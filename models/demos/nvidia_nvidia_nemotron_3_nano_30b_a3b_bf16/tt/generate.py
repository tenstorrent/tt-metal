# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Token generation loop for NemotronH-30B on QB TP=4 (4× Blackhole).

Flow
----
1. Tokenize prompt.
2. Prefill: one eager S=1 step per prompt token to build up SSM state and KV
   cache.  (A batched prefill kernel would be faster but S=1 reuse is simpler
   and correct.)
3. Decode: capture a single-token trace on the pre-warmed state, then replay
   it for each generated token.  Between replay executions:
     a. Copy new SSM states back to state inputs via ttnn.assign.
     b. Write the next token ID and incremented position to the persistent
        device tensors via copy_host_to_device_tensor.

Usage
-----
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.generate import generate

    # Chat API (recommended) — applies the model's chat template automatically:
    text = generate(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        mesh_device=mesh_device,
        wc=weight_cache,
        max_new_tokens=50,
    )

    # Raw string API (backward-compatible):
    text = generate(
        prompt="Hello, I am a language model",
        mesh_device=mesh_device,
        wc=weight_cache,
        max_new_tokens=50,
    )
    print(text)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.common.sampling.sampling_params import SamplingParams
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import sample_top_p

from .kv_cache import DEFAULT_BLOCK_SIZE, DEFAULT_MAX_SEQ_LEN, MODEL_MAX_SEQ_LEN, SNAP, allocate_decoder_state
from .model import WeightCache, nemotron_h_forward_stateful
from .tp import _R, _host_rep


@dataclass
class GenerationMetrics:
    """Per-run performance metrics returned when generate(..., return_metrics=True).

    Timing breakdown:
      prefill_compile_s   — first prefill token (kernel JIT compilation).
      prefill_inference_s — remaining prefill tokens (JIT cache hits).
      ttft_s              — time-to-first-token: full prefill + warmup + trace capture.
      warmup_s            — warmup forward pass (cpu_gate=False path only).
      trace_capture_s     — ttnn trace capture (cpu_gate=False path only).
      decode_compile_s    — first decode step (trace replay or first eager forward).
      decode_inference_s  — steady-state decode (all steps after the first).
      decode_toks_s       — throughput: generated_tokens / decode_inference_s.
    """

    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_compile_s: float = 0.0
    prefill_inference_s: float = 0.0
    ttft_s: float = 0.0
    warmup_s: float = 0.0
    trace_capture_s: float = 0.0
    decode_compile_s: float = 0.0
    decode_inference_s: float = 0.0
    decode_toks_s: float = 0.0

    def print_summary(self) -> None:
        print("\n--- Generation Benchmark ---")
        print(f"  Prompt tokens       : {self.prompt_tokens}")
        print(f"  Generated tokens    : {self.generated_tokens}")
        print(f"  Prefill compile     : {self.prefill_compile_s * 1000:.0f} ms (first token, includes JIT)")
        if self.prefill_inference_s > 0:
            avg_ms = self.prefill_inference_s / max(self.prompt_tokens - 1, 1) * 1000
            print(f"  Prefill inference   : {self.prefill_inference_s * 1000:.0f} ms total ({avg_ms:.0f} ms/tok)")
        if self.warmup_s > 0:
            print(f"  Warmup pass         : {self.warmup_s * 1000:.0f} ms")
        if self.trace_capture_s > 0:
            print(f"  Trace capture       : {self.trace_capture_s * 1000:.0f} ms")
        print(f"  TTFT                : {self.ttft_s * 1000:.0f} ms")
        if self.decode_compile_s > 0:
            print(f"  Decode compile      : {self.decode_compile_s * 1000:.0f} ms (first step)")
        if self.decode_inference_s > 0:
            print(f"  Decode inference    : {self.decode_inference_s * 1000:.0f} ms total")
        print(f"  Decode throughput   : {self.decode_toks_s:.2f} tok/s")
        print("----------------------------")


def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(SNAP)


def _apply_penalties(
    logits: torch.Tensor,
    generated_ids: list[int],
    repetition_penalty: float,
    frequency_penalty: float,
    presence_penalty: float,
) -> torch.Tensor:
    """Apply repetition / frequency / presence penalties to logits in-place."""
    if not generated_ids:
        return logits
    if repetition_penalty != 1.0 or frequency_penalty != 0.0 or presence_penalty != 0.0:
        counts = torch.zeros_like(logits)
        for tid in generated_ids:
            counts[tid] += 1
        if repetition_penalty != 1.0:
            # Scale down logits for tokens that have already appeared.
            logits = torch.where(logits < 0, logits * repetition_penalty, logits / repetition_penalty)
        if frequency_penalty != 0.0:
            logits -= frequency_penalty * counts
        if presence_penalty != 0.0:
            logits -= presence_penalty * (counts > 0).float()
    return logits


def _sample_token(
    logits: torch.Tensor,
    sp: SamplingParams,
    generated_ids: list[int] | None = None,
) -> int:
    """Sample next token using SamplingParams (temperature / top_k / top_p / penalties).

    logits: [1, 1, vocab_size] bfloat16 on CPU (first replica from host_rep).
    """
    logits = logits[0, 0].float()  # [vocab_size]

    # Penalties applied before softmax.
    logits = _apply_penalties(
        logits,
        generated_ids or [],
        sp.repetition_penalty,
        sp.frequency_penalty,
        sp.presence_penalty,
    )

    temperature = sp.temperature if isinstance(sp.temperature, float) else sp.temperature[0]
    top_k = sp.top_k if isinstance(sp.top_k, int) else sp.top_k[0]
    top_p = sp.top_p if isinstance(sp.top_p, float) else sp.top_p[0]

    if temperature == 0.0:
        return int(logits.argmax())

    logits = logits / temperature

    # top-k filter before softmax.
    if top_k > 0:
        topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < topk_vals[-1]] = float("-inf")

    probs = torch.softmax(logits, dim=-1)

    # top-p (nucleus) via the shared implementation.
    if 0.0 < top_p < 1.0:
        return int(sample_top_p(probs.unsqueeze(0), top_p).squeeze())

    return int(torch.multinomial(probs, 1))


def _to_device_token(token_id: int, mesh_device) -> ttnn.Tensor:
    cpu = torch.tensor([[token_id]], dtype=torch.int32)
    return ttnn.from_torch(
        cpu,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )


def _update_ids(ids_tt: ttnn.Tensor, token_id: int):
    cpu = torch.tensor([[token_id]], dtype=torch.int32)
    ids_host = ttnn.from_torch(cpu, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn.copy_host_to_device_tensor(ids_host, ids_tt)


def _update_pos(current_pos_tt: ttnn.Tensor, pos: int):
    cpu = torch.tensor([pos], dtype=torch.int32)
    pos_host = ttnn.from_torch(cpu, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn.copy_host_to_device_tensor(pos_host, current_pos_tt)


def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> from post-generation text (strip_thinking=False path).

    Used when the caller opted into CoT (strip_thinking=False) and wants the
    final answer extracted from the generated text after </think>.
    The chat template adds <think> as the last INPUT token, so generated text
    starts INSIDE the block — strip up to and including </think>.
    """
    import re

    # Full block in text (rare — raw-prompt callers who manually included <think>).
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Text starts mid-block (normal chat mode with CoT).
    end = text.find("</think>")
    if end != -1:
        text = text[end + len("</think>") :].lstrip()
    return text.strip()


def generate(
    prompt: str | None = None,
    mesh_device=None,
    wc: WeightCache | None = None,
    max_new_tokens: int = 100,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    block_size: int = DEFAULT_BLOCK_SIZE,
    temperature: float = 0.0,
    top_p: float = 0.9,
    verbose: bool = True,
    cpu_gate: bool = False,
    messages: list | None = None,
    system_prompt: str | None = None,
    strip_thinking: bool = True,
    sampling_params: SamplingParams | None = None,
    return_metrics: bool = False,
) -> str | tuple[str, GenerationMetrics]:
    """Generate text from a raw prompt string or a list of chat messages.

    Args:
        prompt:         Raw text input (backward-compatible). Ignored when
                        `messages` is provided.
        mesh_device:    Open TTNN MeshDevice (TP=4 QB).
        wc:             WeightCache; created from default SNAP path if None.
        max_new_tokens: Maximum new tokens to generate.
        max_seq_len:    Maximum total sequence length (prompt + generated).
                        Default 32768. Model supports up to 262144; KV cache
                        memory at 256k is ~1.57 GB. Prefill is sequential
                        (~45 ms/tok) so very long contexts are slow to prefill.
        block_size:     Paged KV cache block size (tokens per block).
        temperature:    Sampling temperature; 0 = greedy.
        top_p:          Nucleus sampling threshold.
        verbose:        Print progress.
        cpu_gate:       If False (default), compute MoE gate on device in bfloat16 —
                        trace-compatible; decode loop uses ttnn.execute_trace for
                        ~15 tok/s on TP=4 QB.
                        If True, gate runs on CPU in float32 — exact HF routing
                        but not trace-compatible (~7 tok/s).
        messages:       List of {"role": ..., "content": ...} dicts. When provided,
                        the tokenizer's chat template is applied automatically and
                        only the newly generated text is returned (not the echoed
                        prompt). Takes precedence over `prompt`.
        system_prompt:  Optional system message prepended to `messages`. Ignored
                        when `messages` is None.
        sampling_params: SamplingParams instance for fine-grained control (top_k,
                        repetition_penalty, frequency_penalty, presence_penalty, seed).
                        When provided, its temperature and top_p override the
                        corresponding kwargs.
        return_metrics: When True, return (text, GenerationMetrics) instead of just text.
        strip_thinking: When True (default), suppress the model's chain-of-thought
                        reasoning. The chat template normally appends <think> to the
                        generation prompt, putting the model into reasoning mode before
                        every reply (often hundreds of tokens of CoT). With
                        strip_thinking=True (default) that token is omitted so the model
                        answers directly — better for demos and short factual queries.
                        Set to False when you want full CoT reasoning (set
                        max_new_tokens high enough for </think> to be generated).
                        Only applies when `messages` is provided.

    Returns:
        When `messages` is provided: the assistant reply only (no prompt echo;
        thinking stripped from prompt when strip_thinking=True).
        When only `prompt` is provided: the full text (prompt + new tokens).
    """
    if wc is None:
        wc = WeightCache()

    # Resolve SamplingParams — explicit instance takes precedence; kwargs are the fallback.
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=temperature, top_k=0, top_p=top_p)
    else:
        # sampling_params overrides temperature/top_p kwargs
        temperature = (
            sampling_params.temperature
            if isinstance(sampling_params.temperature, float)
            else sampling_params.temperature[0]
        )
        top_p = sampling_params.top_p if isinstance(sampling_params.top_p, float) else sampling_params.top_p[0]

    if sampling_params.seed is not None:
        torch.manual_seed(sampling_params.seed if isinstance(sampling_params.seed, int) else sampling_params.seed[0])

    tokenizer = _load_tokenizer()

    # Build input_ids from messages (chat template) or raw prompt.
    chat_mode = messages is not None
    if chat_mode:
        msgs = list(messages)
        if system_prompt is not None and (not msgs or msgs[0]["role"] != "system"):
            msgs.insert(0, {"role": "system", "content": system_prompt})
        # Get formatted string first so we can optionally strip <think>.
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if strip_thinking and formatted.endswith("<think>\n"):
            # Drop the <think> trigger — model answers directly without CoT.
            formatted = formatted[: -len("<think>\n")]
        input_ids = tokenizer.encode(formatted, add_special_tokens=False)
    else:
        if prompt is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")
        input_ids = tokenizer(prompt, return_tensors=None)["input_ids"]

    prompt_len = len(input_ids)

    # --- Input validation ---
    if max_seq_len > MODEL_MAX_SEQ_LEN:
        raise ValueError(
            f"max_seq_len={max_seq_len} exceeds the model's max_position_embeddings " f"({MODEL_MAX_SEQ_LEN})."
        )
    if prompt_len >= max_seq_len:
        raise ValueError(
            f"Prompt is {prompt_len} tokens but max_seq_len={max_seq_len}. "
            f"Increase max_seq_len or shorten the prompt. "
            f"Model supports up to {MODEL_MAX_SEQ_LEN} tokens total."
        )
    available = max_seq_len - prompt_len - 1  # -1 so we don't write at exactly max_seq_len
    if max_new_tokens > available:
        if verbose:
            print(
                f"Warning: max_new_tokens={max_new_tokens} exceeds available space "
                f"({available} tokens after {prompt_len}-token prompt in max_seq_len={max_seq_len}). "
                f"Capping to {available}."
            )
        max_new_tokens = available
    if max_new_tokens <= 0:
        raise ValueError(f"No room to generate tokens: prompt_len={prompt_len}, max_seq_len={max_seq_len}.")

    if verbose:
        print(f"Prompt tokens: {prompt_len}")

    profiler = BenchmarkProfiler()
    metrics = GenerationMetrics(prompt_tokens=prompt_len)

    state = allocate_decoder_state(mesh_device, B=1, max_seq_len=max_seq_len, block_size=block_size)

    # Pre-allocate persistent device token tensor (updated between trace replays).
    ids_tt = _to_device_token(input_ids[0], mesh_device)

    # --- Prefill: process prompt tokens one at a time (S=1 decode steps) ---
    # Token 0 includes kernel JIT compilation (compile step); tokens 1+ are cache hits.
    if verbose:
        print("Prefilling...")
    for pos, tok in enumerate(input_ids):
        step_name = "prefill_compile" if pos == 0 else f"prefill_{pos}"
        if verbose:
            print(f"  prefill token {pos + 1}/{len(input_ids)}...", flush=True)
        _update_ids(ids_tt, tok)
        _update_pos(state.current_pos, pos)
        with profiler(step_name):
            logits_tt = nemotron_h_forward_stateful(mesh_device, ids_tt, wc, state, cpu_gate=True)
            ttnn.synchronize_device(mesh_device)
        state.advance()
        if verbose:
            elapsed = profiler.get_duration("prefill_compile" if pos == 0 else f"prefill_{pos}")
            print(f"  prefill token {pos + 1}/{len(input_ids)} done ({elapsed:.1f}s)", flush=True)

    metrics.prefill_compile_s = profiler.get_duration("prefill_compile")
    metrics.prefill_inference_s = sum(
        profiler.get_duration(f"prefill_{i}")
        for i in range(1, len(input_ids))
        if profiler.contains_step(f"prefill_{i}")
    )
    elapsed_prefill = metrics.prefill_compile_s + metrics.prefill_inference_s
    if verbose:
        print(f"Prefill done: {len(input_ids)} tokens in {elapsed_prefill:.1f}s")

    # Sample first decode token from prefill logits.
    logits_cpu = _host_rep(logits_tt, mesh_device, 1)
    generated = []
    next_token = _sample_token(logits_cpu, sampling_params, generated)
    generated = [next_token]
    if verbose:
        tok_str = tokenizer.decode([next_token])
        print(f"First decoded token: {repr(tok_str)}")

    decode_pos = len(input_ids)

    if cpu_gate:
        # cpu_gate=True: D2H inside forward makes trace incompatible — run eager decode loop.
        if verbose:
            print("Decode mode: eager (cpu_gate=True, no trace)")
        metrics.ttft_s = elapsed_prefill
        for step in range(max_new_tokens - 1):
            decode_pos += 1
            _update_ids(ids_tt, next_token)
            _update_pos(state.current_pos, decode_pos - 1)
            step_name = "decode_compile" if step == 0 else f"decode_{step}"
            with profiler(step_name):
                logits_tt = nemotron_h_forward_stateful(mesh_device, ids_tt, wc, state, cpu_gate=True)
                ttnn.synchronize_device(mesh_device)
            state.advance()

            logits_cpu = _host_rep(logits_tt, mesh_device, 1)
            next_token = _sample_token(logits_cpu, sampling_params, generated)
            generated.append(next_token)

            if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
                break
            if decode_pos >= max_seq_len:
                break

        n_gen = len(generated)
        metrics.generated_tokens = n_gen
        metrics.decode_compile_s = (
            profiler.get_duration("decode_compile") if profiler.contains_step("decode_compile") else 0.0
        )
        metrics.decode_inference_s = sum(
            profiler.get_duration(f"decode_{i}") for i in range(1, n_gen) if profiler.contains_step(f"decode_{i}")
        )
        metrics.decode_toks_s = (n_gen - 1) / metrics.decode_inference_s if metrics.decode_inference_s > 0 else 0.0
        elapsed_decode = metrics.decode_compile_s + metrics.decode_inference_s
        if verbose and n_gen > 1:
            print(f"Generated {n_gen} tokens in {elapsed_decode:.2f}s ({metrics.decode_toks_s:.2f} tok/s)")
    else:
        # cpu_gate=False: on-device gate is trace-compatible — use traced decode.
        #
        # Trace decode contract:
        #   1. Trace capture runs token_0 at decode_pos → produces logits_tt for token_1.
        #   2. Read logits_tt immediately, sample token_1, advance state, increment pos.
        #   3. Loop re-executes the trace with token_1 at (decode_pos+1), token_2 at
        #      (decode_pos+2), etc.  Each execute_trace updates logits_tt in-place.
        #
        # BUG that this fixes: old code advanced state + pos BEFORE reading logits_tt,
        # then started the loop with next_token = token_0 at the incremented pos,
        # processing token_0 a second time and discarding the trace-capture logits.
        remaining = max_new_tokens - 1  # token_0 already in generated
        if remaining > 0:
            _update_ids(ids_tt, next_token)
            _update_pos(state.current_pos, decode_pos)

            # Warmup pass (cpu_gate=False, outside trace).
            if verbose:
                print("Warmup pass: priming F32 gate weight caches (cpu_gate=False)...")
            with profiler("warmup"):
                nemotron_h_forward_stateful(mesh_device, ids_tt, wc, state, cpu_gate=False)
                ttnn.synchronize_device(mesh_device)
            metrics.warmup_s = profiler.get_duration("warmup")

            if verbose:
                print("Capturing decode trace...")
            with profiler("trace_capture"):
                trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                logits_tt = nemotron_h_forward_stateful(mesh_device, ids_tt, wc, state, cpu_gate=False)
                ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
                ttnn.synchronize_device(mesh_device)
            metrics.trace_capture_s = profiler.get_duration("trace_capture")
            metrics.ttft_s = elapsed_prefill + metrics.warmup_s + metrics.trace_capture_s
            if verbose:
                print(f"Trace captured. TTFT: {metrics.ttft_s * 1000:.0f} ms")

            # Consume trace-capture output: this IS decode step 1.
            logits_cpu = _host_rep(logits_tt, mesh_device, 1)
            state.advance()
            decode_pos += 1
            next_token = _sample_token(logits_cpu, sampling_params, generated)
            generated.append(next_token)
            remaining -= 1

            for step in range(remaining):
                if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
                    break
                if decode_pos >= max_seq_len:
                    break

                ids_host = ttnn.from_torch(
                    torch.tensor([[next_token]], dtype=torch.int32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                pos_host = ttnn.from_torch(
                    torch.tensor([decode_pos], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                ttnn.copy_host_to_device_tensor(ids_host, ids_tt)
                ttnn.copy_host_to_device_tensor(pos_host, state.current_pos)

                step_name = "decode_compile" if step == 0 else f"decode_{step}"
                with profiler(step_name):
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)

                state.advance()
                decode_pos += 1

                logits_cpu = _host_rep(logits_tt, mesh_device, 1)
                next_token = _sample_token(logits_cpu, sampling_params, generated)
                generated.append(next_token)

            n_gen = len(generated)
            metrics.generated_tokens = n_gen
            metrics.decode_compile_s = (
                profiler.get_duration("decode_compile") if profiler.contains_step("decode_compile") else 0.0
            )
            metrics.decode_inference_s = sum(
                profiler.get_duration(f"decode_{i}") for i in range(1, n_gen) if profiler.contains_step(f"decode_{i}")
            )
            metrics.decode_toks_s = (n_gen - 1) / metrics.decode_inference_s if metrics.decode_inference_s > 0 else 0.0
            elapsed_decode = metrics.decode_compile_s + metrics.decode_inference_s
            if verbose and n_gen > 1:
                print(f"Generated {n_gen} tokens in {elapsed_decode:.2f}s ({metrics.decode_toks_s:.2f} tok/s)")

            ttnn.release_trace(mesh_device, trace_id)

    if chat_mode:
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if strip_thinking:
            text = _strip_thinking(text)
    else:
        text = tokenizer.decode(input_ids + generated, skip_special_tokens=True)

    if return_metrics:
        return text, metrics
    return text
