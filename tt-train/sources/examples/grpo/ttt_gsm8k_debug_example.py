#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone TTT (tt-transformers) GRPO-completer smoke test for Blackhole.

Runs :class:`TttStandaloneCompleter` over either a single detailed
question (GSM8K, default) or a bulk readability sweep (BoolQ, top-100
by default) so garbled output on BH is easy to spot.

This is a single-rank debug driver that models the GRPO rollout worker on
branch ``ichovpan/grpo-speedup``
(``tt-train/sources/examples/grpo/utils/ttt_generation_worker.py::TttGenerationWorker``)
but strips out everything that only exists to serve the 2-rank tt-run
lifecycle. Specifically it drops:

* the ``dummy_weights=True`` boot -- boot with real HF weights instead so
  a single-process run can print coherent completions without waiting
  for a ``WeightBridge`` push;
* ``disable_disk_cache=True`` -- that flag only exists to avoid a
  flatbuffer collective deadlocking against an asymmetric ttml peer
  (``[1, 2]`` vs ``[1, 1]``); with no peer, the standard on-disk cache
  is fine;
* MPI init, the ``TttInferenceServer`` RPC loop, and any use of
  ``ttml.autograd.AutoContext`` -- this script only touches the
  ``models.tt_transformers`` stack, exactly the code path
  ``TttGenerationWorker`` drives during a real rollout.

Everything else is intentionally the same:

* ``ModelArgs`` -> ``Transformer`` -> ``Generator`` construction from
  ``models.tt_transformers.tt.*``.
* Paged attention sizing formula
  (``blocks_per_user = max(min_num_blocks, B * ceil(seq_len / block_size)) // B``).
* On-device sampling via ``SamplingParams`` baked into the captured
  decode trace on first ``generate()`` call.
* Prefill returns ``(output_tokens, log_probs)`` when
  ``sampling_params != None``; the sampled token is written directly
  into the trace's next-step ``tokens`` input buffer, closing the
  decode loop on device.

The point of the script is to answer: *does Llama-3.2-1B-Instruct
produce garbage tokens on Blackhole through the exact tt-transformers
stack that GRPO rollout uses?*

The ``simple_text_demo`` demo is CI-validated on WH N150 only for
Llama-3.2-1B (see ``models/model_ci_tiers.md``); when GRPO rollout runs
the same stack on BH the failure surfaces as garbage completions.
Reproducing it here in a single rank makes it a lot easier to bisect
(swap ``--optimizations``, disable trace, try greedy vs. sampling,
inspect per-step tokens) without having to keep an ttml peer alive.

Usage
=====

Single-chip BH (P150):

    export TT_METAL_HOME=/localdev/ichovpan/tt-metal
    export HF_TOKEN=<your token>   # or use --model unsloth/Llama-3.2-1B-Instruct
    cd $TT_METAL_HOME/tt-train/sources/examples/grpo
    # One detailed GSM8K question (verbose diagnostics)
    python3 ttt_gsm8k_debug_example.py
    # Top-100 BoolQ questions in batches of 8 (readability sweep)
    python3 ttt_gsm8k_debug_example.py --dataset boolq

Handy debug variants:

    # Non-gated mirror -- avoids HF_TOKEN
    python3 ttt_gsm8k_debug_example.py --model unsloth/Llama-3.2-1B-Instruct

    # Turn OFF the captured decode trace to rule out trace-side issues
    python3 ttt_gsm8k_debug_example.py --no-trace

    # Sample instead of greedy (default is temperature=0.0 = greedy)
    python3 ttt_gsm8k_debug_example.py --temperature 1.0 --top-p 0.95 --seed 42

    # Feed the prompt in raw, bypassing apply_chat_template
    python3 ttt_gsm8k_debug_example.py --raw-prompt

    # Show tokenised prompt as well as tokenised completion
    python3 ttt_gsm8k_debug_example.py --dump-prompt-tokens

    # Re-run only the corrupted BoolQ questions from a previous sweep
    python3 ttt_gsm8k_debug_example.py --dataset boolq \
        --question-ids 8,39,43,50,61,67,88,94,95,106,126,148,162,187
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, List, Optional

# tt-transformers reads HF_MODEL from the environment during ``ModelArgs``.
# Import order matters: we must set HF_MODEL BEFORE ``ModelArgs`` is
# imported anywhere (it's captured once at construction, not at import,
# but keeping this at the very top avoids surprises with lazy imports
# elsewhere in this file).
_DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


def _dump_tokens(tokenizer: Any, ids: List[int], label: str, limit: Optional[int] = None) -> None:
    """Print token ids alongside their per-token decoded string.

    ``skip_special_tokens=False`` so ``<|eot_id|>`` etc. are visible; the
    per-token decode is what makes garbage output (invalid UTF-8 bytes,
    unexpected byte-fallback pieces) obvious.
    """
    shown = ids if limit is None else ids[:limit]
    print(f"--- {label} ({len(ids)} tokens" f"{'' if limit is None else f', showing first {len(shown)}'}) ---")
    for i, tok in enumerate(shown):
        piece = tokenizer.decode([int(tok)], skip_special_tokens=False)
        print(f"  [{i:4d}] id={int(tok):>6d}  {piece!r}")
    if limit is not None and len(ids) > limit:
        print(f"  ... {len(ids) - limit} more tokens omitted")


SYSTEM_PROMPT_GSM8K = (
    "You are a careful math tutor. Solve the problem step by step. "
    "After your reasoning, put the final numeric answer on its own line "
    "prefixed by ####."
)

SYSTEM_PROMPT_BOOLQ = (
    "You are a concise assistant that outputs short sentences. Print Yes or No "
    "in the first sentence. Make sure your Yes/No answer is factually correct."
)


def _apply_chat(tokenizer: Any, system_prompt: str, user_content: str, raw: bool) -> str:
    if raw:
        return user_content
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _fetch_gsm8k_examples(n: int, tokenizer: Any, system_prompt: str, raw_prompt: bool) -> List[dict]:
    """Return the first ``n`` GSM8K test-split rows as {index, question, golden, prompt}.

    Matches the loader used in ``sources/examples/gsm8k_finetune/gsm8k_finetune.py``.
    """
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test", verification_mode="no_checks")
    out: List[dict] = []
    for i in range(min(n, len(ds))):
        row = ds[i]
        out.append(
            {
                "index": i,
                "question": row["question"],
                "golden": row["answer"],
                "prompt": _apply_chat(tokenizer, system_prompt, row["question"], raw_prompt),
            }
        )
    return out


def _fetch_boolq_examples(n: int, tokenizer: Any, system_prompt: str, raw_prompt: bool) -> List[dict]:
    """Return the first ``n`` BoolQ validation-split rows as {index, question, golden, prompt}.

    Matches the loader used in ``boolq_accuracy_example.py`` (validation
    split, ``Question: ... Context: ...`` user template).
    """
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split="validation")
    out: List[dict] = []
    for i in range(min(n, len(ds))):
        row = ds[i]
        user_content = f"Question: {row['question']}? Context: {row['passage']}"
        out.append(
            {
                "index": i,
                "question": row["question"],
                "golden": "yes" if row["answer"] else "no",
                "prompt": _apply_chat(tokenizer, system_prompt, user_content, raw_prompt),
            }
        )
    return out


# ``--dataset`` -> (per-question default max_tokens, default num_questions,
# system prompt, loader). Loader signature is
# ``(n, tokenizer, system_prompt, raw_prompt) -> list[dict]``.
_DATASETS = {
    "gsm8k": {
        "default_num_questions": 1,
        "default_max_tokens": 256,
        "system_prompt": SYSTEM_PROMPT_GSM8K,
        "loader": _fetch_gsm8k_examples,
    },
    "boolq": {
        "default_num_questions": 100,
        "default_max_tokens": 64,
        "system_prompt": SYSTEM_PROMPT_BOOLQ,
        "loader": _fetch_boolq_examples,
    },
}


def _prefill_bucket(seq_len: int) -> int:
    """Same bucket function as ``models/tt_transformers/tt/common.py::get_padded_prefill_len``.

    Duplicated here (instead of imported) so ``--dump-prompt-lengths``-style
    reporting works even when the module isn't importable yet (e.g. before
    ``HF_MODEL`` is set). Kept in sync with tt-transformers.
    """
    if seq_len <= 128:
        return 128
    if seq_len <= 1024:
        return 1024
    return 1 << (seq_len - 1).bit_length()


def _has_replacement_chars(s: str) -> bool:
    """True if the decoded string contains U+FFFD or U+FFFE.

    The tokenizer decodes byte-fallback pieces as the Unicode replacement
    character when the underlying bytes don't form valid UTF-8, which is
    the canonical "garbage tokens" signature we're trying to catch on BH.
    """
    return "\ufffd" in s or "\ufffe" in s


def _oneline(s: str, limit: int = 80) -> str:
    """Collapse whitespace + truncate ``s`` for a single-line log line."""
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 1] + "\u2026"


def _format_answer(s: str, *, indent: str = "           ", max_chars: int = 0) -> str:
    """Format a possibly multi-line completion for the bulk-sweep view.

    * Preserves the model's own line breaks (unlike ``_oneline``): garbled
      but grammatically-plausible output like Q8 in the terminal transcript
      often spans multiple lines, and collapsing them into 80 chars hides
      exactly the tail that gives the garbage away.
    * Every line after the first is prefixed with ``indent`` so it lines up
      under the ``A:`` marker.
    * When ``max_chars > 0`` and the answer is longer, it's truncated with
      a trailing ``...(<remaining> more chars)`` note so you know something
      is being hidden.
    """
    text = s.rstrip()
    if max_chars > 0 and len(text) > max_chars:
        remaining = len(text) - max_chars
        text = text[:max_chars].rstrip() + f"\n{indent}...({remaining} more chars, use --answer-chars 0 for full text)"
    lines = text.splitlines() or [""]
    return ("\n" + indent).join(lines)


def _bf16_attn_bfp8_mlp_optimizations(num_decoders: int, model_name: str) -> Any:
    """Llama-family per-layer precision preset.

    Copied verbatim (semantics-wise) from
    ``utils/llama_ttt_presets.py::bf16_attn_bfp8_mlp_optimizations`` on
    branch ``ichovpan/grpo-speedup``: bf16 attention (Q/K/V/O + KV cache),
    BFP8 MLP (FF1/FF2/FF3), HIFI4 on the attention path, HIFI2_FP16 on
    the MLP path.
    """
    from models.tt_transformers.tt.model_config import (
        DecodersPrecision,
        MathFidelitySetting,
        ModelOptimizations,
        OpGroup,
        PrecisionSetting,
        TensorGroup,
    )

    conf = ModelOptimizations(
        {
            "TensorPrecision": {
                TensorGroup.FF1_FF3: PrecisionSetting.BFP8,
                TensorGroup.FF2: PrecisionSetting.BFP8,
                TensorGroup.WQKV: PrecisionSetting.BF16,
                TensorGroup.WO: PrecisionSetting.BF16,
                TensorGroup.KV_CACHE: PrecisionSetting.BF16,
            },
            "OpFidelity": {
                OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI2_FP16,
                OpGroup.LI_FF2: MathFidelitySetting.HIFI2_FP16,
                OpGroup.LI_QKV_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.LI_QKV_PREFILL: MathFidelitySetting.HIFI4,
                OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI4,
                OpGroup.LI_O_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI4,
            },
        }
    )
    conf.__name__ = "bf16_attn_bfp8_mlp"
    inst = DecodersPrecision(num_decoders, model_name, decoder_conf=conf)
    inst.__name__ = "bf16_attn_bfp8_mlp"
    return inst


def _llama_stop_and_pad(tokenizer: Any) -> tuple[frozenset[int], int]:
    """Extract stop / pad token IDs from an already-loaded HF tokenizer.

    Same logic as ``utils/llama_ttt_presets.py::llama_stop_and_pad`` on
    branch ``ichovpan/grpo-speedup``, minus the tokenizer load (we
    already have one here).
    """
    stop_strs = ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>")
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for s in stop_strs:
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
            ids.add(int(tid))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise RuntimeError("tokenizer exposes neither pad_token_id nor eos_token_id; cannot derive a filler id.")
    return frozenset(ids), int(pad)


# ---------------------------------------------------------------------------
# Standalone completer -- mirrors TttGenerationWorker's model construction
# and generate() loop, but with real HF weights and no RPC surface.
# ---------------------------------------------------------------------------


class TttStandaloneCompleter:
    """Real-weights, single-rank version of ``TttGenerationWorker``.

    Constructed once, then reused across ``generate()`` calls. Same
    on-device sampling / paged-attention / trace behaviour as the branch
    worker; the only differences are the missing ``dummy_weights`` boot
    and the missing MPI machinery.
    """

    def __init__(
        self,
        *,
        mesh_device: Any,
        model_source: str,
        max_batch_size: int,
        max_seq_len: int,
        instruct: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: Optional[int],
        paged_block_size: int = 32,
        min_num_blocks: int = 1024,
    ) -> None:
        import ttnn
        import torch

        from models.common.sampling import SamplingParams
        from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model

        self.mesh_device = mesh_device
        self._max_batch_size = int(max_batch_size)

        os.environ["HF_MODEL"] = model_source

        # Same block-table sizing formula as ``TttGenerationWorker``: big
        # enough that the worst-case prompt + decode never overflows
        # ``max_seq_len``, rounded so ``max_num_blocks`` is a multiple of
        # ``max_batch_size``.
        required_blocks_per_user = (max_seq_len + paged_block_size - 1) // paged_block_size
        max_num_blocks = max(min_num_blocks, max_batch_size * required_blocks_per_user)
        blocks_per_user = max_num_blocks // max_batch_size
        max_num_blocks = blocks_per_user * max_batch_size
        self._paged_cache_max_seq_len = paged_block_size * blocks_per_user
        self._paged_attention_config = PagedAttentionConfig(
            block_size=paged_block_size,
            max_num_blocks=max_num_blocks,
        )
        # Sequential (unshuffled) page table, same as the branch worker.
        # ``simple_text_demo.py`` shuffles via ``torch.randperm``; we
        # deliberately don't so this driver matches the rollout path
        # 1:1 during debugging.
        self.page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(max_batch_size, blocks_per_user)

        # ``ModelArgs`` calls ``self.optimizations = optimizations(self)``
        # -- i.e. a single-arg call with the ModelArgs instance. Our
        # preset takes (num_decoders, model_name), so wrap it in a
        # lambda the same way ``TttGenerationWorker`` does on branch
        # ``ichovpan/grpo-speedup``:
        #     optimizations=lambda ma: optimizations(ma.n_layers, ma.model_name)
        model_args, model, tt_kv_cache, _state_dict = create_tt_model(
            mesh_device,
            instruct=instruct,
            max_batch_size=max_batch_size,
            optimizations=lambda ma: _bf16_attn_bfp8_mlp_optimizations(ma.n_layers, ma.model_name),
            max_seq_len=max_seq_len,
            paged_attention_config=self._paged_attention_config,
            dtype=ttnn.bfloat16,
        )
        # Downstream code (Generator, our loop) assumes a single-DP
        # topology: one ModelArgs, one Transformer, one kv-cache list.
        # ``create_tt_model`` already gives us that (no list wrapping).
        self.model_args = model_args
        self.model = model
        self.kv_cache = tt_kv_cache

        # ``create_tt_model`` loaded the real HF tokenizer into
        # ``model_args`` (dummy_weights=False path). Grab it here so
        # ``generate_str`` doesn't need a separate AutoTokenizer round-trip.
        self.tokenizer = model_args.tokenizer

        from models.tt_transformers.tt.generator import Generator

        self.generator = Generator(
            model=[self.model],
            model_args=[self.model_args],
            mesh_device=self.mesh_device,
            tokenizer=self.tokenizer,
        )

        assert self.model.sampling is not None, (
            "On-device sampling is required (SamplingParams is baked into the trace on first "
            "generate call). Got model.sampling=None -- unsupported vocab/mesh combo."
        )
        self._sampling_params = SamplingParams(
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            seed=seed,
        )

        stop_ids, pad_id = _llama_stop_and_pad(self.tokenizer)
        self._stop_token_ids = stop_ids
        self._pad_token_id = pad_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: List[List[int]],
        *,
        max_new_tokens: int = 128,
        enable_trace: bool = True,
        stop_at_eos: bool = True,
    ) -> List[List[int]]:
        """Prefill + decode a batch of token-ID prompts.

        Mirrors ``TttGenerationWorker.generate`` but drops the async
        chunked host readback (single-prompt debug driver -- one blocking
        read per step is fine, and it makes token-by-token print easier).
        """
        import torch

        if max_new_tokens == 0:
            return [[] for _ in prompts]

        prompts, prompt_lens, active_batch_size = self._prepare_prompt_batch(prompts, max_new_tokens)
        batch_size = len(prompts)
        max_prompt_len = max(prompt_lens)

        pad_id = self._pad_token_id
        input_tokens_prefill_pt = torch.full((batch_size, max_prompt_len), pad_id, dtype=torch.int32)
        for i, p in enumerate(prompts):
            input_tokens_prefill_pt[i, : len(p)] = torch.tensor(p, dtype=torch.int32)

        kv_cache = [self.kv_cache]
        self._reset_kv_cache()

        _t_prefill = time.perf_counter()
        # ``sampling_params != None`` -> prefill returns (tokens, log_probs).
        output_tokens, _log_probs = self.generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=self.page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            sampling_params=self._sampling_params,
            warmup_prefill=False,
            enable_trace=enable_trace,
        )
        prefilled_token = output_tokens.reshape(-1)
        prefill_s = time.perf_counter() - _t_prefill
        print(
            f"[TttStandaloneCompleter] prefill done in {prefill_s:.2f}s "
            f"(batch_size={batch_size}, max_prompt_len={max_prompt_len})"
        )

        completions: List[List[int]] = [[] for _ in range(batch_size)]
        user_done = [False] * batch_size
        for u in range(active_batch_size, batch_size):
            user_done[u] = True
        stop_ids = self._stop_token_ids if stop_at_eos else frozenset()

        for u in range(batch_size):
            if user_done[u]:
                continue
            tok = int(prefilled_token[u].item())
            if stop_at_eos and tok in stop_ids:
                user_done[u] = True
            else:
                completions[u].append(tok)

        if all(user_done) or max_new_tokens <= 1:
            return completions[:active_batch_size]

        current_pos = torch.tensor(prompt_lens, dtype=torch.int32)
        out_tok = prefilled_token.unsqueeze(1)

        _t_decode = time.perf_counter()
        steps_executed = 0
        for step in range(max_new_tokens - 1):
            # ``read_from_device=True`` + ``sampling_params != None`` ->
            # ``decode_forward`` internally does the full
            # ``read_decode_output`` -> ``process_decode_output_host`` chain
            # (with ``is_tokens=True``) and returns
            # ``(tokens_torch, log_probs)``. ``tokens_torch`` is 1-D
            # shape ``[B_total]`` int torch; do NOT call
            # ``process_output_decode`` on it again -- that path expects
            # a ttnn host tensor and would try to ``ttnn.reshape`` a
            # torch tensor. See generator.py:1354-1356 and
            # process_decode_output_host in the same file.
            tokens_torch, _log_probs = self.generator.decode_forward(
                out_tok,
                current_pos,
                page_table=self.page_table,
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                sampling_params=self._sampling_params,
                reset_batch=(step == 0),
                prompt_tokens=input_tokens_prefill_pt,
                output_tokens=out_tok,
                read_from_device=True,
            )
            tok_arr = tokens_torch.to(torch.int64).view(-1).tolist()

            for u in range(batch_size):
                if user_done[u]:
                    continue
                tok = int(tok_arr[u])
                if stop_at_eos and tok in stop_ids:
                    user_done[u] = True
                else:
                    completions[u].append(tok)

            current_pos = current_pos + 1
            out_tok = tokens_torch.view(batch_size, 1).to(torch.int32)
            steps_executed += 1

            if stop_at_eos and all(user_done):
                break

        decode_s = time.perf_counter() - _t_decode
        per_step_ms = (decode_s / steps_executed * 1000.0) if steps_executed > 0 else 0.0
        active_tokens = sum(len(c) for c in completions[:active_batch_size])
        active_tok_s = (active_tokens / decode_s) if decode_s > 0 else 0.0
        print(
            f"[TttStandaloneCompleter] decode done in {decode_s:.2f}s "
            f"({steps_executed} steps, {per_step_ms:.1f} ms/step, "
            f"active_tokens={active_tokens} -> {active_tok_s:.1f} tok/s)"
        )
        return completions[:active_batch_size]

    def generate_str(
        self,
        prompt_strs: List[str],
        *,
        max_new_tokens: int = 128,
        enable_trace: bool = True,
        stop_at_eos: bool = True,
    ) -> List[str]:
        prompt_ids = [self.tokenizer.encode(s) for s in prompt_strs]
        completions = self.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            enable_trace=enable_trace,
            stop_at_eos=stop_at_eos,
        )
        return [self.tokenizer.decode(c, skip_special_tokens=False) for c in completions]

    # ------------------------------------------------------------------
    # Internal helpers (mirrored from TttGenerationWorker)
    # ------------------------------------------------------------------

    def _reset_kv_cache(self) -> None:
        import ttnn

        for layer in self.model.layers:
            k_cache, v_cache = layer.attention.layer_past
            ttnn.mul(k_cache, 0, output_tensor=k_cache)
            ttnn.mul(v_cache, 0, output_tensor=v_cache)
        self.generator.prev_page_table = None

    def _prepare_prompt_batch(
        self, prompts: List[List[int]], max_new_tokens: int
    ) -> tuple[List[List[int]], List[int], int]:
        active_batch_size = len(prompts)
        assert 0 < active_batch_size <= self._max_batch_size, (
            f"generate() got {active_batch_size} prompts but was built with " f"max_batch_size={self._max_batch_size}"
        )

        normalized_prompts = [[int(tok) for tok in prompt] for prompt in prompts]
        prompt_lens = [len(p) for p in normalized_prompts]
        assert min(prompt_lens) > 0, "empty prompts are not supported"

        max_prefill_len = self.model_args.max_seq_len - max_new_tokens
        assert (
            max_prefill_len > 0
        ), f"max_new_tokens ({max_new_tokens}) must be < max_seq_len ({self.model_args.max_seq_len})"
        if max(prompt_lens) > max_prefill_len:
            normalized_prompts = [p[-max_prefill_len:] for p in normalized_prompts]
            prompt_lens = [len(p) for p in normalized_prompts]

        max_prompt_len = max(prompt_lens)
        assert max_prompt_len + max_new_tokens <= self.model_args.max_seq_len, (
            f"prompt ({max_prompt_len}) + decode ({max_new_tokens}) > max_seq_len " f"({self.model_args.max_seq_len})"
        )
        assert max_prompt_len + max_new_tokens <= self._paged_cache_max_seq_len, (
            f"prompt ({max_prompt_len}) + decode ({max_new_tokens}) > paged-cache capacity "
            f"({self._paged_cache_max_seq_len})"
        )

        if active_batch_size < self._max_batch_size:
            filler_prompt = [int(self._pad_token_id)]
            pad_slots = self._max_batch_size - active_batch_size
            normalized_prompts.extend([filler_prompt] * pad_slots)
            prompt_lens.extend([1] * pad_slots)

        return normalized_prompts, prompt_lens, active_batch_size


# ---------------------------------------------------------------------------
# stdout/stderr tee (OS-fd level)
# ---------------------------------------------------------------------------


class _FdTee:
    """OS-fd tee of file descriptors 1 and 2 into a log file.

    A pure Python-level ``sys.stdout`` wrapper is *not* enough for
    tt-metal, because the C++ side of the runtime (loguru's compiled
    sink, ``fprintf(stderr, ...)`` calls inside kernels, HTTP libs
    invoked from native code, etc.) writes directly to fd 1 / fd 2 and
    never touches ``sys.stdout``. Those writes bypass any
    ``sys.stdout = FooWrapper`` interception -- which is why the earlier
    Python-level tee only captured the pre-``create_tt_model`` portion of
    the run.

    Design:

    1. Duplicate fds 1 and 2 into ``_saved_stdout_fd`` / ``_saved_stderr_fd``
       so we can restore them at teardown *and* keep a handle on the
       real terminal for the pump.
    2. Create a pipe. ``dup2`` its write end onto both fd 1 and fd 2.
       Every subsequent write (Python or C++) goes into the pipe.
    3. Reconfigure ``sys.stdout`` / ``sys.stderr`` back to line
       buffering: ``TextIOWrapper`` normally switches to block buffering
       when the underlying fd looks non-TTY (a pipe), which would make
       ``print`` output arrive in the log in ~8 KB chunks.
    4. Start a daemon thread that blocks on ``os.read(read_fd, ...)``,
       and forwards every byte to both the saved terminal fd and the
       log file. That keeps the console alive while duplicating the
       stream to disk, byte-for-byte, at the kernel level.

    Teardown restores fd 1 / fd 2, which lets the pipe close (no more
    writers -> ``os.read`` returns ``b""``) and the pump thread exits.
    """

    def __init__(self, log_file: BinaryIO) -> None:
        # Flush any pre-tee buffered output so the split point is clean.
        with contextlib.suppress(Exception):
            sys.stdout.flush()
        with contextlib.suppress(Exception):
            sys.stderr.flush()

        self._saved_stdout_fd: int = os.dup(1)
        self._saved_stderr_fd: int = os.dup(2)

        self._read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)  # fd 1 / fd 2 now hold the only writer refs.

        # ``TextIOWrapper.reconfigure`` is 3.7+; safe on all supported
        # tt-metal Python versions. Without this the log lags the run
        # by a whole write buffer.
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(line_buffering=True)
        with contextlib.suppress(Exception):
            sys.stderr.reconfigure(line_buffering=True)

        self._log_file = log_file
        self._log_fd: int = log_file.fileno()

        self._closed = False
        self._pump_thread = threading.Thread(target=self._pump, name="ttt_gsm8k_debug_log_pump", daemon=True)
        self._pump_thread.start()

    def _pump(self) -> None:
        while True:
            try:
                data = os.read(self._read_fd, 65536)
            except OSError:
                break
            if not data:
                break
            # Best-effort: if either sink is gone (terminal closed,
            # log file torn down early), keep going for the other.
            with contextlib.suppress(OSError):
                os.write(self._saved_stdout_fd, data)
            with contextlib.suppress(OSError):
                os.write(self._log_fd, data)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        # Drain Python-side buffers into the pipe before we detach it.
        with contextlib.suppress(Exception):
            sys.stdout.flush()
        with contextlib.suppress(Exception):
            sys.stderr.flush()

        # Restore real fd 1 / fd 2. Once the last writer to the pipe
        # goes away the pump's blocking ``os.read`` returns b"" and it
        # falls out of the loop.
        os.dup2(self._saved_stdout_fd, 1)
        os.dup2(self._saved_stderr_fd, 2)
        with contextlib.suppress(OSError):
            os.close(self._saved_stdout_fd)
        with contextlib.suppress(OSError):
            os.close(self._saved_stderr_fd)

        self._pump_thread.join(timeout=5.0)
        with contextlib.suppress(OSError):
            os.close(self._read_fd)


def _resolve_log_dir(cli_log_dir: Optional[str]) -> Path:
    """Where to put the run log.

    ``--log-dir`` wins if set; otherwise fall back to a timestamped
    subdirectory under ``$TT_METAL_RUNTIME_ROOT`` (or ``$TT_METAL_HOME``,
    or the repo root inferred from this file's path) matching the
    ``boolq_accuracy_example.py`` layout:

        <repo>/generated/tt-train/ttt_gsm8k_debug_runs/<UTC-timestamp>/
    """
    if cli_log_dir:
        return Path(cli_log_dir).expanduser()
    root = (
        os.environ.get("TT_METAL_RUNTIME_ROOT")
        or os.environ.get("TT_METAL_HOME")
        or str(Path(__file__).resolve().parents[4])
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(root) / "generated" / "tt-train" / "ttt_gsm8k_debug_runs" / stamp


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=_DEFAULT_MODEL_ID, help=f"HF model id (default: {_DEFAULT_MODEL_ID})")
    parser.add_argument(
        "--dataset",
        choices=list(_DATASETS.keys()),
        default="gsm8k",
        help="Which HF dataset to sweep. gsm8k = 1 detailed math question (default), "
        "boolq = top-N validation-split Yes/No questions.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="How many questions to run. Defaults: gsm8k=1, boolq=100. " "Ignored when --question is set.",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Override the dataset with a single custom prompt (implies num_questions=1).",
    )
    parser.add_argument(
        "--question-ids",
        default=None,
        help="Comma-separated list of dataset indices to run (e.g. "
        "'8,39,43,50,61,67,88,94,95,106,126,148,162,187'). The dataset loader is called for "
        "max(id)+1 rows and the result is filtered to just those indices, so the printed "
        "Q-numbers still match the numbering of a full sweep. Overrides --num-questions; "
        "cannot be combined with --question.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max new tokens to generate per question. Defaults: gsm8k=256, boolq=64.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 = greedy; sampling params are BAKED INTO THE TRACE "
        "at first generate call, so this cannot be changed per-call once the trace is captured).",
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p (default: 1.0).")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k (default: 0 = disabled).")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed (default: None).")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Model max_seq_len; must fit prompt + max_new_tokens (default: 2048).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Prompts per generate() call (also becomes ``max_batch_size`` on the completer, so it is "
        "baked into the captured trace). Default: min(num_questions, 8) for boolq, 1 for gsm8k.",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable the captured decode trace. Slower per step, but rules out trace-side issues "
        "when debugging garbled output.",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Skip apply_chat_template and feed the question in as-is. Useful for isolating "
        "tokenizer/chat-template issues from model issues.",
    )
    parser.add_argument(
        "--dump-prompt-tokens",
        action="store_true",
        help="Also print the tokenised prompt (only makes sense for single-question runs).",
    )
    parser.add_argument(
        "--dump-completion-tokens",
        action="store_true",
        help="Print per-token id + decoded piece for every completion. Very verbose; leave off for "
        "bulk sweeps unless you're chasing a specific token-level bug.",
    )
    parser.add_argument(
        "--answer-chars",
        type=int,
        default=0,
        help="Cap the printed answer at this many characters in the bulk-sweep view. "
        "0 (default) = print the full answer (recommended: garbled tails are often "
        "the only garbage signal, U+FFFD doesn't catch grammatical-but-nonsense output).",
    )
    parser.add_argument(
        "--print-full-prompt",
        action="store_true",
        help="Before each question in the bulk sweep, print the full chat-templated prompt "
        "(system prompt + user turn + assistant-turn marker) exactly as the tokenizer sees "
        "it. Useful for confirming the template is well-formed and that the system prompt "
        "actually reached the model. In single-question mode the full prompt is always "
        "printed regardless.",
    )
    parser.add_argument("--no-instruct", action="store_true", help="Load the model in base (non-instruct) mode.")
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory to write the run log into (as ``run.log``). "
        "Defaults to ``$TT_METAL_RUNTIME_ROOT/generated/tt-train/ttt_gsm8k_debug_runs/<UTC-timestamp>/``. "
        "If neither TT_METAL_RUNTIME_ROOT nor TT_METAL_HOME is set, the repo root is inferred from "
        "this file's path.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable stdout/stderr teeing to a run log (still prints to the terminal).",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # OS-fd tee of stdout/stderr -> run.log. Must be set up BEFORE any
    # C++ code prints (tt-metal, loguru's compiled sink, ttnn kernel
    # warnings) so those bytes go through our pipe rather than straight
    # to the terminal. A pure ``sys.stdout = _Tee(...)`` wrapper would
    # miss all of them because they call fprintf(stderr, ...) directly
    # at the fd level.
    # -----------------------------------------------------------------
    log_file: Optional[BinaryIO] = None
    log_file_path: Optional[Path] = None
    fd_tee: Optional[_FdTee] = None
    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / "run.log"
        # Unbuffered binary open: the pump writes raw bytes and expects
        # every write to hit the disk immediately (so ``tail -f`` sees
        # progress live during long BoolQ sweeps).
        log_file = open(log_file_path, "wb", buffering=0)
        fd_tee = _FdTee(log_file)
        # These prints happen after the tee is live -> they land in
        # both the terminal and the log file.
        print(f"[log] Writing run log to {log_file_path}", flush=True)
        print(f"[log] argv: {sys.argv}", flush=True)
        print(f"[log] cwd:  {os.getcwd()}", flush=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    dataset_cfg = _DATASETS[args.dataset]
    max_tokens = args.max_tokens if args.max_tokens is not None else dataset_cfg["default_max_tokens"]

    question_ids: Optional[List[int]] = None
    if args.question_ids is not None:
        if args.question is not None:
            parser.error("--question-ids cannot be combined with --question.")
        try:
            parsed = sorted({int(x.strip()) for x in args.question_ids.split(",") if x.strip()})
        except ValueError as exc:
            parser.error(f"--question-ids must be a comma-separated list of integers: {exc}")
        if not parsed:
            parser.error("--question-ids was empty.")
        if parsed[0] < 0:
            parser.error("--question-ids values must be non-negative.")
        question_ids = parsed

    if args.question is not None:
        num_questions = 1
    elif question_ids is not None:
        # Load enough rows to cover the largest requested index; the
        # filter step below drops everything outside ``question_ids``.
        num_questions = question_ids[-1] + 1
    else:
        num_questions = args.num_questions if args.num_questions is not None else dataset_cfg["default_num_questions"]
    # For anything that scales with the *actual* number of prompts we'll
    # feed the model (batch-size default, config print, single-question
    # detail mode), use the filtered count rather than the raw load size.
    effective_count = len(question_ids) if question_ids is not None else num_questions
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif args.dataset == "boolq":
        batch_size = max(1, min(effective_count, 8))
    else:
        batch_size = 1

    # Import ttnn late so ``HF_MODEL`` is set first for anything downstream
    # that consults os.environ during import.
    os.environ["HF_MODEL"] = args.model
    import ttnn

    print("\n=== Run configuration ===")
    print(f"Model         : {args.model}")
    if question_ids is not None:
        print(
            f"Dataset       : {args.dataset} ({effective_count} question(s), "
            f"filtered from top-{num_questions}; ids={question_ids})"
        )
    else:
        print(f"Dataset       : {args.dataset} ({effective_count} question(s))")
    print(f"Max tokens    : {max_tokens}")
    print(f"Batch size    : {batch_size} (also max_batch_size on the trace)")
    print(f"Temperature   : {args.temperature} (top_k={args.top_k}, top_p={args.top_p}, seed={args.seed})")
    print(f"Trace enabled : {not args.no_trace}")
    print(f"Raw prompt    : {args.raw_prompt}")

    # Open a single-chip mesh. On P150 this maps to the one BH chip; on
    # multi-board setups the environment / mesh graph descriptor picks
    # which chip. No fabric setup is needed for [1, 1].
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        offset=ttnn.MeshCoordinate(0, 0),
    )

    completer: Optional[TttStandaloneCompleter] = None
    try:
        completer = TttStandaloneCompleter(
            mesh_device=mesh_device,
            model_source=args.model,
            max_batch_size=batch_size,
            max_seq_len=args.max_seq_len,
            instruct=not args.no_instruct,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
        )
        tokenizer = completer.tokenizer
        print(
            f"\nTokenizer     : {tokenizer.__class__.__name__} "
            f"(vocab={len(tokenizer)}, eos_id={tokenizer.eos_token_id}, "
            f"pad_id={tokenizer.pad_token_id}, bos_id={tokenizer.bos_token_id})"
        )

        # ------------------------------------------------------------------
        # Build the prompt list
        # ------------------------------------------------------------------
        if args.question is not None:
            examples = [
                {
                    "index": 0,
                    "question": args.question,
                    "golden": "(user-supplied prompt; no golden answer)",
                    "prompt": _apply_chat(tokenizer, dataset_cfg["system_prompt"], args.question, args.raw_prompt),
                }
            ]
        else:
            examples = dataset_cfg["loader"](num_questions, tokenizer, dataset_cfg["system_prompt"], args.raw_prompt)
            if question_ids is not None:
                wanted = set(question_ids)
                examples = [ex for ex in examples if ex["index"] in wanted]
                loaded_ids = {ex["index"] for ex in examples}
                missing = sorted(wanted - loaded_ids)
                if missing:
                    print(
                        f"[warn] --question-ids requested {missing} but the dataset only "
                        f"has {num_questions} rows; those indices were skipped."
                    )
                if not examples:
                    print("[warn] no examples matched --question-ids; nothing to do.")

        # ------------------------------------------------------------------
        # Tokenise every prompt once up front so we can:
        #   * report per-question token lengths (system prompt included --
        #     ``ex["prompt"]`` is the chat-templated string, so encoding it
        #     counts the whole system + user + assistant-marker sequence),
        #   * feed the pre-tokenised ids straight into ``completer.generate``
        #     inside the batch loop (avoids re-encoding the same string).
        # The tokeniser is fast enough that doing this for a few hundred
        # prompts is negligible relative to prefill/decode.
        # ------------------------------------------------------------------
        for ex in examples:
            ids = tokenizer.encode(ex["prompt"])
            ex["prompt_ids"] = ids
            ex["prompt_token_len"] = len(ids)
            ex["prefill_bucket"] = _prefill_bucket(len(ids))

        if examples:
            lens = [ex["prompt_token_len"] for ex in examples]
            bucket_hist = Counter(ex["prefill_bucket"] for ex in examples)
            bucket_str = ", ".join(f"{b}: {n}" for b, n in sorted(bucket_hist.items()))
            print(
                f"\nPrompt lengths: n={len(lens)}, min={min(lens)}, max={max(lens)}, "
                f"mean={sum(lens) / len(lens):.1f} (system prompt included, before pad)"
            )
            print(f"Prefill buckets: {{{bucket_str}}}")

        if len(examples) == 1:
            # Verbose single-question mode -- show the full prompt going in.
            ex = examples[0]
            print("\n=== Prompt fed to the model ===")
            print(ex["prompt"])
            print(f"\nGolden answer : {ex['golden']}")
            print(
                f"Prompt tokens : {ex['prompt_token_len']} "
                f"(prefill bucket = {ex['prefill_bucket']}, system prompt included)"
            )
            if args.dump_prompt_tokens:
                _dump_tokens(tokenizer, ex["prompt_ids"], "prompt tokens", limit=64)

        print(f"\n=== Generating (first generate() call captures the trace @ " f"batch_size={batch_size}) ===")

        # ------------------------------------------------------------------
        # Batched generation loop
        # ------------------------------------------------------------------
        replacement_count = 0
        boolq_correct = 0
        boolq_answered = 0
        overall_start = time.perf_counter()

        for batch_start in range(0, len(examples), batch_size):
            batch = examples[batch_start : batch_start + batch_size]
            prompt_ids = [ex["prompt_ids"] for ex in batch]
            completion_ids_batch = completer.generate(
                prompt_ids, max_new_tokens=max_tokens, enable_trace=not args.no_trace
            )
            assert len(completion_ids_batch) == len(batch)

            for ex, completion_ids in zip(batch, completion_ids_batch):
                completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                has_garbage = _has_replacement_chars(completion_text)
                if has_garbage:
                    replacement_count += 1

                if len(examples) == 1:
                    # Full-detail print for the single-question path.
                    if args.dump_completion_tokens:
                        _dump_tokens(tokenizer, completion_ids, "completion tokens")
                    print("\n=== Decoded completion (skip_special_tokens=False) ===")
                    print(repr(tokenizer.decode(completion_ids, skip_special_tokens=False)))
                    print("\n=== Decoded completion (skip_special_tokens=True) ===")
                    print(completion_text)
                    continue

                # Bulk sweep: one short line per question so the whole
                # thing is skimmable for readability.
                garbage_tag = "  [GARBAGE:U+FFFD]" if has_garbage else ""
                # ``tokens=N`` is the length of the tokenised chat-templated
                # prompt (system + user + assistant marker), i.e. exactly
                # what tt-transformers sees before it rounds up to the next
                # prefill bucket. ``bucket=M`` is that padded length.
                len_tag = f" tokens={ex['prompt_token_len']:>4d} bucket={ex['prefill_bucket']:>4d}"
                header = (
                    f"[Q{ex['index']:3d}] golden={ex['golden']:>3}{len_tag}"
                    if args.dataset == "boolq"
                    else f"[Q{ex['index']:3d}]{len_tag}"
                )
                if args.print_full_prompt:
                    print(f"\n----- {header} full prompt -----")
                    print(ex["prompt"])
                    print(f"----- end {header} full prompt -----")
                print(f"{header} | Q: {_oneline(ex['question'], 70)}{garbage_tag}")
                formatted_answer = _format_answer(completion_text, max_chars=args.answer_chars)
                if args.dataset == "boolq":
                    first_word = (
                        completion_text.strip().split()[0].lower().rstrip(".,!?;:") if completion_text.strip() else ""
                    )
                    correct = first_word.startswith(ex["golden"].lower()) if first_word else False
                    boolq_correct += int(correct)
                    boolq_answered += 1
                    print(f"        A: {formatted_answer}")
                    print(
                        f"        model_first_word={first_word!r:<15} "
                        f"correct={correct} running_acc={boolq_correct/boolq_answered:.3f} "
                        f"({boolq_correct}/{boolq_answered})"
                    )
                else:
                    print(f"        A: {formatted_answer}")

                if args.dump_completion_tokens:
                    _dump_tokens(tokenizer, completion_ids, f"Q{ex['index']} completion tokens")

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        elapsed = time.perf_counter() - overall_start
        print("\n=== Summary ===")
        print(f"Total questions       : {len(examples)}")
        print(f"With U+FFFD garbage   : {replacement_count} " f"({100.0 * replacement_count / len(examples):.1f}%)")
        if args.dataset == "boolq" and boolq_answered > 0:
            print(
                f"BoolQ accuracy        : {boolq_correct}/{boolq_answered} "
                f"({100.0 * boolq_correct / boolq_answered:.1f}%)"
            )
        print(f"Elapsed               : {elapsed:.1f}s")
    finally:
        completer = None
        try:
            ttnn.close_mesh_device(mesh_device)
        except Exception:  # noqa: BLE001 -- best-effort teardown
            pass
        # Print the final log-path banner BEFORE closing the tee, so it
        # lands in the log file. Then tear down the fd redirection
        # (restores real fd 1 / fd 2), then close the file handle.
        if fd_tee is not None:
            with contextlib.suppress(Exception):
                print(f"[log] Run log written to {log_file_path}", flush=True)
            fd_tee.close()
        if log_file is not None:
            with contextlib.suppress(Exception):
                log_file.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
