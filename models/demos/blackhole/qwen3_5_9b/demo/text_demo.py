# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-9B end-to-end text generation test on Blackhole P150.

Consolidates all e2e demo tests into a single parametrized test that covers:
  - Short prompt text generation (128 tokens)
  - Medium prefill with traced decode (2048 tokens)
  - Long-context prefill + decode (4k, 8k tokens)

Run all:    pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py -v -s
Run short:  pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py -v -s -k "prefill_128"
Run 2k:     pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py -v -s -k "prefill_2k"
"""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import requests
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model
from models.tt_transformers.tt.generator import Generator

# Multi-device (TP) is selected via MESH_DEVICE (e.g. P150x4). On a single device
# the mesh is (1,1) and the model runs its validated single-device path; on a
# multi-device mesh it needs FABRIC_1D for the TP collectives (see tp_common notes).
_MESH_SHAPE = {"N150": (1, 1), "N300": (1, 2), "P150x4": (1, 4), "N150x4": (1, 4), "T3K": (1, 8)}.get(
    os.environ.get("MESH_DEVICE"), (1, 1)
)
_MULTI = _MESH_SHAPE != (1, 1)
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {}),
    }
]

SAMPLE_PROMPTS_DIR = "models/demos/blackhole/qwen3_5_9b/demo/sample_prompts"
SHARED_PROMPTS_DIR = "models/demos/llama3_70b_galaxy/demo/sample_prompts"

PERF_TARGETS = {
    128: {"min_decode_tok_s": 5.0, "max_ttft_s": 2.0},
    2048: {"min_decode_tok_s": 4.0, "max_ttft_s": 5.0},
    4096: {"min_decode_tok_s": 2.0, "max_ttft_s": 10.0},
    8192: {"min_decode_tok_s": 1.0, "max_ttft_s": 20.0},
    16384: {"min_decode_tok_s": 0.5, "max_ttft_s": 35.0},
    32768: {"min_decode_tok_s": 0.5, "max_ttft_s": 70.0},
}

# Frankenstein prompt config: seqlen → json_index in eval_frankenstein_long.json.
# Each entry clips Frankenstein text to enough characters to exceed the target token count
# (English text ≈ 4.3 chars/token). Full Frankenstein is ~450K chars ≈ 104K tokens.
_FRANKENSTEIN_CONFIGS = {
    8192: 0,  # 70k chars ≈ 16k tokens (covers 8k with margin)
    16384: 1,  # 140k chars ≈ 32k tokens
    32768: 1,  # 140k chars ≈ 32k tokens
    65536: 2,  # 300k chars ≈ 70k tokens
    131072: 3,  # 500k chars ≈ 104k tokens (full text, Frankenstein caps at ~104k tokens)
}


def _load_and_cache_context(context_url, max_length=None):
    """Download text from URL, cache locally, clip to max_length."""
    cache_dir = Path(SAMPLE_PROMPTS_DIR) / ".context_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / hashlib.md5(context_url.encode()).hexdigest()

    if cache_file.exists():
        context_text = cache_file.read_text()
        logger.info(f"Loaded context from cache: {context_url}")
    else:
        response = requests.get(context_url)
        response.raise_for_status()
        context_text = response.text
        cache_file.write_text(context_text)
        logger.info(f"Downloaded and cached context: {context_url}")

    if max_length:
        context_text = context_text[:max_length]
    return context_text


def _get_prompt(seqlen, tokenizer):
    """Load a prompt of approximately seqlen tokens, clipped but not padded.

    Uses shared input data files from llama3_70b_galaxy for consistency with
    other model implementations. No padding is applied because the Qwen model
    takes logits from the last token position — pad tokens would corrupt output.
    Tile alignment (multiples of 32) ensures the same programs compile regardless
    of small differences between actual and target token counts.
    """
    if seqlen <= 128:
        # Use the same prompt file as other models (Llama, etc.)
        path = f"{SHARED_PROMPTS_DIR}/input_data_questions_prefill_128.json"
        with open(path) as f:
            data = json.load(f)
        inputs = tokenizer(data[0]["prompt"], return_tensors="pt")
        return inputs["input_ids"][:, :seqlen]

    # For long sequences (16k+), use Frankenstein from Project Gutenberg.
    # Feed the raw text and let the model continue it — tests long-context processing.
    if seqlen in _FRANKENSTEIN_CONFIGS:
        idx = _FRANKENSTEIN_CONFIGS[seqlen]
        path = f"{SAMPLE_PROMPTS_DIR}/eval_frankenstein_long.json"
        with open(path) as f:
            data = json.load(f)
        entry = data[idx]
        context = _load_and_cache_context(entry["context"], entry.get("max_length"))
        instruction = entry["prompt"]
        prefix = "<|im_start|>user\n"
        # Seed <think> to start the reasoning chain. At very long contexts (100K+),
        # the DeltaNet recurrent state's finite capacity [B,32,128,128] dilutes the
        # suffix signal, making argmax split ~33% <|im_end|> vs ~15% <think>.
        # Explicit <think> ensures the model enters reasoning mode.
        suffix = f"\n\nBased on the above text: {instruction}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        wrapper_ids = tokenizer(prefix + suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        max_context_tokens = seqlen - wrapper_ids.shape[1]
        context_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"][
            :, :max_context_tokens
        ]
        prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        suffix_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return torch.cat([prefix_ids, context_ids, suffix_ids], dim=1)[:, :seqlen]

    # For medium sequences (1k-8k), use static prompt files
    size_label = f"{seqlen // 1024}k" if seqlen >= 1024 else str(seqlen)
    path = f"{SAMPLE_PROMPTS_DIR}/input_data_long_{size_label}.json"
    with open(path) as f:
        data = json.load(f)
    prompt_text = data[0]["prompt"]
    inputs = tokenizer(prompt_text, return_tensors="pt")
    return inputs["input_ids"][:, :seqlen]


def _warmup_prefill(model, device, token_ids):
    """Run prefill to compile all programs. Discards results.

    Following the Llama/tt_transformers pattern (simple_text_demo.py:1059-1068),
    this separates compilation from inference so TTFT and decode throughput
    reflect actual device compute, not program compilation.

    For long sequences (> 4096), warmup uses a truncated prefix to avoid
    L1 clashes in the non-paged concat path. The paged prefill path compiles
    different kernels (chunked_sdpa, paged_fill_cache) that get compiled
    during the actual prefill_paged call.
    """
    T = token_ids.shape[1]
    # Cap warmup to 4096 tokens: the non-paged concat path hits L1 clashes
    # at 8K+ (attn_chunk_size=4096, second chunk produces 8192-length KV).
    # Paged prefill kernels (chunked_sdpa) are compiled during prefill_paged.
    warmup_tokens = token_ids[:, : min(T, 4096)]
    warmup_len = warmup_tokens.shape[1]
    logger.info(f"Warmup prefill ({warmup_len} tokens) — compiling programs...")
    t0 = time.time()
    logits = model.prefill(warmup_tokens)
    ttnn.synchronize_device(device)

    # Decode warmup is handled by Generator.decode_forward() on the first decode call
    # (it captures its own trace then), so we only warmup prefill here.

    compile_time = time.time() - t0
    logger.info(f"Warmup complete: {compile_time:.1f}s (programs now cached)")

    # Reset state so the actual inference starts clean
    model.reset_state(batch_size=token_ids.shape[0])


BLOCK_SIZE = 64
MAX_NUM_BLOCKS = 2048  # Fixed block budget: 2048 blocks × 64 tokens = 128K token capacity


@run_for_blackhole()
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "seqlen, max_generated_tokens, use_trace",
    [
        (128, 50, True),
        (128, 50, False),
        (4096, 100, True),
        (4096, 100, False),
        (8192, 100, True),
        (8192, 100, False),
        (16384, 100, True),
        (32768, 100, True),
        (65536, 100, True),
        (65536, 100, False),
        (131072, 100, True),
    ],
    ids=[
        "traced_128",
        "paged_128",
        "traced_4k",
        "paged_4k",
        "traced_8k",
        "paged_8k",
        "traced_16k",
        "traced_32k",
        "traced_64k",
        "paged_64k",
        "traced_128k",
    ],
)
def test_demo_text(
    mesh_device,
    seqlen,
    max_generated_tokens,
    use_trace,
):
    """End-to-end text generation: prefill + decode with performance validation."""
    from transformers import AutoTokenizer

    device = mesh_device
    device.enable_program_cache()
    # Fixed block budget — max_seq_len derived from it
    max_seq_len = MAX_NUM_BLOCKS * BLOCK_SIZE

    t0 = time.time()
    model = Qwen35Model.from_pretrained(
        device,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        # n_layers=4,  # uncomment for fast iteration; default uses 32-layer config
    )
    logger.info(f"Model load: {time.time() - t0:.1f}s")
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)

    token_ids = _get_prompt(seqlen, tokenizer)
    actual_len = token_ids.shape[1]
    logger.info(
        f"Prompt: {actual_len} tokens (block budget: {MAX_NUM_BLOCKS} blocks × {BLOCK_SIZE} = {max_seq_len} tokens)"
    )

    # Multi-device (TP): route through the non-traced stateful generate path
    # (prefill fills KV cache + GDN state, then incremental decode). The paged/
    # traced machinery is single-device-only for now, so both traced_* and paged_*
    # configs run via this path on a mesh.
    if model.num_devices > 1:
        generated, perf = _run_tp_generation(model, tokenizer, token_ids, max_generated_tokens)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        logger.info(f"[TP {model.num_devices}-dev] ttft={perf['ttft_s']:.2f}s decode={perf['decode_tok_s']:.2f} tok/s")
        logger.info(f"[TP] GENERATED: {text!r}")
        assert len(generated) == max_generated_tokens, f"{len(generated)} != {max_generated_tokens}"
        assert len(set(generated)) > 1, f"degenerate generation: {generated}"
        return

    # Warmup: compile programs (not counted in TTFT). A short traced prompt takes the masked
    # fixed-bucket path, whose programs are compiled inside capture_prefill_trace_chunked
    # (warmup_prefill_masked_buckets), so the legacy model.prefill warmup is redundant there —
    # and it would hit the pre-existing small-T L1 clash in the non-paged concat path. Skip it.
    PREFILL_CHUNK = 2048
    t_compile = time.time()
    if not (use_trace and actual_len < PREFILL_CHUNK):
        _warmup_prefill(model, device, token_ids)
    t_compile = time.time() - t_compile

    if use_trace:
        generated, perf = _run_traced_generation(
            model,
            tokenizer,
            device,
            token_ids,
            max_generated_tokens,
        )
    else:
        generated, perf = _run_paged_generation(
            model,
            tokenizer,
            device,
            token_ids,
            max_generated_tokens,
        )

    perf["compile_time"] = t_compile
    text = tokenizer.decode(generated, skip_special_tokens=True)
    _log_results(perf, actual_len, len(generated), text)
    _assert_results(perf, actual_len, len(generated))


def _should_use_chunked_trace(model):
    """Whether to capture ONE chunk's prefill forward and replay it per chunk (chunk-outer)
    instead of one whole-sequence trace. The chunk-seq GDN prefill kernel is always on, so
    chunk-outer is always selected: it keeps the captured trace small (under the 4 GiB ceiling
    at long context) with every GDN call at <=2048 tokens.

    ``attn.weights.use_chunk_seq_prefill`` is always True; the gate stays data-driven off the
    GDN weights so a future per-model toggle would flow through here unchanged.
    """
    return any(
        (not layer.is_full_attention)
        and getattr(getattr(layer.attention, "weights", None), "use_chunk_seq_prefill", False)
        for layer in model.layers
    )


def _run_tp_generation(model, tokenizer, token_ids, max_generated_tokens):
    """Multi-device (TP) non-traced generation: stateful prefill (fills KV cache +
    GDN recurrent/conv state) then incremental single-token decode. Returns
    (generated_tokens, perf_dict) with TTFT and decode tok/s."""
    import math

    prompt_ids = token_ids[0].tolist()
    model.reset_tp()
    T = len(prompt_ids)
    T_pad = max(128, math.ceil(T / 128) * 128)
    padded = prompt_ids + [0] * (T_pad - T)

    t0 = time.time()
    logits = model.prefill_tp(torch.tensor([padded], dtype=torch.long), valid_len=T)
    ttnn.synchronize_device(model.device)
    ttft = time.time() - t0

    nxt = int(torch.argmax(logits).item())
    generated = [nxt]
    pos = T
    t1 = time.time()
    for _ in range(max_generated_tokens - 1):
        logits = model.decode_tp(nxt, pos)
        nxt = int(torch.argmax(logits).item())
        generated.append(nxt)
        pos += 1
    ttnn.synchronize_device(model.device)
    dt = time.time() - t1
    n_dec = max(1, len(generated) - 1)
    return generated, {"ttft_s": ttft, "decode_tok_s": (n_dec / dt) if dt > 0 else 0.0}


def _run_traced_generation(model, tokenizer, device, token_ids, max_generated_tokens):
    """Prefill + paged traced decode loop. Returns (generated_tokens, perf_dict).

    Uses paged attention inside the trace: paged_update_cache + paged_sdpa_decode
    run inside the captured trace. Between replays, inputs are updated via DMA.
    No post-trace cache/mask update ops needed.
    """
    T = token_ids.shape[1]

    # Allocate paged KV caches + external DeltaNet state (fixed block budget)
    num_kv_heads = model.args.n_kv_heads
    head_dim = model.args.head_dim
    kv_cache_shape = [MAX_NUM_BLOCKS, num_kv_heads, BLOCK_SIZE, head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)

    # Identity page table (host torch.Tensor — model converts internally)
    page_table = torch.arange(MAX_NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)

    # Prefill via trace capture+replay. The chunk-seq GDN kernel is correct only at
    # <=16 sub-chunks (2048 tokens) per call and is now the only prefill engine, so we
    # always capture ONE chunk's all-layer forward and replay it per chunk (chunk-outer),
    # keeping the captured trace under tt-metal's 4 GiB ceiling at long context.
    assert _should_use_chunked_trace(model), "chunk-seq GDN prefill must be enabled"
    chunk_size = 2048
    bucket_size = ((T + chunk_size - 1) // chunk_size) * chunk_size
    logger.info(f"Capturing prefill trace at bucket_size={bucket_size} (prompt {T} tokens, chunk-outer replay)...")
    t_cap = time.time()
    # Only warm the masked short-prompt buckets when this prompt will actually take the masked
    # path (T < chunk_size). For long prompts the prefill uses chunk-trace replay + eager tail,
    # so warming buckets is wasted capture work (and can perturb the memory state the eager tail
    # compiles into). vLLM keeps the default (warm all buckets once at startup, sizes unknown there).
    model.capture_prefill_trace_chunked(
        device, page_table, chunk_size=chunk_size, warmup_masked_buckets=(T < chunk_size)
    )
    logger.info(f"Prefill trace captured in {time.time() - t_cap:.1f}s")
    pad_len = bucket_size - T
    # Pad with the prompt's last real token rather than 0. The DeltaNet recurrence
    # is sequential and updates state for every input token in the captured bucket;
    # padding with token 0 corrupts state with the embedding of `<pad>`/`<unk>`, while
    # repeating the last real token produces a smoother (still imperfect) post-prefill
    # state. The logit is still extracted at position actual_len-1 so the next-token
    # prediction itself is unaffected — only decode quality is.
    last_token = token_ids[:, -1:].expand(1, pad_len) if pad_len > 0 else token_ids[:, :0]
    padded_token_ids = torch.cat([token_ids, last_token], dim=1)

    t0 = time.time()
    if T < chunk_size:
        # Short prompt (whole prompt fits under one chunk): masked fixed-bucket prefill —
        # bounded program set, no request-time compile, decode-correct GDN state. Mirrors the
        # vLLM prefill_dispatch routing. capture_prefill_trace_chunked above already warmed the
        # bucket programs and put the GDN in in-place mode that the decode trace continues from.
        logits = model.prefill_masked_bucket(token_ids, page_table, actual_len=T)
    else:
        logits = model.prefill_traced_chunked(padded_token_ids, page_table, actual_len=T)
    ttft = time.time() - t0

    logits_torch = ttnn.to_torch(logits).squeeze()
    assert not torch.isnan(logits_torch).any(), "NaN in prefill logits"
    next_token = logits_torch.argmax().item()
    gen = Generator([model], [model.args], device)

    # Capture the decode trace once with GDN-state save/restore so the loop replays from the
    # correct post-prefill state. The stock Generator capture runs the forward twice (compile +
    # capture), which would otherwise double-advance the in-place GDN recurrent state.
    from models.demos.blackhole.qwen3_5_9b.tt.generator_interface import prime_decode_trace

    prime_decode_trace(gen, model, torch.tensor([[next_token]], dtype=torch.long), torch.tensor([T]), page_table)

    generated = [next_token]
    decode_times = []
    current_pos = T

    for i in range(max_generated_tokens - 1):
        t_step = time.time()
        out = gen.decode_forward(
            torch.tensor([[next_token]], dtype=torch.long),
            torch.tensor([current_pos]),
            page_table=page_table,
            kv_cache=None,
            enable_trace=True,
            read_from_device=True,
        )
        decode_times.append(time.time() - t_step)

        dl = (out[0] if isinstance(out, tuple) else out).squeeze().float()
        assert not torch.isnan(dl).any(), f"NaN in traced decode at step {i}"
        next_token = int(dl.argmax())
        generated.append(next_token)
        current_pos += 1

        if next_token == tokenizer.eos_token_id:
            break

    avg_decode = sum(decode_times) / len(decode_times) if decode_times else float("inf")
    return generated, {"ttft": ttft, "avg_decode_s": avg_decode, "decode_steps": len(decode_times)}


def _run_paged_generation(model, tokenizer, device, token_ids, max_generated_tokens):
    """Prefill + paged decode loop (non-traced). Returns (generated_tokens, perf_dict).

    Uses paged KV cache for attention layers. DeltaNet uses external state.
    """
    T = token_ids.shape[1]

    # Allocate paged KV caches + external DeltaNet state (fixed block budget)
    num_kv_heads = model.args.n_kv_heads
    head_dim = model.args.head_dim
    kv_cache_shape = [MAX_NUM_BLOCKS, num_kv_heads, BLOCK_SIZE, head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)

    # Identity page table (host torch.Tensor)
    page_table = torch.arange(MAX_NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)

    # Prefill
    t0 = time.time()
    logits = model.prefill_paged(token_ids, page_table)
    ttnn.synchronize_device(device)
    ttft = time.time() - t0

    logits_torch = ttnn.to_torch(logits).squeeze()
    assert not torch.isnan(logits_torch).any(), "NaN in paged prefill logits"
    next_token = logits_torch.argmax().item()

    gen = Generator([model], [model.args], device)

    generated = [next_token]
    decode_times = []

    for i in range(max_generated_tokens - 1):
        t_step = time.time()
        out = gen.decode_forward(
            torch.tensor([[next_token]], dtype=torch.long),
            torch.tensor([T + i]),
            page_table=page_table,
            kv_cache=None,
            enable_trace=False,
            read_from_device=True,
        )
        decode_times.append(time.time() - t_step)

        dl = (out[0] if isinstance(out, tuple) else out).squeeze().float()
        assert not torch.isnan(dl).any(), f"NaN in paged decode logits at step {i}"
        next_token = int(dl.argmax())

        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)

    avg_decode = sum(decode_times) / len(decode_times) if decode_times else float("inf")
    return generated, {"ttft": ttft, "avg_decode_s": avg_decode, "decode_steps": len(decode_times)}


def _log_results(perf, prompt_len, num_generated, text):
    ttft = perf["ttft"]
    avg_ms = perf["avg_decode_s"] * 1000
    tok_s = 1000.0 / avg_ms if avg_ms > 0 else 0
    compile_time = perf.get("compile_time", 0)

    logger.info("=" * 70)
    logger.info(f"  Compile (warmup):    {compile_time:.3f}s")
    logger.info(f"  Prefill {prompt_len} tokens:  TTFT = {ttft:.3f}s ({prompt_len / ttft:.0f} tok/s)")
    logger.info(f"  Decode:  {avg_ms:.1f}ms/token  ({tok_s:.1f} tok/s)")
    logger.info(f"  Generated {num_generated} tokens in {perf['decode_steps']} steps")
    logger.info(f"  Text: {text[:200]}")
    logger.info("=" * 70)


def _assert_results(perf, prompt_len, num_generated):
    assert num_generated >= 1, "Should generate at least 1 token"

    targets = PERF_TARGETS.get(prompt_len)
    if targets is None:
        return

    # Decode throughput is only meaningful with enough steps — the first decode
    # includes compilation overhead and a single sample is not representative.
    if perf["decode_steps"] >= 3:
        tok_s = 1.0 / perf["avg_decode_s"] if perf["avg_decode_s"] > 0 else 0
        min_tok_s = targets["min_decode_tok_s"]
        assert (
            tok_s >= min_tok_s
        ), f"Decode throughput {tok_s:.1f} tok/s below target {min_tok_s} tok/s at seqlen={prompt_len}"

    max_ttft = targets["max_ttft_s"]
    assert perf["ttft"] < max_ttft, f"TTFT {perf['ttft']:.1f}s exceeds target {max_ttft}s at seqlen={prompt_len}"
