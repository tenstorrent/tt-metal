# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-9B end-to-end text generation test on Blackhole P150.

Consolidates all e2e demo tests into a single parametrized test that covers:
  - Short prompt text generation (128 tokens)
  - Medium prefill with traced decode (2048 tokens)
  - Long-context prefill + decode (4k, 8k tokens)

Run all:    pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s
Run short:  pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "prefill_128"
Run 2k:     pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "prefill_2k"
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
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import determine_device_name

# Multi-device (TP) is selected via MESH_DEVICE. The 27B (default) runs on a P150x4
# (1,4) Blackhole mesh; the 9B can run on a single P150 (1,1) via MESH_DEVICE=P150.
# P150x8 is a (1,8) LINE of 8 P150s (BH LoudBox) for TP=8 — the 4 KV heads are replicated
# 2x across the mesh (see attention/tp.py). The mesh MUST be a 1-D line (1,8), not (2,4):
# the TP collectives take the reduce_scatter-over-line path only when an axis is 1.
# On a single device the model runs its validated single-device path, on a multi-device
# mesh it needs a 1-D fabric for the TP collectives (see tp_common notes). The 4-device
# configs use FABRIC_1D; the 8-device (1,8) LoudBox uses FABRIC_1D_RING — plain FABRIC_1D
# has no deadlock avoidance and wedges the erisc routers under full-context (>=64K) CCL
# traffic, whereas FABRIC_1D_RING adds dateline deadlock avoidance (its larger router now
# fits the ETH kernel-config buffer, see bh dev_mem_map.h). Ring vs line topology of the
# CCL op itself is unchanged (ccl_topology() stays Linear); only the fabric layer changes.
_MESH_SHAPE = {"P150": (1, 1), "P150x4": (1, 4), "P150x8": (1, 8)}.get(os.environ.get("MESH_DEVICE"), (1, 4))
_MULTI = _MESH_SHAPE != (1, 1)
_TP_FABRIC = ttnn.FabricConfig.FABRIC_1D_RING if _MESH_SHAPE == (1, 8) else ttnn.FabricConfig.FABRIC_1D
# Multi-device (TP) long-context prefill replays a captured per-chunk trace, so the mesh
# needs a trace region (ttnn's DEFAULT_TRACE_REGION_SIZE is 0). 256 MiB matches the validated
# TP serving config and is ample for the single 2048-token chunk trace — negligible vs the
# per-device DRAM that chunk-outer prefill frees. Single-device params are left unchanged.
_TP_TRACE_REGION_SIZE = 256 * 1024 * 1024
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        **({"fabric_config": _TP_FABRIC, "trace_region_size": _TP_TRACE_REGION_SIZE} if _MULTI else {}),
    }
]

SAMPLE_PROMPTS_DIR = "models/demos/blackhole/qwen36/demo/sample_prompts"
SHARED_PROMPTS_DIR = "models/demos/llama3_70b_galaxy/demo/sample_prompts"


# Frankenstein prompt config: seqlen → json_index in eval_frankenstein_long.json.
# Each entry clips Frankenstein text to enough characters to exceed the target token count
# (English text ≈ 4.3 chars/token). Full Frankenstein is ~450K chars ≈ 104K tokens.
_FRANKENSTEIN_CONFIGS = {
    8192: 0,  # 70k chars ≈ 16k tokens (covers 8k with margin)
    16384: 1,  # 140k chars ≈ 32k tokens
    32768: 1,  # 140k chars ≈ 32k tokens
    65536: 2,  # 300k chars ≈ 70k tokens
    131072: 3,  # 500k chars ≈ 104k tokens (full text, Frankenstein caps at ~104k tokens)
    # 256k needs a corpus longer than Frankenstein (~104k tokens). Index 4 is War and Peace
    # (pg2600, ~3.2M chars) clipped to 1.2M chars ≈ 256k tokens so the prompt actually fills the
    # context (the actual_len length guard in test_demo_text enforces this).
    262144: 4,
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
    # QWEN35_REF_PROMPT=1: replicate the REFERENCE 27B's exact 64k task (quote-extraction + AI
    # metaphors) for an apples-to-apples comparison — same corpus (pg84.txt), the reference's
    # "You are a helpful assistant" system prompt + apply_chat_template (default thinking), and NO
    # manual <think> seed. Resolves whether the demo's summary-gibberish is a forward bug or just
    # the hard generative-summary task (the reference is coherent on this extractive task, greedy).
    if os.environ.get("QWEN35_REF_PROMPT") and seqlen >= 4096:
        with open(f"{SHARED_PROMPTS_DIR}/input_data_long_64k.json") as f:
            rd = json.load(f)[0]
        context = _load_and_cache_context(rd["context"], rd.get("max_length"))
        instruction = rd["prompt"]
        sys_msg = "You are a helpful assistant."
        overhead = len(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": "\n\n" + instruction}],
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        context = tokenizer.decode(ctx_ids[: max(0, seqlen - overhead - 8)])
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": context + "\n\n" + instruction}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :seqlen]

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
        # QWEN35_NO_THINK=1 disables thinking (Qwen /no_think + no <think> seed) to test whether
        # the long-context degeneration is the known thinking-mode loop vs a real forward issue.
        if os.environ.get("QWEN35_NO_THINK"):
            # Correct Qwen enable_thinking=False scaffold: seed an EMPTY <think></think> block
            # (chat_template.jinja line 150) so the model skips reasoning. The bare "/no_think" text
            # token is ignored by the model in this manual-prompt path (observed: it thinks anyway).
            suffix = (
                f"\n\nBased on the above text: {instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        else:
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
PREFILL_CHUNK = 2048  # chunked-prefill chunk; prompts are padded up to a multiple of this
# 256k ceiling: 4096 blocks × 64 tokens = 262144 token capacity = the model's native context.
# The block budget (and the max_seq_len + KV cache + RoPE table derived from it) is sized per-seqlen
# via _blocks_for, so short tests (128, 4k) keep a cheap cache and only the 256k case pays for 256k.
MAX_BLOCK_BUDGET = 4096


def _blocks_for(seqlen, max_generated_tokens):
    """Paged-KV block budget for `seqlen`, rounded up to a block multiple and capped at the 256k
    ceiling. Sized to cover the padded prefill bucket (seqlen rounded up to PREFILL_CHUNK) PLUS the
    tokens we will decode, so prompt + generation both fit in the RoPE table / KV cache / position
    space. Floored at 64 blocks (4k) so short prompts still get a sane cache."""
    bucket = ((seqlen + PREFILL_CHUNK - 1) // PREFILL_CHUNK) * PREFILL_CHUNK
    needed = bucket + max_generated_tokens
    blocks = max(64, (needed + BLOCK_SIZE - 1) // BLOCK_SIZE)
    # Flexible chunked SDPA reads the page table as a ROW_MAJOR int32 stick of width num_blocks;
    # sdpa_program_factory requires page_table_stick_size (= num_blocks * 4 bytes) % 32 == 0, i.e.
    # num_blocks % 8 == 0. Round up (only enlarges the cache by <=7 blocks; the 4096 cap is %8).
    blocks = ((blocks + 7) // 8) * 8
    return min(MAX_BLOCK_BUDGET, blocks)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "seqlen, max_generated_tokens, use_trace, repeat_batches",
    [
        pytest.param(128, 50, True, 1, id="traced_128"),
        pytest.param(128, 50, False, 1, id="paged_128"),
        pytest.param(4096, 100, True, 1, id="traced_4k"),
        pytest.param(4096, 100, False, 1, id="paged_4k"),
        pytest.param(8192, 500, True, 1, id="traced_8k"),
        pytest.param(8192, 100, False, 1, id="paged_8k"),
        pytest.param(16384, 100, True, 1, id="traced_16k"),
        pytest.param(32768, 100, True, 1, id="traced_32k"),
        pytest.param(65536, 500, True, 1, id="traced_64k"),
        pytest.param(65536, 100, False, 1, id="paged_64k"),
        pytest.param(131072, 100, True, 1, id="traced_128k"),
        pytest.param(262144, 100, True, 1, id="traced_256k"),
        pytest.param(128, 50, True, 2, id="determinism_128"),
    ],
)
def test_demo_text(
    mesh_device,
    seqlen,
    max_generated_tokens,
    use_trace,
    repeat_batches,
):
    """End-to-end text generation: prefill + decode with performance validation."""
    from transformers import AutoTokenizer

    device = mesh_device
    device.enable_program_cache()
    # Per-seqlen block budget — max_seq_len (and the KV cache + RoPE table) derived from it.
    # Sized to hold the padded prompt bucket plus the decoded tokens.
    num_blocks = _blocks_for(seqlen, max_generated_tokens)
    max_seq_len = num_blocks * BLOCK_SIZE

    t0 = time.time()
    model = Qwen36Model.from_pretrained(
        device,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        # n_layers=4,  # uncomment for fast iteration; default uses 32-layer config
    )
    logger.info(f"Model load: {time.time() - t0:.1f}s")
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)

    token_ids = _get_prompt(seqlen, tokenizer)
    # Reserve context for the tokens we generate: prefill writes the padded prompt bucket and decode
    # extends by max_generated_tokens, so prompt + generation must fit within max_seq_len (the RoPE
    # table / KV cache / position span). This only bites when the prompt fills the whole context
    # (the 256k case, where seqlen == the model's native ceiling); smaller prompts sit well under it.
    # Floor to a 128-multiple (GDN sub-chunk) so program shapes stay aligned.
    max_prompt_len = ((max_seq_len - max_generated_tokens) // 128) * 128
    if token_ids.shape[1] > max_prompt_len:
        token_ids = token_ids[:, :max_prompt_len]
    actual_len = token_ids.shape[1]
    # Long-context prompts are built from a corpus and clipped to seqlen; if the corpus is too
    # short the clip silently shortens the prompt (e.g. a 256k case quietly running at ~104k).
    # Guard the cases backed by the large (War and Peace, index 4) corpus, which is sized to fill
    # the context, so they fail loudly instead of silently under-running. The Frankenstein-backed
    # cases (indices 0-3) intentionally cap at ~104k tokens and are not guarded.
    if _FRANKENSTEIN_CONFIGS.get(seqlen) == 4:
        assert (
            actual_len >= 0.95 * seqlen
        ), f"prompt clipped to {actual_len} tokens, expected ~{seqlen} (corpus too short for seqlen={seqlen})"
    logger.info(
        f"Prompt: {actual_len} tokens (block budget: {num_blocks} blocks x {BLOCK_SIZE} = {max_seq_len} tokens)"
    )

    # Multi-device (TP): route through the chunk-outer prefill + paged decode path.
    # Prefill runs each ~2048-token chunk through all layers, carrying GDN recurrent/
    # conv state + paged KV across chunks (so the GDN seq kernel never sees the whole
    # sequence — the long-context OOM fix), then incremental paged single-token decode.
    if model.num_devices > 1:
        if repeat_batches > 1:
            results = []
            for run in range(repeat_batches):
                gen, _ = _run_tp_generation(model, tokenizer, token_ids, max_generated_tokens, num_blocks)
                results.append(gen)
                if run < repeat_batches - 1:
                    model.free_kv_caches()
            for i in range(1, repeat_batches):
                assert results[0] == results[i], (
                    f"Non-deterministic output between run 0 and run {i}.\n"
                    f"Run 0: {results[0]}\nRun {i}: {results[i]}"
                )
            return
        generated, perf = _run_tp_generation(model, tokenizer, token_ids, max_generated_tokens, num_blocks)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        logger.info(f"[TP {model.num_devices}-dev] ttft={perf['ttft_s']:.2f}s decode={perf['decode_tok_s']:.2f} tok/s")
        logger.info(f"[TP] GENERATED: {text!r}")
        assert len(generated) == max_generated_tokens, f"{len(generated)} != {max_generated_tokens}"
        assert len(set(generated)) > 1, f"degenerate generation: {generated}"
        # Emit perf metrics for the centralized target check (no-op outside CI). Perf is
        # NOT asserted here — validate_perf_targets.py compares this against model_targets.yaml.
        _save_tp_benchmark(perf, model, seqlen=seqlen, prompt_len=actual_len, num_generated=len(generated))
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

    if repeat_batches > 1:
        results = []
        for run in range(repeat_batches):
            if use_trace:
                gen, _ = _run_traced_generation(model, tokenizer, device, token_ids, max_generated_tokens, num_blocks)
            else:
                gen, _ = _run_paged_generation(model, tokenizer, device, token_ids, max_generated_tokens, num_blocks)
            results.append(gen)
            if run < repeat_batches - 1:
                model.free_kv_caches()
        for i in range(1, repeat_batches):
            assert results[0] == results[i], (
                f"Non-deterministic output between run 0 and run {i}.\n" f"Run 0: {results[0]}\nRun {i}: {results[i]}"
            )
        return

    if use_trace:
        generated, perf = _run_traced_generation(
            model,
            tokenizer,
            device,
            token_ids,
            max_generated_tokens,
            num_blocks,
        )
    else:
        generated, perf = _run_paged_generation(
            model,
            tokenizer,
            device,
            token_ids,
            max_generated_tokens,
            num_blocks,
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


def _run_tp_generation(model, tokenizer, token_ids, max_generated_tokens, num_blocks):
    """Multi-device (TP) generation via traced chunk-outer prefill + paged decode.

    Prefill captures ONE 2048-token chunk's all-layer forward as a trace and replays it
    per chunk (chunk-outer): each chunk passes through all layers while the GDN recurrent/
    conv state and the paged KV cache carry across chunks in place, so the GDN seq kernel
    never materializes full-sequence float32 tensors (the long-context OOM fix). The traced
    replay DMA-s per-chunk inputs into persistent buffers; the eager (non-traced) fallback
    instead allocates fresh host tensors every chunk and crashes past ~7872 tokens
    (command-queue pressure), so it cannot reach long ISL. Decode then continues from the
    carried state via the standard paged contract (prepare_inputs_decode /
    ttnn_decode_forward / process_output_decode). Returns (generated_tokens, perf_dict).

    Mirrors the validated TP flow in tests/test_model_tp_contract.py
    (test_model_tp_long_prefill_traced) and tt/qwen36_vllm.py. prefill_tp / decode_tp /
    generate_tp are left as the bespoke oracle those tests compare against.
    """
    vocab = model.args.vocab_size
    T = token_ids.shape[1]

    # Benchmark profiler (no-op outside CI). Brackets the phases the centralized perf check
    # consumes: compile_prefill (trace capture), inference_prefill (TTFT), compile_decode
    # (decode trace capture) and inference_decode (steady-state throughput).
    profiler = BenchmarkProfiler()
    profiler.start("run")

    # The flexible (position-general) chunked SDPA requires the page-table width to be a
    # multiple of 32; round the block budget up so both the captured chunk trace's page
    # table and the per-request page table satisfy it. Mirrors test_model_tp_long_prefill_traced.
    num_blocks = ((num_blocks + 31) // 32) * 32

    # Allocate the replicated paged KV cache for the 8 full-attention layers and reset/
    # bind the GDN state (sets _stable_state so prefill/decode update it in place). The
    # per-device cache holds n_local_kv_heads (NOT n_kv_heads) — at TP=4 that is 1.
    kv_cache_shape = [num_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)
    # Identity page table covering the whole block budget (prompt + generated tokens).
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)

    # ---- Capture the per-chunk prefill trace (warmup; NOT counted in TTFT). ----
    # Also warms the masked-bucket programs (short prompt + the long-prompt tail) before the
    # trace is parked, so a request never compiles a program that could clobber the trace.
    CHUNK = 2048
    t_cap = time.time()
    profiler.start("compile_prefill")
    model.capture_prefill_trace_chunked(model.mesh_device, page_table, chunk_size=CHUNK)
    profiler.end("compile_prefill")
    logger.info(f"[TP] prefill chunk-trace captured in {time.time() - t_cap:.1f}s")

    # ---- Chunk-outer prefill (real prompt only; the tail is masked internally). ----
    t0 = time.time()
    profiler.start("inference_prefill")
    logits_dev = model.prefill_traced_chunked(token_ids[:, :T], page_table, actual_len=T)
    ttnn.synchronize_device(model.device)
    profiler.end("inference_prefill")
    ttft = time.time() - t0

    # Decode token selection: greedy by default; QWEN35_TEMP>0 enables temperature sampling.
    _temp = float(os.environ.get("QWEN35_TEMP", "0") or 0)
    # Anti-repetition (both default OFF, decoding-only). On soft long-context logits (>=64k) greedy
    # decode can loop; QWEN35_REP_PENALTY (HF-style, ~1.3) discounts emitted tokens and
    # QWEN35_NO_REPEAT_NGRAM (~3) blocks repeating n-grams. Off by default because over long runs
    # (~100+ tok) the cumulative penalty poisons common tokens and no-repeat-ngram cascades at high
    # entropy; the loops are a decode-drift symptom (forward/retrieval are correct) and vLLM samples.
    _rep_pen = float(os.environ.get("QWEN35_REP_PENALTY", "1.0") or 1.0)
    _no_repeat = int(os.environ.get("QWEN35_NO_REPEAT_NGRAM", "0") or 0)
    generated = []

    def _pick(vec):
        v = vec.float()
        if _rep_pen != 1.0 and generated:
            idx = torch.tensor(sorted(set(generated)))
            s = v[idx]
            v[idx] = torch.where(s > 0, s / _rep_pen, s * _rep_pen)
        if _no_repeat > 0 and len(generated) >= _no_repeat - 1:
            prefix = tuple(generated[-(_no_repeat - 1) :]) if _no_repeat > 1 else ()
            n = _no_repeat
            for i in range(len(generated) - n + 1):
                if tuple(generated[i : i + n - 1]) == prefix:
                    v[generated[i + n - 1]] = float("-inf")
        if _temp > 0:
            return int(torch.multinomial(torch.softmax(v / _temp, dim=-1), 1).item())
        return int(torch.argmax(v).item())

    # Logits are replicated across the mesh ([1,1,vocab]); gather one replica.
    lt = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
    nxt = _pick(lt.reshape(-1, vocab)[0])
    generated.append(nxt)

    # ---- Traced paged single-token decode, continuing from the carried GDN + KV state. ----
    # Decode is captured ONCE as a trace and replayed with ttnn.execute_trace, so each step costs
    # a single device dispatch instead of re-issuing every op of all 64 layers from the host. The
    # eager loop (one ttnn_decode_forward + host readback per token) was the 27B decode bottleneck
    # (~3.5 tok/s -> ~18 tok/s traced). Mirrors the single-device _run_traced_generation here and
    # the validated TP traced decode in qwen35_27b/.../test_e2e_generate.py.
    #
    # The begin/end_trace_capture run is a THROWAWAY: in this tt-metal version its output is
    # unreliable and it advances the GDN recurrent/conv state (and writes paged KV at pos T). So we
    # snapshot the post-prefill GDN state, capture, then RESTORE it — preserving buffer addresses so
    # the trace stays valid. (KV[T] is harmlessly overwritten by the first real replay; positions
    # <T were written by the real prefill.) Every real decode step is then a pure execute_trace
    # replay, which is bit-faithful to an eager decode re-issue (PCC == 1.0).
    # Snapshot/restore is O(GDN state) and context-length independent — unlike re-prefill it does
    # not re-run multi-chunk prefill (which corrupted decode at >=4k). QWEN35_TP_DECODE_EAGER=1
    # forces the old eager loop (A/B comparison / fallback).
    from models.tt_transformers.tt.common import copy_host_to_device

    mesh = model.mesh_device
    eager = os.environ.get("QWEN35_TP_DECODE_EAGER") == "1"

    def _read(out):
        return _pick(model.process_output_decode(out, B=1, S=1).reshape(-1)[:vocab])

    def _update(token, position):
        host = model.prepare_decode_inputs_host(
            torch.tensor([[token]], dtype=torch.int32),
            torch.tensor([position], dtype=torch.int32),
            page_table=page_table,
        )
        copy_host_to_device(host, device_tensors=dev)

    # GDN recurrent/conv state is per-device (each TP rank owns its value heads); snapshot ALL
    # ranks (ConcatMeshToTensor) and restore by sharding back into the SAME buffers (ttnn.copy
    # preserves the addresses the trace baked in). Only GDN layers carry recurrent state.
    _gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]

    def _snapshot_gdn():
        comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
        return [
            (
                ttnn.to_torch(dn.rec_state, mesh_composer=comp),
                [ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states],
            )
            for dn in _gdn
        ]

    def _restore_gdn(snap):
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)

        def _back(t, dtype):
            # Match the target buffer dtype — rec_state may be fp32 (fp32 by default);
            # restoring a hardcoded-bf16 tensor into an fp32 buffer corrupts the state.
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)

        for dn, (rec, convs) in zip(_gdn, snap):
            r = _back(rec, dn.rec_state.dtype)
            ttnn.copy(r, dn.rec_state)
            ttnn.deallocate(r)
            for j, c in enumerate(convs):
                cc = _back(c, dn.conv_states[j].dtype)
                ttnn.copy(cc, dn.conv_states[j])
                ttnn.deallocate(cc)

    # Persistent device input buffers (token, cur_pos, packed rope, page_table).
    dev = model.prepare_inputs_decode(
        torch.tensor([[nxt]], dtype=torch.int32),
        torch.tensor([T], dtype=torch.int32),
        page_table=page_table,
    )

    trace_id = None
    tt_logits = None
    profiler.start("compile_decode")
    if not eager:
        gdn_snap = _snapshot_gdn()  # exact post-prefill GDN state
        # Compile the decode programs (eager) then capture a throwaway trace; both advance GDN
        # state, so restore the snapshot afterward.
        model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        tt_logits, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        _restore_gdn(gdn_snap)
    profiler.end("compile_decode")

    pos = T
    decode_times = []
    profiler.start("inference_decode")
    while len(generated) < max_generated_tokens:
        _update(nxt, pos)
        t_step = time.time()
        if eager:
            tt_logits, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        else:
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        decode_times.append(time.time() - t_step)
        nxt = _read(tt_logits)
        generated.append(nxt)
        pos += 1
    if trace_id is not None:
        ttnn.release_trace(mesh, trace_id)
    profiler.end("inference_decode")

    # Steady-state throughput (drop the first step, which can carry one-time costs).
    steady = decode_times[1:] if len(decode_times) > 1 else decode_times
    avg = (sum(steady) / len(steady)) if steady else float("inf")
    profiler.end("run")
    return generated, {"ttft_s": ttft, "decode_tok_s": (1.0 / avg) if avg > 0 else 0.0, "profiler": profiler}


def _run_traced_generation(model, tokenizer, device, token_ids, max_generated_tokens, num_blocks):
    """Prefill + paged traced decode loop. Returns (generated_tokens, perf_dict).

    Uses paged attention inside the trace: paged_update_cache + paged_sdpa_decode
    run inside the captured trace. Between replays, inputs are updated via DMA.
    No post-trace cache/mask update ops needed.
    """
    T = token_ids.shape[1]

    # Allocate paged KV caches + external DeltaNet state (per-seqlen block budget)
    num_kv_heads = model.args.n_kv_heads
    head_dim = model.args.head_dim
    kv_cache_shape = [num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)

    # Identity page table (host torch.Tensor — model converts internally)
    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

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
    from models.demos.blackhole.qwen36.tt.generator_interface import prime_decode_trace

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


def _run_paged_generation(model, tokenizer, device, token_ids, max_generated_tokens, num_blocks):
    """Prefill + paged decode loop (non-traced). Returns (generated_tokens, perf_dict).

    Uses paged KV cache for attention layers. DeltaNet uses external state.
    """
    T = token_ids.shape[1]

    # Allocate paged KV caches + external DeltaNet state (per-seqlen block budget)
    num_kv_heads = model.args.n_kv_heads
    head_dim = model.args.head_dim
    kv_cache_shape = [num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)

    # Identity page table (host torch.Tensor)
    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

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


def _save_tp_benchmark(perf, model, seqlen, prompt_len, num_generated):
    """Emit a benchmark partial-run JSON for the centralized perf-target check.

    No-op outside CI (BenchmarkData/save self-gate on CI=true). The standalone
    .github/scripts/utils/validate_perf_targets.py pass compares these metrics against
    models/model_targets.yaml. batch_size is 1, so decode tokens/s == tokens/s/user, and
    prefill_time_to_token is reported in seconds (the validator converts the ms target).

    input_sequence_length is the NOMINAL ``seqlen`` (128, 4096, ...), not the tile-clipped
    actual prompt length, because target lookup requires an exact seq_len match against the
    per-ISL entries in model_targets.yaml. prefill_t/s still uses the actual tokens processed.
    """
    profiler = perf["profiler"]
    ttft_s = perf["ttft_s"]
    decode_tok_s = perf["decode_tok_s"]
    measurements = {
        "compile_prefill": profiler.get_duration("compile_prefill"),
        "compile_decode": profiler.get_duration("compile_decode"),
        "prefill_t/s": (prompt_len / ttft_s) if ttft_s > 0 else 0.0,
        "prefill_time_to_token": ttft_s,
        "decode_t/s": decode_tok_s,
        "decode_t/s/u": decode_tok_s,
    }
    benchmark_data = create_benchmark_data(profiler, measurements, {"inference_prefill": 0, "inference_decode": 1}, {})
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="demo",
        ml_model_name=model.args.base_model_name,
        ml_model_type="llm",
        device_name=determine_device_name(model.mesh_device),
        num_layers=model.args.n_layers,
        batch_size=1,
        input_sequence_length=seqlen,
        output_sequence_length=num_generated,
    )


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
    logger.info(f"  Text: {text[:6000]}")
    logger.info("=" * 70)


def _assert_results(perf, prompt_len, num_generated):
    # Correctness only. Perf/accuracy targets are checked by the centralized
    # validate_perf_targets.py pass against models/model_targets.yaml, not asserted here.
    assert num_generated >= 1, "Should generate at least 1 token"
