# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5/3.6 end-to-end text generation test on Blackhole (P150 / P150x4).

A single parametrized test covering prefill + decode across ISLs from 128 up to 256k
(single-user) and batched serving (B=8/B=32, multi-device TP) up to 64k.

Run all:      pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s
Run 128:      pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128"
Run batched:  MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "b8"

GDN prefill runs the fast fused path by DEFAULT — no env vars needed: chunk-parallel phase-split
(PREP fanned across the grid + V-block SCAN), fp32 o output, fp32 state, and flat token-major q/k/v
with in-kernel L2-norm (eliminates the head-split relayouts + host l2_norm — the bulk of the
preprocessing cost). Two opt-out flags exist only for benchmarking/debug:
  QWEN_GDN_PHASED=0    fall back to the monolithic single-kernel fused op (no phase split).
  QWEN_GDN_FLAT_QKV=0  fall back to head-split q/k/v + host l2_norm (no flat token-major reads).
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
from tracy import signpost

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import determine_device_name

_MESH_SHAPE = {"P150": (1, 1), "P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 4))
_MULTI = _MESH_SHAPE != (1, 1)
_TP_TRACE_REGION_SIZE = 1024 * 1024 * 1024
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        **(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": _TP_TRACE_REGION_SIZE} if _MULTI else {}
        ),
    }
]

SAMPLE_PROMPTS_DIR = "models/demos/blackhole/qwen36/demo/sample_prompts"
SHARED_PROMPTS_DIR = "models/demos/llama3_70b_galaxy/demo/sample_prompts"


_FRANKENSTEIN_CONFIGS = {
    8192: 0,
    16384: 1,
    32768: 1,
    65536: 2,
    131072: 3,  # Frankenstein caps ~104k
    262144: 4,  # War and Peace (full context)
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


def _get_prompt(seqlen, tokenizer, max_prompt_len=None):
    """Load ~seqlen tokens (clipped, not padded). Tile-aligned; no pad tokens (last-token logits)."""
    cap = seqlen if max_prompt_len is None else min(seqlen, max_prompt_len)
    # QWEN35_REF_PROMPT=1: match the reference 27B 64k extractive task (pg84, default chat template, no thinking seed).
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
        context = tokenizer.decode(ctx_ids[: max(0, cap - overhead - 8)])
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": context + "\n\n" + instruction}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :cap]

    if seqlen <= 256:
        path = f"{SHARED_PROMPTS_DIR}/input_data_questions_prefill_128.json"
        with open(path) as f:
            data = json.load(f)
        ids = tokenizer(data[0]["prompt"], return_tensors="pt")["input_ids"]
        # The shared prompt is ~128 tokens; repeat it to guarantee >= cap tokens, then clip.
        while ids.shape[1] < cap:
            ids = torch.cat([ids, ids], dim=1)
        return ids[:, :cap]

    # Long sequences (16k+): Frankenstein corpus + continuation task
    if seqlen in _FRANKENSTEIN_CONFIGS:
        idx = _FRANKENSTEIN_CONFIGS[seqlen]
        path = f"{SAMPLE_PROMPTS_DIR}/eval_frankenstein_long.json"
        with open(path) as f:
            data = json.load(f)
        entry = data[idx]
        context = _load_and_cache_context(entry["context"], entry.get("max_length"))
        instruction = entry["prompt"]
        prefix = "<|im_start|>user\n"
        # Seed <think> for reasoning; QWEN35_NO_THINK=1 disables it
        if os.environ.get("QWEN35_NO_THINK"):
            # Empty thinking block (enable_thinking=False)
            suffix = (
                f"\n\nBased on the above text: {instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        else:
            suffix = f"\n\nBased on the above text: {instruction}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        wrapper_ids = tokenizer(prefix + suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        max_context_tokens = cap - wrapper_ids.shape[1]
        context_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"][
            :, :max_context_tokens
        ]
        prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        suffix_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return torch.cat([prefix_ids, context_ids, suffix_ids], dim=1)[:, :cap]

    # Medium sequences (1k–8k): static prompt files
    size_label = f"{seqlen // 1024}k" if seqlen >= 1024 else str(seqlen)
    path = f"{SAMPLE_PROMPTS_DIR}/input_data_long_{size_label}.json"
    with open(path) as f:
        data = json.load(f)
    prompt_text = data[0]["prompt"]
    inputs = tokenizer(prompt_text, return_tensors="pt")
    return inputs["input_ids"][:, :cap]


def _warmup_prefill(model, device, token_ids):
    """Warmup prefill to compile programs (results discarded). Truncates at 4096 for non-paged path."""
    T = token_ids.shape[1]
    warmup_tokens = token_ids[:, : min(T, 4096)]
    warmup_len = warmup_tokens.shape[1]
    logger.info(f"Warmup prefill ({warmup_len} tokens) — compiling programs...")
    t0 = time.time()
    logits = model.prefill(warmup_tokens)
    ttnn.synchronize_device(device)

    compile_time = time.time() - t0
    logger.info(f"Warmup complete: {compile_time:.1f}s (programs now cached)")

    # Reset state before timed inference
    model.reset_state(batch_size=token_ids.shape[0])


BLOCK_SIZE = 64
PREFILL_CHUNK = 2048
# 256k ceiling (4096×64 tokens); _blocks_for sizes the cache per seqlen so short tests stay cheap.
MAX_BLOCK_BUDGET = 4096


def _blocks_for(seqlen, max_generated_tokens):
    """Paged-KV block budget for ``seqlen`` (prompt bucket + decode, capped at 256k, min 4k)."""
    bucket = ((seqlen + PREFILL_CHUNK - 1) // PREFILL_CHUNK) * PREFILL_CHUNK
    needed = bucket + max_generated_tokens
    blocks = max(64, (needed + BLOCK_SIZE - 1) // BLOCK_SIZE)
    # Round num_blocks up to a multiple of 8 for SDPA page-table stick alignment (<=7 extra blocks).
    blocks = ((blocks + 7) // 8) * 8
    return min(MAX_BLOCK_BUDGET, blocks)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "seqlen, max_generated_tokens, use_trace, batch, repeat_batches",
    [
        pytest.param(128, 50, True, 1, 1, id="traced_128"),
        pytest.param(4096, 100, True, 1, 1, id="traced_4k"),
        pytest.param(8192, 100, True, 1, 1, id="traced_8k"),
        pytest.param(16384, 100, True, 1, 1, id="traced_16k"),
        pytest.param(32768, 100, True, 1, 1, id="traced_32k"),
        pytest.param(65536, 500, True, 1, 1, id="traced_64k"),
        pytest.param(131072, 100, True, 1, 1, id="traced_128k"),
        pytest.param(262144, 100, True, 1, 1, id="traced_256k", marks=pytest.mark.timeout(900)),
        # Determinism: re-run the traced 128 case and assert identical output across runs.
        pytest.param(128, 50, True, 1, 2, id="determinism_128"),
        # Batched decode (TP only): B users share one paged KV + batched GDN state.
        pytest.param(128, 50, True, 8, 1, id="batched_128_b8"),
        pytest.param(128, 50, True, 32, 1, id="batched_128_b32"),
        # 128<T<=256 → grouped single-pass at B=2 groups (batched-GDN L1 ceiling for bucket 256).
        pytest.param(256, 50, True, 8, 1, id="batched_256_b8"),
        pytest.param(256, 50, True, 32, 1, id="batched_256_b32"),
        # T>256 → prefill_chunked_peruser (per-user; GDN can't batch large chunks).
        pytest.param(4096, 50, True, 8, 1, id="batched_4k_b8"),
        pytest.param(4096, 50, True, 32, 1, id="batched_4k_b32"),
        # B=8 long-context ladder. Paged KV scales as B x ISL (~1 GB/device at 8k to ~8 GB at
        # 64k), within the P150x4 budget. Each user prefilled via prefill_chunked_peruser, then
        # all 8 decode together in one B-wide trace; identical prompts decode identically.
        pytest.param(8192, 50, True, 8, 1, id="batched_8k_b8"),
        pytest.param(16384, 50, True, 8, 1, id="batched_16k_b8"),
        pytest.param(32768, 50, True, 8, 1, id="batched_32k_b8"),
        # Per-user prefill is sequential, so the 64k TTFT (~357s) exceeds pytest.ini's 300s
        # default; give it a generous per-test timeout.
        pytest.param(65536, 50, True, 8, 1, id="batched_64k_b8", marks=pytest.mark.timeout(900)),
    ],
)
def test_demo_text(
    mesh_device,
    seqlen,
    max_generated_tokens,
    use_trace,
    batch,
    repeat_batches,
):
    """E2e text generation: prefill + decode."""
    from transformers import AutoTokenizer

    device = mesh_device
    if batch > 1 and not _MULTI:
        pytest.skip("batched decode is the TP (multi-device) path; run with MESH_DEVICE=P150x4")
    device.enable_program_cache()
    # Block budget → max_seq_len, KV cache, and RoPE table
    num_blocks = _blocks_for(seqlen, max_generated_tokens)
    max_seq_len = num_blocks * BLOCK_SIZE

    t0 = time.time()
    model = Qwen36Model.from_pretrained(
        device,
        max_batch_size=batch,
        max_seq_len=max_seq_len,
        # n_layers=4,  # fast iteration
        # layer_indices=[0, 3],  # profile specific layers
    )
    logger.info(f"Model load: {time.time() - t0:.1f}s")
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)

    # Reserve generation budget; trim context from the middle via max_prompt_len
    max_prompt_len = ((max_seq_len - max_generated_tokens) // 128) * 128
    token_ids = _get_prompt(seqlen, tokenizer, max_prompt_len=max_prompt_len)
    actual_len = token_ids.shape[1]
    # Assert War-and-Peace configs (index 4) hit target seqlen; Frankenstein configs (0-3) intentionally cap lower.
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
    if model.num_devices > 1 and batch > 1:
        # Batched serving: B users share one paged KV + batched GDN state. The demo replicates the
        # one loaded prompt to all B users, so every row must generate identical tokens (asserted
        # below as a batched-correctness check).
        rows, perf = _run_tp_generation_batched(model, tokenizer, token_ids, max_generated_tokens, batch)
        text0 = tokenizer.decode(rows[0], skip_special_tokens=True)
        logger.info(
            f"[TP {model.num_devices}-dev B={batch}] ttft={perf['ttft_s']:.2f}s "
            f"per-user-decode={perf['decode_tok_s']:.2f} tok/s aggregate={perf['agg_tok_s']:.1f} tok/s"
        )
        logger.info(f"[TP B={batch}] GENERATED (row 0): {text0!r}")
        for u in range(batch):
            assert len(rows[u]) == max_generated_tokens, f"row {u}: {len(rows[u])} != {max_generated_tokens}"
            assert rows[u] == rows[0], f"row {u} diverged from row 0 (identical prompts must decode identically)"
        assert len(set(rows[0])) > 1, f"degenerate generation: {rows[0]}"
        return

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
        # Perf JSON for CI target check (validate_perf_targets.py)
        _save_tp_benchmark(perf, model, seqlen=seqlen, prompt_len=actual_len, num_generated=len(generated))
        return

    # Skip legacy prefill warmup for short traced prompts (masked-bucket path compiles in capture)
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
    """True when chunk-outer prefill trace is used (GDN chunk-seq always on)."""
    return any(
        (not layer.is_full_attention)
        and getattr(getattr(layer.attention, "weights", None), "use_chunk_seq_prefill", False)
        for layer in model.layers
    )


def _run_tp_generation(model, tokenizer, token_ids, max_generated_tokens, num_blocks):
    """TP generation: traced chunk-outer prefill + paged decode. Returns (tokens, perf_dict)."""
    vocab = model.args.vocab_size
    T = token_ids.shape[1]

    # Benchmark profiler (no-op outside CI)
    profiler = BenchmarkProfiler()
    profiler.start("run")

    # Round block budget to multiple of 32 for chunked SDPA page-table alignment
    num_blocks = ((num_blocks + 31) // 32) * 32

    # Paged KV cache + in-place GDN state (n_local_kv_heads per device at TP>1)
    kv_cache_shape = [num_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)
    # Identity page table for prompt + generation
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)

    # Capture chunk prefill trace (warmup; also warms masked-bucket programs)
    CHUNK = 2048
    t_cap = time.time()
    signpost("compile_prefill")
    profiler.start("compile_prefill")
    model.capture_prefill_trace_chunked(model.mesh_device, page_table, chunk_size=CHUNK)
    profiler.end("compile_prefill")
    logger.info(f"[TP] prefill chunk-trace captured in {time.time() - t_cap:.1f}s")

    _temp = float(os.environ.get("QWEN35_TEMP", "0") or 0)
    _rep_pen = float(os.environ.get("QWEN35_REP_PENALTY", "1.0") or 1.0)
    _no_repeat = int(os.environ.get("QWEN35_NO_REPEAT_NGRAM", "0") or 0)
    _top_k = int(os.environ.get("QWEN35_TOP_K", "0") or 0)
    _top_p = float(os.environ.get("QWEN35_TOP_P", "1.0") or 1.0)
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
            probs = torch.softmax(v / _temp, dim=-1)
            # top_k then top_p (vLLM order); multinomial takes unnormalized weights.
            if _top_k > 0:
                probs[probs < torch.topk(probs, min(_top_k, probs.numel())).values[-1]] = 0
            if _top_p < 1.0:
                srt, idx = torch.sort(probs, descending=True)
                remove = torch.cumsum(srt, dim=-1) > _top_p
                remove[1:] = remove[:-1].clone()  # keep first token crossing p
                remove[0] = False
                probs = torch.zeros_like(probs).scatter(0, idx, srt.masked_fill(remove, 0))
            return int(torch.multinomial(probs, 1).item())
        return int(torch.argmax(v).item())

    # Chunk-outer prefill; TTFT includes first-token sampling
    t0 = time.time()
    signpost("inference_prefill")
    profiler.start("inference_prefill")
    logits_dev = model.prefill_traced_chunked(token_ids[:, :T], page_table, actual_len=T)
    ttnn.synchronize_device(model.device)
    # Gather one mesh replica of replicated logits
    lt = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
    nxt = _pick(lt.reshape(-1, vocab)[0])
    generated.append(nxt)
    profiler.end("inference_prefill")
    ttft = time.time() - t0

    # Traced decode with GDN snapshot/restore (QWEN35_TP_DECODE_EAGER=1 for eager)
    from models.tt_transformers.tt.common import copy_host_to_device

    mesh = model.mesh_device
    eager = os.environ.get("QWEN35_TP_DECODE_EAGER") == "1"

    def _read(out):
        return _pick(model.process_output_decode(out, B=1, S=1).reshape(-1)[:vocab])

    # On-device sampling when sampler exists, temp>0, no rep-penalty/no-repeat (not wired on device); else greedy/host.
    _ondev_sample = model.sampling is not None and _temp > 0 and _rep_pen == 1.0 and _no_repeat == 0
    if _ondev_sample:
        from models.common.sampling.generator import SamplingParams, format_sampling_params

        _sbatch = model.sampling.tt_sampling.max_batch_size
        # No explicit seed: a seed would bake a fixed value into the folded trace → same
        # token every step. Unseeded, the device RNG self-advances each replay (varied tokens).
        _sp = format_sampling_params(
            SamplingParams(temperature=_temp, top_k=_top_k, top_p=_top_p),
            _sbatch,
        )
        model.sampling.apply_prefill_state(sampling_params=_sp, prompt_tokens=None, empty_slots=[0])

    # On-device per-shard argmax+max for pure greedy (skips full-vocab gather/readback)
    _greedy = (not eager) and (not _ondev_sample) and _temp == 0 and _rep_pen == 1.0 and _no_repeat == 0
    model._ondev_argmax = _greedy
    _per_shard = vocab // model.num_devices

    # Multi-core max over vocab shard: reduce dim=-1 parallelizes over tile-ROWS (R/32 cores), so
    # a tall/narrow grid (C=32 → R≈1952 → ~61 cores) beats the old 256×256 (8 tile-rows → 8 cores).
    _MAXVAL_C = 32
    _MAXVAL_R = (((_per_shard + _MAXVAL_C - 1) // _MAXVAL_C) + 31) // 32 * 32

    def _maxval_dev(sharded_logits):
        padded = ttnn.pad(
            sharded_logits, [(0, 0), (0, 0), (0, 0), (0, _MAXVAL_R * _MAXVAL_C - _per_shard)], value=-1e30
        )
        grid = ttnn.reshape(padded, (1, 1, _MAXVAL_R, _MAXVAL_C))
        part = ttnn.max(grid, dim=-1)
        part_row = ttnn.reshape(part, (1, 1, 1, _MAXVAL_R))
        val = ttnn.max(part_row, dim=-1)
        ttnn.deallocate(padded)
        ttnn.deallocate(grid)
        ttnn.deallocate(part)
        ttnn.deallocate(part_row)
        return val

    def _argmax_dev(sharded_logits):
        # Per-device local argmax + max value (untilize for multi-core argmax path)
        logits_rm = ttnn.to_layout(sharded_logits, ttnn.ROW_MAJOR_LAYOUT)
        idx = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
        ttnn.deallocate(logits_rm)
        return idx, _maxval_dev(sharded_logits)

    _read_comp = ttnn.ConcatMeshToTensor(mesh, dim=0)

    def _read_tok(idx_t, val_t):
        # Winner device = argmax of per-shard max vals; global token = d * per_shard + local_idx[d].
        # One mesh-composed D2H per tensor (2 total) vs one-per-shard (2*num_devices) — same
        # device order for idx and val, so d maps to device d = vocab shard d.
        idxs = ttnn.to_torch(idx_t, mesh_composer=_read_comp).reshape(-1)
        vals = ttnn.to_torch(val_t, mesh_composer=_read_comp).reshape(-1)
        d = int(torch.argmax(vals).item())
        return d * _per_shard + int(idxs[d].item())

    def _update(token, position):
        # page_table is a constant identity table for the whole sequence; it was uploaded once into
        # dev[3] by prepare_inputs_decode (init) and the trace bakes in its address, so it never needs
        # to change per token. Rebuilding it from torch every token was O(num_blocks) redundant host
        # work that grows with context (~1056 blocks at 64k). Update only the per-token inputs
        # (tokens, cur_pos, rope) and leave dev[3] as-is.
        host = model.prepare_decode_inputs_host(
            torch.tensor([[token]], dtype=torch.int32),
            torch.tensor([position], dtype=torch.int32),
            page_table=None,
        )
        copy_host_to_device(host[:3], device_tensors=dev[:3])

    # Snapshot/restore GDN state across all TP ranks (ttnn.copy preserves trace buffer addresses)
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
            # Match target buffer dtype on restore
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)

        for dn, (rec, convs) in zip(_gdn, snap):
            r = _back(rec, dn.rec_state.dtype)
            ttnn.copy(r, dn.rec_state)
            ttnn.deallocate(r)
            for j, c in enumerate(convs):
                cc = _back(c, dn.conv_states[j].dtype)
                ttnn.copy(cc, dn.conv_states[j])
                ttnn.deallocate(cc)

    # Persistent decode input buffers
    dev = model.prepare_inputs_decode(
        torch.tensor([[nxt]], dtype=torch.int32),
        torch.tensor([T], dtype=torch.int32),
        page_table=page_table,
    )

    def _decode_fwd():
        # on_device_logits=True → bare (sharded, padded) tensor for the sampler; else (logits, None).
        out = model.ttnn_decode_forward(
            dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=_ondev_sample
        )
        return out if _ondev_sample else out[0]

    trace_id = None
    tt_logits = None
    tt_tok = None
    tt_idx = tt_val = None
    signpost("compile_decode")
    profiler.start("compile_decode")
    if not eager:
        gdn_snap = _snapshot_gdn()
        # Eager compile + throwaway capture; restore GDN state and warm argmax kernels before trace
        _warm_logits = _decode_fwd()
        if _greedy:
            _wi, _wv = _argmax_dev(_warm_logits)
            ttnn.deallocate(_wi)
            ttnn.deallocate(_wv)
        if _ondev_sample:
            # Warm sampling kernels + advance the seed to SKIP before capture, so the trace self-advances the RNG each replay.
            model.sampling.seed_manager.get_new_values([0])
            model.sampling.sample(_warm_logits, enable_trace=False)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        tt_logits = _decode_fwd()
        # Fold per-shard argmax+max into trace for tiny readback when greedy
        if _greedy:
            tt_idx, tt_val = _argmax_dev(tt_logits)
        if _ondev_sample:
            # Fold sampling INTO the decode trace: a separate trace aliases the model trace's
            # buffers → intermittent garbage (the op itself is fine — see test_ondevice_sampling.py).
            model.sampling.seed_manager.get_new_values([0])  # steady no-op; keeps the machine in sync
            tt_tok, _ = model.sampling.sample(tt_logits, enable_trace=False)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        _restore_gdn(gdn_snap)
    profiler.end("compile_decode")

    pos = T
    decode_times = []
    signpost("inference_decode")
    profiler.start("inference_decode")
    _DEBUG_TIMING = os.environ.get("QWEN36_DEBUG_DECODE_TIMING") == "1"
    _phase_times = {"update": [], "exec_sync": [], "readback": []}
    while len(generated) < max_generated_tokens:
        # Time the FULL decode step (input update + device decode + host readback + sampling),
        # matching _run_tp_generation_batched and the tt_transformers reference convention —
        # not just the device compute — so the reported tok/s is real end-to-end throughput.
        t_step = time.time()
        t0 = time.time()
        _update(nxt, pos)
        t1 = time.time()
        if eager:
            tt_logits = _decode_fwd()
            if _ondev_sample:
                model.sampling.seed_manager.get_new_values([0])
                tt_tok, _ = model.sampling.sample(tt_logits, enable_trace=False)
        else:
            # Folded decode+sampling trace: tt_tok is written by this same execute_trace.
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        t2 = time.time()
        if _ondev_sample:
            nxt = int(model.process_output_decode(tt_tok, B=1, is_tokens=True)[0])
            if not (0 <= nxt < vocab):
                # Should never fire (fold fixes the aliasing). Cheap guard: log + argmax-fallback
                # instead of crashing _update on a garbage id.
                logger.warning(f"[ondev-sample] step={len(generated)} out-of-vocab id {nxt}; falling back to argmax")
                _g = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3)).float()
                nxt = int(_g.reshape(model.sampling.tt_sampling.max_batch_size, -1)[0, :vocab].argmax())
        elif _greedy and not eager:
            nxt = _read_tok(tt_idx, tt_val)
        else:
            nxt = _read(tt_logits)
        t3 = time.time()
        if _DEBUG_TIMING:
            _phase_times["update"].append(t1 - t0)
            _phase_times["exec_sync"].append(t2 - t1)
            _phase_times["readback"].append(t3 - t2)
        decode_times.append(time.time() - t_step)
        generated.append(nxt)
        pos += 1
    if _DEBUG_TIMING:
        for k, vs in _phase_times.items():
            vs_steady = vs[1:] if len(vs) > 1 else vs
            m = (sum(vs_steady) / len(vs_steady)) if vs_steady else 0.0
            logger.info(f"[DEBUG_DECODE_TIMING] {k}: avg={m*1000:.2f}ms")
    if trace_id is not None:
        ttnn.release_trace(mesh, trace_id)
    profiler.end("inference_decode")

    # Drop first decode step for steady-state throughput
    steady = decode_times[1:] if len(decode_times) > 1 else decode_times
    avg = (sum(steady) / len(steady)) if steady else float("inf")
    profiler.end("run")
    return generated, {"ttft_s": ttft, "decode_tok_s": (1.0 / avg) if avg > 0 else 0.0, "profiler": profiler}


def _run_tp_generation_batched(model, tokenizer, token_ids, max_generated_tokens, batch):
    """Multi-device (TP) batched generation: B users decoded together.

    Each user is prefilled into its own blocks of one shared paged KV cache (+ its row of the
    batched GDN state), then one traced decode step advances all B users per iteration at their
    own positions. The demo replicates the one loaded prompt to all B users; the caller asserts
    every row decodes identically. Decode is captured once as a B-wide trace and replayed; as in
    _run_tp_generation, the post-prefill GDN state is snapshotted before the throwaway capture run
    and restored after so the baked buffer addresses stay valid. Returns (generated_rows, perf)
    with generated_rows a list of B token lists.
    """
    from models.tt_transformers.tt.common import copy_host_to_device

    B = batch
    vocab = model.args.vocab_size
    mesh = model.mesh_device
    T = token_ids.shape[1]

    # Per-user block budget covering the prompt + decode (one contiguous range/user). Round up to a
    # multiple of 8: chunked SDPA reads each user's page-table row as a ROW_MAJOR int32 stick and
    # requires stick_size (= bpu * 4 bytes) % 32 == 0, i.e. bpu % 8 == 0 (as _blocks_for enforces for
    # the single-user path). A misaligned bpu makes the long-prefill SDPA read the wrong KV.
    bpu = max(8, -(-(T + max_generated_tokens) // BLOCK_SIZE))
    bpu = ((bpu + 7) // 8) * 8
    total_blocks = B * bpu
    kv_cache_shape = [total_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=B)
    page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])  # [B, bpu]

    # Prefill routes: T<=256 grouped single-pass; T>256 prefill_chunked_peruser (per-user).
    # QWEN_BATCHED_GROUPED=1 (default): group short prompts for ~4x (T<=128, B=4 groups) / ~1.6x
    # (T<=256, B=2 groups) TTFT. prefill_paged_grouped auto-caps group size by the GDN kernel's
    # per-bucket L1 ceiling. Long prompts (T>256) can't batch GDN (chunk clash) -> stay per-user.
    bucket = 128
    eager = os.environ.get("QWEN35_TP_PREFILL_EAGER") == "1"
    grouped_short = os.environ.get("QWEN_BATCHED_GROUPED", "1") != "0" and T <= 256
    use_traced_bucket = (T == bucket) and not eager and not grouped_short
    token_list = [token_ids[:, :T] for _ in range(B)]
    if use_traced_bucket:
        # Capture is a one-time startup cost, so it runs outside the TTFT timer (mirrors the
        # single-user traced path's capture_prefill_trace_chunked before t0). Capture against one
        # row (buffer width is fixed across replays); each replay DMAs the user's page-table row in.
        model.capture_prefill_trace_bucket(mesh, page_table[0:1].contiguous(), bucket=bucket)
    t0 = time.time()
    if grouped_short:
        pf_logits = model.prefill_paged_grouped(token_list, page_table, valid_lens=[T] * B, group_size=4)
    elif use_traced_bucket:
        pf_logits = model.prefill_traced_bucket_batched(token_list, page_table, valid_lens=[T] * B)
    elif T > bucket:
        # Long prompts: per-user chunk-outer prefill (correct for any length; eager in this stage).
        pf_logits = model.prefill_chunked_peruser(token_list, page_table, valid_lens=[T] * B)
    else:
        pf_logits = model.prefill_paged_peruser(token_list, page_table, valid_lens=[T] * B)
    ttnn.synchronize_device(mesh)
    ttft = time.time() - t0
    if use_traced_bucket:
        model.release_prefill_trace_bucket()

    comp0 = ttnn.ConcatMeshToTensor(mesh, dim=0)

    def _pick(vec):
        return int(torch.argmax(vec.float()).item())

    nxt = [_pick(ttnn.to_torch(pf_logits[u], mesh_composer=comp0).reshape(-1, vocab)[0]) for u in range(B)]
    generated = [[nxt[u]] for u in range(B)]

    # ---- traced batched decode (snapshot/restore GDN around the throwaway capture run) ----
    eager = os.environ.get("QWEN35_TP_DECODE_EAGER") == "1"
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
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)

        for dn, (rec, convs) in zip(_gdn, snap):
            r = _back(rec, dn.rec_state.dtype)
            ttnn.copy(r, dn.rec_state)
            ttnn.deallocate(r)
            for j, c in enumerate(convs):
                cc = _back(c, dn.conv_states[j].dtype)
                ttnn.copy(cc, dn.conv_states[j])
                ttnn.deallocate(cc)

    def _update(tokens_row, positions):
        # page_table is a constant per-user block mapping for the whole decode loop; it was
        # uploaded once into dev[3] by prepare_inputs_decode (init) and the trace bakes in its
        # address, so it never needs to change per step. Rebuilding + re-uploading it from torch
        # every step was O(B * num_blocks_per_user) redundant host work that grows with both
        # batch and context length. Update only the per-step inputs (tokens, cur_pos, rope).
        host = model.prepare_decode_inputs_host(
            torch.tensor(tokens_row, dtype=torch.int32).reshape(B, 1),
            torch.tensor(positions, dtype=torch.int32),
            page_table=None,
        )
        copy_host_to_device(host[:3], device_tensors=dev[:3])

    pos = [T] * B
    dev = model.prepare_inputs_decode(
        torch.tensor(nxt, dtype=torch.int32).reshape(B, 1),
        torch.tensor(pos, dtype=torch.int32),
        page_table=page_table,
    )

    # Batched decode readback mode (QWEN36_BATCHED_DECODE_MODE):
    #   "shard" (default) - per-shard on-device argmax+max (generalizes _run_tp_generation's
    #            B=1 fast path to B>1): each device reduces its OWN vocab shard to (idx, max)
    #            on device, then only 2 tiny [num_devices, B] tensors are read to host — no
    #            full-vocab all-gather, no [B,1,vocab] logits transfer.
    #   "sample" - TTSampling force-argmax path (all-gathers full logits across devices before
    #            arg-maxing). Avoids the full logits HOST transfer but adds a real device-side
    #            all-gather; measured slower overall than "shard" — kept for comparison.
    #   "host"  - legacy: full [B,1,vocab] logits to host, then torch.argmax. Baseline.
    _mode = os.environ.get("QWEN36_BATCHED_DECODE_MODE", "shard")
    if _mode == "sample" and model.sampling is None:
        _mode = "host"
    if _mode == "shard":
        _per_shard = vocab // model.num_devices
        _MAXVAL_C = 32
        _MAXVAL_R = (((_per_shard + _MAXVAL_C - 1) // _MAXVAL_C) + 31) // 32 * 32
        _read_comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
        _nd = model.num_devices

        def _maxval_dev_b(sharded_logits, Bn):
            # sharded_logits: [1,1,Bn,per_shard] -> per-row (per-user) max, tile-parallel reduce.
            padded = ttnn.pad(
                sharded_logits, [(0, 0), (0, 0), (0, 0), (0, _MAXVAL_R * _MAXVAL_C - _per_shard)], value=-1e30
            )
            grid = ttnn.reshape(padded, (1, Bn, _MAXVAL_R, _MAXVAL_C))
            part = ttnn.max(grid, dim=-1)
            part_row = ttnn.reshape(part, (1, 1, Bn, _MAXVAL_R))
            val = ttnn.max(part_row, dim=-1)
            ttnn.deallocate(padded)
            ttnn.deallocate(grid)
            ttnn.deallocate(part)
            ttnn.deallocate(part_row)
            return val

        def _argmax_dev_b(sharded_logits, Bn):
            # Per-device, per-user local argmax (ttnn.argmax reduces the last dim per row,
            # so this generalizes to Bn>1 unchanged from the B=1 version).
            logits_rm = ttnn.to_layout(sharded_logits, ttnn.ROW_MAJOR_LAYOUT)
            idx = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
            ttnn.deallocate(logits_rm)
            return idx, _maxval_dev_b(sharded_logits, Bn)

        def _read_tok_b(idx_t, val_t, Bn):
            # [num_devices, Bn] after mesh-concat (Bn may be sampler-padded); winning device per
            # USER (not globally): global token = winning_device*per_shard + its local argmax idx.
            # Only the first B columns are real users -- the rest are the sampler-width pad.
            idxs = ttnn.to_torch(idx_t, mesh_composer=_read_comp).reshape(_nd, Bn)[:, :B].to(torch.int64)
            vals = ttnn.to_torch(val_t, mesh_composer=_read_comp).reshape(_nd, Bn)[:, :B]
            d = torch.argmax(vals, dim=0)  # [B] winning device index per user
            tok = d * _per_shard + idxs[d, torch.arange(B)]
            return tok.tolist()

    if _mode == "sample":
        from models.common.sampling.generator import SamplingParams

        _sbatch = model.sampling.tt_sampling.max_batch_size
        _greedy_params = SamplingParams(
            temperature=[1.0] * _sbatch, top_k=[1] * _sbatch, top_p=[1.0] * _sbatch, seed=[0] * _sbatch
        )
        model.sampling.apply_decode_state([_greedy_params], reset_batch=True)

    _sharded_logits_mode = _mode in ("shard", "sample")
    trace_id, tt_logits, tt_idx, tt_val, tt_tok = None, None, None, None, None
    if not eager:
        snap = _snapshot_gdn()
        _warm_logits = model.ttnn_decode_forward(
            dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=_sharded_logits_mode
        )
        if _mode == "shard":
            # ttnn_decode_forward(on_device_logits=True) pads the batch dim up to the sampler's
            # width (e.g. 32), not our real B -- use the actual shape, slice to B on readback.
            _wi, _wv = _argmax_dev_b(_warm_logits, _warm_logits.shape[2])
            ttnn.deallocate(_wi)
            ttnn.deallocate(_wv)
        elif _mode == "sample":
            model.sampling.sample(_warm_logits, enable_trace=False)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        tt_logits = model.ttnn_decode_forward(
            dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=_sharded_logits_mode
        )
        if _mode == "shard":
            # Fold per-shard argmax+max into the trace: tiny [num_devices,padded_B] readback/step.
            tt_idx, tt_val = _argmax_dev_b(tt_logits, tt_logits.shape[2])
        elif _mode == "sample":
            # Fold sampling INTO the same trace as the forward pass (mirrors _run_tp_generation).
            tt_tok, _ = model.sampling.sample(tt_logits, enable_trace=False)
        else:
            tt_logits = tt_logits[0]
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        _restore_gdn(snap)

    # Time the FULL decode step (input update + device decode + host logit read + token select)
    # so the reported tok/s is real end-to-end throughput, not just the device compute.
    decode_times = []
    _DEBUG_TIMING = os.environ.get("QWEN36_DEBUG_DECODE_TIMING") == "1"
    _phase_times = {"update": [], "exec_sync": [], "readback": [], "argmax": []}
    while len(generated[0]) < max_generated_tokens:
        t_step = time.time()
        t0 = time.time()
        _update([generated[u][-1] for u in range(B)], pos)
        t1 = time.time()
        if eager:
            out = model.ttnn_decode_forward(
                dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3], on_device_logits=_sharded_logits_mode
            )
            if _mode == "shard":
                tt_idx, tt_val = _argmax_dev_b(out, out.shape[2])
            elif _mode == "sample":
                tt_tok, _ = model.sampling.sample(out, enable_trace=False)
            else:
                tt_logits = out[0]
        else:
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        t2 = time.time()
        if _mode == "shard":
            # Two tiny [num_devices, padded_B] tensors, not the full [B,1,vocab] logits.
            next_toks = _read_tok_b(tt_idx, tt_val, tt_idx.shape[-1])
        elif _mode == "sample":
            # Sampled ids only: a tiny [B] tensor, not the full [B,1,vocab] logits.
            next_toks = model.process_output_decode(tt_tok, B, is_tokens=True).reshape(-1).tolist()
        else:
            logits_step = model.process_output_decode(tt_logits, B)  # [B, 1, vocab]
            # Vectorized argmax over the whole batch in ONE call (matches tt_transformers'
            # batched decode pattern) instead of B separate full-vocab argmax + .item() calls.
            next_toks = torch.argmax(logits_step[:, 0, :vocab].float(), dim=-1).tolist()
        t3 = time.time()
        for u in range(B):
            generated[u].append(next_toks[u])
        pos = [p + 1 for p in pos]
        t4 = time.time()
        if _DEBUG_TIMING:
            _phase_times["update"].append(t1 - t0)
            _phase_times["exec_sync"].append(t2 - t1)
            _phase_times["readback"].append(t3 - t2)
            _phase_times["argmax"].append(t4 - t3)
        decode_times.append(time.time() - t_step)
    if trace_id is not None:
        ttnn.release_trace(mesh, trace_id)

    steady = decode_times[1:] if len(decode_times) > 1 else decode_times
    avg = (sum(steady) / len(steady)) if steady else float("inf")
    if _DEBUG_TIMING:
        for k, vs in _phase_times.items():
            vs_steady = vs[1:] if len(vs) > 1 else vs
            m = (sum(vs_steady) / len(vs_steady)) if vs_steady else 0.0
            logger.info(f"[DEBUG_DECODE_TIMING] {k}: avg={m*1000:.2f}ms")
    return generated, {
        "ttft_s": ttft,
        "decode_tok_s": (1.0 / avg) if avg > 0 else 0.0,  # per decode step (advances all B users)
        "agg_tok_s": (B / avg) if avg > 0 else 0.0,  # aggregate tokens/s across the B users
    }


def _run_traced_generation(model, tokenizer, device, token_ids, max_generated_tokens, num_blocks):
    """Traced prefill + paged decode. Returns (generated_tokens, perf_dict)."""
    T = token_ids.shape[1]

    # Paged KV cache + DeltaNet state
    num_kv_heads = model.args.n_kv_heads
    head_dim = model.args.head_dim
    kv_cache_shape = [num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)

    # Identity page table
    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

    # Chunk-outer prefill trace (one 2048-token chunk replayed per chunk)
    assert _should_use_chunked_trace(model), "chunk-seq GDN prefill must be enabled"
    chunk_size = 2048
    bucket_size = ((T + chunk_size - 1) // chunk_size) * chunk_size
    logger.info(f"Capturing prefill trace at bucket_size={bucket_size} (prompt {T} tokens, chunk-outer replay)...")
    signpost("compile_prefill")
    t_cap = time.time()
    # Warm masked buckets only for short prompts (T < chunk_size)
    model.capture_prefill_trace_chunked(
        device, page_table, chunk_size=chunk_size, warmup_masked_buckets=(T < chunk_size)
    )
    logger.info(f"Prefill trace captured in {time.time() - t_cap:.1f}s")
    pad_len = bucket_size - T
    # Pad bucket with last real token (token 0 corrupts DeltaNet state)
    last_token = token_ids[:, -1:].expand(1, pad_len) if pad_len > 0 else token_ids[:, :0]
    padded_token_ids = torch.cat([token_ids, last_token], dim=1)

    signpost("inference_prefill")
    t0 = time.time()
    if T < chunk_size:
        # Short prompt: masked fixed-bucket prefill
        logits = model.prefill_masked_bucket(token_ids, page_table, actual_len=T)
    else:
        logits = model.prefill_traced_chunked(padded_token_ids, page_table, actual_len=T)
    logits_torch = ttnn.to_torch(logits).squeeze()
    next_token = logits_torch.argmax().item()
    ttft = time.time() - t0

    assert not torch.isnan(logits_torch).any(), "NaN in prefill logits"
    gen = Generator([model], [model.args], device)

    # Decode trace with GDN snapshot/restore (stock capture would double-advance state)
    from models.demos.blackhole.qwen36.tt.generator_interface import prime_decode_trace

    signpost("compile_decode")
    prime_decode_trace(gen, model, torch.tensor([[next_token]], dtype=torch.long), torch.tensor([T]), page_table)

    generated = [next_token]
    decode_times = []
    current_pos = T

    signpost("inference_decode")
    for i in range(max_generated_tokens - 1):
        # Timing includes forward + sampling
        t_step = time.time()
        out = gen.decode_forward(
            torch.tensor([[next_token]], dtype=torch.long),
            torch.tensor([current_pos]),
            page_table=page_table,
            kv_cache=None,
            enable_trace=True,
            read_from_device=True,
        )
        dl = (out[0] if isinstance(out, tuple) else out).squeeze().float()
        next_token = int(dl.argmax())
        decode_times.append(time.time() - t_step)

        assert not torch.isnan(dl).any(), f"NaN in traced decode at step {i}"
        generated.append(next_token)
        current_pos += 1

        if next_token == tokenizer.eos_token_id:
            break

    avg_decode = sum(decode_times) / len(decode_times) if decode_times else float("inf")
    return generated, {"ttft": ttft, "avg_decode_s": avg_decode, "decode_steps": len(decode_times)}


def _run_paged_generation(model, tokenizer, device, token_ids, max_generated_tokens, num_blocks):
    """Non-traced prefill + paged decode. Returns (generated_tokens, perf_dict)."""
    T = token_ids.shape[1]

    # Paged KV cache + DeltaNet state
    num_kv_heads = model.args.n_kv_heads
    head_dim = model.args.head_dim
    kv_cache_shape = [num_blocks, num_kv_heads, BLOCK_SIZE, head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)

    # Identity page table
    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

    signpost("inference_prefill")
    t0 = time.time()
    logits = model.prefill_paged(token_ids, page_table)
    ttnn.synchronize_device(device)
    logits_torch = ttnn.to_torch(logits).squeeze()
    next_token = logits_torch.argmax().item()
    ttft = time.time() - t0

    assert not torch.isnan(logits_torch).any(), "NaN in paged prefill logits"

    gen = Generator([model], [model.args], device)

    generated = [next_token]
    decode_times = []

    signpost("inference_decode")
    for i in range(max_generated_tokens - 1):
        # Timing includes forward + sampling
        t_step = time.time()
        out = gen.decode_forward(
            torch.tensor([[next_token]], dtype=torch.long),
            torch.tensor([T + i]),
            page_table=page_table,
            kv_cache=None,
            enable_trace=False,
            read_from_device=True,
        )
        dl = (out[0] if isinstance(out, tuple) else out).squeeze().float()
        next_token = int(dl.argmax())
        decode_times.append(time.time() - t_step)

        assert not torch.isnan(dl).any(), f"NaN in paged decode logits at step {i}"
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)

    avg_decode = sum(decode_times) / len(decode_times) if decode_times else float("inf")
    return generated, {"ttft": ttft, "avg_decode_s": avg_decode, "decode_steps": len(decode_times)}


def _save_tp_benchmark(perf, model, seqlen, prompt_len, num_generated):
    """Emit CI benchmark JSON (no-op outside CI; uses nominal ``seqlen`` for target lookup)."""
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
    # Correctness only; perf targets checked by validate_perf_targets.py
    assert num_generated >= 1, "Should generate at least 1 token"
