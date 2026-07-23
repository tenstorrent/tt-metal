# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 text generation demo.

Simple prefill + decode loop following gpt-oss text_demo.py pattern.

Long-context policy (see ``GEMMA4_LONG_CONTEXT_POLICY`` in generator_trace.py)
matches ``text_demo_v2.py``: per-(model, device) bounded sliding / chunked
prefill cutovers on QB2 (P150x4 / P300x2). Overrides:
``GEMMA4_BOUNDED_SLIDING``, ``GEMMA4_GEN_PREFILL_CHUNK``, ``GEMMA4_DEMO_SINGLE_CHUNK``,
``GEMMA4_MAX_SEQ_LEN``, ``GEMMA4_MAX_NEW_TOKENS``.

Usage:
    pytest models/demos/gemma4/demo/text_demo.py -v --timeout=600

    # With fewer layers for testing:
    pytest models/demos/gemma4/demo/text_demo.py -v --timeout=600 -k "test_demo"

    # Long-context (64k/128k/256k) on QB2:
    MESH_DEVICE=P150x4 HF_MODEL=google/gemma-4-12B-it pytest \\
        models/demos/gemma4/demo/text_demo.py -k long-context-128k -s --timeout 1800

    # Batch-32 batched prefill:
    pytest models/demos/gemma4/demo/text_demo.py::test_demo_batch_prefill -k "prefill_128-1x1" -v

    # Batch-32 prefill + decode; override batch via GEMMA4_BATCH_DEMO_SIZE=8:
    pytest models/demos/gemma4/demo/text_demo.py::test_demo_batch_32 -k "prefill_128 and 1x1" -v
    GEMMA4_BATCH_DEMO_SIZE=8 pytest models/demos/gemma4/demo/text_demo.py::test_demo_batch_32 -k "prefill_2048 and 1x4" -v

    # 128k batched-prefill ceiling documentation:
    pytest models/demos/gemma4/demo/text_demo.py::test_demo_batch_prefill_4096_ceiling -v
"""

import gc
import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import PREFILL_BUCKETS, parametrize_mesh_with_fabric
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.gemma4.tt.generator import GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN, Gemma4Generator
from models.demos.gemma4.tt.generator_trace import (
    should_auto_enable_bounded_sliding,
    should_auto_enable_chunked_bounded,
)
from models.demos.utils.llm_demo_utils import create_benchmark_data
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    get_padded_prefill_len,
    preprocess_inputs_prefill,
    sample_host,
)
from models.tt_transformers.tt.generator import SUPPORTED_PREFILL_BATCH_SIZES
from models.tt_transformers.tt.model_config import determine_device_name

_TT_TRANSFORMERS_PROMPTS_DIR = "models/tt_transformers/demo/sample_prompts"
_CONTEXT_CACHE_DIR = "models/tt_transformers/demo/context_cache"


def _snap_to_bucket(prompt_len, max_seq_len):
    """Round prompt length up to the next PREFILL_BUCKETS value within max_seq_len.

    Each bucket corresponds to a separately-compiled prefill kernel; snapping
    keeps the number of compiled kernel variants finite. Buckets above
    max_seq_len are filtered, and prompts longer than the largest usable
    bucket fall back to max_seq_len itself.
    """
    usable = [b for b in PREFILL_BUCKETS if b <= max_seq_len]
    if not usable:
        raise ValueError(f"max_seq_len={max_seq_len} is below the smallest prefill bucket ({PREFILL_BUCKETS[0]})")
    for b in usable:
        if prompt_len <= b:
            return b
    return max_seq_len


def _shorten_for_log(prompt, head=200, tail=200):
    """Return a head/tail excerpt for logging; long contexts otherwise flood logs."""
    if len(prompt) <= head + tail + 50:
        return prompt
    return f"{prompt[:head]}\n<long prompt not printed in full ({len(prompt)} chars)>\n{prompt[-tail:]}"


def _load_and_cache_context(url, max_chars=None):
    """Fetch a long-context source from URL with on-disk caching.

    Mirrors tt_transformers' load_and_cache_context: hashes the URL into a
    cache file, downloads on miss, and clips to max_chars. Reuses the
    tt_transformers cache so files fetched there are visible here too.
    """
    import hashlib
    from pathlib import Path

    import requests

    cache_dir = Path(_CONTEXT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / hashlib.md5(url.encode()).hexdigest()

    if cache_file.exists():
        text = cache_file.read_text()
    else:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        text = resp.text
        cache_file.write_text(text)
        logger.info(f"Cached context from {url} ({len(text)} chars)")

    if max_chars:
        text = text[:max_chars]
    return text


def _bucket_to_prompt_file(target_bucket):
    """Map a prefill bucket length to a tt_transformers sample prompts file.

    Buckets ≤ 256 use the hand-written question prompts; ≥ 1024 use the
    long-context files (with URL fetch + caching). Sizes between (e.g. 512)
    fall back to the 256 file and rely on bucket-snap zero-padding.
    """
    if target_bucket <= 128:
        return f"{_TT_TRANSFORMERS_PROMPTS_DIR}/input_data_questions_prefill_128.json"
    if target_bucket <= 256:
        return f"{_TT_TRANSFORMERS_PROMPTS_DIR}/input_data_questions_prefill_256.json"
    if target_bucket < 1024:
        return f"{_TT_TRANSFORMERS_PROMPTS_DIR}/input_data_questions_prefill_256.json"
    # Long-context files step in 1k/2k/.../128k/256k. Cap at 256k (largest
    # source file); buckets above that reuse the 256k file and bucket-snap
    # pads to the larger length.
    long_size = min(target_bucket, 262144)
    size_k = long_size // 1024
    return f"{_TT_TRANSFORMERS_PROMPTS_DIR}/input_data_long_{size_k}k.json"


def _resolve_bounded_sliding(max_seq_len, mesh_device, model_path, *, paged_attention=True) -> bool:
    """Match text_demo_v2: policy auto-enable, overridable via GEMMA4_BOUNDED_SLIDING."""
    _bs_env = os.environ.get("GEMMA4_BOUNDED_SLIDING")
    if _bs_env is None:
        bounded_sliding = should_auto_enable_bounded_sliding(max_seq_len, mesh_device, model_path)
    else:
        bounded_sliding = _bs_env.lower() in ("1", "true", "yes")
    return bool(bounded_sliding and paged_attention)


def _right_size_page_max_num_blocks(batch_size, max_seq_len, page_params) -> int:
    """Match text_demo_v2 page-pool right-sizing for batch=1 long-context."""
    block_size = int(page_params["page_block_size"])
    needed_blocks = batch_size * math.ceil(max_seq_len / block_size)
    configured_blocks = page_params.get("page_max_num_blocks")
    if batch_size <= 1 or configured_blocks is None:
        return needed_blocks
    return int(configured_blocks)


def _install_hybrid_page_tables(model, model_args, batch_size, block_size, max_seq_len, num_layers=None):
    """Install per-layer page tables required for bounded sliding KV."""
    from models.demos.gemma4.tt.attention.kv_cache_hybrid import build_hybrid_page_tables

    n_layers = num_layers or model_args.num_hidden_layers
    sliding_mask = [model_args.layer_types[i] == "sliding_attention" for i in range(n_layers)]
    per_layer_pts = build_hybrid_page_tables(
        n_layers,
        sliding_mask,
        num_users=batch_size,
        block_size=block_size,
        max_seq_len=max_seq_len,
        sliding_window=model_args.sliding_window,
    )
    model._active_page_tables_per_layer = per_layer_pts
    logger.info(f"Bounded sliding: installed {len(per_layer_pts)} per-layer page tables")
    return per_layer_pts


def _mesh_shape_from_env():
    """MESH_DEVICE → mesh shape. QB2 is P150x4 or P300x2 (both 1x4)."""
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "P150": (1, 1),
        "P300": (1, 2),
        "P150x4": (1, 4),
        "P300x2": (1, 4),
        "P300X2": (1, 4),
        "P150x8": (1, 8),
        "T3K": (1, 8),
    }.get(os.environ.get("MESH_DEVICE"), (1, 4))


def _host_sample_greedy(logits):
    """Greedy argmax host sample; logits [B, vocab] or [B, 1, vocab] → [B, 1]."""
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    return logits.argmax(dim=-1, keepdim=True)


def load_demo_prompt(target_bucket, instruct=True):
    """Load a single prompt suitable for the target prefill bucket.

    Short prompts (≤256) come from the hand-written question files. Long
    prompts pull a gutenberg-style context via URL (cached), clipped to the
    file's max_length, then wrapped in a markdown block per the
    tt_transformers convention when instruct=True.
    """
    import json

    path = _bucket_to_prompt_file(target_bucket)
    with open(path) as f:
        entry = json.load(f)[0]

    prompt = entry["prompt"]
    if "context" in entry:
        max_chars = entry.get("max_length")
        context = _load_and_cache_context(entry["context"], max_chars=max_chars)
        prompt = "```" + context + "```\n\n" + prompt if instruct else context
    return prompt


# Batch batched-prefill helpers and tests (default batch 32; override with GEMMA4_BATCH_DEMO_SIZE).
_BATCH32_DRAM_OOM_SIZE = 32
_DEFAULT_BATCH_DEMO_SIZE = 32
# Prefill lengths for ``test_demo_batch_32`` (4096 hits the 128k batched-prefill ceiling).
_DEMO_BATCH_PREFILL_LENGTHS = [128, 512, 1024, 2048, 4096]
# Prefill-only batch coverage; batch-32 2048/4096 may xfail on DRAM-limited meshes (see below).
_BATCH_PREFILL_LENGTHS = [128, 1024, 2048]


def _batch_demo_size():
    """Demo concurrent-user count; defaults to 32, overridable for smaller-batch experiments."""
    size = int(os.getenv("GEMMA4_BATCH_DEMO_SIZE", str(_DEFAULT_BATCH_DEMO_SIZE)))
    if size not in SUPPORTED_PREFILL_BATCH_SIZES:
        supported = ", ".join(str(b) for b in SUPPORTED_PREFILL_BATCH_SIZES)
        raise ValueError(f"GEMMA4_BATCH_DEMO_SIZE={size} must be one of: {supported}")
    return size


def _mesh_shape_str(mesh_device):
    return "x".join(str(d) for d in mesh_device.shape)


def _is_31b_model(model_path):
    name = os.path.basename(str(model_path).rstrip("/")).lower()
    return "31b" in name


def _batch_prefill_known_dram_oom(mesh_device, model_path, batch_size, prefill_len):
    """True for batch-32 long prefill on 31B 1×4 (~64k batched token DRAM budget, e.g. 32×2048)."""
    if batch_size != _BATCH32_DRAM_OOM_SIZE:
        return False
    if prefill_len not in (2048, 4096):
        return False
    if _mesh_shape_str(mesh_device) != "1x4":
        return False
    return _is_31b_model(model_path)


def _maybe_xfail_batch_prefill_dram(mesh_device, model_path, batch_size, prefill_len):
    if _batch_prefill_known_dram_oom(mesh_device, model_path, batch_size, prefill_len):
        pytest.xfail(
            f"Batch-{batch_size} prefill_len={prefill_len} exceeds ~64k batched token DRAM budget "
            f"on 31B 1×4 (32×2048=65536); run on 1×8 or a smaller model."
        )


def _batch_prefill_hits_ceiling(batch_size, prompt_len):
    """True when ``batch_size × padded prefill length`` meets/exceeds the 128k cap."""
    kernel_len = get_padded_prefill_len(prompt_len)
    return batch_size * kernel_len >= GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN


def _batch_page_params(batch_size, prefill_len, max_new_tokens, page_block_size=64):
    """Size the paged-attention pool for ``batch_size`` concurrent users."""
    blocks_per_user = (prefill_len + max_new_tokens + page_block_size - 1) // page_block_size
    return {
        "page_block_size": page_block_size,
        "page_max_num_blocks": batch_size * blocks_per_user,
    }


def _create_tt_page_table(global_batch_size, paged_attention_config: PagedAttentionConfig):
    """Map virtual paged-attention blocks to physical blocks for a batch."""
    max_num_blocks = paged_attention_config.max_num_blocks
    permutation = torch.randperm(max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    return reverse_permutation.reshape(global_batch_size, max_num_blocks // global_batch_size).to(torch.int32)


def load_batch_demo_prompts(batch_size, prefill_len, instruct=True):
    """Replicate one bucket-sized prompt so every user has identical token length."""
    prompt = load_demo_prompt(prefill_len, instruct=instruct)
    return [prompt] * batch_size


def _encode_demo_prompt(tokenizer, prompt, instruct=True):
    """Tokenize a demo prompt the same way as ``run_generation``."""
    if instruct and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        chat_result = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return chat_result["input_ids"].squeeze(0).tolist()
    return tokenizer.encode(prompt, add_special_tokens=True)


def _prepare_batch_prefill_tokens(prompts, tokenizer, target_prefill_len, instruct=True):
    """Build ``[batch, target_prefill_len]`` tokens and per-user prompt lengths."""
    encoded_prompts = [_encode_demo_prompt(tokenizer, prompt, instruct=instruct) for prompt in prompts]
    rows = []
    decoding_pos = []
    for encoded in encoded_prompts:
        if len(encoded) > target_prefill_len:
            encoded = encoded[:target_prefill_len]
        prompt_len = len(encoded)
        row = torch.zeros(target_prefill_len, dtype=torch.int32)
        row[:prompt_len] = torch.tensor(encoded, dtype=torch.int32)
        rows.append(row)
        decoding_pos.append(prompt_len)
    return torch.stack(rows), decoding_pos, encoded_prompts


def run_batch_generation(
    mesh_device,
    model_path,
    batch_size=None,
    prefill_len=128,
    max_new_tokens=32,
    num_layers=None,
    max_seq_len=4096,
    page_params=None,
    enable_prefill_trace=True,
    enable_decode_trace=False,
):
    """Run batched text generation through ``Gemma4Generator``.

    Exercises batched prefill (including chunking at the 128k token ceiling)
    via ``generator.prefill_forward_text``. Decode is best-effort for batch>1:
    batch>1: Gemma4's decode path currently prepares inputs for a single active
    user, so this demo primarily validates batched prefill coverage from #44952.

    Args:
        mesh_device: TT mesh device
        model_path: HuggingFace model path or ID
        batch_size: Number of concurrent users (default from ``GEMMA4_BATCH_DEMO_SIZE``, else 32)
        prefill_len: Target prefill bucket length
        max_new_tokens: Decode tokens per user after prefill
        max_seq_len: KV cache / context budget
        page_params: Paged-attention block config
        enable_prefill_trace: Enable prefill tracing where supported
        enable_decode_trace: Enable decode tracing (single-user oriented today)

    Returns:
        dict with ``prefilled_tokens`` [B, 1], ``generated_texts`` [B],
        ``prefill_seq_len`` (padded kernel length), and ``prompt_lens``.
    """
    del num_layers  # Full-model demo; layer override not wired through Generator factory yet.

    if batch_size is None:
        batch_size = _batch_demo_size()

    is_ci_env = os.environ.get("CI") == "true"
    profiler = BenchmarkProfiler()
    profiler.start("run")

    prompts = load_batch_demo_prompts(batch_size, prefill_len, instruct=True)
    logger.info(f"Loaded {batch_size} prompts for prefill bucket {prefill_len}")

    if page_params is None:
        page_params = _batch_page_params(batch_size, prefill_len, max_new_tokens)
    page_max_num_blocks = _right_size_page_max_num_blocks(batch_size, max_seq_len, page_params)
    paged_attention_config = PagedAttentionConfig(
        block_size=int(page_params["page_block_size"]),
        max_num_blocks=page_max_num_blocks,
    )
    page_table = _create_tt_page_table(batch_size, paged_attention_config)

    bounded_sliding = _resolve_bounded_sliding(max_seq_len, mesh_device, model_path)
    logger.info(
        f"Creating Gemma4Generator (batch={batch_size}, max_seq_len={max_seq_len}, "
        f"prefill_len={prefill_len}, bounded_sliding={bounded_sliding})..."
    )
    t0 = time.time()
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=bounded_sliding,
    )
    model_args = generator.model_args[0]
    if bounded_sliding:
        _install_hybrid_page_tables(
            generator.model[0],
            model_args,
            batch_size=batch_size,
            block_size=int(page_params["page_block_size"]),
            max_seq_len=max_seq_len,
        )
    if not hasattr(tokenizer, "stop_tokens"):
        tokenizer.stop_tokens = [tokenizer.eos_token_id]
    logger.info(f"Generator ready in {time.time() - t0:.1f}s")

    input_tokens_prefill_pt, decoding_pos, encoded_prompts = _prepare_batch_prefill_tokens(
        prompts,
        tokenizer,
        target_prefill_len=prefill_len,
        instruct=True,
    )
    prefill_seq_len = get_padded_prefill_len(decoding_pos[0])
    logger.info(
        f"Batch prefill: users={batch_size}, prompt_len={decoding_pos[0]}, "
        f"kernel_seq_len={prefill_seq_len}, tensor_shape={tuple(input_tokens_prefill_pt.shape)}"
    )

    logger.info("Starting batched prefill warmup...")
    profiler.start("compile_prefill")
    generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        warmup_prefill=True,
        enable_trace=enable_prefill_trace,
    )
    profiler.end("compile_prefill")
    logger.info(f"Prefill warmup done in {profiler.get_duration('compile_prefill'):.2f}s")

    logger.info("Starting batched prefill (measured)...")
    profiler.start("inference_prefill")
    prefill_out = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        warmup_prefill=False,
        enable_trace=enable_prefill_trace,
    )
    prefilled_token = torch.argmax(prefill_out, dim=-1)
    profiler.end("inference_prefill")
    logger.info(
        f"Batched prefill finished in {profiler.get_duration('inference_prefill'):.2f}s; "
        f"first tokens sample: {prefilled_token[:4].reshape(-1).tolist()}"
    )

    all_outputs = [encoded_prompts[b][: decoding_pos[b]] for b in range(batch_size)]
    for user in range(batch_size):
        all_outputs[user].append(int(prefilled_token[user].item()))

    current_pos = torch.tensor(decoding_pos, dtype=torch.int32)
    out_tok = prefilled_token
    user_done = [False] * batch_size
    users_decoding = max_new_tokens > 1

    if users_decoding:
        logger.info(f"Starting batched decode ({max_new_tokens - 1} steps)...")
        profiler.start("inference_decode")
        for iteration in range(max_new_tokens - 1):
            if iteration == 0:
                profiler.start("compile_decode")
            else:
                profiler.start(f"inference_decode_time_{iteration}")

            decode_out = generator.decode_forward(
                out_tok,
                current_pos,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                enable_trace=enable_decode_trace,
                reset_batch=(iteration == 0),
            )
            if isinstance(decode_out, tuple):
                logits = decode_out[0]
            else:
                logits = decode_out
            _, out_tok = sample_host(logits, temperature=0, top_p=1.0, on_host=True)

            if iteration == 0:
                profiler.end("compile_decode")
            else:
                profiler.end(f"inference_decode_time_{iteration}")

            current_pos += 1
            for user in range(batch_size):
                user_tok = int(out_tok[user].item())
                if user_tok not in tokenizer.stop_tokens and not user_done[user]:
                    all_outputs[user].append(user_tok)
                else:
                    user_done[user] = True
            if all(user_done):
                break
        profiler.end("inference_decode")

    generated_texts = [tokenizer.decode(all_outputs[b]) for b in range(batch_size)]
    profiler.end("run")

    logger.info("")
    logger.info("=== Batch performance metrics ===")
    logger.info(f"Prefill compile time: {profiler.get_duration('compile_prefill'):.2f}s")
    logger.info(f"Prefill inference time (TTFT): {profiler.get_duration('inference_prefill') * 1000:.2f}ms")
    if users_decoding and profiler.get_duration("compile_decode") > 0:
        logger.info(f"Decode compile time: {profiler.get_duration('compile_decode'):.2f}s")
    logger.info(f"Full batch demo runtime: {profiler.get_duration('run'):.2f}s")

    if is_ci_env:
        targets = {}
        bench_n_warmup_iter = {"inference_prefill": 0, "inference_decode": 1}
        measurements = ["compile_prefill", "inference_prefill"]
        if users_decoding:
            measurements.extend(["compile_decode", "inference_decode"])
        benchmark_data = create_benchmark_data(profiler, measurements, bench_n_warmup_iter, targets)
        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo",
            ml_model_name="gemma4",
            ml_model_type="llm",
            device_name=determine_device_name(mesh_device),
            num_layers=model_args.num_hidden_layers,
            batch_size=batch_size,
            config_params={"prefill_len": prefill_len, "batched_prefill": True},
            input_sequence_length=prefill_len,
            output_sequence_length=max_new_tokens,
        )

    return {
        "prefilled_tokens": prefilled_token,
        "generated_texts": generated_texts,
        "prefill_seq_len": prefill_seq_len,
        "prompt_lens": decoding_pos,
    }


def _run_generation_via_generator(
    mesh_device,
    model_path,
    prompts,
    max_new_tokens,
    num_layers,
    max_seq_len,
    page_params,
    bounded_sliding,
    enable_decode_trace=True,
):
    """Long-context / policy path via ``Gemma4Generator`` (matches text_demo_v2).

    Required for bounded sliding + auto multi-chunk prefill (e.g. 256k on 31B/12B/26B).
    """
    is_ci_env = os.environ.get("CI") == "true"
    batch_size = len(prompts)
    assert batch_size == 1, "Generator long-context path is batch=1 in text_demo.py"

    profiler = BenchmarkProfiler()
    profiler.start("run")

    block_size = int(page_params["page_block_size"])
    page_max_num_blocks = _right_size_page_max_num_blocks(batch_size, max_seq_len, page_params)
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=page_max_num_blocks)
    page_table = _create_tt_page_table(batch_size, paged_attention_config)

    logger.info(
        f"Loading Gemma4 via Generator (layers={num_layers or 'all'}, max_seq_len={max_seq_len}, "
        f"bounded_sliding={bounded_sliding})..."
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=bounded_sliding,
    )
    model_args_list = generator.model_args
    model_args = model_args_list[0]
    if not hasattr(tokenizer, "stop_tokens"):
        tokenizer.stop_tokens = [tokenizer.eos_token_id]

    if bounded_sliding:
        _install_hybrid_page_tables(
            generator.model[0],
            model_args,
            batch_size=batch_size,
            block_size=block_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
        )

    prefill_trace_max = int(os.environ.get("GEMMA4_PREFILL_TRACE_MAX_SEQ", 4096))
    prefill_enable_trace = enable_decode_trace and max_seq_len < prefill_trace_max
    if enable_decode_trace and not prefill_enable_trace:
        logger.info(
            f"Prefill trace disabled (max_seq_len={max_seq_len} >= {prefill_trace_max}); "
            f"decode stays traced. Set GEMMA4_PREFILL_TRACE_MAX_SEQ to override."
        )

    logger.info("Warming up prefill...")
    generator.warmup_model_prefill(
        kv_cache=tt_kv_cache,
        enable_trace=prefill_enable_trace,
        can_sample_on_device=False,
        greedy_only=True,
    )
    logger.info("Warmup complete")

    input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        prompts, tokenizer, model_args_list, True, max_new_tokens, max_prefill_len=max_seq_len
    )
    max_encoded_prompt_len = max(len(p) for p in encoded_prompts)
    assert max_new_tokens + max_encoded_prompt_len <= max_seq_len, (
        f"prompt ({max_encoded_prompt_len}) + max_new_tokens ({max_new_tokens}) "
        f"must be <= max_seq_len ({max_seq_len})"
    )
    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)

    logger.info("Starting prefill...")
    profiler.start("inference_prefill")
    prefill_logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        warmup_prefill=False,
        enable_trace=prefill_enable_trace,
    )
    prefilled_token = _host_sample_greedy(prefill_logits)
    profiler.end("inference_prefill")

    prefilled_flat = prefilled_token.view(batch_size, -1).squeeze(-1)
    all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(batch_size)]
    for user in range(batch_size):
        all_outputs[user].append(int(prefilled_flat[user].item()))

    current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)])
    out_tok = prefilled_flat.reshape(batch_size, 1)
    iteration = 0
    logger.info("Starting decode loop...")
    profiler.start("inference_decode")
    while iteration < max_new_tokens:
        profiler.start(f"inference_decode_time_{iteration}")
        logits, _ = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=enable_decode_trace,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=None,
        )
        out_tok = _host_sample_greedy(logits)
        profiler.end(f"inference_decode_time_{iteration}")
        current_pos += 1
        for user in range(batch_size):
            tok = int(out_tok[user, 0].item())
            if tok not in tokenizer.stop_tokens:
                all_outputs[user].append(tok)
        if not is_ci_env:
            for user in range(batch_size):
                text = tokenizer.decode(all_outputs[user][prefill_lens[user] :])
                text = ("..." + text[-97:]) if len(text) > 100 else text
                logger.info(f"[User {user}] {text.replace(chr(10), ' ')}")
        iteration += 1
    profiler.end("inference_decode")
    profiler.end("run")

    generated_texts = []
    for i, output in enumerate(all_outputs):
        gen_text = tokenizer.decode(output[prefill_lens[i] :])
        full = tokenizer.decode(output)
        generated_texts.append(full)
        logger.info(f"\n==USER {i} - GENERATION ONLY\n{gen_text.strip()}\n")

    ttft_ms = profiler.get_duration("inference_prefill") * 1000
    logger.info(f"Time to First Token (TTFT): {ttft_ms:.1f} ms")
    return generated_texts


def run_generation(
    mesh_device,
    model_path,
    prompts,
    max_new_tokens=32,
    num_layers=None,
    max_seq_len=4096,
    page_params=None,
    enable_decode_trace=True,
    target_prefill_len=None,
):
    """
    Run text generation with Gemma4.

    Args:
        mesh_device: TT device
        model_path: Path to model weights
        prompts: List of prompt strings
        max_new_tokens: Number of tokens to generate per prompt
        num_layers: Override layer count (for quick testing)
        max_seq_len: Maximum sequence length (determines KV cache size)
        page_params: Paged attention params dict with "page_block_size" and "page_max_num_blocks"
        target_prefill_len: If set, force the prefill bucket to this exact length
            (truncating the tokenized prompt if necessary). Used by the
            length-parametrized demo tests; otherwise the bucket is chosen
            dynamically via _snap_to_bucket.

    Returns:
        List of generated text strings
    """
    from transformers import AutoTokenizer

    max_new_tokens = int(os.environ.get("GEMMA4_MAX_NEW_TOKENS", max_new_tokens))
    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", max_seq_len))
    _layers_env = os.environ.get("GEMMA4_NUM_LAYERS")
    if _layers_env:
        num_layers = int(_layers_env)

    is_ci_env = os.environ.get("CI") == "true"
    batch_size = 1  # Gemma4 demo is single-user

    if page_params is None:
        page_params = {"page_block_size": 64, "page_max_num_blocks": max_seq_len // 64}
    page_params = dict(page_params)
    page_params["page_max_num_blocks"] = _right_size_page_max_num_blocks(batch_size, max_seq_len, page_params)

    bounded_sliding = _resolve_bounded_sliding(max_seq_len, mesh_device, model_path)
    needs_chunk = should_auto_enable_chunked_bounded(
        max_seq_len, mesh_device, model_path, bounded_sliding=bounded_sliding
    )
    # Long-context / policy path: Generator provides bounded sliding + multi-chunk
    # prefill (hand-rolled ttnn_prefill_forward is single-chunk only).
    if bounded_sliding or needs_chunk or max_seq_len >= 65536:
        return _run_generation_via_generator(
            mesh_device=mesh_device,
            model_path=model_path,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            page_params=page_params,
            bounded_sliding=bounded_sliding,
            enable_decode_trace=enable_decode_trace,
        )

    profiler = BenchmarkProfiler()
    profiler.start("run")

    # Load tokenizer
    profiler.start("loading_inputs")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info(f"Tokenizer loaded from {model_path}")
    profiler.end("loading_inputs")

    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )

    # Create page table (identity mapping for single-user)
    page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
        batch_size, paged_attention_config.max_num_blocks
    )

    # Create model
    logger.info(
        f"Creating model with {num_layers or 'all'} layers, max_seq_len={max_seq_len}, "
        f"bounded_sliding={bounded_sliding}..."
    )
    t0 = time.time()
    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        model_path=model_path,
        create_kv_cache=True,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=bounded_sliding,
    )
    if bounded_sliding:
        _install_hybrid_page_tables(
            model,
            model_args,
            batch_size=batch_size,
            block_size=int(page_params["page_block_size"]),
            max_seq_len=max_seq_len,
            num_layers=num_layers,
        )
    logger.info(f"Model created in {time.time() - t0:.1f}s")

    is_mesh = hasattr(mesh_device, "shape")
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    # Page table on device
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
    )

    generated_texts = []

    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f"\n{'='*60}")
        logger.info(f"Prompt {prompt_idx}: {_shorten_for_log(prompt)}")

        # Tokenize using chat template for instruct models
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            chat_result = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )
            input_ids = chat_result["input_ids"].squeeze(0)  # [seq_len]
        else:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)

        prompt_len = input_ids.shape[0]
        if target_prefill_len is not None:
            padded_len = target_prefill_len
        else:
            padded_len = _snap_to_bucket(prompt_len, max_seq_len)

        # Tokenized prompt may exceed the chosen bucket (long-context files
        # tokenize to more tokens than their character cap predicts) — truncate
        # in that case before zero-padding up to the bucket.
        if prompt_len > padded_len:
            input_ids = input_ids[:padded_len]
            prompt_len = padded_len
        input_ids_padded = torch.nn.functional.pad(input_ids, (0, padded_len - prompt_len), value=0)
        logger.info(f"Prompt tokens: {prompt_len} (padded to {padded_len})")

        # Prefill — two calls so TTFT is measured *after* kernel compilation.
        # The first call (warmup) compiles the prefill program for this bucket
        # length and writes the prompt's K/V into the cache. The second call
        # hits the same cached program and overwrites the same K/V slots with
        # identical data — the timing is dominated by inference, not compile.
        # Pattern mirrors tt_transformers simple_text_demo.py and gpt_oss
        # text_demo.py; `prefill_time_to_token` below now reflects inference
        # only.
        import traceback as tb

        # CPU tensors for per-layer input computation (E2B/E4B) — reusable
        # across both prefill calls below.
        import torch.nn.functional as F

        embeds_torch = (
            F.embedding(
                input_ids_padded.unsqueeze(0).long(),
                state_dict.get(
                    "model.language_model.embed_tokens.weight",
                    state_dict.get("model.embed_tokens.weight", torch.zeros(1)),
                ),
            )
            * model.embed_scale
        ).float()

        # Get last token tile for first decode token
        get_last_token = ((prompt_len - 1) // 32) * 32

        def _build_prefill_embeds():
            """Build a fresh ttnn embeds tensor for ttnn_prefill_forward.

            The model deallocates intermediate hidden_states tensors as it
            walks layers (memory pressure on long prompts), which frees the
            input embeds buffer too. Rebuild before each prefill call so the
            warmup pass and the measured pass each see a live tensor.
            Pattern matches tt_transformers' generator.prefill_forward_text,
            which receives torch tokens and re-tokenizes/re-embeds internally
            each call.
            """
            tokens_tt = ttnn.from_torch(
                input_ids_padded.unsqueeze(0).to(torch.int32),
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                mesh_mapper=replicate,
            )
            e = model.embed_tokens(tokens_tt)
            e = ttnn.reshape(e, (1, 1, padded_len, model_args.hidden_size))
            return ttnn.to_layout(e, ttnn.TILE_LAYOUT)

        # ── Warmup prefill (compile cost, untimed for TTFT) ─────────────
        logger.info("Prefill warmup (compiling)...")
        profiler.start(f"compile_prefill", iteration=prompt_idx)
        try:
            warmup_embeds = _build_prefill_embeds()
            warmup_logits = model.ttnn_prefill_forward(
                warmup_embeds,
                page_table=page_table_tt,
                kv_cache=tt_kv_cache,
                get_last_token=get_last_token,
                input_ids_torch=input_ids_padded.unsqueeze(0),
                embeds_torch=embeds_torch,
            )
        except Exception as e:
            logger.error(f"Prefill warmup failed: {e}")
            tb.print_exc()
            raise
        warmup_logits.deallocate(True)
        profiler.end(f"compile_prefill", iteration=prompt_idx)
        logger.info(f"Prefill warmup done in {profiler.get_duration('compile_prefill', iteration=prompt_idx):.2f}s")

        # ── Measured prefill (TTFT) ─────────────────────────────────────
        logger.info("Prefilling (measured)...")
        profiler.start(f"inference_prefill", iteration=prompt_idx)
        try:
            embeds = _build_prefill_embeds()
            logits = model.ttnn_prefill_forward(
                embeds,
                page_table=page_table_tt,
                kv_cache=tt_kv_cache,
                get_last_token=get_last_token,
                input_ids_torch=input_ids_padded.unsqueeze(0),
                embeds_torch=embeds_torch,
            )
        except Exception as e:
            logger.error(f"Prefill failed: {e}")
            tb.print_exc()
            raise

        # Sample first token (argmax from last position) — included in the
        # TTFT window so the metric matches the user-visible "time to first
        # token" (tt_transformers / gpt_oss put argmax inside the same window).
        if is_mesh:
            logits_cpu = ttnn.to_torch(ttnn.get_device_tensors(logits)[0])
        else:
            logits_cpu = ttnn.to_torch(logits)
        logits.deallocate(True)

        # Get logits at the actual last prompt position within the tile
        pos_in_tile = (prompt_len - 1) - get_last_token
        next_token = logits_cpu[0, 0, pos_in_tile, :].argmax().item()
        profiler.end(f"inference_prefill", iteration=prompt_idx)

        logger.info(
            f"Prefill measured in {profiler.get_duration('inference_prefill', iteration=prompt_idx):.2f}s, "
            f"first token: {next_token} = '{tokenizer.decode([next_token])}'"
        )

        # Decode loop
        generated_tokens = [next_token]
        current_pos = prompt_len
        iteration = 0
        trace_id = None
        trace_output = None
        trace_device_inputs = None

        # ── Decode helpers ─────────────────────────────────────────────────
        # Token IDs (+ optional PLI) staged on host; embedding lookup runs on
        # device inside ``ttnn_decode_forward``. Trace captures decoder onward.
        # Sampling: SamplingGenerator for TP >= 2, host torch.argmax for TP = 1.
        on_device_sampling = model.sampling is not None

        def _make_decode_inputs(tok, pos):
            """Create host tensors for one decode iteration."""
            pli_torch = model.compute_host_pli(tok)
            tokens_h = ttnn.from_torch(
                torch.tensor([[tok]], dtype=torch.int32),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                mesh_mapper=replicate,
            )
            pos_padded = torch.nn.functional.pad(
                torch.tensor([pos], dtype=torch.int32).reshape(1, 1), (0, 31), "constant", 0
            )
            inputs = {
                "tokens": tokens_h,
                "position": ttnn.from_torch(
                    pos_padded,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.uint32,
                    mesh_mapper=replicate,
                ),
                "position_int32": ttnn.from_torch(
                    torch.tensor([pos], dtype=torch.int32),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.int32,
                    mesh_mapper=replicate,
                ),
            }
            if pli_torch is not None:
                inputs["pli"] = ttnn.from_torch(
                    pli_torch.to(torch.bfloat16),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=replicate,
                )
            return inputs

        def _fwd(device_inputs):
            out = model.ttnn_decode_forward(
                x=device_inputs["tokens"],
                current_pos=device_inputs["position"],
                rot_mat_idxs=device_inputs["position_int32"],  # pos_int32 passed as rot_mat_idxs
                page_table=page_table_tt,
                kv_cache=tt_kv_cache,
                on_device_logits=on_device_sampling,
                pli_combined=device_inputs.get("pli"),
            )
            # ttnn_decode_forward returns a bare logits tensor on the on-device-sampling
            # path (on_device_logits=True) and a (logits, None) tuple otherwise. Normalize
            # to a 2-tuple so the trace-capture call sites below can always unpack two values.
            return out if isinstance(out, tuple) else (out, None)

        def _inputs_to_device(inputs):
            return {k: ttnn.to_device(v, device=mesh_device) for k, v in inputs.items() if v is not None}

        def _copy_inputs_to_trace(host_inputs):
            for k, v in host_inputs.items():
                if v is not None and k in trace_device_inputs:
                    ttnn.copy_host_to_device_tensor(v, trace_device_inputs[k])

        def _extract_token(decode_output):
            """Extract next token from model output (token IDs or logits)."""
            if on_device_sampling:
                # Keep main behavior: decode sampling in this demo remains untraced.
                # SamplingGenerator.sample() returns (tt_tokens, tt_log_probs); take the tokens.
                sampled = model.sampling.sample(decode_output, enable_trace=False)
                tt_tokens = sampled[0] if isinstance(sampled, tuple) else sampled
                sampled_cpu = (
                    ttnn.to_torch(ttnn.get_device_tensors(tt_tokens)[0]) if is_mesh else ttnn.to_torch(tt_tokens)
                )
                return sampled_cpu.reshape(-1)[0].item()
            else:
                output_cpu = (
                    ttnn.to_torch(ttnn.get_device_tensors(decode_output)[0])
                    if is_mesh
                    else ttnn.to_torch(decode_output)
                )
                return output_cpu.squeeze().argmax().item()

        sample_mode = "device" if on_device_sampling else "host"
        logger.info(
            f"Decoding (trace={'ON' if enable_decode_trace else 'OFF'}, "
            f"embedding=device, sampling={sample_mode})..."
        )
        profiler.start(f"inference_decode", iteration=prompt_idx)

        # Disable Python GC during decode to avoid pause spikes; collect once before.
        # The decode loop, trace capture, and trace execution all sit inside a try
        # so that GC is always restored and any captured trace is always released
        # — otherwise an exception leaves GC disabled for the rest of the pytest
        # worker (contaminating unrelated tests) and leaks the trace handle.
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()

        try:
            # ── Main decode loop (mode-agnostic) ──────────────────────────────
            for step in range(max_new_tokens - 1):
                if iteration == 0:
                    profiler.start(f"compile_decode", iteration=prompt_idx)
                else:
                    profiler.start(f"inference_decode_time_{iteration}", iteration=prompt_idx)

                t_make_start = time.perf_counter()
                inputs_h = _make_decode_inputs(next_token, current_pos)
                t_make_end = time.perf_counter()

                if enable_decode_trace and trace_id is not None:
                    # ── Traced execution: copy inputs and replay ──
                    _copy_inputs_to_trace(inputs_h)
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                    decode_logits = trace_output
                    t_enq_end = time.perf_counter()

                elif enable_decode_trace and iteration == 0:
                    # ── Iteration 0: compile run + trace capture ──
                    # 1. Compile run (un-traced)
                    inputs_d = _inputs_to_device(inputs_h)
                    decode_logits, _ = _fwd(inputs_d)
                    next_token = _extract_token(decode_logits)
                    generated_tokens.append(next_token)
                    current_pos += 1
                    profiler.end(f"compile_decode", iteration=prompt_idx)
                    decode_iteration_time = profiler.get_duration("compile_decode", iteration=prompt_idx)
                    logger.debug(
                        f"Iteration {iteration} (compile): {1000*decode_iteration_time:.0f}ms @ "
                        f"{1/decode_iteration_time:.1f} tok/s/user"
                    )
                    iteration += 1

                    # 2. Capture trace with fresh device buffers
                    logger.info("Capturing decode trace...")
                    inputs_h2 = _make_decode_inputs(next_token, current_pos)
                    trace_device_inputs = _inputs_to_device(inputs_h2)

                    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                    trace_output, _ = _fwd(trace_device_inputs)
                    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
                    logger.info("Decode trace captured")

                    # 3. Execute trace for current iteration
                    profiler.start(f"inference_decode_time_{iteration}", iteration=prompt_idx)
                    _copy_inputs_to_trace(inputs_h2)
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                    decode_logits = trace_output
                    t_enq_end = time.perf_counter()

                else:
                    # ── No tracing: straightforward forward ──
                    inputs_d = _inputs_to_device(inputs_h)
                    decode_logits, _ = _fwd(inputs_d)
                    t_enq_end = time.perf_counter()

                next_token = _extract_token(decode_logits)
                t_sync_end = time.perf_counter()
                generated_tokens.append(next_token)
                current_pos += 1

                if iteration == 0:
                    profiler.end(f"compile_decode", iteration=prompt_idx)
                    decode_iteration_time = profiler.get_duration("compile_decode", iteration=prompt_idx)
                else:
                    profiler.end(f"inference_decode_time_{iteration}", iteration=prompt_idx)
                    decode_iteration_time = profiler.get_duration(
                        f"inference_decode_time_{iteration}", iteration=prompt_idx
                    )

                tokens_per_second_per_user = 1 / decode_iteration_time
                host_inputs_ms = 1000 * (t_make_end - t_make_start)
                copy_enq_ms = 1000 * (t_enq_end - t_make_end)
                exec_sync_ms = 1000 * (t_sync_end - t_enq_end)
                logger.debug(
                    f"Iteration {iteration}: {1000*decode_iteration_time:.0f}ms @ "
                    f"{tokens_per_second_per_user:.1f} tok/s/user ({batch_size*tokens_per_second_per_user:.1f} tok/s throughput) "
                    f"| host_inputs={host_inputs_ms:.1f}ms copy+enq={copy_enq_ms:.1f}ms exec+sync={exec_sync_ms:.1f}ms"
                )

                iteration += 1

                # Check for EOS
                if next_token == tokenizer.eos_token_id:
                    break

        finally:
            if gc_was_enabled:
                gc.enable()
            if trace_id is not None:
                ttnn.release_trace(mesh_device, trace_id)

        profiler.end(f"inference_decode", iteration=prompt_idx)

        # Final output
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_text = prompt + generated_text
        generated_texts.append(full_text)

        logger.info(
            f"\n==PROMPT {prompt_idx}\n{_shorten_for_log(prompt)}\n==OUTPUT {prompt_idx}\n{generated_text.strip()}\n"
        )

    num_tokens_generated_decode = iteration  # from last prompt

    profiler.end("run")

    # ── Performance metrics ──────────────────────────────────────────────
    # compile_prefill = first (warmup) prefill call, includes kernel compile.
    # inference_prefill = second prefill call, kernels already cached — drives TTFT.
    compile_prefill_time = profiler.get_duration("compile_prefill")
    compile_decode_time = profiler.get_duration("compile_decode")
    total_inference_prefill_time = profiler.get_duration("inference_prefill")

    total_inference_decode_time = 0
    for i in range(1, num_tokens_generated_decode):  # Iteration 0 is the compile time
        total_inference_decode_time += profiler.get_duration(f"inference_decode_time_{i}")

    avg_time_to_first_token = total_inference_prefill_time / batch_size
    avg_decode_iteration_time = (
        total_inference_decode_time / (num_tokens_generated_decode - 1) if num_tokens_generated_decode > 1 else 0
    )

    prefill_tok_s = prompt_len / total_inference_prefill_time * batch_size if total_inference_prefill_time > 0 else 0
    decode_tok_s_user = (
        (num_tokens_generated_decode - 1) / total_inference_decode_time
        if num_tokens_generated_decode > 1 and total_inference_decode_time > 0
        else 0
    )
    decode_tok_s = decode_tok_s_user * batch_size

    measurements = {
        # Required measurements
        "compile_prefill": compile_prefill_time,
        "compile_decode": compile_decode_time,
        "inference_prefill": total_inference_prefill_time,
        "inference_decode": total_inference_decode_time,
        "prefill_time_to_token": avg_time_to_first_token,
        "prefill_t/s": prefill_tok_s,
        "decode_t/s/u": decode_tok_s_user,
        "decode_t/s": decode_tok_s,
        # Optional measurements
        "Total compile time": compile_prefill_time + compile_decode_time,
        "Full demo runtime": profiler.get_duration("run"),
    }

    # Decode performance at specific token milestones
    tok_1_perf = profiler.get_duration("inference_decode_time_1") if 1 < num_tokens_generated_decode else 0
    tok_128_perf = profiler.get_duration("inference_decode_time_127") if 127 < num_tokens_generated_decode else 0

    logger.info("")
    logger.info("=== Performance metrics ===")
    if tok_1_perf > 0:
        logger.info(
            f"1st token decode time: {tok_1_perf * 1000:.2f}ms "
            f"[{round(1 / tok_1_perf, 2)} t/s/u, {round((1 / tok_1_perf) * batch_size, 2)} t/s]"
        )
    if tok_128_perf > 0:
        logger.info(
            f"128th token decode time: {tok_128_perf * 1000:.2f}ms "
            f"[{round(1 / tok_128_perf, 2)} t/s/u, {round((1 / tok_128_perf) * batch_size, 2)} t/s]"
        )
    logger.info("==")
    logger.info(f"Prefill compile time: {round(compile_prefill_time, 2)}s")
    logger.info(f"Decode compile time: {round(compile_decode_time, 2)}s")
    logger.info("")
    logger.info(f"Average Time to First Token (TTFT): {round(avg_time_to_first_token * 1000, 2)}ms")
    logger.info(
        f"Average speed: {round(avg_decode_iteration_time * 1000, 2)}ms @ "
        f"{round(decode_tok_s_user, 2)} tok/s/user ({round(decode_tok_s, 2)} tok/s throughput)"
    )
    logger.info(f"Generated {num_tokens_generated_decode} tokens")
    logger.info(f"Full demo runtime: {round(profiler.get_duration('run'), 2)}s")

    # Save benchmark data for CI dashboard
    if is_ci_env:
        targets = {}  # No perf targets for Gemma4 yet
        bench_n_warmup_iter = {"inference_prefill": 0, "inference_decode": 1}
        benchmark_data = create_benchmark_data(profiler, measurements, bench_n_warmup_iter, targets)

        # Save the decode performance of every iteration for plotting
        for i in range(1, num_tokens_generated_decode):
            benchmark_data.add_measurement(
                profiler,
                0,
                "inference_decode",
                f"time_to_token_{i}",
                profiler.get_duration(f"inference_decode_time_{i}") * 1000,
                step_warm_up_num_iterations=None,
                target=None,
            )

        # Average decode performance for first 128 iterations (excluding compile)
        num_iterations_for_avg = min(128, num_tokens_generated_decode)
        inference_decode_time_first_128 = sum(
            profiler.get_duration(f"inference_decode_time_{i}") for i in range(1, num_iterations_for_avg)
        )
        benchmark_data.add_measurement(
            profiler,
            0,
            "inference_decode",
            "avg_decode_time_first_128",
            inference_decode_time_first_128 * 1000 / max(1, num_iterations_for_avg - 1),
            step_warm_up_num_iterations=None,
            target=None,
        )

        benchmark_data.save_partial_run_json(
            profiler,
            run_type="demo_perf",
            ml_model_name="gemma4",
            ml_model_type="llm",
            device_name=determine_device_name(mesh_device),
            num_layers=num_layers or model_args.num_hidden_layers,
            batch_size=batch_size,
            config_params={},
            input_sequence_length=prompt_len,
            output_sequence_length=num_tokens_generated_decode,
        )

    return generated_texts


# ── Pytest entry points ──────────────────────────────────────────────────


@pytest.fixture
def model_path():
    return os.getenv("HF_MODEL") or os.getenv(
        "GEMMA4_MODEL_PATH", "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it"
    )


def test_demo_single_layer(device, model_path):
    """Quick demo with 1 layer — verifies the pipeline works on single device."""
    prompts = ["The capital of France is"]
    results = run_generation(
        mesh_device=device,
        model_path=model_path,
        prompts=prompts,
        max_new_tokens=8,
        num_layers=1,
    )
    assert len(results) == 1
    assert len(results[0]) > len(prompts[0])


_DEMO_PREFILL_LENGTHS = [128, 4096]

# NOTE on long-context-64k/128k/256k (see GEMMA4_LONG_CONTEXT_POLICY):
#   Per-(model, device) cutovers on QB2 (P150x4 / P300x2):
#     31B: bounded @ 64k, chunked @ 256k
#     12B/26B-A4B: unbounded through 128k; bounded(+chunked) @ 256k
#     E2B/E4B: unbounded through 256k (HF native max_pos is 128k)
#   Override: GEMMA4_BOUNDED_SLIDING, GEMMA4_GEN_PREFILL_CHUNK,
#   GEMMA4_DEMO_SINGLE_CHUNK.
_LONG_CONTEXT_CASES = [
    # (id, max_seq_len, page_block_size, page_max_num_blocks)
    ("long-context-64k", 64 * 1024, 64, 1024),
    ("long-context-128k", 128 * 1024, 64, 2048),
    ("long-context-256k", 256 * 1024, 64, 4096),
]


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("prefill_len", _DEMO_PREFILL_LENGTHS, ids=[f"prefill_{b}" for b in _DEMO_PREFILL_LENGTHS])
def test_demo(mesh_device, model_path, prefill_len, request):
    """Full model demo — runs on any multi-device mesh, parametrized over a
    short and a long prefill bucket.

    Loads a tt_transformers sample prompt sized for the target bucket
    (short hand-written prompt for 128, long-context book excerpt for 4k),
    forces the prefill kernel to that exact length, and runs 200 decode
    iterations so the full prefill→decode pipeline is exercised end-to-end.
    Wider per-length kernel coverage lives in the unit tests, which sweep
    the full PREFILL_BUCKETS list under --max-prefill.

    Long-context 64k/128k/256k rows live in ``test_demo_long_context``.

    Filter by mesh shape:
        pytest -k "1x2"               # N300 / TP=2
        pytest -k "1x8"               # T3K  / TP=8
    Filter by prefill length:
        pytest -k "prefill_4096"      # 4k prefill only
    """
    max_prefill = request.config.getoption("--max-prefill")
    if prefill_len > max_prefill:
        pytest.skip(f"prefill_len={prefill_len} > --max-prefill={max_prefill}")

    if os.environ.get("CI") == "true" and prefill_len != 128:
        pytest.skip(f"CI: only prefill_128 runs in CI; skipping prefill_{prefill_len}")

    prompt = load_demo_prompt(prefill_len, instruct=True)

    # KV cache must hold the prefill plus the 200 decode tokens. Keep a small
    # floor so short-bucket runs still allocate a usable cache.
    max_new_tokens = int(os.environ.get("GEMMA4_MAX_NEW_TOKENS", 200))
    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", max(prefill_len + max_new_tokens, 4096)))
    page_block_size = 64
    page_params = {
        "page_block_size": page_block_size,
        "page_max_num_blocks": math.ceil(max_seq_len / page_block_size),
    }

    results = run_generation(
        mesh_device=mesh_device,
        model_path=model_path,
        prompts=[prompt],
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        page_params=page_params,
        enable_decode_trace=True,
        target_prefill_len=prefill_len,
    )
    assert len(results) == 1
    logger.info(f"Full model output: {_shorten_for_log(results[0])}")


@pytest.mark.parametrize(
    "case_id, max_seq_len, page_block_size, page_max_num_blocks",
    _LONG_CONTEXT_CASES,
    ids=[c[0] for c in _LONG_CONTEXT_CASES],
)
@pytest.mark.parametrize(
    "mesh_device",
    [_mesh_shape_from_env()],
    indirect=True,
)
def test_demo_long_context(
    mesh_device, model_path, case_id, max_seq_len, page_block_size, page_max_num_blocks, request
):
    """Long-context demo rows aligned with ``text_demo_v2`` / ``GEMMA4_LONG_CONTEXT_POLICY``.

    Uses ``MESH_DEVICE`` (default P150x4 / QB2; also accepts P300x2). Policy auto-enables
    bounded sliding / multi-chunk prefill per model × device.

    Examples:
        MESH_DEVICE=P150x4 HF_MODEL=google/gemma-4-12B-it pytest \\
            models/demos/gemma4/demo/text_demo.py -k long-context-128k -s --timeout 1800
        MESH_DEVICE=P300x2 GEMMA4_BOUNDED_SLIDING=1 HF_MODEL=google/gemma-4-26B-A4B pytest \\
            models/demos/gemma4/demo/text_demo.py -k long-context-256k -s --timeout 7200
    """
    del case_id  # used only as pytest id
    max_prefill = request.config.getoption("--max-prefill")
    if max_seq_len > max_prefill:
        pytest.skip(f"max_seq_len={max_seq_len} > --max-prefill={max_prefill}")

    if os.environ.get("CI") == "true":
        pytest.skip("CI: long-context rows are local/sweep only")

    max_new_tokens = int(os.environ.get("GEMMA4_MAX_NEW_TOKENS", 200))
    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", max_seq_len))
    prompt = load_demo_prompt(max_seq_len, instruct=True)
    page_params = {
        "page_block_size": page_block_size,
        "page_max_num_blocks": page_max_num_blocks,
    }

    results = run_generation(
        mesh_device=mesh_device,
        model_path=model_path,
        prompts=[prompt],
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        page_params=page_params,
        enable_decode_trace=True,
        target_prefill_len=None,  # Generator path uses preprocess_inputs_prefill
    )
    assert len(results) == 1
    logger.info(f"Long-context output: {_shorten_for_log(results[0])}")


@parametrize_mesh_with_fabric()
@pytest.mark.gemma4_batched_prefill
@pytest.mark.parametrize(
    "prefill_len",
    _BATCH_PREFILL_LENGTHS,
    ids=[f"prefill_{b}" for b in _BATCH_PREFILL_LENGTHS],
)
def test_demo_batch_prefill(mesh_device, model_path, prefill_len, request):
    """Batch-32 prefill — validates ``Gemma4Generator.prefill_forward_text``.

    Parametrized over prefill_len ∈ {128, 1024, 2048}. Uses identical prompts
    for all 32 users so batched prefill is eligible. Prefill-only
    (``max_new_tokens=1``).

    Filter examples:
        pytest -k "test_demo_batch_prefill and prefill_128 and 1x1"
        pytest -k "test_demo_batch_prefill and blackhole and 1x4"
    """
    max_prefill = request.config.getoption("--max-prefill")
    if prefill_len > max_prefill:
        pytest.skip(f"prefill_len={prefill_len} > --max-prefill={max_prefill}")

    if os.environ.get("CI") == "true" and prefill_len != 128:
        pytest.skip(f"CI: only prefill_128 runs in CI; skipping prefill_{prefill_len}")

    batch_size = _batch_demo_size()
    _maybe_xfail_batch_prefill_dram(mesh_device, model_path, batch_size, prefill_len)

    max_new_tokens = 1
    max_seq_len = max(prefill_len + max_new_tokens, 4096)
    page_params = _batch_page_params(batch_size, prefill_len, max_new_tokens)

    result = run_batch_generation(
        mesh_device=mesh_device,
        model_path=model_path,
        batch_size=batch_size,
        prefill_len=prefill_len,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        page_params=page_params,
        enable_prefill_trace=True,
    )

    prefilled_tokens = result["prefilled_tokens"]
    expected_kernel_len = get_padded_prefill_len(result["prompt_lens"][0])

    assert prefilled_tokens.shape[0] == batch_size
    assert result["prefill_seq_len"] == expected_kernel_len
    logger.info(
        f"Batch-{batch_size} prefill_{prefill_len} ok — kernel_seq_len={result['prefill_seq_len']}, "
        f"sample token: {int(prefilled_tokens[0].item())}"
    )


@pytest.mark.gemma4_batched_prefill
@parametrize_mesh_with_fabric()
@pytest.mark.parametrize(
    "prefill_len",
    _DEMO_BATCH_PREFILL_LENGTHS,
    ids=[f"prefill_{b}" for b in _DEMO_BATCH_PREFILL_LENGTHS],
)
def test_demo_batch_32(mesh_device, model_path, prefill_len, request):
    """Batch-32 demo — validates batched prefill via ``Gemma4Generator``.

    Parametrized over prefill_len ∈ {128, 512, 1024, 2048, 4096}.
    Uses identical prompts for all 32 users so batched prefill is eligible,
    and exercises the chunking override at 32×4096 (128k token ceiling).

    Filter examples:
        pytest -k "test_demo_batch_32 and prefill_4096"
        pytest -k "test_demo_batch_32 and 1x8"
    """
    max_prefill = request.config.getoption("--max-prefill")
    if prefill_len > max_prefill:
        pytest.skip(f"prefill_len={prefill_len} > --max-prefill={max_prefill}")

    if os.environ.get("CI") == "true" and prefill_len != 128:
        pytest.skip(f"CI: only prefill_128 runs in CI; skipping prefill_{prefill_len}")

    batch_size = _batch_demo_size()
    _maybe_xfail_batch_prefill_dram(mesh_device, model_path, batch_size, prefill_len)

    max_new_tokens = 32
    max_seq_len = max(prefill_len + max_new_tokens, 4096)
    page_params = _batch_page_params(batch_size, prefill_len, max_new_tokens)

    result = run_batch_generation(
        mesh_device=mesh_device,
        model_path=model_path,
        batch_size=batch_size,
        prefill_len=prefill_len,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        page_params=page_params,
        enable_prefill_trace=True,
        enable_decode_trace=False,
    )

    prefilled_tokens = result["prefilled_tokens"]
    generated_texts = result["generated_texts"]
    assert prefilled_tokens.shape[0] == batch_size
    assert len(generated_texts) == batch_size
    for user, text in enumerate(generated_texts):
        assert len(text) > 0, f"User {user} produced empty output"
    logger.info(
        f"Batch-{batch_size} prefill_{prefill_len} ok — kernel_seq_len={result['prefill_seq_len']}, "
        f"sample output: {_shorten_for_log(generated_texts[0])}"
    )


@parametrize_mesh_with_fabric()
@pytest.mark.gemma4_batched_prefill
def test_demo_batch_prefill_4096_ceiling(mesh_device, model_path, request):
    """Document the 128k batched-prefill ceiling at batch 32 × seq 4096.

    32 × 4096 = 131072 tokens, which meets/exceeds ``GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN`` (128k).
    ``Gemma4Generator`` chunks into smaller user batches (e.g. 32→16+16 at 4096).
    This test documents the threshold; use ``test_demo_batch_32`` for the device run.

    Filter examples:
        pytest -k "test_demo_batch_prefill_4096_ceiling and 1x1"
    """
    del mesh_device, model_path  # Documentation-only; device run is test_demo_batch_32.

    max_prefill = request.config.getoption("--max-prefill")
    prefill_len = 4096
    if prefill_len > max_prefill:
        pytest.skip(f"prefill_len={prefill_len} > --max-prefill={max_prefill}")

    batch_size = _BATCH32_DRAM_OOM_SIZE
    kernel_len = get_padded_prefill_len(prefill_len)
    total_tokens = batch_size * kernel_len

    assert _batch_prefill_hits_ceiling(batch_size, prefill_len)

    pytest.skip(
        f"Batch-32 prefill at seq_len={prefill_len} totals {total_tokens} tokens "
        f"({batch_size}×{kernel_len}), which exceeds "
        f"GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN ({GEMMA4_MAX_BATCHED_PREFILL_SEQ_LEN}). "
        f"Gemma4Generator chunks this case automatically. "
        f"Run test_demo_batch_32[prefill_4096] for batched-path coverage."
    )
