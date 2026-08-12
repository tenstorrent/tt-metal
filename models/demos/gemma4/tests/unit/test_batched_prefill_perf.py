# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched vs sequential prefill throughput.

Logs batched vs sequential aggregate prefill tok/s. Default shape: **31B blackhole
1×4, batch=32, prefill_len=1024** (fits DRAM; ``32×4096`` may OOM on KV init).
On smaller batches and ISLs, sequential prefill is often faster today — use this
test to compare modes rather than assuming batched always wins.

Example (31B blackhole 1×4 — timings only, no speedup assert):

    unset TT_VISIBLE_DEVICES TT_MESH_GRAPH_DESC_PATH
    export HF_MODEL=google/gemma-4-31B-it
    export TT_ARCH=blackhole

    pytest models/demos/gemma4/tests/unit/test_batched_prefill_perf.py \\
        -k "1x4" -v -s --timeout=7200

Override shape (``32×2048`` / ``32×4096`` xfail on 31B 1×4):

    export GEMMA4_PREFILL_PERF_BATCH_SIZE=32
    export GEMMA4_PREFILL_PERF_PREFILL_LEN=2048

Enable speedup assertion when a target threshold is defined:

    export GEMMA4_PREFILL_PERF_ASSERT=1
    export GEMMA4_PREFILL_PERF_MIN_SPEEDUP=1.5
"""

import os
import time

import pytest
from loguru import logger

import ttnn
from models.demos.gemma4.demo.text_demo import (
    _batch_page_params,
    _create_tt_page_table,
    _maybe_xfail_batch_prefill_dram,
    _prepare_batch_prefill_tokens,
    load_batch_demo_prompts,
)
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.demos.gemma4.tt.generator_trace import skip_gemma4_full_prefill_warmup
from models.tt_transformers.tt.common import PagedAttentionConfig, get_padded_prefill_len

from ..test_factory import TestFactory, _get_model_path, parametrize_mesh_with_fabric

# Default perf logging shape (31B blackhole 1×4). Override via GEMMA4_PREFILL_PERF_*.
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_PREFILL_LEN = 1024
_DEFAULT_MIN_SPEEDUP = 1.5
_TARGET_MESH = "1x4"


def _perf_batch_size():
    return int(os.environ.get("GEMMA4_PREFILL_PERF_BATCH_SIZE", str(_DEFAULT_BATCH_SIZE)))


def _perf_prefill_len():
    return int(os.environ.get("GEMMA4_PREFILL_PERF_PREFILL_LEN", str(_DEFAULT_PREFILL_LEN)))


def _mesh_shape_str(mesh_device):
    return "x".join(str(d) for d in mesh_device.shape)


def _sync_device(mesh_device):
    ttnn.synchronize_device(mesh_device)


def _assert_speedup_enabled():
    return os.environ.get("GEMMA4_PREFILL_PERF_ASSERT", "0") == "1"


def _min_speedup():
    return float(os.environ.get("GEMMA4_PREFILL_PERF_MIN_SPEEDUP", str(_DEFAULT_MIN_SPEEDUP)))


def _aggregate_prefill_tok_s(batch_size, prompt_len, elapsed_s):
    if elapsed_s <= 0:
        return 0.0
    return batch_size * prompt_len / elapsed_s


def _timed_prefill(generator, tokens, page_table, kv_cache, prompt_lens, *, batched: bool):
    """Warm up then time one prefill forward for batched or sequential mode."""
    generator.model_args[0].disable_batched_prefill = not batched
    skip_gemma4_full_prefill_warmup(generator)
    generator.prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        enable_trace=False,
        warmup_prefill=False,
    )
    _sync_device(generator.mesh_device)
    start = time.perf_counter()
    generator.prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        enable_trace=False,
        warmup_prefill=False,
    )
    _sync_device(generator.mesh_device)
    return time.perf_counter() - start


def _measure_batched_vs_sequential_prefill(
    mesh_device,
    model_path,
    *,
    batch_size,
    prefill_len,
):
    max_new_tokens = 1
    # Size KV to this prefill bucket only — do not floor at 4096 (OOM on 31B 1×4 batch 32).
    max_seq_len = prefill_len + max_new_tokens
    page_params = _batch_page_params(batch_size, prefill_len, max_new_tokens)
    paged_attention_config = PagedAttentionConfig(
        block_size=int(page_params["page_block_size"]),
        max_num_blocks=int(page_params["page_max_num_blocks"]),
    )
    page_table = _create_tt_page_table(batch_size, paged_attention_config)

    generator, kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_attention_config,
    )

    prompts = load_batch_demo_prompts(batch_size, prefill_len, instruct=True)
    tokens, prompt_lens, _encoded_prompts = _prepare_batch_prefill_tokens(
        prompts,
        tokenizer,
        target_prefill_len=prefill_len,
        instruct=True,
    )
    prompt_len = int(prompt_lens[0])
    kernel_len = get_padded_prefill_len(prompt_len)

    batched_s = _timed_prefill(generator, tokens, page_table, kv_cache, prompt_lens, batched=True)
    sequential_s = _timed_prefill(generator, tokens, page_table, kv_cache, prompt_lens, batched=False)

    batched_tok_s = _aggregate_prefill_tok_s(batch_size, prompt_len, batched_s)
    sequential_tok_s = _aggregate_prefill_tok_s(batch_size, prompt_len, sequential_s)
    speedup = batched_tok_s / sequential_tok_s if sequential_tok_s > 0 else float("inf")

    return {
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "kernel_len": kernel_len,
        "batched_s": batched_s,
        "sequential_s": sequential_s,
        "batched_tok_s": batched_tok_s,
        "sequential_tok_s": sequential_tok_s,
        "speedup": speedup,
    }


@pytest.mark.gemma4_batched_prefill
@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric()
def test_batched_prefill_speedup_over_sequential(mesh_device, reset_seeds, request):
    """Log batched vs sequential prefill tok/s; assert speedup when enabled via env."""
    batch_size = _perf_batch_size()
    prefill_len = _perf_prefill_len()

    max_prefill = request.config.getoption("--max-prefill")
    if prefill_len > max_prefill:
        pytest.skip(f"prefill_len={prefill_len} > --max-prefill={max_prefill}")

    hf_config = TestFactory.create_hf_config()
    if int(getattr(hf_config, "hidden_size_per_layer_input", 0) or 0) > 0:
        pytest.skip("PLI models not in scope for batched prefill perf")

    model_path = _get_model_path()
    mesh_key = _mesh_shape_str(mesh_device)
    if mesh_key != _TARGET_MESH:
        pytest.skip(
            f"Default perf logging targets mesh {_TARGET_MESH} (31B blackhole); got {mesh_key}. "
            f"Set GEMMA4_PREFILL_PERF_BATCH_SIZE / GEMMA4_PREFILL_PERF_PREFILL_LEN to run elsewhere."
        )

    _maybe_xfail_batch_prefill_dram(mesh_device, model_path, batch_size, prefill_len)

    logger.info(
        "Batched prefill perf: model={} mesh={} batch={} prefill_len={} assert_speedup={} min_speedup={}",
        os.path.basename(model_path.rstrip("/")),
        mesh_key,
        batch_size,
        prefill_len,
        _assert_speedup_enabled(),
        _min_speedup() if _assert_speedup_enabled() else "n/a",
    )

    results = _measure_batched_vs_sequential_prefill(
        mesh_device,
        model_path,
        batch_size=batch_size,
        prefill_len=prefill_len,
    )

    logger.info(
        "Prefill perf batch={} prompt_len={} kernel_len={}: "
        "batched={:.3f}s ({:.1f} tok/s) sequential={:.3f}s ({:.1f} tok/s) speedup={:.2f}x",
        results["batch_size"],
        results["prompt_len"],
        results["kernel_len"],
        results["batched_s"],
        results["batched_tok_s"],
        results["sequential_s"],
        results["sequential_tok_s"],
        results["speedup"],
    )

    if not _assert_speedup_enabled():
        logger.info(
            "Speedup assertion disabled (set GEMMA4_PREFILL_PERF_ASSERT=1 and "
            "GEMMA4_PREFILL_PERF_MIN_SPEEDUP to enforce a target speedup)"
        )
        return

    min_speedup = _min_speedup()
    assert results["speedup"] >= min_speedup, (
        f"Batched prefill speedup {results['speedup']:.2f}x below {min_speedup}x "
        f"(batched={results['batched_tok_s']:.1f} tok/s, "
        f"sequential={results['sequential_tok_s']:.1f} tok/s)"
    )
