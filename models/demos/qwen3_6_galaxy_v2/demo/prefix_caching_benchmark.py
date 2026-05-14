# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama3_70b_galaxy.demo.text_demo import create_tt_model
from models.demos.llama3_70b_galaxy.tt.generator import Generator, SamplingParams
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from models.perf.benchmarking_utils import BenchmarkProfiler

PREFILL_BENCHMARK_OUTPUT = Path(__file__).resolve().parent / "output" / "prefill_prefix_caching_benchmark.json"
PREFILL_BENCHMARK_TARGETS = Path(__file__).resolve().parent / "prefill_prefix_caching_targets.json"
PREFILL_BENCHMARK_MAX_SLOWDOWN = 1.05
PREFILL_BENCHMARK_MIN_SPEEDUP = 0.80

# Seq lengths (powers of 2 from 128 to 128k).
PREFILL_BENCHMARK_SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
PREFILL_BENCHMARK_BLOCK_SIZE = 64


def _make_synthetic_prefill_input(batch_size, seq_len, vocab_size, dtype=torch.long):
    """Create synthetic token ids for prefill (no file load)."""
    return torch.randint(0, vocab_size, (batch_size, seq_len), dtype=dtype)


def _prefill_benchmark_result_key(result):
    return (
        result["seq_len"],
        bool(result["use_prefix_caching"]),
        int(round(result["prefix_cached_ratio"] * 100)),
    )


def _load_prefill_benchmark_targets():
    with open(PREFILL_BENCHMARK_TARGETS, "r") as f:
        payload = json.load(f)
    return {_prefill_benchmark_result_key(result): result["prefill_s"] for result in payload["results"]}


def _check_prefill_benchmark_results(results):
    expected_results = _load_prefill_benchmark_targets()
    measured_results = {_prefill_benchmark_result_key(result): result for result in results}

    missing_targets = sorted(set(expected_results) - set(measured_results))
    unexpected_results = sorted(set(measured_results) - set(expected_results))
    assert not missing_targets, f"Missing measured benchmark rows for target keys: {missing_targets}"
    assert not unexpected_results, f"Measured benchmark rows missing targets: {unexpected_results}"

    failures = []
    for key in sorted(expected_results):
        expected_prefill_s = expected_results[key]
        measured_prefill_s = measured_results[key]["prefill_s"]
        lower_bound = expected_prefill_s * PREFILL_BENCHMARK_MIN_SPEEDUP
        upper_bound = expected_prefill_s * PREFILL_BENCHMARK_MAX_SLOWDOWN

        if measured_prefill_s > upper_bound:
            failures.append(
                f"{key}: measured {measured_prefill_s:.6f}s is slower than allowed upper bound "
                f"{upper_bound:.6f}s (+5%) for target {expected_prefill_s:.6f}s"
            )
        elif measured_prefill_s < lower_bound:
            failures.append(
                f"{key}: measured {measured_prefill_s:.6f}s is faster than allowed lower bound "
                f"{lower_bound:.6f}s (-20%) for target {expected_prefill_s:.6f}s. "
                "Please update prefill_prefix_caching_targets.json."
            )

    assert not failures, "Prefill prefix-caching benchmark check failed:\n" + "\n".join(failures)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 216580672,
            "num_command_queues": 1,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "worker_l1_size": 1345000,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_prefill_prefix_caching_benchmark(mesh_device):
    """
    Measure prefill time (after warmup) for seq_len in [128..32k] (powers of 2),
    with no prefix caching vs 50%/75%/90% prefix cached. Uses synthetic input tokens.
    Results written to demo/output/.
    """
    page_params = {"page_block_size": PREFILL_BENCHMARK_BLOCK_SIZE, "page_max_num_blocks": 2048}
    batch_size = 1

    model_args, model, page_table, tt_kv_cache = create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=batch_size,
        optimizations=LlamaOptimizations.performance,
        max_seq_len=128 * 1024,
        num_layers=80,
        dummy_weights=False,
        page_params=page_params,
        dtype=ttnn.bfloat8_b,
        use_paged_kv_cache=True,
        prefill_profile=False,
    )
    model_args.tokenizer = Tokenizer(model_args.tokenizer_path)
    generator = Generator(model, model_args, mesh_device, tokenizer=model_args.tokenizer)
    vocab_size = model_args.vocab_size

    sampling_params = SamplingParams(temperature=0.0, top_p=0.05, top_k=32)
    results = []

    for seq_len in PREFILL_BENCHMARK_SEQ_LENS:
        # Align to block_size for prefix-caching (generator asserts alignment)
        seq_len = (seq_len // PREFILL_BENCHMARK_BLOCK_SIZE) * PREFILL_BENCHMARK_BLOCK_SIZE
        if seq_len == 0:
            continue

        input_tokens_prefill_pt = _make_synthetic_prefill_input(batch_size, seq_len, vocab_size)
        decoding_pos = torch.tensor([seq_len], dtype=torch.long)

        for use_prefix_caching, prefix_cached_ratio in [(False, 0.0), (True, 0.5), (True, 0.75), (True, 0.90)]:
            # Compute start_pos for prefix-cached case (warmup and measured use same input lengths)
            num_cached = 0
            if use_prefix_caching:
                num_cached = int(seq_len * prefix_cached_ratio)
                num_cached = min(num_cached, seq_len - 1)
                num_cached = (num_cached // PREFILL_BENCHMARK_BLOCK_SIZE) * PREFILL_BENCHMARK_BLOCK_SIZE
            start_pos = [num_cached] if use_prefix_caching else None

            # Two batches: 0=warmup, 1=timed (both same input lengths; no KV cache clear needed since we don't check outputs)
            profiler = BenchmarkProfiler()
            for batch_idx in range(2):
                if batch_idx == 0:
                    # Warmup (same as measured: full prefill or prefix-cached prefill)
                    generator.prefill_forward_text(
                        input_tokens_prefill_pt,
                        page_table=page_table,
                        kv_cache=tt_kv_cache,
                        prompt_lens=decoding_pos,
                        enable_trace=True,
                        tt_out_logits_all_users=None,
                        sampling_params=sampling_params,
                        start_pos=start_pos,
                    )
                else:
                    # Timed run
                    profiler.start("prefill")
                    generator.prefill_forward_text(
                        input_tokens_prefill_pt,
                        page_table=page_table,
                        kv_cache=tt_kv_cache,
                        prompt_lens=decoding_pos,
                        enable_trace=True,
                        tt_out_logits_all_users=None,
                        sampling_params=sampling_params,
                        start_pos=start_pos,
                    )
                    profiler.end("prefill")
                    prefill_s = profiler.get_duration("prefill")

                    row = {
                        "seq_len": seq_len,
                        "use_prefix_caching": use_prefix_caching,
                        "prefix_cached_ratio": prefix_cached_ratio,
                        "prefill_s": prefill_s,
                    }
                    results.append(row)
                    logger.info(
                        f"seq_len={seq_len} prefix_cached={use_prefix_caching} ratio={prefix_cached_ratio:.0%} -> {prefill_s:.4f}s"
                    )

    PREFILL_BENCHMARK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PREFILL_BENCHMARK_OUTPUT, "w") as f:
        json.dump({"results": results}, f, indent=2)
    logger.info(f"Results written to {PREFILL_BENCHMARK_OUTPUT}")
    _check_prefill_benchmark_results(results)

    by_len = {}
    for result in results:
        seq_len = result["seq_len"]
        if seq_len not in by_len:
            by_len[seq_len] = {}
        if not result["use_prefix_caching"]:
            label = "no_cache"
        else:
            label = f"{int(result['prefix_cached_ratio'] * 100)}%_cache"
        by_len[seq_len][label] = result["prefill_s"]

    cache_cols = ["50%_cache", "75%_cache", "90%_cache"]
    header = f"{'seq_len':>8}  {'no_cache':>10}"
    for col in cache_cols:
        header += f"  {col:>10}  {'spdup':>5}"
    print("\n=== Prefill time (s) after warmup ===")
    print(header)
    print("-" * len(header))
    for seq_len in sorted(by_len.keys()):
        row = by_len[seq_len]
        no_cache = row.get("no_cache", 0)
        line = f"{seq_len:>8}  {no_cache:>10.4f}"
        for col in cache_cols:
            cached_value = row.get(col, 0)
            speedup = f"{no_cache / cached_value:.2f}x" if cached_value > 0 else "-"
            line += f"  {cached_value:>10.4f}  {speedup:>5}"
        print(line)
