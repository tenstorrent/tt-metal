# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Single-chip SDPA tests for Blackhole covering WAN 2.2 and DeepSeek MLA 100K configurations.

Layout and naming mirror tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py so each "suite"
(WAN, MLA, ...) owns its own chunk-size sweep and dtype/shape parameters. The MLA suite is
sized to replicate the work performed by ring_joint_sdpa at ring iteration 0 — where each
device runs plain causal SDPA on its local Q/K/V slice (``balancing=false``, ``causality=true``,
no KV halving).
"""

import os
import math
import torch
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
import ttnn
from loguru import logger
import pytest

from tests.nightly.sdpa_perf_utils import (
    post_process_ops_log,
    compute_cores_used,
    compute_math_utilization,
    compute_sdpa_flops,
)


# ============================================================================
# HARDWARE CONFIGURATION (single chip, BH)
# ============================================================================

SDPA_GRID = [11, 10]  # 11 cols x 10 rows; matches Galaxy per-chip SDPA grid
NUM_CORES = SDPA_GRID[0] * SDPA_GRID[1]  # 110

BATCH_SIZE = 1


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================


@dataclass
class ModelConfig:
    """Benchmark configuration for a single-chip SDPA suite (WAN, MLA, ...)."""

    name: str
    nhq: int
    nkv: int  # 1 for MLA/MQA; nhq for standard MHA
    seq_len: int
    d_q: int  # Q head dim
    d_k: int  # K head dim (must equal d_q: QK^T matmul constraint). Kept for parity with ring_joint_sdpa.
    d_v: int  # V head dim (may differ from d_q/d_k for MLA)
    is_causal: bool
    q_dtype: ttnn.DataType
    kv_dtype: ttnn.DataType

    # Chunk-size sweep (cross-product is per-model)
    q_chunk_sizes: List[int]
    k_chunk_sizes: List[int]


def generate_model_configs() -> Dict[str, ModelConfig]:
    """Single-chip suites sized to match per-device work on Galaxy/QB."""
    configs = [
        # WAN 2.2 — 1xGalaxy analog (single-chip seq = per-device seq on Galaxy)
        # ModelConfig(
        #     name="wan2_2_1xGLX_analog",
        #     nhq=10,
        #     nkv=10,
        #     seq_len=9472,
        #     d_q=128,
        #     d_k=128,
        #     d_v=128,
        #     is_causal=False,
        #     q_dtype=ttnn.bfloat16,
        #     kv_dtype=ttnn.bfloat16,
        #     q_chunk_sizes=[224, 256, 288],
        #     k_chunk_sizes=[128, 256, 512],
        # ),
        # # WAN 2.2 — 4xGalaxy analog
        # ModelConfig(
        #     name="wan2_2_4xGLX_analog",
        #     nhq=10,
        #     nkv=10,
        #     seq_len=2368,
        #     d_q=128,
        #     d_k=128,
        #     d_v=128,
        #     is_causal=False,
        #     q_dtype=ttnn.bfloat16,
        #     kv_dtype=ttnn.bfloat16,
        #     q_chunk_sizes=[224, 256, 288],
        #     k_chunk_sizes=[128, 256, 512],
        # ),
        # DeepSeek MLA 100K — replicates ring_joint_sdpa ring iter 0 per-device work.
        # At iter 0 for causal+balanced: causality=true, balancing=false, iter_num_kv_chunks
        # is not halved — i.e. plain causal SDPA on local Q/K/V. nhq=32 matches Galaxy per-device
        # head count; grid 11x10 matches Galaxy per-chip SDPA columns.
        ModelConfig(
            name="mla_100k_ring_iter_0",
            nhq=32,
            nkv=1,
            seq_len=3200,
            d_q=576,
            d_k=576,
            d_v=128,
            is_causal=True,
            q_dtype=ttnn.bfloat16,
            kv_dtype=ttnn.bfloat8_b,
            q_chunk_sizes=[160],
            k_chunk_sizes=[160],
        ),
        # DeepSeek MLA 128K — same ring iter 0 semantics as the 100K suite, longer local seq.
        ModelConfig(
            name="mla_128k_ring_iter_0",
            nhq=32,
            nkv=1,
            seq_len=4096,
            d_q=576,
            d_k=576,
            d_v=128,
            is_causal=True,
            q_dtype=ttnn.bfloat16,
            kv_dtype=ttnn.bfloat8_b,
            q_chunk_sizes=[128],
            k_chunk_sizes=[128],
        ),
    ]
    return {c.name: c for c in configs}


MODEL_CONFIGS = generate_model_configs()


def get_test_case_id(config: ModelConfig, q_chunk_size: int, k_chunk_size: int) -> str:
    return f"{config.name}-q{q_chunk_size}-k{k_chunk_size}"


def generate_test_configs(model_configs: Dict[str, ModelConfig]):
    """Build flat (config_tuple, id) list for parametrize across all suites."""
    configs = []
    config_ids = []
    for model in model_configs.values():
        for q_chunk, k_chunk in product(model.q_chunk_sizes, model.k_chunk_sizes):
            assert model.d_k == model.d_q, (
                f"QK^T requires d_q == d_k; got d_q={model.d_q}, d_k={model.d_k} " f"in model {model.name}"
            )
            configs.append(
                (
                    BATCH_SIZE,
                    model.nhq,
                    model.nkv,
                    model.seq_len,
                    model.d_q,
                    model.d_k,
                    model.d_v,
                    q_chunk,
                    k_chunk,
                    model.is_causal,
                    model.q_dtype,
                    model.kv_dtype,
                )
            )
            config_ids.append(get_test_case_id(model, q_chunk, k_chunk))
    return configs, config_ids


TEST_CONFIGS, TEST_CONFIG_IDS = generate_test_configs(MODEL_CONFIGS)
TEST_CONFIG_MODELS = list(MODEL_CONFIGS.keys())


# ============================================================================
# HELPERS
# ============================================================================


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None


def _build_tt_qkv(device, Q, K, V, q_dtype, kv_dtype):
    tt_Q = ttnn.from_torch(Q, dtype=q_dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    return tt_Q, tt_K, tt_V


def _torch_reference(Q, K, V, nhq, nkv, is_causal):
    """Expand K/V to match Q heads (GQA/MQA), then run torch SDPA."""
    if nkv != nhq:
        assert nhq % nkv == 0
        K = K.repeat_interleave(nhq // nkv, dim=1)
        V = V.repeat_interleave(nhq // nkv, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)


# ============================================================================
# CORE RUNNERS
# ============================================================================


def run_sdpa(
    device,
    b,
    nhq,
    nkv,
    sq,
    d_q,
    d_k,
    d_v,
    q_chunk_size,
    k_chunk_size,
    is_causal,
    q_dtype,
    kv_dtype,
    *,
    pcc_threshold=0.9997,
    rmse_threshold=None,
    do_check=True,
):
    """Single-chip SDPA. Handles MHA (d_q == d_v, nkv == nhq) and MLA (d_v < d_q, nkv == 1)."""
    assert d_k == d_q, f"QK^T requires d_q == d_k; got d_q={d_q}, d_k={d_k}"
    torch.manual_seed(1234)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=SDPA_GRID,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
        flatten_work=os.environ.get("TT_METAL_FLATTEN_WORK", "0") not in ("", "0"),
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nhq, sq, d_q)
    K = fa_rand(b, nkv, sq, d_k)
    V = fa_rand(b, nkv, sq, d_v)

    tt_Q, tt_K, tt_V = _build_tt_qkv(device, Q, K, V, q_dtype, kv_dtype)
    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        is_causal=is_causal,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )

    if not do_check:
        return

    tt_back = ttnn.to_torch(tt_back)[:, :, :sq, :d_v]
    gt = _torch_reference(Q, K, V, nhq, nkv, is_causal)

    out_pass, out_pcc = comp_pcc(gt, tt_back, pcc_threshold)
    rmse = torch.sqrt(((gt - tt_back) ** 2).mean()).item()
    logger.debug(f"python vs pytorch: {out_pcc}")
    logger.debug(f"rmse: {rmse}")
    if rmse_threshold is not None:
        assert rmse < rmse_threshold
    assert out_pass


def run_sdpa_determinism(
    device,
    b,
    nhq,
    nkv,
    sq,
    d_q,
    d_k,
    d_v,
    q_chunk_size,
    k_chunk_size,
    is_causal,
    q_dtype,
    kv_dtype,
    num_iterations=10,
):
    """Run SDPA ``num_iterations`` times on the same input and assert bit-exact match."""
    assert d_k == d_q, f"QK^T requires d_q == d_k; got d_q={d_q}, d_k={d_k}"
    torch.manual_seed(1234)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=SDPA_GRID,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
        flatten_work=os.environ.get("TT_METAL_FLATTEN_WORK", "0") not in ("", "0"),
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nhq, sq, d_q)
    K = fa_rand(b, nkv, sq, d_k)
    V = fa_rand(b, nkv, sq, d_v)
    tt_Q, tt_K, tt_V = _build_tt_qkv(device, Q, K, V, q_dtype, kv_dtype)

    reference = None
    for i in range(num_iterations):
        tt_out = ttnn.transformer.scaled_dot_product_attention(
            tt_Q,
            tt_K,
            tt_V,
            is_causal=is_causal,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
        torch_out = ttnn.to_torch(tt_out)[:, :, :sq, :d_v]
        if reference is None:
            reference = torch_out
        elif not torch.equal(reference, torch_out):
            diff_mask = reference != torch_out
            num_diffs = diff_mask.sum().item()
            max_diff = (reference - torch_out).abs().max().item()
            pytest.fail(
                f"SDPA output at iteration {i} differs from iteration 0: "
                f"{num_diffs} differing elements, max diff = {max_diff}"
            )
    logger.info(f"SDPA determinism verified: all {num_iterations} outputs are exactly equal")


# ============================================================================
# TESTS
# ============================================================================

_PARAMS = "b,nhq,nkv,s,d_q,d_k,d_v,q_chunk_size,k_chunk_size,is_causal,q_dtype,kv_dtype"


# === TEST 1: PERFORMANCE SWEEP (skipped on CI) ===
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.parametrize(_PARAMS, TEST_CONFIGS, ids=TEST_CONFIG_IDS)
def test_sdpa_sweep_perf_impl(
    device, b, nhq, nkv, s, d_q, d_k, d_v, q_chunk_size, k_chunk_size, is_causal, q_dtype, kv_dtype
):
    run_sdpa(
        device, b, nhq, nkv, s, d_q, d_k, d_v, q_chunk_size, k_chunk_size, is_causal, q_dtype, kv_dtype, do_check=False
    )


# === TEST 2: ACCURACY VERIFICATION ===
@pytest.mark.parametrize(_PARAMS, TEST_CONFIGS, ids=TEST_CONFIG_IDS)
def test_sdpa_accuracy(device, b, nhq, nkv, s, d_q, d_k, d_v, q_chunk_size, k_chunk_size, is_causal, q_dtype, kv_dtype):
    """PCC + RMSE accuracy check against a torch SDPA reference."""
    # MLA with bfloat8_b K/V needs a looser PCC threshold than bf16 MHA.
    if kv_dtype == ttnn.bfloat8_b:
        pcc_threshold = 0.994
        rmse_threshold = None  # bf8 accumulation noise makes RMSE thresholds brittle
    else:
        pcc_threshold = 0.9997
        rmse_threshold = 4e-2

    run_sdpa(
        device,
        b,
        nhq,
        nkv,
        s,
        d_q,
        d_k,
        d_v,
        q_chunk_size,
        k_chunk_size,
        is_causal,
        q_dtype,
        kv_dtype,
        pcc_threshold=pcc_threshold,
        rmse_threshold=rmse_threshold,
        do_check=True,
    )


# === TEST 3: DETERMINISM VERIFICATION ===
@pytest.mark.parametrize(_PARAMS, TEST_CONFIGS, ids=TEST_CONFIG_IDS)
def test_sdpa_determinism(
    device, b, nhq, nkv, s, d_q, d_k, d_v, q_chunk_size, k_chunk_size, is_causal, q_dtype, kv_dtype
):
    run_sdpa_determinism(
        device,
        b,
        nhq,
        nkv,
        s,
        d_q,
        d_k,
        d_v,
        q_chunk_size,
        k_chunk_size,
        is_causal,
        q_dtype,
        kv_dtype,
        num_iterations=10,
    )


# === TEST 4: PERFORMANCE TABLE (skipped on CI) ===
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.parametrize("model_name", TEST_CONFIG_MODELS)
def test_sdpa_create_perf_table(model_name):
    """
    Sweep a model's chunk-size configurations and print a ranked performance table.
    Skipped on CI — run locally with the tracy profiler.
    """
    from tracy.process_model_log import run_device_profiler

    model = MODEL_CONFIGS[model_name]

    # Collect the parametrize ids that belong to this model.
    sweep = [(cfg, cfg_id) for cfg, cfg_id in zip(TEST_CONFIGS, TEST_CONFIG_IDS) if cfg_id.startswith(model_name + "-")]

    subdir = "ttnn_sdpa_performance"
    perf_results = []

    for config, config_id in sweep:
        (b, nhq, nkv, s, d_q, d_k, d_v, q_chunk_size, k_chunk_size, is_causal, q_dtype, kv_dtype) = config
        float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]
        cols = ["ATTRIBUTES"]
        command = (
            f"pytest tests/nightly/blackhole/sdpa/"
            f"test_scaled_dot_product_attention_sprint.py::test_sdpa_sweep_perf_impl"
            f"[{config_id}]"
        )

        try:
            run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
            r = post_process_ops_log(
                subdir, float_columns=float_cols, columns=cols, op_name="", sum_vals=False, has_signposts=False
            )
            core_count = int(r["CORE COUNT"][0])
            duration_ns = int(r["DEVICE KERNEL DURATION [ns]"].min())

            batch_parallel = min(b, NUM_CORES)
            nh_parallel = min(NUM_CORES // batch_parallel, nhq)
            max_q_parallel = NUM_CORES // (batch_parallel * nh_parallel)

            cores_used = compute_cores_used(s, q_chunk_size, compute_cores=NUM_CORES, num_heads=nhq)
            cores_idle = NUM_CORES - cores_used
            core_util_pct = (cores_used * 100.0) / NUM_CORES

            k_num_chunks = math.ceil(s / k_chunk_size)
            q_num_chunks = math.ceil(s / q_chunk_size)
            q_per_core = math.ceil(q_num_chunks / max_q_parallel)
            iters_per_core = q_per_core * k_num_chunks

            q_padded_total = q_num_chunks * q_chunk_size
            k_padded_total = k_num_chunks * k_chunk_size
            actual_work = s * s
            padded_work = q_padded_total * k_padded_total
            total_waste_pct = ((padded_work - actual_work) / padded_work) * 100 if padded_work > 0 else 0

            total_q_slots = max_q_parallel * q_per_core
            wasted_q_slots = max(0, total_q_slots - q_num_chunks)
            slot_waste_pct = (wasted_q_slots / total_q_slots) * 100 if total_q_slots > 0 else 0

            # Math util uses distinct d_q/d_v and respects causal masking.
            utilization = compute_math_utilization(s, s, d_q, d_v, nhq, duration_ns, core_count, is_causal=is_causal)

            perf_results.append(
                {
                    "q_chunk_size": q_chunk_size,
                    "k_chunk_size": k_chunk_size,
                    "core_count": core_count,
                    "cores_used": cores_used,
                    "cores_idle": cores_idle,
                    "core_util_pct": core_util_pct,
                    "iters_per_core": iters_per_core,
                    "total_waste_pct": total_waste_pct,
                    "slot_waste_pct": slot_waste_pct,
                    "duration_ns": duration_ns,
                    "duration_ms": duration_ns / 1e6,
                    "utilization": utilization,
                }
            )
            logger.info(
                f"[{model_name}] q={q_chunk_size}, k={k_chunk_size}: {duration_ns/1e6:.3f} ms, "
                f"util={utilization:.1f}%, cores={cores_used}/{NUM_CORES} ({core_util_pct:.0f}%), "
                f"iters/core={iters_per_core}"
            )

        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            logger.error(
                f"Error running [{model_name}] with q_chunk_size={q_chunk_size}, k_chunk_size={k_chunk_size}: {e}"
            )
            perf_results.append({"q_chunk_size": q_chunk_size, "k_chunk_size": k_chunk_size, "duration_ns": None})

    valid_results = [r for r in perf_results if r["duration_ns"] is not None]
    valid_results.sort(key=lambda x: x["duration_ns"])

    # Use the MLA config's first chunk sizes for the model summary — b/nhq/s/d_q/d_v are constant per model.
    mm_flops = compute_sdpa_flops(model.seq_len, model.seq_len, model.d_q, model.d_v, model.nhq, model.is_causal)

    print(f"\n{'='*170}")
    print(
        f"SDPA Performance Sweep ({model_name.upper()}): "
        f"b={BATCH_SIZE}, nhq={model.nhq}, nkv={model.nkv}, s={model.seq_len}, "
        f"d_q={model.d_q}, d_v={model.d_v}, causal={model.is_causal}"
    )
    print(f"MM FLOPs: {mm_flops:,} ({mm_flops/1e9:.2f} GFLOPs)")
    print(f"{'='*170}")
    header = "| Rank | q_chunk | k_chunk | Duration (ms) | Cores Used | Cores Idle | Core Util | Iters/Core | Pad Waste | Slot Waste | Math Util |"
    sep = "|------|---------|---------|---------------|------------|------------|-----------|------------|-----------|------------|-----------|"
    print(header)
    print(sep)

    for rank, result in enumerate(valid_results, 1):
        print(
            f"| {rank:4d} | {result['q_chunk_size']:7d} | {result['k_chunk_size']:7d} | {result['duration_ms']:13.3f} | "
            f"{result['cores_used']:10d} | {result['cores_idle']:10d} | {result['core_util_pct']:8.0f}% | "
            f"{result['iters_per_core']:10d} | {result['total_waste_pct']:8.1f}% | {result['slot_waste_pct']:9.1f}% | {result['utilization']:8.1f}% |"
        )

    failed_results = [r for r in perf_results if r["duration_ns"] is None]
    if failed_results:
        print(f"\nFailed configurations:")
        for result in failed_results:
            print(f"  q_chunk_size={result['q_chunk_size']}, k_chunk_size={result['k_chunk_size']}")

    if valid_results:
        best = valid_results[0]
        print(
            f"\nBest configuration: q_chunk_size={best['q_chunk_size']}, k_chunk_size={best['k_chunk_size']} "
            f"({best['duration_ms']:.3f} ms, {best['utilization']:.1f}% math util, "
            f"{best['cores_used']}/{NUM_CORES} cores, {best['iters_per_core']} iters/core, "
            f"{best['total_waste_pct']:.1f}% pad waste, {best['slot_waste_pct']:.1f}% slot waste)"
        )
    print(f"{'='*170}\n")
