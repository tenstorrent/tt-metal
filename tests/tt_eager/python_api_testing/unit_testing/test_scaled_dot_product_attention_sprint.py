# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
from itertools import product
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import ttnn
from loguru import logger
import pytest
from models.common.utility_functions import skip_for_wormhole_b0, skip_for_blackhole

from tracy.process_model_log import (
    get_latest_ops_log_filename,
    run_device_profiler,
)


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None


def run_sdpa_noncausal(
    device,
    b,
    nh,
    nkv,
    sq,
    d,
    q_chunk_size,
    k_chunk_size,
    dtype,
    sk=None,
    pcc_threshold=0.9998,
    rmse_threshold=None,
    do_check=True,
):
    # Ensure same seed, reproducibility
    torch.manual_seed(1234)
    if sk is None:
        sk = sq

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=[11, 10],
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, sq, d)
    K = fa_rand(b, nkv, sk, d)
    V = fa_rand(b, nkv, sk, d)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        is_causal=False,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )

    if not do_check:
        return

    tt_back = ttnn.to_torch(tt_back)
    # Slice out any tile-padding
    tt_back = tt_back[:, :, :sq, :]

    if nkv > 1 and nkv != nh:
        assert nh % nkv == 0
        K = K.reshape(b, nkv, 1, sk, d).repeat(1, 1, nh // nkv, 1, 1).reshape(b, nh, sk, d)
        V = V.reshape(b, nkv, 1, sk, d).repeat(1, 1, nh // nkv, 1, 1).reshape(b, nh, sk, d)

    gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)

    out_pass, out_pcc = comp_pcc(gt, tt_back, pcc_threshold)
    logger.debug(f"python vs pytorch: {out_pcc}")
    rmse = torch.sqrt(((gt - tt_back) ** 2).mean()).item()
    logger.debug(f"rmse: {rmse}")
    if rmse_threshold is not None:
        assert rmse < rmse_threshold

    assert out_pass


def run_sdpa_determinism(
    device,
    b,
    nh,
    nkv,
    sq,
    d,
    q_chunk_size,
    k_chunk_size,
    dtype,
    num_iterations=10,
    sk=None,
):
    """
    Run SDPA multiple times with the same inputs and return all outputs.
    Efficient: creates inputs once and reuses them for all iterations.
    """
    torch.manual_seed(1234)
    if sk is None:
        sk = sq

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=[11, 10],
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Create inputs once
    Q = fa_rand(b, nh, sq, d)
    K = fa_rand(b, nkv, sk, d)
    V = fa_rand(b, nkv, sk, d)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)

    # Run SDPA multiple times and collect outputs
    outputs = []
    for i in range(num_iterations):
        tt_out = ttnn.transformer.scaled_dot_product_attention(
            tt_Q,
            tt_K,
            tt_V,
            is_causal=False,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
        # Convert to torch and slice out padding
        torch_out = ttnn.to_torch(tt_out)[:, :, :sq, :]
        outputs.append(torch_out)

    return outputs


INPUT_SHAPES = [
    # batch, num_heads, sequence_length, head_dim
    [1, 10, 9472, 128],
    [1, 10, 2368, 128],
]
INPUT_IDS = [
    "wan_1xGLX_analog",
    "wan_4xGLX_analog",
]

Q_CHUNK_SIZES = [224, 256, 288]
K_CHUNK_SIZES = [128, 256, 512]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", Q_CHUNK_SIZES, ids=[f"q{s}" for s in Q_CHUNK_SIZES])
@pytest.mark.parametrize("k_chunk_size", K_CHUNK_SIZES, ids=[f"k{s}" for s in K_CHUNK_SIZES])
@pytest.mark.parametrize(
    "b, nh, s, d",
    INPUT_SHAPES,
    ids=INPUT_IDS,
)
def test_sdpa_sweep_perf_impl(device, b, nh, s, d, q_chunk_size, k_chunk_size, dtype):
    # nkv = nh for non-GQA case
    run_sdpa_noncausal(device, b, nh, nh, s, d, q_chunk_size, k_chunk_size, dtype, do_check=False)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", Q_CHUNK_SIZES, ids=[f"q{s}" for s in Q_CHUNK_SIZES])
@pytest.mark.parametrize("k_chunk_size", K_CHUNK_SIZES, ids=[f"k{s}" for s in K_CHUNK_SIZES])
@pytest.mark.parametrize(
    "b, nh, s, d",
    INPUT_SHAPES,
    ids=INPUT_IDS,
)
def test_sdpa_accuracy(device, b, nh, s, d, q_chunk_size, k_chunk_size, dtype):
    """
    Test SDPA accuracy for the given shapes and chunk size configurations.
    Verifies PCC > 0.994 against PyTorch reference.
    """
    # nkv = nh for non-GQA case
    pcc_threshold = 0.9997
    rmse_threshold = 4e-2
    run_sdpa_noncausal(
        device,
        b,
        nh,
        nh,
        s,
        d,
        q_chunk_size,
        k_chunk_size,
        dtype,
        pcc_threshold=pcc_threshold,
        rmse_threshold=rmse_threshold,
        do_check=True,
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", Q_CHUNK_SIZES, ids=[f"q{s}" for s in Q_CHUNK_SIZES])
@pytest.mark.parametrize("k_chunk_size", K_CHUNK_SIZES, ids=[f"k{s}" for s in K_CHUNK_SIZES])
@pytest.mark.parametrize(
    "b, nh, s, d",
    INPUT_SHAPES,
    ids=INPUT_IDS,
)
def test_sdpa_determinism(device, b, nh, s, d, q_chunk_size, k_chunk_size, dtype):
    """
    Test SDPA determinism: run 10 times with same inputs and verify outputs match exactly.
    """
    num_iterations = 10
    outputs = run_sdpa_determinism(
        device, b, nh, nh, s, d, q_chunk_size, k_chunk_size, dtype, num_iterations=num_iterations
    )

    # Compare all outputs to the first one - they should be exactly equal
    reference = outputs[0]
    for i in range(1, num_iterations):
        if not torch.equal(reference, outputs[i]):
            # Find where they differ for debugging
            diff_mask = reference != outputs[i]
            num_diffs = diff_mask.sum().item()
            max_diff = (reference - outputs[i]).abs().max().item()
            logger.error(
                f"Iteration {i} differs from iteration 0: " f"{num_diffs} differing elements, max diff = {max_diff}"
            )
            assert False, f"SDPA output at iteration {i} differs from iteration 0"

    logger.info(f"SDPA determinism verified: all {num_iterations} outputs are exactly equal")


def post_process_ops_log(
    output_logs_subdir, float_columns=None, columns=None, sum_vals=True, op_name="", has_signposts=False
):
    """Process the ops log CSV and extract performance data."""
    filename = get_latest_ops_log_filename(output_logs_subdir)
    import pandas as pd

    df = pd.read_csv(filename)

    if has_signposts:
        # there are explicit start and stop points in the model we want to measure between
        markers = df[df["OP TYPE"] == "signpost"]["OP CODE"]
        start = markers[markers == "start"].index[0]
        stop = markers[markers == "stop"].index[0]
        df = df.iloc[start + 1 : stop]
    if op_name != "":
        df = df[df["OP CODE"] == op_name]

    results = {}
    if float_columns:
        assert (
            type(float_columns) == list
        ), f"Bad columns name type, requested columns should be of type list but {type(float_columns)} was provided"
        for col in float_columns:
            df_filtered = df[df[col] != "-"]
            if sum_vals:
                results[col] = df_filtered[col].astype(float).sum()
            else:
                results[col] = df_filtered[col].astype(float).to_numpy()
    if columns:
        assert (
            type(columns) == list
        ), f"Bad columns name type, requested columns should be of type list but {type(columns)} was provided"
        for col in columns:
            df_filtered = df[df[col] != "-"]
            results[col] = df_filtered[col]
    else:
        results = df
    return results


def compute_cores_used(seqlen, q_chunk_size, num_cores, num_heads):
    """
    Compute number of cores actually used based on parallelization scheme.

    Parallelization hierarchy (from sdpa_program_factory.cpp):
    1. batch_parallel_factor = min(B, num_cores)
    2. nh_parallel_factor = min(num_cores / batch_parallel_factor, NQH)
    3. q_parallel_factor = min(num_cores / (batch * nh), q_num_chunks)
    """
    import math

    B = 1  # batch size
    q_num_chunks = math.ceil(seqlen / q_chunk_size)

    batch_parallel = min(B, num_cores)
    nh_parallel = min(num_cores // batch_parallel, num_heads)
    q_parallel = min(num_cores // (batch_parallel * nh_parallel), q_num_chunks)

    cores_used = batch_parallel * nh_parallel * q_parallel
    return cores_used


def compute_sdpa_utilization(seqlen, head_dim, num_heads, duration_ns, core_count):
    """
    Compute math utilization for SDPA.

    Args:
        seqlen: Sequence length
        head_dim: Head dimension
        num_heads: Number of attention heads
        duration_ns: Measured kernel duration in nanoseconds
        core_count: Number of cores used

    Returns:
        Utilization as a percentage (0-100)
    """
    # MM FLOPs for SDPA: 4 * seqlen^2 * head_dim * num_heads
    mm_flops = 4 * seqlen * seqlen * head_dim * num_heads

    # Convert nanoseconds to cycles (clock is 1.35 GHz = 1.35 cycles per ns)
    cycles = duration_ns * 1.35

    # Each core can perform 2048 MM flops per cycle
    theoretical_flops = core_count * cycles * 2048

    # Utilization percentage
    utilization = (mm_flops / theoretical_flops) * 100

    return utilization


# @pytest.mark.skip(reason="Manual performance sweep - run explicitly when needed")
@pytest.mark.parametrize(
    "b, nh, s, d",
    INPUT_SHAPES,
    ids=INPUT_IDS,
)
def test_sdpa_create_perf_table(b, nh, s, d):
    """
    Sweep chunk sizes for a given SDPA shape and print a performance table.
    Shows the best chunk size configurations ranked by kernel duration.
    """
    # NOTE: Hardcoded for Blackhole (11x10 grid = 110 cores)
    # Cannot query device here as it causes TLB resource contention with subprocess tests
    num_cores = 110

    subdir = "ttnn_sdpa_performance"
    perf_results = []

    for q_chunk_size, k_chunk_size in product(Q_CHUNK_SIZES, K_CHUNK_SIZES):
        float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]
        cols = ["ATTRIBUTES"]

        # Build the test command for this specific configuration
        test_id = f"k{k_chunk_size}-q{q_chunk_size}-bf16"
        shape_id = INPUT_IDS[INPUT_SHAPES.index([b, nh, s, d])]
        command = (
            f"pytest tests/tt_eager/python_api_testing/unit_testing/"
            f"test_scaled_dot_product_attention_sprint.py::test_sdpa_sweep_perf_impl"
            f"[{shape_id}-{test_id}]"
        )

        try:
            run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
            r = post_process_ops_log(
                subdir, float_columns=float_cols, columns=cols, op_name="", sum_vals=False, has_signposts=False
            )

            core_count = int(r["CORE COUNT"][0])
            duration_ns = int(r["DEVICE KERNEL DURATION [ns]"].min())

            # Compute parallelization factors
            B = 1  # batch size
            batch_parallel = min(B, num_cores)
            nh_parallel = min(num_cores // batch_parallel, nh)
            max_q_parallel = num_cores // (batch_parallel * nh_parallel)

            # Compute cores actually used based on parallelization scheme
            cores_used = compute_cores_used(s, q_chunk_size, num_cores=num_cores, num_heads=nh)
            cores_idle = num_cores - cores_used
            core_util_pct = (cores_used * 100.0) / num_cores

            # Compute iterations per core
            k_num_chunks = math.ceil(s / k_chunk_size)
            q_num_chunks = math.ceil(s / q_chunk_size)
            q_per_core = math.ceil(q_num_chunks / max_q_parallel)
            iters_per_core = q_per_core * k_num_chunks

            # Compute padding waste
            q_padded_total = q_num_chunks * q_chunk_size
            k_padded_total = k_num_chunks * k_chunk_size
            q_waste = q_padded_total - s
            k_waste = k_padded_total - s
            actual_work = s * s
            padded_work = q_padded_total * k_padded_total
            total_waste_pct = ((padded_work - actual_work) / padded_work) * 100 if padded_work > 0 else 0

            # Compute work distribution (slot) waste
            # Use the maximum available cores for q parallelization, not just what we need
            total_q_slots = max_q_parallel * q_per_core
            wasted_q_slots = total_q_slots - q_num_chunks
            slot_waste_pct = (wasted_q_slots / total_q_slots) * 100 if total_q_slots > 0 else 0

            # Compute math utilization
            utilization = compute_sdpa_utilization(s, d, nh, duration_ns, core_count)

            perf_results.append(
                {
                    "q_chunk_size": q_chunk_size,
                    "k_chunk_size": k_chunk_size,
                    "core_count": core_count,
                    "cores_used": cores_used,
                    "cores_idle": cores_idle,
                    "core_util_pct": core_util_pct,
                    "iters_per_core": iters_per_core,
                    "q_waste": q_waste,
                    "k_waste": k_waste,
                    "total_waste_pct": total_waste_pct,
                    "slot_waste_pct": slot_waste_pct,
                    "duration_ns": duration_ns,
                    "duration_ms": duration_ns / 1e6,
                    "utilization": utilization,
                }
            )
            logger.info(
                f"q={q_chunk_size}, k={k_chunk_size}: {duration_ns/1e6:.3f} ms, "
                f"util={utilization:.1f}%, cores={cores_used}/{num_cores} ({core_util_pct:.0f}%), "
                f"iters/core={iters_per_core}"
            )

        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            logger.error(f"Error running SDPA with q_chunk_size={q_chunk_size}, k_chunk_size={k_chunk_size}: {e}")
            perf_results.append(
                {
                    "q_chunk_size": q_chunk_size,
                    "k_chunk_size": k_chunk_size,
                    "core_count": None,
                    "cores_used": None,
                    "cores_idle": None,
                    "core_util_pct": None,
                    "iters_per_core": None,
                    "q_waste": None,
                    "k_waste": None,
                    "total_waste_pct": None,
                    "slot_waste_pct": None,
                    "duration_ns": None,
                    "duration_ms": None,
                    "utilization": None,
                }
            )

    # Sort by duration (best first)
    valid_results = [r for r in perf_results if r["duration_ns"] is not None]
    valid_results.sort(key=lambda x: x["duration_ns"])

    # Compute total MM FLOPs for reference
    mm_flops = 4 * s * s * d * nh

    # Print summary table
    print(f"\n{'='*170}")
    print(f"SDPA Performance Sweep: b={b}, nh={nh}, s={s}, d={d}")
    print(f"MM FLOPs: {mm_flops:,} ({mm_flops/1e9:.2f} GFLOPs)")
    print(f"{'='*170}")
    header = "| Rank | q_chunk | k_chunk | Duration (ms) | Cores Used | Cores Idle | Core Util | Iters/Core | Pad Waste | Slot Waste | Math Util |"
    sep = "|------|---------|---------|---------------|------------|------------|-----------|------------|-----------|------------|-----------|"
    print(header)
    print(sep)

    for rank, result in enumerate(valid_results, 1):
        q = result["q_chunk_size"]
        k = result["k_chunk_size"]
        dur_ms = result["duration_ms"]
        cores_used = result["cores_used"]
        cores_idle = result["cores_idle"]
        core_util = result["core_util_pct"]
        iters = result["iters_per_core"]
        pad_waste = result["total_waste_pct"]
        slot_waste = result["slot_waste_pct"]
        math_util = result["utilization"]
        print(
            f"| {rank:4d} | {q:7d} | {k:7d} | {dur_ms:13.3f} | "
            f"{cores_used:10d} | {cores_idle:10d} | {core_util:8.0f}% | "
            f"{iters:10d} | {pad_waste:8.1f}% | {slot_waste:9.1f}% | {math_util:8.1f}% |"
        )

    # Also show failed configs if any
    failed_results = [r for r in perf_results if r["duration_ns"] is None]
    if failed_results:
        print(f"\nFailed configurations:")
        for result in failed_results:
            print(f"  q_chunk_size={result['q_chunk_size']}, k_chunk_size={result['k_chunk_size']}")

    best = valid_results[0]
    print(
        f"\nBest configuration: q_chunk_size={best['q_chunk_size']}, "
        f"k_chunk_size={best['k_chunk_size']} "
        f"({best['duration_ms']:.3f} ms, {best['utilization']:.1f}% math util, "
        f"{best['cores_used']}/{num_cores} cores, {best['iters_per_core']} iters/core, "
        f"{best['total_waste_pct']:.1f}% pad waste, {best['slot_waste_pct']:.1f}% slot waste)"
    )
    print(f"{'='*170}\n")
