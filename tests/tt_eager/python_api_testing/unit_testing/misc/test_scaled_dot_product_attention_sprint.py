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


def create_sdpa_configs(device_or_mesh, q_chunk_size, k_chunk_size, is_mesh_device=False):
    """Create program and compute kernel configs for SDPA operations"""
    if is_mesh_device:
        # For mesh device, use reduced grid size for CCL
        compute_grid_size = device_or_mesh.compute_with_storage_grid_size()
        sdpa_compute_grid = (compute_grid_size.x, compute_grid_size.y - 1)
    else:
        # For single device, use full grid
        sdpa_compute_grid = device_or_mesh.compute_with_storage_grid_size()

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_compute_grid,
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

    return program_config, compute_kernel_config


def generate_sdpa_tensors(b, nh, nkv, sq, d, dtype):
    """Generate Q, K, V tensors for SDPA testing"""
    Q = fa_rand(b, nh, sq, d)
    K = fa_rand(b, nkv, sq, d)  # nkv instead of nh for GQA support
    V = fa_rand(b, nkv, sq, d)
    return Q, K, V


def expand_kv_for_gqa(K, V, nh, nkv, b, sq, d):
    """Expand K, V tensors for Group Query Attention if needed"""
    if nkv > 1 and nkv != nh:
        assert nh % nkv == 0
        K_expanded = K.reshape(b, nkv, 1, sq, d).repeat(1, 1, nh // nkv, 1, 1).reshape(b, nh, sq, d)
        V_expanded = V.reshape(b, nkv, 1, sq, d).repeat(1, 1, nh // nkv, 1, 1).reshape(b, nh, sq, d)
        return K_expanded, V_expanded
    return K, V


def validate_sdpa_output(gt, tt_back, pcc_threshold, rmse_threshold=None):
    """Validate SDPA output against ground truth"""
    out_pass, out_pcc = comp_pcc(gt, tt_back, pcc_threshold)
    logger.debug(f"SDPA vs PyTorch: {out_pcc}")

    rmse = torch.sqrt(((gt - tt_back) ** 2).mean()).item()
    logger.debug(f"rmse: {rmse}")

    if rmse_threshold is not None:
        assert rmse < rmse_threshold, f"RMSE {rmse} >= {rmse_threshold}"

    assert out_pass, f"PCC {out_pcc} < {pcc_threshold}"


def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None


def run_ring_attention_noncausal(
    mesh_device,
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
    """Run Ring Attention on MeshDevice (1D Ring)"""
    # Common setup
    torch.manual_seed(1234)
    if sk is None:
        sk = sq

    # Get number of devices for 1D Ring
    num_devices = mesh_device.get_num_devices()
    ring_size = num_devices

    # Multiply sequence length by number of devices for Ring Attention
    total_seq_len = sq * num_devices
    logger.info(f"Ring Attention: {num_devices} devices, base seq_len={sq}, total seq_len={total_seq_len}")

    # Setup 1D Ring submesh
    current_shape = (mesh_device.shape[0], mesh_device.shape[1])  # (height, width)
    if current_shape == (ring_size, 1):
        submesh = mesh_device.create_submesh(ttnn.MeshShape(ring_size, 1))
        cluster_axis = 0  # Ring along height
    elif current_shape == (1, ring_size):
        submesh = mesh_device.create_submesh(ttnn.MeshShape(1, ring_size))
        cluster_axis = 1  # Ring along width
    else:
        mesh_device.reshape(ttnn.MeshShape(1, ring_size))
        submesh = mesh_device.create_submesh(ttnn.MeshShape(1, ring_size))
        cluster_axis = 1  # Ring along width

    # Setup sub-device management for CCL operations
    compute_grid_size = submesh.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    sub_device_manager = submesh.create_sub_device_manager([worker_sub_device], 0)
    submesh.load_sub_device_manager(sub_device_manager)
    submesh.set_sub_device_stall_group(sub_device_stall_group)

    # Create global semaphore handles for CCL
    ccl_semaphore_handles = [ttnn.create_global_semaphore(submesh, ccl_sub_device_crs, 0) for _ in range(2)]
    ccl_core_grid_offset = (0, compute_grid_size.y - 1)

    # Create shared configs
    program_config, compute_kernel_config = create_sdpa_configs(
        submesh, q_chunk_size, k_chunk_size, is_mesh_device=True
    )

    # Generate input tensors with total sequence length
    Q, K, V = generate_sdpa_tensors(b, nh, nkv, total_seq_len, d, dtype)

    # Create persistent output buffers for Ring Attention AllGather
    persistent_output_buffers = [
        ttnn.from_torch(
            torch.zeros((b, nh, total_seq_len, d)),
            device=submesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=[None, None]),
        )
        for _ in range(2)  # K, V buffers
    ]

    # Shard tensors across ring devices (shard on sequence dimension)
    shard_dims = [None, None]
    shard_dims[cluster_axis] = 2  # Shard sequence dimension along cluster axis

    tt_Q = ttnn.from_torch(
        Q,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        pad_value=0.0,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=shard_dims),
    )
    tt_K = ttnn.from_torch(
        K,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        pad_value=0.0,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=shard_dims),
    )
    tt_V = ttnn.from_torch(
        V,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        pad_value=0.0,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=shard_dims),
    )

    # Use Ring Joint SDPA but with empty joint tensors (just regular Ring Attention)
    empty_joint_shape = (b, nh, 0, d)  # Empty joint tensors
    tt_joint_Q = ttnn.from_torch(torch.zeros(empty_joint_shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=submesh)
    tt_joint_K = ttnn.from_torch(torch.zeros(empty_joint_shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=submesh)
    tt_joint_V = ttnn.from_torch(torch.zeros(empty_joint_shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=submesh)

    # Run Ring Attention
    tt_out, tt_joint_out, tt_lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        tt_joint_Q,
        tt_joint_K,
        tt_joint_V,
        persistent_output_buffer_k=persistent_output_buffers[0],
        persistent_output_buffer_v=persistent_output_buffers[1],
        joint_strategy="rear",
        logical_n=total_seq_len,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=ccl_semaphore_handles,
        num_links=1,
        cluster_axis=cluster_axis,
        mesh_device=submesh,
        topology=ttnn.Topology.Linear,
        subdevice_id=worker_sub_device_id,
        ccl_core_grid_offset=ccl_core_grid_offset,
    )

    if not do_check:
        # Clean up and return
        submesh.reset_sub_device_stall_group()
        submesh.clear_loaded_sub_device_manager()
        return

    # Convert back to torch for validation
    output_dims = [None, None]
    output_dims[cluster_axis] = 2  # Concatenate along sequence dimension

    tt_back = ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(submesh, mesh_shape=tuple(submesh.shape), dims=output_dims)
    )

    # Slice out any tile-padding
    tt_back = tt_back[:, :, :total_seq_len, :]

    # Expand K, V for GQA and validate
    K_expanded, V_expanded = expand_kv_for_gqa(K, V, nh, nkv, b, total_seq_len, d)
    gt = torch.nn.functional.scaled_dot_product_attention(Q, K_expanded, V_expanded, is_causal=False)
    validate_sdpa_output(gt, tt_back, pcc_threshold, rmse_threshold)

    # Clean up sub-device management
    submesh.reset_sub_device_stall_group()
    submesh.clear_loaded_sub_device_manager()


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
    """Run standard SDPA on single device"""
    # Common setup
    torch.manual_seed(1234)
    if sk is None:
        sk = sq

    # Create shared configs
    program_config, compute_kernel_config = create_sdpa_configs(
        device, q_chunk_size, k_chunk_size, is_mesh_device=False
    )

    # Generate input tensors
    Q, K, V = generate_sdpa_tensors(b, nh, nkv, sq, d, dtype)

    # Convert to TT tensors
    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)

    # Run standard SDPA
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

    # Convert back to torch and slice out padding
    tt_back = ttnn.to_torch(tt_back)
    tt_back = tt_back[:, :, :sq, :]

    # Expand K, V for GQA and validate
    K_expanded, V_expanded = expand_kv_for_gqa(K, V, nh, nkv, b, sq, d)
    gt = torch.nn.functional.scaled_dot_product_attention(Q, K_expanded, V_expanded, is_causal=False)
    validate_sdpa_output(gt, tt_back, pcc_threshold, rmse_threshold)


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
    # Common setup
    torch.manual_seed(1234)
    if sk is None:
        sk = sq

    # Create shared configs
    program_config, compute_kernel_config = create_sdpa_configs(
        device, q_chunk_size, k_chunk_size, is_mesh_device=False
    )

    # Generate input tensors once
    Q, K, V = generate_sdpa_tensors(b, nh, nkv, sq, d, dtype)

    # Convert to TT tensors
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

# Ring Attention shapes - base seq_len (will be multiplied by num_devices in the test)
RING_INPUT_SHAPES = [
    # batch, num_heads, base_sequence_length (will be multiplied by device count), head_dim
    [1, 10, 9472, 128],  # Total will be 9472 * num_devices
    [1, 10, 2368, 128],  # Total will be 2368 * num_devices
]
RING_INPUT_IDS = [
    "ring_wan_1xGLX_analog",
    "ring_wan_4xGLX_analog",
]

Q_CHUNK_SIZES = [64, 128, 256, 512]
K_CHUNK_SIZES = [128, 256, 512]


@pytest.mark.parametrize("mesh_device", [(1, 4), (1, 8)], indirect=True)  # 1D Ring with 4 or 8 devices
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", Q_CHUNK_SIZES, ids=[f"q{s}" for s in Q_CHUNK_SIZES])
@pytest.mark.parametrize("k_chunk_size", K_CHUNK_SIZES, ids=[f"k{s}" for s in K_CHUNK_SIZES])
@pytest.mark.parametrize(
    "b, nh, s, d",
    RING_INPUT_SHAPES,
    ids=RING_INPUT_IDS,
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}], indirect=True
)
def test_ring_attention_sweep_perf_impl(mesh_device, b, nh, s, d, q_chunk_size, k_chunk_size, dtype):
    """Ring Attention performance test on MeshDevice (1D Ring)

    Note: The sequence length 's' represents the base sequence length.
    The total distributed sequence length will be s * num_devices across all devices.

    Works with both 4 and 8 device configurations (BH QB: 4 devices, other systems: 8 devices).
    """
    # nkv = nh for non-GQA case
    run_ring_attention_noncausal(mesh_device, b, nh, nh, s, d, q_chunk_size, k_chunk_size, dtype, do_check=False)


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
    subdir = "ttnn_sdpa_performance"
    perf_results = []

    for q_chunk_size, k_chunk_size in product(Q_CHUNK_SIZES, K_CHUNK_SIZES):
        float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]
        cols = ["ATTRIBUTES"]

        # Build the test command for this specific configuration
        test_id = f"k{k_chunk_size}-q{q_chunk_size}-bf16"
        shape_id = INPUT_IDS[INPUT_SHAPES.index([b, nh, s, d])]
        command = (
            f"pytest tests/tt_eager/python_api_testing/unit_testing/misc/"
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

            # Compute utilization
            utilization = compute_sdpa_utilization(s, d, nh, duration_ns, core_count)

            perf_results.append(
                {
                    "q_chunk_size": q_chunk_size,
                    "k_chunk_size": k_chunk_size,
                    "core_count": core_count,
                    "duration_ns": duration_ns,
                    "duration_ms": duration_ns / 1e6,
                    "utilization": utilization,
                }
            )
            logger.info(f"q={q_chunk_size}, k={k_chunk_size}: {duration_ns/1e6:.3f} ms, util={utilization:.1f}%")

        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            logger.error(f"Error running SDPA with q_chunk_size={q_chunk_size}, k_chunk_size={k_chunk_size}: {e}")
            perf_results.append(
                {
                    "q_chunk_size": q_chunk_size,
                    "k_chunk_size": k_chunk_size,
                    "core_count": None,
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
    print(f"\n{'='*85}")
    print(f"SDPA Performance Sweep: b={b}, nh={nh}, s={s}, d={d}")
    print(f"MM FLOPs: {mm_flops:,} ({mm_flops/1e9:.2f} GFLOPs)")
    print(f"{'='*85}")
    header = "| Rank | q_chunk | k_chunk | Duration (ms) | Core Count | Util (%) |"
    sep = "|------|---------|---------|---------------|------------|----------|"
    print(header)
    print(sep)

    for rank, result in enumerate(valid_results, 1):
        q = result["q_chunk_size"]
        k = result["k_chunk_size"]
        dur_ms = result["duration_ms"]
        cores = result["core_count"]
        util = result["utilization"]
        print(f"| {rank:4d} | {q:7d} | {k:7d} | {dur_ms:13.3f} | {cores:10d} | {util:8.1f} |")

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
        f"({best['duration_ms']:.3f} ms, {best['utilization']:.1f}% util)"
    )
    print(f"{'='*85}\n")
