# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import pytest
import ttnn
from loguru import logger
from models.common.utility_functions import is_blackhole
from tests.nightly.t3000.ccl.test_neighbor_pad_async import (
    run_neighbor_pad_1d_impl,
    run_neighbor_pad_2d_impl,
    compute_2d_pad_golden,
    pad_chunks_along_dim,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from ttnn import ShardTensor2dMesh, ConcatMesh2dToTensor


@pytest.mark.timeout(200)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 3),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "input_shape, halo_shard_dim, other_shard_dim, layout, input_dtype, padding_left, padding_right, padding_mode, cluster_axis, num_links, skip_for_ci_env, use_persistent_output_buffer",
    [
        ([1, 6, 184 * 4, 160 * 8, 96], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 4, True, False),
        ([6, 186 * 4, 160 * 8, 96], 2, 1, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 1, 4, True, False),
        ([28, 5, 106, 32], 0, 2, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 0, "replicate", 1, 1, False, False),
        ([28, 5, 106, 32], 2, 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 2, "replicate", 1, 1, False, False),
        ([28, 5, 106, 32], 2, 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 2, "zeros", 1, 1, False, False),
        ([1, 3, 23 * 4, 20 * 8, 32], 2, 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 1, 1, "zeros", 0, 3, False, True),
    ],
    ids=[
        "Wan_shape_17",
        "Wan_shape_18",
        "replicate_T_dim",
        "replicate_W_dim",
        "zeros_W_dim",
        "persistent_buffer",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_output",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "device_params, neighbor_pad_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_neighbor_pad_async_1d(
    mesh_device,
    input_shape,
    halo_shard_dim,
    other_shard_dim,
    padding_left,
    padding_right,
    padding_mode,
    cluster_axis,
    num_links,
    input_dtype,
    layout,
    mem_config_input,
    mem_config_output,
    enable_trace,
    neighbor_pad_topology,
    num_iters,
    is_ci_env,
    skip_for_ci_env,
    use_persistent_output_buffer,
):
    if is_ci_env:
        if skip_for_ci_env:
            pytest.skip("Skipping certain shapes in CI to reduce pipeline time")

    if is_blackhole() and num_links > 2:
        pytest.skip("Skipping num_links > 2 on BH (only 2 ethernet channels available)")
    if not is_blackhole() and num_links == 2:
        pytest.skip("Skipping num_links==2 on WH to remove a redundant test case")

    run_neighbor_pad_1d_impl(
        mesh_device,
        input_shape=input_shape,
        halo_shard_dim=halo_shard_dim,
        other_shard_dim=other_shard_dim,
        padding_left=padding_left,
        padding_right=padding_right,
        padding_mode=padding_mode,
        cluster_axis=cluster_axis,
        num_links=num_links,
        input_dtype=input_dtype,
        layout=layout,
        mem_config_input=mem_config_input,
        mem_config_output=mem_config_output,
        enable_trace=enable_trace,
        neighbor_pad_topology=neighbor_pad_topology,
        num_iters=num_iters,
        use_persistent_output_buffer=use_persistent_output_buffer,
    )


# ---------------------------------------------------------------------------
# 2D tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "mesh_device, num_links",
    [
        [(4, 8), 4],
        [(4, 8), 1],
        [(4, 8), 2],
    ],
    ids=[
        "wh_4x8_4link",
        "general_4x8_1link",
        "bh_4x8_2link",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, use_persistent_output_buffer",
    [
        # 5D: [B, T, H, W, C] — H along axis 0, W along axis 1
        ([1, 3, 12, 16, 32], 2, 3, 0, 1, 1, 1, False),
        # VAE conv_0 shape (full H=90, W=160)
        ([1, 3, 92, 160, 32], 2, 3, 0, 1, 1, 1, False),
        # Flipped axes: H along axis 1, W along axis 0
        ([1, 2, 16, 8, 32], 2, 3, 1, 0, 1, 1, False),
        # 4D tensor [B, H, W, C]
        ([2, 8, 16, 32], 1, 2, 0, 1, 1, 1, False),
        # Larger channel dim
        ([1, 2, 8, 16, 384], 2, 3, 0, 1, 1, 1, False),
        # Padding > 1
        ([1, 2, 8, 16, 32], 2, 3, 0, 1, 2, 2, False),
        # Persistent output buffer
        ([1, 2, 8, 16, 32], 2, 3, 0, 1, 1, 1, True),
    ],
    ids=[
        "medium_5d_h0w1",
        "vae_conv0_h0w1",
        "small_5d_h1w0",
        "small_4d_h0w1",
        "small_5d_largeC",
        "small_5d_pad2",
        "small_5d_persistent",
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_neighbor_pad_async_2d(
    mesh_device,
    input_shape,
    h_dim,
    w_dim,
    h_axis,
    w_axis,
    pH,
    pW,
    num_links,
    use_persistent_output_buffer,
    device_params,
):
    if is_blackhole() and num_links > 2:
        pytest.skip("Skipping num_links > 2 on BH (only 2 ethernet channels available)")
    if not is_blackhole() and num_links == 2:
        pytest.skip("Skipping num_links==2 on WH to remove a redundant test case")
    # Calculate maximum num_links this shape can support
    # For h_dim: outer_dim_size = product of dims before h_dim
    h_outer_dim_size = 1
    for d in range(h_dim):
        h_outer_dim_size *= input_shape[d]

    # For w_dim: outer_dim_size = product of dims before w_dim
    w_outer_dim_size = 1
    for d in range(w_dim):
        w_outer_dim_size *= input_shape[d]

    # Skip if this shape doesn't have enough work for the requested num_links
    max_supported_links = min(h_outer_dim_size, w_outer_dim_size)
    if num_links > max_supported_links:
        print(
            "Warning, reducing num_links from {} to {} for shape {}, h_dim {}, w_dim {} as it is the max allowed for that shape".format(
                num_links, max_supported_links, input_shape, h_dim, w_dim
            )
        )
        num_links = max_supported_links

    run_neighbor_pad_2d_impl(
        mesh_device,
        input_shape=list(input_shape),
        h_dim=h_dim,
        w_dim=w_dim,
        h_axis=h_axis,
        w_axis=w_axis,
        pH=pH,
        pW=pW,
        padding_mode="zeros",
        num_links=num_links,
        input_dtype=ttnn.bfloat16,
        topology=ttnn.Topology.Linear,
        use_persistent_output_buffer=use_persistent_output_buffer,
    )


# ---------------------------------------------------------------------------
# Shared sub-device setup helper
# ---------------------------------------------------------------------------


def _setup_sub_device(mesh_device):
    """Create and load a full-grid sub-device. Returns (core_range_set, stall_group)."""
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([crs])
    stall_group = [ttnn.SubDeviceId(0)]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(stall_group)
    return crs, stall_group


# ---------------------------------------------------------------------------
# 2D + t_front_pad fusion tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(300)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, t_front_pad, num_links, use_persistent_output_buffer, skip_for_ci_env",
    [
        # Slowest NeighborPadAsync from profiler: VAE 720p full res, H=736, W=1280, C=96
        # Per-device: [1, 81, 184, 160, 96], FW ~17.6ms
        ([1, 81, 736, 1280, 96], 2, 3, 0, 1, 1, 1, 2, 2, True, True),
        # Same shape without t_front_pad (regression check)
        ([1, 83, 736, 1280, 96], 2, 3, 0, 1, 1, 1, 0, 2, True, True),
    ],
    ids=[
        "vae_720p_full_res_t_front_2",
        "vae_720p_full_res_no_t_front",
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}],
    indirect=True,
)
def test_neighbor_pad_async_2d_t_front_pad(
    mesh_device,
    input_shape,
    h_dim,
    w_dim,
    h_axis,
    w_axis,
    pH,
    pW,
    t_front_pad,
    num_links,
    use_persistent_output_buffer,
    skip_for_ci_env,
    is_ci_env,
    device_params,
):
    if is_ci_env and skip_for_ci_env:
        pytest.skip("Skipping large shapes in CI to reduce pipeline time")
    if is_blackhole() and num_links > 2:
        pytest.skip("Skipping num_links > 2 on BH (only 2 ethernet channels available)")

    run_neighbor_pad_2d_combined_impl(
        mesh_device,
        input_shape=list(input_shape),
        h_dim=h_dim,
        w_dim=w_dim,
        h_axis=h_axis,
        w_axis=w_axis,
        pH=pH,
        pW=pW,
        t_front_pad=t_front_pad,
        num_links=num_links,
        use_persistent_output_buffer=use_persistent_output_buffer,
    )


# ---------------------------------------------------------------------------
# t_front_pad fusion tests
# ---------------------------------------------------------------------------


def run_neighbor_pad_t_front_pad_impl(
    mesh_device,
    input_shape,
    h_dim,
    h_axis,
    other_shard_dim,
    padding_h,
    t_front_pad,
    num_links,
    input_dtype=ttnn.bfloat16,
):
    """
    Test t_front_pad fusion in neighbor_pad_async.

    The expected per-device output is [B, T+t_front_pad, H_chunk+2*pH, W_chunk, C]:
    - T-front rows (0..t_front_pad-1): all zeros (interior + H halo)
    - Input T rows (t_front_pad..T+t_front_pad-1): input data padded with neighbor H halo
    """
    torch.manual_seed(42)
    mesh_shape = tuple(mesh_device.shape)
    h_factor = mesh_shape[h_axis]
    w_axis = 1 - h_axis
    w_factor = mesh_shape[w_axis]

    assert input_shape[h_dim] % h_factor == 0
    assert input_shape[other_shard_dim] % w_factor == 0
    assert input_shape[0] == 1, "t_front_pad requires B=1"

    input_tensor = torch.rand(input_shape).bfloat16()

    # Golden: per-device comparison (accounts for both H and W sharding).
    # For each W column, independently apply H halo padding and prepend T-front zeros.
    t_dim = h_dim - 1
    h_chunks = list(torch.chunk(input_tensor, h_factor, dim=h_dim))
    # Chunk each H slice along the W (other_shard) dimension
    goldens = {}
    for h_idx in range(h_factor):
        w_chunks = list(torch.chunk(h_chunks[h_idx], w_factor, dim=other_shard_dim))
        for w_idx in range(w_factor):
            # H halo for this (h_idx, w_idx): need neighbor H data from same W column
            col_h_chunks = [torch.chunk(h_chunks[hi], w_factor, dim=other_shard_dim)[w_idx] for hi in range(h_factor)]
            padded_col = pad_chunks_along_dim(col_h_chunks, h_dim, padding_h, padding_h, "zeros")
            padded_hw = padded_col[h_idx]
            # Prepend T-front zero frames
            zero_shape = list(padded_hw.shape)
            zero_shape[t_dim] = t_front_pad
            zero_front = torch.zeros(zero_shape, dtype=padded_hw.dtype)
            golden = torch.cat([zero_front, padded_hw], dim=t_dim)
            key = (h_idx, w_idx) if h_axis == 0 else (w_idx, h_idx)
            goldens[key] = golden

    crs, stall_group = _setup_sub_device(mesh_device)

    h_neighbor_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)

    dims = [None, None]
    dims[h_axis] = h_dim
    dims[1 - h_axis] = other_shard_dim
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    input_tensor_mesh = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=input_dtype,
        memory_config=mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )

    output_tensor = ttnn.experimental.neighbor_pad_async(
        input_tensor_mesh,
        [h_dim],
        [padding_h],
        [padding_h],
        "zeros",
        [h_axis],
        [h_neighbor_sem],
        [barrier_sem],
        num_links=[num_links],
        memory_config=mem_config,
        topology=ttnn.Topology.Linear,
        t_front_pad=t_front_pad,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=stall_group)

    output_host = ttnn.from_device(output_tensor)
    device_tensors = ttnn.get_device_tensors(output_host)

    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            device_idx = row * mesh_shape[1] + col
            dev_tensor = ttnn.to_torch(device_tensors[device_idx])
            golden = goldens[(row, col)]
            assert (
                dev_tensor.shape == golden.shape
            ), f"Device ({row},{col}): shape mismatch: got {dev_tensor.shape}, expected {golden.shape}"
            eq, msg = comp_equal(dev_tensor, golden)
            assert eq, f"Device ({row},{col}): {msg}"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


# ---------------------------------------------------------------------------
# logical_h masking tests
# ---------------------------------------------------------------------------


def run_neighbor_pad_logical_h_impl(
    mesh_device,
    input_shape,
    halo_dim,
    other_shard_dim,
    cluster_axis,
    padding_left,
    padding_right,
    logical_h,
    num_links,
    input_dtype=ttnn.bfloat16,
):
    """
    Test that neighbor_pad with logical_h zeros interior rows at/beyond logical_h,
    matching the behavior of masking before neighbor_pad.

    The input tensor has random (non-zero) data in all rows, including those at/beyond
    logical_h. The expected output treats those rows as if they were zero before padding.
    """
    torch.manual_seed(42)
    mesh_shape = tuple(mesh_device.shape)
    h_factor = mesh_shape[cluster_axis]

    assert (
        input_shape[halo_dim] % h_factor == 0
    ), f"input_shape[{halo_dim}]={input_shape[halo_dim]} must be divisible by h_factor={h_factor}"
    assert (
        0 < logical_h <= input_shape[halo_dim]
    ), f"logical_h={logical_h} must be in (0, input_shape[{halo_dim}]={input_shape[halo_dim]}]"

    input_tensor = torch.rand(input_shape).bfloat16()

    # Golden: zero rows >= logical_h, then chunk and pad
    masked = input_tensor.clone()
    slices = [slice(None)] * input_tensor.ndim
    slices[halo_dim] = slice(logical_h, None)
    masked[tuple(slices)] = 0.0
    chunks = list(torch.chunk(masked, h_factor, halo_dim))
    padded = pad_chunks_along_dim(chunks, halo_dim, padding_left, padding_right, "zeros")
    golden = torch.cat(padded, dim=halo_dim)

    crs, stall_group = _setup_sub_device(mesh_device)

    neighbor_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)

    dims = [None, None]
    dims[cluster_axis] = halo_dim
    dims[1 - cluster_axis] = other_shard_dim
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    # Pass UNMASKED input — logical_h in neighbor_pad should do the masking
    input_tensor_mesh = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=input_dtype,
        memory_config=mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )

    output_tensor = ttnn.experimental.neighbor_pad_async(
        input_tensor_mesh,
        [halo_dim],
        [padding_left],
        [padding_right],
        "zeros",
        [cluster_axis],
        [neighbor_sem],
        [barrier_sem],
        num_links=[num_links],
        memory_config=mem_config,
        topology=ttnn.Topology.Linear,
        logical_h=logical_h,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=stall_group)

    tt_out = ttnn.to_torch(
        ttnn.from_device(output_tensor),
        mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )

    logger.info(f"logical_h={logical_h}, input_shape={input_shape}, output shape={tt_out.shape}")
    eq, msg = comp_equal(tt_out, golden)
    assert eq, f"logical_h masking mismatch: {msg}"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


# ---------------------------------------------------------------------------
# 2D logical_h masking tests — exercises W reader Phase 1 masking
# ---------------------------------------------------------------------------


def run_neighbor_pad_2d_logical_h_impl(
    mesh_device,
    input_shape,
    h_dim,
    w_dim,
    h_axis,
    w_axis,
    pH,
    pW,
    logical_h,
    num_links,
    input_dtype=ttnn.bfloat16,
    use_persistent_output_buffer=False,
):
    """
    Test 2D neighbor_pad with logical_h masking.

    Rows at global H index >= logical_h in the input contain non-zero random data
    (simulating mesh-partition padding). The op must zero those rows both in the
    local copy and in the W fabric exchange, so the output matches a golden
    computed by zeroing those rows before padding.

    This specifically exercises the bug where W reader Phase 1 read unmasked
    input boundary columns for rows >= logical_h and sent non-zero data to the
    W neighbor.
    """
    torch.manual_seed(0)
    mesh_shape = tuple(mesh_device.shape)
    h_factor = mesh_shape[h_axis]
    w_factor = mesh_shape[w_axis]

    assert (
        input_shape[h_dim] % h_factor == 0
    ), f"input_shape[{h_dim}]={input_shape[h_dim]} must be divisible by h_factor={h_factor}"
    assert (
        input_shape[w_dim] % w_factor == 0
    ), f"input_shape[{w_dim}]={input_shape[w_dim]} must be divisible by w_factor={w_factor}"
    assert (
        0 < logical_h <= input_shape[h_dim]
    ), f"logical_h={logical_h} must be in (0, input_shape[{h_dim}]={input_shape[h_dim]}]"

    input_tensor = torch.rand(input_shape).bfloat16()

    # Golden: zero rows >= logical_h, then apply 2D padding
    masked_input = input_tensor.clone()
    slices = [slice(None)] * input_tensor.ndim
    slices[h_dim] = slice(logical_h, None)
    masked_input[tuple(slices)] = 0.0
    goldens = compute_2d_pad_golden(masked_input, mesh_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, "zeros")

    crs, stall_group = _setup_sub_device(mesh_device)

    h_neighbor_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    w_neighbor_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)

    dims = [None, None]
    dims[h_axis] = h_dim
    dims[w_axis] = w_dim
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    persistent_output_buffer = None
    if use_persistent_output_buffer:
        output_shape = list(input_shape)
        output_shape[h_dim] += h_factor * (pH + pH)
        output_shape[w_dim] += w_factor * (pW + pW)
        persistent_output_buffer = ttnn.from_torch(
            torch.zeros(output_shape).bfloat16(),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=input_dtype,
            memory_config=mem_config,
            mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
        )

    input_tensor_mesh = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=input_dtype,
        memory_config=mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )

    output_tensor = ttnn.experimental.neighbor_pad_async(
        input_tensor_mesh,
        [h_dim, w_dim],
        [pH, pW],
        [pH, pW],
        "zeros",
        [h_axis, w_axis],
        [h_neighbor_sem, w_neighbor_sem],
        [barrier_sem],
        num_links=[num_links, num_links],
        memory_config=mem_config,
        topology=ttnn.Topology.Linear,
        persistent_output_buffer=persistent_output_buffer,
        logical_h=logical_h,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=stall_group)

    output_host = ttnn.from_device(output_tensor)
    device_tensors = ttnn.get_device_tensors(output_host)

    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            device_idx = row * mesh_shape[1] + col
            dev_tensor = ttnn.to_torch(device_tensors[device_idx])
            golden = goldens[(row, col)]
            assert (
                dev_tensor.shape == golden.shape
            ), f"Device ({row},{col}): shape mismatch: got {dev_tensor.shape}, expected {golden.shape}"
            eq, msg = comp_equal(dev_tensor, golden)
            assert eq, f"Device ({row},{col}): {msg}"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


# ===========================================================================
# Comprehensive 4x8 BH sweep — 100 tests, all runnable with 2 links
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared impl for combined logical_h + t_front_pad + 2D
# ---------------------------------------------------------------------------


def run_neighbor_pad_2d_combined_impl(
    mesh_device,
    input_shape,
    h_dim,
    w_dim,
    h_axis,
    w_axis,
    pH,
    pW,
    t_front_pad,
    num_links,
    logical_h=None,
    input_dtype=ttnn.bfloat16,
    use_persistent_output_buffer=False,
):
    """
    2D neighbor pad with optional logical_h masking and t_front_pad fusion.

    logical_h=None means no masking (equivalent to logical_h=input_shape[h_dim]).
    t_front_pad=0 means no T-frame prepend. Covers all 2D feature combinations.

    Golden: zero input rows >= logical_h, apply 2D spatial padding, prepend
    t_front_pad zero T-frames.
    """
    torch.manual_seed(7)
    mesh_shape = tuple(mesh_device.shape)
    h_factor = mesh_shape[h_axis]
    w_factor = mesh_shape[w_axis]

    if logical_h is None:
        logical_h = input_shape[h_dim]

    assert input_shape[h_dim] % h_factor == 0
    assert input_shape[w_dim] % w_factor == 0
    if t_front_pad > 0:
        assert input_shape[0] == 1, "t_front_pad requires B=1"
    assert 0 < logical_h <= input_shape[h_dim]

    input_tensor = torch.rand(input_shape).bfloat16()
    t_dim = h_dim - 1

    # Golden: mask then 2D-pad then prepend T-front zeros
    masked = input_tensor.clone()
    slices = [slice(None)] * input_tensor.ndim
    slices[h_dim] = slice(logical_h, None)
    masked[tuple(slices)] = 0.0
    goldens_2d = compute_2d_pad_golden(masked, mesh_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, "zeros")
    goldens = {}
    for key, padded in goldens_2d.items():
        zero_shape = list(padded.shape)
        zero_shape[t_dim] = t_front_pad
        front = torch.zeros(zero_shape, dtype=padded.dtype)
        goldens[key] = torch.cat([front, padded], dim=t_dim) if t_front_pad > 0 else padded

    crs, stall_group = _setup_sub_device(mesh_device)

    h_neighbor_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    w_neighbor_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)

    dims = [None, None]
    dims[h_axis] = h_dim
    dims[w_axis] = w_dim
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    persistent_output_buffer = None
    if use_persistent_output_buffer:
        out_shape = list(input_shape)
        out_shape[t_dim] += t_front_pad
        out_shape[h_dim] += h_factor * (pH + pH)
        out_shape[w_dim] += w_factor * (pW + pW)
        persistent_output_buffer = ttnn.from_torch(
            torch.zeros(out_shape).bfloat16(),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=input_dtype,
            memory_config=mem_config,
            mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
        )

    input_mesh = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=input_dtype,
        memory_config=mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )

    output_tensor = ttnn.experimental.neighbor_pad_async(
        input_mesh,
        [h_dim, w_dim],
        [pH, pW],
        [pH, pW],
        "zeros",
        [h_axis, w_axis],
        [h_neighbor_sem, w_neighbor_sem],
        [barrier_sem],
        num_links=[num_links, num_links],
        memory_config=mem_config,
        topology=ttnn.Topology.Linear,
        persistent_output_buffer=persistent_output_buffer,
        logical_h=logical_h,
        t_front_pad=t_front_pad,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=stall_group)

    output_host = ttnn.from_device(output_tensor)
    device_tensors = ttnn.get_device_tensors(output_host)

    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            dev_tensor = ttnn.to_torch(device_tensors[row * mesh_shape[1] + col])
            golden = goldens[(row, col)]
            assert dev_tensor.shape == golden.shape, f"Device ({row},{col}): shape {dev_tensor.shape} != {golden.shape}"
            eq, msg = comp_equal(dev_tensor, golden)
            assert eq, f"Device ({row},{col}): {msg}"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


# ---------------------------------------------------------------------------
# test_np_bh_2d_shapes — 25 cases: shape/padding/channel/persistent sweep
# ---------------------------------------------------------------------------

_BH_2D_SHAPES = [
    # (input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, persistent, skip_ci)
    ([1, 2, 4 * 4, 4 * 8, 32], 2, 3, 0, 1, 1, 1, False, False),
    ([1, 2, 4 * 4, 4 * 8, 32], 2, 3, 0, 1, 2, 2, False, True),
    ([1, 2, 4 * 4, 4 * 8, 32], 2, 3, 0, 1, 1, 2, False, True),
    ([1, 2, 4 * 4, 4 * 8, 32], 2, 3, 0, 1, 2, 1, False, False),
    ([1, 3, 8 * 4, 8 * 8, 32], 2, 3, 0, 1, 1, 1, False, True),
    ([1, 3, 8 * 4, 8 * 8, 64], 2, 3, 0, 1, 1, 1, False, True),
    ([1, 3, 8 * 4, 8 * 8, 96], 2, 3, 0, 1, 1, 1, False, True),
    ([1, 3, 8 * 4, 8 * 8, 128], 2, 3, 0, 1, 1, 1, False, True),
    ([1, 4, 12 * 4, 16 * 8, 32], 2, 3, 0, 1, 1, 1, False, True),
    ([1, 4, 12 * 4, 16 * 8, 64], 2, 3, 0, 1, 1, 1, False, True),
    ([2, 2, 8 * 4, 8 * 8, 32], 2, 3, 0, 1, 1, 1, False, True),  # B=2
    ([1, 2, 16 * 4, 8 * 8, 32], 2, 3, 0, 1, 1, 1, False, True),  # tall H
    ([1, 2, 8 * 4, 16 * 8, 32], 2, 3, 0, 1, 1, 1, False, True),  # wide W
    ([1, 2, 4 * 4, 4 * 8, 128], 2, 3, 0, 1, 1, 1, False, True),
    ([1, 2, 4 * 4, 4 * 8, 256], 2, 3, 0, 1, 1, 1, False, True),
    ([1, 2, 4 * 4, 4 * 8, 384], 2, 3, 0, 1, 1, 1, False, True),
    ([1, 3, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, False, True),  # VAE-like shape
    ([1, 3, 23 * 4, 20 * 8, 96], 2, 3, 0, 1, 1, 1, False, True),
    ([1, 2, 4 * 4, 4 * 8, 32], 2, 3, 0, 1, 1, 1, True, False),  # persistent
    ([1, 3, 8 * 4, 8 * 8, 64], 2, 3, 0, 1, 1, 1, True, True),  # persistent medium
    ([1, 2, 8 * 4, 8 * 8, 32], 2, 3, 0, 1, 2, 2, False, True),
    ([1, 5, 4 * 4, 4 * 8, 32], 2, 3, 0, 1, 1, 1, False, True),  # T=5
    ([1, 7, 4 * 4, 4 * 8, 32], 2, 3, 0, 1, 1, 1, False, True),  # T=7 odd
    ([1, 2, 4 * 8, 4 * 4, 32], 2, 3, 1, 0, 1, 1, False, True),  # flipped axes
    ([1, 2, 8 * 4, 8 * 8, 32], 2, 3, 0, 1, 1, 2, True, True),  # persistent pW=2
]

_BH_2D_SHAPES_IDS = [
    "t2_H16_W32_C32_p1x1",
    "t2_H16_W32_C32_p2x2",
    "t2_H16_W32_C32_p1x2",
    "t2_H16_W32_C32_p2x1",
    "t3_H32_W64_C32",
    "t3_H32_W64_C64",
    "t3_H32_W64_C96",
    "t3_H32_W64_C128",
    "t4_H48_W128_C32",
    "t4_H48_W128_C64",
    "B2_t2_H32_W64_C32",
    "t2_H64_W64_tallH",
    "t2_H32_W128_wideW",
    "t2_H16_W32_C128",
    "t2_H16_W32_C256",
    "t2_H16_W32_C384",
    "t3_vae_C32",
    "t3_vae_C96",
    "t2_H16_W32_persist",
    "t3_H32_W64_persist",
    "t2_H32_W64_p2x2",
    "t5_H16_W32",
    "t7_H16_W32_odd",
    "t2_flipped_axes",
    "t2_H32_W64_pW2_persist",
]


@pytest.mark.timeout(300)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, use_persistent_output_buffer, skip_for_ci_env",
    _BH_2D_SHAPES,
    ids=_BH_2D_SHAPES_IDS,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_np_bh_2d_shapes(
    mesh_device,
    input_shape,
    h_dim,
    w_dim,
    h_axis,
    w_axis,
    pH,
    pW,
    use_persistent_output_buffer,
    skip_for_ci_env,
    is_ci_env,
    device_params,
):
    """Shape/padding/channel sweep for 2D neighbor pad on 4x8 BH with 2 links."""
    if is_ci_env and skip_for_ci_env:
        pytest.skip("Skipping sweep shape in CI to reduce pipeline time")
    if not is_blackhole():
        pytest.skip("Sized for 4x8 BH mesh")

    num_links = 2
    # Reduce links if shape can't support them (T=1 edge case)
    outer = 1
    for d in range(h_dim):
        outer *= input_shape[d]
    if outer < num_links:
        num_links = outer

    run_neighbor_pad_2d_impl(
        mesh_device,
        input_shape=list(input_shape),
        h_dim=h_dim,
        w_dim=w_dim,
        h_axis=h_axis,
        w_axis=w_axis,
        pH=pH,
        pW=pW,
        padding_mode="zeros",
        num_links=num_links,
        input_dtype=ttnn.bfloat16,
        topology=ttnn.Topology.Linear,
        use_persistent_output_buffer=use_persistent_output_buffer,
    )


# ---------------------------------------------------------------------------
# test_np_bh_logical_h_1d — 15 cases: 1D H-only logical_h masking
# ---------------------------------------------------------------------------

_BH_LH_1D = [
    # (input_shape, halo_dim, other_shard_dim, cluster_axis, pad_l, pad_r, logical_h, skip_ci)
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 90, True),  # 2 excess rows, C=32
    ([1, 3, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 90, True),  # T=3
    ([1, 2, 23 * 4, 20 * 8, 64], 2, 3, 0, 1, 1, 90, True),  # C=64
    ([1, 2, 23 * 4, 20 * 8, 96], 2, 3, 0, 1, 1, 90, True),  # C=96
    ([1, 2, 23 * 4, 20 * 8, 128], 2, 3, 0, 1, 1, 90, True),  # C=128
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 23 * 4, True),  # no masking
    ([1, 2, 24 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 90, True),  # 6 excess rows
    ([1, 2, 24 * 4, 20 * 8, 64], 2, 3, 0, 1, 1, 90, True),  # 6 excess, C=64
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 3, 0, 2, 2, 90, True),  # pad=2
    ([1, 2, 8 * 4, 8 * 8, 32], 2, 3, 0, 1, 1, 30, False),  # small, 2 excess
    ([1, 2, 8 * 4, 8 * 8, 64], 2, 3, 0, 1, 1, 30, True),  # small C=64
    ([1, 2, 4 * 4, 4 * 8, 32], 2, 3, 0, 1, 1, 14, True),  # tiny, 2 excess
    ([1, 4, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 90, True),  # T=4
    ([1, 2, 8 * 4, 8 * 8, 32], 2, 3, 0, 1, 1, 8 * 4, False),  # no masking small
    ([1, 2, 4 * 4, 4 * 8, 96], 2, 3, 0, 1, 1, 14, True),  # tiny C=96
]

_BH_LH_1D_IDS = [
    "vae92_lh90_C32",
    "vae92_lh90_T3",
    "vae92_lh90_C64",
    "vae92_lh90_C96",
    "vae92_lh90_C128",
    "vae92_lh92_none",
    "vae96_lh90_C32",
    "vae96_lh90_C64",
    "vae92_lh90_pad2",
    "H32_lh30_C32",
    "H32_lh30_C64",
    "H16_lh14_C32",
    "vae92_lh90_T4",
    "H32_lh32_none",
    "H16_lh14_C96",
]


@pytest.mark.timeout(300)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "input_shape, halo_dim, other_shard_dim, cluster_axis, padding_left, padding_right, logical_h, skip_for_ci_env",
    _BH_LH_1D,
    ids=_BH_LH_1D_IDS,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}],
    indirect=True,
)
def test_np_bh_logical_h_1d(
    mesh_device,
    input_shape,
    halo_dim,
    other_shard_dim,
    cluster_axis,
    padding_left,
    padding_right,
    logical_h,
    skip_for_ci_env,
    is_ci_env,
    device_params,
):
    """1D H-only neighbor pad with logical_h masking on 4x8 BH with 2 links."""
    if is_ci_env and skip_for_ci_env:
        pytest.skip("Skipping sweep shape in CI to reduce pipeline time")
    if not is_blackhole():
        pytest.skip("Sized for 4x8 BH mesh")

    run_neighbor_pad_logical_h_impl(
        mesh_device,
        input_shape=list(input_shape),
        halo_dim=halo_dim,
        other_shard_dim=other_shard_dim,
        cluster_axis=cluster_axis,
        padding_left=padding_left,
        padding_right=padding_right,
        logical_h=logical_h,
        num_links=2,
    )


# ---------------------------------------------------------------------------
# test_np_bh_logical_h_2d — 20 cases: 2D H+W logical_h masking (the bug fix)
# ---------------------------------------------------------------------------

_BH_LH_2D = [
    # (input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, logical_h, persistent, skip_ci)
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 90, False, True),  # core repro
    ([1, 3, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 90, False, True),  # T=3
    ([1, 2, 23 * 4, 20 * 8, 64], 2, 3, 0, 1, 1, 1, 90, False, True),  # C=64
    ([1, 2, 23 * 4, 20 * 8, 96], 2, 3, 0, 1, 1, 1, 90, False, True),  # C=96
    ([1, 2, 23 * 4, 20 * 8, 128], 2, 3, 0, 1, 1, 1, 90, False, True),  # C=128
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 23 * 4, False, True),  # no masking
    ([1, 2, 24 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 90, False, True),  # 6 excess rows
    ([1, 2, 24 * 4, 20 * 8, 64], 2, 3, 0, 1, 1, 1, 90, False, True),  # 6 excess C=64
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 2, 1, 90, False, True),  # pH=2
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 2, 90, False, True),  # pW=2
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 2, 2, 90, False, True),  # pH=pW=2
    ([1, 4, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 90, False, True),  # T=4 (2 per W link)
    ([1, 5, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 90, False, True),  # T=5
    ([1, 2, 8 * 4, 8 * 8, 32], 2, 3, 0, 1, 1, 1, 30, False, False),  # small 2 excess
    ([1, 2, 8 * 4, 8 * 8, 64], 2, 3, 0, 1, 1, 1, 30, False, True),  # small C=64
    ([1, 2, 4 * 4, 4 * 8, 32], 2, 3, 0, 1, 1, 1, 14, False, True),  # tiny 2 excess
    ([1, 2, 24 * 4, 20 * 8, 96], 2, 3, 0, 1, 1, 1, 90, False, True),  # 6 excess C=96
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 90, True, False),  # persistent
    ([1, 2, 23 * 4, 20 * 8, 64], 2, 3, 0, 1, 1, 1, 90, True, True),  # persistent C=64
    ([2, 2, 8 * 4, 8 * 8, 32], 2, 3, 0, 1, 1, 1, 30, False, True),  # B=2
]

_BH_LH_2D_IDS = [
    "vae92_lh90_C32",
    "vae92_lh90_T3",
    "vae92_lh90_C64",
    "vae92_lh90_C96",
    "vae92_lh90_C128",
    "vae92_lh92_none",
    "vae96_lh90_C32",
    "vae96_lh90_C64",
    "vae92_lh90_pH2",
    "vae92_lh90_pW2",
    "vae92_lh90_p2x2",
    "vae92_lh90_T4",
    "vae92_lh90_T5",
    "H32_lh30_C32",
    "H32_lh30_C64",
    "H16_lh14_C32",
    "vae96_lh90_C96",
    "vae92_lh90_persist",
    "vae92_lh90_C64_persist",
    "B2_H32_lh30",
]


@pytest.mark.timeout(300)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, logical_h, use_persistent_output_buffer, skip_for_ci_env",
    _BH_LH_2D,
    ids=_BH_LH_2D_IDS,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}],
    indirect=True,
)
def test_np_bh_logical_h_2d(
    mesh_device,
    input_shape,
    h_dim,
    w_dim,
    h_axis,
    w_axis,
    pH,
    pW,
    logical_h,
    use_persistent_output_buffer,
    skip_for_ci_env,
    is_ci_env,
    device_params,
):
    """2D H+W neighbor pad with logical_h masking — verifies W reader Phase 1 fix."""
    if is_ci_env and skip_for_ci_env:
        pytest.skip("Skipping sweep shape in CI to reduce pipeline time")
    if not is_blackhole():
        pytest.skip("Sized for 4x8 BH mesh")

    run_neighbor_pad_2d_logical_h_impl(
        mesh_device,
        input_shape=list(input_shape),
        h_dim=h_dim,
        w_dim=w_dim,
        h_axis=h_axis,
        w_axis=w_axis,
        pH=pH,
        pW=pW,
        logical_h=logical_h,
        num_links=2,
        use_persistent_output_buffer=use_persistent_output_buffer,
    )


# ---------------------------------------------------------------------------
# test_np_bh_t_front_1d — 10 cases: 1D H-only t_front_pad fusion
# ---------------------------------------------------------------------------

_BH_TF_1D = [
    # (input_shape, h_dim, h_axis, other_shard_dim, padding_h, t_front_pad, skip_ci)
    ([1, 3, 23 * 4, 20 * 8, 32], 2, 0, 3, 1, 2, True),
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 0, 3, 1, 1, True),
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 0, 3, 1, 0, False),  # t_front=0 regression
    ([1, 4, 8 * 4, 8 * 8, 32], 2, 0, 3, 1, 2, False),
    ([1, 2, 8 * 4, 8 * 8, 64], 2, 0, 3, 1, 1, True),
    ([1, 5, 4 * 4, 4 * 8, 32], 2, 0, 3, 1, 4, True),
    ([1, 3, 4 * 4, 4 * 8, 96], 2, 0, 3, 1, 2, True),
    ([1, 3, 23 * 4, 20 * 8, 64], 2, 0, 3, 1, 2, True),
    ([1, 2, 8 * 4, 8 * 8, 128], 2, 0, 3, 1, 1, True),
    ([1, 7, 4 * 4, 4 * 8, 32], 2, 0, 3, 1, 4, True),
]

_BH_TF_1D_IDS = [
    "t3_vae_tf2",
    "t2_vae_tf1",
    "t2_vae_tf0",
    "t4_H32_tf2",
    "t2_H32_C64",
    "t5_H16_tf4",
    "t3_H16_C96",
    "t3_vae_C64",
    "t2_H32_C128",
    "t7_H16_tf4",
]


@pytest.mark.timeout(300)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "input_shape, h_dim, h_axis, other_shard_dim, padding_h, t_front_pad, skip_for_ci_env",
    _BH_TF_1D,
    ids=_BH_TF_1D_IDS,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}],
    indirect=True,
)
def test_np_bh_t_front_1d(
    mesh_device,
    input_shape,
    h_dim,
    h_axis,
    other_shard_dim,
    padding_h,
    t_front_pad,
    skip_for_ci_env,
    is_ci_env,
    device_params,
):
    """1D H-only neighbor pad with t_front_pad fusion on 4x8 BH with 2 links."""
    if is_ci_env and skip_for_ci_env:
        pytest.skip("Skipping sweep shape in CI to reduce pipeline time")
    if not is_blackhole():
        pytest.skip("Sized for 4x8 BH mesh")

    run_neighbor_pad_t_front_pad_impl(
        mesh_device,
        input_shape=list(input_shape),
        h_dim=h_dim,
        h_axis=h_axis,
        other_shard_dim=other_shard_dim,
        padding_h=padding_h,
        t_front_pad=t_front_pad,
        num_links=2,
    )


# ---------------------------------------------------------------------------
# test_np_bh_t_front_2d — 15 cases: 2D H+W t_front_pad fusion
# ---------------------------------------------------------------------------

_BH_TF_2D = [
    # (input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, t_front_pad, persistent, skip_ci)
    ([1, 3, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 2, False, True),
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 1, False, True),
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 0, False, True),  # regression
    ([1, 4, 8 * 4, 8 * 8, 32], 2, 3, 0, 1, 1, 1, 2, False, False),
    ([1, 2, 8 * 4, 8 * 8, 64], 2, 3, 0, 1, 1, 1, 1, False, True),
    ([1, 5, 4 * 4, 4 * 8, 32], 2, 3, 0, 1, 1, 1, 4, False, True),
    ([1, 3, 23 * 4, 20 * 8, 96], 2, 3, 0, 1, 1, 1, 2, False, True),
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 1, True, False),  # persistent
    ([1, 2, 4 * 4, 4 * 8, 128], 2, 3, 0, 1, 1, 1, 2, False, True),
    ([1, 3, 4 * 4, 4 * 8, 32], 2, 3, 0, 1, 2, 2, 1, False, True),  # pH=pW=2
    ([1, 4, 4 * 4, 4 * 8, 32], 2, 3, 0, 1, 2, 1, 2, False, True),  # pH=2
    ([1, 2, 8 * 4, 8 * 8, 32], 2, 3, 0, 1, 1, 2, 1, False, True),  # pW=2
    ([1, 9, 4 * 4, 4 * 8, 32], 2, 3, 0, 1, 1, 1, 2, False, True),  # T=9 odd
    ([1, 3, 8 * 4, 8 * 8, 64], 2, 3, 0, 1, 1, 1, 2, True, True),  # persist medium
    ([1, 2, 23 * 4, 20 * 8, 64], 2, 3, 0, 1, 1, 1, 2, False, True),
]

_BH_TF_2D_IDS = [
    "t3_vae_tf2",
    "t2_vae_tf1",
    "t2_vae_tf0",
    "t4_H32_tf2",
    "t2_H32_C64",
    "t5_H16_tf4",
    "t3_vae_C96",
    "t2_vae_persist",
    "t2_H16_C128",
    "t3_H16_p2x2",
    "t4_H16_pH2",
    "t2_H32_pW2",
    "t9_H16_odd",
    "t3_H32_persist",
    "t2_vae_C64",
]


@pytest.mark.timeout(300)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, t_front_pad, use_persistent_output_buffer, skip_for_ci_env",
    _BH_TF_2D,
    ids=_BH_TF_2D_IDS,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}],
    indirect=True,
)
def test_np_bh_t_front_2d(
    mesh_device,
    input_shape,
    h_dim,
    w_dim,
    h_axis,
    w_axis,
    pH,
    pW,
    t_front_pad,
    use_persistent_output_buffer,
    skip_for_ci_env,
    is_ci_env,
    device_params,
):
    """2D H+W neighbor pad with t_front_pad fusion on 4x8 BH with 2 links."""
    if is_ci_env and skip_for_ci_env:
        pytest.skip("Skipping sweep shape in CI to reduce pipeline time")
    if not is_blackhole():
        pytest.skip("Sized for 4x8 BH mesh")

    run_neighbor_pad_2d_combined_impl(
        mesh_device,
        input_shape=list(input_shape),
        h_dim=h_dim,
        w_dim=w_dim,
        h_axis=h_axis,
        w_axis=w_axis,
        pH=pH,
        pW=pW,
        t_front_pad=t_front_pad,
        num_links=2,
        use_persistent_output_buffer=use_persistent_output_buffer,
    )


# ---------------------------------------------------------------------------
# test_np_bh_combined — 15 cases: 2D + logical_h + t_front_pad together
# ---------------------------------------------------------------------------

_BH_COMBINED = [
    # (input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, logical_h, t_front_pad, persistent, skip_ci)
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 90, 1, False, True),
    ([1, 3, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 90, 2, False, True),
    ([1, 2, 23 * 4, 20 * 8, 96], 2, 3, 0, 1, 1, 1, 90, 1, False, True),
    ([1, 4, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 90, 2, False, True),
    ([1, 2, 24 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 90, 1, False, True),  # 6 excess
    ([1, 3, 24 * 4, 20 * 8, 64], 2, 3, 0, 1, 1, 1, 90, 2, False, True),
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 90, 1, True, True),  # persistent
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 23 * 4, 2, False, False),  # no H mask
    ([1, 2, 8 * 4, 8 * 8, 32], 2, 3, 0, 1, 1, 1, 30, 1, False, False),
    ([1, 3, 8 * 4, 8 * 8, 64], 2, 3, 0, 1, 1, 1, 30, 2, False, True),
    ([1, 5, 4 * 4, 4 * 8, 32], 2, 3, 0, 1, 1, 1, 14, 3, False, True),
    ([1, 2, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 1, 90, 4, False, True),  # large tf
    ([1, 3, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 2, 1, 90, 1, False, True),  # pH=2
    ([1, 3, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 2, 90, 2, False, True),  # pW=2
    ([1, 2, 24 * 4, 20 * 8, 96], 2, 3, 0, 1, 1, 1, 90, 2, False, True),
]

_BH_COMBINED_IDS = [
    "vae92_lh90_tf1",
    "vae92_lh90_T3_tf2",
    "vae92_lh90_C96_tf1",
    "vae92_lh90_T4_tf2",
    "vae96_lh90_tf1",
    "vae96_lh90_C64_tf2",
    "vae92_lh90_persist",
    "vae92_lh92_tf2",
    "H32_lh30_tf1",
    "H32_lh30_C64_tf2",
    "H16_lh14_tf3",
    "vae92_lh90_tf4",
    "vae92_lh90_pH2_tf1",
    "vae92_lh90_pW2_tf2",
    "vae96_lh90_C96_tf2",
]


@pytest.mark.timeout(300)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, logical_h, t_front_pad, use_persistent_output_buffer, skip_for_ci_env",
    _BH_COMBINED,
    ids=_BH_COMBINED_IDS,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}],
    indirect=True,
)
def test_np_bh_combined(
    mesh_device,
    input_shape,
    h_dim,
    w_dim,
    h_axis,
    w_axis,
    pH,
    pW,
    logical_h,
    t_front_pad,
    use_persistent_output_buffer,
    skip_for_ci_env,
    is_ci_env,
    device_params,
):
    """2D neighbor pad with both logical_h masking and t_front_pad active."""
    if is_ci_env and skip_for_ci_env:
        pytest.skip("Skipping sweep shape in CI to reduce pipeline time")
    if not is_blackhole():
        pytest.skip("Sized for 4x8 BH mesh")

    run_neighbor_pad_2d_combined_impl(
        mesh_device,
        input_shape=list(input_shape),
        h_dim=h_dim,
        w_dim=w_dim,
        h_axis=h_axis,
        w_axis=w_axis,
        pH=pH,
        pW=pW,
        logical_h=logical_h,
        t_front_pad=t_front_pad,
        num_links=2,
        use_persistent_output_buffer=use_persistent_output_buffer,
    )
