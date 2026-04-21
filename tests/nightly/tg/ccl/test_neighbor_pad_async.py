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

    assert input_shape[halo_dim] % h_factor == 0, (
        f"input_shape[{halo_dim}]={input_shape[halo_dim]} must be divisible by h_factor={h_factor}"
    )
    assert 0 < logical_h <= input_shape[halo_dim], (
        f"logical_h={logical_h} must be in (0, input_shape[{halo_dim}]={input_shape[halo_dim]}]"
    )

    input_tensor = torch.rand(input_shape).bfloat16()

    # Golden: zero rows >= logical_h, then chunk and pad
    masked = input_tensor.clone()
    slices = [slice(None)] * input_tensor.ndim
    slices[halo_dim] = slice(logical_h, None)
    masked[tuple(slices)] = 0.0
    chunks = list(torch.chunk(masked, h_factor, halo_dim))
    padded = pad_chunks_along_dim(chunks, halo_dim, padding_left, padding_right, "zeros")
    golden = torch.cat(padded, dim=halo_dim)

    # Sub-device setup
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    neighbor_sem = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)

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
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

    tt_out = ttnn.to_torch(
        ttnn.from_device(output_tensor),
        mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )

    logger.info(f"logical_h={logical_h}, input_shape={input_shape}, output shape={tt_out.shape}")
    eq, msg = comp_equal(tt_out, golden)
    assert eq, f"logical_h masking mismatch: {msg}"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.timeout(120)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "input_shape, halo_dim, other_shard_dim, cluster_axis, padding_left, padding_right, logical_h, num_links",
    [
        # 5D [B, T, H, W, C]: H=92 (23*4) sharded across 4 H-devices, logical_h=90
        # Device 3 has rows 69..91; rows 90-91 (local rows 21-22) should be zeroed.
        ([1, 3, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 90, 2),
        # Same shape but logical_h aligns exactly to shard boundary (no masking on any device)
        ([1, 3, 23 * 4, 20 * 8, 32], 2, 3, 0, 1, 1, 23 * 4, 2),
        # Larger channel dim, more padding rows masked
        ([1, 3, 24 * 4, 20 * 8, 96], 2, 3, 0, 1, 1, 90, 2),
    ],
    ids=[
        "vae_720p_h90_of_92",
        "vae_no_masking",
        "vae_large_c_h90_of_96",
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}],
    indirect=True,
)
def test_neighbor_pad_logical_h(
    mesh_device,
    input_shape,
    halo_dim,
    other_shard_dim,
    cluster_axis,
    padding_left,
    padding_right,
    logical_h,
    num_links,
    device_params,
):
    if is_blackhole() and num_links > 2:
        pytest.skip("Skipping num_links > 2 on BH")

    run_neighbor_pad_logical_h_impl(
        mesh_device,
        input_shape=list(input_shape),
        halo_dim=halo_dim,
        other_shard_dim=other_shard_dim,
        cluster_axis=cluster_axis,
        padding_left=padding_left,
        padding_right=padding_right,
        logical_h=logical_h,
        num_links=num_links,
    )
