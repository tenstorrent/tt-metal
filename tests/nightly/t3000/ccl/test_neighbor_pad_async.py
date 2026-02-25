# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc

from ttnn import ShardTensor2dMesh, ConcatMesh2dToTensor


# ---------------------------------------------------------------------------
# 1D neighbor pad
# ---------------------------------------------------------------------------


def run_neighbor_pad_1d_impl(
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
):
    torch.manual_seed(0)

    ##### All gather setup #####
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    neighbor_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]
    barrier_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    ##### Neighbor pad input setup #####
    logger.info(f"Neighbor pad input shape: {input_shape}")
    logger.info(f"Neighbor pad dim: {halo_shard_dim}")

    input_tensor_mesh_list = []
    np_output_tensor_goldens_list = []

    # Make sure input shape is padded
    input_shape[halo_shard_dim] = (
        math.ceil(input_shape[halo_shard_dim] / mesh_device.shape[cluster_axis]) * mesh_device.shape[cluster_axis]
    )
    input_shape[other_shard_dim] = (
        math.ceil(input_shape[other_shard_dim] / mesh_device.shape[1 - cluster_axis])
        * mesh_device.shape[1 - cluster_axis]
    )

    for i in range(num_iters):
        input_tensor = torch.rand(input_shape).bfloat16()
        num_chunks = mesh_device.shape[cluster_axis]
        chunks = torch.chunk(input_tensor, num_chunks, halo_shard_dim)
        np_output_tensor = []
        # pad left
        if padding_mode == "zeros":
            slice_shape = list(chunks[0].shape)
            slice_shape[halo_shard_dim] = 1
            first_slice_front = torch.zeros(slice_shape)
        else:
            first_slice_front = torch.narrow(chunks[0], halo_shard_dim, 0, 1)
        first_slice = torch.cat((first_slice_front, chunks[0]), dim=halo_shard_dim)
        np_output_tensor.append(first_slice)
        for p in range(padding_left - 1):
            np_output_tensor[0] = torch.cat((first_slice_front, np_output_tensor[0]), dim=halo_shard_dim)
        for k in range(1, num_chunks):
            prev_halo = torch.narrow(
                chunks[k - 1], halo_shard_dim, chunks[k - 1].shape[halo_shard_dim] - padding_left, padding_left
            )
            np_output_tensor.append(torch.cat((prev_halo, chunks[k]), dim=halo_shard_dim))

        # pad right
        if padding_mode == "zeros":
            slice_shape = list(np_output_tensor[num_chunks - 1].shape)
            slice_shape[halo_shard_dim] = 1
            last_slice_back = torch.zeros(slice_shape)
        else:
            last_slice_size = np_output_tensor[num_chunks - 1].shape[halo_shard_dim]
            last_slice_back = torch.narrow(np_output_tensor[num_chunks - 1], halo_shard_dim, last_slice_size - 1, 1)
        for p in range(padding_right):
            np_output_tensor[num_chunks - 1] = torch.cat(
                (np_output_tensor[num_chunks - 1], last_slice_back), dim=halo_shard_dim
            )
        for k in range(0, num_chunks - 1):
            next_halo = torch.narrow(chunks[k + 1], halo_shard_dim, 0, padding_right)
            np_output_tensor[k] = torch.cat((np_output_tensor[k], next_halo), dim=halo_shard_dim)
        np_output_tensor_goldens_list.append(torch.cat(np_output_tensor, dim=halo_shard_dim))

        dims = [None, None]
        dims[cluster_axis] = halo_shard_dim
        dims[1 - cluster_axis] = other_shard_dim
        input_tensor_mesh = ttnn.from_torch(
            input_tensor,
            device=mesh_device,
            layout=layout,
            dtype=input_dtype,
            memory_config=mem_config_input,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        )

        input_tensor_mesh_list.append(input_tensor_mesh)

    ##### Perform the TT ops #####
    tt_neighbor_pad_out_tensor_list = []

    def run_op(i):
        tt_neighbor_pad_out_tensor = ttnn.experimental.neighbor_pad_async(
            input_tensor_mesh_list[i],
            [halo_shard_dim],
            [padding_left],
            [padding_right],
            padding_mode,
            [cluster_axis],
            [neighbor_semaphore_handles[i]],
            [barrier_semaphore_handles[i]],
            num_links=[num_links],
            memory_config=mem_config_output,
            topology=neighbor_pad_topology,
        )

        return tt_neighbor_pad_out_tensor

    if enable_trace:
        # Compile the op
        tt_neighbor_pad_out_tensor = run_op(0)
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_neighbor_pad_out_tensor = run_op(0)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done capturing trace")

        # Execute trace
        for i in range(num_iters):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            tt_neighbor_pad_out_tensor_list.append(tt_neighbor_pad_out_tensor)
        logger.info(f"Done executing trace")
    else:
        for i in range(num_iters):
            tt_neighbor_pad_out_tensor = run_op(i)
            tt_neighbor_pad_out_tensor_list.append(tt_neighbor_pad_out_tensor)
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done iteration {i}")

    for i in range(num_iters):
        tt_np_out_tensor = tt_neighbor_pad_out_tensor_list[i]
        torch_np_out_tensor = np_output_tensor_goldens_list[i if not enable_trace else 0]
        tt_np_out = ttnn.from_device(tt_np_out_tensor)
        dims[cluster_axis] = halo_shard_dim
        dims[1 - cluster_axis] = other_shard_dim
        tt_np_out = ttnn.to_torch(
            tt_np_out,
            mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        )
        eq, output = comp_pcc(tt_np_out, torch_np_out_tensor, 1)
        logger.info(f"{output}, iteration {i}")
        assert eq, f"{i} FAILED np: {output}"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


# ---------------------------------------------------------------------------
# 2D neighbor pad
# ---------------------------------------------------------------------------


def compute_2d_pad_golden(input_tensor, mesh_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, padding_mode="zeros"):
    """
    Compute per-device golden output for 2D neighbor pad.
    Pads h_dim along h_axis, then w_dim along w_axis.
    Returns dict mapping (mesh_row, mesh_col) -> expected per-device tensor.
    """
    assert padding_mode == "zeros", f"2D golden only supports zeros mode, got {padding_mode}"
    h_factor = mesh_shape[h_axis]
    w_factor = mesh_shape[w_axis]

    # Step 1: Chunk along H, pad H boundaries
    h_chunks = list(torch.chunk(input_tensor, h_factor, dim=h_dim))
    h_padded = []
    for i in range(h_factor):
        chunk = h_chunks[i]
        left_parts = []
        for p in range(pH):
            if i == 0:
                left_parts.append(torch.zeros_like(torch.narrow(chunk, h_dim, 0, 1)))
            else:
                left_parts.append(torch.narrow(h_chunks[i - 1], h_dim, h_chunks[i - 1].shape[h_dim] - pH + p, 1))
        right_parts = []
        for p in range(pH):
            if i == h_factor - 1:
                right_parts.append(torch.zeros_like(torch.narrow(chunk, h_dim, 0, 1)))
            else:
                right_parts.append(torch.narrow(h_chunks[i + 1], h_dim, p, 1))
        h_padded.append(torch.cat(left_parts + [chunk] + right_parts, dim=h_dim))

    # Step 2: For each H chunk, chunk along W and pad W boundaries
    goldens = {}
    for h_idx in range(h_factor):
        w_chunks = list(torch.chunk(h_padded[h_idx], w_factor, dim=w_dim))
        for w_idx in range(w_factor):
            chunk = w_chunks[w_idx]
            left_parts = []
            for p in range(pW):
                if w_idx == 0:
                    left_parts.append(torch.zeros_like(torch.narrow(chunk, w_dim, 0, 1)))
                else:
                    left_parts.append(
                        torch.narrow(w_chunks[w_idx - 1], w_dim, w_chunks[w_idx - 1].shape[w_dim] - pW + p, 1)
                    )
            right_parts = []
            for p in range(pW):
                if w_idx == w_factor - 1:
                    right_parts.append(torch.zeros_like(torch.narrow(chunk, w_dim, 0, 1)))
                else:
                    right_parts.append(torch.narrow(w_chunks[w_idx + 1], w_dim, p, 1))
            key = (h_idx, w_idx) if h_axis == 0 else (w_idx, h_idx)
            goldens[key] = torch.cat(left_parts + [chunk] + right_parts, dim=w_dim)

    return goldens


def run_neighbor_pad_2d_impl(
    mesh_device,
    input_shape,
    h_dim,
    w_dim,
    h_axis,
    w_axis,
    pH,
    pW,
    padding_mode,
    input_dtype,
    topology,
):
    """Run fused 2D neighbor pad and compare per-device against golden."""
    mesh_shape = tuple(mesh_device.shape)
    h_factor = mesh_shape[h_axis]
    w_factor = mesh_shape[w_axis]

    assert input_shape[h_dim] % h_factor == 0, f"H dim {input_shape[h_dim]} not divisible by {h_factor}"
    assert input_shape[w_dim] % w_factor == 0, f"W dim {input_shape[w_dim]} not divisible by {w_factor}"

    torch.manual_seed(42)
    input_tensor = torch.rand(input_shape).bfloat16()
    goldens = compute_2d_pad_golden(input_tensor, mesh_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, padding_mode)

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

    # Semaphores
    h_neighbor_sem = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)
    w_neighbor_sem = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)

    # Shard input to mesh
    dims = [None, None]
    dims[h_axis] = h_dim
    dims[w_axis] = w_dim
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    input_tensor_mesh = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=input_dtype,
        memory_config=mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )

    # Run fused 2D neighbor pad
    output_tensor = ttnn.experimental.neighbor_pad_async(
        input_tensor_mesh,
        [h_dim, w_dim],
        [pH, pW],
        [pH, pW],
        padding_mode,
        [h_axis, w_axis],
        [h_neighbor_sem, w_neighbor_sem],
        [barrier_sem],
        num_links=[1, 1],
        memory_config=mem_config,
        topology=topology,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

    # Compare per-device
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
            eq, output = comp_equal(dev_tensor, golden)
            assert eq, f"Device ({row},{col}): {output}"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


# ---------------------------------------------------------------------------
# 1D tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "input_shape, halo_shard_dim, other_shard_dim, layout, input_dtype, padding_left, padding_right, padding_mode, cluster_axis, enable_trace, num_iters",
    [
        ([28, 60, 106, 768], 0, 2, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 0, "replicate", 1, True, 10),  # perf
        ([82, 120, 212, 512], 0, 2, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 0, "replicate", 1, False, 1),  # check
        ([28, 60, 106, 768], 2, 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 2, "replicate", 1, True, 10),  # perf
        ([28, 60, 106, 768], 2, 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, 2, 2, "zeros", 1, False, 1),  # check
    ],
    ids=[
        "mochi_vae_1-perf",
        "mochi_vae_2-check",
        "replicate_width_dim-perf",
        "zeros_width_dim-check",
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
):
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
    )


# ---------------------------------------------------------------------------
# 2D tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
@pytest.mark.parametrize("mesh_device", [(2, 4), (4, 8)], ids=["2x4", "4x8"], indirect=True)
@pytest.mark.parametrize(
    "input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW",
    [
        # 5D: [B, T, H, W, C] — H along axis 0, W along axis 1
        ([1, 2, 8, 16, 32], 2, 3, 0, 1, 1, 1),
        ([1, 3, 10, 16, 32], 2, 3, 0, 1, 1, 1),
        # VAE conv_0 shape (full H=90, W=160)
        ([1, 3, 90, 160, 32], 2, 3, 0, 1, 1, 1),
        # Flipped axes: H along axis 1, W along axis 0
        ([1, 2, 16, 8, 32], 2, 3, 1, 0, 1, 1),
        # 4D tensor [B, H, W, C]
        ([2, 8, 16, 32], 1, 2, 0, 1, 1, 1),
    ],
    ids=[
        "small_5d_h0w1",
        "medium_5d_h0w1",
        "vae_conv0_h0w1",
        "small_5d_h1w0",
        "small_4d_h0w1",
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
    device_params,
):
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
        input_dtype=ttnn.bfloat16,
        topology=ttnn.Topology.Linear,
    )
