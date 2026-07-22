# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl
from models.common.utility_functions import comp_pcc, skip_for_blackhole, skip_for_wormhole_b0


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("num_links", [4], ids=["4links"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout, rs_input_dtype, enable_trace, num_iters,cluster_axis",
    [
        # Perf variants (with tracing)
        (8, [8, 1, 512, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10, 0),  # use batching when fused
        (8, [4, 1, 1024, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10, 0),  # use batching when fused
        (8, [1, 1, 32, 4096], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10, 0),  # use batching when fused
        (8, [1, 1, 32, 1536], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10, 0),  # from CSV
        # Check variants (without tracing)
        (4, [1, 1, 333, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1, 1),  # use batching when fused
        (8, [2, 1, 2048, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1, 0),  # use batching when fused
        (8, [1, 1, 32, 4096], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1, 0),  # use batching when fused
        (8, [1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1, 0),  # from CSV
    ],
    ids=[
        "batch_8-perf",
        "batch_4-perf",
        "batch_1-perf",
        "deepseek_1-perf",
        "batch_1_sd35_prompt-check",
        "batch_2-check",
        "batch_1-check",
        "deepseek_2-check",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
        ),
        # (
        #    {
        #        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        #        "fabric_manager": ttnn.FabricManagerMode.ENABLED,
        #        "trace_region_size": 90112,
        #    },
        #    ttnn.Topology.Linear,
        # ),
    ],
    indirect=["device_params"],
    ids=[
        "fabric_ring",
        # "fabric_manager_enabled_linear" # test removed due to issue 35320
    ],
)
@pytest.mark.parametrize("chunks_per_sync", [1])
@pytest.mark.parametrize("num_workers_per_link", [1])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_reduce_scatter_async(
    mesh_device,
    num_devices,
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    num_iters,
    rs_topology,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
    cluster_axis,
):
    if cluster_axis == 0:
        submesh_device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    else:
        submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
    run_reduce_scatter_impl(
        submesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("num_links", [1], ids=["1links"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout, rs_input_dtype",
    [
        (8, [1, 1, 8, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
    ],
    ids=[
        "deepseek_like",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (False, 1),
    ],
    ids=["check"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
                "trace_region_size": 90112,
            },
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
def test_reduce_scatter_async_big_mesh(
    mesh_device,
    num_devices,
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    num_iters,
    rs_topology,
):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
    cluster_axis = None
    run_reduce_scatter_impl(
        submesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("num_links", [4], ids=["4links"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout, rs_input_dtype",
    [
        (16, [1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
    ],
    ids=[
        "deepseek_4host",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (False, 1),
    ],
    ids=["check"],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@pytest.mark.parametrize("mesh_device", [(8, 16)], indirect=True)
def test_reduce_scatter_async_quad_host_mesh(
    mesh_device,
    num_devices,
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    num_iters,
    rs_topology,
):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
    cluster_axis = None
    run_reduce_scatter_impl(
        submesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("use_barrier_semaphore", [False, True], ids=["no_barrier_sem", "barrier_sem"])
def test_reduce_scatter_on_reshaped_submesh_linear(
    *, mesh_device: ttnn.MeshDevice, use_barrier_semaphore: bool
) -> None:
    # Regression for the line reduce-scatter device-side barrier on a submesh whose logical line is
    # NOT a contiguous physical line: a 2x2 block reshaped to 1x4. The barrier must use only 1-hop
    # per-link handshakes (not a multi-hop line-multicast), otherwise the barrier_sem path hangs.
    submesh = mesh_device.create_submesh(ttnn.MeshShape(2, 2))
    submesh.reshape(ttnn.MeshShape(1, 4))

    compute_grid = submesh.compute_with_storage_grid_size()
    ccl_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    rs_semaphores = [ttnn.create_global_semaphore(submesh, ccl_cores, 0) for _ in range(3)]
    barrier_semaphore = ttnn.create_global_semaphore(submesh, ccl_cores, 0)

    torch_x = torch.randn(1, 1, 64, 1024, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        torch_x,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(submesh),
    )

    if not use_barrier_semaphore:
        ttnn.synchronize_device(submesh)

    tt_out = ttnn.experimental.reduce_scatter_minimal_async(
        x,
        dim=3,
        cluster_axis=1,
        num_links=1,
        topology=ttnn.Topology.Linear,
        multi_device_global_semaphore=rs_semaphores,
        barrier_semaphore=barrier_semaphore if use_barrier_semaphore else None,
    )

    ttnn.synchronize_device(submesh)

    # Input is replicated on every device, so reduce (sum over the cluster axis) == num_devices * torch_x,
    # then scattered along dim=3. ConcatMeshToTensor reassembles the per-device shards into the full tensor.
    num_devices = submesh.get_num_devices()
    torch_out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=3))
    golden = num_devices * torch_x.float()

    passing, pcc = comp_pcc(golden, torch_out)
    assert passing, f"PCC check failed (cluster_axis=1, dim=3, num_devices={num_devices}): {pcc}"


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("mesh_device", [(8, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("use_barrier_semaphore", [False, True], ids=["no_barrier_sem", "barrier_sem"])
def test_reduce_scatter_on_reshaped_submesh_ring(*, mesh_device: ttnn.MeshDevice, use_barrier_semaphore: bool) -> None:
    # Regression for the line reduce-scatter device-side barrier on a submesh whose logical line is
    # NOT a contiguous physical line: a 2x2 block reshaped to 1x4. The barrier must use only 1-hop
    # per-link handshakes (not a multi-hop line-multicast), otherwise the barrier_sem path hangs.
    submesh = mesh_device.create_submesh(ttnn.MeshShape(2, 2))
    submesh.reshape(ttnn.MeshShape(1, 4))

    compute_grid = submesh.compute_with_storage_grid_size()
    ccl_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    rs_semaphores = [ttnn.create_global_semaphore(submesh, ccl_cores, 0) for _ in range(3)]
    barrier_semaphore = ttnn.create_global_semaphore(submesh, ccl_cores, 0)

    torch_x = torch.randn(1, 1, 64, 1024, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        torch_x,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(submesh),
    )

    if not use_barrier_semaphore:
        ttnn.synchronize_device(submesh)

    tt_out = ttnn.experimental.reduce_scatter_minimal_async(
        x,
        dim=3,
        cluster_axis=None,
        num_links=1,
        topology=ttnn.Topology.Ring,
        multi_device_global_semaphore=rs_semaphores,
        barrier_semaphore=barrier_semaphore if use_barrier_semaphore else None,
    )

    ttnn.synchronize_device(submesh)

    # Input is replicated on every device, so reduce (sum over the cluster axis) == num_devices * torch_x,
    # then scattered along dim=3. ConcatMeshToTensor reassembles the per-device shards into the full tensor.
    num_devices = submesh.get_num_devices()
    torch_out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=3))
    golden = num_devices * torch_x.float()

    passing, pcc = comp_pcc(golden, torch_out)
    assert passing, f"PCC check failed (cluster_axis=1, dim=3, num_devices={num_devices}): {pcc}"
