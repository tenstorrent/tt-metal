# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
#  SPDX-License-Identifier: Apache-2.0

"""All-gather tests for multi-galaxy exabox mesh configurations (DUAL_BH / QUAD_BH).

Tests cover:
- Sync API (ttnn.all_gather) on 16x4 and 32x4 meshes
- Async API (ttnn.experimental.all_gather_async) on 32x4 mesh
- Model-representative shapes (MLA dim=1/2, MLP dim=3)
- Multiple fabric configs (FABRIC_1D, FABRIC_1D_RING) and topologies (Linear, Ring)
- Both cluster axes (0 and 1)
- DRAM and L1 memory configs
- num_links variation (1 and 2 links for BH Galaxy)
- bfloat16, bfloat8_b, and float32 dtypes
"""

import math

import pytest
import torch

import ttnn
from tests.sweep_framework.sweep_utils.ccl_common import get_mem_configs
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.utils_for_testing import maybe_trace

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_tensors(input_shape, mesh_shape, dim, cluster_axis, buffer_type, dtype, layout, device):
    num_devices = math.prod(mesh_shape)
    replicate = mesh_shape[cluster_axis]
    torch.manual_seed(0)
    if dtype == ttnn.float32:
        torch_input = torch.cat([torch.rand(input_shape).float() for _ in range(replicate)], dim=dim)
    else:
        torch_input = torch.cat([torch.rand(input_shape).bfloat16() for _ in range(replicate)], dim=dim)

    input_memory_config, output_memory_config = get_mem_configs(buffer_type, None, layout, torch_input.shape)

    shard_dims = (dim, None) if cluster_axis == 0 else (None, dim)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=dtype,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=shard_dims, mesh_shape=mesh_shape),
        device=device,
    )

    torch_reference = torch_input.repeat([num_devices] + [1] * (len(input_shape) - 1))
    return tt_input, torch_reference, output_memory_config


def _verify_all_gather_output(tt_output_tensor, torch_reference, mesh_device, pcc_threshold=0.9999):
    coords = list(tt_output_tensor.tensor_topology().mesh_coords())
    coord_to_index = {coord: idx for idx, coord in enumerate(coords)}
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    device_tensors = ttnn.get_device_tensors(tt_output_tensor)
    coord_iter = coords
    if view is not None and len(device_tensors) != len(coords):
        coord_iter = [coord for coord in coords if view.is_local(coord)]

    per_device_batch = torch_reference.shape[0] // math.prod(mesh_device.shape)
    torch_reference_slices = torch_reference.split(per_device_batch, dim=0)
    for coord, tt_out in zip(coord_iter, device_tensors):
        if view is not None and not view.is_local(coord):
            continue
        device_idx = coord_to_index[coord]
        eq, mess = comp_pcc(torch_reference_slices[device_idx], ttnn.to_torch(tt_out), pcc_threshold)
        assert eq, mess


def _run_all_gather_test(
    mesh_device,
    input_shape,
    dim,
    cluster_axis,
    buffer_type,
    dtype,
    topology,
    enable_trace,
    num_links=None,
):
    tt_input, torch_reference, output_mem_config = _get_tensors(
        input_shape, tuple(mesh_device.shape), dim, cluster_axis, buffer_type, dtype, ttnn.TILE_LAYOUT, mesh_device
    )

    all_gather_kwargs = dict(
        cluster_axis=cluster_axis,
        topology=topology,
        memory_config=output_mem_config,
    )
    if num_links is not None:
        all_gather_kwargs["num_links"] = num_links

    def run_op():
        return ttnn.all_gather(tt_input, dim, **all_gather_kwargs)

    tt_output_tensor = maybe_trace(run_op, enable_trace=enable_trace, device=mesh_device)
    _verify_all_gather_output(tt_output_tensor, torch_reference, mesh_device)


# ---------------------------------------------------------------------------
# Fabric / topology parametrize combos (reused by multiple tests)
# ---------------------------------------------------------------------------

FABRIC_TOPOLOGY_COMBOS = [
    pytest.param(
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
        ttnn.Topology.Linear,
        id="fabric_1d-linear",
    ),
    pytest.param(
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
        ttnn.Topology.Linear,
        id="fabric_1d_ring-linear",
    ),
    pytest.param(
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
        ttnn.Topology.Ring,
        id="fabric_1d_ring-ring",
    ),
]


# ---------------------------------------------------------------------------
# Test: sync all_gather on 16x4 mesh (DUAL_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["DUAL_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        pytest.param(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
            ttnn.Topology.Linear,
            id="fabric_1d-linear",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((16, 4), id="16x4_grid")], indirect=True)
@pytest.mark.parametrize("cluster_axis", [pytest.param(0, id="axis0"), pytest.param(1, id="axis1")])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2links"])
@pytest.mark.parametrize("enable_trace", [True, False])
@pytest.mark.parametrize(
    "input_shape, dtype, buffer_type",
    [
        ([1, 1, 32, 224], ttnn.bfloat16, ttnn.BufferType.DRAM),
        ([1, 1, 32, 896], ttnn.bfloat16, ttnn.BufferType.DRAM),
        ([1, 1, 32, 224], ttnn.bfloat16, ttnn.BufferType.L1),
    ],
    ids=["small_dram", "large_dram", "small_l1"],
)
def test_all_gather_16x4(
    mesh_device,
    cluster_axis,
    dim,
    topology,
    enable_trace,
    input_shape,
    dtype,
    buffer_type,
    num_links,
):
    _run_all_gather_test(
        mesh_device, input_shape, dim, cluster_axis, buffer_type, dtype, topology, enable_trace, num_links=num_links
    )


# ---------------------------------------------------------------------------
# Test: sync all_gather on 32x4 mesh (QUAD_BH)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    FABRIC_TOPOLOGY_COMBOS,
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((32, 4), id="32x4_grid")], indirect=True)
@pytest.mark.parametrize("cluster_axis", [pytest.param(0, id="axis0"), pytest.param(1, id="axis1")])
@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2links"])
@pytest.mark.parametrize("enable_trace", [True, False])
@pytest.mark.parametrize(
    "input_shape, dtype, buffer_type",
    [
        ([1, 1, 32, 224], ttnn.bfloat16, ttnn.BufferType.DRAM),
        ([1, 1, 32, 896], ttnn.bfloat16, ttnn.BufferType.DRAM),
        ([1, 1, 32, 224], ttnn.bfloat16, ttnn.BufferType.L1),
        ([1, 1, 32, 896], ttnn.bfloat16, ttnn.BufferType.L1),
        ([1, 1, 32, 896], ttnn.bfloat8_b, ttnn.BufferType.DRAM),
        ([1, 1, 32, 224], ttnn.float32, ttnn.BufferType.DRAM),
    ],
    ids=["small_dram", "large_dram", "small_l1", "large_l1", "bfloat8_dram", "float32_dram"],
)
def test_all_gather_32x4(
    mesh_device,
    cluster_axis,
    dim,
    topology,
    enable_trace,
    input_shape,
    dtype,
    buffer_type,
    num_links,
):
    _run_all_gather_test(
        mesh_device, input_shape, dim, cluster_axis, buffer_type, dtype, topology, enable_trace, num_links=num_links
    )


# ---------------------------------------------------------------------------
# Test: sync all_gather with model-representative shapes (dim=1,2,3)
# ---------------------------------------------------------------------------


@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        pytest.param(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
            ttnn.Topology.Linear,
            id="fabric_1d-linear",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((32, 4), id="32x4_grid")], indirect=True)
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2links"])
@pytest.mark.parametrize("enable_trace", [True, False])
@pytest.mark.parametrize(
    "input_shape, dim, cluster_axis, dtype, buffer_type",
    [
        # MLA wq_kv_a decode: gather on dim=1 across cols (cluster_axis=1)
        pytest.param([1, 1, 32, 2112], 1, 1, ttnn.bfloat16, ttnn.BufferType.DRAM, id="mla_wq_kv_a_dim1"),
        # MLA wo decode: gather on dim=2 across cols (cluster_axis=1)
        pytest.param([1, 4, 128, 128], 2, 1, ttnn.bfloat16, ttnn.BufferType.L1, id="mla_wo_dim2"),
        # MLP: gather on dim=3 across rows (cluster_axis=0)
        pytest.param([1, 1, 32, 896], 3, 0, ttnn.bfloat16, ttnn.BufferType.DRAM, id="mlp_dim3_axis0"),
        # MLP: gather on dim=3 across cols (cluster_axis=1)
        pytest.param([1, 1, 32, 224], 3, 1, ttnn.bfloat16, ttnn.BufferType.DRAM, id="mlp_dim3_axis1"),
    ],
)
def test_all_gather_32x4_model_shapes(
    mesh_device,
    topology,
    num_links,
    enable_trace,
    input_shape,
    dim,
    cluster_axis,
    dtype,
    buffer_type,
):
    _run_all_gather_test(
        mesh_device, input_shape, dim, cluster_axis, buffer_type, dtype, topology, enable_trace, num_links=num_links
    )


# ---------------------------------------------------------------------------
# Test: async all_gather on 32x4 mesh (QUAD_BH)
# ---------------------------------------------------------------------------


def _run_all_gather_async_test(
    mesh_device,
    input_shape,
    dim,
    cluster_axis,
    buffer_type,
    dtype,
    topology,
    num_links=None,
    num_iters=1,
):
    """Run all_gather_async with sub-device and semaphore management."""
    num_devices = math.prod(mesh_device.shape)
    replicate = mesh_device.shape[cluster_axis]
    torch.manual_seed(0)
    if dtype == ttnn.float32:
        torch_input = torch.cat([torch.rand(input_shape).float() for _ in range(replicate)], dim=dim)
    else:
        torch_input = torch.cat([torch.rand(input_shape).bfloat16() for _ in range(replicate)], dim=dim)

    input_memory_config, output_memory_config = get_mem_configs(buffer_type, None, ttnn.TILE_LAYOUT, torch_input.shape)

    shard_dims = (dim, None) if cluster_axis == 0 else (None, dim)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape),
        device=mesh_device,
    )

    torch_reference = torch_input.repeat([num_devices] + [1] * (len(input_shape) - 1))

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

    ccl_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2 * num_iters)
    ]

    try:
        for i in range(num_iters):
            async_kwargs = dict(
                multi_device_global_semaphore=[ccl_semaphore_handles[2 * i], ccl_semaphore_handles[2 * i + 1]],
                memory_config=output_memory_config,
                subdevice_id=worker_sub_device_id,
            )
            if num_links is not None:
                async_kwargs["num_links"] = num_links

            tt_output_tensor = ttnn.experimental.all_gather_async(
                tt_input, dim, cluster_axis, mesh_device, topology, **async_kwargs
            )
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

            if i < num_iters - 1:
                tt_output_tensor.deallocate(True)

        # Verify final iteration output
        _verify_all_gather_output(tt_output_tensor, torch_reference, mesh_device)
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(sub_device_manager)


@pytest.mark.requires_device(["QUAD_BH"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        pytest.param(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
            ttnn.Topology.Linear,
            id="fabric_1d-linear",
        ),
        pytest.param(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
            ttnn.Topology.Ring,
            id="fabric_1d_ring-ring",
        ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((32, 4), id="32x4_grid")], indirect=True)
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2links"])
@pytest.mark.parametrize(
    "input_shape, dim, cluster_axis, dtype, buffer_type",
    [
        # MLA wq_kv_a decode: gather on dim=1 across cols
        pytest.param([1, 1, 32, 2112], 1, 1, ttnn.bfloat16, ttnn.BufferType.L1, id="mla_wq_kv_a"),
        # MLP: gather on dim=3 across cols
        pytest.param([1, 1, 32, 896], 3, 1, ttnn.bfloat16, ttnn.BufferType.DRAM, id="mlp_dim3"),
        # Embedding: gather on dim=3 across rows (cluster_axis=0)
        pytest.param([1, 1, 32, 224], 3, 0, ttnn.bfloat16, ttnn.BufferType.DRAM, id="embedding_dim3_axis0"),
    ],
)
def test_all_gather_async_32x4(
    mesh_device,
    topology,
    num_links,
    input_shape,
    dim,
    cluster_axis,
    dtype,
    buffer_type,
):
    _run_all_gather_async_test(
        mesh_device,
        input_shape,
        dim,
        cluster_axis,
        buffer_type,
        dtype,
        topology,
        num_links=num_links,
        num_iters=1,
    )
