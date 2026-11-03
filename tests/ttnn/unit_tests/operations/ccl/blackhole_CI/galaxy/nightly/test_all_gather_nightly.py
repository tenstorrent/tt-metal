# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.nightly.t3000.ccl.test_all_gather import run_all_gather_impl
from tt_py_test_utils_common.utility_functions import skip_for_wormhole_b0, skip_for_n_or_less_dev


def ti_cond_skip(condition, reason):
    if condition:
        pytest.skip("Skipping unsupported case: " + reason)


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout",
    [
        (2, [1, 1, 256, 256], 3, ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.bfloat16,
        ttnn.uint32,
        ttnn.bfloat8_b,
    ],
    ids=[
        "float_16",
        "uint_32",
        "bfloat_8",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 3),
        (False, 3),
    ],
    ids=["trace", "non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
    ],
    indirect=True,
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_all_gather_linear_2D_nightly(
    bh_2d_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
    cluster_axis,
):
    if cluster_axis == 0:
        submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    else:
        submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout",
    [
        (4, [1, 1, 128, 2048], 3, ttnn.TILE_LAYOUT),
        (4, [1, 1, 256, 768], 3, ttnn.TILE_LAYOUT),
    ],
    ids=["4_device_test", "4_device_large_test"],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.uint32,
        ttnn.bfloat8_b,
    ],
    ids=[
        "uint_32",
        "bfloat_8",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 3),
        (False, 3),
    ],
    ids=["trace", "non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
    ],
    indirect=True,
)
@pytest.mark.parametrize("cluster_axis", [0])
def test_all_gather_linear_4D_nightly(
    bh_2d_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
    cluster_axis,
):
    if cluster_axis == 0:
        submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    else:
        submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(2)
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout",
    [
        ([1, 1, 128, 2048], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 256, 512], 3, ttnn.TILE_LAYOUT),
        ([1, 1, 256, 768], 3, ttnn.TILE_LAYOUT),
    ],
    ids=["4_device_test", "2_device_test", "4_device_large_test"],
)
@pytest.mark.parametrize(
    "ag_input_dtype",
    [
        ttnn.bfloat16,
    ],
    ids=[
        "float_16",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 3),
        (False, 3),
    ],
    ids=["trace", "non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
    ],
    indirect=True,
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_all_gather_ring_nightly(
    bh_2d_mesh_device,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
    cluster_axis,
):
    num_devices = bh_2d_mesh_device.shape[cluster_axis]
    if cluster_axis == 0:
        submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    else:
        submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, num_devices)))
    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


# MiniMax-H3 DiT gathers on the 4x8 Galaxy with the row count as a runtime word: one program per
# gather serves every served per-chip length (74..470 tile rows).
def _h3_tile_rows():
    return [148, 74, 287, 427, 470]


def _h3_all_gather_sweep(mesh_device, dim, cluster_axis, width, rows_list, num_workers_per_link, miss_indices=()):
    """Gather [1,1,rows,width] per chip along `dim` over `cluster_axis`, alternating two semaphore sets."""
    import torch
    from tests.ttnn.utils_for_testing import assert_program_cache_flat

    torch.manual_seed(0)
    mesh_shape = tuple(mesh_device.shape)
    ring = mesh_shape[cluster_axis]
    compute_grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    semaphore_sets = [[ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(2)] for _ in range(2)]
    barriers = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(2)]
    calls = [0]

    def run(variant):
        rows, w = variant
        full_shape = [1, 1, rows * 32, w]
        full_shape[dim] *= ring
        torch_full = torch.randn(full_shape, dtype=torch.bfloat16)
        shard_dims = (None, dim) if cluster_axis == 1 else (dim, None)
        # Tilize on the host: a device-side tilize would add a program per new shape.
        tt_host = ttnn.from_torch(
            torch_full,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )
        tt_input = ttnn.to_device(tt_host, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        i = calls[0] % 2
        calls[0] += 1
        tt_output = ttnn.experimental.all_gather_async(
            tt_input,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=semaphore_sets[i],
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            cluster_axis=cluster_axis,
            barrier_semaphore=barriers[i],
            num_workers_per_link=num_workers_per_link,
        )
        ttnn.synchronize_device(mesh_device)
        for coord, tt_out in zip(tt_output.tensor_topology().mesh_coords(), ttnn.get_device_tensors(tt_output)):
            assert torch.equal(ttnn.to_torch(tt_out), torch_full), f"device {coord} mismatch at rows={rows} w={w}"

    variants = [(rows, width) for rows in rows_list]
    if miss_indices:
        variants.append((rows_list[0], width * 2))
    mesh_device.disable_and_clear_program_cache()
    mesh_device.enable_program_cache()
    assert_program_cache_flat(mesh_device, run, variants, miss_indices=miss_indices)


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(31)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize(
    "dim, cluster_axis, width",
    [(2, 1, 96), (2, 1, 32), (3, 0, 1344)],
    ids=["sp_video_dim2_w96", "sp_audio_dim2_w32", "tp_hidden_dim3_w1344"],
)
@pytest.mark.parametrize("num_workers_per_link", [4, None], ids=["workers4", "workers_default"])
def test_all_gather_h3_dynamic_rows_flat_cache(bh_2d_mesh_device, dim, cluster_axis, width, num_workers_per_link):
    _h3_all_gather_sweep(bh_2d_mesh_device, dim, cluster_axis, width, _h3_tile_rows(), num_workers_per_link)
    assert bh_2d_mesh_device.num_program_cache_entries() == 1


@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(31)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
def test_all_gather_h3_width_change_misses(bh_2d_mesh_device):
    """Only the row count is runtime: a new width adds a program."""
    _h3_all_gather_sweep(bh_2d_mesh_device, 2, 1, 96, [148, 74], 4, miss_indices=(2,))
