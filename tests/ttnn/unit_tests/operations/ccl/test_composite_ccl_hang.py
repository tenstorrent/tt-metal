# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def _setup_device(mesh_device):
    # Set up sub-devices for async operation
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])
    return ccl_sub_device_crs, worker_sub_device_id


def _setup_l1_overwrite(mesh_device):
    tile_height = 32
    tile_width = 32
    element_size = 4  # int32

    mv = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    nbytes = mv.largest_contiguous_bytes_free_per_bank
    nbanks = mv.num_banks
    ntiles = nbytes // (tile_height * tile_width * element_size)

    def run_l1_overwrite():
        t = ttnn.allocate_tensor_on_device(
            shape=ttnn.Shape((tile_height, tile_width * ntiles * nbanks)),
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.mul_(t, 0)
        ttnn.add_(t, 1000)

    while True:
        try:
            run_l1_overwrite()
            ttnn.synchronize_device(mesh_device)
            break
        except RuntimeError:
            ntiles -= 1
            if ntiles <= 0:
                pytest.fail("Cannot allocate any L1 tiles for overwrite op")

    return run_l1_overwrite


def _trace_corruption_and_rerun(mesh_device, run_ccl_op, run_l1_overwrite):
    run_l1_overwrite()
    ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device)
    run_l1_overwrite()
    ttnn.end_trace_capture(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    run_ccl_op()
    ttnn.synchronize_device(mesh_device)

    ttnn.execute_trace(mesh_device, trace_id)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    out = run_ccl_op()
    ttnn.synchronize_device(mesh_device)
    return out


def _make_sem(mesh_device, cores, buffer_type, initial_value=0):
    return ttnn.create_global_semaphore(mesh_device, cores, initial_value, buffer_type)


def _cluster_axis_with_more_than_one_device(mesh_device):
    shape = mesh_device.shape
    if hasattr(shape, "dims") and shape.dims() >= 2:
        if shape[0] == 1 and shape[1] > 1:
            return 1
        return 0
    try:
        return 1 if len(shape) >= 2 and shape[0] == 1 and shape[1] > 1 else 0
    except TypeError:
        return 0


# External semaphores are intentionally allocated in L1.
# This test reproduces the real L1 trace-overwrite scenario while verifying that
# explicitly passed semaphores are correctly forwarded in composite paths.
SEM_BUFFER_TYPE = ttnn.BufferType.L1


def _check_mesh_output(result, golden, opname):
    for i, t in enumerate(ttnn.get_device_tensors(result)):
        tt_out = ttnn.to_torch(t)
        eq, msg = comp_pcc(tt_out, golden)
        assert eq, f"{opname} correctness failed on device {i}: {msg}"


# Covers every composite all_gather_async forwarding path:
#   - subcore                : wrapper_sub_core_grids (FABRIC_1D, (1,8))
#   - tiled_padded           : use_composite_all_gather TILE+padding (FABRIC_1D, (1,8))
#   - persistent_2d / mesh_2d: wrapper_persistent_buffer / wrapper_mesh_device on FABRIC_2D (2,4)
#   - persistent_1d_* / mesh_1d_*: same two wrappers on FABRIC_1D (1,8), with row_major and tile_padded triggers
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "device_params, mesh_device, case",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, (1, 8), "subcore"),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, (1, 8), "tiled_padded"),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_2D}, (2, 4), "persistent_2d"),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_2D}, (2, 4), "mesh_2d"),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, (1, 8), "persistent_1d_rm"),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, (1, 8), "mesh_1d_rm"),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, (1, 8), "persistent_1d_tp"),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, (1, 8), "mesh_1d_tp"),
    ],
    ids=[
        "subcore",
        "tiled_padded",
        "persistent_2d",
        "mesh_2d",
        "persistent_1d_rm",
        "mesh_1d_rm",
        "persistent_1d_tp",
        "mesh_1d_tp",
    ],
    indirect=["device_params", "mesh_device"],
)
def test_composite_all_gather_hang(mesh_device, case):
    logger.info(f"all_gather[{case}]: a hang during execution means this test failed")
    torch.manual_seed(42)
    ccl_sub_device_crs, worker_sub_device_id = _setup_device(mesh_device)
    run_l1_overwrite = _setup_l1_overwrite(mesh_device)

    semaphores = [_make_sem(mesh_device, ccl_sub_device_crs, SEM_BUFFER_TYPE) for _ in range(2)]
    barrier_sem = _make_sem(mesh_device, ccl_sub_device_crs, SEM_BUFFER_TYPE)

    use_persistent = case.startswith("persistent_")
    use_mesh_overload = case.startswith("mesh_")
    use_cluster_axis = use_persistent or use_mesh_overload

    if case == "subcore":
        num_gather_devices = mesh_device.get_num_devices()
        per_device_width = 64
        golden = torch.randn(32, per_device_width * num_gather_devices).bfloat16()
        ag_input = ttnn.from_torch(
            golden,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )
        dim = 1
    elif case == "tiled_padded":
        num_gather_devices = mesh_device.get_num_devices()
        per_device_rows = 33  # not tile-aligned on gather dim 0
        golden = torch.randn(per_device_rows * num_gather_devices, 64).bfloat16()
        ag_input = ttnn.from_torch(
            golden,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        dim = 0
    elif case in ("persistent_2d", "mesh_2d"):
        num_gather_devices = mesh_device.shape[0]
        per_device_width = 128
        golden = torch.randn(32, per_device_width * num_gather_devices).bfloat16()
        ag_input = ttnn.from_torch(
            golden,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(2, 4), dims=(1, None)),
        )
        dim = 1
    else:  # persistent_1d_rm, mesh_1d_rm, persistent_1d_tp, mesh_1d_tp
        num_gather_devices = mesh_device.get_num_devices()
        if case.endswith("_rm"):
            per_device_extent = 64
            dim = 1
            layout = ttnn.ROW_MAJOR_LAYOUT
            golden = torch.randn(32, per_device_extent * num_gather_devices).bfloat16()
            mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=1)
        else:  # _tp
            per_device_extent = 33
            dim = 0
            layout = ttnn.TILE_LAYOUT
            golden = torch.randn(per_device_extent * num_gather_devices, 64).bfloat16()
            mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        ag_input = ttnn.from_torch(
            golden,
            device=mesh_device,
            layout=layout,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

    cluster_axis = _cluster_axis_with_more_than_one_device(mesh_device) if use_cluster_axis else None

    def run_op():
        if use_persistent:
            return ttnn.experimental.all_gather_async(
                input_tensor=ag_input,
                persistent_output_buffer=None,
                dim=dim,
                multi_device_global_semaphore=semaphores,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
                barrier_semaphore=barrier_sem,
            )
        if use_mesh_overload:
            return ttnn.experimental.all_gather_async(
                input_tensor=ag_input,
                dim=dim,
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                topology=ttnn.Topology.Linear,
                multi_device_global_semaphore=semaphores,
                persistent_output_tensor=None,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                subdevice_id=worker_sub_device_id,
                barrier_semaphore=barrier_sem,
            )
        return ttnn.experimental.all_gather_async(
            input_tensor=ag_input,
            dim=dim,
            multi_device_global_semaphore=semaphores,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            subdevice_id=worker_sub_device_id,
            barrier_semaphore=barrier_sem,
        )

    result = _trace_corruption_and_rerun(mesh_device, run_op, run_l1_overwrite)
    _check_mesh_output(result, golden, f"all_gather_{case}")


# Covers both all_reduce_async composite overloads:
#   - 2D: mesh_device + cluster_axis path on FABRIC_2D (2,4)
#   - 1D: num_devices path on FABRIC_1D (1,8)
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "device_params, mesh_device, case",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_2D}, (2, 4), "2D"),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, (1, 8), "1D"),
    ],
    ids=["2D", "1D"],
    indirect=["device_params", "mesh_device"],
)
def test_composite_all_reduce_hang(mesh_device, case):
    logger.info(f"all_reduce[{case}]: a hang during execution means this test failed")
    torch.manual_seed(42)
    ccl_sub_device_crs, worker_sub_device_id = _setup_device(mesh_device)
    run_l1_overwrite = _setup_l1_overwrite(mesh_device)

    rs_sems = [_make_sem(mesh_device, ccl_sub_device_crs, SEM_BUFFER_TYPE) for _ in range(3)]
    ag_sems = [_make_sem(mesh_device, ccl_sub_device_crs, SEM_BUFFER_TYPE) for _ in range(2)]
    barrier_sems = [_make_sem(mesh_device, ccl_sub_device_crs, SEM_BUFFER_TYPE) for _ in range(2)]

    if case == "2D":
        per_chip_shape = [32, 256]
        num_reduce_devices = mesh_device.shape[0]
        inputs = [torch.randn(per_chip_shape).bfloat16() for _ in range(num_reduce_devices)]
        golden = sum(inputs)
        full_input = torch.cat(inputs, dim=0)
        input_mesh = ttnn.from_torch(
            full_input,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(2, 4), dims=(0, None)),
        )

        def run_op():
            return ttnn.experimental.all_reduce_async(
                input_mesh,
                cluster_axis=0,
                mesh_device=mesh_device,
                barrier_semaphores=barrier_sems,
                rs_global_semaphores=rs_sems,
                ag_global_semaphores=ag_sems,
                math_op=ttnn.ReduceType.Sum,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
                subdevice_id=worker_sub_device_id,
            )

    else:  # 1D
        num_devices = mesh_device.get_num_devices()
        input_single = torch.randn(32, 256).bfloat16()
        golden = input_single * num_devices
        input_mesh = ttnn.from_torch(
            input_single,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        def run_op():
            return ttnn.experimental.all_reduce_async(
                input_tensor=input_mesh,
                num_devices=num_devices,
                barrier_semaphores=barrier_sems,
                rs_global_semaphores=rs_sems,
                ag_global_semaphores=ag_sems,
                math_op=ttnn.ReduceType.Sum,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
                num_links=1,
                subdevice_id=None,
            )

    result = _trace_corruption_and_rerun(mesh_device, run_op, run_l1_overwrite)
    _check_mesh_output(result, golden, f"all_reduce_{case}")


# Covers all_to_all_async composite path forwarding (barrier path when branch supports it).
@pytest.mark.timeout(60)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=["mesh_device"])
def test_composite_all_to_all_hang(mesh_device):
    logger.info("all_to_all: a hang during execution means this test failed")
    torch.manual_seed(42)
    ccl_sub_device_crs, worker_sub_device_id = _setup_device(mesh_device)
    run_l1_overwrite = _setup_l1_overwrite(mesh_device)

    ccl_sem = _make_sem(mesh_device, ccl_sub_device_crs, SEM_BUFFER_TYPE)
    barrier_sem = _make_sem(mesh_device, ccl_sub_device_crs, SEM_BUFFER_TYPE)

    num_devices = mesh_device.get_num_devices()
    in_dim = 0
    out_dim = 1
    logical_shape = [256, 256]

    golden_full = torch.randn(logical_shape).bfloat16()
    golden = list(torch.chunk(golden_full, num_devices, dim=out_dim))

    input_mesh = ttnn.from_torch(
        golden_full,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=in_dim),
    )

    output_shape = list(logical_shape)
    output_shape[out_dim] //= num_devices
    persistent_intermediate = ttnn.from_torch(
        torch.zeros(output_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    persistent_output = ttnn.from_torch(
        torch.zeros(output_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        return ttnn.experimental.all_to_all_async(
            input_tensor=input_mesh,
            persistent_intermediate_buffer=persistent_intermediate,
            persistent_output_buffer=persistent_output,
            in_dim=in_dim,
            out_dim=out_dim,
            multi_device_global_semaphore=ccl_sem,
            barrier_semaphore=barrier_sem,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            subdevice_id=worker_sub_device_id,
        )

    result = _trace_corruption_and_rerun(mesh_device, run_op, run_l1_overwrite)
    for i, t in enumerate(ttnn.get_device_tensors(result)):
        tt_out = ttnn.to_torch(t)
        eq, msg = comp_pcc(tt_out, golden[i])
        assert eq, f"all_to_all correctness failed on device {i}: {msg}"
