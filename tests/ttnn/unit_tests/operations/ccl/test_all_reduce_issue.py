import ttnn
from tests.sweep_framework.sweeps.ccl.generality.all_reduce import run as run_all_reduce_test, LEAD_MODEL_SHARD_SPECS
from tests.sweep_framework.sweep_utils.ccl_common import get_mem_configs
import torch
import pytest
from loguru import logger
import math
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def run_all_reduce_test0(
    mesh_device,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    input_memory_config,
    output_memory_config,
    function_level_defaults,
    num_iters=1,
    topology=ttnn.Topology.Linear,
):
    if len(mesh_device.get_device_ids()) < num_devices:
        pytest.skip(
            f"Not enough devices on machine to implement test case. Wanted {num_devices} but found {len(mesh_device.get_device_ids())}"
        )

    ttnn.synchronize_device(mesh_device)

    sub_device_stall_group = []
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    print("compute grid size: ", compute_grid_size)
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)
    # create global semaphore handles
    rs_global_semaphores = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(3)]
    ag_global_semaphores = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)]
    barrier_semaphores = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)]

    debug = False

    logger.info(f"Per chip output shape: {per_chip_output_shape}, devices: {num_devices}")
    # Generate input tensors
    canonical_input_tensors = []
    input_tensors = []

    numel = math.prod(per_chip_output_shape)
    if debug:
        input_tensors[-1] = torch.arange(numel).reshape(per_chip_output_shape).bfloat16()
    for i in range(num_devices):
        input_tensor = torch.rand(per_chip_output_shape).bfloat16()
        canonical_input_tensors.append(input_tensor)
        input_tensor = input_tensor.view(1, -1, input_tensor.shape[2], input_tensor.shape[3])
        input_tensors.append(input_tensor)

    unchunked_input_tensor = torch.cat(input_tensors)

    assert len(canonical_input_tensors) == num_devices
    input_tensor_mesh = ttnn.from_torch(
        torch.cat(canonical_input_tensors),
        dtype=input_dtype,
        layout=layout,
        device=mesh_device,
        memory_config=input_memory_config,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(0)], ttnn.MeshShape(1, num_devices)),
        ),
    )
    # Run the op
    for i in range(num_iters):
        output_tensor_mesh = ttnn.experimental.all_reduce_async(
            input_tensor_mesh,
            num_devices=num_devices,
            barrier_semaphores=barrier_semaphores,
            rs_global_semaphores=rs_global_semaphores,
            ag_global_semaphores=ag_global_semaphores,
            math_op=math_op,
            num_links=num_links,
            memory_config=output_memory_config,
            topology=ttnn.Topology.Linear,
            subdevice_id=worker_sub_device_id,
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

    mesh_device.reset_sub_device_stall_group()

    tt_out_tensors = ttnn.get_device_tensors(output_tensor_mesh)
    logger.info(f"Compare")
    golden_canonical_out_tensor = torch.sum(unchunked_input_tensor, 0, keepdim=True)
    golden_canonical_out_tensor = golden_canonical_out_tensor.view(per_chip_output_shape)
    # Compare
    mismatch = False
    for i, t in enumerate(tt_out_tensors):
        tt_output_tensor = ttnn.to_torch(t)

        eq, output = comp_pcc(tt_output_tensor, golden_canonical_out_tensor)
        mismatch = mismatch or not eq
        if not eq:
            logger.error(f"output mismatch for tensor {i}. Mesh device ID: {mesh_device.get_device_ids()[i]}")
            if debug:
                for w in range(tt_output_tensor.shape[0]):
                    for z in range(tt_output_tensor.shape[1]):
                        for y in range(tt_output_tensor.shape[2]):
                            for x in range(tt_output_tensor.shape[3]):
                                if tt_output_tensor[w, z, y, x] != golden_canonical_out_tensor[w, z, y, x]:
                                    logger.error(
                                        f"mismatch at {w}, {z}, {y}, {x}: {tt_output_tensor[w, z, y, x]} != {golden_canonical_out_tensor[w, z, y, x]}"
                                    )

        else:
            logger.info(f"output match for tensor {i}")
    assert not mismatch, f"{i} FAILED: {output}"


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "num_devices",
    [
        2,
    ],
)
@pytest.mark.parametrize(
    "num_links",
    [1],
)
@pytest.mark.parametrize(
    "per_chip_output_shape",
    [
        ([1, 1, 32, 1280]),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize("math_op", [ttnn.ReduceType.Sum])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_ring_all_reduce_post_commit(
    mesh_device,
    mesh_shape,
    num_devices,
    per_chip_output_shape,
    num_links,
    math_op,
    input_dtype,
    layout,
    function_level_defaults,
    num_iters=2,
):
    shard_specs = LEAD_MODEL_SHARD_SPECS[0]
    input_memory_config, output_memory_config = get_mem_configs(ttnn.BufferType.L1, shard_specs, per_chip_output_shape)
    run_all_reduce_test0(
        mesh_device,
        num_devices,
        per_chip_output_shape,
        num_links,
        math_op,
        input_dtype,
        layout,
        input_memory_config,
        output_memory_config,
        function_level_defaults,
        num_iters=num_iters,
    )
