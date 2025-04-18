import torch
import pytest
import ttnn
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_typecasting(mesh_device):
    device = mesh_device.get_device(mesh_device.get_device_ids()[0])
    torch_input = torch.randint(0, 32000, (1, 1, 32, 32), dtype=torch.long)
    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Prepare outout tensors

    # torch_output_tensor = torch.zeros(1, 1, 32, 32, dtype=torch.long)

    # start_core = ttnn.CoreCoord(1, 0)
    # core_grid = ttnn.CoreRangeSet(
    #     [
    #         ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
    #         ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
    #     ]
    # )
    # num_cores = 1
    # shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
    #     start_core, num_cores, core_grid, row_wise=False
    # )

    # indices_memory_config = ttnn.create_sharded_memory_config(
    #     shape=(1, 1, 32, 32),
    #     core_grid=shard_grid,
    #     strategy=ttnn.ShardStrategy.WIDTH,
    #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
    #     use_height_and_width_as_shard_shape=True,
    # )

    # tt_output = ttnn.from_torch(
    #     torch_output_tensor,
    #     device=device,
    #     dtype=ttnn.uint32,
    #     layout=ttnn.TILE_LAYOUT,
    #     memory_config=indices_memory_config,
    # )
    # tt_output_int32 = ttnn.from_torch(
    #     torch_output_tensor,
    #     device=device,
    #     dtype=ttnn.int32,
    #     layout=ttnn.TILE_LAYOUT,
    #     memory_config=indices_memory_config,
    # )

    # # Gives correct results but needs to be on sub core grids
    # tt_input = ttnn.typecast(tt_input, dtype=ttnn.uint32) # , output_tensor=tt_output
    # tt_input = ttnn.typecast(tt_input, dtype=ttnn.int32) # , output_tensor=tt_output_int32

    # tt_input = ttnn.experimental.typecast(tt_input, dtype=ttnn.uint32, sub_core_grids=sub_core_grids) # does not support uint16 inputs

    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    tt_input = ttnn.typecast(tt_input, dtype=ttnn.uint32, sub_core_grids=sub_core_grids)
    tt_input = ttnn.typecast(tt_input, dtype=ttnn.int32, sub_core_grids=sub_core_grids)

    breakpoint()

    tt_input_to_torch = ttnn.to_torch(tt_input)
    passing, pcc = comp_pcc(tt_input_to_torch, torch_input, 0.99)
    print(passing, pcc)
    assert passing, f"Typecasting failed"
