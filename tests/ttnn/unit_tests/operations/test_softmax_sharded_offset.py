import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull

def run_softmax_sharded_offset_test(device, batch_size, h, w, offset, orientation, is_masked):
    # Grid setup
    grid_start = offset
    grid_end = (offset[0] + 0, offset[1] + (batch_size - 1)) if orientation == ttnn.ShardOrientation.ROW_MAJOR else (offset[0] + (batch_size - 1), offset[1] + 0)
    
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(grid_start[0], grid_start[1]), ttnn.CoreCoord(grid_end[0], grid_end[1]))})
    shard_shape = (h, w)
    
    input_shape = (1, 1, batch_size * h, w)
    torch_input = torch.randn(input_shape).bfloat16().float()
    
    # Sharded memory config
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, orientation)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    
    # Move to device
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config)
    
    if is_masked:
        mask_shape = (1, 1, 1, w)
        torch_mask = torch.randn(mask_shape).bfloat16().float()
        tt_mask = ttnn.from_torch(torch_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_output = ttnn.scale_mask_softmax_in_place(tt_input, scale=1.0, mask=tt_mask)
        
        # Reference
        torch_output = torch.nn.functional.softmax(torch_input + torch_mask, dim=-1)
    else:
        tt_output = ttnn.softmax_in_place(tt_input)
        torch_output = torch.nn.functional.softmax(torch_input, dim=-1)

    tt_output_torch = ttnn.to_torch(tt_output)

    # Shard-by-shard validation
    assert_with_pcc(torch_output, tt_output_torch, 0.99)

@skip_for_grayskull("Softmax sharded is optimized for Wormhole/Blackhole")
@pytest.mark.parametrize("offset", [(0, 0), (1, 0), (0, 1)])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("is_masked", [True, False])
def test_softmax_sharded_on_offset_grids(device, offset, orientation, is_masked):
    run_softmax_sharded_offset_test(
        device=device,
        batch_size=4,
        h=32,
        w=64,
        offset=offset,
        orientation=orientation,
        is_masked=is_masked
    )
