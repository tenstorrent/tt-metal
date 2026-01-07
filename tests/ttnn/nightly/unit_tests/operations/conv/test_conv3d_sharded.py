import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull

@skip_for_grayskull("Conv3d not supported on Grayskull")
@pytest.mark.parametrize("sharding_config", [
    ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_conv3d_sharded_comprehensive(device, sharding_config, input_layout):
    torch.manual_seed(0)
    input_shape = (1, 3, 32, 32, 32)
    weight_shape = (32, 32, 3, 3, 3) 
    bias_shape = (32,)

    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_weight = torch.randn(weight_shape, dtype=torch.bfloat16)
    torch_bias = torch.randn(bias_shape, dtype=torch.bfloat16)

    torch_output = torch.nn.functional.conv3d(torch_input, torch_weight, torch_bias, stride=1, padding=1)

    tt_input = ttnn.from_torch(torch_input, device=device, layout=input_layout, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_input = ttnn.to_memory_config(tt_input, sharding_config)
    
    tt_weight = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(torch_bias.view(1, 1, 1, 1, -1), device=device, layout=ttnn.TILE_LAYOUT)

    tt_output = ttnn.experimental.conv3d(tt_input, tt_weight, bias=tt_bias, stride=[1, 1, 1], padding=[1, 1, 1], dtype=ttnn.bfloat16)

    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.99)