import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("shape", [[1, 50, 1, 3, 768], [1, 1370, 1, 3, 1280]])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_to_layout_5D(shape, input_layout, output_layout, device):
    torch.manual_seed(2005)
    input_a = torch.randn(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_a, device=device, layout=input_layout, dtype=ttnn.bfloat16)
    output_tensor = ttnn.to_layout(input_tensor, output_layout)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_a, output_tensor)
