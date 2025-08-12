import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_div(device):
    input_a = torch.randn(1, 1, 32, 32)  # *4.7
    # input_a = torch.randn(1,1,1024,1)
    input_b = torch.randn(1, 1, 32, 32)  # *1.4+1

    result1 = input_a / input_b
    input_a = ttnn.from_torch(input_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(input_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    print(input_a.shape)
    print(input_b.shape)

    result2 = ttnn.xlogy(input_a, input_b)
    result2 = ttnn.to_torch(result2)
    print(result1)
    print(result2)

    assert_with_pcc(result1, result2, 0.99)
