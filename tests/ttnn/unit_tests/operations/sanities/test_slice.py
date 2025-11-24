import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_slice_my_case(device):
    # Input tensor
    x = torch.tensor([0.4963, 0.7682], dtype=torch.float32)
    print("Input tensor x =", x)

    # Convert to TT format
    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)

    sliced_tt = ttnn.slice(x_tt, (0,), (1,))

    # Convert back to torch
    sliced = ttnn.to_torch(sliced_tt)

    # Reference in PyTorch
    expected = x[0:1]

    assert_with_pcc(sliced, expected)

    print("Sliced TT tensor =", sliced)
    print("Expected tensor  =", expected)
    print("torch.allclose(sliced, expected) =", torch.allclose(sliced, expected))
    print("max_abs_diff =", torch.max(torch.abs(sliced - expected)).item())
