import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_matmul_my_case(device):
    x = torch.load("inputs/matmul_input_0.pt")
    y = torch.load("inputs/matmul_input_1.pt")

    print("contents of inputs")
    print("x = \n", x)
    print("y = \n", y)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.matmul(x_tt, y_tt)
    z = ttnn.to_torch(z_tt)
    z_t = torch.matmul(x, y)
    assert_with_pcc(z_t, z)

    print("TT output tensor (z)       =", z)
    print("Expected torch.matmul(z_t) =", z_t)
    print("torch.allclose(z, z_t)     =", torch.allclose(z, z_t))
    print("max_abs_diff               =", torch.max(torch.abs(z - z_t)).item())
