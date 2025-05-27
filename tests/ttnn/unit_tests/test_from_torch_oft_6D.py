import ttnn
import torch


def test_from_torch_oft(device):
    torch_x = torch.rand((1, 4, 160, 160, 3, 1), dtype=torch.bfloat16)
    x_tensor = ttnn.from_torch(torch_x, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    x_tensor = ttnn.to_torch(x_tensor)
    assert torch.allclose(torch_x, x_tensor)
