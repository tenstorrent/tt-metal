import torch
import ttnn


def test_compare(device):
    # 1D ones [32] vs 2D [1,32] - run sum on both
    for x in [
        torch.ones([5], dtype=torch.float32),
        torch.ones([16], dtype=torch.float32),
        torch.ones([32], dtype=torch.float32),
        torch.ones([100], dtype=torch.float32),
        torch.ones([1, 5], dtype=torch.float32),
        torch.ones([1, 16], dtype=torch.float32),
        torch.ones([1, 32], dtype=torch.float32),
        torch.ones([1, 100], dtype=torch.float32),
        torch.ones([5, 1], dtype=torch.float32),
        torch.ones([16, 1], dtype=torch.float32),
        torch.ones([32, 1], dtype=torch.float32),
    ]:
        tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.from_torch(torch.tensor([0.0]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        r = ttnn.operations.moreh.sum(tt, dim=None, keepdim=False, output=out)
        val = ttnn.to_torch(r).reshape([-1])[0].item()
        expected = float(x.sum().item())
        print(f"shape={list(x.shape)} padded={tt.padded_shape} got={val:>12.3g} expected={expected:>8.1f}")
