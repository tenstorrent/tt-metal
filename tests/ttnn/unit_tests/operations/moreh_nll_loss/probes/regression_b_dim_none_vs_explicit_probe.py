import torch
import ttnn


def test_dim_explicit(device):
    # dim=None (problematic) vs dim=0 (explicit)
    for shape in [[5], [16], [32]]:
        x = torch.ones(shape, dtype=torch.float32)
        tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.from_torch(torch.tensor([0.0]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # dim=None
        out_none_buf = ttnn.from_torch(torch.tensor([0.0]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        r_none = ttnn.operations.moreh.sum(tt, dim=None, keepdim=False, output=out_none_buf)
        v_none = ttnn.to_torch(r_none).reshape([-1])[0].item()

        # dim=0
        out_dim0_buf = ttnn.from_torch(torch.tensor([0.0]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        r_dim0 = ttnn.operations.moreh.sum(tt, dim=0, keepdim=False, output=out_dim0_buf)
        v_dim0 = ttnn.to_torch(r_dim0).reshape([-1])[0].item()

        print(f"shape={shape}  dim=None: {v_none:>10.3g}  dim=0: {v_dim0:>10.3g}")
