import torch, ttnn
from ttnn.operations.backward_softmax import backward_softmax

device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(42)
    shape = (1, 1, 32, 64)

    for dtype_name, dtype in [("float32", ttnn.float32), ("bfloat16", ttnn.bfloat16), ("bfloat8_b", ttnn.bfloat8_b)]:
        for dim in [-1, -2]:
            torch_dy = torch.randn(shape, dtype=torch.float32)
            torch_y = torch.randn(shape, dtype=torch.float32)
            s = (torch_y * torch_dy).sum(dim=dim, keepdim=True)
            expected = (torch_y * (torch_dy - s)).float()

            if dtype == ttnn.bfloat16:
                torch_dy_h = torch_dy.to(torch.bfloat16)
                torch_y_h = torch_y.to(torch.bfloat16)
            else:
                torch_dy_h = torch_dy
                torch_y_h = torch_y

            dy = ttnn.from_torch(
                torch_dy_h, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            y = ttnn.from_torch(
                torch_y_h, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            out = backward_softmax(dy, y, dim=dim)
            actual = ttnn.to_torch(out).float()

            max_abs = (actual - expected).abs().max().item()
            mean_abs = (actual - expected).abs().mean().item()
            rms_rel = (
                (actual - expected).pow(2).mean().sqrt() / max(expected.pow(2).mean().sqrt().item(), 1e-12)
            ).item()
            print(
                f"{dtype_name:10s} dim={dim:+d}: out.dtype={out.dtype} max_abs={max_abs:.4f} mean_abs={mean_abs:.4f} rms_rel={rms_rel:.4f}"
            )
finally:
    ttnn.close_device(device)
