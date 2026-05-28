"""Quick smoke test: TILE-layout input + TILE-layout gamma/beta."""
import torch
import ttnn
from ttnn.operations.layer_norm_rm import layer_norm

torch.manual_seed(42)


def pytorch_ref(x, g=None, b=None, eps=1e-5):
    xf = x.to(torch.float32)
    mean = xf.mean(dim=-1, keepdim=True)
    var = xf.var(dim=-1, keepdim=True, unbiased=False)
    y = (xf - mean) / torch.sqrt(var + eps)
    if g is not None:
        y = y * g.reshape(-1).to(torch.float32)
    if b is not None:
        y = y + b.reshape(-1).to(torch.float32)
    return y.to(x.dtype)


device = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 64, 128)
    W = shape[-1]
    torch_x = torch.randn(shape, dtype=torch.float32)
    torch_g = torch.randn((1, 1, 1, W), dtype=torch.float32) * 0.5 + 1.0
    torch_b = torch.randn((1, 1, 1, W), dtype=torch.float32) * 0.1
    torch_y = pytorch_ref(torch_x, torch_g, torch_b)

    for cfg_name, in_layout, g_layout, b_layout in [
        ("RM-RM-RM", ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        ("TILE-RM-RM", ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        ("RM-TILE-TILE", ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
        ("TILE-TILE-TILE", ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    ]:
        x = ttnn.from_torch(
            torch_x, dtype=ttnn.float32, layout=in_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        g = ttnn.from_torch(
            torch_g, dtype=ttnn.float32, layout=g_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        b = ttnn.from_torch(
            torch_b, dtype=ttnn.float32, layout=b_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        y = layer_norm(x, g, b)
        y_t = ttnn.to_torch(y)
        diff = (y_t - torch_y).abs().max().item()
        print(f"{cfg_name}: in.layout={x.layout} out.layout={y.layout} max_abs_diff={diff:.6e}")
        assert y.layout == in_layout, f"Layout mismatch for {cfg_name}: got {y.layout}, expected {in_layout}"
        assert diff < 0.03, f"Diff too large for {cfg_name}: {diff}"

    print("OK")
finally:
    ttnn.close_device(device)
