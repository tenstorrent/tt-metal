import torch, ttnn
from ttnn.operations.rms_norm.rms_norm import rms_norm, default_compute_kernel_config

dev = ttnn.open_device(device_id=0)
try:

    def run(shape, dtype, layout, tag, gdtype=None):
        torch.manual_seed(0)
        t = torch.randn(*shape, dtype=torch.float32)
        g = torch.randn(shape[-1], dtype=torch.float32)
        x = ttnn.from_torch(t, dtype=dtype, layout=layout, device=dev)
        gg = ttnn.from_torch(g, dtype=gdtype or dtype, layout=ttnn.TILE_LAYOUT, device=dev)
        y = rms_norm(x, gamma=gg)
        out = ttnn.to_torch(y).to(torch.float32)
        ref = t / torch.sqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * g
        pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
        print(f"PROBE {tag:32s} shape={tuple(shape)} pcc={pcc:.6f}")

    run((1, 1, 64, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, "TILE bf16 2048B page")
    run((1, 1, 64, 1024), ttnn.float32, ttnn.TILE_LAYOUT, "TILE fp32 4096B page")
    run((1, 1, 64, 1024), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, "TILE bf8 1088B page")
    run((1, 1, 64, 1024), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "RM bf16 sticks")
    run((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, "TILE bf16 w_nonalign")
    run((1, 1, 32, 16384), ttnn.bfloat16, ttnn.TILE_LAYOUT, "TILE bf16 wide W-split")
    run((1, 1, 32, 16384), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "RM bf16 W=16384 (32KB row)")
finally:
    ttnn.close_device(dev)
