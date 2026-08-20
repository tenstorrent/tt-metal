import torch, ttnn
from ttnn.operations.rms_norm.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:

    def run(shape, dtype, layout, tag):
        torch.manual_seed(0)
        t = torch.randn(*shape, dtype=torch.float32)
        g = torch.randn(shape[-1], dtype=torch.float32)
        x = ttnn.from_torch(t, dtype=dtype, layout=layout, device=dev)
        gg = ttnn.from_torch(g, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
        out = ttnn.to_torch(rms_norm(x, gamma=gg)).to(torch.float32)
        ref = t / torch.sqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * g
        pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
        print(f"SANITIZE {tag:38s} shape={tuple(shape)} pcc={pcc:.6f}")

    # ROW_MAJOR at large W: the stick chunk is the only transfer whose size can
    # approach / exceed NOC_MAX_BURST_SIZE (16384 B on Blackhole).
    run((1, 1, 32, 16384), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "RM bf16 W=16384 (32 KB row)")
    run((1, 1, 32, 32768), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "RM bf16 W=32768 (64 KB row)")
    run((1, 1, 64, 12288), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "RM bf16 W=12288")
    run((1, 1, 32, 8192), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, "RM fp32 W=8192 (32 KB row)")
    run((1, 1, 32, 4095), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "RM bf16 W=4095 (partial stick)")
    # TILE side, all three page sizes, so the page-size ASSERT is exercised too.
    run((1, 1, 64, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, "TILE bf16 2048 B page")
    run((1, 1, 64, 1024), ttnn.float32, ttnn.TILE_LAYOUT, "TILE fp32 4096 B page")
    run((1, 1, 64, 1024), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, "TILE bf8 1088 B page")
finally:
    ttnn.close_device(dev)
