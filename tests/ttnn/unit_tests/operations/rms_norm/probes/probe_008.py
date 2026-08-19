import torch, ttnn
from ttnn.operations.rms_norm import rms_norm, default_compute_kernel_config


def ref(x, g, eps=1e-6):
    x32 = x.to(torch.float32)
    o = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return o * g.to(torch.float32)


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


torch.manual_seed(0)
for shape in [(1, 1, 32, 64), (1, 1, 128, 512), (1, 1, 32, 4096)]:
    tx = torch.randn(shape)
    tg = torch.randn((1, 1, 1, shape[-1]))
    for dtype, name in [(ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32"), (ttnn.bfloat8_b, "bf8b")]:
        for acc in [True, False]:
            if dtype == ttnn.float32 and not acc:
                continue
            cfg = default_compute_kernel_config()
            cfg.fp32_dest_acc_en = acc
            x = ttnn.from_torch(tx, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
            g = ttnn.from_torch(tg, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
            try:
                out = ttnn.to_torch(rms_norm(x, gamma=g, compute_kernel_config=cfg)).float()
                e = ref(tx, tg)
                print(f"RESULT {shape} {name} acc={acc} PCC={pcc(e,out):.6f} maxabs={(out-e).abs().max():.4f}")
            except Exception as ex:
                print(f"RESULT {shape} {name} acc={acc} FAIL {type(ex).__name__}: {str(ex)[:200]}")
