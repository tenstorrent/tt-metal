import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)


def run(tag, shape, gamma_layout="tile", gamma=True):
    W = shape[-1]
    torch.manual_seed(0)
    ti = torch.randn(shape, dtype=torch.bfloat16)
    x = ti.to(torch.float32)
    ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    gt = None
    if gamma:
        tg = torch.randn(W, dtype=torch.bfloat16)
        ref = ref * tg.to(torch.float32).reshape(-1)
        gl = ttnn.ROW_MAJOR_LAYOUT if gamma_layout == "rm" else ttnn.TILE_LAYOUT
        gt = ttnn.from_torch(tg.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=gl, device=device)
    xt = ttnn.from_torch(ti, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)  # interleaved DRAM
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.fp32_dest_acc_en = True
    out = rms_norm(xt, gamma=gt, epsilon=1e-6, compute_kernel_config=cfg)
    r = ttnn.to_torch(out).to(torch.float32)
    pcc = torch.corrcoef(torch.stack([r.flatten(), ref.flatten()]))[0, 1].item()
    print(f"RESULT {tag} {shape}: PCC={pcc:.6f} maxdiff={(r-ref).abs().max().item():.4f}")


try:
    run("decode-1024", (1, 1, 32, 1024))
    run("decode-2304", (1, 1, 32, 2304))
    run("wide-16384", (1, 1, 32, 16384))
    run("wide-32768", (1, 1, 32, 32768))
    run("R2-12288", (1, 1, 64, 12288))
    run("small-256", (1, 1, 32, 256))
    run("nogamma-1024", (1, 1, 32, 1024), gamma=False)
    run("rmgamma-1024", (1, 1, 32, 1024), gamma_layout="rm")
    run("nonalign-W4100", (1, 1, 32, 4100))
finally:
    ttnn.close_device(device)
