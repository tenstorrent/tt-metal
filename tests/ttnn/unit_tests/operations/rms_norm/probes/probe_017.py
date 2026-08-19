import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def cfg_of(acc, fid=ttnn.MathFidelity.HiFi2):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = fid
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    for W in [1024, 7168]:
        tx = torch.randn((1, 1, 32, W)).to(torch.bfloat16)
        x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        x32 = tx.float()
        s_ref = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)[0, 0, :, 0]
        cases = {}
        for gv in [1.0, 2.0, 0.5]:
            for gdt, dn in [(ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32")]:
                cases[f"g{gv}/{dn}"] = (
                    gv,
                    ttnn.from_torch(
                        torch.full((1, 1, 1, W), gv).to(torch.bfloat16 if gdt == ttnn.bfloat16 else torch.float32),
                        dtype=gdt,
                        layout=ttnn.TILE_LAYOUT,
                        device=dev,
                    ),
                )
        for fid, fn in [(ttnn.MathFidelity.HiFi2, "HiFi2"), (ttnn.MathFidelity.HiFi4, "HiFi4")]:
            for acc in [False]:
                for tag, (gv, gg) in cases.items():
                    out = ttnn.to_torch(rms_norm(x, gamma=gg, compute_kernel_config=cfg_of(acc, fid))).float()
                    num = (out[0, 0] * x32[0, 0]).sum(-1)
                    den = (x32[0, 0] * x32[0, 0]).sum(-1)
                    k = num / den / gv
                    print(f"RESULT W={W:>5} {fn} acc={acc} {tag:<11} rowscale_bias_mean={(k/s_ref-1).mean():+.5f}")
finally:
    ttnn.close_device(dev)
