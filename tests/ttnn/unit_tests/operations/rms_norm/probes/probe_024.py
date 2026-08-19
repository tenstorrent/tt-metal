import torch, ttnn
from ttnn.operations.rms_norm import rms_norm, default_compute_kernel_config
from ttnn.operations.rms_norm.rms_norm_program_descriptor import blocking_plan


def stats(e, a):
    rms = (a - e).pow(2).mean().sqrt().item() / e.std().item()
    pcc = torch.corrcoef(torch.stack([e.flatten().double(), a.flatten().double()]))[0, 1].item()
    return pcc, rms


def cfg_of(acc, fid=ttnn.MathFidelity.HiFi2):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = fid
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    TOL = {ttnn.float32: 0.02, ttnn.bfloat16: 0.04, ttnn.bfloat8_b: 0.10}
    for W in [17, 50, 64, 512, 5120, 7168, 11008]:
        shape = (1, 1, 64, W)
        tx = torch.randn(shape)
        tg = torch.randn(W).reshape(1, 1, 1, W)
        for in_dt, dn in [(ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16"), (ttnn.bfloat8_b, "bf8b")]:
            if in_dt == ttnn.bfloat8_b and W % 32:
                continue  # INVALID: bf8b x non-aligned
            for acc in [True, False]:
                if in_dt == ttnn.float32 and not acc:
                    continue  # EXCLUDED cell
                td = torch.float32 if in_dt == ttnn.float32 else torch.bfloat16
                x = ttnn.from_torch(tx.to(td), dtype=in_dt, layout=ttnn.TILE_LAYOUT, device=dev)
                for gdt, gn in [(ttnn.bfloat8_b, "bf8b"), (ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32")]:
                    g = ttnn.from_torch(
                        tg.to(torch.float32 if gdt == ttnn.float32 else torch.bfloat16),
                        dtype=gdt,
                        layout=ttnn.TILE_LAYOUT,
                        device=dev,
                    )
                    x32 = ttnn.to_torch(x).float()
                    e = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6) * ttnn.to_torch(g).float()[..., :W]
                    p = blocking_plan(x, g, None, dev, cfg_of(acc), None)
                    out = ttnn.to_torch(rms_norm(x, gamma=g, compute_kernel_config=cfg_of(acc))).float()
                    pcc, rms = stats(e, out)
                    bad = "FAIL" if rms > TOL[in_dt] else "ok"
                    if bad == "FAIL" or W in (17, 7168):
                        print(
                            f"RESULT W={W:>5} in={dn} g={gn} acc={acc!s:<5} reg={p.regime} "
                            f"via={p.reduce_via_add} PCC={pcc:.6f} rms={rms:.5f} {bad}"
                        )
    print("RESULT done")
finally:
    ttnn.close_device(dev)
