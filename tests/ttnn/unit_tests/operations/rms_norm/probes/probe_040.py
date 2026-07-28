import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False

    def check(shape, gmode, gdtype=ttnn.bfloat16, glayout=ttnn.TILE_LAYOUT):
        torch.manual_seed(1)
        x = torch.randn(shape, dtype=torch.bfloat16)
        W = shape[-1]
        if gmode == "ramp":
            g = torch.arange(W, dtype=torch.float32).to(torch.bfloat16) / W + 0.5
        elif gmode == "ones":
            g = torch.ones(W, dtype=torch.bfloat16)
        else:
            g = torch.randn(W, dtype=torch.bfloat16)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        tg = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=gdtype, layout=glayout, device=dev)
        out = ttnn.to_torch(rms_norm(tx, gamma=tg, epsilon=1e-6, compute_kernel_config=cfg)).to(torch.float32)
        xf = x.to(torch.float32)
        exp = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6) * g.to(torch.float32).reshape(-1)
        pcc = ((out * exp).sum() / (out.norm() * exp.norm())).item()
        ttnn.deallocate(tx)
        ttnn.deallocate(tg)
        return pcc

    # all-ones ABSOLUTE element-count check (PCC is scale-blind for this op)
    def absolute(shape):
        W = shape[-1]
        tx = ttnn.from_torch(
            torch.ones(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
        )
        tg = ttnn.from_torch(
            torch.ones(1, 1, 1, W, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
        )
        out = ttnn.to_torch(rms_norm(tx, gamma=tg, epsilon=0.25, compute_kernel_config=cfg)).to(torch.float32)
        want = 1.0 / (1.0 + 0.25) ** 0.5
        err = (out - want).abs().max().item()
        ttnn.deallocate(tx)
        ttnn.deallocate(tg)
        return want, out.min().item(), out.max().item(), err

    ok = True
    for shape in [(1, 1, 8192, 1024), (1, 1, 8192, 7168), (1, 1, 8192, 1000), (1, 1, 32, 1024), (1, 1, 256, 512)]:
        for gm in ("ramp", "ones", "rand"):
            p = check(shape, gm)
            flag = "OK " if p > 0.999 else "FAIL"
            if p <= 0.999:
                ok = False
            print(f"  {flag} {str(shape):20s} gamma={gm:5s} pcc={p:.7f}")
    # mixed-precision gamma + RM gamma on the engaged shape
    for gd, gl, tag in [
        (ttnn.float32, ttnn.TILE_LAYOUT, "fp32 TILE gamma"),
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "bf16 RM gamma"),
    ]:
        p = check((1, 1, 8192, 1024), "ramp", gd, gl)
        print(f"  {'OK ' if p>0.999 else 'FAIL'} (1,1,8192,1024) {tag:18s} pcc={p:.7f}")
        if p <= 0.999:
            ok = False
    print()
    for shape in [(1, 1, 8192, 1024), (1, 1, 8192, 7168)]:
        want, lo, hi, err = absolute(shape)
        print(f"  ABSOLUTE {shape}: want={want:.7f} got [{lo:.7f},{hi:.7f}] maxerr={err:.6f}")
        if err > 5e-3:
            ok = False
    print("\n" + ("ALL GAMMA-BROADCAST GATES PASS" if ok else "*** GATE FAILURE ***"))
finally:
    ttnn.close_device(dev)
