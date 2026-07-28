import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
try:
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False

    def check(shape, gmode, tag):
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
        tg = ttnn.from_torch(g.reshape(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        out = ttnn.to_torch(rms_norm(tx, gamma=tg, epsilon=1e-6, compute_kernel_config=cfg)).to(torch.float32)
        xf = x.to(torch.float32)
        exp = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6) * g.to(torch.float32).reshape(-1)
        num = (out * exp).sum()
        den = out.norm() * exp.norm()
        pcc = (num / den).item()
        maxrel = ((out - exp).abs() / exp.abs().clamp(min=1e-3)).max().item()
        print(f"  {tag:32s} gamma={gmode:6s} pcc={pcc:.7f} maxrel={maxrel:.4f}")
        ttnn.deallocate(tx)
        ttnn.deallocate(tg)
        return pcc

    # engaged? report the plan for each shape
    for shape in [(1, 1, 8192, 1024), (1, 1, 8192, 7168), (1, 1, 8192, 1000)]:
        tx = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
        )
        tg = ttnn.from_torch(
            torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
        )
        grid = dev.compute_with_storage_grid_size()
        ht, wt = pd._tile_geometry(tx)
        runs = pd._virtual_x_runs(dev, grid)
        pl = pd._placement_rows(grid, ht)
        blk = pd._derive_blocking(tx, tg, grid.x * grid.y, pl, l1_total_budget=pd._l1_total_budget(dev))
        plan = pd._gamma_mcast_plan(pl, runs, blk, tg, False)
        print(
            f"{shape}: cw={pl.cw} cores={len([w for w in pl.works if w.num_rows>0])} "
            f"gamma_res={blk.gamma_resident} -> mcast ENGAGED={plan is not None}"
            + (f" families={[f[0] for f in plan[0]]} injectors={[f[1] for f in plan[0]]}" if plan else "")
        )
        ttnn.deallocate(tx)
        ttnn.deallocate(tg)
    print()
    for shape in [(1, 1, 8192, 1024), (1, 1, 8192, 7168), (1, 1, 8192, 1000), (1, 1, 32, 1024)]:
        for gm in ("ramp", "ones", "rand"):
            p = check(shape, gm, str(shape))
            assert p > 0.999, f"PCC FAIL {shape} {gm} {p}"
    print("\nALL GAMMA-BROADCAST CORRECTNESS GATES PASS")
finally:
    ttnn.close_device(dev)
