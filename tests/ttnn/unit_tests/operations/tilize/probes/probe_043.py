"""op-vs-host block-float quality, UNPADDED and tile-aligned, across the pack knobs.
Confirms the descriptor actually took the setting before reading the numbers."""
import importlib, torch, ttnn

M = importlib.import_module("ttnn.operations.tilize.tilize")
import ttnn.operations.tilize.tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)
device.disable_and_clear_program_cache()
_plan, _ccd = pd.derive_plan, ttnn.ComputeConfigDescriptor
CFG = {"fp32": None, "precise": False, "seen": None}


def plan_wrap(*a, **kw):
    p = _plan(*a, **kw)
    if CFG["fp32"] is not None:
        p.fp32_dest_acc_en = CFG["fp32"]
    return p


def ccd_wrap(**kw):
    kw["bfp8_pack_precise"] = CFG["precise"]
    c = _ccd(**kw)
    CFG["seen"] = (bool(c.fp32_dest_acc_en), bool(c.bfp8_pack_precise))
    return c


pd.derive_plan, ttnn.ComputeConfigDescriptor = plan_wrap, ccd_wrap


def pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    va, vb = a - a.mean(), b - b.mean()
    d = va.norm() * vb.norm()
    return float((va * vb).sum() / d) if d > 0 else 1.0


torch.manual_seed(0)
shape = (1, 1, 64, 128)
for in_dt, tdt in ((ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)):
    x = torch.randn(shape).to(tdt)
    for out in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        host = ttnn.to_torch(ttnn.from_torch(x, dtype=out, layout=ttnn.TILE_LAYOUT)).float()
        hp = pcc(host, x.float())
        for fp32, prec in ((None, False), (True, True), (False, True)):
            CFG["fp32"], CFG["precise"] = fp32, prec
            pd._PLAN_CACHE.clear()
            tt = ttnn.from_torch(
                x, dtype=in_dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            got = ttnn.to_torch(M.tilize(tt, dtype=out)).float()
            print(
                f"{str(in_dt)[9:]:9s}->{str(out)[9:]:10s} cfg(dest,prec)={CFG['seen']} "
                f"op={pcc(got, x.float()):.6f} host={hp:.6f} "
                f"ndiff_op_vs_host={int((got != host).sum())}/{got.numel()}",
                flush=True,
            )
ttnn.close_device(device)
