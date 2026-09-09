"""Block-float OUTPUT precision: does packing from fp32 DEST *precisely*
(one quantization instead of fp32 -> Bfp8_b -> Bfp4_b) lift the PCC?"""
import importlib, torch, ttnn

M = importlib.import_module("ttnn.operations.tilize.tilize")
import ttnn.operations.tilize.tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)
device.disable_and_clear_program_cache()

_plan, _ccd = pd.derive_plan, ttnn.ComputeConfigDescriptor
CFG = {"fp32": None, "precise": False}


def plan_wrap(*a, **kw):
    p = _plan(*a, **kw)
    if CFG["fp32"] is not None:
        p.fp32_dest_acc_en = CFG["fp32"]
    return p


def ccd_wrap(**kw):
    kw["bfp8_pack_precise"] = CFG["precise"]
    return _ccd(**kw)


pd.derive_plan, ttnn.ComputeConfigDescriptor = plan_wrap, ccd_wrap


def pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    va, vb = a - a.mean(), b - b.mean()
    d = va.norm() * vb.norm()
    return float((va * vb).sum() / d) if d > 0 else 1.0


torch.manual_seed(11)
CASES = [
    ("randn 1x1x64x128", (1, 1, 64, 128), None),
    ("randn+pad-7 1x1x50x50", (1, 1, 50, 50), -7),
    ("rank1 64 pad0", (64,), 0),
]
for name, shape, pv in CASES:
    for in_dt, tdt in ((ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)):
        x = torch.randn(shape).to(tdt)
        for out in (ttnn.bfloat8_b, ttnn.bfloat4_b):
            row = []
            for fp32, prec in ((None, False), (True, False), (True, True), (None, True)):
                CFG["fp32"], CFG["precise"] = fp32, prec
                pd._PLAN_CACHE.clear()
                tt = ttnn.from_torch(
                    x, dtype=in_dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                kw = {} if pv is None else {"pad_value": pv}
                o = M.tilize(tt, dtype=out, **kw)
                got = ttnn.to_torch(o).float()
                row.append(f"dest={'dflt' if fp32 is None else 'fp32'},prec={int(prec)}:{pcc(got, x.float()):.6f}")
            print(f"{name:24s} {str(in_dt)[9:]:9s}->{str(out)[9:]:10s} " + "  ".join(row), flush=True)
ttnn.close_device(device)
