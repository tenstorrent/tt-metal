"""uint8 levers round 2: skip_format_reconfig x math_fidelity x fp32_dest."""
import torch, ttnn
from ttnn.operations import tilize as T
import ttnn.operations.tilize.tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)
T.SUPPORTED["dtype"] = [ttnn.bfloat16, ttnn.uint8]
T.SUPPORTED["output_dtype"] = [ttnn.bfloat16, ttnn.uint8]

_orig_plan, _orig_ccd = pd.derive_plan, ttnn.ComputeConfigDescriptor
CFG = {"fp32": False, "fid": ttnn.MathFidelity.HiFi4}


def plan_wrap(*a, **kw):
    p = _orig_plan(*a, **kw)
    p.fp32_dest_acc_en = CFG["fp32"]
    return p


def ccd_wrap(**kw):
    kw["math_fidelity"] = CFG["fid"]
    return _orig_ccd(**kw)


pd.derive_plan, ttnn.ComputeConfigDescriptor = plan_wrap, ccd_wrap

lin = (torch.arange(32).reshape(32, 1) * 32 + torch.arange(32).reshape(1, 32)) % 100
x = lin.to(torch.uint8).reshape(1, 1, 32, 32)
exp = lin.to(torch.int64)

for skip in (True, False):
    for fid in (ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.LoFi):
        for fp32 in (False, True):
            pd.COMPUTE_SKIP_FORMAT_RECONFIG = skip
            CFG["fid"], CFG["fp32"] = fid, fp32
            pd._PLAN_CACHE.clear()
            try:
                tt = ttnn.from_torch(
                    x,
                    dtype=ttnn.uint8,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                out = T.tilize(tt, dtype=ttnn.uint8)
                got = ttnn.to_torch(out).reshape(32, 32).to(torch.int64)
                print(
                    f"skip={skip} fid={fid} fp32={fp32}: nmis={int((got!=exp).sum())}/1024 got0[:8]={got[0,:8].tolist()}",
                    flush=True,
                )
            except Exception as e:
                print(f"skip={skip} fid={fid} fp32={fp32}: EXC {type(e).__name__}: {str(e)[:200]}", flush=True)
ttnn.close_device(device)
