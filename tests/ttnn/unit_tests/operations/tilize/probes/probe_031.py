"""uint8 levers: fp32_dest_acc_en x dst_full_sync_en, on a 32x32 uint8 identity."""
import torch, ttnn
from ttnn.operations import tilize as T
import ttnn.operations.tilize.tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)
T.SUPPORTED["dtype"] = [ttnn.bfloat16, ttnn.uint8]
T.SUPPORTED["output_dtype"] = [ttnn.bfloat16, ttnn.uint8]

_orig_plan = pd.derive_plan
_orig_ccd = ttnn.ComputeConfigDescriptor

FP32 = [False]
FULLSYNC = [False]


def plan_wrap(*a, **kw):
    p = _orig_plan(*a, **kw)
    p.fp32_dest_acc_en = FP32[0]
    return p


def ccd_wrap(**kw):
    kw["dst_full_sync_en"] = FULLSYNC[0]
    return _orig_ccd(**kw)


pd.derive_plan = plan_wrap
ttnn.ComputeConfigDescriptor = ccd_wrap

x = (
    ((torch.arange(32).reshape(32, 1) * 32 + torch.arange(32).reshape(1, 32)) % 251)
    .to(torch.uint8)
    .reshape(1, 1, 32, 32)
)

for fp32 in (False, True):
    for fs in (False, True):
        FP32[0], FULLSYNC[0] = fp32, fs
        pd._PLAN_CACHE.clear()
        try:
            tt = ttnn.from_torch(
                x, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            out = T.tilize(tt, dtype=ttnn.uint8)
            got = ttnn.to_torch(out).reshape(32, 32).to(torch.int64)
            exp = x.reshape(32, 32).to(torch.int64)
            print(
                f"fp32_dest={fp32} full_sync={fs}: nmismatch={int((got!=exp).sum())}/1024  got_row0[:8]={got[0,:8].tolist()} exp={exp[0,:8].tolist()}",
                flush=True,
            )
        except Exception as e:
            print(f"fp32_dest={fp32} full_sync={fs}: EXC {type(e).__name__}: {str(e)[:250]}", flush=True)
ttnn.close_device(device)
