import os, sys, shutil, json
import torch, ttnn

IN_BT   = os.environ["ZP_IN"]     # DRAM | L1
OUT_BT  = os.environ["ZP_OUT"]    # DRAM | L1
TID     = int(os.environ.get("ZP_TID", "0"))   # 0=no gamma, 1=gamma, 2=gamma+beta
TAG     = os.environ["ZP_TAG"]
OUTDIR  = "/localdev/dnijemcevic/2026_09_10_port/l1res"

from tt_lib.utils import pad_weight, tilize_to_list

bt = {"DRAM": ttnn.BufferType.DRAM, "L1": ttnn.BufferType.L1}
in_mc  = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, bt[IN_BT])
out_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, bt[OUT_BT])

dev = ttnn.open_device(device_id=0)
torch.manual_seed(1234)
N, C, H, W = 1, 9, 384, 1024
epsf = 1e-2
x = torch.rand((N, C, H, W)) * 2 - 0.95
ttx = ttnn.Tensor(tilize_to_list(x), [N, C, H, W], ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, in_mc)

kwargs = {}
if TID >= 1:
    g = torch.rand(1, 1, 1, W) * 2 - 1
    kwargs["weight"] = ttnn.Tensor(tilize_to_list(pad_weight(g)), [1, 1, 32, W], ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, in_mc)
if TID >= 2:
    b = torch.rand(1, 1, 1, W) * 2.0 - 1.1
    kwargs["bias"] = ttnn.Tensor(tilize_to_list(pad_weight(b)), [1, 1, 32, W], ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, in_mc)

SIDE = os.environ.get("ZP_SIDE", "generated")
if SIDE == "native":
    from ttnn.operations import normalization as Nrm
    op = Nrm._native_rms_norm
else:
    from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn as op

def one():
    ttnn.ReadDeviceProfiler(dev)                      # flush
    z = op(ttx, epsilon=epsf, memory_config=out_mc, **kwargs)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    per_chip = ttnn.get_latest_programs_perf_data()
    tot, best = 0.0, (None, None)
    for progs in (per_chip or {}).values():
        for p in progs:
            e = (getattr(p, "program_analyses_results", None) or {}).get("DEVICE KERNEL DURATION [ns]")
            if e is None: continue
            d = float(e.duration); tot += d
            c = getattr(p, "core_count", None)
            if best[0] is None or d > best[0]: best = (d, c)
    z.deallocate()
    return tot, best

# warm (compile + program cache), then the measured launch
one()
tot, best = one()
print(f"ZP_RESULT {TAG} side={SIDE} in={IN_BT} out={OUT_BT} tid={TID} total_ns={tot:.0f} dominant_ns={best[0]} cores={best[1]}")

ttnn.close_device(dev)

src = os.path.join(os.environ["TT_METAL_HOME"], "generated/profiler/.logs/profile_log_device.csv")
if os.path.exists(src):
    dst = os.path.join(OUTDIR, f"devlog_{TAG}.csv")
    shutil.copy(src, dst)
    print(f"ZP_CSV {dst} lines={sum(1 for _ in open(dst))}")
else:
    print(f"ZP_CSV MISSING {src}")
