import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
import statistics, torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn
from eval.sharding import shard_config

_ML = ttnn.TensorMemoryLayout
K = "DEVICE KERNEL DURATION [ns]"


def ns(d):
    ttnn.ReadDeviceProfiler(d)
    pc = ttnn.get_latest_programs_perf_data()
    tot = 0.0
    f = False
    for progs in (pc or {}).values():
        for p in progs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(K)
            if e is not None:
                tot += float(e.duration)
                f = True
    return tot if f else None


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


d = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 32, 7168)
    W = 7168
    mc = shard_config([32, 256], (7, 4), _ML.WIDTH_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=d)
    torch.manual_seed(0)
    x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.float32).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=d,
        memory_config=mc,
    )
    g = ttnn.from_torch(
        torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=d,
    )
    for label, kw in [("full(gamma)", {"weight": g}), ("ABL no_gamma", {})]:
        s = []
        for i in range(5):
            rms_norm_ttnn(x, epsilon=1e-12, compute_kernel_config=cfg(), memory_config=x.memory_config(), **kw)
            ttnn.synchronize_device(d)
            v = ns(d)
            if v:
                s.append(v)
        print(f"RESULT {label:16s} med={statistics.median(s):8.0f} min={min(s):8.0f}")
finally:
    ttnn.close_device(d)
