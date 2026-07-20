import os

os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
import glob, statistics, torch, ttnn
import pandas as pd
from ttnn.operations.tilize import tilize as gen_tilize  # generated op

prod_tilize = ttnn.tilize  # production C++ op

SHAPES = [[1, 1, 32, 32], [3, 1, 320, 384], [1, 1, 128, 7328], [1, 1, 2048, 2048]]
K = 16
LOGDIR = "generated/profiler/.logs"


def latest_csv():
    fs = glob.glob(os.path.join(LOGDIR, "profile_log_device*.csv"))
    return max(fs, key=os.path.getmtime) if fs else None


def dur_col(df):
    for c in df.columns:
        if "DEVICE KERNEL DURATION" in c:
            return c
    raise RuntimeError(f"no duration col in {list(df.columns)}")


dev = ttnn.open_device(device_id=0)
seen = 0
results = {}  # (label, tuple(shape)) -> median ns


def measure(label, fn, shape):
    global seen
    t = torch.randn(shape, dtype=torch.float32)
    tin = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for _ in range(3):  # warmup
        o = fn(tin)
        ttnn.synchronize_device(dev)
    for _ in range(K):
        o = fn(tin)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    df = pd.read_csv(latest_csv())
    new = df.iloc[seen:]
    seen = len(df)
    col = dur_col(new)
    vals = pd.to_numeric(new[col], errors="coerce").dropna().tolist()
    med = statistics.median(vals) if vals else float("nan")
    results[(label, tuple(shape))] = (med, len(vals))
    print(f"  {label:10s} shape={shape} -> median {med:,.0f} ns  ({len(vals)} device-op rows)")


for shape in SHAPES:
    print(f"shape {shape}")
    measure("prod C++", lambda x: prod_tilize(x, use_multicore=True), shape)
    measure("generated", lambda x: gen_tilize(x, use_multicore=True), shape)

print("\n=== SUMMARY: DEVICE KERNEL DURATION median ns (multicore) ===")
print(f"{'shape':22s} {'prod C++ ns':>14s} {'generated ns':>14s} {'gen/prod':>10s}")
for shape in SHAPES:
    p = results[("prod C++", tuple(shape))][0]
    g = results[("generated", tuple(shape))][0]
    ratio = g / p if p else float("nan")
    print(f"{str(shape):22s} {p:14,.0f} {g:14,.0f} {ratio:9.2f}x")

ttnn.close_device(dev)
