import os

os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
import csv, glob, statistics, torch, ttnn
from ttnn.operations.tilize import tilize as gen_tilize  # generated (this branch)

prod_tilize = ttnn.tilize  # native (ttnn main C++)

# Device profiler writes under $TT_METAL_HOME, not necessarily the cwd/clone.
TT_HOME = os.environ.get("TT_METAL_HOME", ".")
LOGDIR = os.path.join(TT_HOME, "generated", "profiler", ".logs")
DEVLOG = os.path.join(LOGDIR, "profile_log_device.csv")

# Interleaved model shapes both ops support (bf16, DRAM interleaved, multicore).
SHAPES = [
    [1, 1, 32, 16384],  # DeepSeek MLA wo  (nt_h=1 -> single-core-collapse suspect)
    [1, 8, 128, 7168],  # DeepSeek unit large
    [8, 1, 32, 7168],  # DeepSeek unit large
    [1, 1, 2048, 2048],  # Falcon mask
    [512, 512],  # fp32-truncation shape (run bf16)
    [1, 1, 32, 256],  # small (host-dispatch dominated in wall-clock)
]
WARMUP, K = 3, 15

# Start clean so stale run-host-IDs don't leak in.
for f in glob.glob(os.path.join(LOGDIR, "profile_log_device*.csv")):
    try:
        os.remove(f)
    except OSError:
        pass


def read_kernel_spans():
    """Return {run_host_id: kernel_span_cycles} from the raw device log.
    Kernel span = max(END) - min(START) over all *-KERNEL zones across all
    cores for that op invocation (run host ID)."""
    if not os.path.exists(DEVLOG):
        return {}
    with open(DEVLOG) as fh:
        lines = fh.readlines()
    # line 0 = "ARCH: ... CHIP_FREQ ...", line 1 = header.
    # Header fields carry leading spaces (" run host ID") — strip them.
    header = [c.strip() for c in next(csv.reader([lines[1]]))]
    rdr = csv.DictReader(lines[2:], fieldnames=header)
    per = {}  # rhid -> [min_start, max_end]
    for r in rdr:
        zone = (r.get("zone name") or "").strip()
        if "KERNEL" not in zone:
            continue
        try:
            rhid = int(r["run host ID"])
            t = int(r["time[cycles since reset]"])
        except (ValueError, KeyError, TypeError):
            continue
        typ = (r.get("type") or "").strip()
        lo, hi = per.get(rhid, (t, t))
        per[rhid] = (min(lo, t), max(hi, t))
    return {k: hi - lo for k, (lo, hi) in per.items()}


dev = ttnn.open_device(device_id=0)
freq_mhz = dev.get_clock_rate_mhz() if hasattr(dev, "get_clock_rate_mhz") else 1000.0
print(f"clock ~{freq_mhz} MHz; log={DEVLOG}")
results = {}  # (impl, tuple(shape)) -> (median_ns, n)
seen_max = -1


def measure(label, fn, shape):
    global seen_max
    t = torch.randn(shape, dtype=torch.float32)
    tin = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for _ in range(WARMUP):
        fn(tin)
        ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    spans = read_kernel_spans()
    base = max(spans.keys()) if spans else seen_max
    for _ in range(K):
        fn(tin)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    spans = read_kernel_spans()
    fresh = [c for rhid, c in spans.items() if rhid > base]
    seen_max = max(spans.keys()) if spans else seen_max
    ns = [c / freq_mhz * 1000.0 for c in fresh]  # cycles / MHz * 1000 = ns
    med = statistics.median(ns) if ns else float("nan")
    results[(label, tuple(shape))] = (med, len(ns))
    print(f"  {label:10s} {str(shape):22s} -> {med:10,.0f} ns  ({len(ns)} ops)")


for shape in SHAPES:
    print(f"shape {shape}")
    measure("native", lambda x: prod_tilize(x, use_multicore=True), shape)
    measure("generated", lambda x: gen_tilize(x, use_multicore=True), shape)

print("\n=== DEVICE KERNEL DURATION (median ns, multicore, DRAM interleaved bf16) ===")
print(f"{'shape':24s} {'native ns':>12s} {'generated ns':>13s} {'gen/native':>11s}")
for shape in SHAPES:
    n = results.get(("native", tuple(shape)), (float("nan"), 0))[0]
    g = results.get(("generated", tuple(shape)), (float("nan"), 0))[0]
    ratio = g / n if n and n == n else float("nan")
    print(f"{str(shape):24s} {n:12,.0f} {g:13,.0f} {ratio:10.2f}x")

ttnn.close_device(dev)
