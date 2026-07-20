import os

os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
import csv, glob, statistics, torch, ttnn
from ttnn.operations.tilize import tilize as gen_tilize

prod_tilize = ttnn.tilize

TT_HOME = os.environ.get("TT_METAL_HOME", ".")
LOGDIR = os.path.join(TT_HOME, "generated", "profiler", ".logs")
DEVLOG = os.path.join(LOGDIR, "profile_log_device.csv")
SHAPE = [int(x) for x in os.environ["SHAPE"].split(",")]
IMPL = os.environ.get("IMPL", "gen")
DT = os.environ.get("DT", "bf16")
K = 15

for f in glob.glob(os.path.join(LOGDIR, "profile_log_device*.csv")):
    try:
        os.remove(f)
    except OSError:
        pass


def parse():
    if not os.path.exists(DEVLOG):
        return float("nan"), 0, 0
    with open(DEVLOG) as fh:
        lines = fh.readlines()
    header = [c.strip() for c in next(csv.reader([lines[1]]))]
    rdr = csv.DictReader(lines[2:], fieldnames=header)
    per = {}  # rhid -> [min,max]
    cores = {}  # rhid -> set((x,y))
    for r in rdr:
        zone = (r.get("zone name") or "").strip()
        if "KERNEL" not in zone:
            continue
        try:
            rhid = int(r["run host ID"])
            t = int(r["time[cycles since reset]"])
            cx = int(r["core_x"])
            cy = int(r["core_y"])
        except (ValueError, KeyError, TypeError):
            continue
        lo, hi = per.get(rhid, (t, t))
        per[rhid] = (min(lo, t), max(hi, t))
        cores.setdefault(rhid, set()).add((cx, cy))
    spans = {k: hi - lo for k, (lo, hi) in per.items()}
    if not spans:
        return float("nan"), 0, 0
    ns = [c / 1000.0 * 1000.0 for c in spans.values()]  # 1GHz -> cycles==ns
    med = statistics.median(ns)
    # core count: max distinct cores across invocations (the steady-state K runs)
    ncores = max(len(s) for s in cores.values())
    return med, len(ns), ncores


dtmap = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32, "u32": ttnn.uint32}
dt = dtmap[DT]
dev = ttnn.open_device(device_id=0)
try:
    if DT == "u32":
        t = torch.randint(0, 100000, SHAPE, dtype=torch.int32)
    else:
        t = torch.randn(SHAPE, dtype=torch.float32)
    tin = ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    fn = (
        (lambda x: prod_tilize(x, use_multicore=True))
        if IMPL == "native"
        else (lambda x: gen_tilize(x, use_multicore=True))
    )
    for _ in range(3):
        fn(tin)
        ttnn.synchronize_device(dev)
    for _ in range(K):
        fn(tin)
    ttnn.synchronize_device(dev)
finally:
    ttnn.close_device(dev)  # flush profiler to .logs

med, n, ncores = parse()
print(f"RESULT IMPL={IMPL} DT={DT} SHAPE={SHAPE} median_ns={med:,.0f} n_ops={n} cores={ncores}")
