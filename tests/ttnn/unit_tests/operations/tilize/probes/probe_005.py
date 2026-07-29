import torch, ttnn
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
KEY = "DEVICE KERNEL DURATION [ns]"
t = ttnn.from_torch(
    torch.randn(1, 1, 256, 256).bfloat16(),
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=dev,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)


def read():
    ttnn.ReadDeviceProfiler(dev)
    per_chip = ttnn.get_latest_programs_perf_data()
    print("  per_chip type:", type(per_chip), "len:", len(per_chip) if per_chip else None)
    tot, found = 0.0, False
    for programs in (per_chip or {}).values():
        for p in programs:
            r = getattr(p, "program_analyses_results", None) or {}
            print("   prog keys:", list(r.keys())[:6])
            e = r.get(KEY)
            if e is not None:
                tot += float(e.duration)
                found = True
    return tot if found else None


for _ in range(3):
    tilize(t)
ttnn.synchronize_device(dev)
print("flush:", read())
for _ in range(5):
    tilize(t)
print("work:", read())
ttnn.close_device(dev)
