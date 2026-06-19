import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc

dev = device
grid = dev.compute_with_storage_grid_size()
print("GRID", grid.x, grid.y, "total", grid.x * grid.y)
KEY = "DEVICE KERNEL DURATION [ns]"


def read_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per = ttnn.get_latest_programs_perf_data()
    tot = 0.0
    found = False
    for progs in (per or {}).values():
        for p in progs:
            r = getattr(p, "program_analyses_results", None) or {}
            e = r.get(KEY)
            if e is None:
                continue
            tot += float(e.duration)
            found = True
    return tot if found else None


def meas(x, iters=7):
    ti = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    rms_norm(ti)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    s = []
    for _ in range(iters):
        rms_norm(ti)
        ttnn.synchronize_device(dev)
        ns = read_ns(dev)
        if ns:
            s.append(ns)
    return min(s) if s else None


cfg = desc._resolve_compute_config(None)


def route(Wt, nrg):
    return desc._select_k(Wt, nrg, grid, grid.x * grid.y, False, ttnn.bfloat16, True, ttnn.bfloat16)


for label, Wt, nrg in [
    ("(1,1,32,8192)", 256, 1),
    ("(1,1,64,8192)", 256, 2),
    ("(256,8192)", 256, 8),
    ("(512,8192)", 256, 16),
    ("(1,1,32,16384)", 512, 1),
    ("(256,16384)", 512, 8),
]:
    K = route(Wt, nrg)
    print(f"{label}: Wt={Wt} nrg={nrg} -> K={K} used={nrg*K if K else None}")
print("=== device time (us) ===")
for shape in [(1, 1, 32, 8192), (1, 1, 64, 8192), (1, 256, 8192), (1, 512, 8192), (1, 1, 32, 16384), (1, 256, 16384)]:
    t = meas(torch.randn(*shape))
    print(f"{shape}: {t/1000:.2f} us" if t else f"{shape}: None")
