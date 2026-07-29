"""Refinement 1 / A0 knee sweep: device core-count sweep on the low-work-per-core
regimes. Measures the CORE_CAP_OVERRIDE hook in the planner, so the kernels are
byte-identical across every point -- only the active-core count (and the chunk
width the split derives from it) changes.
"""
import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import statistics
import torch
import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd

_KEY = "DEVICE KERNEL DURATION [ns]"
N_WARMUP, N_TRIALS, N_ROUNDS = 3, 10, 5


def read_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for p in programs:
            res = getattr(p, "program_analyses_results", None) or {}
            e = res.get(_KEY)
            if e is not None:
                total += float(e.duration)
                found = True
    return total if found else None


def measure(device, fn):
    for _ in range(N_WARMUP):
        fn()
    ttnn.synchronize_device(device)
    read_ns(device)
    samples = []
    for _ in range(N_ROUNDS):
        for _ in range(N_TRIALS):
            fn()
        v = read_ns(device)
        if v is not None:
            samples.append(v / N_TRIALS)
    if not samples:
        return None, None
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return statistics.median(samples), std


SHAPES = {
    "d_tall_narrow": (1, 1, 2048, 32),  # nt_h=64 Wt=1  -> 64 tiles
    "n_tall_narrow2": (1, 1, 2048, 64),  # nt_h=64 Wt=2  -> 128 tiles
    "n_wide_short_sm": (1, 1, 32, 4096),  # nt_h=1  Wt=128 -> 128 tiles
    "n_tiny": (1, 1, 64, 128),  # 8 tiles (below the knee)
}
CAPS = [None, 64, 32, 16, 8, 4, 2, 1]

device = ttnn.open_device(device_id=0)
try:
    rows = []
    for name, shape in SHAPES.items():
        torch.manual_seed(0)
        ti = ttnn.from_torch(
            torch.randn(shape).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for cap in CAPS:
            tpd.CORE_CAP_OVERRIDE = cap
            probe_out = ttnn.allocate_tensor_on_device(
                ttnn.Shape(list(shape)),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            plan = tpd.build_plan(ti, probe_out, device, use_multicore=True)
            ns, std = measure(device, lambda t=ti: tilize(t, ttnn.DRAM_MEMORY_CONFIG))
            traffic = plan["folded_h"] * plan["width"] * 2 + plan["total_tiles"] * plan["tile_out"]
            rows.append(
                (
                    name,
                    cap,
                    plan["ncores"],
                    plan["chunk_wt"],
                    plan["cb_bytes_per_core"],
                    ns,
                    std / ns * 100 if ns else 0,
                    traffic / ns if ns else 0,
                )
            )
    tpd.CORE_CAP_OVERRIDE = None

    print("\n=== A0 knee sweep (bf16, DRAM->DRAM, multicore) ===")
    print(f"{'regime':<17}{'cap':>5}{'cores':>6}{'chk':>5}{'cbB':>8}{'ns':>10}{'cv%':>6}{'GB/s':>8}{'vs64':>7}")
    base = {}
    for r in rows:
        if r[1] is None:
            base[r[0]] = r[5]
    for name, cap, cores, chk, cbb, ns, cv, gbps in rows:
        rel = base[name] / ns if ns else 0
        capstr = "auto" if cap is None else str(cap)
        print(f"{name:<17}{capstr:>5}{cores:>6}{chk:>5}{cbb:>8}{ns:>10.0f}{cv:>6.1f}{gbps:>8.1f}{rel:>7.2f}x")
finally:
    ttnn.close_device(device)
