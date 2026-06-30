# Measure WORKER L1 budget per bank on a carved (1,1) denoise chip of a (4,8)
# parent, under FABRIC_1D vs FABRIC_2D (+ optional reduced routing planes), to
# resolve: does FABRIC_2D actually shrink worker L1 (so a fabric-side knob can
# reclaim it), or is the only reclaim model-side (weight -> DRAM)?
#
# Usage: PROBE_FABRIC=1d|2d  PROBE_PLANES=<int or empty>  python _bench_runs/probe_l1_budget.py
import os
import sys
import time
import subprocess

import ttnn

FABRIC = os.environ.get("PROBE_FABRIC", "2d").lower()
PLANES = os.environ.get("PROBE_PLANES", "").strip()
fabric_cfg = ttnn.FabricConfig.FABRIC_2D if FABRIC == "2d" else ttnn.FabricConfig.FABRIC_1D
print(f"[l1probe] FABRIC={FABRIC} PLANES={PLANES or 'default'}", flush=True)


def set_fabric():
    kw = {}
    if PLANES.isdigit():
        kw["num_planes"] = int(PLANES)
    ttnn.set_fabric_config(fabric_cfg, **kw)


def open_parent():
    last = None
    for attempt in range(3):
        try:
            set_fabric()
            return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8), l1_small_size=24576)
        except Exception as e:
            last = e
            print(f"[l1probe] open attempt {attempt} failed: {e!r}", flush=True)
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:
                pass
            try:
                subprocess.run(["tt-smi", "-r"], check=False, timeout=180)
            except FileNotFoundError:
                pass
            time.sleep(10)
    raise last


def dump_l1(name, mesh):
    mv = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    print(
        f"[l1probe] {name}: num_banks={mv.num_banks} "
        f"total/bank={mv.total_bytes_per_bank} "
        f"allocated/bank={mv.total_bytes_allocated_per_bank} "
        f"free/bank={mv.total_bytes_free_per_bank} "
        f"largest_free/bank={mv.largest_contiguous_bytes_free_per_bank}",
        flush=True,
    )


t0 = time.time()
parent = open_parent()
print(f"[l1probe] (4,8) opened in {time.time()-t0:.1f}s", flush=True)

denoise_chip = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(1, 0))
prefill_row = parent.create_submesh(ttnn.MeshShape(1, 8), ttnn.MeshCoordinate(0, 0))

dump_l1("denoise_chip(1,1)@(1,0)", denoise_chip)
dump_l1("prefill_row(1,8)@(0,0)", prefill_row)

ttnn.close_mesh_device(denoise_chip)
ttnn.close_mesh_device(prefill_row)
ttnn.close_mesh_device(parent)
try:
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
except Exception:
    pass
print("[l1probe] done", flush=True)
sys.exit(0)
