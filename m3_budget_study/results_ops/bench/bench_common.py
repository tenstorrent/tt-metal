# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared plumbing for the M3 prefill op microbenchmarks: mesh open, per-op device timing, CSV rows.

Timing (BENCH_TIMING=device, the default): the device profiler without tracy capture. The env below must
be set before ttnn is imported (setup_env). After every timed call the harness syncs, calls
ttnn.ReadDeviceProfiler(submesh) and reads ttnn.get_latest_programs_perf_data(); it sums
"DEVICE KERNEL DURATION [ns]" over the programs that window added on each chip. Programs are de-duplicated
by program_execution_uid, so a chip whose "latest" set was not refreshed is not counted twice. Per chip the
median over repeats is taken, then worst / mean / min across the chips of the sub-mesh.

BENCH_TIMING=wall (or no perf data at all): ttnn.synchronize_device around each call; worst / mean / min
are then over repeats (no per-chip split).
"""

import csv
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DURATION_KEYS = ("DEVICE KERNEL DURATION [ns]", "DEVICE FW DURATION [ns]")

# M3 dims (models/demos/deepseek_v3_d_p/reference/minimax_m3_config.py, configs/MiniMax-M3/config.json).
EMB = 6144
MOE_INTER = 3072
N_EXPERTS = 128
TOPK = 4
N_Q_HEADS = 64
N_KV_HEADS = 4
HEAD_DIM = 128
MSA_BLOCK = 128
MSA_TOPK_BLOCKS = 16
N_INDEX_HEADS = 4
INDEX_DIM = 128
SUBMESH = (2, 4)  # SP=2 on axis 0, TP=4 on axis 1, EP=8
SP, TP = SUBMESH

# Bytes per element (tile layout incl. shared exponents: 1024 elems -> 512 B + 64 B for bfp4, 1024 + 64 for bfp8).
BPE = {"bf16": 2.0, "bf8": 1088 / 1024, "bf4": 576 / 1024, "u32": 4.0, "u16": 2.0}


def timing_mode():
    return os.getenv("BENCH_TIMING", "device").strip().lower()


def setup_env():
    """Profiler env for the programmatic perf-data API. Call before `import ttnn`."""
    if timing_mode() == "device":
        os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
        os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
        os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
        # No ops CSV / profile_log_device.csv on disk: the API is the only consumer.
        os.environ.setdefault("TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES", "1")
    os.environ.setdefault(
        "TT_MESH_GRAPH_DESC_PATH",
        str(REPO / "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"),
    )


def open_submesh():
    """8x4 galaxy with FABRIC_1D, returns (galaxy, first (2,4) sub-mesh)."""
    import ttnn
    from models.demos.minimax_m3.tt.ccl import L1_SMALL_SIZE

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    galaxy = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), l1_small_size=L1_SMALL_SIZE)
    sub = galaxy.create_submeshes(ttnn.MeshShape(*SUBMESH))[0]
    print(
        f"[bench] galaxy {tuple(galaxy.shape)} -> sub-mesh {tuple(sub.shape)} ndev={sub.get_num_devices()}", flush=True
    )
    return galaxy, sub


def close_mesh(galaxy):
    import ttnn

    # Sub-meshes first: closing the parent with a live child queue is refused.
    for sub in galaxy.get_submeshes():
        ttnn.close_mesh_device(sub)
    ttnn.close_mesh_device(galaxy)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _deallocate(obj):
    import ttnn

    if obj is None:
        return
    if isinstance(obj, (list, tuple)):
        for o in obj:
            _deallocate(o)
        return
    if isinstance(obj, ttnn.Tensor):
        try:
            ttnn.deallocate(obj)
        except Exception:
            pass


class OpTimer:
    """Per-call device kernel time on every chip of `mesh` (see module docstring)."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.mode = timing_mode()
        try:
            self.chip_ids = set(int(i) for i in mesh.get_device_ids())
        except Exception:
            self.chip_ids = None
        self._seen = set()
        if self.mode == "device":
            self._read()  # drain whatever setup enqueued (weight uploads, tilizes)

    def _read(self):
        """Sync + profiler read; returns {chip: [(duration_ns, core_count), ...]} of programs new since last read."""
        import ttnn

        ttnn.synchronize_device(self.mesh)
        ttnn.ReadDeviceProfiler(self.mesh)
        data = ttnn.get_latest_programs_perf_data() or {}
        out = {}
        for chip, programs in data.items():
            chip = int(chip)
            if self.chip_ids is not None and chip not in self.chip_ids:
                continue
            for p in programs:
                uid = p.program_execution_uid
                key = (chip, uid.runtime_id, uid.trace_id, uid.trace_id_counter)
                if key in self._seen:
                    continue
                self._seen.add(key)
                dur = None
                for k in DURATION_KEYS:
                    if k in p.program_analyses_results:
                        dur = p.program_analyses_results[k].duration
                        break
                if dur is None:
                    continue
                out.setdefault(chip, []).append((int(dur), int(p.core_count)))
        return out

    def measure(self, fn, warmup=2, repeats=5):
        """fn() -> output(s) (deallocated after each call). Returns a result dict for the CSV row."""
        import ttnn

        for _ in range(warmup):
            out = fn()
            if self.mode == "device":
                self._read()
            else:
                ttnn.synchronize_device(self.mesh)
            _deallocate(out)

        walls, windows = [], []
        for _ in range(repeats):
            ttnn.synchronize_device(self.mesh)
            t0 = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(self.mesh)
            walls.append((time.perf_counter() - t0) * 1e3)
            if self.mode == "device":
                windows.append(self._read())
            _deallocate(out)

        res = {"wall_ms": statistics.mean(walls)}
        if self.mode == "device" and any(windows):
            chips = sorted(set().union(*[w.keys() for w in windows]))
            per_chip = {}
            for c in chips:
                sums = [sum(d for d, _ in w.get(c, [])) for w in windows]
                per_chip[c] = statistics.median(sums) / 1e6
            vals = list(per_chip.values())
            n_prog = statistics.median([len(w.get(chips[0], [])) for w in windows])
            cores = max((cc for w in windows for progs in w.values() for _, cc in progs), default=0)
            res.update(
                timing_src="device",
                worst_ms=max(vals),
                mean_ms=statistics.mean(vals),
                min_ms=min(vals),
                n_chips=len(vals),
                n_programs=n_prog,
                cores_used=cores,
                per_chip_ms=";".join(f"{c}:{v:.4f}" for c, v in per_chip.items()),
            )
        else:
            if self.mode == "device":
                print("[bench] WARNING: no device perf data returned; falling back to wall time", flush=True)
            res.update(
                timing_src="wall",
                worst_ms=max(walls),
                mean_ms=statistics.mean(walls),
                min_ms=min(walls),
                n_chips="",
                n_programs="",
                cores_used="",
                per_chip_ms="",
            )
        return res


class CsvOut:
    """Row-at-a-time CSV (flushed per row, so a timeout keeps the finished points)."""

    def __init__(self, path, columns):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = list(columns)
        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.columns, extrasaction="ignore")
        self._w.writeheader()
        self._f.flush()

    def row(self, **kw):
        clean = {}
        for k in self.columns:
            v = kw.get(k, "")
            clean[k] = f"{v:.6g}" if isinstance(v, float) else v
        self._w.writerow(clean)
        self._f.flush()
        print("[bench] " + ",".join(str(clean[k]) for k in self.columns), flush=True)

    def close(self):
        self._f.close()


TIMING_COLUMNS = ["timing_src", "n_chips", "n_programs", "wall_ms", "per_chip_ms", "status"]


def add_common_args(p, default_csv):
    p.add_argument("--dry-run", action="store_true", help="print the grid and exit (no ttnn import, no device)")
    p.add_argument("--out", default=str(HERE / default_csv))
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    return p


def int_list(s):
    return [int(x) for x in s.split(",") if x]
