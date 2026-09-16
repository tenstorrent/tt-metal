#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TopK campaign on the fresh main-tip build (T2.6 driver; plan in sweep_plan.md).

NOT executed on hardware by the author. Host-side checks only: `python3 -m py_compile` and
`--list` (which imports neither torch nor ttnn).

Run (one process per class group, options BEFORE the script path for tracy):
    cd $TT_METAL_HOME
    python -m tracy -r -p -v $HANDOFF/topk/topk_campaign.py \
        --classes L1,L2,L3,L4,L5 --out $HANDOFF/data/topk
    python -m tracy -r -p --perf-counter-multipass --profiler-capture-perf-counters=all \
        topk_campaign.py --classes COUNTERS --iters 2 --out .../data/topk
Then join the tracy ops report to the cells:
    python3 topk_campaign.py --postprocess generated/profiler/reports/<date>/ops_perf_results_<date>.csv \
        .../data/topk/cells_L1_L2_L3_L4_L5.csv

Every duration cell: program cache ON, trace OFF, 1 cache-warming call + 2 warmups discarded,
`--iters` (default 3) timed calls, ttnn.synchronize_device after each call. Per-op device kernel
duration comes from the tracy ops CSV (`DEVICE KERNEL DURATION [ns]`), not from host timers.
Any exception in a cell is recorded (first 400 chars) and the run continues.

Cell classes (sweep_plan.md section 3): L1..L6 (topk_large_indices), A..G (generic ttnn.topk),
R1..R4 (gates, moe_grouped_topk, sort, indexer_score_dsa), COUNTERS, SMOKE.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

# Two layouts (PORTABLE_CONTRACT.md): this file in $WORK/handoff/revamp/topk, where --out defaults to
# $HANDOFF/data/topk; or copied into a tt-metal checkout, where it defaults to $PWD, never the repo tree.
_SD = Path(__file__).resolve().parent
_REPO = next((p for p in (_SD, *_SD.parents) if (p / "ttnn").is_dir() and (p / "tt_metal").is_dir()), None)
HANDOFF = os.environ.get("HANDOFF", str(_SD.parent))
DEFAULT_OUT = os.getcwd() if _REPO else os.path.join(HANDOFF, "data", "topk")

# --------------------------------------------------------------------------------------
# Route prediction (mirrors origin/main 2dbd14bf632: topk.cpp:272-364, topk_device_operation.cpp:97-159,
# topk_utils.cpp:27-44, topk_constants.hpp:13-31, topk_large_indices_program_factory.cpp:33-41,130-143)
# --------------------------------------------------------------------------------------


def snap_k(k: int) -> int:
    return 512 if k <= 512 else (1024 if k <= 1024 else 2048)


def li_body_mode(k: int, n_phys: int) -> str:
    kk = snap_k(k)
    if kk >= 1024:
        return "FusedSegmented"
    return "FusedEndToEnd" if math.ceil(n_phys / kk) <= 32 else "Classic"


def is_pow2(w: int) -> bool:
    return w > 0 and (w & (w - 1)) == 0


def structurally_eligible(w: int, ht: int, k: int) -> bool:
    width_gate = w >= 8192 or (ht <= 2 and w >= 1024)
    return width_gate and w < 65535 and is_pow2(w) and k <= 64


def predict_topk_path(
    rows,
    n,
    k,
    dtype="bf16",
    layout="TILE",
    largest=True,
    stable=False,
    sub_core_grids=None,
    indices_tensor=False,
    preallocated=False,
    sharded=False,
    arch="BLACKHOLE",
    grid=(11, 10),
):
    """Returns (path, [expected OP CODE substrings in dispatch order])."""
    w_pad = max(64, 32 * math.ceil(n / 32))
    ht = math.ceil(rows / 32)
    k16 = 16 * math.ceil(k / 16)
    k32 = 32 * math.ceil(k / 32)
    route = largest and not stable and not (indices_tensor or preallocated or sub_core_grids is not None)
    route = route and not sharded and k <= 2048
    if route and k <= 64:
        route = (not structurally_eligible(w_pad, 3, k)) and w_pad >= 4096
    route = route and dtype == "bf16" and layout == "TILE" and arch == "BLACKHOLE"
    route = route and k16 <= n <= (1 << 19)
    if route:
        kk = snap_k(k16)
        ops = [
            "TopkRoutePrep",
            f"TopkLargeIndices[K'={kk},mode={li_body_mode(k16, n)},chunks={math.ceil(n / kk)}]",
            "TopkRouteFinish",
        ] + (["Slice"] if k16 != k else [])
        return "composite", ops
    gx, gy = grid if sub_core_grids is None else sub_core_grids
    mc = structurally_eligible(w_pad, ht, k32) and gx >= 2 and gy >= 3 and (gx - 1) * (gy - 2) >= 2
    pre = (["Pad"] if n < 64 else []) + ["FillImplicitTilePadding?"]
    post = ["Slice"] if (k32 != k or w_pad != n) else []
    if mc:
        return "stock_multi_core", pre + [f"TopK[multi,Wt={w_pad // 32},Kt={k32 // 32}]"] + post
    cores = min(ht, gx * gy)
    return "stock_single_core", pre + [f"TopK[single,Ht={ht},Wt={w_pad // 32},Kt={k32 // 32},cores={cores}]"] + post


# --------------------------------------------------------------------------------------
# Cell enumeration (sweep_plan.md section 3). C = compute grid core count, read from the device at
# run time; for --list a nominal C_NOMINAL is used so counts are exact and shapes are illustrative.
# --------------------------------------------------------------------------------------

C_NOMINAL = 110  # p100a 11x10 worker grid measured in campaign_b (data/b_topk_r1.log)


def _cell(cls, op, iters=3, **params):
    return {"cls": cls, "op": op, "iters": iters, "params": params}


EXCLUDE_OPS = set()  # ops skipped at run time (--exclude-op), e.g. indexer on main tip


def enumerate_cells(C: int = C_NOMINAL):
    cells = []

    # ---- Grid L: topk_large_indices ------------------------------------------------------
    ns = [256, 512, 1024, 4096, 5000, 5300, 16384, 65536, 131072]
    ks = [16, 32, 64, 256, 512, 1024, 2048]
    for n in ns:
        for k in ks:
            if k <= n:
                cells.append(_cell("L1", "li", R=1, N=n, K=k))
    for n, k in [
        (16896, 512),
        (32768, 512),
        (32768, 1024),
        (33792, 1024),
        (67584, 2048),
        (262144, 2048),
        (65536, 528),
        (65536, 1040),
    ]:
        cells.append(_cell("L2", "li", R=1, N=n, K=k))
    for r in [8, 32, C, C + 1, 2 * C]:
        cells.append(_cell("L3", "li", R=r, N=65536, K=2048))
    for n in [4096, 16384]:
        cells.append(_cell("L3", "li", R=C, N=n, K=2048))
    cells.append(_cell("L4", "li", iters=6, R=640, N=51200, K=1536, pin_ns=1_286_400))
    cells.append(_cell("L4", "li", iters=6, R=2, N=102400, K=1536, valid_length=56320, pin_ns=242_560))
    cells.append(_cell("L4", "li", R=1, N=131072, K=2048, valid_length=65536))
    for n in [4096, 16384, 65536]:
        for k in [512, 2048]:
            cells.append(_cell("L5", "li", R=C, N=n, K=k, mem="l1"))
    for r, n, k, vl in [
        (2048, 8192, 1024, None),
        (2048, 8192, 2048, None),
        (2048, 131072, 2048, None),
        (160, 1048576, 2048, 524288),
        (640, 440, 16, None),
        (1, 8400, 304, None),
    ]:
        p = dict(R=r, N=n, K=k)
        if vl is not None:
            p["valid_length"] = vl
        cells.append(_cell("L6", "li", **p))

    # ---- Grid G: generic ttnn.topk -------------------------------------------------------
    a_ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65504, 65568, 131072]
    a_ks = [32, 64, 128, 256, 512, 2048]
    for n in a_ns:
        for k in a_ks:
            if k <= n:
                cells.append(_cell("A", "topk", rows=32, N=n, K=k, grid="single"))
    for k in [5, 50]:
        cells.append(_cell("A", "topk", rows=32, N=4096, K=k, grid="single"))
    for n in [4096, 65504]:
        cells.append(_cell("B", "topk", rows=32, N=n, K=32, grid="single", indices_tensor=True))
    for n in [1024, 4096, 16384]:
        for k in [32, 256]:
            cells.append(_cell("B", "topk", rows=32, N=n, K=k, grid="single", mem="l1"))
    for ht in [1, 2, 4, 8, C // 2, C, C + 1, 2 * C, 4 * C]:
        cells.append(_cell("C", "topk", rows=32 * ht, N=4224, K=32, grid="full"))
    for dt in ["bfp8", "fp32"]:
        for n in [1024, 8192, 65536]:
            for k in [32, 256]:
                cells.append(_cell("D", "topk", rows=32, N=n, K=k, grid="single", dtype=dt))
    for k in [32, 256]:
        cells.append(_cell("D", "topk", rows=32, N=4096, K=k, grid="single", largest=False))
    e_ns = [1024, 2048, 4096, 8192, 16384, 32768]
    for n in e_ns:
        for k in [32, 64]:
            cells.append(_cell("E", "topk", rows=32, N=n, K=k, grid="full"))
    for g in [(2, 3), (3, 4), (5, 6), (9, 10)]:
        cells.append(_cell("E", "topk", rows=32, N=16384, K=32, grid=g))
    for ht in [2, 4]:
        cells.append(_cell("E", "topk", rows=32 * ht, N=16384, K=32, grid="full"))
    cells.append(_cell("E", "topk", rows=128, N=4096, K=32, grid="full"))
    for n in e_ns:
        for k in [32, 64]:
            cells.append(_cell("F", "topk", rows=32, N=n, K=k, grid=None))
    for k in [128, 512, 2048]:
        for n in [8192, 65536, 262144]:
            cells.append(_cell("F", "topk", rows=32, N=n, K=k, grid=None))
    g_cells = [
        dict(rows=32, N=128, K=4, grid=None, tag="M1 gpt-oss decode DRAM"),
        dict(rows=32, N=128, K=4, grid=None, mem="l1", tag="M1 gpt-oss decode L1"),
        dict(rows=4096, N=128, K=4, grid=None, tag="M2 deepseek d_p gpt-style prefill"),
        dict(rows=32, N=512, K=10, grid=None, tag="M4 qwen3.5 fallback"),
        dict(rows=32, N=256, K=32, grid=None, tag="S5 log-probs"),
        dict(rows=32, N=64128, K=32, grid="full", tag="S4 sampling1d llama3 stock"),
        dict(rows=32, N=75968, K=32, grid="full", tag="S4 sampling1d qwen3 stock u32"),
        dict(rows=32, N=128256, K=64, grid=None, sampling_query=True, tag="S1 ttsampling llama3"),
        dict(rows=32, N=151936, K=128, grid=None, sampling_query=True, tag="S1b ttsampling qwen3"),
        dict(rows=32, N=256000, K=128, grid=None, sampling_query=True, tag="S1c ttsampling gemma"),
        dict(rows=32, N=151936, K=128, grid="full", tag="S1b stock contrast Kt=4 one core"),
        dict(rows=32, N=16032, K=32, grid=None, sampling_query=True, tag="S3 shard non-pow2"),
        dict(rows=32, N=16384, K=32, grid=None, sampling_query=True, tag="S3 shard pow2"),
        dict(rows=256, N=96, K=32, grid=None, tag="M6 informer"),
        dict(rows=4096, N=384, K=8, grid=None, tag="M2 kimi ungrouped"),
        dict(rows=1024, N=128, K=8, grid=None, tag="M3 gemma4"),
    ]
    for p in g_cells:
        cells.append(_cell("G", "topk", **p))

    # ---- Grid R: routers, sort, indexer --------------------------------------------------
    for b in [1, 32, C]:
        for k in [4, 8]:
            for sm in [False, True]:
                cells.append(_cell("R1", "gmg", B=b, K=k, softmax=sm, experts=256))
    for sig in [False, True]:
        cells.append(_cell("R1", "gmg", B=32, K=8, softmax=False, experts=256, sigmoid=sig, tag="sigmoid toggle"))
    for sm in [False, True]:
        cells.append(_cell("R1", "gmg", B=32, K=8, softmax=sm, experts=512))
    for b in [1, 32, C]:
        cells.append(_cell("R1", "dmg", B=b))
    for b in [32, 128, 512]:
        for sm in [False, True]:
            cells.append(_cell("R1", "gate_wrapper", B=b, K=8, N=128, softmax=sm))
    for t in [32, 128, 1024, 4096]:
        cells.append(_cell("R2", "mgt", T=t, N=128, K=4, groups=1))
    for t in [32, 1024, 4096]:
        cells.append(_cell("R2", "mgt", T=t, N=256, K=8, groups=8))
    cells.append(_cell("R2", "mgt", T=4096, N=384, K=8, groups=1))
    for w in [512, 2048, 8192]:
        cells.append(_cell("R3", "sort", rows=32, N=w))
    for sq, t in [(2048, 8192), (2048, 32768), (512, 65536)]:
        cells.append(_cell("R4", "indexer", Sq=sq, T=t, Hi=64, D=128))

    # ---- Counter cells (multipass) ---------------------------------------------------------
    for n in [16384, 131072]:
        for k in [512, 2048]:
            cells.append(_cell("COUNTERS", "li", iters=2, R=C, N=n, K=k))
    cells.append(_cell("COUNTERS", "topk", iters=2, rows=32, N=4096, K=32, grid="single"))
    cells.append(_cell("COUNTERS", "topk", iters=2, rows=32, N=4096, K=512, grid="single"))
    cells.append(_cell("COUNTERS", "topk", iters=2, rows=32, N=16384, K=32, grid="full"))
    cells.append(_cell("COUNTERS", "topk", iters=2, rows=32, N=65536, K=128, grid=None))
    cells.append(_cell("COUNTERS", "gmg", iters=2, B=C, K=8, softmax=False, experts=256))
    cells.append(_cell("COUNTERS", "mgt", iters=2, T=4096, N=128, K=4, groups=1))
    cells.append(_cell("COUNTERS", "indexer", iters=2, Sq=2048, T=8192, Hi=64, D=128))

    # ---- Smoke: one cell per op ------------------------------------------------------------
    cells.append(_cell("SMOKE", "li", R=1, N=4096, K=32))
    cells.append(_cell("SMOKE", "topk", rows=32, N=4096, K=32, grid="single"))
    cells.append(_cell("SMOKE", "topk", rows=32, N=65536, K=128, grid=None))
    cells.append(_cell("SMOKE", "gmg", B=1, K=8, softmax=False, experts=256))
    cells.append(_cell("SMOKE", "mgt", T=32, N=128, K=4, groups=1))
    cells.append(_cell("SMOKE", "sort", rows=32, N=512))
    cells.append(_cell("SMOKE", "indexer", Sq=512, T=2048, Hi=64, D=128))

    for i, c in enumerate(cells):
        c["cell_id"] = f"{c['cls']}-{i:03d}-{c['op']}-" + "-".join(
            f"{k}{v}" for k, v in c["params"].items() if k not in ("tag", "pin_ns", "sampling_query")
        )
        c["predicted"] = predict(c, grid=(11, 10))
    return cells


def predict(cell, grid):
    p = cell["params"]
    if cell["op"] == "li":
        kk = snap_k(p["K"])
        n_valid = p.get("valid_length", p["N"])
        return json.dumps(
            {
                "path": "topk_large_indices_direct",
                "Kp": kk,
                "mode": li_body_mode(p["K"], p["N"]),
                "chunks": math.ceil(n_valid / kk),
                "rows_per_core": math.ceil(p["R"] / (grid[0] * grid[1])),
            }
        )
    if cell["op"] == "topk":
        g = p.get("grid")
        scg = None if g is None else ((1, 1) if g == "single" else (grid if g == "full" else tuple(g)))
        path, ops = predict_topk_path(
            p["rows"],
            p["N"],
            p["K"],
            dtype=p.get("dtype", "bf16"),
            largest=p.get("largest", True),
            sub_core_grids=scg,
            indices_tensor=p.get("indices_tensor", False),
            grid=grid,
        )
        return json.dumps({"path": path, "ops": ops})
    return json.dumps({"path": cell["op"]})


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------


def _git(args, cwd):
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e.__class__.__name__})"


def provenance_line(device=None, extra=""):
    home = os.environ.get("TT_METAL_HOME", os.getcwd())
    sha = _git(["rev-parse", "HEAD"], home)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], home)
    dirty = _git(["status", "--porcelain", "--untracked-files=no"], home)
    brisc = "n/a"
    try:
        with open(os.path.join(home, "tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h")) as f:
            for line in f:
                if "define MEM_BRISC_FIRMWARE_SIZE" in line:
                    brisc = line.strip()
    except OSError:
        pass
    dev = ""
    if device is not None:
        try:
            g = device.compute_with_storage_grid_size()
            dev = f" arch={device.arch()} grid={g.x}x{g.y} cores={g.x * g.y}"
        except Exception:  # noqa: BLE001
            dev = " device=unreadable"
    counters = os.environ.get("TT_METAL_PROFILE_PERF_COUNTERS", "")
    return (
        f"# PROVENANCE git_sha={sha} branch={branch} dirty={'yes' if dirty else 'no'} TT_METAL_HOME={home}"
        f"{dev} host={platform.node()} date={_dt.datetime.utcnow().isoformat()}Z"
        f" cmd={' '.join(sys.argv)} perf_counters_env={counters!r} {brisc} {extra}"
    )


# --------------------------------------------------------------------------------------
# Device-side cell runners (only imported/called when actually running; torch/ttnn lazy)
# --------------------------------------------------------------------------------------


class Runner:
    def __init__(self, device, iters, warmups=2):
        import torch  # noqa: F401
        import ttnn  # noqa: F401

        self.ttnn = ttnn
        self.torch = torch
        self.device = device
        self.iters = iters
        self.warmups = warmups
        g = device.compute_with_storage_grid_size()
        self.grid = (g.x, g.y)
        self.C = g.x * g.y

    # -- helpers ------------------------------------------------------------------------
    def core_grid(self, spec):
        ttnn = self.ttnn
        if spec is None:
            return None
        gx, gy = (1, 1) if spec == "single" else (self.grid if spec == "full" else tuple(spec))
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])

    def memcfg(self, mem):
        return self.ttnn.L1_MEMORY_CONFIG if mem == "l1" else self.ttnn.DRAM_MEMORY_CONFIG

    def loop(self, fn, iters):
        """1 cache-warming call + warmups discarded, then `iters` timed calls; each synchronized."""
        outs = []
        for _ in range(1 + self.warmups + iters):
            out = fn()
            self.ttnn.synchronize_device(self.device)
            for t in out if isinstance(out, (list, tuple)) else [out]:
                try:
                    t.deallocate()
                except Exception:  # noqa: BLE001
                    pass
        return outs

    # -- topk_large_indices ----------------------------------------------------------------
    def run_li(self, p):
        ttnn, torch = self.ttnn, self.torch
        x = torch.randn(1, 1, p["R"], p["N"], dtype=torch.bfloat16)
        tx = ttnn.from_torch(
            x, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, memory_config=self.memcfg(p.get("mem"))
        )
        kw = {"k": p["K"]}
        if "valid_length" in p:
            kw["valid_length"] = p["valid_length"]
        self.loop(lambda: ttnn.experimental.topk_large_indices(tx, **kw), p["_iters"])
        tx.deallocate()
        return {}

    # -- generic ttnn.topk -------------------------------------------------------------------
    def run_topk(self, p):
        ttnn, torch = self.ttnn, self.torch
        dt = p.get("dtype", "bf16")
        tdtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32}[dt]
        x = torch.randn(1, 1, p["rows"], p["N"], dtype=torch.float32)
        tx = ttnn.from_torch(
            x, tdtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=self.memcfg(p.get("mem"))
        )
        kw = dict(dim=-1, largest=p.get("largest", True), sorted=True)
        scg = self.core_grid(p.get("grid"))
        if scg is not None:
            kw["sub_core_grids"] = scg
        extra = {}
        if p.get("indices_tensor"):
            iota = torch.arange(p["N"], dtype=torch.int32).view(1, 1, 1, -1).expand(1, 1, p["rows"], -1)
            extra["indices_tensor"] = ttnn.from_torch(
                iota.contiguous(),
                ttnn.uint16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            kw["indices_tensor"] = extra["indices_tensor"]
        info = {}
        if p.get("sampling_query"):
            try:
                info["sampling_query_routes"] = bool(
                    ttnn._ttnn.operations.reduction._sampling_topk_would_route_to_large_indices(tx, p["K"])
                )
            except Exception as e:  # noqa: BLE001
                info["sampling_query_routes"] = f"error: {e.__class__.__name__}"
        self.loop(lambda: ttnn.topk(tx, p["K"], **kw), p["_iters"])
        tx.deallocate()
        for t in extra.values():
            t.deallocate()
        return info

    # -- gates (tensor construction per test_generalized_moe_gate_program_cache.py:29-98) ------
    def _gate_tensors(self, B, experts):
        ttnn, torch = self.ttnn, self.torch
        blocks = (
            experts // 256
        )  # 1 -> (B,16,16) shards (32,32); 2 -> (B,2,16,16) shards (64,32), tt_moe_gate.py:532-545
        in_shape = (B, 16, 16) if blocks == 1 else (B, 2, 16, 16)
        shard = (32, 32) if blocks == 1 else (64, 32)
        g = self.device.compute_with_storage_grid_size()
        core_grid = ttnn.num_cores_to_corerangeset(B, ttnn.CoreCoord(g.x, g.y), row_wise=True)
        in_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
        )
        out_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
        )
        tile = ttnn.Tile((32, 32))
        xin = torch.sigmoid(2 * torch.rand(in_shape, dtype=torch.bfloat16) - 1)
        bias = torch.transpose(2 * torch.rand(in_shape, dtype=torch.bfloat16) - 1, -2, -1)
        idx = torch.arange(experts, dtype=torch.int32).unsqueeze(0).expand(B, -1).reshape(in_shape)
        idx = torch.transpose(idx, -2, -1).to(torch.uint16)
        mk = lambda t, dt, mem: ttnn.from_torch(
            t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=mem, tile=tile  # noqa: E731
        )
        return dict(
            x=mk(xin, ttnn.bfloat16, in_mem),
            bias=mk(bias, ttnn.bfloat16, in_mem),
            idx=mk(idx, ttnn.uint16, in_mem),
            out=mk(torch.zeros((B, 1, 16), dtype=torch.bfloat16), ttnn.bfloat16, out_mem),
            out_idx=mk(torch.zeros((B, 1, 16), dtype=torch.uint16), ttnn.uint16, out_mem),
        )

    def run_gmg(self, p):
        ttnn = self.ttnn
        t = self._gate_tensors(p["B"], p["experts"])
        fn = lambda: ttnn.experimental.deepseek.moe.generalized_moe_gate(  # noqa: E731
            t["x"],
            bias_tensor=t["bias"],
            input_indices_tensor=t["idx"],
            output_tensor=t["out"],
            output_indices_tensor=t["out_idx"],
            eps=1e-20,
            scaling_factor=2.5,
            enable_sigmoid=p.get("sigmoid", False),
            topk=p["K"],
            output_softmax=p["softmax"],
            grouped=False,
        )
        # outputs are the preallocated tensors: do not deallocate per call
        for _ in range(1 + self.warmups + p["_iters"]):
            fn()
            ttnn.synchronize_device(self.device)
        for v in t.values():
            v.deallocate()
        return {}

    def run_dmg(self, p):
        ttnn = self.ttnn
        t = self._gate_tensors(p["B"], 256)
        for _ in range(1 + self.warmups + p["_iters"]):
            ttnn.experimental.deepseek.moe.deepseek_moe_gate(
                t["x"],
                bias_tensor=t["bias"],
                input_indices_tensor=t["idx"],
                output_tensor=t["out"],
                output_indices_tensor=t["out_idx"],
                eps=1e-20,
                scaling_factor=2.5,
                enable_sigmoid=True,
            )
            ttnn.synchronize_device(self.device)
        for v in t.values():
            v.deallocate()
        return {}

    def run_gate_wrapper(self, p):
        """TTMoEGate.forward at B tokens (loops ceil(B/C) launches inside, tt_moe_gate.py:523-526).
        Config fields per models/common/modules/moe/tt_moe_gate_config.py; adjust names if they drifted."""
        ttnn, torch = self.ttnn, self.torch
        from models.common.modules.moe.tt_moe_gate import TTMoEGate  # noqa: PLC0415
        from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig  # noqa: PLC0415

        H = 2048
        cfg = TTMoEGateConfig(
            num_routed_experts=p["N"],
            select_experts_k=p["K"],
            hidden_size=H,
            batch_per_device=p["B"],
            n_group=1,
            score_func="softmax" if p["softmax"] else "sigmoid",
        )
        W = torch.randn(H, p["N"], dtype=torch.bfloat16)
        # No score-correction bias: these cells are the n_group=1 generalized gate, and the config leaves
        # score_correction_bias False, which forbids passing torch_gate_bias (it is the deepseek path).
        gate = TTMoEGate(self.device, cfg, W)
        x = ttnn.from_torch(
            torch.randn(1, 1, p["B"], H, dtype=torch.bfloat16),
            ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.loop(lambda: gate.forward(x), p["_iters"])
        x.deallocate()
        return {}

    # -- moe_grouped_topk ---------------------------------------------------------------------
    def run_mgt(self, p):
        ttnn, torch = self.ttnn, self.torch
        scores = ttnn.from_torch(
            torch.randn(p["T"], p["N"], dtype=torch.bfloat16),
            ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        bias = ttnn.from_torch(
            torch.randn(p["T"], p["N"], dtype=torch.bfloat16) * 0.1,
            ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        if p["groups"] == 1:
            fn = lambda: ttnn.experimental.deepseek_prefill.moe_grouped_topk(  # noqa: E731
                scores,
                bias,
                n_groups=1,
                summed_experts_per_group=1,
                topk_groups=1,
                n_activated_experts=p["K"],
                route_scale=2.0,
                epsilon=1e-20,
            )
        else:
            fn = lambda: ttnn.experimental.deepseek_prefill.moe_grouped_topk(  # noqa: E731
                scores, bias, n_groups=8, summed_experts_per_group=2, topk_groups=4, n_activated_experts=8
            )
        self.loop(fn, p["_iters"])
        scores.deallocate()
        bias.deallocate()
        return {}

    # -- sort ------------------------------------------------------------------------------------
    def run_sort(self, p):
        ttnn, torch = self.ttnn, self.torch
        x = ttnn.from_torch(
            torch.randn(1, 1, p["rows"], p["N"], dtype=torch.bfloat16),
            ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.loop(lambda: ttnn.sort(x, dim=-1, descending=True), p["_iters"])
        x.deallocate()
        return {}

    # -- indexer_score_dsa (analysis/indexer_probe.py construction) --------------------------------
    def run_indexer(self, p):
        ttnn, torch = self.ttnn, self.torch
        hi, d, sq, t = p["Hi"], p["D"], p["Sq"], p["T"]
        q = ttnn.from_torch(
            torch.randn(1, hi, sq, d, dtype=torch.bfloat16), device=self.device, layout=ttnn.TILE_LAYOUT
        )
        k = ttnn.from_torch(torch.randn(1, 1, t, d, dtype=torch.bfloat16), device=self.device, layout=ttnn.TILE_LAYOUT)
        w = ttnn.from_torch(
            torch.randn(1, hi, sq, 1, dtype=torch.bfloat16), device=self.device, layout=ttnn.TILE_LAYOUT
        )
        self.loop(lambda: ttnn.experimental.indexer_score_dsa(q, k, w, chunk_start_idx=t - sq), p["_iters"])
        for x in (q, k, w):
            x.deallocate()
        return {}

    def run(self, cell):
        p = dict(cell["params"])
        p["_iters"] = cell["iters"]
        # replace the nominal C with the measured grid for row-count cells
        return getattr(self, f"run_{cell['op']}")(p)


def rebind_core_count(cells, C):
    """Cells enumerated with C_NOMINAL are re-enumerated with the measured core count so R/Ht/B track C."""
    return enumerate_cells(C)


# --------------------------------------------------------------------------------------
# Main run loop
# --------------------------------------------------------------------------------------

CELL_FIELDS = ["cell_id", "cls", "op", "params", "iters", "predicted", "status", "info", "error", "host_seconds"]


def run_campaign(classes, iters_override, out_dir):
    import time  # noqa: PLC0415
    import ttnn  # noqa: PLC0415

    try:
        from tracy import signpost  # noqa: PLC0415
    except Exception:  # noqa: BLE001

        def signpost(header, message=None):  # noqa: ARG001
            print(f"[signpost] {header} {message}", flush=True)

    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()
    C = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    cells = [c for c in enumerate_cells(C) if c["cls"] in classes and c["op"] not in EXCLUDE_OPS]
    os.makedirs(out_dir, exist_ok=True)
    tag = "_".join(classes)
    cells_csv = os.path.join(out_dir, f"cells_{tag}.csv")
    runner = Runner(device, iters=3)
    with open(cells_csv, "w", newline="") as f:
        f.write(provenance_line(device, extra=f"classes={tag}") + "\n")
        wr = csv.DictWriter(f, fieldnames=CELL_FIELDS)
        wr.writeheader()
        for c in cells:
            if iters_override:
                c["iters"] = iters_override
            row = {k: c.get(k, "") for k in CELL_FIELDS}
            row["params"] = json.dumps(c["params"], sort_keys=True)
            t0 = time.time()
            print(f"[topk_campaign] BEGIN {c['cell_id']} iters={c['iters']}", flush=True)
            signpost("CELL_BEGIN", c["cell_id"])
            try:
                info = runner.run(c)
                row["status"] = "ok"
                row["info"] = json.dumps(info, sort_keys=True)
            except Exception as e:  # noqa: BLE001  (TT_FATAL -> RuntimeError, OOM included)
                row["status"] = "error"
                row["error"] = (e.__class__.__name__ + ": " + str(e).replace("\n", " "))[:400]
                traceback.print_exc()
                try:
                    ttnn.synchronize_device(device)
                except Exception:  # noqa: BLE001
                    pass
            signpost("CELL_END", c["cell_id"])
            row["host_seconds"] = f"{time.time() - t0:.1f}"
            wr.writerow(row)
            f.flush()
            print(f"[topk_campaign] END {c['cell_id']} {row['status']} {row['host_seconds']}s", flush=True)
    ttnn.close_device(device)
    print(f"[topk_campaign] wrote {cells_csv}", flush=True)


# --------------------------------------------------------------------------------------
# Post-processing: join the tracy ops report to the cells by signpost order
# --------------------------------------------------------------------------------------

RESULT_FIELDS = [
    "cell_id",
    "cls",
    "op",
    "params",
    "iters",
    "predicted",
    "status",
    "info",
    "error",
    "op_index",
    "op_code",
    "n_calls",
    "dev_kernel_ns_median",
    "dev_kernel_ns_min",
    "dev_kernel_ns_max",
    "per_iter_sum_all_ops_ns_median",
    "op_codes_in_cell",
]


def postprocess(report_csv, cells_csv, out_csv=None):
    import statistics  # noqa: PLC0415

    prov = ""
    cells = []
    with open(cells_csv) as f:
        first = f.readline()
        if first.startswith("# PROVENANCE"):
            prov = first.rstrip("\n")
        else:
            f.seek(0)
        cells = list(csv.DictReader(f))
    by_id = {c["cell_id"]: c for c in cells}
    # walk the ops report in order, bucket device ops between CELL_BEGIN/CELL_END signposts
    buckets = {}
    current = None
    with open(report_csv, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            code = (r.get("OP CODE") or "").strip()
            typ = (r.get("OP TYPE") or "").strip()
            if typ == "signpost":
                if code.startswith("CELL_BEGIN"):
                    # tracy puts signpost(header, message) as OP CODE = header and message = ATTRIBUTES
                    msg = (r.get("ATTRIBUTES") or "").strip()
                    current = code.split(" ", 1)[1].strip() if " " in code else (msg if msg else code)
                    buckets.setdefault(current, [])
                elif code.startswith("CELL_END"):
                    current = None
                continue
            if current is None:
                continue
            dur = r.get("DEVICE KERNEL DURATION [ns]", "")
            try:
                dur = float(dur)
            except ValueError:
                continue
            buckets[current].append((code, dur))
    out_csv = out_csv or os.path.join(os.path.dirname(cells_csv), "topk_campaign_results.csv")
    with open(out_csv, "w", newline="") as f:
        if prov:
            f.write(prov + f" report={report_csv}\n")
        wr = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        wr.writeheader()
        for cid, ops in buckets.items():
            c = by_id.get(cid, {"cell_id": cid})
            iters = int(c.get("iters") or 3)
            # ops repeat per call: 1 cache-warming + 2 warmups + iters calls -> ops per call = len(ops) / n_calls
            n_calls = 3 + iters
            per_call = len(ops) // n_calls if n_calls and len(ops) % n_calls == 0 else 0
            codes = [o[0] for o in ops[:per_call]] if per_call else sorted({o[0] for o in ops})
            base = {k: c.get(k, "") for k in CELL_FIELDS}
            base["op_codes_in_cell"] = json.dumps(codes)
            if per_call == 0:
                row = dict(base, op_index="", op_code="UNALIGNED", n_calls=len(ops))
                wr.writerow({k: row.get(k, "") for k in RESULT_FIELDS})
                continue
            timed = ops[3 * per_call :]  # drop the 3 untimed calls
            sums = [sum(d for _, d in timed[i * per_call : (i + 1) * per_call]) for i in range(iters)]
            for j in range(per_call):
                durs = [timed[i * per_call + j][1] for i in range(iters)]
                row = dict(
                    base,
                    op_index=j,
                    op_code=timed[j][0],
                    n_calls=iters,
                    dev_kernel_ns_median=statistics.median(durs),
                    dev_kernel_ns_min=min(durs),
                    dev_kernel_ns_max=max(durs),
                    per_iter_sum_all_ops_ns_median=statistics.median(sums),
                )
                wr.writerow({k: row.get(k, "") for k in RESULT_FIELDS})
    print(f"[topk_campaign] wrote {out_csv}", flush=True)


# --------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default="SMOKE", help="comma list of L1..L6,A..G,R1..R4,COUNTERS,SMOKE, or ALL")
    ap.add_argument("--iters", type=int, default=0, help="override timed iterations per cell (default per cell)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--list", action="store_true", help="enumerate cells (no torch/ttnn) and exit")
    ap.add_argument("--postprocess", nargs=2, metavar=("REPORT_CSV", "CELLS_CSV"))
    ap.add_argument("--exclude-op", default="", help="comma list of cell ops to skip (e.g. indexer)")
    args = ap.parse_args()

    if args.postprocess:
        postprocess(*args.postprocess)
        return
    all_cells = enumerate_cells()
    all_classes = sorted({c["cls"] for c in all_cells}, key=lambda s: (len(s), s))
    classes = all_classes if args.classes == "ALL" else args.classes.split(",")
    unknown = [c for c in classes if c not in all_classes]
    if unknown:
        sys.exit(f"unknown classes {unknown}; known: {all_classes}")
    if args.list:
        sel = [c for c in all_cells if c["cls"] in classes]
        counts = {}
        for c in sel:
            counts[c["cls"]] = counts.get(c["cls"], 0) + 1
        for k in all_classes:
            if k in counts:
                print(f"{k:9s} {counts[k]:4d}")
        dur = sum(v for k, v in counts.items() if k not in ("COUNTERS", "SMOKE"))
        print(
            f"duration cells (excl. COUNTERS, SMOKE): {dur}; COUNTERS: {counts.get('COUNTERS', 0)}; "
            f"SMOKE: {counts.get('SMOKE', 0)}; total selected: {len(sel)}"
        )
        if len(classes) <= 2:
            for c in sel:
                print(c["cell_id"], c["predicted"])
        return
    if args.exclude_op:
        EXCLUDE_OPS.update(args.exclude_op.split(","))
    run_campaign(classes, args.iters, args.out)


if __name__ == "__main__":
    main()
