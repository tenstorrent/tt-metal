# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.83 probe: ttnn.bringup.mhc_post as DeepSeek-V4.1 calls it on the 8x4 (the residual pass with POST_COLLAPSE +
NORM_IN_MHC, and attn_residual's with FFN_FEED), [160, 5120] bf16 a chip, the 11 x 10 grid cap, the model's
FABRIC_2D_TORUS_XY; random inputs. An op-level probe (no model load). It opens its own mesh (not BATCH_SHARED): the
device profiler must be on at the open and its CSV is complete only after the close, so each run is a fresh device
open (host RAM leaks a few GiB an open on the Galaxy): zone studies only; correctness runs on the resident runner
(tests/test_mhc_post_fast.py).

Per call it reads the device profiler (TT_METAL_DEVICE_PROFILER=1, set by this file) and prints, for the slowest chip,
the program span and every device zone of the kernels (start / end offsets from the program start, min / median / max
over cores): where the call's time goes.

  MHCP_SITES=ffn,attn      the calls (ffn: norm, no feed; attn: norm + feed, 2 links)
  MHCP_FAST=0|1|both       the fork's `fast` option (P.83): both = run both and check them against each other
  MHCP_CALLS=4             profiled calls a site (after 2 warm-ups)
  MHCP_ZONES=1             print the zone table (0: spans only)
"""

import csv
import os
import statistics

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")

import torch

import ttnn

N = 4
NG = N * (N + 2)
T, C = 160, 5120
CFG = dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False)
_CSV = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated/profiler/.logs/profile_log_device.csv")


def _open():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
    shape = ttnn.MeshShape(8, 4) if ttnn.get_num_devices() == 32 else None
    return ttnn.open_mesh_device(mesh_shape=shape, l1_small_size=24576) if shape else ttnn.open_mesh_device()


def _close(m):
    _SEMS.pop(id(m), None)
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    return ttnn.CoreCoord(min(g.x, 11), min(g.y, 10)) if g.x * g.y > 110 else None


_SEMS = {}


def _sems(mesh):
    if id(mesh) not in _SEMS:
        g = _grid(mesh) or mesh.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
        handles = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
        ttnn.synchronize_device(mesh)
        _SEMS[id(mesh)] = (handles, [int(ttnn.get_global_semaphore_address(h)) for h in handles])
    return _SEMS[id(mesh)][1]


def _dev(t, mesh, dtype=ttnn.float32, shard=True):
    return ttnn.from_torch(
        t.float().contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if shard else ttnn.ReplicateTensorToMesh(mesh),
    )


def _inputs(mesh, seed):
    nd = mesh.get_num_devices()
    g = torch.Generator().manual_seed(seed)
    f = _dev(torch.randn(nd, 1, T, C, generator=g), mesh, ttnn.bfloat16)
    x = _dev(torch.randn(nd, 1, T, N * C, generator=g) * 3, mesh, ttnn.bfloat16)
    mix = torch.rand(nd, 1, T, NG, generator=g)
    mix[..., N : 2 * N] *= 2
    w = torch.randn(1, 1, N * C, 32, generator=g) * (N * C) ** -0.5
    hc = dict(
        hc_scale=[float(v) for v in (torch.rand(3, generator=g) * 0.5 + 0.1)],
        hc_base=[float(v) for v in torch.randn(NG, generator=g) * 0.5],
        hc_norm_width=float(N * C),
        hc_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
    )
    gamma = _dev(torch.randn(1, 1, 1, C, generator=g) * 0.2 + 1.0, mesh, shard=False)
    return f, x, _dev(mix, mesh), _dev(w, mesh, shard=False), hc, gamma


def _call(mesh, ins, *, feed=None, fast=False):
    f, x, md, wd, hc, gamma = ins
    cfg = ttnn.ComputeConfigDescriptor(**CFG)
    kw = dict(stats_weight=wd, compute_kernel_config=cfg, post_col=N, comb_col=2 * N, max_grid=_grid(mesh))
    if feed is not None:
        kw.update(feed_output=feed, feed_semaphores=_sems(mesh), feed_num_links=2, feed_cluster_axis=1)
    if fast:
        kw.update(fast=True)
    return ttnn.bringup.mhc_post(f, x, md, md, collapse_pre=md, norm_weight=gamma, norm_eps=1e-6, **kw, **hc)


_MARK = {"c_mm"}  # a zone only the stats program has (both paths)


def _calls():
    """{call base: {chip: [(x, y, risc, cycle, zone, type)]}} of the stats program's launches, in launch order (the
    run host id is base + chip), and ns a cycle."""
    with open(_CSV, newline="") as fh:
        head = fh.readline()
        fh.readline()
        freq = 1000.0
        for part in head.split(","):
            if "CHIP_FREQ" in part:
                freq = float(part.split(":")[1])
        by = {}
        marked = set()
        for r in csv.reader(fh):
            if len(r) < 12:
                continue
            typ = r[11].strip()
            if typ not in ("ZONE_START", "ZONE_END"):
                continue
            zone = r[10].strip()
            if zone.endswith("-FW"):
                continue
            try:
                chip, run, cyc = int(r[0]), int(r[7]), int(r[5])
            except ValueError:
                continue
            key = (run - chip, chip)
            by.setdefault(key, []).append((r[1].strip(), r[2].strip(), r[3].strip(), cyc, zone, typ))
            if zone in _MARK:
                marked.add(run - chip)
    calls = {}
    for (base, chip), ev in by.items():
        if base in marked:
            calls.setdefault(base, {})[chip] = ev
    return [calls[b] for b in sorted(calls)], 1000.0 / freq


def _analyze(chips, ns, label, zones=True):
    """The slowest chip's span (first kernel zone start .. last end over its cores) and its zones (offsets from that
    start, us; min / median / max over cores)."""
    best = None
    for chip, ev in chips.items():
        t0 = min(e[3] for e in ev)
        t1 = max(e[3] for e in ev)
        if best is None or t1 - t0 > best[1] - best[0]:
            best = (t0, t1, chip, ev)
    t0, t1, chip, ev = best
    span = (t1 - t0) * ns / 1000
    if zones:
        open_, per = {}, {}
        for x, y, risc, cyc, zone, typ in sorted(ev, key=lambda e: e[3]):
            key = (x, y, risc, zone)
            if typ == "ZONE_START":
                open_.setdefault(key, []).append(cyc)
            elif open_.get(key):
                s = open_[key].pop()
                per.setdefault((risc, zone), []).append(((s - t0) * ns / 1000, (cyc - t0) * ns / 1000))
        print(f"\n[{label}] chip {chip} span {span:.1f} us")
        print(
            f"  {'risc':8s} {'zone':16s} {'n':>4s} {'dur med':>8s} {'dur max':>8s} {'st min':>7s} {'st med':>7s} "
            f"{'st max':>7s} {'end min':>7s} {'end med':>7s} {'end max':>7s}"
        )
        for (risc, zone), v in sorted(per.items(), key=lambda kv: statistics.median(e for _, e in kv[1])):
            d = [e - s for s, e in v]
            st = [s for s, _ in v]
            en = [e for _, e in v]
            print(
                f"  {risc:8s} {zone[:16]:16s} {len(v):4d} {statistics.median(d):8.1f} {max(d):8.1f} {min(st):7.1f} "
                f"{statistics.median(st):7.1f} {max(st):7.1f} {min(en):7.1f} {statistics.median(en):7.1f} {max(en):7.1f}"
            )
    return span


def _bits(c):
    return c.view(torch.int16) if c.dtype == torch.bfloat16 else c.view(torch.int32)


def _chips(t):
    return [ttnn.to_torch(c) for c in ttnn.get_device_tensors(t)]


def test_mhc_post_probe():
    """Profiled calls first; the device CSV is complete only once the mesh is closed, so the zones are read after."""
    sites = os.environ.get("MHCP_SITES", "ffn,attn").split(",")
    fast_env = os.environ.get("MHCP_FAST", "0")
    fasts = [False, True] if fast_env == "both" else [fast_env == "1"]
    calls = int(os.environ.get("MHCP_CALLS", "4"))
    zones = os.environ.get("MHCP_ZONES", "1") == "1"
    if os.path.exists(_CSV):
        os.remove(_CSV)
    labels, results = [], {}
    mesh = _open()
    try:
        ins = _inputs(mesh, 5)
        nd = mesh.get_num_devices()
        feed = _dev(torch.zeros(nd, 1, 4 * T, C // 4), mesh, ttnn.bfloat16)
        for site in sites:
            for fast in fasts:
                kw = dict(feed=feed if site == "attn" else None, fast=fast)
                for k in range(2 + calls):
                    out = _call(mesh, ins, **kw)
                    labels.append((f"{site} fast={int(fast)}", k >= 2))
                ttnn.synchronize_device(mesh)
                got = [_chips(t) for t in out]
                if site == "attn":
                    got.append(_chips(feed))
                results[(site, fast)] = got
                ttnn.ReadDeviceProfiler(mesh)
    finally:
        _close(mesh)
    launches, ns = _calls()
    print(f"\n{len(launches)} stats launches profiled, {len(labels)} calls made")
    spans = {}
    for (label, timed), chips in zip(labels, launches):
        if timed:
            spans.setdefault(label, []).append(_analyze(chips, ns, label, zones and not spans.get(label)))
    for label, s in spans.items():
        print(f"[{label}] spans (slowest chip, us): {[round(v, 1) for v in s]}  median {statistics.median(s):.1f}")
    names = ["x_out", "stats", "hc", "y", "feed"]
    for site in sites:
        if (site, False) in results and (site, True) in results:
            a, b = results[(site, False)], results[(site, True)]
            for i, (ta, tb) in enumerate(zip(a, b)):
                diff = sum(int((_bits(p) != _bits(q)).sum()) for p, q in zip(ta, tb))
                mx = max(float((p.float() - q.float()).abs().max()) for p, q in zip(ta, tb))
                print(f"[{site}] {names[i]}: differing elements {diff} (max abs {mx:.3g})")
