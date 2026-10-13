# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-perf sweep for the `fabric_reduce_scatter` example: a line / ring reduce-scatter over every fabric config and
every line and ring the box can form, with ttnn.reduce_scatter on the same group and topology as the in-tree baseline.

Every case checks every chip's output against the fp32 sum of its group's partials (the device adds in bf16, in the
relay's order, so the check is a tolerance, not bit-exactness); a combination the fabric cannot route is reported as
unsupported.

    scripts/run_safe_pytest.sh --run-all --no-precompile tests/ttnn/unit_tests/operations/examples/test_fabric_reduce_scatter.py -s

Mirrors test_fabric_all_gather.py; see ttnn/ttnn/operations/examples/fabric_reduce_scatter/report.md.
"""

import os

# "device" = device profiler kernel duration of the slowest chip (needs a profiler build); "rt" = realtime profiler
# program duration of the slowest chip (any build).
PROFILER = os.environ.get("RS_PROFILER", "device")
if PROFILER == "device":
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
    os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
    os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

from ttnn.operations.examples.fabric_all_gather import build_groups
from ttnn.operations.examples.fabric_reduce_scatter.program_descriptor_with_inline_kernels import (
    _PLAN_CACHE,
    fabric_reduce_scatter,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
# per-chip INPUT (the partial) sizes (H, W): chip p gets rows [p H / G, (p + 1) H / G) of the sum. "H,W" or "H,W;H,W"
SHAPES = [
    tuple(int(x) for x in s.split(",")) for s in os.environ.get("RS_SHAPE", "2048,4096;4096,4096").split(";") if s
]
LINK_GBPS = float(os.environ.get("RS_LINK_GBPS", "48.5"))  # one link direction's bare one-hop rate (QuietBox)
TRIALS = int(os.environ.get("RS_TRIALS", "5"))
PAYLOAD = int(os.environ.get("RS_PAYLOAD", "14336"))
LINKS = tuple(int(x) for x in os.environ.get("RS_LINKS", "1,2").split(","))
WORKERS = tuple(int(x) for x in os.environ.get("RS_WORKERS", "1").split(","))  # cores per link and direction
FINALS = tuple(int(x) for x in os.environ.get("RS_FINALS", "2").split(","))  # final cores per link (workers=1)
INC_EVERY = tuple(int(x) for x in os.environ.get("RS_INC_EVERY", "8").split(","))  # arrival increment every N chunks
CB_KB = tuple(int(x) for x in os.environ.get("RS_CB_KB", "112").split(","))  # CB budget per worker, KiB
BALANCE = os.environ.get("RS_BALANCE", "1") == "1"  # rings of even G: split the far block between the directions
MUX_BUFFERS = int(os.environ.get("RS_MUX_BUFFERS", "2"))  # buffers per client channel on the mux
MUX_NOC = int(os.environ.get("RS_MUX_NOC", "1"))
WORKER_PLACEMENT = os.environ.get("RS_WORKER_PLACEMENT", "spread")  # "spread" | "mux" | "dram"
BASELINE = os.environ.get("RS_BASELINE", "1") == "1"  # also time ttnn.reduce_scatter on the same line
# max |device - fp32 sum| allowed: the sum of G bf16 values, each partial add rounded to bf16 (as the source branch's
# MiMo test: max(0.25, 2 x ttnn.reduce_scatter's error))
ATOL = float(os.environ.get("RS_ATOL", "0.25"))
_FABRICS = {
    "1d": "FABRIC_1D",
    "1d_ring": "FABRIC_1D_RING",
    "1d_neighbor_exchange": "FABRIC_1D_NEIGHBOR_EXCHANGE",
    "2d": "FABRIC_2D",
    "2d_torus_x": "FABRIC_2D_TORUS_X",
    "2d_torus_y": "FABRIC_2D_TORUS_Y",
    "2d_torus_xy": "FABRIC_2D_TORUS_XY",
}
FABRICS = tuple(os.environ.get("RS_FABRICS", ",".join(_FABRICS)).split(","))
# (mesh shape, cluster_axis, topology); cluster_axis None = one group over the whole mesh in snake order. The 2x2 axis
# groups have 2 chips, where a ring is a line.
_TOPOS = {
    "2x2_axis0_line": ((2, 2), 0, "Linear"),
    "2x2_axis1_line": ((2, 2), 1, "Linear"),
    "2x2_snake_line": ((2, 2), None, "Linear"),
    "2x2_snake_ring": ((2, 2), None, "Ring"),
    "4x1_line": ((4, 1), 0, "Linear"),
    "4x1_ring": ((4, 1), 0, "Ring"),
    "1x4_line": ((1, 4), 1, "Linear"),
    "1x4_ring": ((1, 4), 1, "Ring"),
}
TOPOS = tuple(os.environ.get("RS_TOPOS", ",".join(_TOPOS)).split(","))
MESH_SHAPES = sorted({_TOPOS[t][0] for t in TOPOS})
_REPORT = []


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    if _REPORT:
        logger.info("\n".join(_REPORT))


def _emit(line):
    _REPORT.append(line)
    print(f"FABRIC_REDUCE_SCATTER {line.strip()}", flush=True)


def _slowest_chip_ns(mesh_device):
    """Per chip, the summed kernel duration of every program of one call (ttnn.reduce_scatter may launch more than
    one); the slowest chip."""
    ttnn.ReadDeviceProfiler(mesh_device)
    chip_ns = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        ns = [
            float(p.program_analyses_results[_DURATION_KEY].duration)
            for p in programs
            if _DURATION_KEY in (getattr(p, "program_analyses_results", None) or {})
        ]
        if ns:
            chip_ns.append(sum(ns))
    assert chip_ns, "profiler produced no kernel durations (profiler-enabled build?)"
    return max(chip_ns)


def _time(mesh_device, run):
    samples = []
    for _ in range(TRIALS):
        if PROFILER == "rt":
            _, records = profile_realtime_program(mesh_device, run, collect_all=True, record_timeout_seconds=5.0)
            assert records, "realtime profiler returned no program"
            samples.append(max(r["duration_ns"] for r in records))
        else:
            ttnn.ReadDeviceProfiler(mesh_device)
            run()
            samples.append(_slowest_chip_ns(mesh_device))
    return statistics.median(samples), min(samples), max(samples)


def _router(payload):
    cfg = ttnn.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = payload
    return cfg


def _positions(mesh_device, cluster_axis, num_links, topology, workers, finals=2):
    """{coord: position p} as fabric_reduce_scatter planned it (chip p gets block p)."""
    key = (id(mesh_device), cluster_axis, num_links, "auto", topology, workers, WORKER_PLACEMENT, finals)
    return {coord: ch["rings"][0]["p"] for coord, ch in _PLAN_CACHE[key][1].items()}


def _max_err(out, ref, groups_of, pos, cols, Sb):
    worst = 0.0
    for d, t in enumerate(ttnn.get_device_tensors(out)):
        coord = divmod(d, cols)
        p = pos[coord]
        want = ref[groups_of[coord]][p * Sb : (p + 1) * Sb]
        got = ttnn.to_torch(t).float().reshape(want.shape)
        worst = max(worst, (got - want).abs().max().item())
    return worst


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "fabric_config": getattr(ttnn.FabricConfig, _FABRICS[f]),
                "fabric_router_config": _router(PAYLOAD),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            id=f"fabric_{f}",
        )
        for f in FABRICS
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", MESH_SHAPES, ids=lambda s: f"mesh{s[0]}x{s[1]}", indirect=True)
def test_fabric_reduce_scatter(mesh_device):
    """Every (size, group, links): output within ATOL of the fp32 sum on every chip, then the median of TRIALS timed
    calls. Reported per case: time, and the busiest link's rate = its blocks / (links x time) (a line's end chip sends
    G - 1 blocks one way; fabric_reduce_scatter's ring G // 2 forward, G/2 - 1/2 balanced), as a % of one link direction's bare rate.
    The same bytes are used for ttnn.reduce_scatter, so its rate compares directly."""
    fabric = str(ttnn.get_fabric_config()).split(".")[-1]
    if not _REPORT:
        _emit(
            f"\n=== fabric_reduce_scatter  box={socket.gethostname()}  arch={mesh_device.arch()}  payload={PAYLOAD}B  "
            f"bf16 tile  trials={TRIALS} (median)  link peak {LINK_GBPS:.1f} GB/s ==="
        )
    for H, W in SHAPES:
        _one_size(mesh_device, H, W, fabric)


def _one_size(mesh_device, H, W, fabric):
    rows, cols = tuple(mesh_device.shape)
    torch.manual_seed(0)
    host = torch.randn((rows, cols, H, W), dtype=torch.bfloat16)
    inp = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )
    inp = ttnn.reshape(inp, (1, 1, H, W))
    in_bytes = H * W * 2
    size = f"{H}x{W} ({in_bytes / 2**20:.0f} MiB in)"
    for topo_name in TOPOS:
        shape, cluster_axis, topo = _TOPOS[topo_name]
        topology = getattr(ttnn.Topology, topo)
        if shape != (rows, cols):
            continue
        groups = build_groups((rows, cols), cluster_axis)
        G = len(groups[0])
        Sb = H // G
        groups_of = {coord: gi for gi, grp in enumerate(groups) for coord in grp}
        ref = [sum(host[r, c].float() for r, c in grp) for grp in groups]
        ring = topo == "Ring" and G > 2
        far = (G / 2 - 0.5 if BALANCE and G % 2 == 0 else G // 2) if ring else G - 1  # blocks one way, busiest link
        busiest = far * Sb * W * 2  # bytes over the busiest link direction (all its links)
        for num_links, workers, inc, cb, fins in [
            (l, w, i, c, fn) for l in LINKS for w in WORKERS for i in INC_EVERY for c in CB_KB for fn in FINALS
        ]:
            tag = f"    {fabric:<27} {topo_name:<15} {size:<22} G={G} links={num_links}"
            if WORKERS != (1,):
                tag += f" w={workers}"
            if INC_EVERY != (8,) or CB_KB != (112,):
                tag += f" inc={inc} cb={cb}"
            if FINALS != (2,):
                tag += f" F={fins}"
            run = lambda: fabric_reduce_scatter(
                inp,
                cluster_axis=cluster_axis,
                num_links=num_links,
                topology=topology,
                balance=BALANCE,
                workers=workers,
                mux_buffers=MUX_BUFFERS,
                mux_noc=MUX_NOC,
                worker_placement=WORKER_PLACEMENT,
                inc_every=inc,
                cb_bytes=cb * 1024,
                finals=fins,
            )
            try:
                out = run()
            except (ValueError, RuntimeError) as e:
                _emit(f"{tag}  fabric_rs unsupported: {str(e).splitlines()[0][:110]}")
                continue
            ttnn.synchronize_device(mesh_device)
            pos = _positions(mesh_device, cluster_axis, num_links, topology, workers, fins)
            err = _max_err(out, ref, groups_of, pos, cols, Sb)
            assert err <= ATOL, f"{fabric}/{topo_name}/{size}/links={num_links}: max abs err {err} > {ATOL}"
            del out
            ns, lo, hi = _time(mesh_device, run)
            per_link = busiest / (num_links * ns)
            _emit(
                f"{tag}  fabric_rs {ns / 1e3:8.1f} us (min {lo / 1e3:.1f} max {hi / 1e3:.1f})  busiest link "
                f"{per_link:5.1f} GB/s = {100 * per_link / LINK_GBPS:3.0f}% of peak  err {err:.3g}  ✓"
            )
            if (
                BASELINE
                and cluster_axis is not None
                and workers == WORKERS[0]
                and (inc, cb) == (INC_EVERY[0], CB_KB[0])
            ):
                _baseline(mesh_device, inp, tag, cluster_axis, num_links, topology, ref, groups_of, cols, Sb, busiest)


def _baseline(mesh_device, inp, tag, cluster_axis, num_links, topology, ref, groups_of, cols, Sb, busiest):
    """ttnn.reduce_scatter(dim=2) on the same group, topology and link count: chip at position p along the axis gets
    block p."""
    run = lambda: ttnn.reduce_scatter(
        inp,
        dim=2,
        cluster_axis=cluster_axis,
        num_links=num_links,
        topology=topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    try:
        out = run()
        ttnn.synchronize_device(mesh_device)
    except (ValueError, RuntimeError) as e:
        _emit(f"{tag}  ttnn.reduce_scatter unsupported: {str(e).splitlines()[0][:110]}")
        return
    pos = {coord: coord[cluster_axis] for coord in groups_of}
    err = _max_err(out, ref, groups_of, pos, cols, Sb)
    del out
    ns, lo, hi = _time(mesh_device, run)
    per_link = busiest / (num_links * ns)
    _emit(
        f"{tag}  ttnn_rs   {ns / 1e3:8.1f} us (min {lo / 1e3:.1f} max {hi / 1e3:.1f})  busiest link "
        f"{per_link:5.1f} GB/s = {100 * per_link / LINK_GBPS:3.0f}% of peak  err {err:.3g}"
    )
