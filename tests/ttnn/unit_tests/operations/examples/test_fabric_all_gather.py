# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `fabric_all_gather` example: line / ring all-gather over every fabric config and topology the box
can form. Every case checks the gathered output bit-exactly on every chip; a combination the fabric cannot route
(no direct link for a hop) is reported as unsupported.

    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather.py -s

See ttnn/ttnn/operations/examples/fabric_all_gather/README.md.
"""

import os

# Under tt-emule (TT_METAL_EMULE_MODE) there is no timing: run correctness only, without the device profiler.
EMULE = bool(os.environ.get("TT_METAL_EMULE_MODE"))
# "device" = device profiler kernel duration of the slowest chip (needs a profiler build); "rt" = realtime profiler
# program duration of the slowest chip (any build; what CI uses).
PROFILER = os.environ.get("AG_PROFILER", "device")
if not EMULE and PROFILER == "device":
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

from ttnn.operations.examples.fabric_all_gather import build_groups, fabric_all_gather, link_load, plan

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
# per-chip shard sizes (H, W), one row of results per size: "H,W" or "H,W;H,W;..."
SHAPES = [tuple(int(x) for x in s.split(",")) for s in os.environ.get("AG_SHAPE", "2048,4096").split(";") if s]
# one link direction's rate (GB/s) that link utilization is measured against: 48.5 = the bare one-hop stream measured
# on a QuietBox (fabric_link_ceiling); a Galaxy's links are slower: its high_bw_all_gather 8-rank gate (94.3 GB/s per
# chip, busiest hop 4 shards over 2 links) implies at least 26.9 GB/s per link, so 27 is the default on 32 chips.
_LINK_GBPS_ENV = os.environ.get("AG_LINK_GBPS")
TRIALS = int(os.environ.get("AG_TRIALS", "3"))
PAYLOAD = int(os.environ.get("AG_PAYLOAD", "14336"))
_DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32, "fp8": ttnn.fp8_e4m3}
DTYPE_NAME = os.environ.get("AG_DTYPE", "bf16")
_LAYOUTS = {"tile": ttnn.TILE_LAYOUT, "rm": ttnn.ROW_MAJOR_LAYOUT}
LAYOUT_NAME = os.environ.get("AG_LAYOUT", "tile")  # rm = ROW_MAJOR (a page is one row)
DIM = int(os.environ.get("AG_DIM", "0"))  # 0, or 2 (= dim -2 of the [1, 1, H, W] shard)
LINKS = tuple(int(x) for x in os.environ.get("AG_LINKS", "1,2").split(","))
# schedule variants: "b" = balanced ring (far shard split between directions), "d" = desynchronized bank walks
VARIANTS = tuple(os.environ.get("AG_VARIANTS", "base").split(","))
_FABRICS = {
    "1d": "FABRIC_1D",
    "1d_ring": "FABRIC_1D_RING",
    "1d_neighbor_exchange": "FABRIC_1D_NEIGHBOR_EXCHANGE",
    "2d": "FABRIC_2D",
    "2d_torus_x": "FABRIC_2D_TORUS_X",
    "2d_torus_y": "FABRIC_2D_TORUS_Y",
    "2d_torus_xy": "FABRIC_2D_TORUS_XY",
}
FABRICS = tuple(os.environ.get("AG_FABRICS", ",".join(_FABRICS)).split(","))
# (mesh shape, cluster_axis, topology[, scheme]); cluster_axis None = one group over the whole mesh
_TOPOS = {
    "2x2_axis0_line": ((2, 2), 0, "Linear"),
    "2x2_axis1_line": ((2, 2), 1, "Linear"),
    "2x2_snake_line": ((2, 2), None, "Linear"),
    "2x2_snake_ring": ((2, 2), None, "Ring"),
    "4x1_line": ((4, 1), 0, "Linear"),
    "4x1_ring": ((4, 1), 0, "Ring"),
    "1x4_line": ((1, 4), 1, "Linear"),
    "1x4_ring": ((1, 4), 1, "Ring"),
    # 32-chip Galaxy (4 x 8 torus): per-axis rings, one snake ring, and two edge-disjoint Hamiltonian cycles
    "4x8_axis0_ring": ((4, 8), 0, "Ring"),
    "4x8_axis1_ring": ((4, 8), 1, "Ring"),
    "4x8_snake_ring": ((4, 8), None, "Ring"),
    "4x8_dual_cycles": ((4, 8), None, "Ring", "dual_cycles"),
    # BH Galaxy as CI opens it (8 x 4, FABRIC_2D_TORUS_XY)
    "8x4_axis0_ring": ((8, 4), 0, "Ring"),
    "8x4_axis1_ring": ((8, 4), 1, "Ring"),
    "8x4_snake_ring": ((8, 4), None, "Ring"),
    "8x4_dual_cycles": ((8, 4), None, "Ring", "dual_cycles"),
    # BH LoudBox (2 x 4)
    "2x4_axis0_line": ((2, 4), 0, "Linear"),
    "2x4_axis1_line": ((2, 4), 1, "Linear"),
    "2x4_axis1_ring": ((2, 4), 1, "Ring"),
    "2x4_snake_ring": ((2, 4), None, "Ring"),
}
# 1 = a topology the fabric cannot route fails the test instead of being reported as unsupported (CI)
STRICT = os.environ.get("AG_STRICT", "0") == "1"
_QB_TOPOS = [t for t, v in _TOPOS.items() if v[0][0] * v[0][1] == 4]
TOPOS = tuple(os.environ.get("AG_TOPOS", ",".join(_QB_TOPOS)).split(","))
MESH_SHAPES = sorted({_TOPOS[t][0] for t in TOPOS})
_REPORT = []


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    if _REPORT:
        logger.info("\n".join(_REPORT))


def _emit(line):
    """Record a result row, and print it right away (flushed): a CI log keeps what was measured even if the job
    is killed later; the whole table is printed again at the end of the module."""
    _REPORT.append(line)
    print(f"FABRIC_ALL_GATHER {line.strip()}", flush=True)


def _slowest_chip_ns(mesh_device):
    ttnn.ReadDeviceProfiler(mesh_device)
    chip_ns = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        ns = [
            float(p.program_analyses_results[_DURATION_KEY].duration)
            for p in programs
            if _DURATION_KEY in (getattr(p, "program_analyses_results", None) or {})
        ]
        if ns:
            chip_ns.append(sum(ns) / len(ns))
    assert chip_ns, "profiler produced no kernel durations (profiler-enabled build?)"
    return max(chip_ns)


def _to_host(t):
    """A device tensor's values on the host, exactly. fp8_e4m3 goes through float32 on device first (some torch
    builds reject fp8 over DLPack; every fp8 value is exact in float32)."""
    if t.dtype == ttnn.fp8_e4m3:
        t = ttnn.typecast(t, ttnn.float32)
    return ttnn.to_torch(t)


def _rt_slowest_chip_ns(mesh_device, run):
    _, records = profile_realtime_program(mesh_device, run, collect_all=True, record_timeout_seconds=5.0)
    assert records, "realtime profiler returned no program"
    return max(r["duration_ns"] for r in records)


def _router(payload):
    cfg = ttnn.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = payload
    return cfg


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
def test_fabric_all_gather(mesh_device):
    """Every (size, topology, links, variant): bit-exact output on every chip, then the median of TRIALS timed calls
    into the same output. Reported per case: time, effective receive bandwidth per chip = shard bytes x (G - 1) / time,
    and link utilization = the busiest hop's bytes / (links x time) as a fraction of one link direction's rate."""
    rows, cols = tuple(mesh_device.shape)
    link_gbps = float(_LINK_GBPS_ENV) if _LINK_GBPS_ENV else (27.0 if rows * cols == 32 else 48.5)
    fabric = str(ttnn.get_fabric_config()).split(".")[-1]
    if not _REPORT:
        _emit(
            f"\n=== fabric_all_gather  box={socket.gethostname()}  arch={mesh_device.arch()}  payload={PAYLOAD}B  "
            f"{DTYPE_NAME} {LAYOUT_NAME} dim={DIM}  trials={TRIALS} (median)  link peak {link_gbps:.1f} GB/s ==="
        )
    for H, W in SHAPES:
        _gather_one_size(mesh_device, H, W, fabric, link_gbps)


def _gather_one_size(mesh_device, H, W, fabric, link_gbps):
    rows, cols = tuple(mesh_device.shape)
    torch.manual_seed(0)
    dtype = _DTYPES[DTYPE_NAME]
    host = torch.randn((rows, cols, H, W), dtype=torch.float32 if dtype == ttnn.float32 else torch.bfloat16)
    inp = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16 if dtype == ttnn.fp8_e4m3 else dtype,  # fp8: made on device from bf16
        layout=_LAYOUTS[LAYOUT_NAME],
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )
    if dtype == ttnn.fp8_e4m3:
        inp = ttnn.typecast(inp, ttnn.fp8_e4m3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if dtype in (ttnn.bfloat8_b, ttnn.fp8_e4m3):  # lossy: compare against what the device holds
        host = torch.stack([_to_host(t) for t in ttnn.get_device_tensors(inp)]).reshape(rows, cols, H, W)
    pages = (H // 32) * (W // 32) if LAYOUT_NAME == "tile" else H
    shard_bytes = pages * int(inp.buffer_aligned_page_size())
    size = f"{H}x{W} ({shard_bytes / 2**20:.1f} MiB)"
    for topo_name in TOPOS:
        shape, cluster_axis, topo = _TOPOS[topo_name][:3]
        scheme = _TOPOS[topo_name][3] if len(_TOPOS[topo_name]) > 3 else "ring"
        if shape != (rows, cols):
            continue
        topology = getattr(ttnn.Topology, topo)
        # output slots are in row-major chip order for every scheme (a snake travels in snake order)
        groups = [sorted(g) for g in build_groups((rows, cols), cluster_axis)]
        G = len(groups[0])
        for num_links, variant in [(l, v) for l in LINKS for v in VARIANTS]:
            balance, desync = "b" in variant and variant != "base", "d" in variant and variant != "base"
            kw = dict(balance=balance, desync=desync)
            tag = f"    {fabric:<27} {topo_name:<15} {size:<22} G={G:<2} links={num_links} {variant:<4}"
            try:
                out = fabric_all_gather(
                    inp, cluster_axis=cluster_axis, topology=topology, num_links=num_links, dim=DIM, scheme=scheme, **kw
                )
            except (ValueError, RuntimeError) as e:
                if STRICT:
                    raise
                msg = str(e).splitlines()[0][:110]
                _emit(f"{tag}  unsupported: {msg}")
                continue
            ttnn.synchronize_device(mesh_device)
            dev = ttnn.get_device_tensors(out)
            for grp in groups:  # one chip's output on the host at a time (a Galaxy's outputs are tens of GB)
                expected = torch.cat([host[r, c].reshape(1, 1, H, W) for r, c in grp], dim=DIM)
                for r, c in grp:
                    got = _to_host(dev[r * cols + c])
                    assert torch.equal(
                        got.reshape(expected.shape).to(expected.dtype), expected
                    ), f"{fabric}/{topo_name}/{size}/links={num_links}: chip ({r},{c}) output != gathered shards"
                    del got
            chips, _ = plan(
                mesh_device,
                cluster_axis=cluster_axis,
                topology=topology,
                num_links=num_links,
                scheme=scheme,
                balance=balance,
            )
            load = link_load(chips)
            nbrs = max(len({p for (a, p) in load if a == coord}) for coord in chips)
            busiest = max(load.values())
            geo = f"busiest hop {busiest:.1f} shards, {nbrs} neighbours/chip"
            if EMULE or TRIALS == 0:
                _emit(f"{tag}  bit-exact on all {rows * cols} chips  ✓  {geo}  (no timing)")
                continue
            run = lambda: fabric_all_gather(
                inp,
                cluster_axis=cluster_axis,
                topology=topology,
                num_links=num_links,
                dim=DIM,
                scheme=scheme,
                output=out,
                **kw,
            )
            samples = []
            for _ in range(TRIALS):
                if PROFILER == "rt":
                    samples.append(_rt_slowest_chip_ns(mesh_device, run))
                else:
                    ttnn.ReadDeviceProfiler(mesh_device)
                    run()
                    samples.append(_slowest_chip_ns(mesh_device))
            ns = statistics.median(samples)
            per_link = busiest * shard_bytes / (num_links * ns)  # GB/s on each link of the busiest hop
            _emit(
                f"{tag}  {ns / 1e3:>9.1f} us  receive {shard_bytes * (G - 1) / ns:6.1f} GB/s/chip  "
                f"busiest link {per_link:5.1f} GB/s = {100 * per_link / link_gbps:3.0f}% of peak  ✓  {geo}"
            )
            del out


# ----------------------------------------------------------------------------------------------------------------------
# Reusing one preallocated output, calls back to back (the fence), and running inside a sub-device
# ----------------------------------------------------------------------------------------------------------------------
REUSE_CALLS = int(os.environ.get("AG_REUSE_CALLS", "8"))
REUSE_SHAPE = (256, 1024)  # small shards: calls are short, so a missing fence would show up as a race


_DELAY_SOURCE = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    const uint32_t chunks = get_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < chunks; ++i) {
        riscv_wait(1000000);
    }
}
"""


def _delay_chip(mesh_device, tensor, coord, chunks):
    """Keep chip `coord` busy for `chunks` x 1M cycles (a program queued there, after whatever came before it)."""
    mesh_desc = ttnn.MeshProgramDescriptor()
    core = ttnn.CoreCoord(0, 0)
    for r in range(mesh_device.shape[0]):
        for c in range(mesh_device.shape[1]):
            program = ttnn.ProgramDescriptor()
            rt = ttnn.RuntimeArgs()
            rt[core.x][core.y] = [chunks if (r, c) == coord else 0]
            program.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=_DELAY_SOURCE,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
                    compile_time_args=[],
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0),
                )
            ]
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    ttnn.generic_op([tensor], mesh_desc)


def _sharded_inputs(mesh_device, n, H, W):
    rows, cols = tuple(mesh_device.shape)
    hosts = [torch.randn((rows, cols, H, W), dtype=torch.bfloat16) for _ in range(n)]
    tensors = [
        ttnn.from_torch(
            h,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
        )
        for h in hosts
    ]
    return hosts, tensors


def _check(host, out, groups, cols, H, W, what):
    dev = ttnn.get_device_tensors(out)
    for grp in groups:
        expected = torch.cat([host[r, c].reshape(1, 1, H, W) for r, c in grp], dim=0)
        for r, c in grp:
            got = ttnn.to_torch(dev[r * cols + c])
            assert torch.equal(got.reshape(expected.shape), expected), f"{what}: chip ({r},{c}) wrong"


def _reuse_topos(mesh_device):
    shape = tuple(mesh_device.shape)
    return [t for t in TOPOS if _TOPOS[t][0] == shape and len(_TOPOS[t]) == 3]


def _preallocated(mesh_device, inp, G):
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([G * inp.shape[0], *list(inp.shape)[1:]]),
        inp.dtype,
        ttnn.TILE_LAYOUT,
        mesh_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )


_REUSE_FABRICS = [f for f in FABRICS if f in ("1d", "2d_torus_xy", "2d")] or [FABRICS[0]]


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
        for f in _REUSE_FABRICS
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", MESH_SHAPES, ids=lambda s: f"mesh{s[0]}x{s[1]}", indirect=True)
def test_fabric_all_gather_output_reuse(mesh_device):
    """REUSE_CALLS calls into ONE preallocated output, a different input each, with no host sync in between; after
    each call a clone snapshots the output (a consumer queued behind the gather on every chip). Every snapshot must
    be that call's gather: a neighbour may not overwrite my output before my consumer of the previous call is done."""
    rows, cols = tuple(mesh_device.shape)
    H, W = REUSE_SHAPE
    torch.manual_seed(1)
    hosts, inputs = _sharded_inputs(mesh_device, REUSE_CALLS, H, W)
    for topo_name in _reuse_topos(mesh_device):
        _, cluster_axis, topo = _TOPOS[topo_name]
        groups = [sorted(g) for g in build_groups((rows, cols), cluster_axis)]
        for num_links, balance in [(l, b) for l in LINKS for b in (False, True)]:
            kw = dict(cluster_axis=cluster_axis, topology=getattr(ttnn.Topology, topo), num_links=num_links)
            try:
                plan(mesh_device, balance=balance, **kw)
            except (ValueError, RuntimeError):
                continue
            out = _preallocated(mesh_device, inputs[0], len(groups[0]))
            snaps = []
            for i, inp in enumerate(inputs):
                res = fabric_all_gather(inp, output=out, balance=balance, **kw)
                assert res.buffer_address() == out.buffer_address(), "the preallocated output must be written in place"
                # one chip (a different one each call) reads its result late: its neighbours must not have written
                # the next call's data into its output by then
                _delay_chip(mesh_device, out, divmod(i % (rows * cols), cols), 200)
                snaps.append(ttnn.clone(out, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            ttnn.synchronize_device(mesh_device)
            for i, (host, snap) in enumerate(zip(hosts, snaps)):
                _check(host, snap, groups, cols, H, W, f"{topo_name} links={num_links} balance={balance} call {i}")
            _emit(f"    reuse  {topo_name:<15} links={num_links} balance={balance!s:<5}  {REUSE_CALLS} calls ✓")


def _gather_strip(mesh_device, columns):
    grid = mesh_device.compute_with_storage_grid_size()
    x0 = grid.x - columns
    rest = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x0 - 1, grid.y - 1))])
    strip = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    return rest, strip


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
        for f in _REUSE_FABRICS
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", MESH_SHAPES, ids=lambda s: f"mesh{s[0]}x{s[1]}", indirect=True)
@pytest.mark.parametrize("external_semaphores", [False, True], ids=["own_semaphores", "external_semaphores"])
def test_fabric_all_gather_subdevice(mesh_device, external_semaphores):
    """The op confined to a 4-column strip of the grid that is its own sub-device (the rest is another), with its
    own or caller-owned semaphores; calls back to back into one preallocated output."""
    rows, cols = tuple(mesh_device.shape)
    H, W = REUSE_SHAPE
    torch.manual_seed(2)
    hosts, inputs = _sharded_inputs(mesh_device, REUSE_CALLS, H, W)
    rest, strip = _gather_strip(mesh_device, 4)
    sems = None
    if external_semaphores:
        sems = [ttnn.create_global_semaphore(mesh_device, strip, 0) for _ in range(2)]
        ttnn.synchronize_device(mesh_device)
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([rest]), ttnn.SubDevice([strip])], 0)
    mesh_device.load_sub_device_manager(manager)
    sd = ttnn.SubDeviceId(1)
    try:
        for topo_name in _reuse_topos(mesh_device):
            _, cluster_axis, topo = _TOPOS[topo_name]
            groups = [sorted(g) for g in build_groups((rows, cols), cluster_axis)]
            for num_links in LINKS:
                kw = dict(cluster_axis=cluster_axis, topology=getattr(ttnn.Topology, topo), num_links=num_links)
                kw.update(subdevice_id=sd, sub_core_grid=strip, balance=topo == "Ring")
                if sems:
                    kw.update(ready_semaphore=sems[0], data_valid_semaphore=sems[1])
                try:
                    chips, _ = plan(
                        mesh_device,
                        sub_core_grid=strip,
                        **{k: kw[k] for k in ("cluster_axis", "topology", "num_links", "balance")},
                    )
                except (ValueError, RuntimeError):
                    continue
                used = {(c.x, c.y) for ch in chips.values() for c in list(ch["ports"].values()) + ch["copy"]}
                x0 = mesh_device.compute_with_storage_grid_size().x - 4
                assert all(x >= x0 for x, _ in used), f"cores outside the strip: {sorted(used)}"
                out = _preallocated(mesh_device, inputs[0], len(groups[0]))
                for i, inp in enumerate(inputs):
                    fabric_all_gather(inp, output=out, **kw)
                    if i == 0:
                        ttnn.synchronize_device(mesh_device, sub_device_ids=[sd])
                        _check(hosts[0], out, groups, cols, H, W, f"subdevice {topo_name} links={num_links} call 0")
                ttnn.synchronize_device(mesh_device, sub_device_ids=[sd])
                _check(hosts[-1], out, groups, cols, H, W, f"subdevice {topo_name} links={num_links} last call")
                _emit(
                    f"    subdevice({'external' if sems else 'own'} sems)  {topo_name:<15} links={num_links}  "
                    f"{len(used)} cores in a 4-column strip ✓"
                )
    finally:
        ttnn.synchronize_device(mesh_device)
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(manager)
