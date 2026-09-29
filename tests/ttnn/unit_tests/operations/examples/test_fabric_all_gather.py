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
if not EMULE:
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

from ttnn.operations.examples.fabric_all_gather import build_groups, fabric_all_gather, link_load, plan

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
SHAPE = tuple(int(x) for x in os.environ.get("FAG_SHAPE", "2048,4096").split(","))  # per-chip shard (H, W), bf16
TRIALS = int(os.environ.get("FAG_TRIALS", "3"))
PAYLOAD = int(os.environ.get("FAG_PAYLOAD", "14336"))
_DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32}
DTYPE_NAME = os.environ.get("FAG_DTYPE", "bf16")
DIM = int(os.environ.get("FAG_DIM", "0"))  # 0, or 2 (= dim -2 of the [1, 1, H, W] shard)
LINKS = tuple(int(x) for x in os.environ.get("FAG_LINKS", "1,2").split(","))
# schedule variants: "b" = balanced ring (far shard split between directions), "d" = desynchronized bank walks
VARIANTS = tuple(os.environ.get("FAG_VARIANTS", "base").split(","))
_FABRICS = {
    "1d": "FABRIC_1D",
    "1d_ring": "FABRIC_1D_RING",
    "1d_neighbor_exchange": "FABRIC_1D_NEIGHBOR_EXCHANGE",
    "2d": "FABRIC_2D",
    "2d_torus_x": "FABRIC_2D_TORUS_X",
    "2d_torus_y": "FABRIC_2D_TORUS_Y",
    "2d_torus_xy": "FABRIC_2D_TORUS_XY",
}
FABRICS = tuple(os.environ.get("FAG_FABRICS", ",".join(_FABRICS)).split(","))
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
}
_QB_TOPOS = [t for t, v in _TOPOS.items() if v[0][0] * v[0][1] == 4]
TOPOS = tuple(os.environ.get("FAG_TOPOS", ",".join(_QB_TOPOS)).split(","))
MESH_SHAPES = sorted({_TOPOS[t][0] for t in TOPOS})
_REPORT = []


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    if _REPORT:
        logger.info("\n".join(_REPORT))


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
    rows, cols = tuple(mesh_device.shape)
    H, W = SHAPE
    torch.manual_seed(0)
    dtype = _DTYPES[DTYPE_NAME]
    host = torch.randn((rows, cols, H, W), dtype=torch.float32 if dtype == ttnn.float32 else torch.bfloat16)
    inp = ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )
    if dtype == ttnn.bfloat8_b:  # lossy: compare against what the device holds
        host = torch.stack([ttnn.to_torch(t) for t in ttnn.get_device_tensors(inp)]).reshape(rows, cols, H, W)
    shard_bytes = (H // 32) * (W // 32) * int(inp.buffer_aligned_page_size())
    fabric = str(ttnn.get_fabric_config()).split(".")[-1]
    for topo_name in TOPOS:
        shape, cluster_axis, topo = _TOPOS[topo_name][:3]
        scheme = _TOPOS[topo_name][3] if len(_TOPOS[topo_name]) > 3 else "ring"
        if shape != (rows, cols):
            continue
        topology = getattr(ttnn.Topology, topo)
        if scheme == "dual_cycles":  # one group over the whole mesh, output in row-major chip order
            groups = [[(r, c) for r in range(rows) for c in range(cols)]]
        else:
            groups = build_groups((rows, cols), cluster_axis)
        G = len(groups[0])
        for num_links, variant in [(l, v) for l in LINKS for v in VARIANTS]:
            balance, desync = "b" in variant and variant != "base", "d" in variant and variant != "base"
            kw = dict(balance=balance, desync=desync)
            tag = f"    {fabric:<27} {topo_name:<15} G={G} links={num_links} {variant:<5}"
            try:
                out = fabric_all_gather(
                    inp, cluster_axis=cluster_axis, topology=topology, num_links=num_links, dim=DIM, scheme=scheme, **kw
                )
            except (ValueError, RuntimeError) as e:
                msg = str(e).splitlines()[0][:110]
                _REPORT.append(f"{tag}  unsupported: {msg}")
                continue
            ttnn.synchronize_device(mesh_device)
            got = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
            for grp in groups:
                parts = [host[r, c].reshape(1, 1, H, W) for r, c in grp]
                expected = torch.cat(parts, dim=DIM)
                for r, c in grp:
                    assert torch.equal(
                        got[r * cols + c].reshape(expected.shape).to(expected.dtype), expected
                    ), f"{fabric}/{topo_name}/links={num_links}: chip ({r},{c}) output != gathered shards"
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
            geo = f"busiest hop carries {busiest:.1f} shards, {nbrs} neighbours used per chip"
            if EMULE or TRIALS == 0:
                _REPORT.append(f"{tag}  bit-exact on all {rows * cols} chips  ✓  {geo}  (no timing)")
                continue
            samples = []
            for _ in range(TRIALS):
                ttnn.ReadDeviceProfiler(mesh_device)
                fabric_all_gather(
                    inp,
                    cluster_axis=cluster_axis,
                    topology=topology,
                    num_links=num_links,
                    dim=DIM,
                    scheme=scheme,
                    output=out,
                    **kw,
                )
                samples.append(_slowest_chip_ns(mesh_device))
            ns = statistics.median(samples)
            _REPORT.append(
                f"{tag}  {ns:>10.0f} ns  effective receive {shard_bytes * (G - 1) / ns:6.2f} GB/s per chip  ✓  {geo}"
            )
    _REPORT.insert(
        0,
        f"\n=== fabric_all_gather  box={socket.gethostname()}  arch={mesh_device.arch()}  payload={PAYLOAD}B  "
        f"shard={H}x{W} {DTYPE_NAME} dim={DIM} ({shard_bytes / 2**20:.0f} MiB/chip)  trials={TRIALS} (median) ===",
    ) if not _REPORT or not _REPORT[0].startswith("\n===") else None
