# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `fabric_ring_exchange` example: a fused CCL op's stats exchange on a 4-chip ring, by itself.

Every relay / direct case checks every chip's landing bit-exactly. Timing is the device profiler's kernel duration per
chip and call: untraced (one call at a time: the last-launched chip, which does not wait at the ready fence, and the
slowest chip) and traced (T calls back to back: the slowest chip per call). Each number is a median over the calls.

    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_ring_exchange.py -s

Overrides: RX_VARIANTS (empty,fence,fence_direct,relay,direct), RX_BYTES (slot bytes per link and wave, comma list),
RX_LINKS (1,2), RX_IDLE (idle compute cores, comma list), RX_UNTRACED, RX_TRACED (calls per case).
See ttnn/ttnn/operations/examples/fabric_ring_exchange/README.md.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics

import pytest
import torch
import ttnn

from ttnn.operations.examples.fabric_ring_exchange import (
    Resources,
    expected_landing,
    idle_cores,
    port_cores,
    ring_exchange,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_list = lambda name, default: [x for x in os.environ.get(name, default).split(",") if x]
VARIANTS = _list("RX_VARIANTS", "empty,fence,fence_direct,relay,direct")
BYTES = [int(x) for x in _list("RX_BYTES", "128,2560,8192,32768")]
LINKS = [int(x) for x in _list("RX_LINKS", "1,2")]
IDLE = [int(x) for x in _list("RX_IDLE", "100")]
UNTRACED = int(os.environ.get("RX_UNTRACED", "7"))
TRACED = int(os.environ.get("RX_TRACED", "8"))
PAYLOAD = 14 * 1024 + 64  # the production router payload on Blackhole (conftest.production_router_config)


def _router():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = PAYLOAD
    return cfg


def _per_chip_ns(mesh):
    """{device id: [kernel ns of each program since the last read, in order]}."""
    ttnn.ReadDeviceProfiler(mesh)
    out = {}
    for dev, programs in (ttnn.get_latest_programs_perf_data() or {}).items():
        out[dev] = [
            float(p.program_analyses_results[_DURATION_KEY].duration)
            for p in programs
            if _DURATION_KEY in (getattr(p, "program_analyses_results", None) or {})
        ]
    return out


def _check(res, variant):
    if variant not in ("relay", "direct"):
        return
    got = ttnn.to_torch(res.scratch, mesh_composer=ttnn.ConcatMeshToTensor(res.mesh, dim=0)).to(torch.int32)
    got = got.reshape(res.G, 2 * res.links, res.words)
    exp = expected_landing(res, variant)
    # Only the slots this variant writes: the landing is shared by every variant of one Resources, and relay and
    # direct land a given peer's slot on different ports (the other port keeps the previous variant's copy).
    # Every written word is non-zero (Resources.pattern), so the expectation marks them.
    written = exp != 0
    bad = (got != exp) & written
    assert not bad.any(), f"{variant}: landing mismatch on {bad.nonzero()[:8].tolist()}"


def _measure(mesh, res, variant, idle):
    run = lambda: ring_exchange(res, variant, idle, PAYLOAD)
    run()  # compile + first call
    ttnn.synchronize_device(mesh)
    _check(res, variant)
    _per_chip_ns(mesh)  # drop the warm-up
    last, slow = [], []
    for _ in range(UNTRACED):
        run()
        ttnn.synchronize_device(mesh)
        per = [v[-1] for v in _per_chip_ns(mesh).values() if v]
        last.append(min(per))
        slow.append(max(per))
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(TRACED):
        run()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.release_trace(mesh, tid)
    ttnn.synchronize_device(mesh)
    _check(res, variant)
    per = [v[-TRACED:] for v in _per_chip_ns(mesh).values() if len(v) >= TRACED]
    traced = [max(chip[i] for chip in per) for i in range(TRACED)]
    us = lambda v: statistics.median(v) / 1000
    return us(last), us(slow), us(traced), (min(traced) / 1000, max(traced) / 1000)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "fabric_router_config": _router(),
            "trace_region_size": 1 << 22,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["mesh1x4"], indirect=True)
def test_fabric_ring_exchange(mesh_device):
    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    print(
        f"\n=== fabric_ring_exchange  box={socket.gethostname()}  arch={mesh.arch()}  ring=1x4 FABRIC_2D_TORUS_XY  "
        f"payload={PAYLOAD}B  untraced={UNTRACED} traced={TRACED}  (us: untraced last-launched / untraced slowest / "
        f"traced slowest [range])",
        flush=True,
    )
    for links in LINKS:
        ports = port_cores(grid, links)
        for nbytes in BYTES:
            res = Resources(mesh, ports, nbytes, num_waves=1)
            for n_idle in IDLE:
                idle = idle_cores(grid, ports, n_idle)
                for variant in VARIANTS:
                    if variant in ("empty", "fence", "fence_direct") and nbytes != BYTES[0]:
                        continue  # no payload: one row per (links, idle) is enough
                    last, slow, traced, (lo, hi) = _measure(mesh, res, variant, idle)
                    size = "-" if variant in ("empty", "fence", "fence_direct") else f"{nbytes}B"
                    print(
                        f"FABRIC_RING_EXCHANGE links={links} idle={n_idle:3d} {variant:12s} slot={size:>7s}  "
                        f"{last:7.2f} / {slow:7.2f} / {traced:7.2f} [{lo:.2f}-{hi:.2f}]",
                        flush=True,
                    )
