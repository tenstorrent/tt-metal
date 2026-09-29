# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `fabric_link_ceiling` example: how fast can the simplest op push bytes over one link?

    # correctness (landing ring == peer's source ring) + device kernel time, per fabric x payload
    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_link_ceiling.py -s

See ttnn/ttnn/operations/examples/fabric_link_ceiling/README.md.
"""

import os

# In-process device profiler (all three, before the device opens) + a quiet C++ logger.
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

from ttnn.operations.examples.fabric_link_ceiling import (
    DIRECTIONS,
    NOC0,
    NOC1,
    VARIANTS,
    fabric_link_ceiling,
    link_cores,
    ring_memory_config,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
SLOTS = 8  # L1 ring depth per link core (source and landing)
MB_PER_LINK = float(os.environ.get("FLC_MB", "64"))  # bytes streamed per link per direction per launch
LINKS = tuple(int(x) for x in os.environ.get("FLC_LINKS", "1,2").split(","))
TRIALS = int(os.environ.get("FLC_TRIALS", "3"))
_FABRICS = {"1d": ttnn.FabricConfig.FABRIC_1D, "2d": ttnn.FabricConfig.FABRIC_2D}
FABRICS = tuple(os.environ.get("FLC_FABRICS", "1d,2d").split(","))
PAYLOADS = tuple(int(x) for x in os.environ.get("FLC_PAYLOADS", "4352,8704,14336,15232").split(","))
RUN_VARIANTS = tuple(os.environ.get("FLC_VARIANTS", ",".join(VARIANTS)).split(","))
RUN_DIRECTIONS = tuple(os.environ.get("FLC_DIRECTIONS", ",".join(DIRECTIONS)).split(","))


def _parse_cores(spec):
    return [ttnn.CoreCoord(*(int(v) for v in c.split(","))) for c in spec.split(";")]


# Placements for the 2-link test, as logical cores ("x,y;x,y"): link l is served by the l-th core.
# `under_eth` puts each core in the NoC column of its link's Ethernet core. Those columns are board-specific:
# these were read from a NoC trace on a 4x p150a QuietBox (links 0/1 on Ethernet cores at NoC x = 3 / 4,
# logical x = 2 / 3). Override with FLC_PLACEMENTS="name=x,y;x,y|name=...".
# NoC0 senders are opt-in (FLC_SENDER_NOCS=1,0): the `adjacent` placement + NoC0 + both directions hangs
# deterministically on a 4x p150a QuietBox (see README).
SENDER_NOCS = tuple(int(x) for x in os.environ.get("FLC_SENDER_NOCS", "1").split(","))
_DEFAULT_PLACEMENTS = "adjacent=0,0;1,0|rows=0,0;0,1|under_eth=2,0;3,0"
PLACEMENTS = {
    name: _parse_cores(spec)
    for name, spec in (p.split("=") for p in os.environ.get("FLC_PLACEMENTS", _DEFAULT_PLACEMENTS).split("|"))
}

_REPORT = []


def _device_params(fabric, payload):
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = payload
    return {
        "fabric_config": _FABRICS[fabric],
        "fabric_router_config": router,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    if _REPORT:
        logger.info("\n".join(_REPORT))


def _kernel_ns_per_launch(mesh_device, launches):
    """Per launch, the slowest chip's kernel time (chips run concurrently), averaged over launches."""
    ttnn.ReadDeviceProfiler(mesh_device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    chip_means = []
    for programs in per_chip.values():
        ns = [
            float(p.program_analyses_results[_DURATION_KEY].duration)
            for p in programs
            if _DURATION_KEY in (getattr(p, "program_analyses_results", None) or {})
        ]
        if ns:
            chip_means.append(sum(ns) / len(ns))
    assert chip_means, "profiler produced no kernel durations (profiler-enabled build?)"
    return max(chip_means)


def _ring(mesh_device, cores, packet_bytes, fill):
    rows, cols = tuple(mesh_device.shape)
    words = packet_bytes // 4
    shape = (rows, cols, len(cores) * SLOTS, words)
    # distinct per chip / link / slot, so a misrouted or stale packet is caught
    host = torch.randint(0, 2**31 - 1, shape, dtype=torch.int32) if fill else torch.zeros(shape, dtype=torch.int32)
    t = ttnn.from_torch(
        host,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ring_memory_config(cores, SLOTS, packet_bytes),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )
    return host, t


def _run_case(mesh_device, *, cores, variant, direction, sender_noc=NOC1):
    """Correctness-check one case, then return (median kernel ns, bytes per link per direction)."""
    rows, cols = tuple(mesh_device.shape)
    packet_bytes = ttnn.get_tt_fabric_max_payload_size_bytes()
    packets = max(1, int(MB_PER_LINK * 2**20) // packet_bytes)
    sem = ttnn.create_global_semaphore(mesh_device, ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]), 0)
    sem_addr = int(ttnn.get_global_semaphore_address(sem))
    src_host, src = _ring(mesh_device, cores, packet_bytes, fill=True)
    _, dst = _ring(mesh_device, cores, packet_bytes, fill=False)
    run = lambda: fabric_link_ceiling(
        mesh_device,
        src,
        dst,
        sem_addr,
        variant=variant,
        direction=direction,
        cores=cores,
        slots=SLOTS,
        packet_bytes=packet_bytes,
        packets_per_link=packets,
        sender_noc=sender_noc,
    )
    run()
    ttnn.synchronize_device(mesh_device)

    # correctness: each receiving chip's landing ring == its peer's source ring
    got = torch.stack([ttnn.to_torch(t).to(torch.int64) for t in ttnn.get_device_tensors(dst)])
    got = got.reshape(rows, cols, len(cores) * SLOTS, packet_bytes // 4)
    for r in range(rows):
        if direction == "bi" or r == 1:
            assert torch.equal(
                got[r], src_host[1 - r].to(torch.int64)
            ), f"{variant}/{direction}: row {r} != peer source"
        else:
            assert not got[r].any(), f"{variant}/{direction}: row {r} received data it should not have"

    samples = []
    for _ in range(TRIALS):
        ttnn.ReadDeviceProfiler(mesh_device)  # flush the window
        run()
        samples.append(_kernel_ns_per_launch(mesh_device, 1))
    return statistics.median(samples), packets * packet_bytes


def _header(mesh_device, title, link_bytes):
    _REPORT.append(
        f"\n=== {title}  box={socket.gethostname()}  arch={mesh_device.arch()}  fabric={ttnn.get_fabric_config()}  "
        f"payload={ttnn.get_tt_fabric_max_payload_size_bytes()}B  {link_bytes / 2**20:.1f} MiB/link/direction  "
        f"trials={TRIALS} (median) ==="
    )


@pytest.mark.parametrize(
    "device_params",
    [pytest.param(_device_params(f, p), id=f"fabric_{f}_payload{p}") for f in FABRICS for p in PAYLOADS],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_fabric_link_ceiling(mesh_device):
    """Default placement (adjacent cores), sender on NoC1: variant x direction x link count."""
    rows_out = []
    link_bytes = None
    for num_links in LINKS:
        for direction in RUN_DIRECTIONS:
            for variant in RUN_VARIANTS:
                ns, link_bytes = _run_case(
                    mesh_device, cores=link_cores(num_links), variant=variant, direction=direction
                )
                rows_out.append(f"    {variant:<17} {direction:<4} {num_links:>5} {ns:>12.0f} {link_bytes / ns:>18.2f}")
    _header(mesh_device, "fabric_link_ceiling", link_bytes)
    _REPORT.append(f"    {'variant':<17} {'dir':<4} {'links':>5} {'kernel ns':>12} {'GB/s per link-dir':>18}")
    _REPORT.extend(rows_out)


@pytest.mark.parametrize(
    "device_params",
    [pytest.param(_device_params("1d", 14336), id="fabric_1d_payload14336")],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_fabric_link_placement(mesh_device):
    """2 links, header_ring: which cores serve the links x which NoC the senders use."""
    rows_out = []
    link_bytes = None
    for name, cores in PLACEMENTS.items():
        noc_xy = " ".join(f"({v.x},{v.y})" for v in (mesh_device.worker_core_from_logical_core(c) for c in cores))
        for sender_noc, noc_name in [((NOC0, NOC1)[n], f"NoC{n}") for n in SENDER_NOCS]:
            for direction in RUN_DIRECTIONS:
                ns, link_bytes = _run_case(
                    mesh_device, cores=cores, variant="header_ring", direction=direction, sender_noc=sender_noc
                )
                rows_out.append(
                    f"    {name:<10} {noc_xy:<15} {noc_name:<5} {direction:<4} {ns:>12.0f} {link_bytes / ns:>18.2f}"
                )
    _header(mesh_device, "fabric_link_ceiling placement (2 links, header_ring)", link_bytes)
    _REPORT.append(
        f"    {'placement':<10} {'NoC coords':<15} {'send':<5} {'dir':<4} {'kernel ns':>12} {'GB/s per link-dir':>18}"
    )
    _REPORT.extend(rows_out)
