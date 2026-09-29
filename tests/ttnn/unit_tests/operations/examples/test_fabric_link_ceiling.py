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
    VARIANTS,
    fabric_link_ceiling,
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


def _ring(mesh_device, num_links, packet_bytes, fill):
    rows, cols = tuple(mesh_device.shape)
    words = packet_bytes // 4
    if fill:
        # distinct per chip / link / slot, so a misrouted or stale packet is caught
        host = torch.randint(0, 2**31 - 1, (rows, cols, num_links * SLOTS, words), dtype=torch.int32)
    else:
        host = torch.zeros((rows, cols, num_links * SLOTS, words), dtype=torch.int32)
    t = ttnn.from_torch(
        host,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ring_memory_config(num_links, SLOTS, packet_bytes),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )
    return host, t


@pytest.mark.parametrize(
    "device_params",
    [pytest.param(_device_params(f, p), id=f"fabric_{f}_payload{p}") for f in FABRICS for p in PAYLOADS],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_fabric_link_ceiling(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    packet_bytes = ttnn.get_tt_fabric_max_payload_size_bytes()
    packets = max(1, int(MB_PER_LINK * 2**20) // packet_bytes)
    link_bytes = packets * packet_bytes
    sem_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(max(LINKS) - 1, 0))])
    sem = ttnn.create_global_semaphore(mesh_device, sem_cores, 0)
    sem_addr = int(ttnn.get_global_semaphore_address(sem))

    fabric = ttnn.get_fabric_config()
    _REPORT.append(
        f"\n=== fabric_link_ceiling  box={socket.gethostname()}  arch={mesh_device.arch()}  fabric={fabric}  "
        f"payload={packet_bytes}B  {link_bytes / 2**20:.1f} MiB/link/direction  trials={TRIALS} (median) ==="
    )
    _REPORT.append(f"    {'variant':<17} {'dir':<4} {'links':>5} {'kernel ns':>12} {'GB/s per link-dir':>18}")
    for num_links in LINKS:
        for direction in RUN_DIRECTIONS:
            for variant in RUN_VARIANTS:
                src_host, src = _ring(mesh_device, num_links, packet_bytes, fill=True)
                _, dst = _ring(mesh_device, num_links, packet_bytes, fill=False)
                run = lambda: fabric_link_ceiling(
                    mesh_device,
                    src,
                    dst,
                    sem_addr,
                    variant=variant,
                    direction=direction,
                    num_links=num_links,
                    slots=SLOTS,
                    packet_bytes=packet_bytes,
                    packets_per_link=packets,
                )
                run()
                ttnn.synchronize_device(mesh_device)

                # correctness: each receiving chip's landing ring == its peer's source ring
                got = torch.stack([ttnn.to_torch(t).to(torch.int64) for t in ttnn.get_device_tensors(dst)])
                got = got.reshape(rows, cols, num_links * SLOTS, packet_bytes // 4)
                for r in range(rows):
                    if direction == "bi" or r == 1:
                        assert torch.equal(
                            got[r], src_host[1 - r].to(torch.int64)
                        ), f"{variant}/{direction}: row {r} landing != peer source"
                    else:
                        assert not got[r].any(), f"{variant}/{direction}: row {r} received data it should not have"

                samples = []
                for _ in range(TRIALS):
                    ttnn.ReadDeviceProfiler(mesh_device)  # flush the window
                    run()
                    samples.append(_kernel_ns_per_launch(mesh_device, 1))
                ns = statistics.median(samples)
                _REPORT.append(f"    {variant:<17} {direction:<4} {num_links:>5} {ns:>12.0f} {link_bytes / ns:>18.2f}")
