# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `fabric_gather_pair` example: a two-chip DRAM -> DRAM all-gather, one core per link.

    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_gather_pair.py -s

See ttnn/ttnn/operations/examples/fabric_gather_pair/README.md.
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
from loguru import logger

from ttnn.operations.examples.fabric_gather_pair import NOC0, NOC1, VARIANTS, fabric_gather_pair

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
SHAPE = tuple(int(x) for x in os.environ.get("FGP_SHAPE", "8192,4096").split(","))  # per-chip shard (H, W), bf16
TRIALS = int(os.environ.get("FGP_TRIALS", "3"))
PAYLOAD = int(os.environ.get("FGP_PAYLOAD", "14336"))
RUN_VARIANTS = tuple(os.environ.get("FGP_VARIANTS", ",".join(VARIANTS)).split(","))
# Diagnostic ablations, "|"-separated sets of "+"-joined names (dram_read, local_copy, fabric); "" = full op.
# NoC for the local copy: "same" (the sender's NoC1) or "noc0" (its own outbound port).
LOCAL_NOCS = tuple(os.environ.get("FGP_LOCAL_NOCS", "same,noc0").split(","))
ABLATIONS = [tuple(x for x in a.split("+") if x) for a in os.environ.get("FGP_ABLATE", "").split("|")]


def _parse_cores(spec):
    return [ttnn.CoreCoord(*(int(v) for v in c.split(","))) for c in spec.split(";")]


# Link cores (logical), link l on the l-th core. `under_eth` puts each core in the NoC column of its link's
# Ethernet core (board-specific; read from a NoC trace on a 4x p150a QuietBox).
_DEFAULT_PLACEMENTS = "1link_under_eth=2,0|2link_adjacent=0,0;1,0|2link_under_eth=2,0;3,0"
PLACEMENTS = {
    name: _parse_cores(spec)
    for name, spec in (p.split("=") for p in os.environ.get("FGP_PLACEMENTS", _DEFAULT_PLACEMENTS).split("|"))
}
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
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_router_config": _router(PAYLOAD),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
    ],
    ids=[f"fabric_1d_payload{PAYLOAD}"],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_fabric_gather_pair(mesh_device):
    rows, cols = tuple(mesh_device.shape)
    H, W = SHAPE
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
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 2 * H, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    shard_bytes = H * W * 2

    lines = [
        f"\n=== fabric_gather_pair  box={socket.gethostname()}  arch={mesh_device.arch()}  "
        f"fabric={ttnn.get_fabric_config()}  payload={ttnn.get_tt_fabric_max_payload_size_bytes()}B  "
        f"shard={H}x{W} bf16 ({shard_bytes / 2**20:.0f} MiB/chip)  trials={TRIALS} (median) ===",
        f"    {'placement':<16} {'NoC coords':<15} {'variant':<26} {'kernel ns':>11} {'GB/s per link-dir':>18} "
        f"{'GB/s per chip':>14}",
    ]
    for name, cores in PLACEMENTS.items():
        noc_xy = " ".join(f"({v.x},{v.y})" for v in (mesh_device.worker_core_from_logical_core(c) for c in cores))
        sem = ttnn.create_global_semaphore(mesh_device, ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]), 0)
        sem_addr = int(ttnn.get_global_semaphore_address(sem))
        for variant, local, ablate in [(v, l, a) for v in RUN_VARIANTS for l in LOCAL_NOCS for a in ABLATIONS]:
            run = lambda: fabric_gather_pair(
                mesh_device,
                inp,
                out,
                sem_addr,
                variant=variant,
                cores=cores,
                local_copy_noc=NOC0 if local == "noc0" else None,
                ablate=ablate,
            )
            run()
            ttnn.synchronize_device(mesh_device)
            got = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
            for r in range(rows):
                for c in range(cols):
                    expected = torch.cat([host[0, c], host[1, c]], dim=-2)
                    assert ablate or torch.equal(
                        got[r * cols + c].reshape(expected.shape), expected
                    ), f"{name}/{variant}: chip ({r},{c}) output != [row-0 shard ; row-1 shard]"
            samples = []
            for _ in range(TRIALS):
                ttnn.ReadDeviceProfiler(mesh_device)
                run()
                samples.append(_slowest_chip_ns(mesh_device))
            ns = statistics.median(samples)
            lines.append(
                f"    {name:<16} {noc_xy:<15} {variant + ('+local_noc0' if local == 'noc0' else '') + ('-' + '-'.join(ablate) if ablate else ''):<26} {ns:>11.0f} {shard_bytes / len(cores) / ns:>18.2f} "
                f"{shard_bytes / ns:>14.2f}"
            )
    _REPORT.extend(lines)
