# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""nd_stage matmul probes in the FF1 run pattern: many short bursts with the host syncing in between.

test_nd_stage runs each probe as one long on-device loop, so a core sees one idle -> full load step per arm.
test_ff1_matmul runs a matmul for about a millisecond and then idles while the host syncs, a thousand times
per run. A fault that needs the load step rather than sustained load shows in FF1 and not in nd_stage.

Here every burst is one short run of a probe on the full grid, analyzed like an nd_stage arm: the on-core
checker compares the burst's later iterations with its first, and the first iteration's output is compared
across all cores and chips, which flags a core whose first iteration after the idle gap is wrong. The gap is
the host's sync and analysis of the previous burst; its length is logged and reported.

Env:
  ND_BURSTS        bursts per probe (default 300)
  ND_BURST_MS      on-device time per burst (default 1.0)
  ND_BURST_PROBES  comma list of nd_stage probes (default MM_DEST,MM_L1ACC)
  ND_FIDELITY      math fidelity (default LoFi, as FF1 runs)
  ND_DATA          random (default) | int
  ND_REPORT_DIR    where the JSON report goes (default generated/nd_stage)
"""

import json
import os
import socket
import time
from collections import Counter
from pathlib import Path

import pytest
from loguru import logger

from models.common.utility_functions import skip_for_wormhole_b0
from tests.didt.nd_stage import nd_stage_lib as nd


def _ms_stats(values):
    if not values:
        return None
    v = sorted(values)
    return {
        q: round(1000 * v[min(len(v) - 1, int(f * len(v)))], 3)
        for q, f in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))
    }


@skip_for_wormhole_b0("Targets the Blackhole FF1 non-determinism")
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="galaxy")], indirect=["mesh_device"])
def test_nd_burst(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    bursts = int(os.environ.get("ND_BURSTS", 300))
    burst_s = float(os.environ.get("ND_BURST_MS", 1.0)) / 1000
    probes = [p.strip() for p in os.environ.get("ND_BURST_PROBES", "MM_DEST,MM_L1ACC").split(",") if p.strip()]
    unknown = [p for p in probes if p not in nd.PROBES]
    assert probes and not unknown, f"ND_BURST_PROBES: unknown {unknown}, choose from {list(nd.PROBES)}"
    fidelity = os.environ.get("ND_FIDELITY", "LoFi")
    mode = os.environ.get("ND_DATA", "random")
    runner = nd.Runner(mesh_device, gx, gy, mode=mode, fidelity=fidelity)
    device_ids = list(mesh_device.get_device_ids())
    identity = nd.device_identity(device_ids)
    logger.info(f"nd_burst: {len(device_ids)} devices, grid {gx}x{gy}, {probes} x {bursts} bursts, {fidelity}")

    findings, notes, log = [], Counter(), []
    for p in probes:
        arm = nd.Arm(f"{p}:burst", {p: runner.cores})
        iters, rate = runner.calibrate(arm, p, burst_s)
        t, walls, gaps, end = None, [], [], None
        failed = Counter()
        for b in range(bursts):
            if end is not None:
                gaps.append(time.perf_counter() - end)
            t, wall = runner.run(arm, {p: iters}, t)
            end = time.perf_counter()
            walls.append(wall)
            f, n = nd.analyze_arm(runner, arm, t, device_ids)
            notes.update(n)
            for x in f:
                x.detail["burst"] = b
                failed[(x.device, x.core, x.kind)] += 1
            findings += f
        nd.free(t)
        rec = {
            "probe": p,
            "iters_per_burst": iters,
            "rate_it_per_s": round(rate, 1),
            "bursts": bursts,
            "burst_wall_ms": _ms_stats(walls),
            "gap_ms": _ms_stats(gaps),
            "failed": [
                {"device": d, "core": list(c), "kind": k, "bursts": n} for (d, c, k), n in sorted(failed.items())
            ],
        }
        log.append(rec)
        logger.info(
            f"{p:10s} {iters} iters/burst, burst {rec['burst_wall_ms']['median']} ms, gap {rec['gap_ms']['median']} ms, "
            f"failing (device, core, kind) {len(failed)}"
        )
        for (d, c, k), n in sorted(failed.items()):
            logger.error(f"  {p} dev {d:2d} core {c} {k}: {n}/{bursts} bursts | {nd.where(identity, d)}")

    for line, n in notes.items():
        logger.warning(f"{line} (x{n})")
    for x in findings[:200]:
        logger.info(f"  {x.arm:14s} dev {x.device:2d} core {x.core} {x.kind:9s} {x.detail}")

    report = {
        "host": socket.gethostname(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "devices": device_ids,
        "identity": identity,
        "grid": [gx, gy],
        "fidelity": fidelity,
        "data": mode,
        "burst_ms": burst_s * 1000,
        "probes": log,
        "findings": [x.__dict__ | {"core": list(x.core)} for x in findings],
        "notes": dict(notes),
    }
    out_dir = Path(os.environ.get("ND_REPORT_DIR", "generated/nd_stage"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"nd_burst_{socket.gethostname()}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=1))
    logger.info(f"report: {path}")

    cores = {(x.device, x.core) for x in findings}
    assert not findings, f"ND or wrong output on {len(cores)} (device, core) pairs; see {path}"
