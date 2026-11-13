# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""ND stage localization: which part of a core goes non-deterministic, not only which core.

A run-to-run matmul gate says "core (6,8) on device 14 drifts". This test runs a set of probes, each of
which loads one part of the Tensix pipeline (or L1, or the NoC) and checks its own output against
iteration 0 on the core itself, then intersects the results per core:

  MM_DEST       unpack -> SrcA/B, FPU matmul with K=7168 accumulated in DEST, pack
  MM_L1ACC      the same, but partials summed by packer L1 accumulation (the production spill path)
  MM_DESTRD     MM_DEST with DEST also read back by the math RISC before the pack; its two records
                split "wrong in DEST" (unpack, FPU, DEST) from "wrong after DEST" (packer, L1)
  FPU_ELTWISE   FPU add, operand via unpacker 1 -> SrcB, DEST -> SrcA moves, no MVMUL
  ELTWISE_SRCB  the same with the roles swapped: operand via unpacker 0 -> SrcA, DEST -> SrcB moves
  SFPU_INT      int32 multiply in DEST on the SFPU (SFPMUL24, IADD, SHFT, LOADMACRO)
  SFPU_FP       fp32 x2 then x0.5 in DEST on the SFPU (SFPMAD), exact
  SFPU_LDST     int32 tiles copied DEST -> LREG -> DEST on the SFPU, no arithmetic
  UNPACK_PACK   int32 tiles cycled L1 -> DEST -> L1, no arithmetic
  COPY_SRCA     bf16 tiles cycled L1 -> SrcA -> DEST -> L1 (FPU datacopy)
  PACK_L1ACC    one DEST packed repeatedly with L1 accumulation, FPU idle
  L1_MARCH      March C- over 512 KB of L1 on BRISC, no NoC
  NOC_MCAST     row multicast of toggling patterns on NoC0, receivers compare with a local copy
  NOC1_MCAST    the same on NoC1

Every non-matmul probe also runs "loaded": on a checkerboard half of the grid with MM_DEST on the other
half, at both parities, so a fault that needs neighbouring matmul di/dt still shows. L1 and NoC loaded arms
run MM_DEST on the same cores. Outputs are also compared across all cores and chips (and against an exact
golden where one exists), which catches a core that is stably wrong.

The verdict for each failing (device, core) is the set of stages every failing probe exercises minus the
stages a passing probe clears. Example: fails the three MM probes and the DEST readback, passes the rest
-> "FPU MVMUL (multiplier array + adder tree)". Fails FPU_ELTWISE only -> "DEST -> SrcA move (MOVD2A)".
Each ND record also carries a per-bit flip histogram and the face columns hit, to compare with the bit
and column signatures syseng reports.

Not separable with kernels alone: DEST storage from its read ports, the multiplier from the adder tree,
unpacker and packer internal stages, the TRISC front ends, and clock or power delivery. L1 ECC counters
are not usable either: the scrubber also walks unpopulated rows and the firmware clears the counts.

The target is a Blackhole Galaxy: every probe runs on all 32 chips at once, on the full worker grid of each
(12x10 = 120 cores). The run-to-run determinism gates cap co-active cores at 110 (11x10); this test does not,
so column x=11 is covered and the matmul aggressors load the whole chip.

Wall time is ~ (number of arms) x ND_PROBE_SECONDS plus compile and calibration: 33 arms x 20 s ~ 11 min,
~ 14 min in all. ND_PROBE_SECONDS=15 trims it to ~ 11 min.

Env:
  ND_PROBE_SECONDS  seconds per arm (default 20)
  ND_ARMS           comma list of probes or arms to run, e.g. "MM_DEST,SFPU_INT:loaded0" (default all)
  ND_DATA           random (default) | int: small integers make the float probes exact against a golden
  ND_FIDELITY       math fidelity (default HiFi4)
  ND_REPORT_DIR     where the JSON report goes (default generated/nd_stage)
  ND_INJECT         positive control, PROBE[:pack]@x,y[@iter]: flip one output bit of a compute probe on that
                    core (every device). The test then passes only if exactly that core and probe are flagged.
                    SFPU_INT and FPU_ELTWISE localize to one stage; iter 0 exercises the consensus path.
                    MM_DESTRD flips a DEST bit before the pack (both of its records fail); MM_DESTRD:pack
                    flips only the packed copy.
"""

import json
import os
import socket
import time
from pathlib import Path

import pytest
from loguru import logger

from models.common.utility_functions import skip_for_wormhole_b0
from tests.didt.nd_stage import nd_stage_lib as nd


def _calibrate(runner, seconds, needed):
    """Iterations per probe for `seconds` of quiet running, measured on the full grid."""
    iters, rates = {}, {}
    for key in needed:
        if key in nd.PROBES:
            arm = nd.Arm(f"{key}:cal", {key: runner.cores})
        else:
            arm = nd.Arm(f"{key}:cal", extra=key)
        iters[key], rates[key] = runner.calibrate(arm, key, seconds)
        logger.info(f"calibrate {key:12s} {rates[key]:12.1f} it/s -> {iters[key]} iters")
    return iters, rates


def _fmt_core(v, identity):
    failed = ", ".join(f"{p}({'+'.join(c)})" for p, c in sorted(v["failed"].items()))
    if v["specific_to"]:
        what = "; ".join(f"{p}: {nd.PROBES[p].what}" if p in nd.PROBES else p for p in v["specific_to"])
        suspects = f"no single stage, every one is cleared by a passing probe; specific to {what}"
    else:
        suspects = "; ".join(v["suspects"])
    line = f"dev {v['device']:2d} core ({v['core'][0]},{v['core'][1]}): FAIL {failed} | suspects: {suspects}"
    if v["didt_only"]:
        line += " | only beside MM aggressors"
    if v["iter0_outlier"]:
        line += f" | iteration 0 was the outlier in {', '.join(v['iter0_outlier'])}"
    if v["dest_vs_pack"]:
        line += f" | MM_DESTRD: {v['dest_vs_pack']}"
    if loc := nd.where(identity, v["device"]):
        line += f" | {loc}"
    return line


@skip_for_wormhole_b0("The probes target the Blackhole matmul di/dt fault")
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="galaxy")], indirect=["mesh_device"])
def test_nd_stage(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    seconds = nd.env_seconds()
    mode = os.environ.get("ND_DATA", "random")
    inject = nd.parse_inject(os.environ.get("ND_INJECT"))
    if inject:
        assert inject.core[0] < gx and inject.core[1] < gy, f"ND_INJECT core {inject.core} outside the {gx}x{gy} grid"
    runner = nd.Runner(mesh_device, gx, gy, mode=mode, fidelity=os.environ.get("ND_FIDELITY", "HiFi4"), inject=inject)
    arms = nd.plan_arms(runner.cores, gx, os.environ.get("ND_ARMS"))
    assert arms, f"ND_ARMS={os.environ.get('ND_ARMS')!r} selects no arm"
    device_ids = list(mesh_device.get_device_ids())
    identity = nd.device_identity(device_ids)
    logger.info(f"nd_stage: {len(device_ids)} devices, grid {gx}x{gy}, {len(arms)} arms x {seconds:.0f} s, data={mode}")

    needed = sorted({p for a in arms for p in a.groups} | {a.extra for a in arms if a.extra})
    t0 = time.perf_counter()
    iters, rates = _calibrate(runner, seconds, needed)
    cal_s = time.perf_counter() - t0

    findings, notes, arm_log = [], [], []
    ran = {}
    for arm in arms:
        keys = list(arm.groups) + ([arm.extra] if arm.extra else [])
        t, wall = runner.run(arm, {k: iters[k] for k in keys})
        f, n = nd.analyze_arm(runner, arm, t, device_ids)
        nd.free(t)
        findings += f
        notes += n
        for k in keys + [nd.DESTRD_DEST for p in arm.groups if nd.PROBES[p].dest_check]:
            ran.setdefault(k, set()).add("loaded" if arm.loaded else "quiet")
        nd_cores = {(x.device, x.core) for x in f if x.kind == "ND"}
        wrong_cores = {(x.device, x.core) for x in f if x.kind != "ND"}
        arm_log.append(
            {
                "arm": arm.name,
                "wall_s": round(wall, 2),
                "iters": {k: iters[k] for k in keys},
                "nd": len(nd_cores),
                "wrong": len(wrong_cores),
            }
        )
        logger.info(
            f"{arm.name:22s} {wall:7.1f} s  ND cores {len(nd_cores):4d}  stable-wrong cores {len(wrong_cores):4d}"
        )

    verdicts = nd.localize(findings, ran)
    for line in notes:
        logger.warning(line)
    for v in verdicts:
        v["identity"] = identity.get(v["device"])
        logger.error(_fmt_core(v, identity))
    for x in findings:
        logger.info(f"  {x.arm:22s} dev {x.device:2d} core {x.core} {x.kind:9s} {x.detail}")

    report = {
        "host": socket.gethostname(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "devices": device_ids,
        "identity": identity,
        "grid": [gx, gy],
        "seconds_per_arm": seconds,
        "data": mode,
        "inject": inject.__dict__ if inject else None,
        "calibration_s": round(cal_s, 1),
        "rates_it_per_s": {k: round(v, 2) for k, v in rates.items()},
        "stages": {p: sorted(nd.stages_of(p)) for p in ran},
        "arms": arm_log,
        "verdicts": verdicts,
        "findings": [x.__dict__ | {"core": list(x.core)} for x in findings],
        "notes": notes,
    }
    out_dir = Path(os.environ.get("ND_REPORT_DIR", "generated/nd_stage"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"nd_stage_{socket.gethostname()}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=1))
    logger.info(f"report: {path}")

    if inject:
        flagged = {(x.probe, x.core) for x in findings}
        missed = set(device_ids) - {x.device for x in findings}
        assert (
            flagged == inject.expected() and not missed
        ), f"ND_INJECT {inject}: flagged {sorted(flagged)}, devices missed {sorted(missed)}; see {path}"
        return

    n_nd = len({(x.device, x.core) for x in findings if x.kind == "ND"})
    n_wrong = len({(x.device, x.core) for x in findings if x.kind != "ND"})
    assert not findings, f"ND on {n_nd} cores, stable-wrong on {n_wrong} cores; see {path}\n" + "\n".join(
        _fmt_core(v) for v in verdicts
    )
