# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-identity of the block's output across two builds, for the cases the planner runs as row blocks (the MoE
head and the MoE tail looping over the token slices): MOE_GOLDEN=record saves each case's eager output (combine snap
on: the combine's sums do not depend on landing order) to MOE_GOLDEN_DIR; MOE_GOLDEN=check runs a later build and
requires every element equal (a missing recording fails it). Opt-in: without MOE_GOLDEN the test skips. Used to prove
that a new implementation of the loops (e.g. kernels that loop natively instead of being relaunched in place) computes
exactly what the old one did."""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.moe_block.tests.test_moe_block_models import CASES, COMBINE_SNAP, MODELS, _params, _setup, _to_host

MODE = os.environ.get("MOE_GOLDEN")  # (record / check; unset: the test skips - no recording to compare with)
DIR = Path(os.environ.get("MOE_GOLDEN_DIR", "/tmp/moe_golden"))
# (the cases the planner runs as row blocks: K3 at 7680 / 10240 - 2 loops on 8x4, 3 / 2 on the 4x4 sub-torus)
LOOP_IDS = {"kimi_k3-8x4-10240", "kimi_k3-4x4-y-7680", "kimi_k3-4x4-y-10240"}
GOLDEN_CASES = [c for c in CASES if c.id in LOOP_IDS]


@pytest.mark.skipif(
    "MOE_GOLDEN" not in os.environ,
    reason="cross-build comparison requires MOE_GOLDEN=record or check",
)
@pytest.mark.parametrize(
    "case, mesh_device, device_params", _params(GOLDEN_CASES), indirect=["mesh_device", "device_params"]
)
@pytest.mark.timeout(1800)
def test_moe_block_loops_golden(case, mesh_device, device_params):
    assert MODE in ("record", "check"), f"MOE_GOLDEN={MODE}: record or check"
    path = DIR / f"{case.id}.pt"
    if MODE == "check" and not path.is_file():  # (asked to compare: a missing recording is a failure, not a skip)
        pytest.fail(f"MOE_GOLDEN=check: no recording {path} - record it with MOE_GOLDEN=record on the old build")
    run = _setup(case, mesh_device, combine_snap=COMBINE_SNAP)
    H = MODELS[case.model].cfg.EMB_SIZE
    for _ in range(2):  # (the first call loads the weights and builds the programs; the second is the one kept)
        y = run(eager=True)
    got = _to_host(y, mesh_device, case.chunk, H, case)
    loops = run.ffn._block(case.chunk).row_blocks
    if MODE == "record":
        DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"y": got, "loops": loops}, path)
        logger.info(f"GOLDEN recorded {case.id}: {loops} row blocks -> {path}")
        return
    want = torch.load(path)
    diff = int((want["y"] != got).sum())
    logger.info(f"GOLDEN {case.id}: {loops} row blocks (recorded with {want['loops']}): {diff} elements differ")
    assert loops == want["loops"], f"{case.id}: {loops} row blocks now, {want['loops']} when recorded"
    if diff:
        # (where the differences sit: the rows in 24 bands - a row block's rows show as whole bands - and the columns
        # in 8; the output saved beside the recording)
        bad = want["y"] != got
        rows, cols = bad.any(1).float(), bad.any(0).float()
        rb = [round(float(b.mean()), 2) for b in rows.chunk(24)]
        cb = [round(float(b.mean()), 2) for b in cols.chunk(8)]
        logger.info(f"GOLDEN {case.id}: differing rows by band {rb}; columns by band {cb}")
        torch.save({"y": got, "loops": loops}, DIR / f"{case.id}.got.pt")
    assert diff == 0, f"{case.id}: {diff} of {got.numel()} elements differ from the recorded build"
