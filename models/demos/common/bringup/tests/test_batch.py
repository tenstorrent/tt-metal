# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F259: several device-test runs in this one process, the mesh opened and the model loaded once (core/batch.py,
testing/batch.py). BRINGUP_BATCH=<plan.json> (the orchestrator's A/B: per run its own env, results dir and log), or
BRINGUP_BATCH_CMD=<a gate unit's command> (its runs in order, output here, stop at the first failure). Each run keeps
its own pytest-timeout; this test has none of its own (hangs: the safe runner's dispatch timeout)."""

from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()


@mesh_parametrize(timeout_s=0)
def test_batch(mesh_device):
    plan, path = batch.load_plan()
    failed = batch.run_plan(mesh_device, S.mesh, plan, path)
    assert not failed, f"batch runs failed: {failed}"
