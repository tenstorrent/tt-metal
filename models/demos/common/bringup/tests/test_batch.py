# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F259: several device-test runs in this one process, the mesh opened and the model loaded once (core/batch.py,
testing/batch.py). BRINGUP_BATCH=<plan.json> (the orchestrator's A/B: per run its own env, results dir and log), or
BRINGUP_BATCH_CMD=<a gate unit's command> (its runs in order, output here, stop at the first failure). Each run keeps
its own pytest-timeout; this test has none of its own (hangs: the safe runner's dispatch timeout). F269: a run
that needs another model ends this process there; the driver (core/batch.drive) runs the rest in a fresh one."""

import os

from models.demos.common.bringup.core import batch as B
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()


@mesh_parametrize(timeout_s=0)
def test_batch(mesh_device):
    plan, path = batch.load_plan()
    failed = batch.run_plan(mesh_device, S.mesh, plan, path)
    assert not failed, f"batch runs failed: {failed}"
    # F269: runs left for a fresh process are the driver's (core/batch.drive); a process nobody drives says so
    assert not batch.DEFERRED or os.environ.get(B.DRIVEN_ENV) == "1", (
        f"batch runs not run (they need another model than this process held; core/batch.drive runs them in a fresh "
        f"process): {batch.DEFERRED}"
    )
