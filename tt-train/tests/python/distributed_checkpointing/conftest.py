# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Makes this directory's helper (``dcp_oracle``) importable whether pytest collects this directory alone or the whole
``tt-train/tests/python`` tree, puts the process-wide generators back after every test here, since these tests reseed
and restore them, and lets each rank of a tt-run module report its outcome to the cross-world module."""

import json
import os
import random
import sys

import numpy as np
import pytest

import ttml

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


@pytest.fixture(autouse=True)
def _restore_process_generators():
    """The ttml, ttml.init, python, numpy and torch generators as they were before the test, so tests collected after
    this directory start from the state they would have had without it. Snapshotted through each generator's own API,
    not through the checkpoint code these tests check."""
    ctx = ttml.autograd.AutoContext.get_instance()
    torch = sys.modules.get("torch")
    saved = (
        ctx.get_generator_state(),
        random.getstate(),
        np.random.get_state(),
        ttml.init.get_rng_state(),
        torch.get_rng_state() if torch is not None else None,
    )
    yield
    ctx.set_generator_state(saved[0])
    random.setstate(saved[1])
    np.random.set_state(saved[2])
    ttml.init.set_rng_state(saved[3])
    if torch is not None:
        torch.set_rng_state(saved[4])


def pytest_sessionfinish(session, exitstatus):
    """With ``DCP_RANK_RESULTS_DIR`` set (``test_dcp_cross_world`` sets it when it runs a tt-run module), every rank
    writes ``rank-<MPI world rank>.json`` there: its exit status and how many tests passed, failed, errored or were
    skipped. The cross-world module reads these files rather than the ranks' output, whose lines tt-run can
    interleave."""
    out = os.environ.get("DCP_RANK_RESULTS_DIR")
    if not out:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    stats = reporter.stats if reporter is not None else {}
    outcome = {"exitstatus": int(exitstatus)}
    outcome.update({k: len(stats.get(k, [])) for k in ("passed", "failed", "error", "skipped", "xfailed", "xpassed")})
    rank = os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMIX_RANK", "0"))
    with open(os.path.join(out, f"rank-{rank}.json"), "w") as f:
        json.dump(outcome, f)
