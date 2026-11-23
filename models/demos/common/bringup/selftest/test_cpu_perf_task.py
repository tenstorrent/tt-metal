# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F302: a perf-step task with `device: false` is gated, not measured as an on / off device A/B."""

from models.demos.common.bringup.orchestrator import Orchestrator


def test_a_cpu_only_perf_task_is_not_a_perf_pick():
    pick = Orchestrator.perf_pick
    assert not pick(None, {"step": "perf", "role": "perf", "device": False})
    assert pick(None, {"step": "perf", "role": "perf", "device": True})
    assert pick(None, {"step": "perf", "role": "perf"})  # unmarked perf picks stay picks
    assert not pick(None, {"step": "perf", "device": True})  # no role: the profile task, as before
