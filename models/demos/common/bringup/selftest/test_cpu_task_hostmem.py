# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F301: a CPU-only task's gate runs without the F284 device-open reserve; a device task keeps it."""

from models.demos.common.bringup.core import gate as G
from models.demos.common.bringup.core import hostmem as H

GB = 1024**3


def test_cpu_task_gate_skips_the_device_reserve():
    mi = {"MemAvailable": int(237 * GB)}  # the PS.1 node: 280 GiB leaked, 237 available
    cpu_env = G.cpu_task_env({"device": False, "step": "perf"})
    assert cpu_env == {H.RESERVE_ENV: "0"}
    ok, line = H.reserve_check(env=cpu_env, mi=mi)
    assert ok and "reserve check off" in line


def test_device_and_unmarked_tasks_keep_the_reserve():
    mi = {"MemAvailable": int(237 * GB)}
    for task in ({"device": True}, {"step": "perf"}):  # unmarked perf tasks count as device (ledger.py)
        env = G.cpu_task_env(task)
        assert env == {}
        ok, line = H.reserve_check(env=env, mi=mi)
        assert not ok and H.INFRA_TAG in line
