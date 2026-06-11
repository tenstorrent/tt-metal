# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
pytest hooks for the accuracy suite (auto-loaded by pytest; nothing imports it).

They bracket the whole run so each per-op CSV reflects only this run:
  - before the run: delete leftover shard files (clear_shards)
  - after the run: merge this run's shards into one CSV per op (merge_shards)

When tests run in parallel (pytest -n) there are many worker processes plus one
controller. We run clear/merge only on the controller (the single process
without `config.workerinput`) so it happens once, not once per worker.
"""

from accuracy.accuracy_harness import OUTPUT_DIR, clear_shards, merge_shards


def _is_controller(config) -> bool:
    return not hasattr(config, "workerinput")


def pytest_configure(config):
    if _is_controller(config):
        clear_shards()


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if _is_controller(config):
        written = merge_shards()
        if written:
            print(f"\n[accuracy] merged {len(written)} per-op CSV(s) into {OUTPUT_DIR}")
