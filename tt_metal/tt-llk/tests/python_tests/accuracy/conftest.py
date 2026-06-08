# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Accuracy-suite collection hooks.

Controller-only lifecycle so the per-op CSVs reflect exactly the current run:
  - pytest_configure  : clear SHARD_DIR before workers spawn (no stale shards)
  - pytest_sessionfinish : merge current-run shards into per-op CSVs

Under xdist, only the controller process lacks `config.workerinput`.
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
