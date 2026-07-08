# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
pytest hooks for the accuracy suite (auto-loaded by pytest; nothing imports it).

They bracket the whole run so each per-op file reflects only this run:
  - before the run: delete leftover shard files (clear_shards)
  - after the run: merge this run's shards into one file per op (merge_shards)

The merged output is Parquet by default; pass --csv (or set
ACCURACY_OUTPUT_FORMAT=csv for the run_test.sh wrapper, which can't forward the
flag) to emit CSV instead.

When tests run in parallel (pytest -n) there are many worker processes plus one
controller. We run clear/merge only on the controller (the single process
without `config.workerinput`) so it happens once, not once per worker.
"""

import os

from helpers.logger import logger

_MODE_TO_FUNC = {
    "accuracy": "test_sfpu_accuracy_sweep",
    "perf": "test_sfpu_perf_sweep",
    "both": "test_sfpu_accuracy_and_perf_sweep",
}
_SWEEP_FUNCS = set(_MODE_TO_FUNC.values())


def pytest_collection_modifyitems(config, items):
    mode = config.getoption("--mode", default=None)
    if not mode:
        return
    target = _MODE_TO_FUNC[mode]
    deselected = [
        it
        for it in items
        if getattr(it, "function", None) is not None
        and it.function.__name__ in _SWEEP_FUNCS
        and it.function.__name__ != target
    ]
    if deselected:
        drop = {id(it) for it in deselected}
        config.hook.pytest_deselected(items=deselected)
        items[:] = [it for it in items if id(it) not in drop]


def pytest_addoption(parser):
    parser.addoption(
        "--csv",
        action="store_true",
        default=False,
        help="Write the merged per-op files as CSV instead of the default "
        "Parquet. Same as setting ACCURACY_OUTPUT_FORMAT=csv.",
    )


def _is_controller(config) -> bool:
    return not hasattr(config, "workerinput")


def pytest_configure(config):
    if _is_controller(config):
        from accuracy.accuracy_harness import clear_shards

        clear_shards()


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if _is_controller(config):
        from accuracy.accuracy_harness import (
            DEFAULT_OUTPUT_FORMAT,
            OUTPUT_DIR,
            merge_shards,
        )

        if config.getoption("--csv"):
            output_format = "csv"
        else:
            output_format = os.getenv("ACCURACY_OUTPUT_FORMAT", DEFAULT_OUTPUT_FORMAT)
        written = merge_shards(output_format)
        if written:
            logger.success(
                "Merged {} per-op {} file(s) into {}",
                len(written),
                output_format,
                OUTPUT_DIR,
            )
