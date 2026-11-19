# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run the site's cases smallest first: the SFPU probe, the 256-wide cases on 1 and 2 cores,
the Hy4 width on 2, 8 and 16 cores, the layer chain, the latency cases, and after all of them the root-exchange
cases (the only ones with a NoC multicast). A case that is numerically off
does not stop the run (every case logs all its numbers and fails at its end); a device operation that times out
does: the board is then unusable and every further case would only wait for its own timeout."""

import pytest


def _size(item) -> int:
    name = item.name
    if "sfpu_projection" in name:
        return 51 if name.startswith("test_latency") else 50
    if "site_dram" in name:
        return 41 if name.startswith("test_latency") else 40
    if "root_exchange" in name:
        # the only cases with a NoC multicast: after every other case, their latency last of all
        return 30 if name.startswith("test_latency") else 20 + ("C256" not in name)
    if name.startswith("test_probe"):
        return 0
    if name.startswith("test_latency"):
        return 9
    if name.startswith("test_layers"):
        return 8
    if "C256-K1" in name:
        return 1
    if "C256" in name:
        return 2
    for rank, tag in ((3, "-K2"), (4, "-K8"), (5, "-K16")):
        if tag + "-" in name or name.endswith(tag + "]"):
            return rank
    return 6


def pytest_collection_modifyitems(config, items):
    mine = [i for i, item in enumerate(items) if "ihc_site/tests" in str(item.fspath)]
    ordered = sorted((items[i] for i in mine), key=_size)
    for slot, item in zip(mine, ordered):
        items[slot] = item


_STOP = ("timed out", "timeout", "unrecoverable", "TT_METAL_OPERATION_TIMEOUT")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if "ihc_site/tests" not in str(item.fspath) or not report.failed or call.excinfo is None:
        return
    text = str(call.excinfo.value).lower()
    if any(word.lower() in text for word in _STOP):
        pytest.exit(f"{item.name}: a device operation timed out, stopping the run", returncode=3)
