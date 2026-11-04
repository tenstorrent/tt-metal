# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0


def pytest_addoption(parser):
    parser.addoption(
        "--noc-wedge-launches",
        action="store",
        default=None,
        help="NoC wedge reproducers: launches per case (default: the case's own count)",
    )
    parser.addoption(
        "--noc-wedge-sync",
        action="store",
        default=None,
        help="NoC wedge reproducers: synchronize (and log) every N launches (default 50)",
    )
    parser.addoption(
        "--noc-wedge-exit-on-hang",
        action="store_true",
        default=False,
        help="NoC wedge reproducers: on a dispatch timeout, os._exit at once (no device close, no further host access "
        "to the hung chip); pair with scripts/run_safe_pytest.sh --no-triage",
    )
