# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run pytest over a file of node ids.

A file rather than argv because a full sweep is tens of thousands of ids and
would blow past ARG_MAX. A separate script rather than an inline heredoc so the
supervisor has a real child process to wait on and, when the card wedges, to
kill.

    python3 _pytest_runner.py IDS_FILE [pytest args...]
"""

import sys

import pytest


def main(argv) -> int:
    if len(argv) < 2:
        print("usage: _pytest_runner.py IDS_FILE [pytest args...]", file=sys.stderr)
        return 4

    with open(argv[1]) as handle:
        ids = [line.rstrip("\n") for line in handle if line.strip()]
    if not ids:
        print("ttnop: nothing collected")
        return 0

    return pytest.main(
        [*ids, *argv[2:], "-p", "ttnop_plugin", "-p", "no:randomly", "-q"]
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
