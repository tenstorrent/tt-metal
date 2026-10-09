# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run pytest over a file of node ids.

A file rather than argv because a full sweep is tens of thousands of ids and
would blow past ARG_MAX. A separate script rather than an inline heredoc so the
supervisor has a real child process to wait on and, when the card wedges, to
kill.

    python3 _pytest_runner.py IDS_FILE PYTEST_ARGS_JSON
"""

import json
import sys

import pytest


def run(ids_file, pytest_args) -> int:
    with open(ids_file) as handle:
        ids = [line.rstrip("\n") for line in handle if line.strip()]
    if not ids:
        raise ValueError("ttnop: no test node IDs to run")

    return pytest.main(
        [*ids, *pytest_args, "-p", "ttnop_plugin", "-p", "no:randomly", "-q"]
    )


def main(argv) -> int:
    _, ids_file, pytest_args_file = argv
    with open(pytest_args_file) as handle:
        pytest_args = json.load(handle)
    return run(ids_file, pytest_args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
