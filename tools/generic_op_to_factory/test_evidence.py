# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reject warmup/crash artifacts and verify observed golden-suite routing."""

import json
import hashlib
import re
from pathlib import Path

from tools.generic_op_to_factory.export_run import ExportError

# Reviewed runtime-tool versions. Keep these pins with flow releases; comparing
# a worktree file with itself would not detect an unreviewed local runner edit.
RUNNER_SHA256 = "72a490dacafc12c92bc83222415906e583721eda87475020bcd60e22e146900d"
ADAPTER_SHA256 = "7a77fc39f28b2d319755ed95c1751950902108059ce7643160d7469e71f657b4"


def check_runner(path, *, precompile, require_raw=True):
    path = Path(path)
    if not path.is_file() or path.resolve() != path:
        raise ExportError("Safe runner must be a regular, unredirected file")
    source = path.read_text()
    if require_raw and "SAFE_PYTEST_RAW_EXIT_CODE=" not in source:
        raise ExportError("Target safe runner must report the underlying pytest exit code; update it before validation")
    if precompile and 'pytest "${PYTEST_ARGS[@]}" --junitxml="${clog%.log}.junit.xml"' not in source:
        raise ExportError(
            "Precompile requires separate warmup JUnit routing; update the safe runner or disable precompile"
        )
    if require_raw and hashlib.sha256(path.read_bytes()).hexdigest() != RUNNER_SHA256:
        raise ExportError("Target safe runner must match this flow's reviewed version before validation")


def verify(log, junit, *, route=None, require_raw=True):
    text = Path(log).read_text()
    statuses = re.findall(r"^SAFE_PYTEST_RAW_EXIT_CODE=(\d+)\s*$", text, re.MULTILINE)
    if not statuses and not require_raw:
        # Pinned historical runners predate the dedicated raw-status marker.
        # Accept only their explicit normal result protocol, never wrapper exit 1 alone.
        statuses = re.findall(
            r"^SAFE_PYTEST_RESULT: FAIL \(pytest exit code: (\d+); wrapper exit: 1\)\s*$", text, re.MULTILINE
        )
        statuses += ["0"] * len(re.findall(r"^SAFE_PYTEST_RESULT: PASS\s*$", text, re.MULTILINE))
    if len(statuses) != 1:
        raise ExportError("Missing or ambiguous underlying pytest exit status")
    if int(statuses[0]) not in (0, 1):
        raise ExportError("Pytest crashed or did not complete normally; JUnit is not accepted")
    if "SAFE_PYTEST_RESULT: HANG" in text:
        raise ExportError("Device hang prevents validation")
    junit = Path(junit)
    if not junit.is_file() or junit.resolve() != junit:
        raise ExportError("Real execution did not produce a regular JUnit report")
    observed = {"pytest_exit_code": int(statuses[0])}
    if route is not None:
        messages = re.findall(r"^MIGRATION_ROUTE=(.+)$", text, re.MULTILINE)
        if len(messages) != 1:
            raise ExportError("Missing or ambiguous observed migration route")
        result = json.loads(messages[0])
        if any(result.get(key) != value for key, value in route.items()):
            raise ExportError("Observed migration route differs from the planned route")
        if type(result.get("calls")) is not int or result["calls"] <= 0:
            raise ExportError("Golden suite did not call the selected operation")
        observed["route"] = result
    return observed
