# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PLAN v1.21 / D3 item 6: ttnn's "active trace" allocation warning fails the test that triggers it.

tt-metal logs it at most once per process, so the test whose allocation first happens under a live trace is the
one that fails; the job fails with it. The captured output is re-emitted so `-s` logs keep their lines."""
import sys

import pytest

ACTIVE_TRACE_WARNING = "potentially unsafe due to the existence of an active trace"


@pytest.fixture(autouse=True)
def fail_on_active_trace_allocation(capfd):
    yield
    out, err = capfd.readouterr()
    with capfd.disabled():
        sys.stdout.write(out)
        sys.stderr.write(err)
    assert ACTIVE_TRACE_WARNING not in out + err, "a device buffer was allocated while a trace was alive (v1.21)"
