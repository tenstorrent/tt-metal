# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Keep this suite's temp files out of the shared system temp directory.

WHY THIS EXISTS. Several tests drive the real gate, and the gate creates its log directory with
`tempfile.mkdtemp(prefix="e2e_gate_")` -- so running the suite scatters `e2e_gate_*` directories
through the system temp dir, holding FIXTURE text: "RuntimeError: boom", "NOC0 is hung on PCIe
device ID 9", "[e2e] denoise step 37/50 done".

That is the same prefix a LIVE bring-up uses, and a live run's agent inspects those directories to
work out why its own gate failed. On 2026-09-28 it read this suite's fixtures twice and concluded
its gate had been SIGKILLed mid-run -- once at "denoise step 37/50", once at 43/50 -- and reported
hours of device time lost that had not been lost. Test data that is indistinguishable from
production evidence does not just litter; it gets believed.

`tempfile.tempdir` is redirected rather than the callers changed, so this holds for every test in
the suite including ones written later, and for any tool code they call. `gettempdir()` reads the
same global, so a test that globs the temp dir for its own leaks still sees exactly what it made.
"""

from __future__ import annotations

import tempfile

import pytest


@pytest.fixture(autouse=True)
def _tempdir_is_not_shared(tmp_path_factory, monkeypatch):
    """Point tempfile at a per-test directory pytest will clean up."""
    private = tmp_path_factory.mktemp("systmp")
    monkeypatch.setattr(tempfile, "tempdir", str(private))
    monkeypatch.setenv("TMPDIR", str(private))
    yield private
