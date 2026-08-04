# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Drift test: the dry run must keep matching the hand-written oracle.

The dry run derives its graph from `models/tt_dit` source; `examples/ltx.py`
states the same graph by hand. When they disagree, either the model changed (and
the example is stale) or the shim's shape/dist rules drifted — both are things to
look at, which is the whole point of keeping `examples/` around.

Each config runs in a **subprocess**: the dry run installs fakes for `ttnn` and
`torch` into `sys.modules`, which must not leak into any other test.

    pytest models/tt_dit/tools/dit_analyzer/spike/
    python3 models/tt_dit/tools/dit_analyzer/spike/test_dryrun_matches_oracle.py
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "run_ltx_block.py")


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, DRIVER, *args],
        capture_output=True,
        text=True,
        cwd=HERE,
        timeout=600,
    )


def test_ring_dryrun_matches_oracle():
    """BH 4x8 / Ring: 6 provable duplicate gathers, byte-identical to the oracle."""
    proc = _run()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "identical multiset" in proc.stdout, proc.stdout
    assert "'duplicate_gather': 6" in proc.stdout, proc.stdout
    assert "unregistered ops: none" in proc.stdout, proc.stdout
    assert "PASS" in proc.stdout


def test_linear_dryrun_is_clean():
    """BH 2x4 / Linear: same source, nothing redundant."""
    proc = _run("--linear")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "identical multiset" in proc.stdout, proc.stdout
    assert "findings: none" in proc.stdout, proc.stdout
    assert "PASS" in proc.stdout


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_")):
        try:
            fn()
            print("PASS %s" % name)
        except AssertionError as exc:
            failed += 1
            print("FAIL %s\n%s" % (name, str(exc)[:2000]))
    sys.exit(1 if failed else 0)
