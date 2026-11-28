"""Pre-flight: ensure `import ttnn` works before any on-device test runs.

On failure it prints a clean diagnosis plus the exact commands to fix it, then
aborts. It does NOT auto-rebuild and does NOT invoke an LLM — fixing a build is a
manual, privileged step the operator should run. Dormant when ttnn imports.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Tuple


def _check_import() -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import ttnn"],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return (proc.returncode == 0), (proc.stderr or proc.stdout or "")


def _diagnose(stderr: str) -> str:
    s = stderr.strip()
    last = s.splitlines()[-1] if s else ""
    m = re.search(r"cannot import name '([^']+)' from '(ttnn[^']*)'", last)
    if m:
        sym, mod = m.group(1), m.group(2)
        return (
            f"ttnn's compiled extension is STALE — `{mod}` does not provide "
            f"`{sym}`, but the checked-out source needs it.\n"
            f"    Cause: _ttnn.so was built from OLDER source (git carries source, "
            f"not the compiled binary), or for a different checkout/arch.\n"
            f"    → It must be rebuilt to match this source."
        )
    if "No module named 'ttnn'" in last or "No module named ttnn" in last:
        return (
            "ttnn is NOT installed/built in this Python env.\n"
            "    → tt-metal needs to be built and the venv created for this checkout."
        )
    if "_ttnn" in last and ("undefined symbol" in last or "ImportError" in last):
        return (
            "ttnn's compiled extension failed to load (ABI/build mismatch between "
            "_ttnn.so and the current source/toolchain).\n"
            "    → It must be rebuilt."
        )
    return f"`import ttnn` failed:\n    {last or '(no error text)'}\n    → likely a stale/missing ttnn build."


def ensure_ttnn_ready(**_ignored) -> bool:
    """Return True if `import ttnn` works (proceed), else print what to run and
    return False (caller aborts). Accepts and ignores extra kwargs so call sites
    don't need to change."""
    ok, err = _check_import()
    if ok:
        return True

    sep = "=" * 78
    print("\n" + sep)
    print("  ✗ PRE-FLIGHT: ttnn is not usable for this checkout — cannot run any on-device test.")
    print(sep)
    print("  " + _diagnose(err).replace("\n", "\n  "))
    print(sep)
    print("  To fix it, run from the tt-metal repo root:")
    print("      git submodule update --init --recursive")
    print("      ./build_metal.sh")
    print('      python -c "import ttnn"      # must print no error')
    print("  then re-run this command.")
    print("  (Stopping here: no component can graduate until import ttnn works.)")
    print(sep)
    return False
