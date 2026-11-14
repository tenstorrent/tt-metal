# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pass/fail for the check_*.py scripts: finite values, then the shared `comp_allclose_and_pcc`."""

from __future__ import annotations

import os
import sys

import numpy as np


def _comp_allclose_and_pcc():
    root = os.environ.get("TT_METAL_HOME")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        from tt_py_test_utils_common.utility_functions import comp_allclose_and_pcc
    except ImportError as e:
        raise SystemExit(f"{e}; set TT_METAL_HOME or PYTHONPATH=$TT_METAL_HOME") from e
    return comp_allclose_and_pcc


class Checks:
    """Records named comparisons; `assert_all()` fails if any did.

    atol is `atol_frac * max|ref|`, so one setting fits tensors of any scale."""

    def __init__(self) -> None:
        self._compare = _comp_allclose_and_pcc()
        self.failed: list[str] = []

    def __call__(self, name: str, got, ref, *, pcc: float = 0.99, rtol: float = 0.05, atol_frac: float = 0.02) -> str:
        import torch

        got, ref = np.ascontiguousarray(got, np.float64), np.ascontiguousarray(ref, np.float64)
        if got.shape != ref.shape:
            ok, detail = False, f"shape {got.shape} vs reference {ref.shape}"
        elif not (np.isfinite(got).all() and np.isfinite(ref).all()):
            ok, detail = False, "non-finite values"  # the shared helper lets matching NaNs pass
        else:
            atol = atol_frac * float(np.abs(ref).max(initial=0.0))
            ok, detail = self._compare(torch.from_numpy(ref), torch.from_numpy(got), rtol=rtol, atol=atol, pcc=pcc)
            ok = bool(ok)
        if not ok:
            self.failed.append(name)
        return "PASS" if ok else f"FAIL ({detail})"

    def expect(self, name: str, ok: bool, detail: str = "") -> str:
        """A non-numeric condition."""
        if not ok:
            self.failed.append(name)
        return "PASS" if ok else f"FAIL {detail}".rstrip()

    def assert_all(self) -> None:
        assert not self.failed, f"{len(self.failed)} check(s) failed: {', '.join(self.failed)}"
        print("all checks passed", flush=True)
