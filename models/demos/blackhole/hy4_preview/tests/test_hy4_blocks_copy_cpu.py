# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the run coalescing / validation of tt/hy4_blocks_copy.py (the device part is tests/b45b_copy_unit.py)."""
import importlib.util
import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# hy4_blocks_copy imports hy4_xrow through the package path; load both standalone with a stub package
pkg = "models.demos.blackhole.hy4_preview.tt"
for p in ("models", "models.demos", "models.demos.blackhole", "models.demos.blackhole.hy4_preview", pkg):
    sys.modules.setdefault(p, types.ModuleType(p))
for n in ("hy4_xrow", "hy4_blocks_copy"):
    s = importlib.util.spec_from_file_location(f"{pkg}.{n}", os.path.join(_HERE, f"../tt/{n}.py"))
    m = importlib.util.module_from_spec(s)
    sys.modules[f"{pkg}.{n}"] = m
    setattr(sys.modules[pkg], n, m)
    s.loader.exec_module(m)
BC = sys.modules[f"{pkg}.hy4_blocks_copy"]


def test_runs():
    assert BC.runs_of([(0, 100), (1, 101), (2, 102), (7, 103), (8, 104)]) == [(0, 100, 3), (7, 103, 2)]
    assert BC.runs_of([]) == []


def test_disjoint_checks():
    BC.check_disjoint([(0, 10), (1, 11)])
    for bad in ([(0, 10), (1, 10)], [(0, 10), (10, 11)]):
        try:
            BC.check_disjoint(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_ref_copy_matches_runs():
    pool = np.random.default_rng(0).integers(0, 255, size=(64 * 20, 8), dtype=np.uint8)
    pairs = [(0, 12), (1, 13), (5, 15)]
    out = BC.ref_copy(pool, pairs, 64)
    for s, d in pairs:
        assert (out[d * 64 : (d + 1) * 64] == pool[s * 64 : (s + 1) * 64]).all()
    assert (out[:64] == pool[:64]).all()
