# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.7C diagnostic (not gated): the glued oracle (tests/packed/test_glued_oracle.py) with one part of
PACKED_GLUED_DYNAMIC=on dropped at a time (tt/page_copy.DYN_PARTS), to find the part that changes a result.

GLUED_DYN_DROPS: ';'-separated sets of ','-separated parts (default: each part alone); DS41_PACKED_LAYERS as the
oracle's.

    <the gate's glued env> DS41_PACKED_LAYERS=0,1,2 GLUED_DYN_DROPS="snap;ranges;rows" \\
        scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_glued_dyn_bisect.py
"""

from __future__ import annotations

import importlib
import os

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
BATCH_SHARED = True
BATCH_ITEM_VARS = ("DS41_PACKED_LAYERS", "GLUED_DYN_DROPS")


@mesh_parametrize(timeout_s=4 * 3600)
def test_glued_dyn_bisect(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tests.packed import test_glued_oracle as O
    from models.demos.deepseek_v41_flash_d_p.tt import page_copy

    O = importlib.import_module(O.__name__)
    every = sorted(page_copy.DYN_PARTS)
    v = os.environ.get("GLUED_DYN_DROPS", "").strip()
    sets = [set(x.split(",")) for x in v.split(";") if x] if v else [{p} for p in every]
    monkeypatch.setattr(metrics, "record", lambda *a, **k: None)  # (the oracle's metrics are not this test's)
    out = []
    for drop in sets:
        keep = set(page_copy.DYN_PARTS)
        page_copy.DYN_PARTS.clear()
        page_copy.DYN_PARTS.update(set(every) - drop)
        try:
            O.test_glued_oracle(mesh_device)
            res = "bit-identical"
        except AssertionError as e:
            res = f"FAILS: {str(e)[:200]}"
        finally:
            page_copy.DYN_PARTS.clear()
            page_copy.DYN_PARTS.update(keep)
        out.append((sorted(drop), res))
        print(f"BISECT drop {sorted(drop)}: {res}", flush=True)
    for d, r in out:
        print(f"BISECT RESULT drop {d}: {r}")
