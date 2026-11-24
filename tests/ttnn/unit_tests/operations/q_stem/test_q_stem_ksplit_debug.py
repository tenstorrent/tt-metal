# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug (Refinement 7): which heads / rows of the Kimi call go wrong under S1_KSPLIT=2 at fp32 DEST. Prints the
per-head max error on device 0; keeps no assertion beyond the run. DO NOT DELETE."""

from __future__ import annotations

import pytest
import torch

from tests.ttnn.unit_tests.operations.q_stem.test_q_stem import (
    CLUSTER_AXIS,
    _DEVICE_PARAMS,
    _MESH_PARAM,
    _make_inputs,
    _read_back,
    reference,
)
import ttnn
from ttnn.operations.q_stem import q_stem
from ttnn.operations.q_stem import q_stem_program_descriptor as _pd


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", _MESH_PARAM, indirect=True)
@pytest.mark.parametrize("ks", [2])
def test_ksplit_head_errors(mesh_device, ks, monkeypatch):
    monkeypatch.setattr(_pd, "S1_KSPLIT", ks)
    monkeypatch.setattr(_pd, "S1_KPART_DEPTH", int(__import__("os").environ.get("KPD", "2")))
    B, S, Q, Hl, Dn, Dr, N = 1, 640, 1536, 24, 128, 64, 512
    G = tuple(mesh_device.shape)[CLUSTER_AXIS]
    x, q_b, wkv, tx, tq, tw = _make_inputs(mesh_device, B, S, Q, Hl, Dn, Dr, N)
    expected = reference(x, q_b, wkv, G)
    for call in range(6):
        out = q_stem(tx, tq, tw, cluster_axis=CLUSTER_AXIS)
        _report(out, expected, mesh_device, G, Hl, S, N, Dr, call)


def _report(out, expected, mesh_device, G, Hl, S, N, Dr, call):
    for idx, dev in enumerate(_read_back(out, mesh_device)):
        got = ttnn.to_torch(dev).float()
        exp = expected[0, idx].float()
        err = (got - exp).abs()
        bad_heads = [(h, round(err[0, h].max().item(), 2)) for h in range(G * Hl) if err[0, h].max() > 0.5]
        print(f"call {call} dev {idx}: bad heads {bad_heads}")
        if bad_heads:
            h = bad_heads[0][0]
            rows = [r for r in range(S // G) if err[0, h, r].max() > 0.5]
            cols_nope = [c for c in range(N) if err[0, h, :, c].max() > 0.5]
            cols_rope = [c for c in range(N, N + Dr) if err[0, h, :, c].max() > 0.5]
            print(
                f"  head {h}: bad rows {rows[:8]}..({len(rows)}) nope cols {len(cols_nope)} rope cols {cols_rope[:8]}..({len(cols_rope)})"
            )
