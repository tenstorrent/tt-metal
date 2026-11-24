# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""K-split stage 1 (regime R6, Refinement 7): the `S1_KSPLIT` knob is parked at 1 (measured slower on the PERF FOCUS),
so the golden suite never runs it. Pin its correctness here: K groups 2 and 3 on the Kimi call, a K-blocked stage 1
(nkb > 1 per group), multi-batch, non-tile-aligned rows (the injector realign with a K offset) and a 2-tile Q (ks
clamps to a divisor of Q_t). Device from the conftest `mesh_device` fixture."""

from __future__ import annotations

import pytest

from tests.ttnn.unit_tests.operations.q_stem.test_q_stem import (
    CLUSTER_AXIS,
    _DEVICE_PARAMS,
    _MESH_PARAM,
    _check,
    _make_inputs,
    reference,
)
from ttnn.operations.q_stem import q_stem
from ttnn.operations.q_stem import q_stem_program_descriptor as _pd

CASES = [
    pytest.param((1, 640, 1536, 24, 128, 64, 512), 2, id="kimi_k31_640-ks2"),
    pytest.param((1, 640, 1536, 24, 128, 64, 512), 3, id="kimi_k31_640-ks3"),
    pytest.param((1, 128, 4096, 4, 128, 64, 512), 2, id="deep_q_k_blocked-ks2"),
    pytest.param((2, 128, 512, 8, 128, 64, 512), 3, id="multi_batch-ks3"),
    pytest.param((1, 160, 512, 4, 128, 64, 512), 2, id="m_non_aligned-ks2"),
    pytest.param((1, 128, 64, 2, 32, 32, 32), 3, id="single_tile-ks3_clamped"),
]


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", _MESH_PARAM, indirect=True)
@pytest.mark.parametrize("dims,ks", CASES)
def test_q_stem_ksplit(mesh_device, dims, ks, monkeypatch):
    monkeypatch.setattr(_pd, "S1_KSPLIT", ks)
    B, S, Q, Hl, Dn, Dr, N = dims
    G = tuple(mesh_device.shape)[CLUSTER_AXIS]
    x, q_b, wkv, tx, tq, tw = _make_inputs(mesh_device, B, S, Q, Hl, Dn, Dr, N)
    expected = reference(x, q_b, wkv, G)
    out = q_stem(tx, tq, tw, cluster_axis=CLUSTER_AXIS)
    _check(out, expected, mesh_device, (B, G * Hl, S // G, N + Dr))
