# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf probe (not a correctness test): run q_stem on the Kimi K3.1 Phase-0 shape a few times so a --profile run
reports device kernel durations. QSTEM_ABLATE (env, read by the program descriptor) stubs one stage for ablation;
the output is then wrong by design, so nothing is checked here."""

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.q_stem.test_q_stem import _DEVICE_PARAMS, _MESH_PARAM, _make_inputs, CLUSTER_AXIS
from ttnn.operations.q_stem import q_stem


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", _MESH_PARAM, indirect=True)
def test_q_stem_perf_probe(mesh_device):
    B, S, Q, Hl, Dn, Dr, N = (1, 640, 1536, 24, 128, 64, 512)
    _, _, _, tx, tq, tw = _make_inputs(mesh_device, B, S, Q, Hl, Dn, Dr, N)
    for _ in range(4):
        out = q_stem(tx, tq, tw, cluster_axis=CLUSTER_AXIS)
    ttnn.synchronize_device(mesh_device)
