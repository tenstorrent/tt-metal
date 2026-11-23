# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf probe (Refinement 2): run the focus case a few times WITHOUT checking the output, so a perf ablation define
(AGRN_ABL_*, wrong output by construction) can be profiled with AGRN_PROFILE_ZONES=1 + --profile and read with
zone_timeline.py. Correctness lives in test_all_gather_rms_norm.py; this file only times."""

import os

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.all_gather_rms_norm.test_all_gather_rms_norm import (
    DEVICE_PARAMS,
    _make_inputs,
    _make_weight,
    _skip_if_unrunnable,
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    gathered_shape,
)
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4")], indirect=True)
def test_perf_probe_focus(mesh_device):
    _skip_if_unrunnable(mesh_device, 1, ttnn.Topology.Ring)
    shape = (1, 1, int(os.environ.get("AGRN_PROBE_H", 640)), int(os.environ.get("AGRN_PROBE_W", 384)))
    G = tuple(mesh_device.shape)[1]
    x = _make_inputs(mesh_device, shape, ttnn.bfloat16, layer_norm=False)
    g = _make_weight(gathered_shape(shape, G)[-1], ttnn.bfloat16, seed=43)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tg = create_ttnn_gamma(g, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    for _ in range(int(os.environ.get("AGRN_PROBE_CALLS", 3))):
        all_gather_rms_norm(tx, cluster_axis=1, epsilon=1e-6, topology=ttnn.Topology.Ring, num_links=2, gamma=tg)
        ttnn.synchronize_device(mesh_device)
