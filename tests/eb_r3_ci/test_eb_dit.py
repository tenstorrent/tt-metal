# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58723): the DiT fused distributed norms at tensor parallelism 1 (local norm, no all-gather) on
one chip, through test_distributed_rmsnorm_fused.py's test_corr_det, whose parameters name a 4x8 mesh."""
import importlib

import pytest
import ttnn

m = importlib.import_module("models.tt_dit.tests.unit.test_distributed_rmsnorm_fused")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 131072}], indirect=True)
@pytest.mark.parametrize("model", ["wan", "ltx", "flux"])
def test_dit_tp1(mesh_device, model):
    m.test_corr_det(mesh_device, model, 1, ttnn.Topology.Linear, None, 1, False)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 131072}], indirect=True)
def test_dit_ln_tp1(mesh_device):
    m.test_layernorm_corr(mesh_device, "wan", 1, ttnn.Topology.Linear, None, 1, False)
