# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Dense MLP at real dims (hidden 12288, intermediate 28672) vs ReferenceMLP, random weights in the
spec's MLP dtypes: column-parallel gate/up, row-parallel down, TP reduce-scatter into the residual."""

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.model import ReferenceMLP, random_layer_state_dict
from models.demos.mistral_medium_3_5_128b.tt.mlp import MLP

from .common import CFG, assert_pcc, randn, residual_to_torch, spec_dtypes, to_full


@pytest.mark.timeout(900)
@pytest.mark.parametrize("seq_len", [10240])
def test_dense_mlp_vs_ref(galaxy_mesh, mesh_config, ccl_manager, seq_len):
    sd = {k[len("mlp.") :]: v for k, v in random_layer_state_dict(CFG, seed=21).items() if k.startswith("mlp.")}
    ref_mlp = ReferenceMLP(CFG).to(torch.bfloat16).eval()
    ref_mlp.load_state_dict(sd)
    x = randn(1, 1, seq_len, CFG.hidden_size, seed=22)
    with torch.no_grad():
        ref = ref_mlp(x)

    dt = spec_dtypes()
    mlp = MLP(
        galaxy_mesh,
        mesh_config,
        ccl_manager,
        sd,
        gate_dtype=dt["mlp_gate"],
        up_dtype=dt["mlp_up"],
        down_dtype=dt["mlp_down"],
    )
    out = residual_to_torch(mlp(to_full(x, galaxy_mesh, mesh_config)), galaxy_mesh, mesh_config)
    assert_pcc("dense_mlp", ref, out)
