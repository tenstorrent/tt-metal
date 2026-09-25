# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The model's final norm: the norm test applied to the tail instance ``Model.norm`` (built from the
weight source's ``norm.weight``), not a decoder layer's."""

import pytest

from models.demos.mistral_medium_3_5_128b.reference.model import rms_norm
from models.demos.mistral_medium_3_5_128b.tt.model import Model
from models.demos.mistral_medium_3_5_128b.tt.weights import StateDictWeights

from .common import CFG, assert_pcc, full_to_torch, randn, spec_dtypes, to_residual


@pytest.mark.parametrize("seq_len", [10240])
def test_final_norm_vs_ref(galaxy_mesh, mesh_config, ccl_manager, seq_len):
    w = 1.0 + randn(CFG.hidden_size, seed=121, scale=0.1)
    x = randn(1, 1, seq_len, CFG.hidden_size, seed=122, scale=4.0)
    model = Model(
        galaxy_mesh,
        mesh_config,
        ccl_manager,
        CFG,
        StateDictWeights({"norm.weight": w}),
        dtypes=spec_dtypes(),
        num_layers=0,
        with_embedding=False,
        with_lm_head=False,
    )
    out = full_to_torch(model.norm(to_residual(x, galaxy_mesh, mesh_config)), galaxy_mesh, mesh_config)
    assert_pcc("final_norm", rms_norm(x, w, CFG.rms_norm_eps), out)
