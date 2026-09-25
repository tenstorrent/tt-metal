# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm on the SP x TP sharded residual vs the torch reference (plain ``weight * x_normed``, no
Gemma ``1 + w`` fold). Full hidden width, full one-shot sequence; a few massive-activation channels."""

import pytest

from models.demos.mistral_medium_3_5_128b.reference.model import rms_norm
from models.demos.mistral_medium_3_5_128b.tt.rms_norm import RMSNorm

from .common import CFG, assert_pcc, full_to_torch, randn, to_residual


@pytest.mark.parametrize("seq_len", [10240])
def test_rms_norm_vs_ref(galaxy_mesh, mesh_config, ccl_manager, seq_len):
    x = randn(1, 1, seq_len, CFG.hidden_size, seed=1)
    x[..., ::997] *= 30.0
    w = 1.0 + randn(CFG.hidden_size, seed=2, scale=0.1)
    ref = rms_norm(x, w, CFG.rms_norm_eps)

    norm = RMSNorm(galaxy_mesh, mesh_config, ccl_manager, w, CFG.rms_norm_eps)
    out = full_to_torch(norm(to_residual(x, galaxy_mesh, mesh_config)), galaxy_mesh, mesh_config)
    assert_pcc("rms_norm", ref, out)
