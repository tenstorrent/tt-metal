# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Dense SiLU-SwiGLU MLP vs the torch reference, at the model's real dims (5120 -> 17408 -> 5120).

Column-parallel gate/up, row-parallel down, TP all-reduce. The activation is plain ``silu(gate) *
up`` — the guard-rail below pins that a clamped swigluoai (the nearest donor's activation) would
not pass this test by accident.
"""

from __future__ import annotations

import torch

from models.common.utility_functions import comp_pcc
from models.demos.qwen_3_8_27b_d_p.reference.modeling import REF_DTYPE, Qwen35MLP, init_random_weights
from models.demos.qwen_3_8_27b_d_p.tt.mlp import MLP

from ..test_factory import mesh_setup, parametrize_mesh, unit_test_config
from .helpers import WEIGHT_DTYPE, assert_tp_replicated, check_pcc, from_sp_sharded, randn, to_sp_sharded


@parametrize_mesh()
def test_dense_mlp_vs_ref(mesh, submesh_shape, device_params):
    cfg = unit_test_config()
    mesh_config, ccl = mesh_setup(mesh)
    tokens = 128 * mesh_config.sp

    ref = Qwen35MLP(cfg).eval()
    init_random_weights(ref, seed=11)
    x = randn(1, 1, tokens, cfg.hidden_size, seed=12)
    with torch.no_grad():
        expected = ref(x)

    mlp = MLP(
        mesh,
        cfg,
        ref.state_dict(),
        mesh_config=mesh_config,
        ccl_manager=ccl,
        weight_dtype=WEIGHT_DTYPE,
    )
    tt_x = to_sp_sharded(x, mesh, mesh_config)
    out = mlp(tt_x)
    assert_tp_replicated(out, mesh_config, "mlp output")
    got = from_sp_sharded(out, mesh_config)
    check_pcc("dense_mlp", expected, got, shape=(1, 1, tokens, cfg.hidden_size))


def test_plain_swiglu_is_not_clamped_swigluoai():
    """Host-only guard-rail. The nearest donor MLP (MiniMax-M3) uses a clamped swigluoai with an
    alpha; porting that math across is a silent PCC loss, so pin that the two differ here."""
    torch.manual_seed(13)
    gate = torch.randn(4096, dtype=REF_DTYPE) * 4
    up = torch.randn(4096, dtype=REF_DTYPE) * 4
    plain = torch.nn.functional.silu(gate) * up
    alpha, limit = 1.702, 7.0
    g = gate.clamp(max=limit)
    u = up.clamp(min=-limit, max=limit)
    oai = g * torch.sigmoid(alpha * g) * (u + 1)
    _, pcc = comp_pcc(plain.float(), oai.float(), 0.0)
    assert float(pcc) < 0.99, f"plain SwiGLU and clamped swigluoai agree at PCC {pcc} — bad test"
