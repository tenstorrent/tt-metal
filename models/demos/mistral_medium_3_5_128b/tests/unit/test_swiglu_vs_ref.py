# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The MLP activation at the model's exact variant: Ministral3 ``hidden_act="silu"`` gating,
``silu(gate) * up`` (no clamp, no alpha), at the per-chip intermediate shard of a one-shot chunk."""

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.tt.mlp import silu_gate

from .common import CFG, assert_pcc, randn, residual_to_torch, to_residual


@pytest.mark.parametrize("seq_len", [10240])
def test_silu_gate_vs_ref(galaxy_mesh, mesh_config, seq_len):
    assert CFG.hidden_act == "silu"
    gate = randn(1, 1, seq_len, CFG.intermediate_size, seed=3, scale=3.0)
    up = randn(1, 1, seq_len, CFG.intermediate_size, seed=4)
    ref = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)

    tt_gate = to_residual(gate, galaxy_mesh, mesh_config)
    tt_up = to_residual(up, galaxy_mesh, mesh_config)
    out = residual_to_torch(silu_gate(tt_gate, tt_up), galaxy_mesh, mesh_config)
    assert_pcc("silu_gate", ref, out)
