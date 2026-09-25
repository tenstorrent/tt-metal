# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SP-sharded ring-joint SDPA with live Q/K/V (no cache) vs unsharded causal GQA attention: gathering
K/V across the 8 SP rows by online softmax equals full attention. 24 q / 2 kv heads per chip."""

import pytest

from models.demos.mistral_medium_3_5_128b.reference.model import causal_attention
from models.demos.mistral_medium_3_5_128b.tt.attention import ring_sdpa_nocache

from .common import CFG, assert_pcc, heads_to_torch, randn, to_heads


@pytest.mark.parametrize("seq_len", [10240, 5120])
def test_ring_joint_sdpa_sp_vs_ref(galaxy_mesh, mesh_config, ccl_manager, seq_len):
    d = CFG.head_dim
    q = randn(1, CFG.num_attention_heads, seq_len, d, seed=41)
    k = randn(1, CFG.num_key_value_heads, seq_len, d, seed=42)
    v = randn(1, CFG.num_key_value_heads, seq_len, d, seed=43)
    ref = causal_attention(q, k, v)

    out = ring_sdpa_nocache(
        to_heads(q, galaxy_mesh, mesh_config),
        to_heads(k, galaxy_mesh, mesh_config),
        to_heads(v, galaxy_mesh, mesh_config),
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        logical_n=seq_len,
        n_kv=CFG.num_key_value_heads,
        head_dim=d,
        scale=d**-0.5,
    )
    assert_pcc(f"ring_joint_sdpa_nocache[{seq_len}]", ref, heads_to_torch(out, galaxy_mesh, mesh_config))
