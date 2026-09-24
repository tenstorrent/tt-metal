# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P2.6: TP4 attention, chunked 2k->2k: chunk 0 (causal SDPA) then chunk 1 (chunked SDPA over the
cache prefix). Inputs are golden attn_norm outputs; checks attention output per chunk and the final KV cache."""

import pytest

from models.demos.ernie45_d_p.tests.conftest import mesh_1x4
from models.demos.ernie45_d_p.tt.attention import TtAttention, TtKVCache
from models.demos.ernie45_d_p.tt.common import replicated_to_torch, to_mesh_activation
from models.demos.ernie45_d_p.tt.ops import TtRope

TASK = "P2.6"


@mesh_1x4
@pytest.mark.parametrize("layer", [0, 13, 27])
def test_attention_chunked(mesh_device, cfg, layer_weights, golden_2k, record, layer):
    G = golden_2k
    rope = TtRope(mesh_device, cfg.head_dim, cfg.rope_theta)
    attn = TtAttention(mesh_device, cfg, layer, layer_weights(layer), rope)
    cache = TtKVCache(mesh_device, cfg, G.seq, [layer])
    n_chunks = G.seq // G.chunk
    for c in range(n_chunks):
        g = G.layer(c, layer)
        out = attn(to_mesh_activation(mesh_device, g["attn_norm"].float()), c * G.chunk, cache)
        record(f"pcc_attn_out_L{layer:02d}_c{c}", replicated_to_torch(out)[0, 0], g["attn_out"], 0.99)
    k, v = cache.to_torch(layer, G.seq)
    gk, gv = G.kv(layer)
    record(f"pcc_kv_k_L{layer:02d}", k, gk, 0.99)
    record(f"pcc_kv_v_L{layer:02d}", v, gv, 0.99)
    record.check()
