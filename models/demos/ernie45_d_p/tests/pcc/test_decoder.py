# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P2.10: full decoder blocks (dense layer 0, MoE layers 1 and 27), chunked 2k->2k with golden inputs."""

import pytest

from models.demos.ernie45_d_p.tests.conftest import mesh_1x4
from models.demos.ernie45_d_p.tt.attention import TtKVCache
from models.demos.ernie45_d_p.tt.common import replicated_to_torch, to_mesh_activation
from models.demos.ernie45_d_p.tt.model import TtDecoderLayer
from models.demos.ernie45_d_p.tt.ops import TtRope

TASK = "P2.10"


@mesh_1x4
@pytest.mark.parametrize("layer", [0, 1, 27])
def test_decoder_block(mesh_device, cfg, layer_weights, golden_2k, record, layer):
    G = golden_2k
    rope = TtRope(mesh_device, cfg.head_dim, cfg.rope_theta)
    block = TtDecoderLayer(mesh_device, cfg, layer, layer_weights(layer), rope)
    cache = TtKVCache(mesh_device, cfg, G.seq, [layer])
    for c in range(G.seq // G.chunk):
        g = G.layer(c, layer)
        out = block(to_mesh_activation(mesh_device, g["in"].float()), c * G.chunk, cache)
        record(f"pcc_block_L{layer:02d}_c{c}", replicated_to_torch(out)[0, 0], g["out"], 0.98)
    k, v = cache.to_torch(layer, G.seq)
    gk, gv = G.kv(layer)
    record(f"pcc_kv_k_L{layer:02d}", k, gk, 0.97)
    record(f"pcc_kv_v_L{layer:02d}", v, gv, 0.97)
    record.check()
