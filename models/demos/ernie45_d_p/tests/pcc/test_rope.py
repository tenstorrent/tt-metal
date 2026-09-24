# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P2.5: interleaved RoPE with chunk offset (chunk 0 at pos 0, chunk 1 at pos 2048) vs golden q/k."""

import pytest
import torch.nn.functional as F

import ttnn
from models.demos.ernie45_d_p.tests.conftest import mesh_1x4
from models.demos.ernie45_d_p.tt.common import replicated_to_torch
from models.demos.ernie45_d_p.tt.ops import TtRope

TASK = "P2.5"


@mesh_1x4
@pytest.mark.parametrize("chunk", [0, 1])
def test_rope(mesh_device, cfg, layer_weights, golden_2k, record, chunk):
    layer = 0
    g = golden_2k.layer(chunk, layer)
    w = layer_weights(layer)
    x = g["attn_norm"].float()
    S, D = x.shape[0], cfg.head_dim
    start = chunk * golden_2k.chunk
    rope = TtRope(mesh_device, D, cfg.rope_theta)
    for name, W, nh in [("q", w.wq, cfg.num_attention_heads), ("k", w.wk, cfg.num_key_value_heads)]:
        pre = F.linear(x, W).view(S, nh, D).transpose(0, 1)[None]  # [1, nh, S, D]
        t = ttnn.from_torch(
            pre,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        y = replicated_to_torch(rope(t, start))[0]
        record(f"pcc_rope_{name}_c{chunk}", y, g[name], 0.999)
    record.check()
