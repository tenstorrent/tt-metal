# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttKDA sequence-parallel prefill on a (rows, 4) block: the prompt split over the rows (row r = slab r of
the chunk, TP over the columns), output rows and the final carries vs HF. The first building block of the
multi-row prefill engine (TTFT).

    TT_VISIBLE_DEVICES=2,3,6,7,10,11,14,15 TT_MESH_GRAPH_DESC_PATH=.../p150_x8_mesh_graph_descriptor.textproto \
    MESH_DEVICE=P150x8 pytest models/demos/glm53_flash/tests/test_mesh_sp_kda.py -s
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tests.test_kda import LAYER, _hf_linear_attention, weights  # noqa: F401 (fixture)
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.kda import TtKDA
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
T = int(os.environ.get("GLM53_SP_T", "512"))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_sp_kda_prefill_vs_hf(mesh_device, weights):
    from transformers import DynamicCache

    snapshot, text, raw = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    hf = _hf_linear_attention(snapshot, text, raw)
    attn = {k[len("self_attn.") :]: v for k, v in raw.items()}
    mesh = GLMMesh.from_device(mesh_device)
    rows = mesh.rows
    tt = TtKDA(mesh_device, cfg, attn, layer_idx=LAYER, mesh=mesh)
    torch.manual_seed(0)
    x = (torch.randn(1, T, cfg.hidden_size) * 0.8).to(torch.bfloat16)
    cache = DynamicCache(config=text)
    with torch.no_grad():
        y_hf = hf(x, cache_params=cache, attention_mask=torch.ones(1, T, dtype=torch.bool))
    hf_layer = cache.layers[LAYER]
    # slab order: row r holds rows [r*T/rows, (r+1)*T/rows) of the chunk
    assert T % (rows * 32) == 0
    x_dev = mesh.upload(x, ttnn.bfloat16, rows=1)  # [1, T/rows, 4096] per row, replicated over the columns
    state = tt.allocate_state(1)
    # the model path: ttKDA for the global active_seq_len (SP over the rows, the grouped scan sized to the local
    # chunk count and the worker grid), the conv-history layout conversions of the block decode path and the TP
    # all-gather of the reduce-scattered output inside
    y, new_state = tt.prefill_forward(x_dev, state, tt.scalar(0), None)
    logger.info(
        f"ttKDA SP output shape per chip {list(y.shape)}, state {list(new_state.recurrent.shape)} / {list(new_state.convolution.shape)}; "
        f"summary group chunks {tt.summary_group_chunks(T // rows // 32)} for {T // rows // 32} local chunks"
    )
    out = mesh.to_host(y, rows=1).float()  # [1, T, 4096] in row (slab) order
    p = pcc(y_hf, out)
    per_row = [
        pcc(y_hf[:, r * T // rows : (r + 1) * T // rows], out[:, r * T // rows : (r + 1) * T // rows])
        for r in range(rows)
    ]
    logger.info(
        f"SP ttKDA layer {LAYER} T={T} on {MESH}: output pcc vs HF {p:.6f}, per row {['%.6f' % v for v in per_row]}"
    )
    # final carries: TP-sharded heads, replicated over the rows?
    rec_shards = [ttnn.to_torch(t) for t in mesh.device_tensors(new_state.recurrent)]
    conv_shards = [ttnn.to_torch(t) for t in mesh.device_tensors(new_state.convolution)]
    if new_state.convolution.layout == ttnn.TILE_LAYOUT:  # block path: tap-major tiles [taps, B, W] -> [B, taps, W]
        conv_shards = [c.permute(1, 0, 2).contiguous() for c in conv_shards]
    H_loc = cfg.kda_num_heads // mesh.cols
    rec = torch.cat([rec_shards[c] for c in range(mesh.cols)], dim=1)  # row 0's columns -> [1, 64, 128, 128]
    p_rec = pcc(hf_layer.recurrent_states[0], rec)
    conv_hf = hf_layer.conv_states[0][:, :, -3:].transpose(1, 2)  # [1,3,W] (q|k|v all heads)
    # per-chip conv [1,3,3*H_loc*128] in [q_c|k_c|v_c] order -> reassemble [q|k|v]
    W_loc = conv_shards[0].shape[-1] // 3
    conv = torch.cat(
        [torch.cat([conv_shards[c][..., i * W_loc : (i + 1) * W_loc] for c in range(mesh.cols)], -1) for i in range(3)],
        -1,
    )
    p_conv = pcc(conv_hf, conv)
    same_rows = all(torch.equal(rec_shards[r * mesh.cols], rec_shards[0]) for r in range(1, rows))
    logger.info(
        f"  final recurrent state pcc vs HF {p_rec:.6f} (replicated over rows: {same_rows}), conv history pcc {p_conv:.6f}"
    )
    assert p > 0.995 and p_rec > 0.995 and p_conv > 0.995
