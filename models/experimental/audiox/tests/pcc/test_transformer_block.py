# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.experimental.audiox.reference.rotary import RotaryEmbedding
from models.experimental.audiox.reference.transformer_block import TransformerBlock
from models.experimental.audiox.tt.rotary import precompute_rotary_cos_sin
from models.experimental.audiox.tt.transformer_block import TtTransformerBlock
from tests.ttnn.utils_for_testing import assert_with_pcc


PCC_THRESHOLD = 0.99


@pytest.mark.parametrize(
    "batch, seq_q, seq_kv, dim, dim_context, dim_heads, cross_attend",
    [
        (1, 64, 128, 384, 384, 64, True),  # dim_context == dim (kv_heads == num_heads)
        (1, 64, 128, 384, 192, 64, True),  # dim_context < dim (kv_heads < num_heads, AudioX case)
    ],
)
def test_transformer_block_pcc(device, batch, seq_q, seq_kv, dim, dim_context, dim_heads, cross_attend):
    torch.manual_seed(0)

    rot_dim = max(dim_heads // 2, 32)
    block = TransformerBlock(dim, dim_heads=dim_heads, cross_attend=cross_attend, dim_context=dim_context).eval()
    x = torch.randn(batch, seq_q, dim)
    context = torch.randn(batch, seq_kv, dim_context) if cross_attend else None

    rotary = RotaryEmbedding(rot_dim).eval()
    with torch.no_grad():
        freqs_pair = rotary.forward_from_seq_len(seq_q)
        ref_out = block(x, context=context, rotary_pos_emb=freqs_pair)

    tt_block = TtTransformerBlock(
        mesh_device=device,
        state_dict=block.state_dict(),
        dim_heads=dim_heads,
        cross_attend=cross_attend,
    )
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_ctx = (
        ttnn.from_torch(context, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) if cross_attend else None
    )
    cos, sin = precompute_rotary_cos_sin(seq_q, rot_dim, mesh_device=device)
    tt_out = tt_block(tt_x, context=tt_ctx, rotary_cos=cos, rotary_sin=sin)

    assert_with_pcc(ref_out, ttnn.to_torch(tt_out), pcc=PCC_THRESHOLD)
