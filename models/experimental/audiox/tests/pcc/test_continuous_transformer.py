# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.experimental.audiox.reference.continuous_transformer import ContinuousTransformer
from models.experimental.audiox.tt.continuous_transformer import TtContinuousTransformer
from tests.ttnn.utils_for_testing import assert_with_pcc


PCC_THRESHOLD = 0.99


@pytest.mark.parametrize(
    "batch, seq_x, prepend_len, seq_kv, dim, dim_in, dim_out, dim_heads, depth, cross_attend",
    [
        (1, 64, 32, 128, 384, 64, 64, 64, 2, True),
    ],
)
def test_continuous_transformer_pcc(
    device, batch, seq_x, prepend_len, seq_kv, dim, dim_in, dim_out, dim_heads, depth, cross_attend
):
    torch.manual_seed(0)

    model = ContinuousTransformer(
        dim=dim,
        depth=depth,
        dim_in=dim_in,
        dim_out=dim_out,
        dim_heads=dim_heads,
        cross_attend=cross_attend,
        cond_token_dim=dim,
    ).eval()

    x = torch.randn(batch, seq_x, dim_in)
    prepend = torch.randn(batch, prepend_len, dim)
    context = torch.randn(batch, seq_kv, dim)

    with torch.no_grad():
        ref_out = model(x, prepend_embeds=prepend, context=context)

    tt_model = TtContinuousTransformer(
        mesh_device=device,
        state_dict=model.state_dict(),
        depth=depth,
        dim_heads=dim_heads,
        cross_attend=cross_attend,
        dim_in=dim_in,
        dim_out=dim_out,
    )
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_prepend = ttnn.from_torch(prepend, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_context = ttnn.from_torch(context, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = tt_model(tt_x, prepend_embeds=tt_prepend, context=tt_context)

    assert_with_pcc(ref_out, ttnn.to_torch(tt_out), pcc=PCC_THRESHOLD)
