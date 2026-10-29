# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.experimental.audiox.reference.dit import DiffusionTransformer
from models.experimental.audiox.tt.dit import TtDiffusionTransformer
from tests.ttnn.utils_for_testing import assert_with_pcc


PCC_THRESHOLD = 0.99


@pytest.mark.parametrize(
    "batch, seq_t, seq_cond, io_channels, embed_dim, depth, num_heads, cond_token_dim",
    [
        # Small AudioX-shaped config: cond_token_dim < embed_dim exercises the
        # kv_heads < num_heads cross-attn path, mirroring the real model.
        (1, 64, 16, 32, 256, 2, 4, 128),
    ],
)
def test_dit_pcc(device, batch, seq_t, seq_cond, io_channels, embed_dim, depth, num_heads, cond_token_dim):
    torch.manual_seed(0)

    model = DiffusionTransformer(
        io_channels=io_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        cond_token_dim=cond_token_dim,
    ).eval()

    x = torch.randn(batch, io_channels, seq_t)
    t = torch.rand(batch)
    cond = torch.randn(batch, seq_cond, cond_token_dim)

    with torch.no_grad():
        ref_out = model(x, t, cross_attn_cond=cond)

    tt_model = TtDiffusionTransformer(
        mesh_device=device,
        state_dict=model.state_dict(),
        depth=depth,
        num_heads=num_heads,
        io_channels=io_channels,
        embed_dim=embed_dim,
    )
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_t = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_cond = ttnn.from_torch(cond, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out = tt_model(tt_x, tt_t, tt_cond)

    assert_with_pcc(ref_out, ttnn.to_torch(tt_out), pcc=PCC_THRESHOLD)
