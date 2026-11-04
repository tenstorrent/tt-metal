# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.experimental.audiox.reference.rotary import RotaryEmbedding, apply_rotary_pos_emb as ref_apply
from models.experimental.audiox.tt.rotary import apply_rotary_pos_emb as tt_apply, precompute_rotary_cos_sin
from tests.ttnn.utils_for_testing import assert_with_pcc


PCC_THRESHOLD = 0.99


@pytest.mark.parametrize(
    "batch, heads, seq, head_dim, rot_dim",
    [
        (1, 24, 64, 64, 32),  # AudioX DiT spec: dim_heads=64, rot_dim=max(dim_heads//2, 32)=32
        (2, 8, 128, 64, 64),  # full-rotary case
    ],
)
def test_rotary_pcc(device, batch, heads, seq, head_dim, rot_dim):
    torch.manual_seed(0)

    rotary = RotaryEmbedding(rot_dim).eval()
    q = torch.randn(batch, heads, seq, head_dim)

    with torch.no_grad():
        freqs, _ = rotary.forward_from_seq_len(seq)
        ref_out = ref_apply(q, freqs)

    cos, sin = precompute_rotary_cos_sin(seq, rot_dim, mesh_device=device)
    tt_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = tt_apply(tt_q, cos, sin)

    assert_with_pcc(ref_out, ttnn.to_torch(tt_out), pcc=PCC_THRESHOLD)
