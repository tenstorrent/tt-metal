# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One complete decoder layer (norms + mixer + MLP + residuals) vs torch, real dims, random weights —
one of each layer type. (pattern: minimax_m3/tests/unit/test_decoder_layer_vs_ref.py)"""

import pytest
import torch

import ttnn
from models.demos.qwen_3_8_27b.config import QWEN38, PrefillSpec
from models.demos.qwen_3_8_27b.reference import qwen3_8_ref as ref
from models.demos.qwen_3_8_27b.tests.common import assert_pcc, from_sp, to_sp
from models.demos.qwen_3_8_27b.tt.context import PrefillCtx
from models.demos.qwen_3_8_27b.tt.kv_cache import allocate_caches, cache_capacity
from models.demos.qwen_3_8_27b.tt.layer import TtDecoderLayer
from models.demos.qwen_3_8_27b.tt.rope import TtRope


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["gdn_layer0", "attn_layer3"])
def test_decoder_layer_vs_ref(mesh, mesh_config, ccl_manager, layer_idx):
    seq = 5120
    m = ref.init_random_(ref.DecoderLayer(QWEN38, layer_idx), seed=31 + layer_idx).to(torch.bfloat16).eval()
    x = torch.randn(1, seq, QWEN38.hidden_size, generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)
    cos, sin = ref.rope_cos_sin(QWEN38, torch.arange(seq))
    with torch.no_grad():
        want, _ = m(x, cos, sin)
    tt = TtDecoderLayer(mesh_config, ccl_manager, QWEN38, m.state_dict(), layer_idx, PrefillSpec.load())
    caches = allocate_caches(
        mesh,
        num_attn_layers=len(QWEN38.full_attention_layers),
        gdn_layers=[layer_idx] if not tt.is_full else [],
        max_seq_len=cache_capacity(seq, [seq]),
        head_dim=QWEN38.head_dim,
    )
    rope = TtRope(mesh_config, QWEN38)
    ctx = PrefillCtx(caches=caches, user_id=0, start=0, valid_end=seq, tokens=seq)
    ctx.cos, ctx.sin = rope.tables(0, seq // mesh_config.sp)
    got = from_sp(tt(to_sp(x[None], mesh_config), ctx, rope), mesh_config)[0]
    assert_pcc(f"decoder_layer_{layer_idx}", got, want.float())
    caches.reset_gdn()
    ttnn.deallocate(caches.k)
    ttnn.deallocate(caches.v)
