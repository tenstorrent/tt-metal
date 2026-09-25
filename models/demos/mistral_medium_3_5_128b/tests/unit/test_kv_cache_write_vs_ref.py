# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cache contents after a write through the production seam (``Attention`` with a KV cache), read back
and PCC'd against the reference's post-RoPE K (in the device's Meta layout) and raw V: the write landed
at the right slot (user 1, layer 2 of 3) and offset, and every other slot is untouched."""

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.model import ReferenceAttention, rope_cos_sin
from models.demos.mistral_medium_3_5_128b.tt.attention import Attention
from models.demos.mistral_medium_3_5_128b.tt.kv_cache import allocate_kv_cache, naturalize, read_slot_kv
from models.demos.mistral_medium_3_5_128b.tt.rope import RopeSetup, hf_to_meta_perm

from .common import CFG, assert_pcc, randn, spec_dtypes, to_full
from .test_attention_vs_ref import attention_weights


@pytest.mark.timeout(900)
@pytest.mark.parametrize("seq_len", [10240])
def test_kv_cache_write_through_attention(galaxy_mesh, mesh_config, ccl_manager, seq_len):
    sd = attention_weights(71)
    ref_attn = ReferenceAttention(CFG).to(torch.bfloat16).eval()
    ref_attn.load_state_dict(sd)
    x = randn(1, 1, seq_len, CFG.hidden_size, seed=72)
    pos = torch.arange(seq_len)
    cos, sin = rope_cos_sin(CFG, pos)
    with torch.no_grad():
        _, ref_k, ref_v = ref_attn(x[0], cos, sin, pos)

    num_layers, num_users, user_id, layer_idx = 3, 2, 1, 2
    cache = allocate_kv_cache(
        galaxy_mesh,
        mesh_config,
        num_layers=num_layers,
        max_seq_len=seq_len,
        num_users=num_users,
        num_local_kv_heads=CFG.num_key_value_heads // mesh_config.tp,
        head_dim=CFG.head_dim,
    )
    rope = RopeSetup(galaxy_mesh, mesh_config, CFG, max_seq_len=seq_len, chunk_size=seq_len)
    attn = Attention(
        galaxy_mesh,
        mesh_config,
        ccl_manager,
        CFG,
        sd,
        layer_idx=layer_idx,
        weight_dtype=spec_dtypes()["attention"],
    )
    attn(to_full(x, galaxy_mesh, mesh_config), rope, kv_cache=cache, user_id=user_id, cached_len=0)

    perm = hf_to_meta_perm(CFG.head_dim)
    for u in range(num_users):
        k_blk, v_blk = read_slot_kv(galaxy_mesh, cache, u)
        for layer in range(num_layers):
            got_k = naturalize(k_blk[layer], seq_len, mesh_config.sp, seq_len, seq_len)
            got_v = naturalize(v_blk[layer], seq_len, mesh_config.sp, seq_len, seq_len)
            if (u, layer) == (user_id, layer_idx):
                assert_pcc("cache_write_k", ref_k[0][..., perm], got_k)
                assert_pcc("cache_write_v", ref_v[0], got_v)
            else:
                assert torch.count_nonzero(got_k) == 0 and torch.count_nonzero(got_v) == 0, (u, layer)
