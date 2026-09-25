# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A 2-chunk sequence through the SAME ``Attention`` module two ways — one-shot over 2C tokens, and
chunk 0 then chunk 1 reading chunk 0 back from the KV cache — and the second chunk's output must
match (and match the reference). Proves the cache-read path is wired, not just callable."""

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.model import ReferenceAttention, rope_cos_sin
from models.demos.mistral_medium_3_5_128b.tt.attention import Attention
from models.demos.mistral_medium_3_5_128b.tt.kv_cache import allocate_kv_cache
from models.demos.mistral_medium_3_5_128b.tt.rope import RopeSetup

from .common import CFG, assert_pcc, randn, residual_to_torch, spec_dtypes, to_full
from .test_attention_vs_ref import attention_weights


@pytest.mark.timeout(900)
@pytest.mark.parametrize("chunk", [5120])
def test_attention_two_chunks_match_one_shot(galaxy_mesh, mesh_config, ccl_manager, chunk):
    total = 2 * chunk
    sd = attention_weights(81)
    ref_attn = ReferenceAttention(CFG).to(torch.bfloat16).eval()
    ref_attn.load_state_dict(sd)
    x = randn(1, 1, total, CFG.hidden_size, seed=82)
    pos = torch.arange(total)
    cos, sin = rope_cos_sin(CFG, pos)
    with torch.no_grad():
        ref, _, _ = ref_attn(x[0], cos, sin, pos)
    ref_chunk1 = ref[None][:, :, chunk:]

    attn = Attention(galaxy_mesh, mesh_config, ccl_manager, CFG, sd, weight_dtype=spec_dtypes()["attention"])
    n_local_kv = CFG.num_key_value_heads // mesh_config.tp

    def cache():
        return allocate_kv_cache(
            galaxy_mesh,
            mesh_config,
            num_layers=1,
            max_seq_len=total,
            num_local_kv_heads=n_local_kv,
            head_dim=CFG.head_dim,
        )

    # One-shot: a single 2C chunk.
    rope_1 = RopeSetup(galaxy_mesh, mesh_config, CFG, max_seq_len=total, chunk_size=total)
    one = residual_to_torch(
        attn(to_full(x, galaxy_mesh, mesh_config), rope_1, kv_cache=cache(), cached_len=0), galaxy_mesh, mesh_config
    )[:, :, chunk:]

    # Chunked: chunk 0 fills the cache, chunk 1 attends it.
    rope_c = RopeSetup(galaxy_mesh, mesh_config, CFG, max_seq_len=total, chunk_size=chunk)
    kv = cache()
    attn(to_full(x[:, :, :chunk], galaxy_mesh, mesh_config), rope_c, kv_cache=kv, cached_len=0)
    two = residual_to_torch(
        attn(to_full(x[:, :, chunk:], galaxy_mesh, mesh_config), rope_c, kv_cache=kv, cached_len=chunk),
        galaxy_mesh,
        mesh_config,
    )

    assert_pcc("chunk1_vs_reference", ref_chunk1, two)
    assert_pcc("chunk1_one_shot_vs_reference", ref_chunk1, one)
    assert_pcc("chunk1_chunked_vs_one_shot", one, two)
