# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One complete decoder layer with residuals at real dims vs ReferenceDecoderLayer, random weights in
the spec's dtypes, one-shot over the full 10240-token prefill; also checks the layer's KV write."""

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.model import (
    ReferenceDecoderLayer,
    random_layer_state_dict,
    rope_cos_sin,
)
from models.demos.mistral_medium_3_5_128b.tt.kv_cache import allocate_kv_cache, naturalize, read_slot_kv
from models.demos.mistral_medium_3_5_128b.tt.layer import DecoderLayer
from models.demos.mistral_medium_3_5_128b.tt.rope import RopeSetup, hf_to_meta_perm

from .common import CFG, assert_pcc, randn, residual_to_torch, spec_dtypes, to_residual


@pytest.mark.timeout(900)
@pytest.mark.parametrize("seq_len", [10240])
def test_decoder_layer_vs_ref(galaxy_mesh, mesh_config, ccl_manager, seq_len):
    sd = random_layer_state_dict(CFG, seed=91)
    ref_layer = ReferenceDecoderLayer(CFG).to(torch.bfloat16).eval()
    ref_layer.load_state_dict(sd)
    x = randn(1, 1, seq_len, CFG.hidden_size, seed=92)
    pos = torch.arange(seq_len)
    cos, sin = rope_cos_sin(CFG, pos)
    with torch.no_grad():
        ref, ref_k, ref_v = ref_layer(x[0], cos, sin, pos)

    cache = allocate_kv_cache(
        galaxy_mesh,
        mesh_config,
        num_layers=1,
        max_seq_len=seq_len,
        num_local_kv_heads=CFG.num_key_value_heads // mesh_config.tp,
        head_dim=CFG.head_dim,
    )
    rope = RopeSetup(galaxy_mesh, mesh_config, CFG, max_seq_len=seq_len, chunk_size=seq_len)
    layer = DecoderLayer(galaxy_mesh, mesh_config, ccl_manager, CFG, sd, layer_idx=0, dtypes=spec_dtypes())
    out = residual_to_torch(
        layer(to_residual(x, galaxy_mesh, mesh_config), rope, kv_cache=cache, cached_len=0), galaxy_mesh, mesh_config
    )
    assert_pcc("decoder_layer", ref[None], out)

    k_blk, v_blk = read_slot_kv(galaxy_mesh, cache, 0)
    perm = hf_to_meta_perm(CFG.head_dim)
    assert_pcc("decoder_layer_k", ref_k[0][..., perm], naturalize(k_blk[0], seq_len, mesh_config.sp, seq_len, seq_len))
    assert_pcc("decoder_layer_v", ref_v[0], naturalize(v_blk[0], seq_len, mesh_config.sp, seq_len, seq_len))
