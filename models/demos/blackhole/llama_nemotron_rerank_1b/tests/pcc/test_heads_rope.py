# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fused head-split + RoPE op vs nlp_create_qkv_heads + rotary_embedding_llama."""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops.fused_heads_rope import fused_heads_rope
from models.tt_transformers.tt.common import gather_cos_sin, get_rot_transformation_mat, precompute_freqs


@pytest.mark.parametrize("in_dtype, out_dtype", [("bfloat16", "bfloat16")])
@pytest.mark.parametrize("batch, seq_len", [(1, 128), (1, 512), (4, 256)])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_fused_heads_rope(device, batch, seq_len, in_dtype, out_dtype):
    nh, nkv, d = 32, 8, 64
    torch.manual_seed(0)
    qkv_t = torch.randn(1, 1, batch * seq_len, (nh + 2 * nkv) * d)
    cos_t, sin_t = precompute_freqs(d, seq_len * 2, 500000.0, 32.0, 8192)
    cos_t, sin_t = gather_cos_sin(torch.arange(seq_len), cos_t, sin_t)
    tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    qkv, cos, sin, trans = tt(qkv_t), tt(cos_t), tt(sin_t), tt(get_rot_transformation_mat(d))
    qkv_in = ttnn.typecast(qkv, getattr(ttnn, in_dtype)) if in_dtype != "bfloat16" else qkv

    q, k, v = fused_heads_rope(
        qkv_in, cos, sin, trans, n_heads=nh, n_kv_heads=nkv, batch=batch, out_dtype=getattr(ttnn, out_dtype)
    )

    ref_in = ttnn.reshape(qkv, [batch, 1, seq_len, qkv.shape[-1]])
    rq, rk, rv = ttnn.experimental.nlp_create_qkv_heads(ref_in, num_heads=nh, num_kv_heads=nkv, transpose_k_heads=False)
    rq = ttnn.experimental.rotary_embedding_llama(rq, cos, sin, trans, is_decode_mode=False)
    rk = ttnn.experimental.rotary_embedding_llama(rk, cos, sin, trans, is_decode_mode=False)
    for name, got, ref in (("q", q, rq), ("k", k, rk), ("v", v, rv)):
        g, r = ttnn.to_torch(got).float(), ttnn.to_torch(ref).float()
        assert g.shape == r.shape, (name, g.shape, r.shape)
        ok, pcc = comp_pcc(r, g, 0.9999 if in_dtype == out_dtype == "bfloat16" else 0.999)
        print(f"{name}: pcc {pcc} max|diff| {(g - r).abs().max().item():.4g}")
        assert ok, (name, pcc)
