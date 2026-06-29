# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Device denoise-conditioning helpers: wte embed, step scatter, TT mask upload.
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_denoise_cond_tt.py -v -s

from __future__ import annotations

import torch
import ttnn

from models.experimental.hunyuan_image_3_0.ref.image_gen.timestep_embedder import TimestepEmbedder
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import scatter_distill_step_embeds
from models.experimental.hunyuan_image_3_0.tt.denoise_cond import (
    make_wte_embed_fn,
    scatter_step_embeds_tt,
    upload_denoise_cond,
)
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def test_scatter_step_embeds_tt_matches_ref(device):
    hidden = 64
    bsz, seq = 1, 8
    ref_emb = TimestepEmbedder(hidden_size=hidden)
    sd = {f"timestep_emb.{k}": v for k, v in ref_emb.state_dict().items()}
    tt_emb = HunyuanTtTimestepEmbedder(device, hidden, sd, "timestep_emb")

    base_host = torch.randn(bsz, seq, hidden)
    idx_t = torch.tensor([[2]], dtype=torch.long)
    idx_g = torch.tensor([[3]], dtype=torch.long)
    idx_r = torch.tensor([[4]], dtype=torch.long)

    ref_out = scatter_distill_step_embeds(
        base_host.clone(),
        t_scalar=100.0,
        gen_timestep_scatter_index=idx_t,
        timestep_emb=ref_emb,
        guidance_scalar=2500.0,
        guidance_scatter_index=idx_g,
        guidance_emb=ref_emb,
        t_r_scalar=50.0,
        gen_timestep_r_scatter_index=idx_r,
        timestep_r_emb=ref_emb,
    )

    base_tt = ttnn.from_torch(
        base_host.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    out_tt = scatter_step_embeds_tt(
        base_tt,
        t_scalar=100.0,
        gen_timestep_scatter_index=idx_t,
        tt_timestep_emb=tt_emb,
        guidance_scalar=2500.0,
        guidance_scatter_index=idx_g,
        tt_guidance_emb=tt_emb,
        t_r_scalar=50.0,
        gen_timestep_r_scatter_index=idx_r,
        tt_timestep_r_emb=tt_emb,
    )
    out_host = ttnn.to_torch(out_tt).float()
    tt_emb.deallocate()

    score = _pcc(ref_out, out_host)
    print(f"scatter_step_embeds_tt PCC={score:.6f}")
    assert score >= 0.99, f"device scatter PCC {score:.6f} < 0.99"
    ttnn.deallocate(out_tt)


def test_wte_embed_tt_matches_torch(device):
    vocab, hidden = 128, 64
    wte = torch.randn(vocab, hidden)
    ids = torch.tensor([[3, 7, 11, 15, 19, 23, 27, 31]])
    ref = torch.nn.functional.embedding(ids, wte)

    embed_fn = make_wte_embed_fn(device, wte)
    out_tt = embed_fn(ids)
    out_host = ttnn.to_torch(out_tt).float()
    ttnn.deallocate(out_tt)
    ttnn.deallocate(embed_fn.weight)

    score = _pcc(ref, out_host)
    print(f"wte embed PCC={score:.6f}")
    assert score >= 0.99, f"wte embed PCC {score:.6f} < 0.99"


def test_upload_denoise_cond_persistent_base(device):
    from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask
    from models.experimental.hunyuan_image_3_0.tt.attention.mask import _NEG

    bsz, seq, hidden = 1, 32, 64
    spans = [slice(4, 20)]
    base = torch.randn(bsz, seq, hidden)
    cond = dict(
        base_embeds_host=base,
        batch=1,
        gen_timestep_scatter_index=torch.tensor([[2]], dtype=torch.long),
    )
    uploaded = upload_denoise_cond(device, cond, seq_len=seq, attn_spans=spans)
    assert uploaded.get("base_embeds_persistent") is True
    assert uploaded.get("base_embeds") is not None

    ref_bool = build_attention_mask(seq, [spans], bsz=1)
    ref_mask = torch.where(
        ref_bool,
        torch.zeros((), dtype=torch.float32),
        torch.full((), _NEG, dtype=torch.float32),
    ).to(torch.bfloat16)
    tt_mask = uploaded["attention_mask"]
    tt_host = ttnn.to_torch(tt_mask)[..., :seq, :seq].to(torch.bfloat16)
    assert torch.equal(tt_host, ref_mask)
    ttnn.deallocate(uploaded["base_embeds"])
    ttnn.deallocate(tt_mask)
