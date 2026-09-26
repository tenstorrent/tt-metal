# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.2 MTP module CPU reference.

The fused input projection is the only new math here; the decoder layer it feeds is
``reference.glm_5_1.block`` unchanged. Concat order is embedding first, then hidden state.
"""

from __future__ import annotations

import torch

from models.demos.deepseek_v3_d_p.reference.glm_5_1.block import glm_decoder_layer_reference, rms_norm


def fused_mtp_reference(
    embed: torch.Tensor,
    hidden: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eh_proj_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """The MTP input projection: ``eh_proj(cat[enorm(embed), hnorm(hidden)])``.

    ``hidden`` is the previous level's state at the same position, taken after ``model.norm`` for
    level 1.
    """
    seq = embed.shape[-2]
    hidden_size = embed.shape[-1]
    assert hidden.shape[-2:] == (seq, hidden_size), f"hidden {tuple(hidden.shape)} != embed {tuple(embed.shape)}"
    assert eh_proj_weight.shape == (
        hidden_size,
        2 * hidden_size,
    ), f"eh_proj must be [hidden, 2*hidden] = [{hidden_size}, {2 * hidden_size}], got {tuple(eh_proj_weight.shape)}"

    e = rms_norm(embed, enorm_weight, eps)
    h = rms_norm(hidden, hnorm_weight, eps)
    return (torch.cat([e, h], dim=-1).float() @ eh_proj_weight.float().t()).to(embed.dtype)


def glm_mtp_module_reference(
    config,
    mla_weights,
    mtp_weights: dict,
    attn_norm_weight: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    embed: torch.Tensor,
    hidden: torch.Tensor,
    seq_len: int,
    *,
    ffn_weights: dict | None = None,
    moe_weights: dict | None = None,
    indexer_topk: torch.Tensor | None = None,
    return_indexer_topk: bool = False,
    mla_ref=None,
    actual_start: int = 0,
    actual_end: int | None = None,
):
    """One complete MTP module on CPU, matching ``TtMTPModule.forward``.

    Returns ``(x, out, out_head_normed, kvpe_cache)``: the projection output, the layer output, and
    the same after ``shared_head.norm``. Every level needs its own ``SparseMLAReference``.
    """
    x = fused_mtp_reference(
        embed,
        hidden,
        mtp_weights["enorm"],
        mtp_weights["hnorm"],
        mtp_weights["eh_proj"],
        config.rms_norm_eps,
    )

    layer_out = glm_decoder_layer_reference(
        config,
        mla_weights,
        attn_norm_weight,
        ffn_norm_weight,
        hidden_states=x,
        seq_len=seq_len,
        ffn_weights=ffn_weights,
        moe_weights=moe_weights,
        indexer_topk=indexer_topk,
        return_indexer_topk=return_indexer_topk,
        mla_ref=mla_ref,
        actual_start=actual_start,
        actual_end=actual_end,
    )
    out, kvpe_cache = layer_out[0], layer_out[1]

    out_head_normed = rms_norm(out, mtp_weights["shared_head_norm"], config.rms_norm_eps)
    if return_indexer_topk:
        return x, out, out_head_normed, kvpe_cache, layer_out[2]
    return x, out, out_head_normed, kvpe_cache


def glm_mtp_predictor_reference(
    config,
    mla_weights,
    mtp_weights: dict,
    attn_norm_weight: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    embeds,
    hidden: torch.Tensor,
    seq_len: int,
    *,
    ffn_weights: dict | None = None,
    moe_weights: dict | None = None,
    num_levels: int | None = None,
    index_share: bool = True,
    hiddens=None,
    mla_refs=None,
    actual_start: int = 0,
    actual_end: int | None = None,
):
    """K MTP levels on CPU, matching ``TtMTPPredictor.forward``.

    One weight set replayed per level, chained through the hidden state; ``hiddens`` teacher-forces
    each level instead. Returns per-level lists plus the K caches stacked as the device returns them.
    """
    embeds = list(embeds)
    if num_levels is None:
        num_levels = len(embeds)
    assert num_levels == len(embeds), f"num_levels={num_levels} but got {len(embeds)} embeddings"
    assert num_levels >= 1, f"num_levels must be >= 1, got {num_levels}"
    if hiddens is not None:
        hiddens = list(hiddens)
        assert num_levels == len(hiddens), f"num_levels={num_levels} but got {len(hiddens)} hidden states"
    if mla_refs is not None:
        mla_refs = list(mla_refs)
        assert num_levels == len(mla_refs), f"num_levels={num_levels} but got {len(mla_refs)} MLA references"

    xs, outs, normeds, kvpes = [], [], [], []
    shared_topk = None
    h = hidden
    for k, embed in enumerate(embeds):
        inject = shared_topk if (index_share and k > 0) else None
        want_topk = index_share and k == 0
        result = glm_mtp_module_reference(
            config,
            mla_weights,
            mtp_weights,
            attn_norm_weight,
            ffn_norm_weight,
            embed,
            h if hiddens is None else hiddens[k],
            seq_len,
            ffn_weights=ffn_weights,
            moe_weights=moe_weights,
            indexer_topk=inject,
            return_indexer_topk=want_topk,
            mla_ref=None if mla_refs is None else mla_refs[k],
            actual_start=actual_start,
            actual_end=actual_end,
        )
        x, out, out_head_normed, kvpe = result[0], result[1], result[2], result[3]
        if want_topk:
            shared_topk = result[4]
        xs.append(x)
        outs.append(out)
        normeds.append(out_head_normed)
        kvpes.append(kvpe)
        h = out_head_normed

    return xs, outs, normeds, torch.cat(kvpes, dim=0)
