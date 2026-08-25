# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.2 MTP (Multi-Token Prediction) module CPU reference.

One MTP module is the fused input projection followed by one full GLM decoder layer:

    x^k      = eh_proj( cat[ enorm(embed(t_{p+k})) , hnorm(h^{k-1}[p]) ] )     <- fused_mtp_reference
    h^k      = GLM_decoder_layer(x^k)                                          <- glm_decoder_layer_reference
    out      = shared_head.norm(h^k)

Only the first line is new math; the decoder layer is ``reference.glm_5_1.block`` verbatim, which is
already the truth for a GLM DSA layer (it composes ``cpu_deepseek_v32.SparseMLAReference`` with
``glm_moe_reference``). Nothing here re-implements attention, the indexer, or MoE.

Concat order is **embedding first**, confirmed independently by vLLM ``deepseek_mtp.py``
(``torch.cat([inputs_embeds, previous_hidden_states], dim=-1)``), vLLM ``glm4_moe_mtp.py``, SGLang
``glm4_moe_nextn.py``, and by the checkpoint's own weight statistics: the second column-half of
``eh_proj`` is heavier-tailed (std 0.0238 vs 0.0149) matching ``hnorm``'s heavy gain
(max/mean 6.2 vs ``enorm``'s 1.3).

The MTP weights live on layer ``num_hidden_layers`` (78 for GLM-5.2) — a layer that is otherwise a
complete, full-size MoE decoder layer: both layernorms, full MLA *with* indexer weights (layer 77
has none), and 256 routed experts. See issue #53533.
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
    *,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """The MTP input projection: ``eh_proj(cat[enorm(embed), hnorm(hidden)])``.

    Args:
        embed: token embeddings of the SHIFTED ids, ``embed(t_{p+k})`` [.., seq, hidden].
        hidden: the previous level's hidden state at position p, ``h^{k-1}[p]`` [.., seq, hidden].
            For level 1 this is the trunk output, taken AFTER ``model.norm`` — vLLM's
            ``deepseek_mtp.py`` says so in as many words ("Recycle the post-final-norm hidden into
            the next draft step"), and both vLLM's and SGLang's GLM model ``forward`` return the
            post-norm tensor.
        enorm_weight / hnorm_weight: the two RMSNorm gains [hidden].
        eh_proj_weight: HF layout ``[hidden, 2 * hidden]`` (out, in) — BF16 in the checkpoint, with
            no ``weight_scale_inv``, unlike every MLA/MoE weight on the same layer.
        eps: ``config.rms_norm_eps``.
        positions: absolute position of each row [.., seq]. Defaults to ``arange(seq)``. Rows at
            absolute position 0 have their EMBEDDING zeroed before the norm, mirroring vLLM
            (``inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)``) — at
            position 0 there is no preceding token to have predicted from. Pass explicit positions
            under chunked prefill or any block-cyclic layout, where row index != absolute position.

    Returns:
        ``x^k`` [.., seq, hidden] in ``embed``'s dtype.
    """
    seq = embed.shape[-2]
    hidden_size = embed.shape[-1]
    assert hidden.shape[-2:] == (seq, hidden_size), f"hidden {tuple(hidden.shape)} != embed {tuple(embed.shape)}"
    assert eh_proj_weight.shape == (
        hidden_size,
        2 * hidden_size,
    ), f"eh_proj must be [hidden, 2*hidden] = [{hidden_size}, {2 * hidden_size}], got {tuple(eh_proj_weight.shape)}"

    if positions is None:
        positions = torch.arange(seq, device=embed.device)
    embed = torch.where(positions.reshape(*positions.shape, 1) == 0, torch.zeros_like(embed), embed)

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
    positions: torch.Tensor | None = None,
):
    """One complete MTP module on CPU, matching ``TtMTPModule.forward``.

    Args:
        config: GLM HF-attribute config (``glm_5_2_hf_config()``); ``config.max_seq_len`` should be set.
        mla_weights: canonical MLA+indexer weights for the MTP layer (78), as returned by
            ``sparse_mla_reference.build_weights`` / ``cpu_deepseek_v32.pretrained_mla_weights``.
            The same dict feeds ``ttMLA``.
        mtp_weights: ``{"enorm", "hnorm", "eh_proj", "shared_head_norm"}``.
        attn_norm_weight / ffn_norm_weight: layer 78's ``input_layernorm`` / ``post_attention_layernorm``.
        embed / hidden: see :func:`fused_mtp_reference`.
        seq_len: sequence length (sizes the sparse-MLA KVPE buffer).
        ffn_weights / moe_weights: exactly one, as for ``glm_decoder_layer_reference``. Layer 78 is
            a 256-expert MoE layer, so serving always takes ``moe_weights``.

    Returns:
        ``(x, out, out_head_normed, kvpe_cache)`` where ``x`` is the fused-projection output (the
        decoder layer's input), ``out`` is the layer output BEFORE ``shared_head.norm``, and
        ``out_head_normed`` is after it. Both output forms are returned deliberately: which one
        feeds level k+1's ``hnorm`` is a live question at MTP2, and returning both makes it a PCC
        comparison rather than a guess.
    """
    x = fused_mtp_reference(
        embed,
        hidden,
        mtp_weights["enorm"],
        mtp_weights["hnorm"],
        mtp_weights["eh_proj"],
        config.rms_norm_eps,
        positions=positions,
    )

    out, kvpe_cache = glm_decoder_layer_reference(
        config,
        mla_weights,
        attn_norm_weight,
        ffn_norm_weight,
        hidden_states=x,
        seq_len=seq_len,
        ffn_weights=ffn_weights,
        moe_weights=moe_weights,
    )

    out_head_normed = rms_norm(out, mtp_weights["shared_head_norm"], config.rms_norm_eps)
    return x, out, out_head_normed, kvpe_cache
