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

Concat order is **embedding first**. This is settled by the checkpoint itself, and corroborated by
vLLM ``deepseek_mtp.py`` (``torch.cat([inputs_embeds, previous_hidden_states], dim=-1)``), vLLM
``glm4_moe_mtp.py`` and SGLang ``glm4_moe_nextn.py``.

The checkpoint evidence is the structure of ``eh_proj``. Because
``cat[a, b] @ W.t() == a @ W[:, :H].t() + b @ W[:, H:].t()``, the two column-halves of the BF16
``[6144, 12288]`` weight *are* the two concat slots' projections — and they are not alike:

    model.layers.78.eh_proj.weight          [:, :6144]      [:, 6144:]
      std(OFF-diagonal entries)               0.014872        0.013868
      mean(diagonal)                         +0.024262       +1.515491
        ... in units of std(off-diagonal)        1.63 s          109.28 s
      mean(diag^2) / mean(off-diag^2)            3.69 x        12017.51 x   generic matrix -> ~1
      diagonal's share of total variance         0.06 %           66.18 %
      diagonal entries that are positive        94.43 %          100.00 %   6144 of 6144

The second half is ``1.5155 * I`` (diag std 0.120, min +0.085) plus a general matrix carrying the
remaining third of its energy. That is a pass-through, and a pass-through can only sit on the HIDDEN
path: copying one token's embedding into layer 78's residual stream is not a coherent function, and
MTP's measured draft-acceptance rate (59.7% for SGLang MTP4 over 176 SPEED-Bench requests on 8xH200,
70.2% on the 10-prompt debug set) requires ``x`` to be dominated by the trunk state and merely nudged
by the next token. So the second concat slot takes the hidden state and the first takes the
embedding.

Do NOT read this off the raw per-half std (0.02384 vs 0.01488) and conclude the second half is
"heavier-tailed" — an earlier version of this docstring did. The two halves' off-diagonal statistics
are the same to within 7% (the second half's is if anything *smaller*); the entire std gap is the
diagonal. The natural follow-up, that ``diag`` of the hidden half inverts ``hnorm``'s per-channel
gain, was also checked and is null: ``corr(diag(W[:, 6144:]), 1 / hnorm.weight) = -0.060`` against a
shuffled null of 0.021. The pass-through is a near-uniform scalar, not an elementwise un-norm.

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
    indexer_topk: torch.Tensor | None = None,
    return_indexer_topk: bool = False,
    mla_ref=None,
    actual_start: int = 0,
    actual_end: int | None = None,
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
        indexer_topk / return_indexer_topk: index sharing across MTP levels; see
            :func:`glm_mtp_predictor_reference`, which is the only caller that needs them.
        mla_ref / actual_start / actual_end: chunked prefill, forwarded to
            :func:`glm_decoder_layer_reference`. One persistent ``SparseMLAReference`` PER LEVEL --
            each MTP level owns its own KV cache, so sharing one instance across levels would let
            level k attend to level k-1's keys. ``positions`` must be set to match: under chunking
            row 0 is absolute position ``actual_start``, not 0, and only the chunk that really holds
            position 0 may have its embedding row zeroed.

    Returns:
        ``(x, out, out_head_normed, kvpe_cache)`` where ``x`` is the fused-projection output (the
        decoder layer's input), ``out`` is the layer output BEFORE ``shared_head.norm``, and
        ``out_head_normed`` is after it. Both output forms are returned deliberately: which one
        feeds level k+1's ``hnorm`` is a live question at MTP2, and returning both makes it a PCC
        comparison rather than a guess. With ``return_indexer_topk`` a fifth element, this level's
        top-k indices.
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


# ``h^{k-1}`` — which of the previous level's two output forms feeds level k's ``hnorm``.
CHAIN_FROM_NORM = "norm"  # out_head_normed, i.e. shared_head.norm(h^{k-1})
CHAIN_FROM_RAW = "raw"  # out, i.e. h^{k-1} straight off the decoder layer
CHAIN_FROM_CHOICES = (CHAIN_FROM_NORM, CHAIN_FROM_RAW)


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
    chain_from: str = CHAIN_FROM_NORM,
    positions: torch.Tensor | None = None,
    hiddens=None,
    mla_refs=None,
    actual_start: int = 0,
    actual_end: int | None = None,
):
    """K MTP levels on CPU, matching ``TtMTPPredictor.forward``.

    The DeepSeek-V3 / GLM-5.2 scheme: K levels predicted at ONE position, K KV caches, ONE shared
    weight module replayed K times. Level k consumes the embedding of ``t_{p+k}`` and the previous
    level's hidden state, seeded by ``H^0`` (the trunk output taken AFTER ``model.norm``)::

        H^0 = hidden
        for k in 1..K:
            x^k, h^k, H^k = glm_mtp_module_reference(embed=embeds[k-1], hidden=H^{k-1})

    Args:
        embeds: K token-embedding tensors, ``embeds[k-1]`` = ``embed(t_{p+k})`` [1, seq, hidden].
            Every level's row at absolute position 0 is zeroed, as in :func:`fused_mtp_reference` —
            vLLM zeroes at position 0 for all k, not just k=1.
        hidden: ``H^0`` [1, seq, hidden].
        num_levels: K. Defaults to ``len(embeds)``; asserted to agree with it.
        index_share: mirror ``config.index_share_for_mtp_iteration``. True (GLM-5.2's setting) runs
            the indexer once, on level 1, and reuses its top-k for levels 2..K — which is what the
            device does when ``TtMTPPredictor`` injects level 1's ``indexer_indices``. This is NOT
            cosmetic: with ``seq_len > config.index_topk`` (5120 > 2048) top-k is selective on ~60%
            of rows, so a reference that recomputed per level would disagree with a sharing device
            on most of the sequence and the disagreement would look like a module bug.
        chain_from: which output form of level k-1 feeds level k's ``hnorm`` —
            ``"norm"`` (``out_head_normed``, the default) or ``"raw"`` (``out``). Kept a parameter
            on both sides so settling it is a flag flip and a PCC comparison, not an edit.
        hiddens: teacher forcing. K hidden states, ``hiddens[k]`` used as level k+1's ``H^k``
            INSTEAD of the chained value. ``None`` (the default) chains, matching the device.
            Pass the DEVICE's own per-level hidden states to compare each level against the
            reference in isolation: the reference then never sees its own level k-1 output, so a
            level's PCC measures only that level's math and the thresholds do not need the
            per-level drift allowance ``_accumulated_pcc`` exists for. ``hiddens[0]`` overrides
            ``hidden`` too, so ``hidden`` may be a dummy when this is set.
        mla_refs: K caller-owned ``SparseMLAReference`` instances, one PER LEVEL (each level owns
            its own KV cache, so one shared instance would let level k attend to level k-1's
            keys). Required for chunked prefill, where the same objects must be handed back on
            every chunk so the caches and fill watermark carry across.
        actual_start / actual_end: this call's cache write window, forwarded to
            :func:`glm_mtp_module_reference`. Set ``positions`` to match --
            ``arange(actual_start, actual_start + seq)`` -- or every chunk zeroes its own row 0.
        positions / ffn_weights / moe_weights / config / mla_weights / mtp_weights /
            attn_norm_weight / ffn_norm_weight: as for :func:`glm_mtp_module_reference`; the same
            single weight set drives every level.

    Returns:
        ``(xs, outs, out_head_normeds, kvpe_cache)`` — three K-element lists, one per level, plus
        the K per-level caches stacked into ``[K, 1, seq, kv_lora_rank + qk_rope_head_dim]``. That
        stacking is deliberate: it is exactly the layout ``ttMLA.kv_cache_to_host`` returns for a
        cache allocated with ``num_kvpe_cache_layers=K``, so level k's slot compares directly and a
        level that wrote the wrong slot fails loudly. Nothing else catches a slot collision — each
        level reads back only what it just wrote, so its *output* is right either way.
    """
    assert chain_from in CHAIN_FROM_CHOICES, f"chain_from must be one of {CHAIN_FROM_CHOICES}, got {chain_from!r}"
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
        # Level 1 always computes its own top-k; it is the level the sharing levels share FROM.
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
            positions=positions,
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
        h = out_head_normed if chain_from == CHAIN_FROM_NORM else out

    return xs, outs, normeds, torch.cat(kvpes, dim=0)
