# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One Kimi-K3 layer on host, in the order ``TtKimiK3Block.forward`` runs it, at a GLOBAL layer index.

    h      = pre-read(live, sealed, q_pre[L])        (layer 0: the live stream itself)
    seal if L % 12 == 0                              (live -> sealed, live restarts empty)
    live  += attention(rms_norm(h, attn_norm))       (KDA carry / MLA KV flow in and out)
    h      = post-read(live, sealed, q_post[L])
    live  += ffn(rms_norm(h, ffn_norm))              (dense at layer 0, LatentMoE after)

The AttnRes reads are ``reference.kimi_k3.attn_res.attn_res.attn_res``; the components are the
functional fp32 references in ``reference_ops`` (pinned against the module references). Everything
after token mixing is token-wise, so the layer is evaluated for a row sample ``rows`` -- exactly,
not approximately -- while the attention's token mixing runs over every token (plan §5.5 lever 2).

Inputs and outputs are in natural token order, ``[T, ...]`` with T the chunk's tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import EPS, attn_res, attn_res_scores
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.weights import fold_queries
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kimi_k3.single_layer.reference_ops import (
    MlaDims,
    PooledExperts,
    dense_mlp,
    kda_forward,
    mla_forward,
    moe_forward,
    rms_norm,
)

BLOCK = KimiK3Config.ATTN_RES_BLOCK_SIZE


@dataclass
class AttentionCarry:
    """What one layer's attention carries between chunks of a request."""

    kda: KDAReferenceState | None = None
    mla_prefix: torch.Tensor | None = None  # [P, 576] KV rows of every earlier token


@dataclass
class LayerResult:
    live: torch.Tensor | None  # [R, d] after the layer (None for a kv_only layer)
    sealed: torch.Tensor  # [R, S_out, d]
    attn_delta: torch.Tensor | None  # [R, d] what attention wrote
    ffn_delta: torch.Tensor | None  # [R, d] what the FFN wrote
    carry: AttentionCarry = field(default_factory=AttentionCarry)
    kvpe: torch.Tensor | None = None  # [T, 576] this chunk's MLA KV rows
    routing: tuple[torch.Tensor, torch.Tensor] | None = None
    model_out: torch.Tensor | None = None  # [R, d] q_out read + final norm (full tail only)
    ffn_input: torch.Tensor | None = None  # [R, d] the FFN's input: normed post-read
    # Mean softmax entropy of each AttnRes read as a fraction of log(candidates): near 0 is one-hot
    # (other planes untested), near 1 uniform (the query untested).
    entropy: dict = field(default_factory=dict)


def _entropy(running_sum, block_residual, q) -> float:
    candidates = torch.cat([block_residual, running_sum.unsqueeze(1)], dim=1).float()
    p = attn_res_scores(candidates, q.float(), EPS).softmax(-1)
    return (-(p * p.clamp_min(1e-30).log()).sum(-1).mean() / math.log(candidates.shape[1])).item()


def layer_queries(attn_res_weights, layer: int, prefix: str):
    """``(q_pre, q_post, q_out)`` folded for one global layer."""
    q_pre, q_post, q_out = fold_queries(attn_res_weights, 1, prefix, first_layer_idx=layer)
    return q_pre[0], q_post[0], q_out


def reference_layer(
    layer: int,
    state: dict,
    queries,
    live: torch.Tensor,
    sealed: torch.Tensor,
    *,
    rows: torch.Tensor,
    carry: AttentionCarry,
    experts: PooledExperts | None = None,
    routing: tuple[torch.Tensor, torch.Tensor] | None = None,
    kv_only: bool = False,
    final_norm: torch.Tensor | None = None,
    drop: str | None = None,
) -> LayerResult:
    """One global layer over a chunk. ``live [T, d]``, ``sealed [T, S, d]`` with ``S == ceil(L/12)``.

    ``experts`` are the device's routed experts (``PooledRoutedExperts.reference_experts()``);
    ``routing`` optionally replaces the reference gate for the sampled rows with the device's.
    ``final_norm`` turns on the model-level ``q_out`` read and final RMSNorm (non-pipelined tail).
    ``drop`` ("attn" or "ffn") zeroes that branch's write -- the counterfactual a test uses to prove its
    oracle would see the branch missing.
    """
    eps = KimiK3Config.RMS_NORM_EPS
    q_pre, q_post, q_out = queries
    live, sealed = live.float(), sealed.float()
    expected_sealed = -(-layer // BLOCK)
    if sealed.shape[1] != expected_sealed:
        raise ValueError(f"layer {layer} reads against {expected_sealed} sealed planes, got {sealed.shape[1]}")

    # Pre-read over every token: the attention mixes all of them.
    entropy = {}
    if sealed.shape[1]:
        entropy["pre"] = _entropy(live, sealed, q_pre)
    h = live if sealed.shape[1] == 0 else attn_res(live, sealed, q_pre, EPS)
    if layer % BLOCK == 0:
        sealed, live = torch.cat([sealed, live.unsqueeze(1)], dim=1), None

    normed = rms_norm(h, state["attn_norm_weight"], eps)
    out = LayerResult(live=None, sealed=sealed.index_select(0, rows), attn_delta=None, ffn_delta=None, entropy=entropy)
    if "mla_weights" in state:
        attn_rows, kvpe = mla_forward(
            normed, state["mla_weights"], MlaDims.from_hf(kimi_k3_hf_config()), prefix_kvpe=carry.mla_prefix, rows=rows
        )
        out.kvpe = kvpe
        out.carry = AttentionCarry(
            mla_prefix=kvpe if carry.mla_prefix is None else torch.cat([carry.mla_prefix, kvpe], dim=0)
        )
    else:
        attn_rows, kda_state = kda_forward(normed, state["kda_weights"], kimi_k3_kda_config(), carry.kda, rows=rows)
        out.carry = AttentionCarry(kda=kda_state)
    if kv_only:
        return out

    if drop == "attn":
        attn_rows = torch.zeros_like(attn_rows)
    out.attn_delta = attn_rows
    live_rows = attn_rows if live is None else live.index_select(0, rows) + attn_rows
    sealed_rows = out.sealed
    entropy["post"] = _entropy(live_rows, sealed_rows, q_post)
    h = attn_res(live_rows, sealed_rows, q_post, EPS)
    normed = rms_norm(h, state["ffn_norm_weight"], eps)
    out.ffn_input = normed
    if "ffn_weights" in state:
        ffn_rows = dense_mlp(normed, state["ffn_weights"])
    else:
        if experts is None:
            raise ValueError(f"layer {layer} is MoE: pass the device's routed experts")
        ffn_rows, out.routing = moe_forward(
            normed, state, experts, top_k=KimiK3Config.NUM_EXPERTS_PER_TOKEN, eps=eps, routing=routing
        )
    if drop == "ffn":
        ffn_rows = torch.zeros_like(ffn_rows)
    out.ffn_delta = ffn_rows
    out.live = live_rows + ffn_rows
    if final_norm is not None:
        entropy["out"] = _entropy(out.live, sealed_rows, q_out)
        out.model_out = rms_norm(attn_res(out.live, sealed_rows, q_out, EPS), final_norm, eps)
    return out
