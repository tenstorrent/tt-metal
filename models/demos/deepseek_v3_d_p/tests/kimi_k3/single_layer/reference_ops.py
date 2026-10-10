# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional fp32 Kimi-K3 component references for the single-layer suite.

The nn.Module references (``MLAReference``, ``KimiSparseMoeBlock``, ``kda_forward_reference``) are the
truth models; these functions reproduce their math (``test_reference_ops.py`` pins each against its
module at reduced dims) but add what a CI-budgeted single-layer reference needs:

* fp32 throughout, so the reference's own rounding stays far below the device's, and bf16 rounding is
  applied only where a caller says the device rounds;
* ``rows``: compute only a sample of output rows. Everything after token mixing is token-wise, so a
  sampled row is exact, not approximate. Token mixing itself (the KDA recurrence, MLA K/V) still
  runs over every token;
* a pooled routed-expert representation: ``E`` experts backed by ``P`` weight sets plus an exact
  per-expert sign transform, with GEMMs grouped by pool.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState, _initial_state
from models.demos.deepseek_v3_d_p.reference.kda.ops import (
    causal_depthwise_conv_reference,
    kda_gate_reference,
    kda_recurrent_reference,
    sigmoid_gated_rms_norm_reference,
)

SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0


def _select(x: torch.Tensor, rows: torch.Tensor | None, dim: int = 0) -> torch.Tensor:
    return x if rows is None else x.index_select(dim, rows)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32 RMSNorm (``KimiRMSNorm`` / ``DeepseekV3RMSNorm`` math without the dtype round trip)."""
    x = x.float()
    return weight.float() * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps))


def situ_and_mul(gate: torch.Tensor, up: torch.Tensor, beta=SITU_BETA, linear_beta=SITU_LINEAR_BETA):
    """``SituAndMul`` from ``modeling_kimi_moe.py``: beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb)."""
    gate, up = gate.float(), up.float()
    activated = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return activated * up


def dense_mlp(x: torch.Tensor, w: Mapping[str, torch.Tensor], rows: torch.Tensor | None = None) -> torch.Tensor:
    """SiTU-GLU MLP (``KimiMLP``): keys ``gate_proj``, ``up_proj``, ``down_proj`` (``[out, in]``)."""
    x = _select(x, rows).float()
    activated = situ_and_mul(F.linear(x, w["gate_proj"].float()), F.linear(x, w["up_proj"].float()))
    return F.linear(activated, w["down_proj"].float())


# ---------------------------------------------------------------------------------------------- KDA


def kda_forward(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    config: KDAConfig,
    state: KDAReferenceState | None = None,
    rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, KDAReferenceState]:
    """``kda_forward_reference`` over ``hidden [T, d]``; the output is computed for ``rows`` only.

    The projections that feed the recurrence (q, k, v, f, b) run over every token; the output gate,
    o_norm and o_proj only over the sampled rows. Returns ``(out [R, d], final state)``.
    """
    if hidden.ndim != 2:
        raise ValueError(f"hidden must be [T, d], got {tuple(hidden.shape)}")
    x = hidden.float().unsqueeze(0)
    state = _initial_state(x, config) if state is None else state
    q, q_state = causal_depthwise_conv_reference(
        F.linear(x, weights["q_proj.weight"].float()), weights["q_conv1d.weight"], state.q_convolution
    )
    k, k_state = causal_depthwise_conv_reference(
        F.linear(x, weights["k_proj.weight"].float()), weights["k_conv1d.weight"], state.k_convolution
    )
    v, v_state = causal_depthwise_conv_reference(
        F.linear(x, weights["v_proj.weight"].float()), weights["v_conv1d.weight"], state.v_convolution
    )
    sequence = x.shape[1]
    heads = config.num_heads
    q = q.reshape(1, sequence, heads, config.head_k_dim)
    k = k.reshape(1, sequence, heads, config.head_k_dim)
    v = v.reshape(1, sequence, heads, config.head_v_dim)
    raw_gate = F.linear(F.linear(x, weights["f_a_proj.weight"].float()), weights["f_b_proj.weight"].float()).reshape(
        1, sequence, heads, config.head_k_dim
    )
    gate = kda_gate_reference(raw_gate, weights["A_log"], weights["dt_bias"], config.gate_lower_bound)
    beta = torch.sigmoid(F.linear(x, weights["b_proj.weight"].float()))
    output, recurrent = kda_recurrent_reference(q, k, v, gate, beta, state.recurrent)

    output = _select(output[0], rows)
    x_rows = _select(x[0], rows)
    if config.use_full_rank_gate:
        output_gate = F.linear(x_rows, weights["g_proj.weight"].float())
    else:
        output_gate = F.linear(F.linear(x_rows, weights["g_a_proj.weight"].float()), weights["g_b_proj.weight"].float())
    output_gate = output_gate.reshape(-1, heads, config.head_v_dim)
    output = sigmoid_gated_rms_norm_reference(output, output_gate, weights["o_norm.weight"], config.norm_eps)
    output = F.linear(output.reshape(-1, config.v_dim), weights["o_proj.weight"].float())
    return output, KDAReferenceState(recurrent, q_state, k_state, v_state)


# ---------------------------------------------------------------------------------------------- MLA


@dataclass(frozen=True)
class MlaDims:
    num_heads: int = 96
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rms_norm_eps: float = 1e-5

    @classmethod
    def from_hf(cls, config) -> "MlaDims":
        return cls(
            num_heads=config.num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            rms_norm_eps=config.rms_norm_eps,
        )

    @property
    def softmax_scale(self) -> float:
        # K3 is NoPE with no YaRN: plain qk_head_dim**-0.5, no mscale.
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5


def mla_kvpe(hidden: torch.Tensor, w: Mapping[str, torch.Tensor], dims: MlaDims) -> torch.Tensor:
    """The per-token cache row ``[T, kv_lora_rank + rope]``: normed latent ‖ unrotated k_pe (NoPE)."""
    compressed = F.linear(hidden.float(), w["kv_a_proj_with_mqa.weight"].float())
    latent, k_pe = torch.split(compressed, [dims.kv_lora_rank, dims.qk_rope_head_dim], dim=-1)
    return torch.cat([rms_norm(latent, w["kv_a_layernorm.weight"], dims.rms_norm_eps), k_pe], dim=-1)


def mla_forward(
    hidden: torch.Tensor,
    w: Mapping[str, torch.Tensor],
    dims: MlaDims,
    *,
    prefix_kvpe: torch.Tensor | None = None,
    rows: torch.Tensor | None = None,
    query_block: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K3 MLA (absorbed, NoPE, output-gated) over ``hidden [T, d]`` following ``prefix_kvpe [P, 576]``.

    ``rows`` index into this call's T tokens; each sampled query attends causally to the prefix and to
    tokens ``<= row``. Returns ``(out [R, d], kvpe [T, 576])`` -- kvpe for every token of the call.
    """
    hidden = hidden.float()
    kvpe = mla_kvpe(hidden, w, dims)
    keys = kvpe if prefix_kvpe is None else torch.cat([prefix_kvpe.float(), kvpe], dim=0)
    prefix_len = keys.shape[0] - kvpe.shape[0]
    rows = torch.arange(hidden.shape[0]) if rows is None else rows
    x = hidden.index_select(0, rows)

    heads, nope, rope, latent = dims.num_heads, dims.qk_nope_head_dim, dims.qk_rope_head_dim, dims.kv_lora_rank
    q = F.linear(
        rms_norm(F.linear(x, w["q_a_proj.weight"].float()), w["q_a_layernorm.weight"], dims.rms_norm_eps),
        w["q_b_proj.weight"].float(),
    ).view(-1, heads, nope + rope)
    q_nope, q_pe = torch.split(q, [nope, rope], dim=-1)
    kv_b = w["kv_b_proj.weight"].float().view(heads, nope + dims.v_head_dim, latent)
    q_latent = torch.einsum("rhn,hnl->rhl", q_nope, kv_b[:, :nope])
    query = torch.cat([q_latent, q_pe], dim=-1)  # [R, H, 576]

    out_latent = torch.empty(query.shape[0], heads, latent)
    positions = rows + prefix_len
    for start in range(0, query.shape[0], query_block):
        end = min(start + query_block, query.shape[0])
        limit = int(positions[start:end].max()) + 1
        scores = torch.einsum("rhd,td->hrt", query[start:end], keys[:limit]) * dims.softmax_scale
        mask = torch.arange(limit).unsqueeze(0) > positions[start:end].unsqueeze(1)
        scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
        out_latent[start:end] = torch.einsum("hrt,tl->rhl", scores.softmax(-1), keys[:limit, :latent])

    attn = torch.einsum("rhl,hvl->rhv", out_latent, kv_b[:, nope:]).reshape(-1, heads * dims.v_head_dim)
    attn = attn * torch.sigmoid(F.linear(x, w["g_proj.weight"].float()))
    return F.linear(attn, w["o_proj.weight"].float()), kvpe


# ---------------------------------------------------------------------------------------------- MoE


@dataclass
class PooledExperts:
    """``E`` routed experts backed by ``P`` weight sets: expert ``e`` is pool ``pool_of[e]`` with its
    intermediate channels multiplied by ``sign[e]`` (±1) before ``down`` -- i.e. the rows of
    ``down^T`` flipped, which commutes exactly with block-float quantization of whole rows.

    ``gate``/``up`` ``[P, I, Hl]``, ``down`` ``[P, Hl, I]``, ``pool_of`` ``[E]``, ``sign`` ``[E, I]``.
    ``P == E`` with all-ones signs is the plain (un-pooled) representation.
    """

    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    pool_of: torch.Tensor
    sign: torch.Tensor

    @property
    def num_experts(self) -> int:
        return self.pool_of.shape[0]

    def expert(self, e: int) -> dict[str, torch.Tensor]:
        """Materialize expert ``e`` as a plain ``{gate_proj, up_proj, down_proj}`` dict."""
        p = int(self.pool_of[e])
        return {
            "gate_proj": self.gate[p],
            "up_proj": self.up[p],
            "down_proj": self.down[p] * self.sign[e].to(self.down.dtype).unsqueeze(0),
        }


def moe_route(
    x: torch.Tensor, gate_weight: torch.Tensor, bias: torch.Tensor, top_k: int, scaling: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """``KimiMoEGate`` (sigmoid, noaux_tc bias for choice only, renormalized). ``x`` is ``[T, d]``."""
    scores = F.linear(x.float(), gate_weight.float()).sigmoid()
    _, idx = torch.topk(scores + bias.float().unsqueeze(0), k=top_k, dim=-1, sorted=False)
    weight = scores.gather(1, idx)
    if top_k > 1:  # moe_renormalize, as KimiMoEGate applies it
        weight = weight / (weight.sum(-1, keepdim=True) + 1e-20)
    return idx, weight * scaling


def routed_experts(
    latent: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor, experts: PooledExperts
) -> torch.Tensor:
    """Σ_k weight[t,k] · expert[idx[t,k]](latent[t]); GEMMs grouped by pool. ``latent [R, Hl]``."""
    out = torch.zeros_like(latent, dtype=torch.float32)
    flat_expert = idx.reshape(-1)
    flat_token = torch.arange(idx.shape[0]).repeat_interleave(idx.shape[1])
    flat_weight = weight.reshape(-1).float()
    flat_pool = experts.pool_of[flat_expert]
    for p in torch.unique(flat_pool).tolist():
        sel = (flat_pool == p).nonzero(as_tuple=True)[0]
        tokens = flat_token[sel]
        h = latent.index_select(0, tokens).float()
        activated = situ_and_mul(F.linear(h, experts.gate[p].float()), F.linear(h, experts.up[p].float()))
        activated = activated * experts.sign.index_select(0, flat_expert[sel]).float()
        y = F.linear(activated, experts.down[p].float()) * flat_weight[sel].unsqueeze(1)
        out.index_add_(0, tokens, y)
    return out


def moe_forward(
    x: torch.Tensor,
    w: Mapping,
    experts: PooledExperts,
    *,
    top_k: int = 16,
    eps: float = 1e-5,
    scaling: float = 1.0,
    rows: torch.Tensor | None = None,
    routing: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """``KimiSparseMoeBlock`` (LatentMoE + latent norm + shared expert) over ``x [T, d]``, sampled rows.

    K3's settings only: sigmoid scoring, one expert group, the latent norm on. A config that changes
    any of them needs its own path, not a flag here.

    ``w``: ``gate_weights {weight, e_score_correction_bias}``, ``latent_weights {down_proj, up_proj,
    norm}``, ``shared_expert_weights {gate_proj, up_proj, down_proj}``. ``routing`` (``idx, weight``
    for the sampled rows) overrides the reference gate, e.g. with the device's selection.
    Returns ``(out [R, d], (idx, weight))``.
    """
    x = _select(x, rows).float()
    idx, weight = (
        routing
        if routing is not None
        else moe_route(x, w["gate_weights"]["weight"], w["gate_weights"]["e_score_correction_bias"], top_k, scaling)
    )
    if idx.shape != (x.shape[0], top_k) or weight.shape != idx.shape:
        raise ValueError(f"routing must be [rows={x.shape[0]}, top_k={top_k}], got {tuple(idx.shape)}")
    latent_w = w["latent_weights"]
    latent = F.linear(x, latent_w["down_proj"].float())
    y = routed_experts(latent, idx, weight, experts)
    y = F.linear(rms_norm(y, latent_w["norm"], eps), latent_w["up_proj"].float())
    return y + dense_mlp(x, w["shared_expert_weights"]), (idx, weight)
