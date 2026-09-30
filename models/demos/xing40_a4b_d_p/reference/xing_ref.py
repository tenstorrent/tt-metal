# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone torch CPU reference for Xing4.0-29B-A4B (text decoder, MTP layer 40 skipped) chunked prefill.

Independent of HF transformers (HF modeling_xing4_0.py is only the oracle in check_hf). Loads the safetensors
checkpoint directly (bf16 -> ``dtype`` cast, no quantization), keeps an explicit per-layer MLA latent cache and runs
each block through ``run_block``, so the block graphs below are the code. Each chunk attends against the cached
latent of the prefix (the prefix is never recomputed through the layers).

Residual: mHC, 4 streams. Block input / output ``[S * 4, H]`` (token-major, stream-minor: row 4t + n is stream n of
token t; HF's layer output [B, S, 4, H] flattened); the embedding is copied to the 4 streams; the final hidden is the
plain mean of the 4 streams, then norm and lm_head.

Block (all 40 layers):
    in                                                     [S * 4, H] residual streams (H = 3584)
    attn_hc       = mHC coefficients [S, 24] fp32 from in  [pre 4 | post 4 | comb 16 (row-major 4x4, comb[i, j])]:
                                                           flatten to [S, 4H], unweighted RMS (eps 1e-6), linear
                                                           attn_hc.hc_fn [24, 4H]; pre = sigmoid(. * s0 + b),
                                                           post = 2 sigmoid(. * s1 + b), comb logits = . * s2 + b,
                                                           clamped to [-30, 30], minus the row max, exp, then 20 x
                                                           (row normalize, column normalize), each "/ (sum + 1e-6)"
    attn_in       = sum_n pre[n] * in[:, n]                [S, H]
    attn_norm     = w * rms(attn_in)                       input_layernorm, eps 1e-6
    q_resid       = q_a_layernorm(q_a_proj(attn_norm))     [S, 768], eps 1e-6
    attn_out      = MLA(attn_norm, q_resid)  [stateful]    dense causal MLA, see below; o_proj -> [S, H]
    h_mid         = post * attn_out + comb @ in            attn_residual: stream i = post_i y + sum_j comb[i, j] in_j
    ffn_hc, ffn_in, ffn_norm                               as above on h_mid (ffn_hc.*, post_attention_layernorm)
  dense (layers 0-1):
    mlp_out       = down(silu(gate(x)) * up(x))            intermediate 9216
  MoE (layers 2-39):
    router        = dense routing [S, 64] fp32             sigmoid(fp32 logits); top-4 on scores + correction bias
                                                           (one group); weights = the chosen (unbiased) scores
                                                           / (sum + 1e-20) x 2.0
    experts_out   = sum_e router[:, e] * expert_e(ffn_norm)   SwiGLU, intermediate 1024, fp32 accumulation in
                                                              expert-id order (HF's order)
    shared_out    = shared expert (SwiGLU 1024)
    mlp_out       = experts_out + shared_out
    out           = post * mlp_out + comb @ h_mid          ffn_residual

MLA (32 heads, q_lora 768, kv_lora 512, qk 128 NoPE + 64 RoPE, v 128), dense causal, no sinks, no output gate:
    q = q_b_proj(q_resid) [S, 32, 192] -> q_nope 128 | q_rope 64 (RoPE)
    kv_a_proj_with_mqa(attn_norm) [S, 576] -> latent 512 (kv_a_layernorm, eps 1e-6) | k_rope 64 (RoPE, one head)
    kv_latent state row = [kv_a_layernorm(latent) | RoPE(k_rope)] (576)
    kv_b_proj(latent) [T, 32, 256] -> k_nope 128 | v 128 per head (expanded from the cached latent each chunk)
    score = [q_nope | q_rope] . [k_nope | k_rope] * scale over keys <= query; scale = 192^-0.5 * mscale^2,
            mscale = 0.1 ln(64) + 1 (YaRN mscale_all_dim 1); softmax fp32; out = p @ v; o_proj
    RoPE: interleaved (rope_interleave true): pair (x[2i], x[2i+1]) rotated by position * inv_freq[i], YaRN
    (factor 64, beta_fast 32, beta_slow 1, original 4096, theta 1e4, truncate), cos / sin scale 1.0. The rotated
    values stay in checkpoint (interleaved) order; HF's apply_rotary_pos_emb_interleave writes the same values
    de-interleaved ([evens | odds]), which permutes q_rope and k_rope alike, so scores are unchanged. k_rope in
    kv_latent is therefore in the interleaved (Meta / GPT-J) order a device rotary_embedding_llama produces.

State per layer: kv_latent [max_seq, 576].

Precision: weights and activations in ``dtype`` (fp32 for the gates); the router, the mHC coefficients and the
attention softmax in fp32, as in HF. Attention query rows run in blocks aligned to absolute positions, the latent
expansion in fixed zero-padded row blocks, and expert token groups are padded to 32 rows, so chunked prefill
reproduces one-shot prefill (MKL's GEMM kernels depend on M).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.demos.common.bringup.reference.interface import Ctx, Step, noop, run_block
from models.demos.xing40_a4b_d_p.reference.weights import WeightLoader


@dataclass
class XingConfig:
    hidden_size: int = 3584
    vocab_size: int = 131072
    num_hidden_layers: int = 40
    rms_norm_eps: float = 1e-6
    # mHC
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    mhc_h_res_clamp_min: float = -30.0
    mhc_h_res_clamp_max: float = 30.0
    # MLA
    num_attention_heads: int = 32
    q_lora_rank: int = 768
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    # MLP / MoE
    intermediate_size: int = 9216
    moe_intermediate_size: int = 1024
    n_routed_experts: int = 64
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    first_k_dense_replace: int = 2
    routed_scaling_factor: float = 2.0
    norm_topk_prob: bool = True

    @classmethod
    def from_json(cls, path: str) -> "XingConfig":
        with open(path) as f:
            raw = json.load(f)
        cfg = cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})
        # what this reference implements; anything else in a new checkpoint must fail loudly
        assert raw.get("rope_interleave", True), "reference implements interleaved RoPE only"
        assert raw["scoring_func"] == "sigmoid" and raw["topk_method"] == "noaux_tc"
        assert raw["n_group"] == 1 and raw["topk_group"] == 1 and raw["n_shared_experts"] == 1
        assert raw["hidden_act"] == "silu" and not raw["attention_bias"] and not raw["tie_word_embeddings"]
        assert raw["num_key_value_heads"] == raw["num_attention_heads"] and raw.get("moe_layer_freq", 1) == 1
        rs = cfg.rope_scaling
        assert rs and rs.get("type", rs.get("rope_type")) == "yarn", rs
        return cfg

    def is_moe(self, i: int) -> bool:
        return i >= self.first_k_dense_replace

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def attn_scale(self) -> float:
        """HF Xing4_0Attention.scaling: qk_head_dim^-0.5 * mscale(factor, mscale_all_dim)^2."""
        rs = self.rope_scaling
        s = self.qk_head_dim**-0.5
        m_all = rs.get("mscale_all_dim", 0)
        if m_all:
            m = yarn_get_mscale(rs["factor"], m_all)
            s = s * m * m
        return s


# --------------------------------------------------------------------------------------
# Functional building blocks
# --------------------------------------------------------------------------------------


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_inv_freq(cfg: XingConfig) -> tuple[torch.Tensor, float]:
    """YaRN inverse frequencies [R / 2] fp32 and the cos / sin scale (transformers _compute_yarn_parameters)."""
    rs, dim, base = cfg.rope_scaling, cfg.qk_rope_head_dim, cfg.rope_theta
    factor, orig = rs["factor"], rs["original_max_position_embeddings"]
    beta_fast, beta_slow = rs.get("beta_fast") or 32, rs.get("beta_slow") or 1
    att = rs.get("attention_factor")
    if att is None:
        m, m_all = rs.get("mscale"), rs.get("mscale_all_dim")
        att = yarn_get_mscale(factor, m) / yarn_get_mscale(factor, m_all) if m and m_all else yarn_get_mscale(factor)

    def corr_dim(rot):
        return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))

    low, high = corr_dim(beta_fast), corr_dim(beta_slow)
    if rs.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    pos_freqs = base ** (torch.arange(0, dim, 2).float() / dim)
    extra, inter = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    keep = 1 - ramp
    return inter * (1 - keep) + extra * keep, float(att)


def rope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor, att: float):
    """cos / sin [S, R / 2] fp32: angle of pair i at each position (HF: (inv_freq @ pos).cos() * attention_scaling)."""
    freqs = positions.float()[:, None] * inv_freq[None, :]
    return freqs.cos() * att, freqs.sin() * att


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved RoPE on the last dim of x [S, ..., R] (pairs (x[2i], x[2i+1])), output in the same order.
    Element-wise the same arithmetic as HF apply_rotary_pos_emb_interleave, computed in fp32."""
    shape = (cos.shape[0],) + (1,) * (x.dim() - 2) + (cos.shape[1],)
    c, s = cos.view(shape), sin.view(shape)
    xf = x.float()
    x1, x2 = xf[..., 0::2], xf[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).flatten(-2).to(x.dtype)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """HF Xing4_0RMSNorm: normalize in fp32, cast back, then * w."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * y.to(x.dtype)


def swiglu(x, w_gate, w_up, w_down):
    return F.linear(F.silu(F.linear(x, w_gate)) * F.linear(x, w_up), w_down)


def hc_weights(x: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, cfg: XingConfig):
    """mHC coefficients [S, 24] fp32 = [pre 4 | post 4 | comb 16 (row-major, comb[i, j])] from the streams
    x [S * 4, H] (HF Xing4_0HyperConnection)."""
    n = cfg.hc_mult
    s = x.shape[0] // n
    flat = x.reshape(s, -1).float()
    flat = (flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)).to(x.dtype)
    mix = F.linear(flat, fn.to(x.dtype)).float()
    pre_w, post_w, comb_w = mix.split([n, n, n * n], dim=-1)
    pre_b, post_b, comb_b = base.split([n, n, n * n])
    s_pre, s_post, s_comb = scale.unbind(0)
    pre = torch.sigmoid(pre_w * s_pre + pre_b)
    post = 2 * torch.sigmoid(post_w * s_post + post_b)
    logits = comb_w.view(s, n, n) * s_comb + comb_b.view(n, n)
    logits = torch.clamp(logits, min=cfg.mhc_h_res_clamp_min, max=cfg.mhc_h_res_clamp_max)
    comb = torch.exp(logits - logits.amax(dim=-1, keepdim=True))
    for _ in range(cfg.hc_sinkhorn_iters):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + cfg.hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + cfg.hc_eps)
    return torch.cat([pre, post, comb.reshape(s, n * n)], dim=-1)


def hc_collapse(x: torch.Tensor, hc: torch.Tensor, n: int) -> torch.Tensor:
    """sum_n pre[:, n] * x[:, n] -> [S, H] in x.dtype (x: [S * N, H])."""
    return (hc[:, :n].unsqueeze(-1).to(x.dtype) * x.view(-1, n, x.shape[-1])).sum(dim=1)


def hc_residual(x: torch.Tensor, hc: torch.Tensor, y: torch.Tensor, n: int) -> torch.Tensor:
    """post[:, i] * y + sum_j comb[i, j] x[:, j] -> [S * N, H] (HF: post * out + comb @ residual)."""
    post = hc[:, n : 2 * n].to(x.dtype)
    comb = hc[:, 2 * n :].reshape(-1, n, n).to(x.dtype)
    out = post.unsqueeze(-1) * y.unsqueeze(-2) + torch.matmul(comb, x.view(-1, n, x.shape[-1]))
    return out.reshape(-1, x.shape[-1])


def route(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, cfg: XingConfig) -> torch.Tensor:
    """HF Xing4_0TopkRouter (sigmoid, noaux_tc, one group) as a dense routing matrix [S, E] fp32."""
    scores = F.linear(x.float(), gate_w.float()).sigmoid()
    ti = torch.topk(scores + bias.float()[None, :], k=cfg.num_experts_per_tok, dim=-1, sorted=False)[1]
    tw = scores.gather(1, ti)
    if cfg.norm_topk_prob:
        tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-20)
    tw = tw * cfg.routed_scaling_factor
    return torch.zeros_like(scores).scatter_(1, ti, tw)


EXPERT_ROW_BLOCK = 32  # per-expert GEMM rows padded to a multiple of this (known issue: MKL sgemm result depends on M)


def experts_forward(x: torch.Tensor, routing: torch.Tensor, experts: list) -> torch.Tensor:
    """sum over selected experts of routing[t, e] * expert_e(x[t]), tokens grouped per expert, accumulated in fp32
    in expert-id order (HF Xing4_0MoE.moe). ``experts[e] = (gate, up, down)``."""
    out = torch.zeros(x.shape, dtype=torch.float32)
    tok_all, exp_all = (routing != 0).nonzero(as_tuple=True)
    order = torch.argsort(exp_all, stable=True)
    tok_all, exp_all = tok_all[order], exp_all[order]
    ids, counts = torch.unique_consecutive(exp_all, return_counts=True)
    off = 0
    for e, n in zip(ids.tolist(), counts.tolist()):
        tok = tok_all[off : off + n]
        off += n
        xe = x[tok]
        pad = -n % EXPERT_ROW_BLOCK
        if pad:
            xe = torch.cat([xe, xe.new_zeros(pad, xe.shape[1])])
        y = swiglu(xe, *experts[e])[:n]
        out.index_add_(0, tok, y * routing[tok, e, None].to(y.dtype))
    return out.to(x.dtype)


# Attention query rows per block, aligned to absolute positions (chunk starts that are multiples of it give the same
# blocks as one-shot); latent rows per kv_b expansion GEMM (zero-padded, fixed M).
Q_BLOCK = 256
EXPAND_BLOCK = 512


def expand_latent(lat: torch.Tensor, kv_b: torch.Tensor) -> torch.Tensor:
    """kv_b_proj over the latent rows [T, 512] -> [T, 32 * 256] in fixed EXPAND_BLOCK-row GEMMs (row results do not
    depend on T)."""
    t = lat.shape[0]
    out = lat.new_empty(t, kv_b.shape[0])
    for a in range(0, t, EXPAND_BLOCK):
        b = min(t, a + EXPAND_BLOCK)
        blk = lat[a:b]
        if b - a < EXPAND_BLOCK:
            blk = torch.cat([blk, blk.new_zeros(EXPAND_BLOCK - (b - a), blk.shape[1])])
        out[a:b] = F.linear(blk, kv_b)[: b - a]
    return out


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int, scale: float) -> torch.Tensor:
    """q [Hq, S, D] at positions [start, start + S), k [Hq, T, D], v [Hq, T, Dv] with T >= start + S. Query rows in
    blocks aligned to absolute multiples of Q_BLOCK, each against keys [0, end of block). Returns [S, Hq, Dv]."""
    hq, s, _ = q.shape
    out = q.new_empty(hq, s, v.shape[-1])
    a = 0
    while a < s:
        b = min(s, (start + a) // Q_BLOCK * Q_BLOCK + Q_BLOCK - start)
        end = start + b
        sc = torch.matmul(q[:, a:b], k[:, :end].transpose(1, 2)) * scale  # [Hq, B, end]
        pos = torch.arange(start + a, end)
        sc = sc.masked_fill(torch.arange(end)[None, :] > pos[:, None], float("-inf"))
        p = F.softmax(sc, dim=-1, dtype=torch.float32).to(q.dtype)
        out[:, a:b] = torch.matmul(p, v[:, :end])
        a = b
    return out.transpose(0, 1)


# --------------------------------------------------------------------------------------
# Weights and state
# --------------------------------------------------------------------------------------

PREFIX = "model."


def load_layer(loader: WeightLoader, cfg: XingConfig, i: int, dtype=torch.float32) -> dict:
    """Name -> tensor for one layer (names below ``model.layers.{i}.``); routed experts as w["experts"][e] =
    (gate, up, down). hc_scale and the router bias stay fp32 (stored F32), everything else in ``dtype``."""
    p = f"{PREFIX}layers.{i}."
    w = {}
    for n in (n[len(p) :] for n in loader.weight_map if n.startswith(p) and ".mlp.experts." not in n):
        keep32 = n.endswith(("hc_scale", "e_score_correction_bias"))
        w[n] = loader.get(p + n).to(torch.float32 if keep32 else dtype)
    if cfg.is_moe(i):
        w["experts"] = [
            tuple(loader.get(f"{p}mlp.experts.{e}.{m}_proj.weight").to(dtype) for m in ("gate", "up", "down"))
            for e in range(cfg.n_routed_experts)
        ]
    return w


class State:
    """Per-layer prefill state: kv_latent [max_seq, 576] = kv_a_layernorm(latent) | RoPE(k_rope)."""

    def __init__(self, cfg: XingConfig, layers: list[int], max_seq: int, dtype=torch.float32):
        self.max_seq = max_seq
        self.kv = {i: torch.zeros(max_seq, cfg.kv_lora_rank + cfg.qk_rope_head_dim, dtype=dtype) for i in layers}


HC_ATTN = [
    Step("attn_hc", ("in",), "attn_hc", "op"),
    Step("attn_collapse", ("in", "attn_hc"), "attn_in", "op"),
    Step("attn_norm", ("attn_in",), "attn_norm", "norm"),
    Step("q_a", ("attn_norm",), "q_resid", "op"),
    Step("attention", ("attn_norm", "q_resid"), "attn_out", "attention", stateful=True),
    Step("attn_residual", ("in", "attn_hc", "attn_out"), "h_mid", "residual"),
]
HC_FFN = [
    Step("ffn_hc", ("h_mid",), "ffn_hc", "op"),
    Step("ffn_collapse", ("h_mid", "ffn_hc"), "ffn_in", "op"),
    Step("ffn_norm", ("ffn_in",), "ffn_norm", "norm"),
]
DENSE_FFN = [Step("mlp", ("ffn_norm",), "mlp_out", "mlp")]
MOE_FFN = [
    Step("router", ("ffn_norm",), "router", "router"),
    Step("experts", ("ffn_norm", "router"), "experts_out", "moe"),
    Step("shared_expert", ("ffn_norm",), "shared_out", "mlp"),
    Step("moe_add", ("experts_out", "shared_out"), "mlp_out", "residual"),
]
FFN_RESIDUAL = [Step("ffn_residual", ("h_mid", "ffn_hc", "mlp_out"), "out", "residual")]


class XingReference:
    """Implements models/demos/common/bringup/reference/interface.py. Keeps the requested layers resident
    (all 40 layers: 122 GB in fp32)."""

    def __init__(self, model_path: str, layers=None, dtype=torch.float32, lm_head: bool = True):
        self.cfg = XingConfig.from_json(os.path.join(model_path, "config.json"))
        self.loader = WeightLoader(model_path)
        self.dtype = dtype
        self.layer_ids = list(range(self.cfg.num_hidden_layers)) if layers is None else list(layers)
        assert all(0 <= i < self.cfg.num_hidden_layers for i in self.layer_ids), "MTP layer is out of scope"
        self.embed = self.loader.get(PREFIX + "embed_tokens.weight").to(dtype)
        self.final_norm_w = self.loader.get(PREFIX + "norm.weight").to(dtype)
        self.lm_head = self.loader.get("lm_head.weight").to(dtype) if lm_head else None
        self.w = {i: load_layer(self.loader, self.cfg, i, dtype) for i in self.layer_ids}
        self.inv_freq, self.rope_att = yarn_inv_freq(self.cfg)
        self._rope_cache = {}

    # ---- interface
    def new_state(self, max_seq: int) -> State:
        return State(self.cfg, self.layer_ids, max_seq, self.dtype)

    def state_tensors(self, state: State, layer: int, length: int) -> dict:
        return {"kv_latent": state.kv[layer][:length].clone()}

    def load_state(self, state: State, layer: int, tensors: dict, length: int) -> None:
        state.kv[layer][:length] = tensors["kv_latent"][:length].to(state.kv[layer].dtype)

    def block_graph(self, layer: int) -> list[Step]:
        ffn = MOE_FFN if self.cfg.is_moe(layer) else DENSE_FFN
        return list(HC_ATTN + HC_FFN + ffn + FFN_RESIDUAL)

    def chunk_context(self, layer: int, start: int, length: int, state) -> Ctx:
        key = (start, length)
        if key not in self._rope_cache:
            cos, sin = rope_cos_sin(torch.arange(start, start + length), self.inv_freq, self.rope_att)
            self._rope_cache = {key: (cos, sin)}
        cos, sin = self._rope_cache[key]
        return Ctx(layer, start, length, state, extra={"cos": cos, "sin": sin})

    def component(self, layer: int, name: str):
        cfg, w, eps, n = self.cfg, self.w[layer], self.cfg.rms_norm_eps, self.cfg.hc_mult
        table = {
            "attn_hc": lambda ctx, x: hc_weights(
                x, w["attn_hc.hc_fn"], w["attn_hc.hc_base"], w["attn_hc.hc_scale"], cfg
            ),
            "attn_collapse": lambda ctx, x, hc: hc_collapse(x, hc, n),
            "attn_norm": lambda ctx, x: rms_norm(x, w["input_layernorm.weight"], eps),
            "q_a": lambda ctx, x: rms_norm(
                F.linear(x, w["self_attn.q_a_proj.weight"]), w["self_attn.q_a_layernorm.weight"], eps
            ),
            "attention": lambda ctx, x, qr: self._mla(layer, ctx, x, qr),
            "attn_residual": lambda ctx, x, hc, y: hc_residual(x, hc, y, n),
            "ffn_hc": lambda ctx, x: hc_weights(x, w["ffn_hc.hc_fn"], w["ffn_hc.hc_base"], w["ffn_hc.hc_scale"], cfg),
            "ffn_collapse": lambda ctx, x, hc: hc_collapse(x, hc, n),
            "ffn_norm": lambda ctx, x: rms_norm(x, w["post_attention_layernorm.weight"], eps),
            "ffn_residual": lambda ctx, x, hc, y: hc_residual(x, hc, y, n),
        }
        if cfg.is_moe(layer):
            p = "mlp.shared_experts."
            table.update(
                router=lambda ctx, x: route(x, w["mlp.gate.weight"], w["mlp.gate.e_score_correction_bias"], cfg),
                experts=lambda ctx, x, r: experts_forward(x, r, w["experts"]),
                shared_expert=lambda ctx, x: swiglu(
                    x, w[p + "gate_proj.weight"], w[p + "up_proj.weight"], w[p + "down_proj.weight"]
                ),
                moe_add=lambda ctx, a, b: a + b,
            )
        else:
            table["mlp"] = lambda ctx, x: swiglu(
                x, w["mlp.gate_proj.weight"], w["mlp.up_proj.weight"], w["mlp.down_proj.weight"]
            )
        return table[name]

    # ---- MLA
    def _mla(self, layer: int, ctx: Ctx, x: torch.Tensor, q_resid: torch.Tensor) -> torch.Tensor:
        cfg, w = self.cfg, self.w[layer]
        a = "self_attn."
        s, start = x.shape[0], ctx.start
        end = start + s
        nh, dn, r, dv, lat = (
            cfg.num_attention_heads,
            cfg.qk_nope_head_dim,
            cfg.qk_rope_head_dim,
            cfg.v_head_dim,
            cfg.kv_lora_rank,
        )
        cos, sin = ctx.extra["cos"], ctx.extra["sin"]
        q = F.linear(q_resid, w[a + "q_b_proj.weight"]).view(s, nh, dn + r)
        q = torch.cat([q[..., :dn], apply_rope(q[..., dn:], cos, sin)], dim=-1)  # [S, nh, 192]
        ckv = F.linear(x, w[a + "kv_a_proj_with_mqa.weight"])
        kv = torch.cat(
            [
                rms_norm(ckv[:, :lat], w[a + "kv_a_layernorm.weight"], cfg.rms_norm_eps),
                apply_rope(ckv[:, lat:], cos, sin),
            ],
            -1,
        )
        st = ctx.state.kv[layer]
        st[start:end] = kv
        cache = st[:end]
        kvb = expand_latent(cache[:, :lat], w[a + "kv_b_proj.weight"]).view(end, nh, dn + dv)
        k = torch.cat([kvb[..., :dn], cache[:, None, lat:].expand(end, nh, r)], dim=-1)  # [T, nh, 192]
        v = kvb[..., dn:]
        o = causal_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1).contiguous(), start, cfg.attn_scale
        )  # [S, nh, dv]
        return F.linear(o.reshape(s, nh * dv), w[a + "o_proj.weight"])

    # ---- forward
    @torch.no_grad()
    def forward_chunk(self, tokens, start, state, rec=noop, logits_last_n=0):
        """tokens [S] at absolute positions [start, start + S). Returns (final_norm [S, H], logits [n, V] | None)."""
        tokens = tokens.long()
        e = F.embedding(tokens, self.embed)
        rec("embed", e)
        h = e.unsqueeze(1).expand(-1, self.cfg.hc_mult, -1).reshape(-1, e.shape[-1])
        for i in self.layer_ids:
            ctx = self.chunk_context(i, start, tokens.shape[0], state)
            h = run_block(self.block_graph(i), lambda name, i=i: self.component(i, name), ctx, h, rec, prefix=f"L{i}.")
        mean = h.view(-1, self.cfg.hc_mult, h.shape[-1]).mean(dim=1)
        rec("hc_mean", mean)
        out = rms_norm(mean, self.final_norm_w, self.cfg.rms_norm_eps)
        rec("final_norm", out)
        logits = None
        if logits_last_n and self.lm_head is not None:
            logits = self.logits(out[-logits_last_n:])
            rec("logits", logits)
        return out, logits

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.lm_head)
