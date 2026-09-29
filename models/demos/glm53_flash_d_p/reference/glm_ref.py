# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone torch CPU reference for GLM-5.3-Flash (text decoder) chunked prefill.

Independent of HF transformers (the vendored HF glm5_next code in reference/hf is only the oracle for check_hf).
Loads the safetensors checkpoint directly (FP8 blocks dequantized as weights.py documents), keeps an explicit
per-layer state and runs each block through ``run_block``, so the block graphs below are the code. Sparse and
chunked where HF is dense: the DSA indexer scores the pooled-key cache, and attention gathers the selected latent
rows; KDA runs the chunked delta rule from the carried recurrent state.

Residual: mHC, 4 streams. Block input / output ``[S * 4, H]`` (token-major, stream-minor: row 4t + n is stream n of
token t; HF's layer output flattened); embed is broadcast to the 4 streams; the final head is the mean of the
streams, then norm and lm_head.

Block (all layers):
    in                                                     [S * 4, H] residual streams
    attn_hc       = mHC weights [S, 24] fp32 from in       [pre 4 | post 4 | comb 16 (row-major 4x4)]: flatten,
                                                           unweighted RMS (eps 1e-5), fp32 linear hc_attn_fn [24, 4H],
                                                           pre = sigmoid(. * s0 + b) + 1e-6, post = 2 sigmoid(. * s1
                                                           + b), comb = softmax(. * s2 + b) + 1e-6, 20 Sinkhorn steps
    attn_in       = sum_n pre[n] * in[:, n]                [S, H]
    attn_norm     = w * rms(attn_in)                       input_layernorm, eps 1e-5
  KDA (kda_dense, kda_moe):
    attn_out      = KDA(attn_norm)       [stateful]        q/k/v proj -> causal conv4 + SiLU (tail in state) ->
                                                           l2norm q, k; forget gate -5 sigmoid(exp(A_log) (f_b f_a x
                                                           + dt_bias)); beta = sigmoid(b_proj x); chunked delta rule
                                                           from the recurrent state; gated RMSNorm (g_b g_a x); o_proj
  DSA (dsa_moe):
    q_resid       = rms(q_a_proj(attn_norm))               q_a_layernorm, [S, 1536]
    topk          = indexer(attn_norm, q_resid) [stateful] int32 [S, 2051]: token ids of the top 512 pools (4 each,
                                                           -1 = none) + up to 3 tail tokens; pools by 4 from token 0,
                                                           pooled key = sum softmax(gate x + ape) * LayerNorm(wk x);
                                                           pool p selectable by query q iff 4p + 3 <= q; score =
                                                           sum_h w_h relu(q_h . k / sqrt(128)); pooled keys cached
    attn_out      = MLA(attn_norm, q_resid, topk) [stateful]   latent = rms(kv_a_proj x) cached [S, 512]; NoPE,
                                                           qk 256 / v 256, 64 heads, scale 256^-0.5, softmax over the
                                                           selected tokens only (absorbed kv_b); o_proj
    attn_residual = post * attn_out + comb^T @ in          [S * 4, H]
    ffn_hc, ffn_in, ffn_norm                               as above on attn_residual (post_attention_layernorm)
  dense (layers 0-2):
    mlp_out       = down(silu(min(g, 10)) * clamp(u, +-10))   intermediate 12288
  MoE:
    router        = dense routing [S, 288] fp32            sigmoid(fp32 logits), top-8 on scores + correction bias,
                                                           weights = chosen scores / sum, x 2.5
    experts_out   = sum_e router[:, e] * expert_e(ffn_norm)   clamped SwiGLU, intermediate 2048
    shared_out    = shared expert (clamped SwiGLU 2048)
    mlp_out       = experts_out + shared_out
    out           = post * mlp_out + comb^T @ attn_residual (ffn_residual)

State per layer (spec state.by_block_type):
    DSA: kv_latent [length, 512] (after kv_a_layernorm), index_key [length // 4, 128] (pooled keys, complete pools).
    KDA: kda_recurrent [64, 128, 128] fp32 and kda_conv [3, 3 * 8192] (the last 3 pre-conv q|k|v projections): the
    current ones (fixed size). A chunk starting at 0 resets them. Chunk starts must be multiples of 4 (pool
    alignment) and, for one-shot == chunked equality, of 64 (the KDA chunk size).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.demos.common.bringup.reference.interface import Ctx, Step, noop, run_block
from models.demos.glm53_flash_d_p.reference.weights import PREFIX, PackedExpert, WeightLoader


@dataclass
class GlmConfig:
    hidden_size: int = 4096
    vocab_size: int = 154880
    num_hidden_layers: int = 45
    rms_norm_eps: float = 1e-5
    # mHC
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    # KDA
    linear_num_heads: int = 64
    linear_head_dim: int = 128
    linear_conv_kernel: int = 4
    linear_lower_bound: float = -5.0
    kda_chunk: int = 64
    # DSA
    num_attention_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_head_dim: int = 256
    v_head_dim: int = 256
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_kpool: int = 4
    # MLP / MoE
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 288
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    swiglu_limit: float = 10.0
    layer_types: tuple = ()
    mlp_layer_types: tuple = ()

    @classmethod
    def from_json(cls, path: str) -> "GlmConfig":
        with open(path) as f:
            raw = json.load(f)["text_config"]
        la = raw["linear_attn_config"]
        cfg = cls(
            **{k: raw[k] for k in cls.__dataclass_fields__ if k in raw and k not in ("layer_types", "mlp_layer_types")}
        )
        cfg.linear_num_heads, cfg.linear_head_dim = la["num_heads"], la["head_dim"]
        cfg.linear_conv_kernel, cfg.linear_lower_bound = la["short_conv_kernel_size"], la["gate_lower_bound"]
        cfg.layer_types, cfg.mlp_layer_types = tuple(raw["layer_types"]), tuple(raw["mlp_layer_types"])
        cfg.qk_head_dim = raw["qk_nope_head_dim"] + raw["qk_rope_head_dim"]
        # what this reference implements; anything else in a new checkpoint must fail loudly
        assert raw["qk_rope_head_dim"] == 0 and raw["mla_use_nope"] and raw["mhc"]
        assert raw["n_group"] == 1 and raw["topk_group"] == 1 and raw["scoring_func"] == "sigmoid"
        assert raw["topk_method"] == "noaux_tc" and raw["hidden_act"] == "silu" and raw["n_shared_experts"] == 1
        assert raw["index_kpool_always_select_tail"] and raw["index_kpool_compress"]
        assert set(raw["indexer_types"]) == {"full"}, "shared indexer layers not implemented"
        assert not raw["attention_bias"] and not raw["tie_word_embeddings"]
        return cfg

    def is_kda(self, i: int) -> bool:
        return self.layer_types[i] == "linear_attention"

    def is_moe(self, i: int) -> bool:
        return self.mlp_layer_types[i] == "sparse"


# --------------------------------------------------------------------------------------
# Functional building blocks
# --------------------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """HF Glm5NextTextRMSNorm: normalize in fp32, cast back, then * w."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * y.to(x.dtype)


def clamped_swiglu(x, w_gate, w_up, w_down, limit: float):
    g = F.linear(x, w_gate).clamp(max=limit)
    u = F.linear(x, w_up).clamp(min=-limit, max=limit)
    return F.linear(F.silu(g) * u, w_down)


def hc_weights(x: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, cfg: GlmConfig):
    """mHC coefficients [S, (2 + N) * N] fp32 = [pre | post | comb (N x N, row-major, Sinkhorn-normalized)] from
    the streams x [S * N, H]."""
    n = cfg.hc_mult
    s = x.shape[0] // n
    flat = x.reshape(s, -1).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    mix = F.linear(flat, fn.float())
    pre_w, post_w, comb_w = mix.split([n, n, n * n], dim=-1)
    pre_b, post_b, comb_b = base.float().split([n, n, n * n])
    s_pre, s_post, s_comb = scale.float().unbind(0)
    pre = torch.sigmoid(pre_w * s_pre + pre_b) + cfg.hc_eps
    post = 2 * torch.sigmoid(post_w * s_post + post_b)
    comb = torch.softmax(comb_w.view(s, n, n) * s_comb + comb_b.view(n, n), dim=-1) + cfg.hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + cfg.hc_eps)
    for _ in range(cfg.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + cfg.hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + cfg.hc_eps)
    return torch.cat([pre, post, comb.reshape(s, n * n)], dim=-1)


def hc_collapse(x: torch.Tensor, hc: torch.Tensor, n: int) -> torch.Tensor:
    """sum_n pre[:, n] * x[:, n] -> [S, H] in x.dtype (x: [S * N, H])."""
    return (hc[:, :n].unsqueeze(-1) * x.view(-1, n, x.shape[-1])).sum(dim=1).to(x.dtype)


def hc_residual(x: torch.Tensor, hc: torch.Tensor, y: torch.Tensor, n: int) -> torch.Tensor:
    """post[:, n] * y + comb^T @ x -> [S * N, H] (HF: post * out + comb.transpose(-1, -2) @ residual)."""
    post = hc[:, n : 2 * n].to(x.dtype)
    comb = hc[:, 2 * n :].reshape(-1, n, n).to(x.dtype)
    out = post.unsqueeze(-1) * y.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), x.view(-1, n, x.shape[-1]))
    return out.reshape(-1, x.shape[-1])


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)


KDA_GROUP = 8  # sub-chunks whose intra-chunk terms are built at once (bounds the [H, G, C, C, K] temporaries)


def kda_chunked(q, k, v, g, beta, state: torch.Tensor, chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """The KDA delta rule over one prefill chunk, HF chunk_kimi_delta_attention's algorithm (chunk size 64).

    q, k [H, S, K] (l2-normed, q already scaled), v [H, S, V], g [H, S, K] (log decay), beta [H, S]; state [H, K, V]
    fp32, the recurrent state before this chunk. Sub-chunks start at this chunk's first token, so chunk starts that
    are multiples of ``chunk`` give the same sub-chunk grid as one-shot. Returns (out [H, S, V], new state)."""
    h, s, dk = k.shape
    pad = -s % chunk
    if pad:
        q, k, v, g = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v, g))
        beta = F.pad(beta, (0, pad))
    n = (s + pad) // chunk
    q, k, v, g = (t.reshape(h, n, chunk, t.shape[-1]) for t in (q, k, v, g))
    beta = beta.reshape(h, n, chunk)
    g = g.cumsum(dim=-2)
    k_beta, v_beta = k * beta.unsqueeze(-1), v * beta.unsqueeze(-1)
    incl = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool), diagonal=0)
    strict = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool), diagonal=1)
    eye = torch.eye(chunk, dtype=q.dtype)

    u = torch.empty_like(v)
    w = torch.empty_like(k)
    a_qk = q.new_empty(h, n, chunk, chunk)
    for c0 in range(0, n, KDA_GROUP):
        c1 = min(n, c0 + KDA_GROUP)
        gg = g[:, c0:c1]
        decay = (gg.unsqueeze(-2) - gg.unsqueeze(-3)).masked_fill(strict[..., None], float("-inf")).exp()
        attn = -(k_beta[:, c0:c1].unsqueeze(-2) * k[:, c0:c1].unsqueeze(-3) * decay).sum(dim=-1).masked_fill(incl, 0)
        for i in range(1, chunk):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + eye
        u[:, c0:c1] = attn @ v_beta[:, c0:c1]
        w[:, c0:c1] = attn @ (k_beta[:, c0:c1] * gg.exp())
        a_qk[:, c0:c1] = (
            (q[:, c0:c1].unsqueeze(-2) * k[:, c0:c1].unsqueeze(-3) * decay).sum(dim=-1).masked_fill(strict, 0)
        )
        del decay, attn

    out = torch.empty_like(v)
    st = state.to(q.dtype)
    for c in range(n):
        q_c, k_c, g_c = q[:, c], k[:, c], g[:, c]
        v_new = u[:, c] - w[:, c] @ st
        out[:, c] = (q_c * g_c.exp()) @ st + a_qk[:, c] @ v_new
        st = st * g_c[:, -1].exp().unsqueeze(-1) + (k_c * (g_c[:, -1:] - g_c).exp()).transpose(-1, -2) @ v_new
    return out.reshape(h, n * chunk, -1)[:, :s], st


def route(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, cfg: GlmConfig) -> torch.Tensor:
    """HF Glm5NextTextTopkRouter (noaux_tc, one group) as a dense routing matrix [S, E] fp32."""
    scores = F.linear(x.float(), gate_w.float()).sigmoid()
    ti = torch.topk(scores + bias.float()[None, :], k=cfg.num_experts_per_tok, dim=-1, sorted=False)[1]
    tw = scores.gather(1, ti)
    if cfg.norm_topk_prob:
        tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-20)
    tw = tw * cfg.routed_scaling_factor
    return torch.zeros_like(scores).scatter_(1, ti, tw)


EXPERT_ROW_BLOCK = 32  # per-expert GEMM rows padded to a multiple of this (known issue: MKL sgemm result depends on M)


def experts_forward(x: torch.Tensor, routing: torch.Tensor, expert_weights, limit: float) -> torch.Tensor:
    """sum over selected experts of routing[t, e] * expert_e(x[t]), tokens grouped per expert, accumulated in fp32
    in expert-id order (the HF order). ``expert_weights(e) -> (gate, up, down)``."""
    out = torch.zeros(x.shape, dtype=torch.float32)
    tok_all, exp_all = (routing != 0).nonzero(as_tuple=True)
    order = torch.argsort(exp_all, stable=True)
    tok_all, exp_all = tok_all[order], exp_all[order]
    experts, counts = torch.unique_consecutive(exp_all, return_counts=True)
    off = 0
    for e, n in zip(experts.tolist(), counts.tolist()):
        tok = tok_all[off : off + n]
        off += n
        xe = x[tok]
        pad = -n % EXPERT_ROW_BLOCK
        if pad:
            xe = torch.cat([xe, xe.new_zeros(pad, xe.shape[1])])
        wg, wu, wd = expert_weights(e)
        y = clamped_swiglu(xe, wg.to(x.dtype), wu.to(x.dtype), wd.to(x.dtype), limit)[:n]
        out.index_add_(0, tok, (y * routing[tok, e, None].to(y.dtype)).float())
    return out.to(x.dtype)


# --------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------


class LayerWeights(dict):
    """Name -> tensor for one layer (names as in the checkpoint below ``layers.{i}.``, convs fused)."""

    experts: list | None = None


def load_layer(loader: WeightLoader, cfg: GlmConfig, i: int, dtype=torch.float32, eager_experts=True) -> LayerWeights:
    p = f"{PREFIX}layers.{i}."
    names = [n[len(p) :] for n in loader.weight_map if n.startswith(p) and ".mlp.experts." not in n]
    w = LayerWeights()
    for n in names:
        if n.endswith("_scale_inv"):
            continue
        keep32 = n.endswith(("A_log", "dt_bias", "e_score_correction_bias", "_base", "_scale"))
        w[n] = loader.weight(p + n, torch.float32 if keep32 else dtype)
    if cfg.is_moe(i):
        w.experts = []
        for e in range(cfg.n_routed_experts):
            pe = PackedExpert(loader, i, e)
            w.experts.append(pe.weights(dtype) if eager_experts else pe)
    return w


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class State:
    """Per-layer prefill state: DSA kv_latent [max_seq, 512] + index_key [max_seq // 4, 128]; KDA recurrent
    [H, K, V] fp32 + conv tail [3, 3 * H * K] (current)."""

    def __init__(self, cfg: GlmConfig, layers: list[int], max_seq: int, dtype=torch.float32):
        self.max_seq = max_seq
        self.kv_latent, self.index_key, self.rec, self.conv = {}, {}, {}, {}
        for i in layers:
            if cfg.is_kda(i):
                hd = cfg.linear_num_heads * cfg.linear_head_dim
                self.rec[i] = torch.zeros(cfg.linear_num_heads, cfg.linear_head_dim, cfg.linear_head_dim)
                self.conv[i] = torch.zeros(cfg.linear_conv_kernel - 1, 3 * hd, dtype=dtype)
            else:
                self.kv_latent[i] = torch.zeros(max_seq, cfg.kv_lora_rank, dtype=dtype)
                self.index_key[i] = torch.zeros(max_seq // cfg.index_kpool, cfg.index_head_dim, dtype=dtype)


HC_ATTN = [
    Step("attn_hc", ("in",), "attn_hc", "op"),
    Step("attn_collapse", ("in", "attn_hc"), "attn_in", "op"),
    Step("attn_norm", ("attn_in",), "attn_norm", "norm"),
]
KDA_ATTN = [Step("attention", ("attn_norm",), "attn_out", "attention", stateful=True)]
DSA_ATTN = [
    Step("q_a", ("attn_norm",), "q_resid", "op"),
    Step("indexer", ("attn_norm", "q_resid"), "topk", "op", stateful=True),
    Step("attention", ("attn_norm", "q_resid", "topk"), "attn_out", "attention", stateful=True),
]
HC_FFN = [
    Step("attn_residual", ("in", "attn_hc", "attn_out"), "h_mid", "residual"),
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

# Up to this many MoE layers the experts are dequantized at load (fp32: 29 GB per layer); beyond it they stay FP8 and
# each expert is dequantized when it runs. Both give the same weights.
EAGER_EXPERT_LAYERS = 2
Q_BLOCK = 128  # query rows per DSA indexer / attention block


class GlmReference:
    """Implements models/demos/common/bringup/reference/interface.py. Keeps the requested layers resident."""

    def __init__(self, model_path: str, layers=None, dtype=torch.float32, lm_head: bool = True, eager_experts=None):
        self.cfg = GlmConfig.from_json(os.path.join(model_path, "config.json"))
        self.loader = WeightLoader(model_path)
        self.dtype = dtype
        self.layer_ids = list(range(self.cfg.num_hidden_layers)) if layers is None else list(layers)
        n_moe = sum(self.cfg.is_moe(i) for i in self.layer_ids)
        self.eager_experts = n_moe <= EAGER_EXPERT_LAYERS if eager_experts is None else eager_experts
        self.embed = self.loader.get(PREFIX + "embed_tokens.weight").to(dtype)
        self.final_norm_w = self.loader.get(PREFIX + "norm.weight").to(dtype)
        self.lm_head = self.loader.get("lm_head.weight").to(dtype) if lm_head else None
        self.w = {i: load_layer(self.loader, self.cfg, i, dtype, self.eager_experts) for i in self.layer_ids}
        for i in self.layer_ids:
            w = self.w[i]
            if self.cfg.is_kda(i):  # fused conv over [q | k | v] channels, fp32 as HF keeps it
                w["conv"] = torch.cat([w.pop(f"self_attn.{n}_conv1d.weight") for n in "qkv"])[:, 0].float()
            else:  # absorbed kv_b: [heads, qk + v, latent]
                w["kv_b"] = w.pop("self_attn.kv_b_proj.weight").view(
                    self.cfg.num_attention_heads, -1, self.cfg.kv_lora_rank
                )

    # ---- interface
    def new_state(self, max_seq: int) -> State:
        return State(self.cfg, self.layer_ids, max_seq, self.dtype)

    def state_tensors(self, state: State, layer: int, length: int) -> dict:
        if self.cfg.is_kda(layer):
            return {"kda_recurrent": state.rec[layer].clone(), "kda_conv": state.conv[layer].clone()}
        return {
            "kv_latent": state.kv_latent[layer][:length].clone(),
            "index_key": state.index_key[layer][: length // self.cfg.index_kpool].clone(),
        }

    def load_state(self, state: State, layer: int, tensors: dict, length: int) -> None:
        if self.cfg.is_kda(layer):
            state.rec[layer].copy_(tensors["kda_recurrent"])
            state.conv[layer].copy_(tensors["kda_conv"])
            return
        npool = length // self.cfg.index_kpool
        state.kv_latent[layer][:length] = tensors["kv_latent"][:length].to(state.kv_latent[layer].dtype)
        state.index_key[layer][:npool] = tensors["index_key"][:npool].to(state.index_key[layer].dtype)

    def block_graph(self, layer: int) -> list[Step]:
        attn = KDA_ATTN if self.cfg.is_kda(layer) else DSA_ATTN
        ffn = MOE_FFN if self.cfg.is_moe(layer) else DENSE_FFN
        return list(HC_ATTN + attn + HC_FFN + ffn + FFN_RESIDUAL)

    def chunk_context(self, layer: int, start: int, length: int, state) -> Ctx:
        return Ctx(layer, start, length, state)

    def expert_weights(self, layer: int):
        experts = self.w[layer].experts
        if self.eager_experts:
            return lambda e: experts[e]
        return lambda e: experts[e].weights(self.dtype)

    def component(self, layer: int, name: str):
        cfg, w, eps, n = self.cfg, self.w[layer], self.cfg.rms_norm_eps, self.cfg.hc_mult
        lim = cfg.swiglu_limit
        table = {
            "attn_hc": lambda ctx, x: hc_weights(x, w["hc_attn_fn"], w["hc_attn_base"], w["hc_attn_scale"], cfg),
            "attn_collapse": lambda ctx, x, hc: hc_collapse(x, hc, n),
            "attn_norm": lambda ctx, x: rms_norm(x, w["input_layernorm.weight"], eps),
            "attn_residual": lambda ctx, x, hc, y: hc_residual(x, hc, y, n),
            "ffn_hc": lambda ctx, x: hc_weights(x, w["hc_ffn_fn"], w["hc_ffn_base"], w["hc_ffn_scale"], cfg),
            "ffn_collapse": lambda ctx, x, hc: hc_collapse(x, hc, n),
            "ffn_norm": lambda ctx, x: rms_norm(x, w["post_attention_layernorm.weight"], eps),
            "ffn_residual": lambda ctx, x, hc, y: hc_residual(x, hc, y, n),
        }
        if cfg.is_kda(layer):
            table["attention"] = lambda ctx, x: self._kda(layer, ctx, x)
        else:
            table.update(
                q_a=lambda ctx, x: rms_norm(
                    F.linear(x, w["self_attn.q_a_proj.weight"]), w["self_attn.q_a_layernorm.weight"], eps
                ),
                indexer=lambda ctx, x, qr: self._indexer(layer, ctx, x, qr),
                attention=lambda ctx, x, qr, idx: self._mla(layer, ctx, x, qr, idx),
            )
        if cfg.is_moe(layer):
            p = "mlp.shared_experts."
            table.update(
                router=lambda ctx, x: route(x, w["mlp.gate.weight"], w["mlp.gate.e_score_correction_bias"], cfg),
                experts=lambda ctx, x, r: experts_forward(x, r, self.expert_weights(layer), lim),
                shared_expert=lambda ctx, x: clamped_swiglu(
                    x, w[p + "gate_proj.weight"], w[p + "up_proj.weight"], w[p + "down_proj.weight"], lim
                ),
                moe_add=lambda ctx, a, b: a + b,
            )
        else:
            table["mlp"] = lambda ctx, x: clamped_swiglu(
                x, w["mlp.gate_proj.weight"], w["mlp.up_proj.weight"], w["mlp.down_proj.weight"], lim
            )
        return table[name]

    # ---- KDA
    def _kda(self, layer: int, ctx: Ctx, x: torch.Tensor) -> torch.Tensor:
        cfg, w, st = self.cfg, self.w[layer], ctx.state
        s, h, d = x.shape[0], cfg.linear_num_heads, cfg.linear_head_dim
        a = "self_attn."
        if ctx.start == 0:
            st.rec[layer].zero_()
            st.conv[layer].zero_()
        qkv = torch.cat([F.linear(x, w[a + f"{n}_proj.weight"]) for n in "qkv"], dim=-1)  # [S, 3 H D] pre-conv
        window = torch.cat([st.conv[layer].to(x.dtype), qkv]).float()  # [K - 1 + S, C]
        conv = F.silu(F.conv1d(window.t().unsqueeze(0), w["conv"].unsqueeze(1), groups=window.shape[1])[0].t())
        st.conv[layer].copy_(window[-(cfg.linear_conv_kernel - 1) :])
        conv = conv.to(x.dtype)
        q, k, v = (t.reshape(s, h, d).transpose(0, 1).float() for t in conv.split(h * d, dim=-1))
        raw_g = F.linear(F.linear(x, w[a + "f_a_proj.weight"]), w[a + "f_b_proj.weight"]).float()
        raw_g = (raw_g + w[a + "dt_bias"].float()).view(s, h, d)
        g = cfg.linear_lower_bound * torch.sigmoid(w[a + "A_log"].float().exp().view(1, h, 1) * raw_g)
        beta = torch.sigmoid(F.linear(x, w[a + "b_proj.weight"])).float()
        q = l2norm(q) * d**-0.5
        k = l2norm(k)
        o, st.rec[layer] = kda_chunked(q, k, v, g.transpose(0, 1), beta.t(), st.rec[layer], cfg.kda_chunk)
        o = o.transpose(0, 1)  # [S, H, D] fp32
        gate = F.linear(F.linear(x, w[a + "g_a_proj.weight"]), w[a + "g_b_proj.weight"]).view(s, h, d)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
        o = (w[a + "o_norm.weight"].float() * o * torch.sigmoid(gate.float())).to(x.dtype)
        return F.linear(o.reshape(s, h * d), w[a + "o_proj.weight"])

    # ---- DSA
    def _indexer(self, layer: int, ctx: Ctx, x: torch.Tensor, q_resid: torch.Tensor) -> torch.Tensor:
        cfg, w, st = self.cfg, self.w[layer], ctx.state
        a = "self_attn.indexer."
        s, start, kp = x.shape[0], ctx.start, cfg.index_kpool
        assert start % kp == 0, f"chunk start {start} is not pool-aligned ({kp})"
        nh, hd = cfg.index_n_heads, cfg.index_head_dim
        q = F.linear(q_resid, w[a + "wq_b.weight"]).view(s, nh, hd)
        k = F.layer_norm(F.linear(x, w[a + "wk.weight"]), (hd,), w[a + "k_norm.weight"], w[a + "k_norm.bias"], 1e-6)
        gate = F.linear(x, w[a + "index_kpool_compress_gate"])
        weights = F.linear(x, w[a + "weights_proj.weight"]).float() * nh**-0.5  # [S, nh]
        # pools completed inside this chunk (start is pool-aligned)
        m = s // kp
        if m:
            logits = gate[: m * kp].view(m, kp, hd).float() + w[a + "index_kpool_compress_ape"].float()[None]
            prob = logits.softmax(dim=1).to(k.dtype)
            p0 = start // kp
            st.index_key[layer][p0 : p0 + m] = (prob * k[: m * kp].view(m, kp, hd)).sum(dim=1)
        n_end = (start + s) // kp  # complete pools after this chunk
        pk = st.index_key[layer][:n_end].float()
        sel_pools = cfg.index_topk // kp
        out = torch.full((s, cfg.index_topk + kp - 1), -1, dtype=torch.int32)
        offs = torch.arange(kp, dtype=torch.int64)
        for t0 in range(0, s, Q_BLOCK):
            t1 = min(s, t0 + Q_BLOCK)
            qpos = torch.arange(start + t0, start + t1)
            n_vis = (qpos + 1) // kp  # pools with end <= q
            p_blk = int(n_vis[-1])
            if p_blk:
                sc = torch.matmul(q[t0:t1].float(), pk[:p_blk].t())  # [B, nh, P]
                sc = F.relu(sc * hd**-0.5)
                idx_scores = torch.matmul(weights[t0:t1].unsqueeze(-2), sc).squeeze(-2)  # [B, P]
                valid = torch.arange(p_blk)[None, :] < n_vis[:, None]
                idx_scores = idx_scores.masked_fill(~valid, torch.finfo(idx_scores.dtype).min)
                ksel = min(sel_pools, p_blk)
                sel = idx_scores.topk(ksel, dim=-1).indices
                sel_valid = valid.gather(-1, sel)
                tok = (sel[..., None] * kp + offs).masked_fill(~sel_valid[..., None], -1).flatten(-2)
                out[t0:t1, : ksel * kp] = tok.to(torch.int32)
            tail_n = (qpos + 1) % kp
            tail = (qpos + 1 - tail_n)[:, None] + offs[None, : kp - 1]
            tail = tail.masked_fill(offs[None, : kp - 1] >= tail_n[:, None], -1)
            out[t0:t1, cfg.index_topk :] = tail.to(torch.int32)
        return out

    def _mla(self, layer: int, ctx: Ctx, x: torch.Tensor, q_resid: torch.Tensor, topk: torch.Tensor) -> torch.Tensor:
        cfg, w, st = self.cfg, self.w[layer], ctx.state
        a = "self_attn."
        s, start = x.shape[0], ctx.start
        nh, dqk, dv, r = cfg.num_attention_heads, cfg.qk_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
        kv = rms_norm(F.linear(x, w[a + "kv_a_proj_with_mqa.weight"]), w[a + "kv_a_layernorm.weight"], cfg.rms_norm_eps)
        st.kv_latent[layer][start : start + s] = kv
        lat_all = st.kv_latent[layer][: start + s]
        q = F.linear(q_resid, w[a + "q_b_proj.weight"]).view(s, nh, dqk)
        w_uk, w_uv = w["kv_b"][:, :dqk], w["kv_b"][:, dqk:]  # [nh, dqk, r], [nh, dv, r]
        q_lat = torch.einsum("shd,hdr->shr", q, w_uk)  # [S, nh, r]
        out = x.new_empty(s, nh, dv)
        scale = dqk**-0.5
        for t0 in range(0, s, Q_BLOCK):
            t1 = min(s, t0 + Q_BLOCK)
            idx = topk[t0:t1].long()
            valid = idx >= 0
            lat = lat_all[idx.clamp(min=0)]  # [B, W, r]
            sc = torch.matmul(q_lat[t0:t1], lat.transpose(-1, -2)) * scale  # [B, nh, W]
            sc = sc.masked_fill(~valid[:, None, :], float("-inf"))
            p = F.softmax(sc, dim=-1, dtype=torch.float32).to(x.dtype)
            o_lat = torch.matmul(p, lat)  # [B, nh, r]
            out[t0:t1] = torch.einsum("bhr,hdr->bhd", o_lat, w_uv)
        return F.linear(out.reshape(s, nh * dv), w[a + "o_proj.weight"])

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
        out = rms_norm(h.view(-1, self.cfg.hc_mult, h.shape[-1]).mean(dim=1), self.final_norm_w, self.cfg.rms_norm_eps)
        rec("final_norm", out)
        logits = None
        if logits_last_n and self.lm_head is not None:
            logits = self.logits(out[-logits_last_n:])
            rec("logits", logits)
        return out, logits

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.lm_head)
