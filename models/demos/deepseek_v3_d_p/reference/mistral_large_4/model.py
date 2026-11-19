# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU reference of Mistral Large 4 (1T-A52B, text decoder) for chunked prefill.

Implements the interface in _bringup_interface.py, independent of transformers (HF is only the oracle,
hf_oracle.py). Weights come straight from the consolidated FP8 checkpoint (weights.py) and are
held in ``weight_dtype(spec)`` (bf16: an FP8 block times its power-of-two scale is exact in bf16); compute runs in
``dtype`` (fp32), each weight cast in the op that uses it. Routed experts stay in the checkpoint's FP8 (the
ExpertStore: 1 TB in bf16 would not fit) and are dequantized straight into the compute dtype for the GEMM that uses
them. Every block runs through ``run_block``, so the block graph below is the code.

Model (72 layers, hidden 6144, 64 heads, vocab 204800, upstream modeling_mistral_large4.py):
    h = embed(tokens); per layer: h = block(h); final_norm = RMSNorm(h); logits = 30 * tanh(lm_head(final_norm) / 30)
Block (layer 0 dense FFN, 1-71 MoE; every 4th layer NoPE; "full" layers run their own DSA indexer, "shared" layers
17-18, 21-22, ... reuse the latest full layer's top-k of the same chunk; layers 39, 43, ..., 71 read layer 35's KV):
    in           [S, H]
    attn_norm    = RMSNorm(in)                                         attention_norm (eps 1e-5)
  full:
    topk         = indexer(attn_norm)                [stateful]        [S, 2048] int32 key positions, ascending, -1
                                                                       padded (a row p < 2048 keeps its p + 1 keys)
  shared:
    topk         = the latest full layer's topk of this chunk (ctx.extra["shared_topk"] when set)
    attn_out     = attention(attn_norm, topk)        [stateful]        sparse MLA over the latent cache, o_proj
    h_mid        = in + attn_out
    ffn_norm     = RMSNorm(h_mid)                                      ffn_norm
  dense (layer 0):
    mlp_out      = w2(silu(w1 x) * w3 x)                               intermediate 24576
  MoE:
    router       = dense routing weights [S, 128] fp32                 softmax over 128 (fp32 logits), top-4,
                                                                       renormalized (x routed_scaling_factor 1.0)
    experts_out  = sum_e router[:, e] * expert_e(ffn_norm)             w2(silu(w1 x) * w3 x), intermediate 6144
    shared_out   = w2(silu(w1 x) * w3 x)                               1 shared expert, intermediate 6144
    mlp_out      = experts_out + shared_out
    out          = h_mid + mlp_out

Attention (MLA, no q-LoRA; 64 heads; qk 64 NoPE + 64 RoPE (NoPE layers: 128 + 0); v 128; kv_lora 512):
    q = wq(x) [S, 64, 128] -> q_nope | q_rot (RoPE); q *= ssmax_h(p) = s_h * log(p + 1) + b_h  (softmax scaling)
    wkv_a_with_mqa(x) [S, 512 + r] -> latent (kv_a_norm, RMSNorm eps 1e-5) | k_rot (RoPE, one head)
    absorbed: q_lat_h = q_nope_h @ W_uk_h (per head nope -> 512, W_uk = wkv_b's k rows); score = [q_lat | q_rot] .
              [latent | k_rot]_t * 128^-0.5 over the selected keys t; softmax; out_lat = p @ latent; o_h = W_uv_h
              out_lat (512 -> 128); attn_out = wo(o)
    Shared-KV layers (kv_source_layers[i] != i) have no wkv_a / kv_a_norm: they read the source layer's latent and
    rotated k_rot (ctx.extra["shared_kv"] when set) with their own wq, wkv_b, wo.
    RoPE: YaRN (factor 2048 over 512, beta 32 / 1, no magnitude scaling) with LeRoPE (inverse frequencies x
    exp(lerope.alpha)); interleaved: pairs (x[2i], x[2i+1]) rotated by frequency i, kept in place (checkpoint order;
    HF's apply_rotary_pos_emb_interleave writes cat(evens, odds), the same scores).
Indexer (32 heads x 128): q = ssmax'_h(p) * wq_b(LayerNorm(x)) (LayerNorm eps 1e-6 with bias; its own ssmax);
    k = LayerNorm(wk(x)) (eps 1e-6, with bias); RoPE on the first 64 of the 128 dims (rope layers only);
    w = weights_proj(x) * 32^-0.5; score[s, t] = sum_h w[s, h] * relu(q[s, h] . k[t] * 128^-0.5) over t <= s;
    the top 2048 (HF: top-k over the causally masked scores; a row with <= 2048 visible keys selects all of them).
State per layer: kv_latent [max_seq, 512] (normed latent) and k_rope [max_seq, r] (RoPE'd, interleaved), [*, 0] on
    NoPE layers and both [*, 0] on shared-KV layers; index_key [max_seq, 128] (LayerNorm'd, RoPE'd on rope layers) on
    full-indexer layers, [*, 0] on shared-index layers. A chunk's indexer scores every earlier key, so chunked
    prefill needs the prefix's index keys: the indexer refuses a prefix whose index keys were not loaded rather than
    score zeros.

Precision: weights in ``weight_dtype``, activations and state in ``dtype`` (fp32 for the gates); norm variance,
softmax, router logits and RoPE tables fp32 as upstream. Chunked == one-shot: attention / indexer row blocks are
aligned to absolute positions and expert token groups are zero-padded to 32 rows (MKL GEMM kernels depend on M).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.mistral_large_4._bringup_interface import Ctx, Step, as_compute, noop, run_block, weight_dtype
from models.demos.deepseek_v3_d_p.reference.mistral_large_4.weights import checkpoint, expert_store, keep_heap, load_dense

ROW_BLOCK = 64  # attention / indexer query rows per block, aligned to absolute positions
EXPERT_ROW_BLOCK = 32  # per-expert GEMM rows are padded to a multiple of this (see experts_forward)
INDEX_NORM_EPS = 1e-6  # indexer q_input_norm and k_norm (nn.LayerNorm(eps=1e-6) upstream; params.json dsa.norm_eps)


class Config:
    """The checkpoint's text_config (config.json) and what upstream derives from it."""

    def __init__(self, path):
        tc = json.loads((Path(path) / "config.json").read_text())["text_config"]
        self.raw = tc
        self.hidden = tc["hidden_size"]
        self.n_layers = tc["num_hidden_layers"]
        self.n_heads = tc["num_attention_heads"]
        self.kv_lora = tc["kv_lora_rank"]
        self.qk_nope = tc["qk_nope_head_dim"]
        self.qk_rope = tc["qk_rope_head_dim"]
        self.v_head = tc["v_head_dim"]
        self.qk_head = self.qk_nope + self.qk_rope
        self.index_heads = tc["index_n_heads"]
        self.index_dim = tc["index_head_dim"]
        self.index_topk = tc["index_topk"]
        self.n_experts = tc["n_routed_experts"]
        self.top_k = tc["num_experts_per_tok"]
        self.routed_scale = float(tc["routed_scaling_factor"])
        self.eps = tc["rms_norm_eps"]
        self.vocab = tc["vocab_size"]
        self.softcap = tc["final_logit_softcapping"]
        self.no_rope_layers = tc["no_rope_layers"]
        self.kv_source_layers = tc["kv_source_layers"]
        self.indexer_types = tc["indexer_types"]
        self.mlp_layer_types = tc["mlp_layer_types"]
        self.rope = tc["rope_parameters"]
        assert tc["hidden_act"] == "silu" and not tc.get("attention_bias", False)
        assert tc["n_group"] == 1 and tc["topk_group"] == 1, "group-limited routing is not implemented"
        assert tc["n_shared_experts"] == 1 and self.rope["rope_type"] == "yarn"
        # YaRN without magnitude scaling: cos / sin unscaled, softmax scale qk_head^-0.5 (no mscale_all_dim)
        assert float(self.rope.get("attention_factor", 1.0)) == 1.0 and not self.rope.get("mscale_all_dim")
        assert self.index_dim - self.qk_rope >= 0

    def use_rope(self, i: int) -> bool:
        return bool(self.no_rope_layers[i])

    def rope_dim(self, i: int) -> int:
        return self.qk_rope if self.use_rope(i) else 0

    def nope_dim(self, i: int) -> int:
        return self.qk_head - self.rope_dim(i)

    def is_moe(self, i: int) -> bool:
        return self.mlp_layer_types[i] == "sparse"

    def is_full(self, i: int) -> bool:
        return self.indexer_types[i] == "full"

    def kv_source(self, i: int) -> int:
        return self.kv_source_layers[i]

    def owns_kv(self, i: int) -> bool:
        return self.kv_source_layers[i] == i

    def kv_shared_later(self, i: int) -> bool:
        return any(src == i for src in self.kv_source_layers[i + 1 :])

    def topk_source(self, i: int) -> int:
        """The layer whose indexer layer i uses: itself on full layers, the latest full layer before it otherwise."""
        return max(j for j in range(i + 1) if self.is_full(j))


# --------------------------------------------------------------------------------------------- building blocks
def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """MistralLarge4RMSNorm: w * (x * rsqrt(mean(x^2) + eps)), the normalization in fp32, cast back before * w."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return as_compute(w, x) * y.to(x.dtype)


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), as_compute(w, x), as_compute(b, x), eps)


def yarn_inv_freq(rope: dict, dim: int) -> torch.Tensor:
    """YaRN inverse frequencies (transformers' _compute_yarn_parameters, truncate=True), fp32 [dim / 2]."""
    base, factor = float(rope["rope_theta"]), float(rope["factor"])
    orig = rope["original_max_position_embeddings"]
    beta_fast, beta_slow = rope.get("beta_fast") or 32, rope.get("beta_slow") or 1

    def corr_dim(rot):
        return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))

    low, high = math.floor(corr_dim(beta_fast)), math.ceil(corr_dim(beta_slow))
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    pos_freqs = base ** (torch.arange(0, dim, 2).to(dtype=torch.float) / dim)
    extra, inter = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    extra_factor = 1 - ramp
    return inter * (1 - extra_factor) + extra * extra_factor


def rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [S, ..., R], cos / sin [S, R / 2] (one angle per pair): pairs (x[2i], x[2i+1]) rotated, in place order."""
    shape = (cos.shape[0],) + (1,) * (x.dim() - 2) + (cos.shape[1],)
    c, s = cos.view(shape).to(x.dtype), sin.view(shape).to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).flatten(-2)


def softmax_scaling(log_pos: torch.Tensor, s: torch.Tensor, b: torch.Tensor, dtype) -> torch.Tensor:
    """[S, heads] = s_h * log(p + 1) + b_h (MistralLarge4TextSoftmaxScaling, in the query dtype)."""
    return s.to(dtype)[None, :] * log_pos.to(dtype)[:, None] + b.to(dtype)[None, :]


def _row_blocks(start: int, length: int, block: int = ROW_BLOCK):
    """Local [a, b) row ranges whose absolute positions are aligned to ``block``."""
    a = 0
    while a < length:
        b = min(length, (start + a) // block * block + block - start)
        yield a, b
        a = b


def indexer_topk(q, k_all, w, start: int, topk: int, scale: float) -> torch.Tensor:
    """Top-k key positions per query for queries at [start, start + S), ascending, -1 padded to ``topk``, int32.

    q [S, Hi, D] (post-RoPE, softmax-scaled), k_all [>= start + S, D] (index keys, prefix + chunk), w [S, Hi] (head
    weights x Hi^-0.5). score[s, t] = sum_h w[s, h] * relu(q[s, h] . k[t] * scale) for t <= start + s
    (MistralLarge4TextIndexer, fp32). A row that sees at most ``topk`` keys keeps all of them."""
    s_len = q.shape[0]
    out = torch.full((s_len, topk), -1, dtype=torch.int32)
    ar = torch.arange(topk, dtype=torch.int32)
    for a, b in _row_blocks(start, s_len):
        pos = torch.arange(start + a, start + b, dtype=torch.int32)
        end = start + b
        if end <= topk:
            out[a:b] = torch.where(ar[None, :] <= pos[:, None], ar[None, :], -1)
            continue
        sc = torch.matmul(q[a:b].float(), k_all[:end].float().t()) * scale  # [n, Hi, end]
        sc = torch.matmul(w[a:b].float().unsqueeze(-2), F.relu(sc)).squeeze(-2)  # [n, end]
        sc = sc.masked_fill(torch.arange(end)[None, :] > pos[:, None], float("-inf"))
        idx = sc.topk(topk, dim=-1).indices.sort(dim=-1).values.to(torch.int32)
        small = pos + 1 <= topk  # these rows keep every key they see
        if small.any():
            idx[small] = torch.where(ar[None, :] <= pos[small, None], ar[None, :], -1)
        out[a:b] = idx
    return out


def sparse_mla(q: torch.Tensor, kv_all: torch.Tensor, idx: torch.Tensor, scale: float, start: int, lat: int):
    """Absorbed sparse MLA. q [S, Hq, lat + r], kv_all [T, lat + r], idx [S, K] (-1 = no key). Softmax over the
    selected keys. Returns [S, Hq, lat]."""
    s_len, hq, _ = q.shape
    out = q.new_empty(s_len, hq, lat)
    for a, b in _row_blocks(start, s_len):
        ii = idx[a:b].long()
        valid = ii >= 0
        g = kv_all[ii.clamp(min=0)]  # [n, K, lat + r]
        sc = torch.bmm(q[a:b], g.transpose(1, 2)) * scale  # [n, Hq, K]
        sc = sc.masked_fill(~valid[:, None, :], float("-inf"))
        p = F.softmax(sc, dim=-1, dtype=torch.float32).to(q.dtype)
        out[a:b] = torch.bmm(p, g[..., :lat])
    return out


def swiglu(x, w1, w3, w2):
    return F.linear(F.silu(F.linear(x, as_compute(w1, x))) * F.linear(x, as_compute(w3, x)), as_compute(w2, x))


def route(x: torch.Tensor, gate_w: torch.Tensor, cfg: Config) -> torch.Tensor:
    """MistralLarge4TextTopkRouter (softmax, 1 group, top-k renormalized) as a dense [S, E] fp32 routing matrix: the
    top-k weight at each selected expert, 0 elsewhere (softmax weights are > 0)."""
    logits = F.linear(x.float(), gate_w.float())
    scores = logits.softmax(-1)
    ti = torch.topk(scores, k=cfg.top_k, dim=-1, sorted=False)[1]
    tw = scores.gather(1, ti)
    tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-20)
    tw = tw * cfg.routed_scale
    return torch.zeros_like(scores).scatter_(1, ti, tw)


def experts_forward(x: torch.Tensor, routing: torch.Tensor, layer_experts) -> torch.Tensor:
    """sum over selected experts of routing[t, e] * expert_e(x[t]), accumulated in fp32 in expert-id order (the HF
    order), returned in x.dtype; expert_e = w2(silu(w1 x) * w3 x), its FP8 weights dequantized into x.dtype here.

    Tokens are grouped per expert and each group is zero-padded to a multiple of EXPERT_ROW_BLOCK rows: MKL sgemm
    picks M-dependent kernels for small or odd M, so without padding a token's expert output would depend on how many
    other tokens chose that expert, and chunked prefill would stop matching one-shot."""
    out = torch.zeros(x.shape, dtype=torch.float32)
    tok_all, exp_all = (routing != 0).nonzero(as_tuple=True)
    if tok_all.numel() == 0:
        return out.to(x.dtype)
    order = torch.argsort(exp_all, stable=True)
    tok_all, exp_all = tok_all[order], exp_all[order]
    experts, counts = torch.unique_consecutive(exp_all, return_counts=True)
    w13 = layer_experts.w[(int(experts[0]), "w1")]
    inter, hidden = w13.shape
    gu = torch.empty(2 * inter, hidden, dtype=x.dtype)  # [w1; w3], as upstream's gate_up_proj
    down = torch.empty(hidden, inter, dtype=x.dtype)
    off = 0
    for e, n in zip(experts.tolist(), counts.tolist()):
        tok = tok_all[off : off + n]
        off += n
        xe = x[tok]
        pad = -n % EXPERT_ROW_BLOCK
        if pad:
            xe = torch.cat([xe, xe.new_zeros(pad, xe.shape[1])])
        layer_experts.weight(e, "w1", x.dtype, gu[:inter])
        layer_experts.weight(e, "w3", x.dtype, gu[inter:])
        layer_experts.weight(e, "w2", x.dtype, down)
        gate, up = F.linear(xe, gu).chunk(2, dim=-1)
        y = F.linear(F.silu(gate) * up, down)[:n]
        out.index_add_(0, tok, (y * routing[tok, e, None].to(y.dtype)).float())
    return out.to(x.dtype)


# --------------------------------------------------------------------------------------------- weights
def layer_names(cfg: Config, i: int) -> dict[str, str]:
    """Dense (non-routed-expert) weights of layer i: key -> checkpoint name."""
    p = f"layers.{i}."
    m = {
        "attn_norm": p + "attention_norm.weight",
        "ffn_norm": p + "ffn_norm.weight",
        "wq": p + "attention.wq.weight",
        "wkv_b": p + "attention.wkv_b.weight",
        "wo": p + "attention.wo.weight",
        "ssmax_s": p + "attention.ssmax_scaling.s",
        "ssmax_b": p + "attention.ssmax_scaling.b",
    }
    if cfg.owns_kv(i):
        m["wkv_a"] = p + "attention.wkv_a_with_mqa.weight"
        m["kv_a_norm"] = p + "attention.kv_a_norm.weight"
    if cfg.is_full(i):
        q = p + "attention.indexer."
        m.update(
            idx_wq_b=q + "wq_b.weight",
            idx_wk=q + "wk.weight",
            idx_k_norm_w=q + "k_norm.weight",
            idx_k_norm_b=q + "k_norm.bias",
            idx_weights_proj=q + "weights_proj.weight",
            idx_q_norm_w=q + "q_input_norm.weight",
            idx_q_norm_b=q + "q_input_norm.bias",
            idx_ssmax_s=q + "ssmax_scaling.s",
            idx_ssmax_b=q + "ssmax_scaling.b",
        )
    if cfg.is_moe(i):
        m["gate"] = p + "gate.weight"
        m.update({f"shared_{w}": p + f"shared_experts.{w}.weight" for w in ("w1", "w2", "w3")})
    else:
        m.update({w: p + f"feed_forward.{w}.weight" for w in ("w1", "w2", "w3")})
    return m


class State:
    """Per layer: kv [max_seq, 512] and kr [max_seq, r] (layers that own their KV), ik [max_seq, 128] (full-indexer
    layers). ik_len: how many positions of each layer's index keys are valid (a missing index-key prefix is caught
    instead of read as zeros)."""

    def __init__(self, cfg: Config, layers: list[int], max_seq: int, dtype):
        self.max_seq = max_seq
        self.kv = {i: torch.zeros(max_seq, cfg.kv_lora, dtype=dtype) for i in layers if cfg.owns_kv(i)}
        self.kr = {i: torch.zeros(max_seq, cfg.rope_dim(i), dtype=dtype) for i in layers if cfg.owns_kv(i)}
        self.ik = {i: torch.zeros(max_seq, cfg.index_dim, dtype=dtype) for i in layers if cfg.is_full(i)}
        self.ik_len = {i: 0 for i in self.ik}


class Reference:
    def __init__(self, spec, layers=None, dtype=torch.float32):
        self.spec = spec
        self.dtype = dtype or torch.float32
        self.wdtype = weight_dtype(spec)
        keep_heap()
        self.ckpt = checkpoint(spec.hf_dir)
        self.cfg = cfg = Config(spec.hf_dir)
        self.layer_ids = list(range(cfg.n_layers)) if layers is None else list(layers)
        self.experts = expert_store(self.ckpt, cfg.n_experts)
        names = {"embed": "tok_embeddings.weight", "final_norm": "norm.weight", "lerope_alpha": "lerope.alpha"}
        for i in self.layer_ids:
            names.update({f"{i}.{k}": v for k, v in layer_names(cfg, i).items()})
        got = load_dense(self.ckpt, names, self.wdtype)
        self.embed, self.final_norm_w = got.pop("embed"), got.pop("final_norm")
        alpha = got.pop("lerope_alpha").float()
        self.w = {i: {} for i in self.layer_ids}
        for k, t in got.items():
            i, name = k.split(".", 1)
            self.w[int(i)][name] = t
        for i in self.layer_ids:
            w, nd = self.w[i], cfg.nope_dim(i)
            kv_b = w["wkv_b"].view(cfg.n_heads, nd + cfg.v_head, cfg.kv_lora)
            w["w_uk"], w["w_uv"] = kv_b[:, :nd], kv_b[:, nd:]  # views of the stored weight
        moe = [i for i in self.layer_ids if cfg.is_moe(i)]
        self.next_moe = dict(zip(moe, moe[1:]))
        self._lm_head = None
        self.inv_freq = yarn_inv_freq(cfg.rope, cfg.qk_rope) * alpha.exp()  # LeRoPE, fp32
        self.scale = cfg.qk_head**-0.5  # no YaRN mscale (rope_parameters has no mscale_all_dim)
        self._cos = self._sin = None
        # Cross-layer values of the latest chunk, kept for the layers that read them: ("topk", src) -> (start,
        # length, topk) from a full layer's indexer; ("kv", src) -> (start, length, latent, k_rot) from a KV source.
        self._xl = {}

    # ---- RoPE tables: fp32, built once for the longest sequence seen, sliced per chunk
    def _rope(self, end: int):
        if self._cos is None or self._cos.shape[0] < end:
            n = max(end, int(self.spec.get("shapes.max_seq_len") or 0))
            freqs = torch.arange(n)[:, None].float() * self.inv_freq[None, :].float()
            self._cos, self._sin = freqs.cos(), freqs.sin()
        return self._cos, self._sin

    @property
    def lm_head(self):
        if self._lm_head is None:
            self._lm_head = load_dense(self.ckpt, {"w": "output.weight"}, self.wdtype)["w"]
        return self._lm_head

    # ---- state
    def new_state(self, max_seq):
        return State(self.cfg, self.layer_ids, max_seq, self.dtype)

    def state_tensors(self, state, layer, length):
        """kv_latent, k_rope and index_key for every layer; a layer without one of them (a shared-KV layer reads its
        source's cache, a NoPE layer has no rotated key, a shared-index layer no indexer) returns it [length, 0]."""
        none = torch.zeros(length, 0, dtype=self.dtype)
        own = self.cfg.owns_kv(layer)
        return {
            "kv_latent": state.kv[layer][:length].clone() if own else none,
            "k_rope": state.kr[layer][:length].clone() if own else none.clone(),
            "index_key": state.ik[layer][:length].clone() if layer in state.ik else none.clone(),
        }

    def load_state(self, state, layer, tensors, length):
        if self.cfg.owns_kv(layer):
            state.kv[layer][:length] = tensors["kv_latent"][:length].to(self.dtype)
            state.kr[layer][:length] = tensors["k_rope"][:length].to(self.dtype)
        if layer in state.ik:
            if "index_key" in tensors:
                state.ik[layer][:length] = tensors["index_key"][:length].to(self.dtype)
                state.ik_len[layer] = length
            else:
                state.ik_len[layer] = 0

    # ---- block graph
    def block_graph(self, layer):
        cfg = self.cfg
        steps = [
            Step("attn_norm", ("in",), "attn_norm", "norm"),
            Step("indexer", ("attn_norm",), "topk", "indexer", stateful=True)
            if cfg.is_full(layer)
            else Step("topk_shared", ("attn_norm",), "topk", "topk"),
            Step("attention", ("attn_norm", "topk"), "attn_out", "attention", stateful=True),
            Step("attn_residual", ("in", "attn_out"), "h_mid", "residual"),
            Step("ffn_norm", ("h_mid",), "ffn_norm", "norm"),
        ]
        if cfg.is_moe(layer):
            steps += [
                Step("router", ("ffn_norm",), "router", "router"),
                Step("experts", ("ffn_norm", "router"), "experts_out", "moe"),
                Step("shared_expert", ("ffn_norm",), "shared_out", "mlp"),
                Step("moe_combine", ("experts_out", "shared_out"), "mlp_out", "residual"),
            ]
        else:
            steps.append(Step("mlp", ("ffn_norm",), "mlp_out", "mlp"))
        steps.append(Step("mlp_residual", ("h_mid", "mlp_out"), "out", "residual"))
        return steps

    def chunk_context(self, layer, start, length, state):
        cos, sin = self._rope(start + length)
        pos = torch.arange(start, start + length)
        return Ctx(
            layer,
            start,
            length,
            state,
            extra={
                "cos": cos[start : start + length],
                "sin": sin[start : start + length],
                "log_pos": (pos + 1).float().log(),
            },
        )

    def component(self, layer, name):
        cfg, w, eps = self.cfg, self.w[layer], self.cfg.eps
        table = {
            "attn_norm": lambda ctx, x: rms_norm(x, w["attn_norm"], eps),
            "indexer": lambda ctx, x: self._indexer(layer, ctx, x),
            "topk_shared": lambda ctx, x: self._shared_topk(layer, ctx),
            "attention": lambda ctx, x, tk: self._attention(layer, ctx, x, tk),
            "attn_residual": lambda ctx, a, b: a + b,
            "ffn_norm": lambda ctx, x: rms_norm(x, w["ffn_norm"], eps),
            "mlp_residual": lambda ctx, a, b: a + b,
        }
        if cfg.is_moe(layer):
            table.update(
                router=lambda ctx, x: route(x, w["gate"], cfg),
                experts=lambda ctx, x, r: experts_forward(
                    x, r, self.experts.get(layer, next_layer=self.next_moe.get(layer))
                ),
                shared_expert=lambda ctx, x: swiglu(x, w["shared_w1"], w["shared_w3"], w["shared_w2"]),
                moe_combine=lambda ctx, a, b: a + b,
            )
        else:
            table["mlp"] = lambda ctx, x: swiglu(x, w["w1"], w["w3"], w["w2"])
        return table[name]

    # ---- attention pieces
    def _indexer(self, layer: int, ctx: Ctx, x: torch.Tensor) -> torch.Tensor:
        cfg, w = self.cfg, self.w[layer]
        s, hi, d, r = x.shape[0], cfg.index_heads, cfg.index_dim, cfg.qk_rope
        start, end = ctx.start, ctx.start + s
        if ctx.state.ik_len[layer] < start:
            raise ValueError(
                f"layer {layer} indexer: no index_key prefix for [0, {start}) in the state (a golden prefix without "
                "index_key: the spec's state.tensors must name index_key for the full-indexer layers)"
            )
        qn = layer_norm(x, w["idx_q_norm_w"], w["idx_q_norm_b"], INDEX_NORM_EPS)
        q = F.linear(qn, as_compute(w["idx_wq_b"], qn)).view(s, hi, d)
        q = q * softmax_scaling(ctx.extra["log_pos"], w["idx_ssmax_s"], w["idx_ssmax_b"], q.dtype)[:, :, None]
        k = layer_norm(F.linear(x, as_compute(w["idx_wk"], x)), w["idx_k_norm_w"], w["idx_k_norm_b"], INDEX_NORM_EPS)
        if cfg.use_rope(layer):
            cos, sin = ctx.extra["cos"], ctx.extra["sin"]
            q = torch.cat([rope_interleaved(q[..., :r], cos, sin), q[..., r:]], dim=-1)
            k = torch.cat([rope_interleaved(k[..., :r], cos, sin), k[..., r:]], dim=-1)
        wts = F.linear(x, as_compute(w["idx_weights_proj"], x)).float() * (hi**-0.5)
        ik = ctx.state.ik[layer]
        ik[start:end] = k
        ctx.state.ik_len[layer] = end
        topk = indexer_topk(q, ik[:end], wts, start, cfg.index_topk, d**-0.5)
        self._xl[("topk", layer)] = (start, s, topk)
        return topk

    def _shared_topk(self, layer: int, ctx: Ctx) -> torch.Tensor:
        if "shared_topk" in ctx.extra:
            return ctx.extra["shared_topk"]
        src = self.cfg.topk_source(layer)
        got = self._xl.get(("topk", src))
        if got is None or got[:2] != (ctx.start, ctx.length):
            raise KeyError(
                f"layer {layer} reuses layer {src}'s top-k, but layer {src} has not run on chunk "
                f"[{ctx.start}, {ctx.start + ctx.length}); run it first or set ctx.extra['shared_topk'] "
                f"(the golden's L{src}.topk)"
            )
        return got[2]

    def _source_kv(self, layer: int, ctx: Ctx, end: int):
        """A shared-KV layer's (latent [end, 512], k_rot [end, r]) of its source layer."""
        if "shared_kv" in ctx.extra:
            st = ctx.extra["shared_kv"]
            return st["kv_latent"][:end].to(self.dtype), st["k_rope"][:end].to(self.dtype)
        src = self.cfg.kv_source(layer)
        got = self._xl.get(("kv", src))
        if got is None or got[:2] != (ctx.start, ctx.length):
            raise KeyError(
                f"layer {layer} reads layer {src}'s KV, but layer {src} has not run on chunk "
                f"[{ctx.start}, {ctx.start + ctx.length}); run it first or set ctx.extra['shared_kv'] "
                f"(the golden's layer-{src} state)"
            )
        return got[2][:end], got[3][:end]

    def _attention(self, layer: int, ctx: Ctx, x: torch.Tensor, topk: torch.Tensor) -> torch.Tensor:
        cfg, w = self.cfg, self.w[layer]
        s, hq, lat = x.shape[0], cfg.n_heads, cfg.kv_lora
        nd, r = cfg.nope_dim(layer), cfg.rope_dim(layer)
        start, end = ctx.start, ctx.start + s
        cos, sin = ctx.extra["cos"], ctx.extra["sin"]
        q = F.linear(x, as_compute(w["wq"], x)).view(s, hq, cfg.qk_head)
        q_nope, q_rot = q[..., :nd], q[..., nd:]
        if r:
            q_rot = rope_interleaved(q_rot, cos, sin)
        if cfg.owns_kv(layer):
            ckv = F.linear(x, as_compute(w["wkv_a"], x))
            latent = rms_norm(ckv[:, :lat], w["kv_a_norm"], cfg.eps)
            k_rot = ckv[:, lat:]
            if r:
                k_rot = rope_interleaved(k_rot, cos, sin)
            kv_st, kr_st = ctx.state.kv[layer], ctx.state.kr[layer]
            kv_st[start:end] = latent
            kr_st[start:end] = k_rot
            if cfg.kv_shared_later(layer):
                self._xl[("kv", layer)] = (start, s, kv_st, kr_st)
            lat_all, rot_all = kv_st[:end], kr_st[:end]
        else:
            lat_all, rot_all = self._source_kv(layer, ctx, end)
        sm = softmax_scaling(ctx.extra["log_pos"], w["ssmax_s"], w["ssmax_b"], q.dtype)[:, :, None]
        q_nope, q_rot = q_nope * sm, q_rot * sm
        q_lat = torch.einsum("shd,hdc->shc", q_nope, as_compute(w["w_uk"], x))  # [S, Hq, 512]
        qq = torch.cat([q_lat, q_rot], dim=-1)
        kv_all = torch.cat([lat_all, rot_all], dim=-1)
        o_lat = sparse_mla(qq, kv_all, topk, self.scale, start, lat)  # [S, Hq, 512]
        o = torch.einsum("shc,hvc->shv", o_lat, as_compute(w["w_uv"], x))  # [S, Hq, 128]
        o = o.reshape(s, hq * cfg.v_head)
        return F.linear(o, as_compute(w["wo"], o))

    # ---- model
    @torch.no_grad()
    def forward_chunk(self, tokens, start, state, rec=noop, logits_last_n=0):
        """tokens [S] at absolute positions [start, start + S). Returns (final_norm [S, H], logits [n, V] | None)."""
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        h = self.embed[tokens].to(self.dtype)
        rec("embed", h)
        n = tokens.shape[0]
        for i in self.layer_ids:
            ctx = self.chunk_context(i, start, n, state)
            h = run_block(self.block_graph(i), lambda name, i=i: self.component(i, name), ctx, h, rec, prefix=f"L{i}.")
        out = rms_norm(h, self.final_norm_w, self.cfg.eps)
        rec("final_norm", out)
        logits = None
        if logits_last_n:
            x = out[-logits_last_n:]
            logits = F.linear(x, as_compute(self.lm_head, x))
            logits = torch.tanh(logits / self.cfg.softcap) * self.cfg.softcap
            rec("logits", logits)
        return out, logits
