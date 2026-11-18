# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Streaming fp32 reference forward for Hy4-preview prefill (HF ``hy_v4`` semantics).

Why not ``HYV4ForCausalLM`` directly: the BF16 checkpoint is 1.5 TB (does not fit host RAM), and HF's
eager DSA attention materializes a dense ``[heads, S, S]`` score tensor (275 GB at 32K, 4.4 TB at
128K). This module computes the same function one decoder layer at a time:

* one layer's weights resident at a time (the next layer prefetches on a thread while this one runs);
* the iHC residual kept as 4 fp32 streams ``[S, 4, D]`` for the whole prompt;
* the DSA indexer scored per query block against the causal key prefix, keeping the top-``index_topk``;
* sparse MLA per query block in the absorbed (latent) form over the gathered top-k latent rows, with
  the learnable per-head sink as an extra softmax logit and the elementwise sigmoid output gate.

All math runs in fp32 on bf16 (or dequantized MXFP8) weights. The HF-equivalence test
(``tests/hy4/test_hy4_golden.py``) checks this against ``HYV4ForCausalLM`` (transformers >= 5.23)
on a random tiny checkpoint, layer by layer.

Per layer, the sink receives exactly what the device writes for that layer:
  * ``kvpe``    ``[S, kv_lora + rope]``  = ``kv_a_layernorm(latent) | k_pe`` with k_pe roped HALF-SPLIT (HF frame)
  * ``index_k`` ``[S, index_head_dim]`` on full-indexer layers = ``LayerNorm(wk(x))`` with RoPE on the
                LAST ``rope`` dims, half-split (HF frame; ``golden.index_k_hf_to_device`` re-frames it)
  * ``topk``    ``[S, index_topk]`` int32 on full-indexer layers, descending score, ``-1`` = no valid key
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.hy4.config import Hy4RefConfig

F32 = torch.float32


# ---------------------------------------------------------------------------------------------- math
def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    x = x.to(F32)
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return y if weight is None else y * weight


def rope_cos_sin(positions: torch.Tensor, dim: int, theta: float, fp64: bool = True):
    """``[N, dim/2]`` cos/sin for absolute ``positions``. HF builds the angle in fp32; ``fp64`` (default)
    builds it in fp64 and rounds once, which matters only at long context (fp32 angle ulp at 128K is ~8e-3)."""
    dt = torch.float64 if fp64 else F32
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dt) / dim))
    ang = positions.to(dt)[:, None] * inv_freq[None, :]
    return ang.cos().to(F32), ang.sin().to(F32)


def apply_rope_half_split(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF ``rotate_half`` RoPE on the last dim of ``x`` ``[N, ..., d]``; cos/sin ``[N, d/2]``."""
    shape = (cos.shape[0],) + (1,) * (x.dim() - 2) + (cos.shape[1],)
    c, s = cos.view(shape), sin.view(shape)
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    return torch.cat((a * c - b * s, b * c + a * s), dim=-1)


def hyper_connection(x4: torch.Tensor, fn, base, scale, cfg: Hy4RefConfig):
    """iHC pre-mix (HF ``HYV4HyperConnection``): ``x4`` ``[N, hc, D]`` -> (post ``[N, hc]``, sublayer input ``[N, D]``)."""
    hc = cfg.hc_mult
    flat = x4.flatten(1)
    mixes = (flat @ fn.T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    pre_l, post_l = mixes.split(hc, dim=-1)
    pre = torch.sigmoid(pre_l * scale[0] + base[:hc]) + cfg.hc_eps
    post = cfg.hc_magnitude * torch.sigmoid(post_l * scale[1] + base[hc:]) + cfg.hc_eps
    return post, (pre.unsqueeze(-1) * x4).sum(1)


def hyper_head(x4: torch.Tensor, fn, base, scale, cfg: Hy4RefConfig) -> torch.Tensor:
    """Final iHC collapse (HF ``HYV4HyperHead``)."""
    flat = x4.flatten(1)
    mixes = (flat @ fn.T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    pre = torch.sigmoid(mixes * scale + base) + cfg.hc_eps
    return (pre.unsqueeze(-1) * x4).sum(1)


def _blocks(n: int, block: int):
    for a in range(0, n, block):
        yield a, min(a + block, n)


# ----------------------------------------------------------------------------------------- weights
def load_layer_weights(ckpt, cfg: Hy4RefConfig, layer: int) -> dict:
    """Every tensor of one decoder layer. Dense/attention weights are upcast to fp32 here; routed
    experts stay in their stored dtype (bf16, or bf16 after MXFP8 dequant) and are upcast per expert."""
    p = f"model.layers.{layer}."
    a = p + "self_attn."
    w = {}

    def f32(name):
        return ckpt.get(name, F32)

    for tag, hc_name in (("attn_hc", "hc_attn_layer"), ("ffn_hc", "hc_mlp_layer")):
        w[f"{tag}.fn"] = f32(p + f"{hc_name}.hc_pre.hc_fn")
        w[f"{tag}.base"] = f32(p + f"{hc_name}.hc_pre.hc_base")
        w[f"{tag}.scale"] = f32(p + f"{hc_name}.hc_pre.hc_scale")
    w["input_layernorm"] = f32(p + "input_layernorm.weight")
    w["post_attention_layernorm"] = f32(p + "post_attention_layernorm.weight")
    for n in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", "linear_gate"):
        w[n] = f32(a + n + ".weight")
    w["q_a_layernorm"] = f32(a + "q_a_layernorm.weight")
    w["kv_a_layernorm"] = f32(a + "kv_a_layernorm.weight")
    w["sinks"] = f32(a + "learnable_sink_param")
    if cfg.is_full_indexer(layer):
        x = a + "indexer."
        for n in ("wq_b", "wk", "weights_proj"):
            w[f"indexer.{n}"] = f32(x + n + ".weight")
        w["indexer.k_norm.weight"] = f32(x + "k_norm.weight")
        w["indexer.k_norm.bias"] = f32(x + "k_norm.bias")
    m = p + "mlp."
    if cfg.is_dense(layer):
        for n in ("gate_proj", "up_proj", "down_proj"):
            w[f"mlp.{n}"] = f32(m + n + ".weight")
    else:
        w["mlp.gate"] = f32(m + "gate.weight")
        w["mlp.gate.bias"] = f32(m + "gate.e_score_correction_bias")
        w["mlp.experts.gate_up_proj"] = ckpt.get(m + "experts.gate_up_proj")
        w["mlp.experts.down_proj"] = ckpt.get(m + "experts.down_proj")
        for n in ("gate_proj", "up_proj", "down_proj"):
            w[f"mlp.shared.{n}"] = f32(m + f"shared_experts.{n}.weight")
    return w


# --------------------------------------------------------------------------------------- reference
class Hy4PrefillReference:
    """Run a prompt through the first ``num_layers`` decoder layers and hand each layer's cache to a sink.

    ``sink.on_layer(layer, kvpe=..., index_k=..., topk=..., streams=...)`` is called after every layer
    (``index_k``/``topk`` are None on shared-indexer layers; ``streams`` is the iHC state ``[S, hc, D]``
    after the layer, owned by the reference -- copy it to keep it). ``sink.on_final(...)`` receives the
    next-token logits when the full stack ran.
    """

    def __init__(
        self,
        ckpt,
        cfg: Hy4RefConfig | None = None,
        *,
        attn_block: int = 128,
        proj_block: int = 2048,
        index_budget_bytes: int = 2 << 30,
        rope_fp64: bool = True,
        prefetch: bool = True,
    ):
        self.ckpt = ckpt
        self.cfg = cfg or ckpt.config
        self.attn_block = attn_block
        self.proj_block = proj_block
        self.index_budget_bytes = index_budget_bytes
        self.rope_fp64 = rope_fp64
        self.prefetch = prefetch

    # ---- attention ----
    def _attention(self, layer: int, w: dict, x4: torch.Tensor, cos, sin, prev_topk):
        cfg = self.cfg
        S = x4.shape[0]
        H, R, L = cfg.num_attention_heads, cfg.qk_rope_head_dim, cfg.kv_lora_rank
        full = cfg.is_full_indexer(layer)

        post = torch.empty(S, cfg.hc_mult, dtype=F32)
        hn = torch.empty(S, cfg.hidden_size, dtype=F32)
        q_resid = torch.empty(S, cfg.q_lora_rank, dtype=F32)
        kvpe = torch.empty(S, cfg.kvpe_dim, dtype=F32)
        index_k = torch.empty(S, cfg.index_head_dim, dtype=F32) if full else None
        index_w = torch.empty(S, cfg.index_n_heads, dtype=F32) if full else None

        # Pass 1 -- per-token: iHC pre-mix, norms, the MLA latent (what the cache stores), the indexer key.
        for a, b in _blocks(S, self.proj_block):
            post[a:b], h = hyper_connection(x4[a:b], w["attn_hc.fn"], w["attn_hc.base"], w["attn_hc.scale"], cfg)
            hn[a:b] = rms_norm(h, w["input_layernorm"], cfg.rms_norm_eps)
            q_resid[a:b] = rms_norm(hn[a:b] @ w["q_a_proj"].T, w["q_a_layernorm"], cfg.low_rank_norm_eps)
            ckv = hn[a:b] @ w["kv_a_proj_with_mqa"].T
            kvpe[a:b, :L] = rms_norm(ckv[:, :L], w["kv_a_layernorm"], cfg.low_rank_norm_eps)
            kvpe[a:b, L:] = apply_rope_half_split(ckv[:, L:], cos[a:b], sin[a:b])
            if full:
                k = F.layer_norm(
                    hn[a:b] @ w["indexer.wk"].T,
                    (cfg.index_head_dim,),
                    w["indexer.k_norm.weight"],
                    w["indexer.k_norm.bias"],
                    cfg.rms_norm_eps,
                )
                nope = cfg.index_head_dim - R  # Hy4 ropes the LAST R dims of the index head
                index_k[a:b, :nope] = k[:, :nope]
                index_k[a:b, nope:] = apply_rope_half_split(k[:, nope:], cos[a:b], sin[a:b])
                index_w[a:b] = (
                    (hn[a:b] @ w["indexer.weights_proj"].T) * cfg.index_n_heads**-0.5 * cfg.index_head_dim**-0.5
                )

        topk = self._indexer_topk(w, q_resid, index_k, index_w, cos, sin) if full else prev_topk
        if topk is None:
            raise ValueError(f"layer {layer} is a shared-indexer layer but no prior full layer produced top-k")

        # Pass 2 -- per query block: sparse MLA (absorbed) over the top-k latent rows, sink, gate, o_proj.
        wkv = w["kv_b_proj"].view(H, cfg.qk_nope_head_dim + cfg.v_head_dim, L)
        w_uk, w_uv = wkv[:, : cfg.qk_nope_head_dim], wkv[:, cfg.qk_nope_head_dim :]
        latent, k_pe = kvpe[:, :L], kvpe[:, L:]
        sinks = w["sinks"].view(1, H, 1)
        for a, b in _blocks(S, self.attn_block):
            n = b - a
            q = (q_resid[a:b] @ w["q_b_proj"].T).view(n, H, cfg.qk_head_dim)
            q_nope, q_pe = q[..., : cfg.qk_nope_head_dim], apply_rope_half_split(
                q[..., cfg.qk_nope_head_dim :], cos[a:b], sin[a:b]
            )
            q_lat = torch.einsum("nhd,hdc->nhc", q_nope, w_uk)
            idx = topk[a:b].long()
            pos = torch.arange(a, b).unsqueeze(1)
            valid = (idx >= 0) & (idx <= pos)
            safe = idx.clamp(min=0)
            lat_g, pe_g = latent[safe], k_pe[safe]  # [n, K, L], [n, K, R]
            scores = (torch.einsum("nhc,nkc->nhk", q_lat, lat_g) + torch.einsum("nhr,nkr->nhk", q_pe, pe_g)) * (
                cfg.softmax_scale
            )
            scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
            logits = torch.cat([scores, sinks.expand(n, H, 1)], dim=-1)
            probs = torch.softmax(logits - logits.amax(-1, keepdim=True), dim=-1)[..., :-1]
            o_lat = torch.einsum("nhk,nkc->nhc", probs, lat_g)
            o = torch.einsum("nhc,hvc->nhv", o_lat, w_uv)
            gate = (hn[a:b] @ w["linear_gate"].T).view(n, H, cfg.v_head_dim)
            o = (o * torch.sigmoid(gate)).reshape(n, H * cfg.v_head_dim)
            out = o @ w["o_proj"].T
            x4[a:b] += post[a:b].unsqueeze(-1) * out.unsqueeze(1)
        return kvpe, index_k, topk

    def _indexer_topk(self, w, q_resid, index_k, index_w, cos, sin):
        cfg = self.cfg
        S = q_resid.shape[0]
        NH, DI, R, K = cfg.index_n_heads, cfg.index_head_dim, cfg.qk_rope_head_dim, cfg.index_topk
        nope = DI - R
        out = torch.full((S, K), -1, dtype=torch.int32)
        block = max(16, min(1024, self.index_budget_bytes // max(1, 4 * NH * S)))
        for a, b in _blocks(S, block):
            n = b - a
            q = (q_resid[a:b] @ w["indexer.wq_b"].T).view(n, NH, DI)
            q = torch.cat([q[..., :nope], apply_rope_half_split(q[..., nope:], cos[a:b], sin[a:b])], dim=-1)
            # Causal: queries in [a, b) see keys [0, b); later keys are masked per row below.
            s = torch.relu(torch.einsum("nhd,td->nht", q, index_k[:b]))
            scores = torch.einsum("nh,nht->nt", index_w[a:b], s)
            causal = torch.arange(b).unsqueeze(0) > torch.arange(a, b).unsqueeze(1)
            scores = scores.masked_fill(causal, float("-inf"))
            k = min(K, b)
            vals, idx = scores.topk(k, dim=-1)
            idx = idx.to(torch.int32).masked_fill(torch.isneginf(vals), -1)
            out[a:b, :k] = idx
        return out

    # ---- feed-forward ----
    def _ffn(self, layer: int, w: dict, x4: torch.Tensor):
        cfg = self.cfg
        S = x4.shape[0]
        post = torch.empty(S, cfg.hc_mult, dtype=F32)
        hn = torch.empty(S, cfg.hidden_size, dtype=F32)
        for a, b in _blocks(S, self.proj_block):
            post[a:b], h = hyper_connection(x4[a:b], w["ffn_hc.fn"], w["ffn_hc.base"], w["ffn_hc.scale"], cfg)
            hn[a:b] = rms_norm(h, w["post_attention_layernorm"], cfg.rms_norm_eps)

        def swiglu_mlp(x, gate_w, up_w, down_w):  # HF HYV4MLP: dense layer and shared expert, no clamp
            return (F.silu(x @ gate_w.T) * (x @ up_w.T)) @ down_w.T

        if cfg.is_dense(layer):
            out = torch.empty_like(hn)
            for a, b in _blocks(S, self.proj_block):
                out[a:b] = swiglu_mlp(hn[a:b], w["mlp.gate_proj"], w["mlp.up_proj"], w["mlp.down_proj"])
        else:
            out = self._moe(w, hn)
            for a, b in _blocks(S, self.proj_block):
                out[a:b] += swiglu_mlp(
                    hn[a:b], w["mlp.shared.gate_proj"], w["mlp.shared.up_proj"], w["mlp.shared.down_proj"]
                )
        for a, b in _blocks(S, self.proj_block):
            x4[a:b] += post[a:b].unsqueeze(-1) * out[a:b].unsqueeze(1)

    def _moe(self, w: dict, hn: torch.Tensor) -> torch.Tensor:
        """HF ``HYV4MoE`` routed half: sigmoid router, bias-corrected top-k choice, normalized weights x
        routed_scaling_factor, SwiGLU with gate clamped above and up clamped both sides at swiglu_limit."""
        cfg = self.cfg
        E, k, lim = cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.swiglu_limit
        scores = torch.sigmoid(hn @ w["mlp.gate"].T)
        choice = scores + w["mlp.gate.bias"]
        if cfg.n_group > 1:
            grp = choice.view(-1, cfg.n_group, E // cfg.n_group).topk(2, dim=-1)[0].sum(-1)
            keep = torch.zeros_like(grp).scatter_(1, grp.topk(cfg.topk_group, dim=-1)[1], 1.0)
            choice = choice.masked_fill(~keep.repeat_interleave(E // cfg.n_group, dim=1).bool(), float("-inf"))
        top_i = choice.topk(k, dim=-1, sorted=False)[1]
        top_w = scores.gather(1, top_i)
        if cfg.norm_topk_prob:
            top_w = top_w / (top_w.sum(-1, keepdim=True) + 1e-20)
        top_w = top_w * cfg.routed_scaling_factor

        out = torch.zeros_like(hn)
        gate_up, down = w["mlp.experts.gate_up_proj"], w["mlp.experts.down_proj"]
        flat_e = top_i.flatten()
        order = flat_e.argsort(stable=True)
        counts = torch.bincount(flat_e, minlength=E).tolist()
        start = 0
        for e, c in enumerate(counts):
            if c == 0:
                continue
            sel = order[start : start + c]
            start += c
            tok, slot = sel // k, sel % k
            gu = hn[tok] @ gate_up[e].to(F32).T
            g, u = gu.chunk(2, dim=-1)
            act = F.silu(g.clamp(max=lim)) * u.clamp(min=-lim, max=lim)
            y = (act @ down[e].to(F32).T) * top_w[tok, slot].unsqueeze(-1)
            out.index_add_(0, tok, y)
        return out

    # ---- driver ----
    def run(
        self,
        token_ids,
        sink,
        *,
        num_layers: int | None = None,
        final_positions=None,
        start_state: dict | None = None,
        on_state=None,
    ):
        """Prefill ``token_ids`` (1-D, positions 0..S-1) through layers ``[0, num_layers)``.

        ``final_positions`` (default: the last token) are the rows whose next-token logits go to
        ``sink.on_final`` after a full-stack run; prefill is causal, so row ``p`` is the last row of the
        ``p + 1``-token prefix. ``start_state`` (``{"layer", "streams", "topk"}``) resumes after a saved
        layer; ``on_state(state)`` is offered the same dict after every layer so a long run can checkpoint.
        """
        cfg = self.cfg
        n_layers = cfg.num_hidden_layers if num_layers is None else num_layers
        token_ids = torch.as_tensor(token_ids, dtype=torch.long).flatten()
        S = token_ids.numel()
        cos, sin = rope_cos_sin(torch.arange(S), cfg.qk_rope_head_dim, cfg.rope_theta, self.rope_fp64)

        if start_state is not None:
            first = start_state["layer"] + 1
            x4, topk = start_state["streams"], start_state["topk"]
        else:
            first, topk = 0, None
            emb = self.ckpt.get("model.embed_tokens.weight")[token_ids].to(F32)
            x4 = emb.unsqueeze(1).expand(S, cfg.hc_mult, cfg.hidden_size).contiguous()

        pool = ThreadPoolExecutor(1) if self.prefetch else None
        load = lambda i: load_layer_weights(self.ckpt, cfg, i)  # noqa: E731
        pending = pool.submit(load, first) if (pool and first < n_layers) else None
        try:
            for layer in range(first, n_layers):
                t0 = time.time()
                w = pending.result() if pending is not None else load(layer)
                t_load = time.time() - t0
                pending = pool.submit(load, layer + 1) if (pool and layer + 1 < n_layers) else None
                kvpe, index_k, topk = self._attention(layer, w, x4, cos, sin, topk)
                self._ffn(layer, w, x4)
                del w
                full = cfg.is_full_indexer(layer)
                sink.on_layer(layer, kvpe=kvpe, index_k=index_k, topk=topk if full else None, streams=x4)
                if on_state is not None:
                    on_state({"layer": layer, "streams": x4, "topk": topk})
                logger.info(
                    f"[hy4-ref] layer {layer:>2}/{n_layers} {'full ' if full else 'shared'} "
                    f"{'dense' if cfg.is_dense(layer) else 'moe  '} S={S} wait-weights={t_load:.1f}s "
                    f"total={time.time() - t0:.1f}s |x4|max={x4.abs().max().item():.3e}"
                )
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

        if n_layers == cfg.num_hidden_layers and hasattr(sink, "on_final"):
            positions = sorted({S - 1} if final_positions is None else {int(p) for p in final_positions})
            h = hyper_head(
                x4[positions],
                self.ckpt.get("model.hc_head.hc_head_fn", F32),
                self.ckpt.get("model.hc_head.hc_head_base", F32),
                self.ckpt.get("model.hc_head.hc_head_scale", F32),
                cfg,
            )
            h = rms_norm(h, self.ckpt.get("model.norm.weight", F32), cfg.rms_norm_eps)
            logits = h @ self.ckpt.get("lm_head.weight", F32).T  # HF keeps lm_head in fp32
            for row, p in enumerate(positions):
                sink.on_final(logits=logits[row], position=p)
        return x4
