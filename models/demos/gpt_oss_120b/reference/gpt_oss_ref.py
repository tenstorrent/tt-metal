# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU reference for gpt-oss-120b prefill (the interface in _bringup_interface.py).

Model (openai/gpt-oss-120b, transformers ``modeling_gpt_oss``): 36 decoder layers alternating sliding-window (128)
and full causal attention, 64 query heads / 8 KV heads of 64, q/k/v/o biases, a learned per-head attention sink
(an extra softmax logit, dropped from the output), half-split YaRN RoPE (theta 150000, factor 32, cos and sin
scaled by 0.1 ln 32 + 1), RMSNorm eps 1e-5, and a 128-expert top-4 MoE (router linear + bias, softmax over the
chosen 4 logits; experts: interleaved gate/up with biases, clamped SwiGLU (limit 7, alpha 1.702, (up + 1) * glu),
down with bias). Expert weights are mxfp4 in the checkpoint and are dequantized exactly at load.

Block graph (every layer): attn_norm -> attention (stateful) -> attn_residual -> ffn_norm -> router -> experts
-> mlp_residual. ``router`` writes dense routing weights [S, 128] (the softmax weight of each chosen expert, 0
elsewhere); ``experts`` reads them with the normed input.

State: per layer ``key`` [S, 8, 64] (after RoPE) and ``value`` [S, 8, 64], every position on both layer kinds (a
sliding layer only reads the last 128). Held head-major [8, max_seq, 64] so attention reads contiguous slices.

Speed: attention runs over query blocks aligned to absolute positions (QBLOCK), each against its causal key range
only (sliding: the 127 keys before the block), so chunked and one-shot prefill compute the same blocks; experts
are dispatched by token sort. Weights are only read for the requested layers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from ._bringup_interface import Ctx, Step, noop, run_block

QBLOCK = 256  # query rows per attention block (absolute-aligned)
SWIGLU_ALPHA = 1.702
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
EXPERT_FP32_MAX_LAYERS = 8  # above this many layers, expert weights are held in bf16 (exact for mxfp4) and cast per use


def mxfp4_dequant(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """[..., G, 16] uint8 nibble pairs (low nibble first) + [..., G] uint8 biased exponents -> [..., G * 32] fp32."""
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32)
    vals = torch.stack((lut[(blocks & 0x0F).long()], lut[(blocks >> 4).long()]), dim=-1)
    vals = vals.reshape(*blocks.shape[:-1], 32)
    vals = torch.ldexp(vals, (scales.to(torch.int32) - 127).unsqueeze(-1))
    return vals.reshape(*blocks.shape[:-2], -1)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Half-split rotation. x [S, H, D], cos / sin [S, 1, D/2]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def yarn_inv_freq(cfg: dict) -> tuple[torch.Tensor, float]:
    """transformers ``_compute_yarn_parameters`` for this config: inverse frequencies and the cos/sin scale."""
    rs = cfg["rope_scaling"]
    base, dim = float(cfg["rope_theta"]), int(cfg["head_dim"])
    factor, orig = float(rs["factor"]), int(rs["original_max_position_embeddings"])
    beta_fast, beta_slow = float(rs.get("beta_fast") or 32), float(rs.get("beta_slow") or 1)
    attention_factor = rs.get("attention_factor") or (0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0)

    def corr_dim(rot):
        return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))

    low, high = corr_dim(beta_fast), corr_dim(beta_slow)
    if rs.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    pos_freqs = base ** (torch.arange(0, dim, 2).to(dtype=torch.float) / dim)
    extra, inter = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    extra_factor = 1 - ramp
    return inter * (1 - extra_factor) + extra * extra_factor, float(attention_factor)


class Checkpoint:
    """Tensor reader over the sharded safetensors checkpoint (read-only, one tensor at a time)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.map = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self._open = {}

    def get(self, name: str) -> torch.Tensor:
        f = self.map[name]
        if f not in self._open:
            self._open[f] = safe_open(str(self.path / f), framework="pt")
        return self._open[f].get_tensor(name)

    def close(self):
        self._open.clear()


class LayerWeights:
    def __init__(self, ck: Checkpoint, i: int, cfg: dict, dtype: torch.dtype, expert_dtype: torch.dtype):
        p = f"model.layers.{i}."
        g = lambda n: ck.get(p + n).to(dtype)  # noqa: E731
        self.attn_norm, self.ffn_norm = g("input_layernorm.weight"), g("post_attention_layernorm.weight")
        self.wq, self.bq = g("self_attn.q_proj.weight"), g("self_attn.q_proj.bias")
        self.wk, self.bk = g("self_attn.k_proj.weight"), g("self_attn.k_proj.bias")
        self.wv, self.bv = g("self_attn.v_proj.weight"), g("self_attn.v_proj.bias")
        self.wo, self.bo = g("self_attn.o_proj.weight"), g("self_attn.o_proj.bias")
        self.sinks = g("self_attn.sinks")
        self.wr, self.br = g("mlp.router.weight"), g("mlp.router.bias")
        E, I = cfg["num_local_experts"], cfg["intermediate_size"]
        # gate_up rows are interleaved (gate 2j, up 2j+1): stored as [gate rows | up rows] so one linear gives both
        perm = torch.cat((torch.arange(0, 2 * I, 2), torch.arange(1, 2 * I, 2)))
        self.b_gu = g("mlp.experts.gate_up_proj_bias")[:, perm].contiguous()
        self.b_d = g("mlp.experts.down_proj_bias")
        gu_b, gu_s = ck.get(p + "mlp.experts.gate_up_proj_blocks"), ck.get(p + "mlp.experts.gate_up_proj_scales")
        d_b, d_s = ck.get(p + "mlp.experts.down_proj_blocks"), ck.get(p + "mlp.experts.down_proj_scales")
        self.w_gu = torch.empty(E, 2 * I, cfg["hidden_size"], dtype=expert_dtype)  # [E, out, in]
        self.w_d = torch.empty(E, cfg["hidden_size"], I, dtype=expert_dtype)
        step = 8
        for e in range(0, E, step):
            self.w_gu[e : e + step] = mxfp4_dequant(gu_b[e : e + step], gu_s[e : e + step])[:, perm]
            self.w_d[e : e + step] = mxfp4_dequant(d_b[e : e + step], d_s[e : e + step])
        self.dtype = dtype


class LayerKV:
    """One layer's KV cache, head-major [n_kv, max_seq, head_dim]."""

    def __init__(self, n_kv: int, max_seq: int, head_dim: int, dtype: torch.dtype):
        self.k = torch.zeros(n_kv, max_seq, head_dim, dtype=dtype)
        self.v = torch.zeros(n_kv, max_seq, head_dim, dtype=dtype)


class State:
    def __init__(self, layers, max_seq, n_kv, head_dim, dtype):
        self.layers = {i: LayerKV(n_kv, max_seq, head_dim, dtype) for i in layers}


class GptOssReference:
    def __init__(self, hf_dir, layers=None, dtype=torch.float32):
        self.hf_dir = Path(hf_dir)
        self.cfg = cfg = json.loads((self.hf_dir / "config.json").read_text())
        self.dtype = dtype = dtype or torch.float32
        self.layer_ids = list(range(cfg["num_hidden_layers"])) if layers is None else list(layers)
        self.H, self.D = cfg["hidden_size"], cfg["head_dim"]
        self.nq, self.nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.top_k, self.E = cfg["num_experts_per_tok"], cfg["num_local_experts"]
        self.I, self.eps = cfg["intermediate_size"], float(cfg["rms_norm_eps"])
        self.limit = float(cfg.get("swiglu_limit", 7.0))
        self.window = int(cfg["sliding_window"])
        self.sliding = {i: cfg["layer_types"][i] == "sliding_attention" for i in range(cfg["num_hidden_layers"])}
        self.scale = self.D**-0.5

        expert_dtype = dtype if len(self.layer_ids) <= EXPERT_FP32_MAX_LAYERS else torch.bfloat16
        ck = Checkpoint(self.hf_dir)
        self.embed = ck.get("model.embed_tokens.weight").to(dtype)
        self.final_norm = ck.get("model.norm.weight").to(dtype)
        self.lm_head = ck.get("lm_head.weight").to(dtype)
        self.w = {i: LayerWeights(ck, i, cfg, dtype, expert_dtype) for i in self.layer_ids}
        ck.close()

        inv_freq, att = yarn_inv_freq(cfg)
        pos = torch.arange(cfg["max_position_embeddings"], dtype=torch.float32)
        freqs = pos[:, None] * inv_freq[None, :]
        self.cos = (freqs.cos() * att).to(dtype)  # [max_pos, D/2], built once
        self.sin = (freqs.sin() * att).to(dtype)
        self._components = {i: self._make_components(i) for i in self.layer_ids}

    # ------------------------------------------------------------------ interface
    def new_state(self, max_seq: int) -> State:
        return State(self.layer_ids, max_seq, self.nkv, self.D, self.dtype)

    def state_tensors(self, state: State, layer: int, length: int) -> dict:
        kv = state.layers[layer]
        return {
            "key": kv.k[:, :length].permute(1, 0, 2).contiguous(),
            "value": kv.v[:, :length].permute(1, 0, 2).contiguous(),
        }

    def load_state(self, state: State, layer: int, tensors: dict, length: int) -> None:
        kv = state.layers[layer]
        for name, dst in (("key", kv.k), ("value", kv.v)):
            src = tensors[name][:length].reshape(length, self.nkv, self.D)
            dst[:, :length] = src.permute(1, 0, 2).to(dst.dtype)

    def block_graph(self, layer: int) -> list[Step]:
        return [
            Step("attn_norm", ("in",), "attn_norm", "norm"),
            Step("attention", ("attn_norm",), "attn_out", "attention", stateful=True),
            Step("attn_residual", ("in", "attn_out"), "h_mid", "residual"),
            Step("ffn_norm", ("h_mid",), "ffn_norm", "norm"),
            Step("router", ("ffn_norm",), "router", "router"),
            Step("experts", ("ffn_norm", "router"), "moe_out", "moe"),
            Step("mlp_residual", ("h_mid", "moe_out"), "out", "residual"),
        ]

    def component(self, layer: int, step_name: str):
        return self._components[layer][step_name]

    def chunk_context(self, layer: int, start: int, length: int, state: State) -> Ctx:
        end = start + length
        return Ctx(
            layer,
            start,
            length,
            state.layers[layer] if state is not None else None,
            {"cos": self.cos[start:end, None, :], "sin": self.sin[start:end, None, :]},
        )

    @torch.no_grad()
    def forward_chunk(self, tokens, start, state, rec=noop, logits_last_n=0):
        h = self.embed[tokens.long()]
        rec("embed", h)
        for i in self.layer_ids:
            ctx = self.chunk_context(i, start, tokens.shape[0], state)
            h = run_block(self.block_graph(i), self._components[i].__getitem__, ctx, h, rec, prefix=f"L{i}.")
        out = rms_norm(h, self.final_norm, self.eps)
        rec("final_norm", out)
        logits = None
        if logits_last_n:
            logits = out[-logits_last_n:] @ self.lm_head.T
            rec("logits", logits)
        return out, logits

    # ------------------------------------------------------------------ steps
    def _make_components(self, i: int) -> dict:
        w = self.w[i]
        return {
            "attn_norm": lambda ctx, x: rms_norm(x, w.attn_norm, self.eps),
            "attention": lambda ctx, x: self._attention(w, i, ctx, x),
            "attn_residual": lambda ctx, a, b: a + b,
            "ffn_norm": lambda ctx, x: rms_norm(x, w.ffn_norm, self.eps),
            "router": lambda ctx, x: self._router(w, x),
            "experts": lambda ctx, x, r: self._experts(w, x, r),
            "mlp_residual": lambda ctx, a, b: a + b,
        }

    @torch.no_grad()
    def _attention(self, w: LayerWeights, layer: int, ctx: Ctx, x: torch.Tensor) -> torch.Tensor:
        S, D, nq, nkv = x.shape[0], self.D, self.nq, self.nkv
        rep = nq // nkv
        start, end = ctx.start, ctx.start + S
        cos, sin = ctx.extra["cos"], ctx.extra["sin"]
        q = apply_rope(F.linear(x, w.wq, w.bq).view(S, nq, D), cos, sin)
        k = apply_rope(F.linear(x, w.wk, w.bk).view(S, nkv, D), cos, sin)
        v = F.linear(x, w.wv, w.bv).view(S, nkv, D)
        kv = ctx.state
        kv.k[:, start:end] = k.permute(1, 0, 2)
        kv.v[:, start:end] = v.permute(1, 0, 2)
        sliding = self.sliding[layer]
        sink = w.sinks.view(nkv, rep, 1, 1)
        out = torch.empty(nq, S, D, dtype=x.dtype)
        qh = q.permute(1, 0, 2)  # [nq, S, D]
        a = start
        while a < end:
            b = min(end, (a // QBLOCK + 1) * QBLOCK)
            nb = b - a
            k0 = max(0, a - self.window + 1) if sliding else 0
            nk = b - k0
            qb = qh[:, a - start : b - start].reshape(nkv, rep * nb, D)
            s = torch.bmm(qb, kv.k[:, k0:b].transpose(1, 2)).mul_(self.scale).view(nkv, rep, nb, nk)
            qpos = torch.arange(a, b)[:, None]
            if sliding:
                kpos = torch.arange(k0, b)[None, :]
                s.masked_fill_((kpos > qpos) | (kpos <= qpos - self.window), float("-inf"))
            else:  # keys before the block are all visible: mask only the diagonal block
                s[..., a - k0 :].masked_fill_(torch.arange(a, b)[None, :] > qpos, float("-inf"))
            m = torch.maximum(s.amax(-1, keepdim=True), sink)
            s.sub_(m).exp_()
            denom = s.sum(-1, keepdim=True) + torch.exp(sink - m)
            o = torch.bmm(s.view(nkv, rep * nb, nk), kv.v[:, k0:b]).view(nkv, rep, nb, D) / denom
            out[:, a - start : b - start] = o.view(nq, nb, D)
            a = b
        return F.linear(out.permute(1, 0, 2).reshape(S, nq * D), w.wo, w.bo)

    def _router(self, w: LayerWeights, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x, w.wr, w.br)
        vals, idx = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.softmax(vals, dim=-1)
        return torch.zeros_like(logits).scatter_(1, idx, weights)

    @torch.no_grad()
    def _experts(self, w: LayerWeights, x: torch.Tensor, routing: torch.Tensor) -> torch.Tensor:
        tok, exp = routing.nonzero(as_tuple=True)
        order = torch.argsort(exp, stable=True)
        tok, exp = tok[order], exp[order]
        counts = torch.bincount(exp, minlength=self.E).tolist()
        out = torch.zeros_like(x)
        I, lim = self.I, self.limit
        lo = 0
        for e, n in enumerate(counts):
            if not n:
                continue
            t = tok[lo : lo + n]
            lo += n
            gu = F.linear(x[t], w.w_gu[e].to(x.dtype), w.b_gu[e])
            gate, up = gu[:, :I].clamp(max=lim), gu[:, I:].clamp(min=-lim, max=lim)
            hdn = (up + 1) * (gate * torch.sigmoid(gate * SWIGLU_ALPHA))
            o = F.linear(hdn, w.w_d[e].to(x.dtype), w.b_d[e]) * routing[t, e, None]
            out.index_add_(0, t, o)
        return out
