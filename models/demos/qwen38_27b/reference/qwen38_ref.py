# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU reference of Qwen3.8-27B (Qwen3_5ForConditionalGeneration, text model) for chunked prefill.

Follows transformers 5.12 ``models/qwen3_5/modeling_qwen3_5.py`` (the code the upstream trace ran):

- decoder RMSNorm: ``x * rsqrt(mean(x^2) + eps) * (1 + w)`` in fp32 (input_layernorm, post_attention_layernorm,
  q_norm, k_norm, the final norm);
- "linear" layers: Gated DeltaNet. in_proj_qkv -> causal depthwise conv (kernel 4) + silu -> q, k (16 heads x 128),
  v (48 heads x 128); q, k l2-normalised and repeated to 48 heads; beta = sigmoid(in_proj_b), g = -exp(A_log) *
  softplus(in_proj_a + dt_bias); the chunked gated delta rule (chunk 64) carries the recurrent state [48, 128, 128]
  (fp32); gated RMSNorm (plain w) with silu(in_proj_z), out_proj;
- "attn" layers: q_proj gives per head [q 256 | gate 256]; q_norm / k_norm per head; partial RoPE on the first 64
  dims (rotate-half, theta 1e7; text-only interleaved mRoPE is plain RoPE); causal GQA (24 q / 4 kv heads), scale
  1/16; output * sigmoid(gate); o_proj;
- dense SwiGLU MLP.

Weights are held in ``weight_dtype(spec)`` (bf16, as the checkpoint stores them) and cast to the activation dtype in
the op that uses them (``as_compute``); the compute dtype is fp32.

State per layer (spec state.by_block_type): attn ``key`` / ``value`` [4, seq, 256] (key after k_norm and RoPE, as the
HF cache keeps them); linear ``conv`` [10240, 4] (the last 4 in_proj_qkv rows before the conv, oldest first, zeros
before position 0: HF's conv_states) and ``recurrent`` [48, 128, 128] fp32 (HF's recurrent_states). Both linear
tensors are fixed-size: state_tensors gives the current ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from ._bringup_interface import Ctx, Step, as_compute, noop, run_block, weight_dtype

PREFIX = "model.language_model."
ATTN_BLOCK = 1024  # full attention runs this many query rows at a time


@dataclass(frozen=True)
class Config:
    hidden: int
    vocab: int
    num_layers: int
    eps: float
    layer_types: tuple[str, ...]
    # full attention
    n_heads: int
    n_kv: int
    head_dim: int
    rope_dim: int
    rope_theta: float
    # Gated DeltaNet
    lk_heads: int
    lv_heads: int
    lk_dim: int
    lv_dim: int
    conv_k: int
    # MLP
    inter: int

    @classmethod
    def load(cls, path: str) -> "Config":
        c = json.loads((Path(path) / "config.json").read_text())
        t = c.get("text_config", c)
        rp = t.get("rope_parameters") or {}
        head_dim = t["head_dim"]
        return cls(
            hidden=t["hidden_size"],
            vocab=t["vocab_size"],
            num_layers=t["num_hidden_layers"],
            eps=t["rms_norm_eps"],
            layer_types=tuple(t["layer_types"]),
            n_heads=t["num_attention_heads"],
            n_kv=t["num_key_value_heads"],
            head_dim=head_dim,
            rope_dim=int(head_dim * rp.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0))),
            rope_theta=float(rp.get("rope_theta", t.get("rope_theta", 10000.0))),
            lk_heads=t["linear_num_key_heads"],
            lv_heads=t["linear_num_value_heads"],
            lk_dim=t["linear_key_head_dim"],
            lv_dim=t["linear_value_head_dim"],
            conv_k=t["linear_conv_kernel_dim"],
            inter=t["intermediate_size"],
        )

    @property
    def key_dim(self) -> int:
        return self.lk_heads * self.lk_dim

    @property
    def value_dim(self) -> int:
        return self.lv_heads * self.lv_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim


class Checkpoint:
    """Reads single tensors of the sharded safetensors checkpoint, in the stored dtype (bf16)."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        with safe_open(str(self.path / self.index[name]), "pt") as f:
            return f.get_tensor(name).to(dtype).contiguous()

    def layer(self, i: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        p = f"{PREFIX}layers.{i}."
        names = [n for n in self.index if n.startswith(p)]
        return {n[len(p) :]: self.get(n, dtype) for n in names}


# ---------------------------------------------------------------- ops


def rms_norm_1p(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3_5RMSNorm: fp32 variance, weight applied as (1 + w), result in x's dtype."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + w.float())).to(x.dtype)


def rms_norm_gated(x: torch.Tensor, w: torch.Tensor, z: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3_5RMSNormGated: norm (plain w), then * silu(z) in fp32."""
    xf = x.float()
    h = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (h * as_compute(w, h) * F.silu(z.float())).to(x.dtype)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def chunk_gated_delta_rule(q, k, v, g, beta, state, chunk: int = 64):
    """transformers' torch_chunk_gated_delta_rule for one sequence, with the state carried in and out.

    q, k [S, H, Dk] (already l2-normalised), v [S, H, Dv], g, beta [S, H] (fp32), state [H, Dk, Dv] fp32.
    Returns out [S, H, Dv] (fp32) and the state after the last position. A tail shorter than ``chunk`` is zero-padded
    (g = 0, beta = 0, k = v = 0 there), which leaves the state unchanged, as upstream.
    """
    S = q.shape[0]
    q, k, v = (t.transpose(0, 1).float() for t in (q, k, v))  # [H, S, D]
    beta, g = beta.transpose(0, 1).float(), g.transpose(0, 1).float()  # [H, S]
    H, _, dk = k.shape
    dv = v.shape[-1]
    pad = (chunk - S % chunk) % chunk
    q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    n = (S + pad) // chunk
    q = q * (dk**-0.5)
    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v, k_beta, v_beta = (t.reshape(H, n, chunk, t.shape[-1]) for t in (q, k, v, k_beta, v_beta))
    g = g.reshape(H, n, chunk).cumsum(-1)
    upper = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool), diagonal=0)
    decay = (g[..., :, None] - g[..., None, :]).tril().exp().tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * decay).masked_fill(upper, 0)
    for i in range(1, chunk):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row[..., None] * sub).sum(-2)
    attn = attn + torch.eye(chunk, dtype=attn.dtype)
    v = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp()[..., None])
    s = state.float()
    out = torch.empty(H, n, chunk, dv, dtype=v.dtype)
    strict = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool), diagonal=1)
    for c in range(n):
        q_c, k_c, v_c, g_c = q[:, c], k[:, c], v[:, c], g[:, c]
        a = (q_c @ k_c.transpose(-1, -2) * decay[:, c]).masked_fill(strict, 0)
        v_new = v_c - k_cumdecay[:, c] @ s
        out[:, c] = (q_c * g_c.exp()[..., None]) @ s + a @ v_new
        s = s * g_c[:, -1, None, None].exp() + (k_c * (g_c[:, -1, None] - g_c).exp()[..., None]).transpose(-1, -2) @ v_new
    out = out.reshape(H, n * chunk, dv)[:, :S].transpose(0, 1)
    return out, s


# ---------------------------------------------------------------- state


class LinearState:
    def __init__(self, cfg: Config):
        self.conv = torch.zeros(cfg.conv_dim, cfg.conv_k)
        self.recurrent = torch.zeros(cfg.lv_heads, cfg.lk_dim, cfg.lv_dim, dtype=torch.float32)


class KVState:
    def __init__(self, cfg: Config, max_seq: int):
        self.key = torch.zeros(cfg.n_kv, max_seq, cfg.head_dim)
        self.value = torch.zeros(cfg.n_kv, max_seq, cfg.head_dim)


class State:
    def __init__(self, ref: "Qwen38Reference", max_seq: int):
        self.max_seq = max_seq
        self.layers = {
            i: (KVState(ref.cfg, max_seq) if ref.is_attn(i) else LinearState(ref.cfg)) for i in ref.layer_ids
        }


# ---------------------------------------------------------------- reference


class Qwen38Reference:
    """The reference interface (``_bringup_interface`` docstring) for Qwen3.8-27B."""

    def __init__(self, spec, path: str, layers=None, dtype=torch.float32):
        self.cfg = Config.load(path)
        self.ckpt = Checkpoint(path)
        self.dtype = dtype
        self.wdtype = weight_dtype(spec)
        self.layer_ids = list(range(self.cfg.num_layers)) if layers is None else [int(i) for i in layers]
        self.embed = self.ckpt.get(f"{PREFIX}embed_tokens.weight", self.wdtype)
        self.final_w = self.ckpt.get(f"{PREFIX}norm.weight", self.wdtype)
        self._lm_head = None  # loaded on first use: only the full stack's logits need it
        self.w = {i: self.ckpt.layer(i, self.wdtype) for i in self.layer_ids}
        c = self.cfg
        self.inv_freq = 1.0 / (c.rope_theta ** (torch.arange(0, c.rope_dim, 2, dtype=torch.int64).float() / c.rope_dim))

    # ---- structure
    def is_attn(self, layer: int) -> bool:
        return self.cfg.layer_types[layer] == "full_attention"

    def block_graph(self, layer):
        attn = Step("attention", ("attn_norm",), "attn_out", "attention", stateful=True)
        return [
            Step("attn_norm", ("in",), "attn_norm", "norm"),
            attn,
            Step("attn_residual", ("in", "attn_out"), "h_mid", "residual"),
            Step("ffn_norm", ("h_mid",), "ffn_norm", "norm"),
            Step("mlp", ("ffn_norm",), "mlp_out", "mlp"),
            Step("mlp_residual", ("h_mid", "mlp_out"), "out", "residual"),
        ]

    # ---- state
    def new_state(self, max_seq):
        return State(self, max_seq)

    def state_tensors(self, state, layer, length):
        st = state.layers[layer]
        if isinstance(st, KVState):
            return {"key": st.key[:, :length].clone(), "value": st.value[:, :length].clone()}
        return {"conv": st.conv.clone(), "recurrent": st.recurrent.clone()}

    def load_state(self, state, layer, tensors, length):
        st = state.layers[layer]
        if isinstance(st, KVState):
            st.key[:, :length] = tensors["key"][:, :length].to(st.key.dtype)
            st.value[:, :length] = tensors["value"][:, :length].to(st.value.dtype)
            return
        st.conv.copy_(tensors["conv"])
        st.recurrent.copy_(tensors["recurrent"])

    def rope(self, start: int, length: int):
        """cos / sin [length, rope_dim] (fp32) for positions [start, start + length), as Qwen3_5TextRotaryEmbedding."""
        pos = torch.arange(start, start + length, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def chunk_context(self, layer, start, length, state):
        extra = {}
        if self.is_attn(layer):
            cos, sin = self.rope(start, length)
            extra = {"cos": cos.to(self.dtype), "sin": sin.to(self.dtype)}
        return Ctx(layer, start, length, state.layers[layer] if state is not None else None, extra)

    # ---- components
    def component(self, layer, name):
        w, eps = self.w[layer], self.cfg.eps
        table = {
            "attn_norm": lambda ctx, x: rms_norm_1p(x, w["input_layernorm.weight"], eps),
            "attention": (lambda ctx, x: self.full_attention(layer, ctx, x))
            if self.is_attn(layer)
            else (lambda ctx, x: self.gated_deltanet(layer, ctx, x)),
            "attn_residual": lambda ctx, a, b: a + b,
            "ffn_norm": lambda ctx, x: rms_norm_1p(x, w["post_attention_layernorm.weight"], eps),
            "mlp": lambda ctx, x: self.mlp(layer, x),
            "mlp_residual": lambda ctx, a, b: a + b,
        }
        return table[name]

    def mlp(self, layer, x):
        w = self.w[layer]
        gate = F.linear(x, as_compute(w["mlp.gate_proj.weight"], x))
        up = F.linear(x, as_compute(w["mlp.up_proj.weight"], x))
        return F.linear(F.silu(gate) * up, as_compute(w["mlp.down_proj.weight"], x))

    def full_attention(self, layer, ctx, x):
        c, w = self.cfg, self.w[layer]
        st: KVState = ctx.state
        S, D = x.shape[0], c.head_dim
        start, end = ctx.start, ctx.start + S
        qg = F.linear(x, as_compute(w["self_attn.q_proj.weight"], x)).view(S, c.n_heads, 2 * D)
        q, gate = qg[..., :D], qg[..., D:].reshape(S, c.n_heads * D)
        q = rms_norm_1p(q, w["self_attn.q_norm.weight"], c.eps)
        k = rms_norm_1p(F.linear(x, as_compute(w["self_attn.k_proj.weight"], x)).view(S, c.n_kv, D),
                        w["self_attn.k_norm.weight"], c.eps)
        v = F.linear(x, as_compute(w["self_attn.v_proj.weight"], x)).view(S, c.n_kv, D)
        cos, sin = ctx.extra["cos"][:, None, :], ctx.extra["sin"][:, None, :]
        r = c.rope_dim
        q = torch.cat((q[..., :r] * cos + rotate_half(q[..., :r]) * sin, q[..., r:]), dim=-1)
        k = torch.cat((k[..., :r] * cos + rotate_half(k[..., :r]) * sin, k[..., r:]), dim=-1)
        st.key[:, start:end] = k.transpose(0, 1)
        st.value[:, start:end] = v.transpose(0, 1)
        rep = c.n_heads // c.n_kv
        qh = q.transpose(0, 1).reshape(c.n_kv, rep, S, D)  # head h = kv head h // rep
        o = torch.empty_like(qh)
        # query blocks at absolute multiples of ATTN_BLOCK (the same blocks chunked or one-shot) bound the score
        # matrix: [24, 1024, 56320] fp32 is 5.5 GB where a whole 5k chunk would take 27 GB
        b0 = start
        while b0 < end:
            b1 = min(end, (b0 // ATTN_BLOCK + 1) * ATTN_BLOCK)
            # causal over the absolute positions: query p sees keys [0, p]
            mask = torch.arange(b1)[None, :] <= torch.arange(b0, b1)[:, None]
            o[:, :, b0 - start : b1 - start] = F.scaled_dot_product_attention(
                qh[:, :, b0 - start : b1 - start], st.key[:, None, :b1], st.value[:, None, :b1], attn_mask=mask,
                scale=D**-0.5)
            b0 = b1
        o = o.reshape(c.n_heads, S, D).transpose(0, 1).reshape(S, c.n_heads * D)
        o = o * torch.sigmoid(gate)
        return F.linear(o, as_compute(w["self_attn.o_proj.weight"], o))

    def gated_deltanet(self, layer, ctx, x):
        c, w = self.cfg, self.w[layer]
        st: LinearState = ctx.state
        S = x.shape[0]
        p = "linear_attn."
        mixed = F.linear(x, as_compute(w[p + "in_proj_qkv.weight"], x))  # [S, conv_dim]
        z = F.linear(x, as_compute(w[p + "in_proj_z.weight"], x)).view(S, c.lv_heads, c.lv_dim)
        b = F.linear(x, as_compute(w[p + "in_proj_b.weight"], x))
        a = F.linear(x, as_compute(w[p + "in_proj_a.weight"], x))
        # causal depthwise conv over [the previous K - 1 inputs | this chunk], as upstream's cached continuation
        K = c.conv_k
        seq = torch.cat((st.conv.to(mixed.dtype), mixed.transpose(0, 1)), dim=-1)  # [conv_dim, K + S]
        st.conv.copy_(seq[:, -K:])
        conv_w = as_compute(w[p + "conv1d.weight"], mixed)  # [conv_dim, 1, K]
        out = F.conv1d(seq[None, :, 1:], conv_w, groups=c.conv_dim)[0]  # [conv_dim, S]
        mixed = F.silu(out).transpose(0, 1)
        q, k, v = torch.split(mixed, [c.key_dim, c.key_dim, c.value_dim], dim=-1)
        q = q.reshape(S, c.lk_heads, c.lk_dim)
        k = k.reshape(S, c.lk_heads, c.lk_dim)
        v = v.reshape(S, c.lv_heads, c.lv_dim)
        beta = torch.sigmoid(b)
        g = -w[p + "A_log"].float().exp() * F.softplus(a.float() + w[p + "dt_bias"].float())
        rep = c.lv_heads // c.lk_heads
        q = l2norm(q.float()).repeat_interleave(rep, dim=1)
        k = l2norm(k.float()).repeat_interleave(rep, dim=1)
        core, s = chunk_gated_delta_rule(q, k, v, g, beta, st.recurrent)
        st.recurrent.copy_(s)
        core = rms_norm_gated(core.to(x.dtype), w[p + "norm.weight"], z, c.eps).reshape(S, c.value_dim)
        return F.linear(core, as_compute(w[p + "out_proj.weight"], core))

    # ---- forward
    @property
    def lm_head(self):
        if self._lm_head is None:
            self._lm_head = self.ckpt.get("lm_head.weight", self.wdtype)
        return self._lm_head

    @torch.no_grad()
    def forward_chunk(self, tokens, start, state, rec=noop, logits_last_n=0):
        h = self.embed[tokens].to(self.dtype)
        rec("embed", h)
        for i in self.layer_ids:
            ctx = self.chunk_context(i, start, tokens.shape[0], state)
            h = run_block(self.block_graph(i), lambda name, i=i: self.component(i, name), ctx, h, rec, prefix=f"L{i}.")
        out = rms_norm_1p(h, self.final_w, self.cfg.eps)
        rec("final_norm", out)
        logits = None
        if logits_last_n:
            last = out[-logits_last_n:]
            logits = F.linear(last, as_compute(self.lm_head, last))
            rec("logits", logits)
        return out, logits
