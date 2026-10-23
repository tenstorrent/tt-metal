# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU reference for the Qwen3.8-27B text decoder (Qwen3.5 architecture), chunked prefill.

Layers alternate three Gated DeltaNet (linear attention) blocks with one gated full-attention block; every block
has a dense SwiGLU MLP. Both block types share one graph shape:

    attn_norm -> attention (stateful) -> attn_residual -> mlp_norm -> mlp -> mlp_residual (= out)

State per layer:
    full attention   key, value  [n_kv_heads, max_seq, head_dim]   (key after k_norm and RoPE)
    linear attention conv        [conv_dim, kernel - 1]           the last kernel-1 pre-conv q|k|v inputs (fixed)
                     recurrent   [n_v_heads, head_k_dim, head_v_dim] fp32 delta-rule state (fixed)
A chunk starting at position 0 starts a new sequence: the fixed-size state resets there.

Weights are read layer by layer from the safetensors checkpoint (read-only); only the selected layers, the
embedding, the final norm and the LM head are loaded. Every weight is held in ``wdtype`` (the spec's weight_dtype,
bf16 = the checkpoint's own numbers) and cast to the activation's dtype in the op that uses it; the norms add their
``1 +`` in fp32 at use, as upstream does.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from models.demos.qwen38_27b.reference._bringup_interface import Ctx, Step, as_compute, noop, run_block

PREFIX = "model.language_model."
ATTN_BLOCK = 512  # query rows per causal attention block (absolute positions)
DELTA_CHUNK = 64  # the delta rule's intra-chunk block (HF's torch_chunk_gated_delta_rule uses 64)


class Config:
    def __init__(self, hf_dir: Path):
        raw = json.loads((Path(hf_dir) / "config.json").read_text())
        c = raw.get("text_config", raw)
        self.hidden = c["hidden_size"]
        self.num_layers = c["num_hidden_layers"]
        self.layer_types = list(c["layer_types"])
        self.vocab = c["vocab_size"]
        self.eps = c["rms_norm_eps"]
        self.inter = c["intermediate_size"]
        # full attention
        self.n_heads = c["num_attention_heads"]
        self.n_kv = c["num_key_value_heads"]
        self.head_dim = c["head_dim"]
        rp = c.get("rope_parameters") or {}
        self.rope_theta = rp.get("rope_theta", c.get("rope_theta"))
        self.rotary_dim = int(self.head_dim * rp.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0)))
        # linear attention (Gated DeltaNet)
        self.lk_heads = c["linear_num_key_heads"]
        self.lv_heads = c["linear_num_value_heads"]
        self.lk_dim = c["linear_key_head_dim"]
        self.lv_dim = c["linear_value_head_dim"]
        self.conv_kernel = c["linear_conv_kernel_dim"]
        self.key_dim = self.lk_heads * self.lk_dim
        self.value_dim = self.lv_heads * self.lv_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim

    def is_linear(self, layer: int) -> bool:
        return self.layer_types[layer] == "linear_attention"


class Checkpoint:
    """Tensor name -> safetensors shard; reads one tensor at a time."""

    def __init__(self, hf_dir: Path):
        self.dir = Path(hf_dir)
        self.map = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, name: str, dtype) -> torch.Tensor:
        with safe_open(str(self.dir / self.map[name]), framework="pt") as f:
            return f.get_tensor(name).to(dtype).contiguous()

    def layer(self, i: int, names: list[str], dtype) -> dict[str, torch.Tensor]:
        by_file: dict[str, list[str]] = {}
        for n in names:
            full = f"{PREFIX}layers.{i}.{n}"
            by_file.setdefault(self.map[full], []).append(n)
        out = {}
        for fname, ns in by_file.items():
            with safe_open(str(self.dir / fname), framework="pt") as f:
                for n in ns:
                    out[n] = f.get_tensor(f"{PREFIX}layers.{i}.{n}").to(dtype).contiguous()
        return out


LINEAR_NAMES = [
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.conv1d.weight",
    "linear_attn.dt_bias",
    "linear_attn.A_log",
    "linear_attn.norm.weight",
    "linear_attn.out_proj.weight",
]
ATTN_NAMES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
]
COMMON_NAMES = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


# ---- math
def lin(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x @ w.T with the stored weight cast to the activation's dtype for this op only."""
    return F.linear(x, as_compute(w, x))


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3.5 RMSNorm: x / rms(x) * (1 + w), computed in fp32; ``w`` is the checkpoint's zero-centred weight."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).to(x.dtype)


def causal_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int, scale: float) -> torch.Tensor:
    """Causal GQA attention of the queries at [start, start + S) over keys [0, end): q [Hq, S, D], k, v [Hkv, end, D].

    Query rows go in blocks aligned to absolute positions (multiples of ATTN_BLOCK), each over only the keys it can
    see, so the score matrix stays small at 56k context and a chunk computes the same blocks as one-shot prefill."""
    S = q.shape[1]
    out = torch.empty(q.shape[0], S, v.shape[-1], dtype=q.dtype)
    p = start
    while p < start + S:
        e = min((p // ATTN_BLOCK + 1) * ATTN_BLOCK, start + S)
        mask = torch.arange(e)[None, :] <= torch.arange(p, e)[:, None]
        out[:, p - start : e - start] = F.scaled_dot_product_attention(
            q[None, :, p - start : e - start],
            k[None, :, :e],
            v[None, :, :e],
            attn_mask=mask,
            scale=scale,
            enable_gqa=True,
        )[0]
        p = e
    return out


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def rope_tables(cfg: Config, max_seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin [max_seq, rotary_dim] (rotate-half layout). Text positions are equal on the three mRoPE axes, so the
    interleaved mRoPE reduces to plain RoPE over the rotary dims."""
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rotary_dim, 2, dtype=torch.int64).float() / cfg.rotary_dim))
    freqs = torch.arange(max_seq, dtype=torch.float32)[:, None] * inv[None]
    emb = torch.cat([freqs, freqs], -1)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [heads, S, D]; RoPE on the first rotary_dim dims, the rest passes through."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    x1, x2 = xr[..., : r // 2], xr[..., r // 2 :]
    rot = torch.cat([-x2, x1], -1)
    return torch.cat([xr * cos + rot * sin, xp], -1)


def chunk_gated_delta_rule(q, k, v, g, beta, state, chunk=DELTA_CHUNK):
    """Gated delta rule over one prefill chunk, blockwise (the WY form HF's torch_chunk_gated_delta_rule uses).

    q, k [H, S, Dk] (l2-normalised, q not yet scaled), v [H, S, Dv], g, beta [H, S], all fp32; ``state``
    [H, Dk, Dv] fp32 is the recurrent state before the chunk and is updated in place. Returns [H, S, Dv]."""
    H, S, Dk = k.shape
    Dv = v.shape[-1]
    pad = (chunk - S % chunk) % chunk
    q = F.pad(q * Dk**-0.5, (0, 0, 0, pad))
    k = F.pad(k, (0, 0, 0, pad))
    v = F.pad(v, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    n = (S + pad) // chunk
    v_beta = (v * beta[..., None]).reshape(H, n, chunk, Dv)
    k_beta = (k * beta[..., None]).reshape(H, n, chunk, Dk)
    q, k = q.reshape(H, n, chunk, Dk), k.reshape(H, n, chunk, Dk)
    g = g.reshape(H, n, chunk).cumsum(-1)
    incl = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool), diagonal=0)
    decay = (g[..., :, None] - g[..., None, :]).tril().exp().tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * decay).masked_fill(incl, 0)
    for i in range(1, chunk):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp()[..., None])
    strict = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool), diagonal=1)
    out = torch.empty(H, n, chunk, Dv)
    s = state
    for i in range(n):
        q_i, k_i, g_i = q[:, i], k[:, i], g[:, i]
        a = (q_i @ k_i.transpose(-1, -2) * decay[:, i]).masked_fill(strict, 0)
        v_new = value[:, i] - k_cumdecay[:, i] @ s
        out[:, i] = (q_i * g_i[..., None].exp()) @ s + a @ v_new
        s = (
            s * g_i[:, -1, None, None].exp()
            + (k_i * (g_i[:, -1, None] - g_i).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    state.copy_(s)
    return out.reshape(H, n * chunk, Dv)[:, :S]


# ---- state
class State:
    def __init__(self, cfg: Config, layers: list[int], max_seq: int, dtype):
        self.max_seq = max_seq
        self.key, self.value, self.conv, self.recurrent = {}, {}, {}, {}
        for i in layers:
            if cfg.is_linear(i):
                self.conv[i] = torch.zeros(cfg.conv_dim, cfg.conv_kernel - 1, dtype=dtype)
                self.recurrent[i] = torch.zeros(cfg.lv_heads, cfg.lk_dim, cfg.lv_dim, dtype=torch.float32)
            else:
                self.key[i] = torch.zeros(cfg.n_kv, max_seq, cfg.head_dim, dtype=dtype)
                self.value[i] = torch.zeros(cfg.n_kv, max_seq, cfg.head_dim, dtype=dtype)


class Reference:
    def __init__(self, hf_dir, layers=None, dtype=torch.float32, wdtype=torch.bfloat16):
        self.cfg = cfg = Config(hf_dir)
        self.dtype = dtype
        self.layer_ids = list(range(cfg.num_layers)) if layers is None else sorted(int(i) for i in layers)
        ck = Checkpoint(hf_dir)
        self.embed = ck.get(PREFIX + "embed_tokens.weight", wdtype)
        self.final_norm_w = ck.get(PREFIX + "norm.weight", wdtype)
        self.lm_head = ck.get("lm_head.weight", wdtype)
        self.w = {}
        for i in self.layer_ids:
            names = COMMON_NAMES + (LINEAR_NAMES if cfg.is_linear(i) else ATTN_NAMES)
            w = ck.layer(i, names, wdtype)
            if cfg.is_linear(i):
                w["linear_attn.conv1d.weight"] = w["linear_attn.conv1d.weight"].reshape(cfg.conv_dim, 1, -1)
            self.w[i] = w
        self._rope = (torch.empty(0), torch.empty(0))

    # ---- interface
    def new_state(self, max_seq):
        return State(self.cfg, self.layer_ids, max_seq, self.dtype)

    def state_tensors(self, state, layer, length):
        if self.cfg.is_linear(layer):
            return {"conv": state.conv[layer].clone(), "recurrent": state.recurrent[layer].clone()}
        return {"key": state.key[layer][:, :length].clone(), "value": state.value[layer][:, :length].clone()}

    def load_state(self, state, layer, tensors, length):
        if self.cfg.is_linear(layer):
            state.conv[layer].copy_(tensors["conv"])
            state.recurrent[layer].copy_(tensors["recurrent"])
            return
        state.key[layer][:, :length] = tensors["key"][:, :length]
        state.value[layer][:, :length] = tensors["value"][:, :length]

    def block_graph(self, layer):
        return [
            Step("attn_norm", ("in",), "attn_norm", "norm"),
            Step("attention", ("attn_norm",), "attn_out", "attention", stateful=True),
            Step("attn_residual", ("in", "attn_out"), "h_mid", "residual"),
            Step("mlp_norm", ("h_mid",), "mlp_norm", "norm"),
            Step("mlp", ("mlp_norm",), "mlp_out", "mlp"),
            Step("mlp_residual", ("h_mid", "mlp_out"), "out", "residual"),
        ]

    def rope(self, end: int):
        if self._rope[0].shape[0] < end:
            self._rope = rope_tables(self.cfg, max(end, 2 * self._rope[0].shape[0]))
        return self._rope

    def chunk_context(self, layer, start, length, state):
        extra = {}
        if not self.cfg.is_linear(layer):
            cos, sin = self.rope(start + length)
            extra = {"cos": cos[start : start + length], "sin": sin[start : start + length]}
        return Ctx(layer, start, length, state, extra)

    def component(self, layer, name):
        w, eps = self.w[layer], self.cfg.eps
        table = {
            "attn_norm": lambda ctx, x: rms_norm(x, w["input_layernorm.weight"], eps),
            "attention": self._linear_attention if self.cfg.is_linear(layer) else self._full_attention,
            "attn_residual": lambda ctx, a, b: a + b,
            "mlp_norm": lambda ctx, x: rms_norm(x, w["post_attention_layernorm.weight"], eps),
            "mlp": lambda ctx, x: lin(
                F.silu(lin(x, w["mlp.gate_proj.weight"])) * lin(x, w["mlp.up_proj.weight"]), w["mlp.down_proj.weight"]
            ),
            "mlp_residual": lambda ctx, a, b: a + b,
        }
        return table[name]

    # ---- token mixers
    def _full_attention(self, ctx: Ctx, x: torch.Tensor) -> torch.Tensor:
        cfg, w, i = self.cfg, self.w[ctx.layer], ctx.layer
        S, D = x.shape[0], cfg.head_dim
        start, end = ctx.start, ctx.start + x.shape[0]
        qg = lin(x, w["self_attn.q_proj.weight"]).view(S, cfg.n_heads, 2 * D)
        q, gate = qg[..., :D], qg[..., D:].reshape(S, cfg.n_heads * D)
        q = rms_norm(q, w["self_attn.q_norm.weight"], cfg.eps).transpose(0, 1)
        k = rms_norm(
            lin(x, w["self_attn.k_proj.weight"]).view(S, cfg.n_kv, D), w["self_attn.k_norm.weight"], cfg.eps
        ).transpose(0, 1)
        v = lin(x, w["self_attn.v_proj.weight"]).view(S, cfg.n_kv, D).transpose(0, 1)
        cos, sin = ctx.extra["cos"], ctx.extra["sin"]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        ctx.state.key[i][:, start:end] = k
        ctx.state.value[i][:, start:end] = v
        keys, vals = ctx.state.key[i][:, :end], ctx.state.value[i][:, :end]
        o = causal_sdpa(q, keys, vals, start, D**-0.5)
        o = o.transpose(0, 1).reshape(S, cfg.n_heads * D) * torch.sigmoid(gate)
        return lin(o, w["self_attn.o_proj.weight"])

    def _linear_attention(self, ctx: Ctx, x: torch.Tensor) -> torch.Tensor:
        cfg, w, i = self.cfg, self.w[ctx.layer], ctx.layer
        S = x.shape[0]
        if ctx.start == 0:
            ctx.state.conv[i].zero_()
            ctx.state.recurrent[i].zero_()
        mixed = lin(x, w["linear_attn.in_proj_qkv.weight"]).T  # [conv_dim, S]
        z = lin(x, w["linear_attn.in_proj_z.weight"]).view(S, cfg.lv_heads, cfg.lv_dim)
        b = lin(x, w["linear_attn.in_proj_b.weight"])
        a = lin(x, w["linear_attn.in_proj_a.weight"])
        # causal depthwise conv over [previous kernel-1 inputs | this chunk], then SiLU
        xin = torch.cat([ctx.state.conv[i], mixed], -1)
        ctx.state.conv[i].copy_(xin[:, -(cfg.conv_kernel - 1) :])
        mixed = F.silu(
            F.conv1d(xin[None], as_compute(w["linear_attn.conv1d.weight"], xin), groups=cfg.conv_dim)[0]
        ).T  # [S, conv_dim]
        q, k, v = torch.split(mixed, [cfg.key_dim, cfg.key_dim, cfg.value_dim], -1)
        rep = cfg.lv_heads // cfg.lk_heads
        q = q.reshape(S, cfg.lk_heads, cfg.lk_dim).repeat_interleave(rep, 1)
        k = k.reshape(S, cfg.lk_heads, cfg.lk_dim).repeat_interleave(rep, 1)
        v = v.reshape(S, cfg.lv_heads, cfg.lv_dim)
        beta = b.float().sigmoid()
        # g = -exp(A_log) * softplus(a + dt_bias), in fp32
        g = -w["linear_attn.A_log"].float().exp() * F.softplus(a.float() + w["linear_attn.dt_bias"].float())
        q, k = l2norm(q.float()), l2norm(k.float())
        core = chunk_gated_delta_rule(
            q.transpose(0, 1).contiguous(),
            k.transpose(0, 1).contiguous(),
            v.float().transpose(0, 1).contiguous(),
            g.T.contiguous(),
            beta.T.contiguous(),
            ctx.state.recurrent[i],
        ).transpose(
            0, 1
        )  # [S, Hv, Dv]
        # gated RMSNorm per value head: norm(core) * w * silu(z)
        core = core.to(x.dtype).float()
        core = core * torch.rsqrt(core.pow(2).mean(-1, keepdim=True) + cfg.eps)
        core = (as_compute(w["linear_attn.norm.weight"], x) * core.to(x.dtype)) * F.silu(z.float())
        return lin(core.to(x.dtype).reshape(S, cfg.value_dim), w["linear_attn.out_proj.weight"])

    # ---- forward
    @torch.no_grad()
    def forward_chunk(self, tokens, start, state, rec=noop, logits_last_n=0):
        h = self.embed[tokens].to(self.dtype)
        rec("embed", h)
        for i in self.layer_ids:
            ctx = self.chunk_context(i, start, tokens.shape[0], state)
            h = run_block(self.block_graph(i), lambda name, i=i: self.component(i, name), ctx, h, rec, prefix=f"L{i}.")
        out = rms_norm(h, self.final_norm_w, self.cfg.eps)
        rec("final_norm", out)
        logits = lin(out[-logits_last_n:], self.lm_head) if logits_last_n else None
        if logits is not None:
            rec("logits", logits)
        return out, logits
