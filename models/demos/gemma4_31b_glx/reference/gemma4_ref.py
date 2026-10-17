# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU reference for the Gemma-4 31B-it text decoder (pure torch, reads the HF safetensors directly).

Matches transformers' Gemma4TextModel (5.12) op for op, in fp32:
  embed = embed_tokens[ids] * sqrt(H)
  per layer (sliding_attention or full_attention):
    attn_norm      x * rsqrt(mean(x^2) + eps) * w              (Gemma-4 RMSNorm is x * w, not 1 + w)
    attention      q = rope(q_norm(q_proj)), k = rope(k_norm(k_proj)), v = v_norm(v_proj) (unscaled RMS norm);
                   full layers have no v_proj (attention_k_eq_v): v = v_norm(raw k_proj). Scale 1.0, causal;
                   sliding layers see the last 1024 positions (kv > q - 1024). GQA 32 q heads over 16 (sliding,
                   head_dim 256) or 4 (full, head_dim 512) kv heads.
                   RoPE: sliding theta 1e4 on all 256 dims; full theta 1e6 "proportional", 25 %: the first 64
                   frequencies of a 512-dim table, the other 192 are 0 (identity), rotate-half layout.
    post_attn_norm, attn_residual (in + .), ffn_norm, mlp (gelu_tanh(gate) * up -> down), post_ffn_norm,
    ffn_residual ((h_mid + .) * layer_scalar)
  final_norm, logits = 30 * tanh((final_norm @ embed^T) / 30)

State per layer: key [Hkv, max_seq, D] (after k_norm + RoPE) and value [Hkv, max_seq, D] (after v_norm), fp32.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from models.demos.gemma4_31b_glx.reference._bringup_interface import Ctx, Step, noop, run_block

PREFIX = "model.language_model."
Q_BLOCK = 512  # query rows per attention block (bounds the score matrix at long context)


def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Gemma4RMSNorm, the same ops as HF (pow(-0.5), weight applied as x * w)."""
    x = x.float()
    y = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return y * w if w is not None else y


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Config:
    def __init__(self, ckpt: Path):
        c = json.loads((ckpt / "config.json").read_text())
        t = c.get("text_config", c)
        self.hidden = t["hidden_size"]
        self.num_layers = t["num_hidden_layers"]
        self.n_heads = t["num_attention_heads"]
        self.n_kv = t["num_key_value_heads"]
        self.n_kv_global = t.get("num_global_key_value_heads") or self.n_kv
        self.head_dim = t["head_dim"]
        self.global_head_dim = t.get("global_head_dim") or self.head_dim
        self.k_eq_v = bool(t.get("attention_k_eq_v"))
        self.eps = t["rms_norm_eps"]
        self.window = t["sliding_window"]
        self.layer_types = t["layer_types"]
        self.rope = t["rope_parameters"]
        self.softcap = t.get("final_logit_softcapping")
        self.vocab = t["vocab_size"]
        assert not t.get("enable_moe_block") and not t.get("hidden_size_per_layer_input"), "31B is dense, no PLE"
        assert not t.get("num_kv_shared_layers"), "no KV sharing in 31B"

    def is_sliding(self, layer: int) -> bool:
        return self.layer_types[layer] == "sliding_attention"

    def attn_dims(self, layer: int) -> tuple[int, int]:
        """(kv heads, head_dim) of a layer."""
        if self.is_sliding(layer):
            return self.n_kv, self.head_dim
        return (self.n_kv_global if self.k_eq_v else self.n_kv), self.global_head_dim

    def inv_freq(self, layer: int) -> torch.Tensor:
        """HF's per-layer-type inverse frequencies (default / proportional RoPE), length head_dim / 2."""
        p = self.rope[self.layer_types[layer]]
        dim = self.attn_dims(layer)[1]
        base = p["rope_theta"]
        if p.get("rope_type", "default") == "default":
            return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        assert p["rope_type"] == "proportional", p
        angles = int(p.get("partial_rotary_factor", 1.0) * dim // 2)
        rot = 1.0 / (base ** (torch.arange(0, 2 * angles, 2, dtype=torch.int64).float() / dim))
        return torch.cat([rot, torch.zeros(dim // 2 - angles)]) / p.get("factor", 1.0)


class State:
    """Per-layer KV cache: key / value [Hkv, max_seq, D]."""

    def __init__(self, cfg: Config, layers, max_seq: int, dtype):
        self.max_seq = max_seq
        self.k, self.v = {}, {}
        for i in layers:
            hk, d = cfg.attn_dims(i)
            self.k[i] = torch.empty(hk, max_seq, d, dtype=dtype)
            self.v[i] = torch.empty(hk, max_seq, d, dtype=dtype)


class Gemma4Reference:
    def __init__(self, ckpt, layers=None, dtype=torch.float32, threads: int = 16):
        self.ckpt = Path(ckpt)
        self.cfg = Config(self.ckpt)
        self.dtype = dtype or torch.float32
        self.layer_ids = list(range(self.cfg.num_layers)) if layers is None else sorted(layers)
        self.w = self._load(threads)
        self.inv_freq = {i: self.cfg.inv_freq(i) for i in self.layer_ids}

    # ---- weights -----------------------------------------------------------------------------------------------
    def _load(self, threads: int) -> dict:
        index = json.loads((self.ckpt / "model.safetensors.index.json").read_text())["weight_map"]
        want = {PREFIX + "embed_tokens.weight", PREFIX + "norm.weight"}
        want |= {
            k
            for k in index
            if k.startswith(PREFIX + "layers.") and int(k[len(PREFIX + "layers.") :].split(".")[0]) in self.layer_ids
        }
        by_file: dict[str, list[str]] = {}
        for k in sorted(want):
            by_file.setdefault(index[k], []).append(k)
        out = {}

        def read(fname_keys):
            fname, keys = fname_keys
            with safe_open(str(self.ckpt / fname), framework="pt") as f:
                for k in keys:
                    out[k[len(PREFIX) :]] = f.get_tensor(k).to(self.dtype).contiguous()

        jobs = [(f, ks[j::threads]) for f, ks in by_file.items() for j in range(threads)]
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(read, [j for j in jobs if j[1]]))
        return out

    def lw(self, layer: int, name: str) -> torch.Tensor:
        return self.w[f"layers.{layer}.{name}"]

    # ---- interface ---------------------------------------------------------------------------------------------
    def new_state(self, max_seq: int) -> State:
        return State(self.cfg, self.layer_ids, max_seq, self.dtype)

    def state_tensors(self, state: State, layer: int, length: int) -> dict:
        return {"key": state.k[layer][:, :length].clone(), "value": state.v[layer][:, :length].clone()}

    def load_state(self, state: State, layer: int, tensors: dict, length: int) -> None:
        state.k[layer][:, :length] = tensors["key"][:, :length].to(self.dtype)
        state.v[layer][:, :length] = tensors["value"][:, :length].to(self.dtype)

    def block_graph(self, layer: int) -> list[Step]:
        return [
            Step("attn_norm", ("in",), "attn_norm", "norm"),
            Step("attention", ("attn_norm",), "attn_out", "attention", stateful=True),
            Step("post_attn_norm", ("attn_out",), "attn_post", "norm"),
            Step("attn_residual", ("in", "attn_post"), "h_mid", "residual"),
            Step("ffn_norm", ("h_mid",), "ffn_norm", "norm"),
            Step("mlp", ("ffn_norm",), "mlp_out", "mlp"),
            Step("post_ffn_norm", ("mlp_out",), "mlp_post", "norm"),
            Step("ffn_residual", ("h_mid", "mlp_post"), "out", "residual"),
        ]

    def chunk_context(self, layer: int, start: int, length: int, state: State) -> Ctx:
        pos = torch.arange(start, start + length, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[layer][None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return Ctx(
            layer, start, length, state, extra={"cos": emb.cos().to(self.dtype), "sin": emb.sin().to(self.dtype)}
        )

    def component(self, layer: int, name: str):
        eps = self.cfg.eps
        if name in ("attn_norm", "post_attn_norm", "ffn_norm", "post_ffn_norm"):
            w = self.lw(
                layer,
                {
                    "attn_norm": "input_layernorm",
                    "post_attn_norm": "post_attention_layernorm",
                    "ffn_norm": "pre_feedforward_layernorm",
                    "post_ffn_norm": "post_feedforward_layernorm",
                }[name]
                + ".weight",
            )
            return lambda ctx, x: rms_norm(x, w, eps)
        if name == "attn_residual":
            return lambda ctx, a, b: a + b
        if name == "ffn_residual":
            scalar = self.lw(layer, "layer_scalar")
            return lambda ctx, a, b: (a + b) * scalar
        if name == "mlp":
            g, u, d = (self.lw(layer, f"mlp.{p}_proj.weight") for p in ("gate", "up", "down"))
            return lambda ctx, x: F.linear(F.gelu(F.linear(x, g), approximate="tanh") * F.linear(x, u), d)
        if name == "attention":
            return lambda ctx, x: self.attention(layer, ctx, x)
        raise KeyError(name)

    # ---- attention ---------------------------------------------------------------------------------------------
    def attention(self, layer: int, ctx: Ctx, x: torch.Tensor) -> torch.Tensor:
        cfg, eps = self.cfg, self.cfg.eps
        hk, d = cfg.attn_dims(layer)
        s, start = x.shape[0], ctx.start
        end = start + s
        cos, sin = ctx.extra["cos"][:, None, :], ctx.extra["sin"][:, None, :]

        q = F.linear(x, self.lw(layer, "self_attn.q_proj.weight")).view(s, cfg.n_heads, d)
        q = rms_norm(q, self.lw(layer, "self_attn.q_norm.weight"), eps)
        q = (q * cos) + (rotate_half(q) * sin)
        k_raw = F.linear(x, self.lw(layer, "self_attn.k_proj.weight")).view(s, hk, d)
        v_w = self.w.get(f"layers.{layer}.self_attn.v_proj.weight")
        v_raw = k_raw if v_w is None else F.linear(x, v_w).view(s, hk, d)
        k = rms_norm(k_raw, self.lw(layer, "self_attn.k_norm.weight"), eps)
        k = (k * cos) + (rotate_half(k) * sin)
        v = rms_norm(v_raw, None, eps)

        st = ctx.state
        st.k[layer][:, start:end] = k.transpose(0, 1)
        st.v[layer][:, start:end] = v.transpose(0, 1)

        g = cfg.n_heads // hk
        qh = q.transpose(0, 1).reshape(hk, g, s, d)  # q head h -> kv head h // g (HF repeat_kv)
        window = cfg.window if cfg.is_sliding(layer) else None
        out = torch.empty(hk, g, s, d, dtype=q.dtype)
        for qs in range(0, s, Q_BLOCK):
            qe = min(s, qs + Q_BLOCK)
            p0, p1 = start + qs, start + qe  # absolute query positions [p0, p1)
            k0 = max(0, p0 - window + 1) if window else 0
            kb, vb = st.k[layer][:, k0:p1], st.v[layer][:, k0:p1]
            scores = torch.matmul(qh[:, :, qs:qe], kb.transpose(-1, -2)[:, None])  # [hk, g, rows, L]; scale 1.0
            qpos = torch.arange(p0, p1)[:, None]
            kpos = torch.arange(k0, p1)[None, :]
            mask = kpos > qpos
            if window:
                mask |= kpos <= qpos - window
            scores.masked_fill_(mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out[:, :, qs:qe] = torch.matmul(probs, vb[:, None])
        out = out.reshape(cfg.n_heads, s, d).transpose(0, 1).reshape(s, cfg.n_heads * d)
        return F.linear(out, self.lw(layer, "self_attn.o_proj.weight"))

    # ---- forward -----------------------------------------------------------------------------------------------
    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        scale = torch.tensor(self.cfg.hidden**0.5).to(self.dtype)
        return self.w["embed_tokens.weight"][tokens] * scale

    def head(self, h: torch.Tensor) -> torch.Tensor:
        logits = F.linear(h, self.w["embed_tokens.weight"])  # tied lm_head
        if self.cfg.softcap:
            logits = torch.tanh(logits / self.cfg.softcap) * self.cfg.softcap
        return logits

    @torch.no_grad()
    def forward_chunk(self, tokens, start, state, rec=noop, logits_last_n=0):
        tokens = torch.as_tensor(tokens).long()
        h = self.embed(tokens)
        rec("embed", h)
        for i in self.layer_ids:
            ctx = self.chunk_context(i, start, tokens.shape[0], state)
            h = run_block(self.block_graph(i), lambda name, i=i: self.component(i, name), ctx, h, rec, prefix=f"L{i}.")
        out = rms_norm(h, self.w["norm.weight"], self.cfg.eps)
        rec("final_norm", out)
        logits = None
        if logits_last_n:
            logits = self.head(out[-logits_last_n:])
            rec("logits", logits)
        return out, logits
