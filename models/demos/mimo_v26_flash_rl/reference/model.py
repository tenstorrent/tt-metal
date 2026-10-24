# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU reference for MiMo-V2.6-Flash (mimo_v2), following the checkpoint's modeling_mimo_v2.py.

Per layer: x -> input_layernorm -> attention -> + x -> post_attention_layernorm -> (dense SwiGLU MLP on layer 0 |
routed MoE elsewhere: sigmoid router, noaux_tc top-8; the gate input and correction bias are rounded to config
moe_router_dtype (bf16, as the upstream trace ran them), logits fp32, 256 mxfp4 experts, no shared expert) -> + residual.

Attention: fused qkv (fp8, per-TP-rank slabs), head_dim 192 for q/k and 128 for v, v scaled by
attention_value_scale before the cache, partial RoPE (half-split rotate_half) on the first 64 dims of each head.
Full layers (hybrid_layer_pattern 0): 4 kv heads, rope theta 1e7, causal, no sink. Sliding layers: 8 kv heads, theta
1e4, window 128 (key j visible to query i when i - 128 < j <= i), and a per-head sink logit appended to the scaled
scores (max subtracted, fp32 softmax, sink column dropped). Scale head_dim ** -0.5.

Weights are held in ``weight_dtype(spec)``; compute is fp32 (``dtype``); norm variance, softmax, router logits and
RoPE tables are fp32 as upstream. State per layer: key [kv_heads, seq, 192] after RoPE, value [kv_heads, seq, 128]
after the value scale, every position (the HF trace's past_key_values layout).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

from models.demos.mimo_v26_flash_rl.reference._bringup_interface import (
    Ctx,
    Step,
    as_compute,
    noop,
    run_block,
    weight_dtype,
)
from models.demos.mimo_v26_flash_rl.reference.weights import Checkpoint

Q_BLOCK = 512  # query rows per attention block, aligned to absolute positions (chunked == one-shot)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return as_compute(w, x) * xf.to(x.dtype)


def rope_tables(inv_freq: torch.Tensor, start: int, length: int):
    pos = torch.arange(start, start + length, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class LayerState:
    def __init__(self, kv_heads: int, max_seq: int, head_dim: int, v_head_dim: int, dtype: torch.dtype):
        self.key = torch.zeros(kv_heads, max_seq, head_dim, dtype=dtype)
        self.value = torch.zeros(kv_heads, max_seq, v_head_dim, dtype=dtype)


class State:
    def __init__(self, layers: dict[int, LayerState]):
        self.layers = layers


class Layer:
    """One decoder layer's config and weights."""

    def __init__(self, ck: Checkpoint, i: int, cfg: dict):
        self.idx = i
        self.is_swa = cfg["hybrid_layer_pattern"][i] == 1
        self.is_moe = bool(cfg["moe_layer_freq"][i])
        p = f"model.layers.{i}"
        sfx = "swa_" if self.is_swa else ""
        self.heads = cfg.get(f"{sfx}num_attention_heads", cfg["num_attention_heads"])
        self.kv_heads = cfg.get(f"{sfx}num_key_value_heads", cfg["num_key_value_heads"])
        self.head_dim = cfg.get(f"{sfx}head_dim", cfg["head_dim"])
        self.v_head_dim = cfg.get(f"{sfx}v_head_dim", cfg["v_head_dim"])
        self.rope_dim = int(self.head_dim * cfg["partial_rotary_factor"])
        self.scale = self.head_dim**-0.5
        self.v_scale = cfg.get("attention_value_scale")
        self.window = cfg["sliding_window"] if self.is_swa else None
        self.eps = cfg["layernorm_epsilon"]
        theta = cfg["swa_rope_theta"] if self.is_swa else cfg["rope_theta"]
        self.inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.rope_dim, 2, dtype=torch.int64).to(torch.float32) / self.rope_dim)
        )

        self.attn_norm_w = ck.get(f"{p}.input_layernorm.weight")
        self.ffn_norm_w = ck.get(f"{p}.post_attention_layernorm.weight")
        self.wq, self.wk, self.wv = ck.qkv(
            f"{p}.self_attn.qkv_proj",
            self.heads * self.head_dim,
            self.kv_heads * self.head_dim,
            self.kv_heads * self.v_head_dim,
        )
        self.wo = ck.get(f"{p}.self_attn.o_proj.weight")
        sink = f"{p}.self_attn.attention_sink_bias"
        has_sink = cfg.get("add_swa_attention_sink_bias" if self.is_swa else "add_full_attention_sink_bias", False)
        self.sink = ck.get(sink) if has_sink else None

        if self.is_moe:
            self.top_k = cfg["num_experts_per_tok"]
            self.n_experts = cfg["n_routed_experts"]
            assert cfg["scoring_func"] == "sigmoid" and cfg["topk_method"] == "noaux_tc"
            assert cfg["n_group"] == 1 and cfg["topk_group"] == 1, "group-limited routing not implemented"
            self.norm_topk = cfg["norm_topk_prob"]
            self.routed_scale = cfg.get("routed_scaling_factor") or 1.0
            self.gate_w = ck.get(f"{p}.mlp.gate.weight")
            # The checkpoint stores the correction bias fp32, but config moe_router_dtype (bf16) is the dtype upstream
            # runs the gate parameters in: the HF trace's top-k reproduces only with the bias rounded to bf16.
            self.router_dtype = getattr(torch, cfg.get("moe_router_dtype") or "float32")
            self.gate_bias = ck.raw(f"{p}.mlp.gate.e_score_correction_bias").to(self.router_dtype)

            def load(e):
                q = f"{p}.mlp.experts.{e}"
                return ck.mxfp4(f"{q}.gate_proj"), ck.mxfp4(f"{q}.up_proj"), ck.mxfp4(f"{q}.down_proj")

            with ThreadPoolExecutor(16) as ex:
                ws = list(ex.map(load, range(self.n_experts)))
            self.w_gate = [w[0] for w in ws]
            self.w_up = [w[1] for w in ws]
            self.w_down = [w[2] for w in ws]
        else:
            self.w_gate = ck.fp8(f"{p}.mlp.gate_proj")
            self.w_up = ck.fp8(f"{p}.mlp.up_proj")
            self.w_down = ck.fp8(f"{p}.mlp.down_proj")

    # ---- block graph
    def graph(self) -> list[Step]:
        steps = [
            Step("attn_norm", ("in",), "attn_norm", kind="norm"),
            Step("attention", ("attn_norm",), "attn_out", kind="attention", stateful=True),
            Step("attn_residual", ("in", "attn_out"), "h", kind="residual"),
            Step("ffn_norm", ("h",), "ffn_norm", kind="norm"),
        ]
        if self.is_moe:
            steps += [
                Step("router", ("ffn_norm",), "router", kind="router"),
                Step("experts", ("ffn_norm", "router"), "mlp_out", kind="moe"),
            ]
        else:
            steps.append(Step("mlp", ("ffn_norm",), "mlp_out", kind="mlp"))
        steps.append(Step("mlp_residual", ("h", "mlp_out"), "out", kind="residual"))
        return steps

    def component(self, name: str):
        return {
            "attn_norm": lambda ctx, x: rms_norm(x, self.attn_norm_w, self.eps),
            "attention": self.attention,
            "attn_residual": lambda ctx, a, b: a + b,
            "ffn_norm": lambda ctx, x: rms_norm(x, self.ffn_norm_w, self.eps),
            "router": self.router,
            "experts": self.experts,
            "mlp": self.mlp,
            "mlp_residual": lambda ctx, a, b: a + b,
        }[name]

    # ---- steps
    def attention(self, ctx: Ctx, x: torch.Tensor) -> torch.Tensor:
        S, start, st = x.shape[0], ctx.start, ctx.state
        H, KV, D, Dv, R = self.heads, self.kv_heads, self.head_dim, self.v_head_dim, self.rope_dim
        rep = H // KV
        q = F.linear(x, as_compute(self.wq, x)).view(S, H, D).transpose(0, 1)
        k = F.linear(x, as_compute(self.wk, x)).view(S, KV, D).transpose(0, 1)
        v = F.linear(x, as_compute(self.wv, x)).view(S, KV, Dv).transpose(0, 1)
        if self.v_scale is not None:
            v = v * self.v_scale
        cos, sin = ctx.extra["cos"].to(x.dtype), ctx.extra["sin"].to(x.dtype)
        q = torch.cat((q[..., :R] * cos + rotate_half(q[..., :R]) * sin, q[..., R:]), dim=-1)
        k = torch.cat((k[..., :R] * cos + rotate_half(k[..., :R]) * sin, k[..., R:]), dim=-1)
        st.key[:, start : start + S] = k
        st.value[:, start : start + S] = v

        end = start + S
        out = torch.empty(H, S, Dv, dtype=x.dtype)
        qg = q.reshape(KV, rep, S, D)
        qblk = Q_BLOCK if self.window is not None else Q_BLOCK // 2  # full layers: scores over the whole prefix
        for qs in range((start // qblk) * qblk, end, qblk):
            a, b = max(qs, start), min(qs + qblk, end)
            n = b - a
            k_lo = 0 if self.window is None else max(0, a - self.window + 1)
            keys = st.key[:, k_lo:b]  # [KV, T, D]
            vals = st.value[:, k_lo:b]
            qb = qg[:, :, a - start : b - start].reshape(KV, rep * n, D)
            scores = torch.matmul(qb, keys.transpose(1, 2)).view(KV, rep, n, -1) * self.scale
            qi = torch.arange(a, b)[:, None]
            kj = torch.arange(k_lo, b)[None, :]
            masked = kj > qi
            if self.window is not None:
                masked |= kj <= qi - self.window
            scores = scores.masked_fill(masked, float("-inf"))
            if self.sink is not None:
                sink = as_compute(self.sink, scores).view(KV, rep, 1, 1).expand(KV, rep, n, 1)
                scores = torch.cat((scores, sink), dim=-1)
            scores = scores - scores.amax(dim=-1, keepdim=True)
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
            if self.sink is not None:
                probs = probs[..., :-1]
            o = torch.matmul(probs.reshape(KV, rep * n, -1), vals).view(H, n, Dv)
            out[:, a - start : b - start] = o
        out = out.transpose(0, 1).reshape(S, H * Dv)
        return F.linear(out, as_compute(self.wo, out))

    def router(self, ctx: Ctx, x: torch.Tensor) -> torch.Tensor:
        """Dense routing weights [S, n_experts] (fp32): the top-k experts' normalized sigmoid scores, zero elsewhere."""
        # Upstream's gate sees bf16 activations (moe_router_dtype); rounding here makes the selection a function of the
        # bf16 input, so it is the same whether x arrives fp32 (one forward) or bf16 (a stored boundary, the device).
        x = x.to(self.router_dtype).to(torch.float32)
        logits = F.linear(x, self.gate_w.to(torch.float32))
        scores = logits.sigmoid()
        choice = scores + self.gate_bias.to(torch.float32)
        idx = torch.topk(choice, k=self.top_k, dim=-1, sorted=False).indices
        w = scores.gather(1, idx)
        if self.top_k > 1 and self.norm_topk:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        w = w * self.routed_scale
        return torch.zeros_like(scores).scatter_(1, idx, w)

    def experts(self, ctx: Ctx, x: torch.Tensor, routing: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape, dtype=torch.float32)
        sel = routing != 0
        for e in torch.nonzero(sel.any(0)).flatten().tolist():
            rows = torch.nonzero(sel[:, e]).flatten()
            xe = x[rows]
            g = F.linear(xe, as_compute(self.w_gate[e], xe))
            u = F.linear(xe, as_compute(self.w_up[e], xe))
            h = F.silu(g) * u
            y = F.linear(h, as_compute(self.w_down[e], h))
            out.index_add_(0, rows, y * routing[rows, e].unsqueeze(-1))
        return out.to(x.dtype)

    def mlp(self, ctx: Ctx, x: torch.Tensor) -> torch.Tensor:
        g = F.linear(x, as_compute(self.w_gate, x))
        u = F.linear(x, as_compute(self.w_up, x))
        h = F.silu(g) * u
        return F.linear(h, as_compute(self.w_down, h))


class MimoReference:
    """The bring-up interface (reference/_bringup_interface.py) over the given layers."""

    def __init__(self, spec, layers=None, dtype=torch.float32):
        self.dtype = dtype or torch.float32
        self.ck = Checkpoint(spec.hf_dir, weight_dtype(spec))
        self.cfg = self.ck.config
        self.num_layers = self.cfg["num_hidden_layers"]
        self.layer_ids = list(layers) if layers is not None else list(range(self.num_layers))
        self.eps = self.cfg["layernorm_epsilon"]
        self.embed_w = self.ck.get("model.embed_tokens.weight")
        self.norm_w = self.ck.get("model.norm.weight")
        self._lm_head = None
        self.layers = {i: Layer(self.ck, i, self.cfg) for i in self.layer_ids}

    @property
    def lm_head(self) -> torch.Tensor:
        if self._lm_head is None:
            self._lm_head = self.ck.get("lm_head.weight")
        return self._lm_head

    # ---- interface
    def new_state(self, max_seq: int) -> State:
        return State(
            {i: LayerState(L.kv_heads, max_seq, L.head_dim, L.v_head_dim, self.dtype) for i, L in self.layers.items()}
        )

    def state_tensors(self, state: State, layer: int, length: int) -> dict[str, torch.Tensor]:
        st = state.layers[layer]
        return {"key": st.key[:, :length].clone(), "value": st.value[:, :length].clone()}

    def load_state(self, state: State, layer: int, tensors: dict[str, torch.Tensor], length: int) -> None:
        st = state.layers[layer]
        st.key[:, :length] = tensors["key"][:, :length].to(st.key.dtype)
        st.value[:, :length] = tensors["value"][:, :length].to(st.value.dtype)

    def block_graph(self, layer: int) -> list[Step]:
        return self.layers[layer].graph()

    def component(self, layer: int, step_name: str):
        return self.layers[layer].component(step_name)

    def chunk_context(self, layer: int, start: int, length: int, state: State) -> Ctx:
        cos, sin = rope_tables(self.layers[layer].inv_freq, start, length)
        return Ctx(layer=layer, start=start, length=length, state=state.layers[layer], extra={"cos": cos, "sin": sin})

    def forward_chunk(self, tokens: torch.Tensor, start: int, state: State, rec=noop, logits_last_n: int = 0):
        tokens = tokens.reshape(-1).long()
        S = tokens.shape[0]
        h = self.embed_w[tokens].to(self.dtype)
        rec("embed", h)
        for i in self.layer_ids:
            L = self.layers[i]
            ctx = self.chunk_context(i, start, S, state)
            h = run_block(L.graph(), L.component, ctx, h, rec=rec, prefix=f"L{i}.")
        # The whole stack returns the final norm's output; a layer subset returns its last layer's output (its logits,
        # when asked for, are the final norm + lm_head over it).
        full = self.layer_ids == list(range(self.num_layers))
        normed = rms_norm(h, self.norm_w, self.eps) if (full or logits_last_n) else None
        if full:
            h = normed
            rec("final_norm", h)
        logits = None
        if logits_last_n:
            tail = normed[-logits_last_n:]
            logits = F.linear(tail, as_compute(self.lm_head, tail))
            rec("logits", logits)
        return h, logits
