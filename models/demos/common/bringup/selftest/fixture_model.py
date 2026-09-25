# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Synthetic test fixture for the reference-side scripts: random weights, one attention head, a KV state.

Not a model bring-up. It implements the reference interface (reference/interface.py) in as few lines as possible,
plus an "HF" twin that computes the same thing through a different code path, so check_hf, check_reference and
generate_golden can be exercised without a checkpoint. Used as the ``hooks`` module of the selftest spec.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.demos.common.bringup.reference.interface import Ctx, Step, noop, run_block

H, V, LAYERS, SEED = 64, 97, 3, 0


def weights():
    g = torch.Generator().manual_seed(SEED)
    r = lambda *s: torch.randn(*s, generator=g) * 0.2  # noqa: E731
    return {
        "embed": r(V, H),
        "layers": [
            {"wq": r(H, H), "wk": r(H, H), "wv": r(H, H), "w1": r(2 * H, H), "w2": r(H, 2 * H)} for _ in range(LAYERS)
        ],
        "head": r(V, H),
    }


def norm(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class State:
    def __init__(self, layers, max_seq):
        self.k = {i: torch.zeros(max_seq, H) for i in layers}
        self.v = {i: torch.zeros(max_seq, H) for i in layers}


class Reference:
    def __init__(self, spec=None, layers=None, dtype=torch.float32, broken_graph=False):
        self.w = weights()
        self.layer_ids = list(range(LAYERS)) if layers is None else list(layers)
        self.broken_graph = broken_graph

    def new_state(self, max_seq):
        return State(self.layer_ids, max_seq)

    def state_tensors(self, state, layer, length):
        return {"key": state.k[layer][:length].clone(), "value": state.v[layer][:length].clone()}

    def load_state(self, state, layer, tensors, length):
        state.k[layer][:length] = tensors["key"][:length]
        state.v[layer][:length] = tensors["value"][:length]

    def block_graph(self, layer):
        return [
            Step("attn_norm", ("in",), "attn_norm", "norm"),
            Step("attention", ("attn_norm",), "attn_out", "attention", stateful=True),
            Step("attn_residual", ("in", "attn_out"), "h_mid", "residual"),
            Step("ffn_norm", ("h_mid",), "ffn_norm", "norm"),
            Step("mlp", ("ffn_norm",), "mlp_out", "mlp"),
            Step("mlp_residual", ("h_mid", "mlp_out"), "out", "residual"),
        ]

    def chunk_context(self, layer, start, length, state):
        return Ctx(layer, start, length, state)

    def component(self, layer, name):
        w = self.w["layers"][layer]

        def attention(ctx, x):
            end = ctx.start + x.shape[0]
            ctx.state.k[layer][ctx.start : end] = x @ w["wk"].T
            ctx.state.v[layer][ctx.start : end] = x @ w["wv"].T
            k, v = ctx.state.k[layer][:end], ctx.state.v[layer][:end]
            s = (x @ w["wq"].T) @ k.T / H**0.5
            mask = torch.arange(end)[None] > torch.arange(ctx.start, end)[:, None]
            return torch.softmax(s.masked_fill(mask, float("-inf")), -1) @ v

        table = {
            "attn_norm": lambda ctx, x: norm(x),
            "attention": attention,
            "attn_residual": lambda ctx, a, b: a + b,
            "ffn_norm": lambda ctx, x: norm(x),
            "mlp": lambda ctx, x: F.silu(x @ w["w1"].T) @ w["w2"].T,
            "mlp_residual": lambda ctx, a, b: a + b + (1e-3 if self.broken_graph else 0.0),
        }
        return table[name]

    @torch.no_grad()
    def forward_chunk(self, tokens, start, state, rec=noop, logits_last_n=0):
        h = self.w["embed"][tokens]
        rec("embed", h)
        for i in self.layer_ids:
            ctx = self.chunk_context(i, start, tokens.shape[0], state)
            comp = (lambda i: lambda name: self.component(i, name))(i)
            steps = self.block_graph(i)
            if self.broken_graph:  # the forward differs from what the graph claims (mlp_residual adds 1e-3)
                h = self._forward_raw(i, h, ctx, rec)
            else:
                h = run_block(steps, comp, ctx, h, rec, prefix=f"L{i}.")
        out = norm(h)
        rec("final_norm", out)
        logits = out[-logits_last_n:] @ self.w["head"].T if logits_last_n else None
        return out, logits

    def _forward_raw(self, i, h, ctx, rec):
        rec(f"L{i}.in", h)
        c = lambda n: self.component(i, n)  # noqa: E731
        x = c("attn_norm")(ctx, h)
        rec(f"L{i}.attn_norm", x)
        a = c("attention")(ctx, x)
        rec(f"L{i}.attn_out", a)
        m = h + a
        rec(f"L{i}.h_mid", m)
        f = c("ffn_norm")(ctx, m)
        rec(f"L{i}.ffn_norm", f)
        o = c("mlp")(ctx, f)
        rec(f"L{i}.mlp_out", o)
        out = m + o
        rec(f"L{i}.out", out)
        return out


class _HFLayer(torch.nn.Module):
    """Independent one-shot implementation (causal attention over the whole sequence, no state)."""

    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, h):
        w = self.w
        x = norm(h)
        s = (x @ w["wq"].T) @ (x @ w["wk"].T).T / H**0.5
        s = s.masked_fill(torch.triu(torch.ones(s.shape, dtype=torch.bool), 1), float("-inf"))
        h = h + torch.softmax(s, -1) @ (x @ w["wv"].T)
        return (h + F.silu(norm(h) @ w["w1"].T) @ w["w2"].T,)


class _HFOut:
    def __init__(self, logits):
        self.logits = logits


class HFTwin(torch.nn.Module):
    def __init__(self, num_layers=None):
        super().__init__()
        self.w = weights()
        n = num_layers or LAYERS
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(_HFLayer(self.w["layers"][i]) for i in range(n))

    def forward(self, tokens, use_cache=False):
        h = self.w["embed"][tokens[0]]
        for layer in self.model.layers:
            h = layer(h)[0]
        return _HFOut((norm(h) @ self.w["head"].T)[None])


class Tokenizer:
    """Byte-level: one token per UTF-8 byte mod V; BOS = 1."""

    bos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + b % (V - 2) for b in text.encode()[:200000]]}


# ---- hooks module interface (the selftest spec points "hooks" at this module)
def reference(spec, layers=None, dtype=torch.float32):
    return Reference(spec, layers, dtype, broken_graph=bool(spec.get("fixture.broken_graph")))


def hf_model(spec, num_layers):
    return HFTwin(num_layers)


def tokenizer(spec):
    return Tokenizer()


# ---- fake device hooks: the reference plus small deterministic noise, for testing the device-side test helpers
NOISE = {"value": 1e-3}


def _noisy(t: torch.Tensor, seed: int) -> torch.Tensor:
    if not t.is_floating_point() or NOISE["value"] == 0:
        return t
    g = torch.Generator().manual_seed(seed)
    return t + NOISE["value"] * t.abs().mean() * torch.randn(t.shape, generator=g)


def device_component(mesh, spec, layer, name):
    ref = Reference(spec, [layer])
    cpu = ref.component(layer, name)

    def fn(ctx, *x):
        state = ref.new_state(ctx.extra["max_seq"])
        if ctx.extra["prefix_len"]:
            ref.load_state(state, layer, ctx.extra["state_prefix"], ctx.extra["prefix_len"])
        return _noisy(cpu(Ctx(layer, ctx.start, ctx.length, state), *x), layer)

    return fn


class _FakeState:
    def __init__(self, ref, max_seq):
        self.ref, self.s = ref, ref.new_state(max_seq)

    def load_prefix(self, layer, tensors, length):
        self.ref.load_state(self.s, layer, tensors, length)

    def to_torch(self, layer, length):
        return self.ref.state_tensors(self.s, layer, length)


class FakeDeviceModel:
    load_seconds = 0.0

    def __init__(self, spec, layers):
        self.ref = Reference(spec, layers)

    def new_state(self, max_seq):
        return _FakeState(self.ref, max_seq)

    def embed(self, tokens):
        return self.ref.w["embed"][tokens]

    def from_host(self, h):
        return h.float()

    def to_host(self, h):
        return h

    def layer(self, i, h, start, state):
        ctx = self.ref.chunk_context(i, start, h.shape[0], state.s)
        out = run_block(self.ref.block_graph(i), lambda n: self.ref.component(i, n), ctx, h)
        return _noisy(out, 100 + i)

    def final_norm(self, h):
        return norm(h)

    def logits(self, hidden, rows):
        return hidden[rows] @ self.ref.w["head"].T

    def free(self, h):
        pass

    def sync(self):
        pass


def device_model(mesh, spec, layers, lm_head=True):
    return FakeDeviceModel(spec, layers)
