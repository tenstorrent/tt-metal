# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""`ConditionalDecoder.forward` in plain torch, driven only by the flat weight export.

This exists to split one question into two. When the TTNN UNet misses its PCC gate
the cause is either *the graph is wrong* -- a swapped index, the wrong padding, a
GroupNorm where a LayerNorm belongs -- or *a TTNN op behaved differently than
expected*. Those need completely different debugging, and on rented silicon the
difference is expensive to establish by bisection.

So this reimplements the reference network from `flow_weights.npz` and nothing
else: no cosyvoice package, no diffusers, no matcha, no device. If it reproduces
the captured `flow.cfm_estimator` output then the graph is provably right and any
device failure is a TTNN question. If it does not, the mistake is here and costs
nothing to find.

It is also the executable statement of what the architecture *is*, which the
config file only implies -- `act_fn: 'gelu'`, for instance, reads like it might
select GEGLU (diffusers' default) and does not: `FeedForward.__init__` tests
`"gelu"` in a bare `if` before the `elif` chain, so the projection is a single
Linear followed by the erf GELU.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

EPS_LN = 1e-5
EPS_GN = 1e-5


def sinusoidal_pos_emb(t: torch.Tensor, dim: int, scale: float = 1000.0) -> torch.Tensor:
    """`SinusoidalPosEmb(dim)(t, scale=1000)`; t is `[B]`, result `[B, dim]`."""
    half = dim // 2
    step = math.log(10000.0) / (half - 1)
    freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -step)
    ang = scale * t.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat((ang.sin(), ang.cos()), dim=-1)


def _block1d(x: torch.Tensor, w: dict, p: str) -> torch.Tensor:
    """Conv1d(3, pad 1) -> GroupNorm(8) -> Mish, on `[B, C, T]`."""
    h = F.conv1d(x, w[f"{p}.block.0.weight"], w[f"{p}.block.0.bias"], padding=1)
    h = F.group_norm(h, 8, w[f"{p}.block.1.weight"], w[f"{p}.block.1.bias"], EPS_GN)
    return F.mish(h)


def _resnet(x: torch.Tensor, t_emb: torch.Tensor, w: dict, p: str) -> torch.Tensor:
    h = _block1d(x, w, f"{p}.block1")
    h = h + F.linear(F.mish(t_emb), w[f"{p}.mlp.1.weight"], w[f"{p}.mlp.1.bias"]).unsqueeze(-1)
    h = _block1d(h, w, f"{p}.block2")
    return h + F.conv1d(x, w[f"{p}.res_conv.weight"], w[f"{p}.res_conv.bias"])


def _transformer(x: torch.Tensor, w: dict, p: str, heads: int, dim_head: int) -> torch.Tensor:
    """BasicTransformerBlock on `[B, T, C]`: self-attn then a GELU feed-forward."""
    b, t, c = x.shape
    n = F.layer_norm(x, (c,), w[f"{p}.norm1.weight"], w[f"{p}.norm1.bias"], EPS_LN)

    def head_split(name):
        return F.linear(n, w[f"{p}.attn1.{name}.weight"]).view(b, t, heads, dim_head).transpose(1, 2)

    a = F.scaled_dot_product_attention(head_split("to_q"), head_split("to_k"), head_split("to_v"))
    a = a.transpose(1, 2).reshape(b, t, heads * dim_head)
    a = F.linear(a, w[f"{p}.attn1.to_out.0.weight"], w[f"{p}.attn1.to_out.0.bias"])
    x = a + x

    n = F.layer_norm(x, (c,), w[f"{p}.norm3.weight"], w[f"{p}.norm3.bias"], EPS_LN)
    f = F.linear(n, w[f"{p}.ff.net.0.proj.weight"], w[f"{p}.ff.net.0.proj.bias"])
    f = F.gelu(f, approximate="none")
    f = F.linear(f, w[f"{p}.ff.net.2.weight"], w[f"{p}.ff.net.2.bias"])
    return f + x


def _n_children(w: dict, prefix: str) -> int:
    base = prefix + "."
    return len({k[len(base) :].split(".", 1)[0] for k in w if k.startswith(base) and k[len(base)].isdigit()})


def conditional_decoder(
    w: dict,
    x: torch.Tensor,
    mu: torch.Tensor,
    t: torch.Tensor,
    spks: torch.Tensor | None = None,
    cond: torch.Tensor | None = None,
    *,
    prefix: str = "decoder.estimator",
    heads: int = 8,
    dim_head: int = 64,
) -> torch.Tensor:
    """x/mu/cond `[B, 80, T]`, spks `[B, 80]`, t `[B]` -> `[B, 80, T]`.

    Channel-first, matching the reference exactly; the TTNN port is the one that
    moves to channels-last.
    """
    p = prefix
    in_channels = w[f"{p}.time_mlp.linear_1.weight"].shape[1]

    emb = sinusoidal_pos_emb(t, in_channels)
    emb = F.linear(emb, w[f"{p}.time_mlp.linear_1.weight"], w[f"{p}.time_mlp.linear_1.bias"])
    emb = F.silu(emb)
    t_emb = F.linear(emb, w[f"{p}.time_mlp.linear_2.weight"], w[f"{p}.time_mlp.linear_2.bias"])

    h = torch.cat([x, mu], dim=1)
    if spks is not None:
        h = torch.cat([h, spks.unsqueeze(-1).expand(-1, -1, h.shape[-1])], dim=1)
    if cond is not None:
        h = torch.cat([h, cond], dim=1)

    skips = []
    n_down = _n_children(w, f"{p}.down_blocks")
    for i in range(n_down):
        q = f"{p}.down_blocks.{i}"
        h = _resnet(h, t_emb, w, f"{q}.0")
        h = h.transpose(1, 2)
        for j in range(_n_children(w, f"{q}.1")):
            h = _transformer(h, w, f"{q}.1.{j}", heads, dim_head)
        h = h.transpose(1, 2).contiguous()
        skips.append(h)
        # Downsample1D nests its conv under `.conv`; the last stage is a bare Conv1d.
        is_last = i == n_down - 1
        key = f"{q}.2" if is_last else f"{q}.2.conv"
        h = F.conv1d(h, w[f"{key}.weight"], w[f"{key}.bias"], stride=1 if is_last else 2, padding=1)

    for i in range(_n_children(w, f"{p}.mid_blocks")):
        q = f"{p}.mid_blocks.{i}"
        h = _resnet(h, t_emb, w, f"{q}.0")
        h = h.transpose(1, 2)
        for j in range(_n_children(w, f"{q}.1")):
            h = _transformer(h, w, f"{q}.1.{j}", heads, dim_head)
        h = h.transpose(1, 2).contiguous()

    n_up = _n_children(w, f"{p}.up_blocks")
    for i in range(n_up):
        q = f"{p}.up_blocks.{i}"
        skip = skips.pop()
        h = torch.cat([h[:, :, : skip.shape[-1]], skip], dim=1)
        h = _resnet(h, t_emb, w, f"{q}.0")
        h = h.transpose(1, 2)
        for j in range(_n_children(w, f"{q}.1")):
            h = _transformer(h, w, f"{q}.1.{j}", heads, dim_head)
        h = h.transpose(1, 2).contiguous()
        is_last = i == n_up - 1
        if is_last:
            h = F.conv1d(h, w[f"{q}.2.weight"], w[f"{q}.2.bias"], padding=1)
        else:
            h = F.conv_transpose1d(h, w[f"{q}.2.conv.weight"], w[f"{q}.2.conv.bias"], stride=2, padding=1)

    h = _block1d(h, w, f"{p}.final_block")
    return F.conv1d(h, w[f"{p}.final_proj.weight"], w[f"{p}.final_proj.bias"])
