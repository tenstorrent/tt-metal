# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent FP32 CPU reference of one Qwen3.5 / 3.6 / 3.8 Gated DeltaNet layer (prefill, streaming state).

Pure torch, no ttnn. Math follows transformers ``Qwen3_5GatedDeltaNet`` (qwen3_5 and qwen3_5_moe share it):
fused ``in_proj_qkv`` -> depthwise causal conv (kernel 4, no bias) + SiLU -> q | k | v; ``beta = sigmoid(b)``;
``g = -exp(A_log) * softplus(a + dt_bias)`` per V head; q/k L2-normalized (eps 1e-6), q scaled by ``K^-0.5``,
V head ``j`` reads K head ``j // (Nv / Nk)``; token-by-token delta rule with an FP32 ``[Nv, K, V]`` state;
gated RMSNorm (raw weight, norm before gate, ``silu(z)``); ``out_proj``. Everything is computed in FP32 from the
stored (bf16) weights, without the intermediate bf16 roundings of the HF module. The recurrence is the naive
per-token form, not the chunked algorithm under test.

``GdnState`` carries what a following chunk needs: the last ``K - 1`` raw (pre-conv) qkv rows in HF channel
order and the recurrent state. Running chunks one after another with the returned state equals one pass over
their concatenation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

PREFIX = "linear_attn."
WEIGHT_NAMES = (
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "in_proj_a.weight",
    "in_proj_b.weight",
    "out_proj.weight",
    "conv1d.weight",
    "A_log",
    "dt_bias",
    "norm.weight",
)


@dataclass(frozen=True)
class GdnShape:
    hidden: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel: int
    eps: float

    @classmethod
    def from_text_config(cls, text_config: dict) -> "GdnShape":
        return cls(
            hidden=text_config["hidden_size"],
            num_k_heads=text_config["linear_num_key_heads"],
            num_v_heads=text_config["linear_num_value_heads"],
            head_k_dim=text_config["linear_key_head_dim"],
            head_v_dim=text_config["linear_value_head_dim"],
            conv_kernel=text_config["linear_conv_kernel_dim"],
            eps=text_config["rms_norm_eps"],
        )

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim


@dataclass
class GdnState:
    conv: torch.Tensor  # [K - 1, conv_dim] fp32, raw qkv rows preceding the next chunk (HF channel order)
    recurrent: torch.Tensor  # [Nv, K, V] fp32

    @classmethod
    def zeros(cls, shape: GdnShape) -> "GdnState":
        return cls(
            conv=torch.zeros(shape.conv_kernel - 1, shape.conv_dim),
            recurrent=torch.zeros(shape.num_v_heads, shape.head_k_dim, shape.head_v_dim),
        )


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def delta_rule_recurrence(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive gated delta rule. q, k [T, H, K] (normalized, q scaled); v [T, H, V]; g, beta [T, H]; state [H, K, V]."""
    s = state.clone()
    out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], dtype=torch.float32)
    decay = g.exp()
    for t in range(q.shape[0]):
        s.mul_(decay[t, :, None, None])
        kv_mem = torch.bmm(k[t].unsqueeze(1), s).squeeze(1)  # [H, V] = k_t^T S
        delta = (v[t] - kv_mem) * beta[t, :, None]
        s.baddbmm_(k[t].unsqueeze(2), delta.unsqueeze(1))  # S += k_t delta^T
        out[t] = torch.bmm(q[t].unsqueeze(1), s).squeeze(1)
    return out, s


def gdn_layer_reference(
    weights: dict[str, torch.Tensor], shape: GdnShape, x: torch.Tensor, state: GdnState | None = None
) -> tuple[torch.Tensor, GdnState]:
    """One chunk of the GDN layer. x [T, hidden] -> (out [T, hidden] fp32, state after the chunk)."""
    if state is None:
        state = GdnState.zeros(shape)
    w = {name: weights[PREFIX + name].float() for name in WEIGHT_NAMES}
    xf = x.float()
    T = xf.shape[0]
    Nk, Nv, Dk, Dv = shape.num_k_heads, shape.num_v_heads, shape.head_k_dim, shape.head_v_dim
    kc = shape.conv_kernel

    qkv = xf @ w["in_proj_qkv.weight"].T  # [T, C]
    z = xf @ w["in_proj_z.weight"].T  # [T, Nv*Dv]
    b = xf @ w["in_proj_b.weight"].T  # [T, Nv]
    a = xf @ w["in_proj_a.weight"].T  # [T, Nv]

    padded = torch.cat([state.conv, qkv], dim=0)  # [K-1+T, C]
    taps = w["conv1d.weight"][:, 0, :]  # [C, K]; tap j multiplies row t - (K-1) + j
    conv = sum(padded[j : j + T] * taps[:, j] for j in range(kc))
    conv = F.silu(conv)
    q = conv[:, : shape.key_dim].reshape(T, Nk, Dk)
    k = conv[:, shape.key_dim : 2 * shape.key_dim].reshape(T, Nk, Dk)
    v = conv[:, 2 * shape.key_dim :].reshape(T, Nv, Dv)

    beta = torch.sigmoid(b)
    g = -w["A_log"].exp() * F.softplus(a + w["dt_bias"])
    q = _l2norm(q) * Dk**-0.5
    k = _l2norm(k)
    group = Nv // Nk
    q = q.repeat_interleave(group, dim=1)
    k = k.repeat_interleave(group, dim=1)

    o, recurrent = delta_rule_recurrence(q, k, v, g, beta, state.recurrent)

    normed = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + shape.eps) * w["norm.weight"]
    gated = (normed * F.silu(z.reshape(T, Nv, Dv))).reshape(T, Nv * Dv)
    out = gated @ w["out_proj.weight"].T
    new_conv = padded[-(kc - 1) :].clone()
    return out, GdnState(conv=new_conv, recurrent=recurrent)


def per_device_conv_columns(conv: torch.Tensor, shape: GdnShape, tp: int) -> torch.Tensor:
    """Reorder conv-state columns from HF order [q | k | v] to the TP order [q_0 k_0 v_0 | q_1 k_1 v_1 | ...]
    that ``tp_common.prepare_gdn_qkv`` gives the device (each rank's K heads with its contiguous V heads)."""
    qs, ks, vs = conv.split([shape.key_dim, shape.key_dim, shape.value_dim], dim=-1)
    parts = []
    for rank in range(tp):
        for t, width in ((qs, shape.key_dim // tp), (ks, shape.key_dim // tp), (vs, shape.value_dim // tp)):
            parts.append(t[..., rank * width : (rank + 1) * width])
    return torch.cat(parts, dim=-1)
