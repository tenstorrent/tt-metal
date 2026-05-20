# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Distributed Flux1 DiT (Diffusion Transformer) using the ttml framework.

Only the transformer is implemented here.  Text encoders, scheduler, and
VAE stay on torch/CPU (see generate.py).

TP strategy (Megatron-LM style — same as the Qwen3 example):
  - Hidden states are **replicated** across TP devices.
  - Attention Q/K/V:       ColumnParallel  (shard heads)
  - Attention O:            RowParallel     (shard input, all-reduce)
  - MLP ff1 (up):          ColumnParallel  (shard intermediate)
  - MLP ff2 (down):        RowParallel     (shard input, all-reduce)
  - AdaLN modulation:      ColumnParallel  gather_output=True (replicated out)
  - Single block proj_out: RowParallel     (shard input, all-reduce)
  - x_embedder:            ColumnParallel  gather_output=True
  - context_embedder:      ColumnParallel  gather_output=True
  - proj_out (final):      Replicated
  - Norms, timestep embed: Replicated      (no TP sharding)

Communication per double block:  4 all-reduces  (attn O + attn add_O + FF + FF_ctx)
Communication per single block:  1 all-reduce   (proj_out)

Optimization log (steps 2-3 steady-state, target ~820ms):
  # | Change                                              | ms   | delta
  0 | Baseline                                            | 1726 | —
  1 | Fused QKV (3 linears → 1 ColumnParallel)            | 1672 | -54
  2 | joint_scaled_dot_product_attention                   | 1230 | -442
  3 | split_query_key_value_and_split_heads                | 1175 | -55
  4 | no_grad context                                      | 1175 |   0
  5 | LayerNormNoAffineFunction fused                      | 1105 | -70
  6 | _adaln (fused layer_norm with scale/shift)           | 1035 | -70
  7 | _chunk4d views + final _adaln + RowParallel bias     | 1007 | -28
  — | Reverted to fully differentiable attention           | 1523.5 | baseline

  Feed-forward and single-block MLP use ``GeluTanhApproximate``: the same closed form as
  ``torch.nn.functional.gelu(x, approximate='tanh')`` (see class docstring), not
  ``ttml.ops.unary.gelu`` (exact GELU).

CODING RULES (do not repeat past mistakes):
  - ``ttnn.sum`` / ``ttnn.mean`` / ``ttnn.max`` / ``ttnn.min`` accept a LIST of dims
    via ``dim=[...]``. NEVER write ``for d in range(N): x = ttnn.sum(x, dim=d, ...)``;
    use ``ttnn.sum(x, dim=[0, 1, 2], keepdim=True)`` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, ModuleList, Parameter


# =====================================================================
# Empty-init context manager
# =====================================================================

_empty_init = False


class empty_init:
    """Skip tensor value initialisation during model construction.

    All weight/bias tensors are allocated directly on device via ``ttnn.empty``
    (no CPU data, no host->device transfer, no tilisation).  Use when you
    plan to load pretrained weights immediately after construction::

        with empty_init():
            model = DistributedFlux1Transformer(config, shard_dim=shard_dim)
        load_weights_from_hf_distributed(model, hf_state_dict, config, ...)
    """

    def __enter__(self):
        global _empty_init
        self._prev = _empty_init
        _empty_init = True
        return self

    def __exit__(self, *args):
        global _empty_init
        _empty_init = self._prev


def is_empty_init():
    return _empty_init


# =====================================================================
# Utilities
# =====================================================================


def _get_device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _get_tp_size():
    ctx = ttml.autograd.AutoContext.get_instance()
    return ctx.get_parallelism_context().get_tp_size()


def _tile_pad(dim: int) -> int:
    return ((dim + 31) // 32) * 32


def _make_empty_on_device(shape):
    """Allocate an empty bfloat16 TILE tensor directly on device."""
    device = _get_device()
    padded = list(shape)
    padded[-2] = _tile_pad(padded[-2])
    padded[-1] = _tile_pad(padded[-1])
    ttnn_tensor = ttnn.empty(padded, ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    return ttml.autograd.create_tensor(ttnn_tensor)


def _make_replicated(data_np):
    return ttml.autograd.Tensor.from_numpy(data_np, ttnn.Layout.TILE, ttnn.bfloat16)


def _make_replicated_zeros(shape):
    if is_empty_init():
        return _make_empty_on_device(shape)
    return _make_replicated(np.zeros(shape, dtype=np.float32))


def _make_replicated_ones(shape):
    if is_empty_init():
        return _make_empty_on_device(shape)
    return _make_replicated(np.ones(shape, dtype=np.float32))


def _make_replicated_weight(shape, std=0.02):
    if is_empty_init():
        return _make_empty_on_device(shape)
    return _make_replicated((torch.randn(shape) * std).float().numpy())


def _make_sharded_weight(shape, shard_dim_tensor, shard_dim_mesh, std=0.02):
    if is_empty_init():
        per_device = list(shape)
        per_device[shard_dim_tensor] //= _get_tp_size()
        return _make_empty_on_device(per_device)
    device = _get_device()
    data = (torch.randn(shape) * std).float().numpy()
    mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, shard_dim_tensor, shard_dim_mesh)
    return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.bfloat16, mapper)


def _make_sharded_zeros(shape, shard_dim_tensor, shard_dim_mesh):
    if is_empty_init():
        per_device = list(shape)
        per_device[shard_dim_tensor] //= _get_tp_size()
        return _make_empty_on_device(per_device)
    device = _get_device()
    data = np.zeros(shape, dtype=np.float32)
    mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, shard_dim_tensor, shard_dim_mesh)
    return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.bfloat16, mapper)


def _linear(x, weight, bias=None):
    return ttml.ops.linear.linear(x, weight, bias)



# =====================================================================
# Configuration
# =====================================================================


@dataclass
class Flux1Config:
    patch_size: int = 1
    in_channels: int = 64
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    out_channels: int = 64
    guidance_embeds: bool = False
    axes_dims_rope: tuple = (16, 56, 56)

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def mlp_hidden_dim(self) -> int:
        return 4 * self.inner_dim


def create_flux1_config_from_hf(hf_config) -> Flux1Config:
    out_channels = getattr(hf_config, "out_channels", None) or hf_config.in_channels
    return Flux1Config(
        patch_size=hf_config.patch_size,
        in_channels=hf_config.in_channels,
        num_layers=hf_config.num_layers,
        num_single_layers=hf_config.num_single_layers,
        attention_head_dim=hf_config.attention_head_dim,
        num_attention_heads=hf_config.num_attention_heads,
        joint_attention_dim=hf_config.joint_attention_dim,
        pooled_projection_dim=hf_config.pooled_projection_dim,
        out_channels=out_channels,
        guidance_embeds=hf_config.guidance_embeds,
        axes_dims_rope=tuple(hf_config.axes_dims_rope),
    )


# =====================================================================
# Custom autograd Functions
# =====================================================================

# Same constants as PyTorch ``F.gelu(..., approximate='tanh')`` / HF ``gelu-approximate``.
_GELU_TANH_C = 0.044715
_GELU_TANH_K = math.sqrt(2.0 / math.pi)  # sqrt(2/pi), not 2/pi


class Typecast(ttml.autograd.Function):
    """Differentiable dtype cast between bfloat16 and float32."""

    @staticmethod
    def forward(ctx, x, dtype):
        ctx.orig_dtype = x.get_value().dtype
        ctx.target_dtype = dtype
        return ttml.autograd.create_tensor(ttnn.typecast(x.get_value(), dtype))

    @staticmethod
    def backward(ctx, grad_output):
        g = grad_output.get_value() if hasattr(grad_output, "get_value") else grad_output
        return ttnn.typecast(g, ctx.orig_dtype)


def _to_float32(x):
    return Typecast.apply(x, ttnn.DataType.FLOAT32)


def _to_bfloat16(x):
    return Typecast.apply(x, ttnn.DataType.BFLOAT16)


class GeluTanhApproximate(ttml.autograd.Function):
    """Tan approximate GELU matching ``torch.nn.functional.gelu(x, approximate='tanh')``.

    .. math::

        \\mathrm{GELU}(x)=\\tfrac12\\,x\\,\\bigl(1+\\tanh(\\sqrt{2/\\pi}\\,(x+0.044715\\,x^3))\\bigr)

    Forward uses elementwise ``ttnn`` ops only (no ``ttnn.gelu`` shortcut). Backward is the
    analytical derivative of that expression, so it stays consistent with PyTorch autograd.
    """

    @staticmethod
    def forward(ctx, x):
        x_val = x.get_value()
        ctx.save_for_backward(x_val)
        x2 = ttnn.mul(x_val, x_val)
        x3 = ttnn.mul(x2, x_val)
        inner = ttnn.add(x_val, ttnn.mul(x3, _GELU_TANH_C))
        u = ttnn.mul(inner, _GELU_TANH_K)
        t = ttnn.tanh(u)
        one_plus_t = ttnn.add(t, 1.0)
        out = ttnn.mul(ttnn.mul(x_val, 0.5), one_plus_t)
        return ttml.autograd.create_tensor(out)

    @staticmethod
    def backward(ctx, grad_output):
        (x_val,) = ctx.saved_tensors
        g = grad_output.get_value() if hasattr(grad_output, "get_value") else grad_output
        x2 = ttnn.mul(x_val, x_val)
        x3 = ttnn.mul(x2, x_val)
        inner = ttnn.add(x_val, ttnn.mul(x3, _GELU_TANH_C))
        u = ttnn.mul(inner, _GELU_TANH_K)
        t = ttnn.tanh(u)
        t2 = ttnn.mul(t, t)
        sech2 = ttnn.subtract(ttnn.ones_like(t2), t2)
        dup_dx = ttnn.mul(ttnn.add(ttnn.mul(x2, 3.0 * _GELU_TANH_C), 1.0), _GELU_TANH_K)
        term_a = ttnn.mul(ttnn.add(t, 1.0), 0.5)
        term_b = ttnn.mul(ttnn.mul(ttnn.mul(x_val, 0.5), sech2), dup_dx)
        dy_dx = ttnn.add(term_a, term_b)
        return ttnn.mul(g, dy_dx)


class ApplyRoPE(ttml.autograd.Function):
    """Apply pre-computed RoPE (cos/sin) to a tensor.

    Forward:  y = x * cos + rotate90(x) * sin
    Backward: grad_x = grad * cos - rotate90(grad) * sin
    """

    @staticmethod
    def forward(ctx, x, cos_t, sin_t):
        x_val = x.get_value()
        cos_val = cos_t.get_value()
        sin_val = sin_t.get_value()
        ctx.save_for_backward(cos_val, sin_val)
        rotated = ttnn.alt_complex_rotate90(x_val)
        out = ttnn.add(ttnn.mul(x_val, cos_val), ttnn.mul(rotated, sin_val))
        return ttml.autograd.create_tensor(out)

    @staticmethod
    def backward(ctx, grad_output):
        cos_val, sin_val = ctx.saved_tensors
        rotated_grad = ttnn.alt_complex_rotate90(grad_output)
        grad_x = ttnn.subtract(ttnn.mul(grad_output, cos_val), ttnn.mul(rotated_grad, sin_val))
        return grad_x, None, None


class HeadsSplit(ttml.autograd.Function):
    """[B, 1, S, D] -> [B, H, S, D/H] via transpose+reshape+transpose."""

    @staticmethod
    def forward(ctx, x, num_heads):
        val = x.get_value()
        B, _, S, D = val.shape
        ctx.save_for_backward(B, 1, S, D)
        ctx.num_heads = num_heads
        t = ttnn.transpose(val, -2, -1)
        t = ttnn.reshape(t, [B, num_heads, D // num_heads, S])
        t = ttnn.transpose(t, -2, -1)
        return ttml.autograd.create_tensor(t)

    @staticmethod
    def backward(ctx, grad):
        B, _, S, D = ctx.saved_tensors
        return ttnn.experimental.nlp_concat_heads(grad)


class HeadsFuse(ttml.autograd.Function):
    """[B, H, S, D/H] -> [B, 1, S, D] via nlp_concat_heads."""

    @staticmethod
    def forward(ctx, x):
        val = x.get_value()
        B, H, S, Dh = val.shape
        ctx.save_for_backward(B, H, S, Dh)
        return ttml.autograd.create_tensor(ttnn.experimental.nlp_concat_heads(val))

    @staticmethod
    def backward(ctx, grad):
        B, H, S, Dh = ctx.saved_tensors
        D = H * Dh
        t = ttnn.transpose(grad, -2, -1)
        t = ttnn.reshape(t, [B, H, Dh, S])
        t = ttnn.transpose(t, -2, -1)
        return t


class ConcatLastDim(ttml.autograd.Function):
    """Concatenate two tensors along the last dimension."""

    @staticmethod
    def forward(ctx, a, b):
        a_shape = a.shape()
        b_shape = b.shape()
        ctx.save_for_backward(a_shape, b_shape)
        return ttml.autograd.create_tensor(ttnn.concat([a.get_value(), b.get_value()], dim=-1))

    @staticmethod
    def backward(ctx, grad_output):
        a_shape, b_shape = ctx.saved_tensors
        a_last = a_shape[-1]
        grad_a = ttnn.slice(grad_output, [0] * len(a_shape), list(a_shape))
        starts = [0] * len(b_shape)
        starts[-1] = a_last
        ends = list(b_shape)
        ends[-1] = a_last + b_shape[-1]
        grad_b = ttnn.slice(grad_output, starts, ends)
        return grad_a, grad_b


class ConcatSeqDim(ttml.autograd.Function):
    """Concatenate two 4-D tensors along dim 2 (sequence dimension)."""

    @staticmethod
    def forward(ctx, a, b):
        a_val = a.get_value()
        b_val = b.get_value()
        ctx.a_seq = a_val.shape[2]
        ctx.full_shape = list(a_val.shape)
        return ttml.autograd.create_tensor(ttnn.concat([a_val, b_val], dim=2))

    @staticmethod
    def backward(ctx, grad_output):
        s = ctx.full_shape
        grad_a = ttnn.slice(grad_output, [0, 0, 0, 0], [s[0], s[1], ctx.a_seq, s[3]])
        grad_b = ttnn.slice(
            grad_output,
            [0, 0, ctx.a_seq, 0],
            [s[0], s[1], grad_output.shape[2], s[3]],
        )
        return grad_a, grad_b


class SplitSeqDim(ttml.autograd.Function):
    """Split a 4-D tensor along dim 2 into two pieces. Backward concatenates."""

    @staticmethod
    def forward(ctx, x, split_at):
        val = x.get_value()
        s = list(val.shape)
        ctx.split_at = split_at
        a = ttml.autograd.create_tensor(ttnn.slice(val, [0, 0, 0, 0], [s[0], s[1], split_at, s[3]]))
        b = ttml.autograd.create_tensor(ttnn.slice(val, [0, 0, split_at, 0], [s[0], s[1], s[2], s[3]]))
        return a, b

    @staticmethod
    def backward(ctx, grad_a, grad_b):
        return ttnn.concat([grad_a, grad_b], dim=2)


class SiLUFunction(ttml.autograd.Function):
    

    @staticmethod
    def forward(ctx, x):
        x_val = x.get_value()
       
        sig = ttnn.sigmoid(x_val)
        out = ttnn.mul(x_val, sig)
        
        ctx.save_for_backward(sig, out)
        return ttml.autograd.create_tensor(out)

    @staticmethod
    def backward(ctx, grad_output):
        sig, fwd = ctx.saved_tensors
        g_raw = grad_output.get_value() if hasattr(grad_output, "get_value") else grad_output
        g = g_raw
        one_minus_sig = ttnn.subtract(ttnn.ones_like(sig), sig)
        fwd_term = ttnn.mul(fwd, one_minus_sig)
        dfdx = ttnn.add(sig, fwd_term)
        dx = ttnn.mul(g, dfdx)
        return dx



def _silu(x):
    return SiLUFunction.apply(x)



# =====================================================================
# Parallel linear layers
# =====================================================================






class ColumnParallelLinear(AbstractModuleBase):
    """Column-parallel linear: shards output features across TP.

    Weight shape: (1, 1, out_features, in_features).
    Per-device:   (1, 1, out_features / tp, in_features).
    """

    def __init__(self, in_features, out_features, has_bias=False, gather_output=False, shard_dim=None):
        super().__init__()
        self.gather_output = gather_output
        self.shard_dim = shard_dim
        self.weight = Parameter(_make_sharded_weight((1, 1, out_features, in_features), 2, shard_dim))
        self.col_bias = Parameter(_make_sharded_zeros((1, 1, 1, out_features), 3, shard_dim)) if has_bias else None

    def forward(self, x):
        x = ttml.ops.distributed.broadcast(x, self.shard_dim)
        bias_t = self.col_bias.tensor if self.col_bias is not None else None
        x = _linear(x, self.weight.tensor, bias_t)
        if self.gather_output:
            x = ttml.ops.distributed.all_gather(x, 3, self.shard_dim, ttml.ops.distributed.GradOutputType.REPLICATED)
        return x


class RowParallelLinear(AbstractModuleBase):
    """Row-parallel linear: shards input features across TP.

    Weight shape: (1, 1, out_features, in_features).
    Per-device:   (1, 1, out_features, in_features / tp).
    Bias added after all_reduce so it is not scaled by TP size.
    """

    def __init__(self, in_features, out_features, has_bias=False, input_is_parallel=False, shard_dim=None):
        super().__init__()
        self.input_is_parallel = input_is_parallel
        self.shard_dim = shard_dim
        self.weight = Parameter(_make_sharded_weight((1, 1, out_features, in_features), 3, shard_dim))
        self.row_bias = Parameter(_make_replicated_zeros((1, 1, 1, out_features))) if has_bias else None

    def forward(self, x):
        if not self.input_is_parallel:
            x = ttml.ops.distributed.scatter(x, 3, self.shard_dim)
        x = _linear(x, self.weight.tensor, None)
        x = ttml.ops.distributed.all_reduce(x, self.input_is_parallel, self.shard_dim)
        if self.row_bias is not None:
            x = ttml.ops.binary.add(x, self.row_bias.tensor)
        return x


class ReplicatedLinear(AbstractModuleBase):
    """Standard (replicated) linear with optional bias and activation."""

    def __init__(self, in_features, out_features, has_bias=False, act_fn=None):
        super().__init__()
        self.act_fn = act_fn
        self.weight = Parameter(_make_replicated_weight((1, 1, out_features, in_features)))
        self.bias = Parameter(_make_replicated_zeros((1, 1, 1, out_features))) if has_bias else None

    def forward(self, x):
        b = self.bias.tensor if self.bias is not None else None
        x = _linear(x, self.weight.tensor, b)
        if self.act_fn == "silu":
            x = _silu(x)
        elif self.act_fn == "gelu":
            x = ttml.ops.unary.gelu(x)
        return x


# =====================================================================
# RMSNorm module
# =====================================================================


class RMSNormFunction(ttml.autograd.Function):
    """RMSNorm via rsqrt: y = (x · rsqrt(mean(x², dim=-1) + ε)) · weight.

    All internal arithmetic runs in float32 to avoid precision loss from
    bfloat16 in the variance / rsqrt path.  Inputs are up-cast at entry,
    and outputs are down-cast back to bfloat16 before leaving each pass.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        x_val = ttnn.typecast(x.get_value(), ttnn.DataType.FLOAT32)
        w_val = ttnn.typecast(weight.get_value(), ttnn.DataType.FLOAT32)

        x_sq = ttnn.mul(x_val, x_val)
        mean_sq = ttnn.mean(x_sq, dim=-1, keepdim=True)
        variance = ttnn.add(mean_sq, eps)
        rrms = ttnn.rsqrt(variance)

        x_hat = ttnn.mul(x_val, rrms)
        out = ttnn.mul(x_hat, w_val)

        ctx.save_for_backward(x_hat, rrms, w_val)
        return ttml.autograd.create_tensor(ttnn.typecast(out, ttnn.DataType.BFLOAT16))

    @staticmethod
    def backward(ctx, grad_output):
        x_hat, rrms, w_val = ctx.saved_tensors
        grad_out_f32 = ttnn.typecast(grad_output, ttnn.DataType.FLOAT32)

        grad_w = ttnn.mul(grad_out_f32, x_hat)
        grad_w = ttnn.sum(grad_w, dim=[0, 1, 2], keepdim=True)

        gw = ttnn.mul(grad_out_f32, w_val)
        dot = ttnn.mul(gw, x_hat)
        dot_mean = ttnn.mean(dot, dim=-1, keepdim=True)
        correction = ttnn.mul(x_hat, dot_mean)
        grad_x = ttnn.mul(ttnn.subtract(gw, correction), rrms)

        return (
            ttnn.typecast(grad_x, ttnn.DataType.BFLOAT16),
            ttnn.typecast(grad_w, ttnn.DataType.BFLOAT16),
        )


class RMSNorm(AbstractModuleBase):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(_make_replicated_ones((1, 1, 1, dim)))

    def forward(self, x):
        return RMSNormFunction.apply(x, self.weight.tensor, self.eps)


# =====================================================================
# Timestep / text embeddings (replicated — not worth TP-sharding)
# =====================================================================


class TimestepEmbedding(AbstractModuleBase):
    """Two-layer MLP with SiLU: linear → silu → linear."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear_1 = ReplicatedLinear(in_channels, out_channels, has_bias=True)
        self.linear_2 = ReplicatedLinear(out_channels, out_channels, has_bias=True)

    def forward(self, x):
        x = self.linear_1(x)
        x = _silu(x)
        return self.linear_2(x)


class TextProjection(AbstractModuleBase):
    """PixArt-alpha style: linear+silu → linear."""

    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear_1 = ReplicatedLinear(in_features, hidden_size, has_bias=True, act_fn="silu")
        self.linear_2 = ReplicatedLinear(hidden_size, hidden_size, has_bias=True)

    def forward(self, x):
        return self.linear_2(self.linear_1(x))


class CombinedTimestepGuidanceTextProjEmbeddings(AbstractModuleBase):
    """Combined timestep + optional guidance + pooled text projection."""

    def __init__(self, embedding_dim, pooled_projection_dim, with_guidance=True):
        super().__init__()
        self.with_guidance = with_guidance
        self.timestep_embedder = TimestepEmbedding(256, embedding_dim)
        self.guidance_embedder = TimestepEmbedding(256, embedding_dim) if with_guidance else None
        self.text_embedder = TextProjection(pooled_projection_dim, embedding_dim)

    def forward(self, timestep_proj, guidance_proj, pooled_projection):
        """All inputs are pre-projected sinusoidal embeddings (bfloat16, 4-D)."""
        timesteps_emb = self.timestep_embedder(timestep_proj)
        text_emb = self.text_embedder(pooled_projection)
        if not self.with_guidance:
            return ttml.ops.binary.add(timesteps_emb, text_emb)
        guidance_emb = self.guidance_embedder(guidance_proj)
        return ttml.ops.binary.add(ttml.ops.binary.add(timesteps_emb, guidance_emb), text_emb)


# =====================================================================
# Attention
# =====================================================================


class Flux1JointAttention(AbstractModuleBase):
    """Joint attention for the Flux1 double (joint) transformer block.

    Spatial and prompt streams share Q/K/V structure but have independent
    projection weights.  Keys and values are concatenated before SDPA.
    """

    def __init__(self, config: Flux1Config, shard_dim):
        super().__init__()
        dim = config.inner_dim
        head_dim = config.attention_head_dim
        n_heads = config.num_attention_heads
        self.head_dim = head_dim
        self.n_local_heads = n_heads // _get_tp_size()
        self.shard_dim = shard_dim

        self.to_qkv = ColumnParallelLinear(dim, 3 * dim, has_bias=True, shard_dim=shard_dim)
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)
        self.to_out = RowParallelLinear(dim, dim, has_bias=True, input_is_parallel=True, shard_dim=shard_dim)

        self.add_qkv_proj = ColumnParallelLinear(dim, 3 * dim, has_bias=True, shard_dim=shard_dim)
        self.norm_added_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = RMSNorm(head_dim, eps=1e-6)
        self.to_add_out = RowParallelLinear(dim, dim, has_bias=True, input_is_parallel=True, shard_dim=shard_dim)

    def _qkv_norm_rope(self, x, qkv_proj, norm_q, norm_k, rope_cos, rope_sin):
        B = x.shape()[0]
        S = x.shape()[2]
        qkv = qkv_proj(x)
        q, k, v = _chunk4d(qkv, 3)

        q_shape = q.shape()
        k_shape = k.shape()
        q = ttml.ops.reshape.reshape(q, [B, 1, S * self.n_local_heads, self.head_dim])
        k = ttml.ops.reshape.reshape(k, [B, 1, S * self.n_local_heads, self.head_dim])
        q = norm_q(q)
        k = norm_k(k)
        q = ttml.ops.reshape.reshape(q, q_shape)
        k = ttml.ops.reshape.reshape(k, k_shape)

        q_heads = HeadsSplit.apply(q, self.n_local_heads)
        k_heads = HeadsSplit.apply(k, self.n_local_heads)
        v_heads = HeadsSplit.apply(v, self.n_local_heads)

        if rope_cos is not None:
            q_heads = ApplyRoPE.apply(q_heads, rope_cos, rope_sin)
            k_heads = ApplyRoPE.apply(k_heads, rope_cos, rope_sin)

        return q_heads, k_heads, v_heads

    def forward(self, spatial, prompt, spatial_rope_cos, spatial_rope_sin, prompt_rope_cos, prompt_rope_sin):
        sq, sk, sv = self._qkv_norm_rope(
            spatial, self.to_qkv, self.norm_q, self.norm_k, spatial_rope_cos, spatial_rope_sin
        )
        pq, pk, pv = self._qkv_norm_rope(
            prompt, self.add_qkv_proj, self.norm_added_q, self.norm_added_k, prompt_rope_cos, prompt_rope_sin
        )

        joint_k = ConcatSeqDim.apply(sk, pk)
        joint_v = ConcatSeqDim.apply(sv, pv)

        spatial_attn = ttml.ops.attention.scaled_dot_product_attention_composite(sq, joint_k, joint_v, None)
        prompt_attn = ttml.ops.attention.scaled_dot_product_attention_composite(pq, joint_k, joint_v, None)

        spatial_out = self.to_out(HeadsFuse.apply(spatial_attn))
        prompt_out = self.to_add_out(HeadsFuse.apply(prompt_attn))
        return spatial_out, prompt_out


class Flux1SingleAttention(AbstractModuleBase):
    """Attention for the Flux1 single transformer block (pre_only, shared weights)."""

    def __init__(self, config: Flux1Config, shard_dim):
        super().__init__()
        dim = config.inner_dim
        head_dim = config.attention_head_dim
        n_heads = config.num_attention_heads
        self.head_dim = head_dim
        self.n_local_heads = n_heads // _get_tp_size()
        self.shard_dim = shard_dim

        self.to_qkv = ColumnParallelLinear(dim, 3 * dim, has_bias=True, shard_dim=shard_dim)
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)

    def _proj_norm_rope(self, x, rope_cos, rope_sin):
        B = x.shape()[0]
        S = x.shape()[2]
        qkv = self.to_qkv(x)
        q, k, v = _chunk4d(qkv, 3)

        q_shape = q.shape()
        k_shape = k.shape()
        q = ttml.ops.reshape.reshape(q, [B, 1, S * self.n_local_heads, self.head_dim])
        k = ttml.ops.reshape.reshape(k, [B, 1, S * self.n_local_heads, self.head_dim])
        q = self.norm_q(q)
        k = self.norm_k(k)
        q = ttml.ops.reshape.reshape(q, q_shape)
        k = ttml.ops.reshape.reshape(k, k_shape)

        q_heads = HeadsSplit.apply(q, self.n_local_heads)
        k_heads = HeadsSplit.apply(k, self.n_local_heads)
        v_heads = HeadsSplit.apply(v, self.n_local_heads)

        if rope_cos is not None:
            q_heads = ApplyRoPE.apply(q_heads, rope_cos, rope_sin)
            k_heads = ApplyRoPE.apply(k_heads, rope_cos, rope_sin)

        return q_heads, k_heads, v_heads

    def forward(self, spatial, prompt, spatial_rope_cos, spatial_rope_sin, prompt_rope_cos, prompt_rope_sin):
        sq, sk, sv = self._proj_norm_rope(spatial, spatial_rope_cos, spatial_rope_sin)
        pq, pk, pv = self._proj_norm_rope(prompt, prompt_rope_cos, prompt_rope_sin)

        joint_k = ConcatSeqDim.apply(sk, pk)
        joint_v = ConcatSeqDim.apply(sv, pv)

        spatial_attn = ttml.ops.attention.scaled_dot_product_attention_composite(sq, joint_k, joint_v, None)
        prompt_attn = ttml.ops.attention.scaled_dot_product_attention_composite(pq, joint_k, joint_v, None)

        return (
            HeadsFuse.apply(spatial_attn),
            HeadsFuse.apply(prompt_attn),
        )


# =====================================================================
# FeedForward
# =====================================================================


class Flux1FeedForward(AbstractModuleBase):
    """Feedforward: ColParallel → GELU (tanh approx, HF ``gelu-approximate``) → RowParallel."""

    def __init__(self, dim, shard_dim):
        super().__init__()
        ff_dim = 4 * dim
        self.ff1 = ColumnParallelLinear(dim, ff_dim, has_bias=True, shard_dim=shard_dim)
        self.ff2 = RowParallelLinear(ff_dim, dim, has_bias=True, input_is_parallel=True, shard_dim=shard_dim)

    def forward(self, x):
        x = self.ff1(x)
        x = GeluTanhApproximate.apply(x)
        return self.ff2(x)


# =====================================================================
# Transformer blocks
# =====================================================================


class _Chunk4D(ttml.autograd.Function):
    """Differentiable chunk along the last dim: backward concatenates gradients."""

    @staticmethod
    def forward(ctx, x, count):
        val = x.get_value()
        size = val.shape[-1] // count
        ctx.count = count
        return tuple(
            ttml.autograd.create_tensor(val[:, :, :, i * size : (i + 1) * size])
            for i in range(count)
        )

    @staticmethod
    def backward(ctx, *grads):
        return ttnn.concat(list(grads), dim=3)


def _chunk4d(t, count):
    """Chunk a 4-D ttml tensor along the last dim into *count* pieces (differentiable)."""
    return list(_Chunk4D.apply(t, count))


# "Precise" compute kernel config for AdaLN reductions, mirroring
# ttml::core::ComputeKernelConfig::precise() in tt-train (HiFi4 + fp32 dest accumulation
# + L1-packer accumulation, no SFPU approximations). Only ttnn.mean / ttnn.sum currently
# accept compute_kernel_config; eltwise ops (mul/add/subtract/rsqrt) do not expose it
# and run with their built-in defaults (HiFi4, math_approx_mode=False, packer_l1_acc=off,
# fp32_dest_acc_en auto-set to True whenever the output dtype is FLOAT32, which is the
# case throughout AdaLN since we typecast to FP32 on entry).
_ADALN_REDUCE_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
_ADALN_REDUCE_COMPUTE_KERNEL_CONFIG=None

class AdaLNFunction(ttml.autograd.Function):
    """AdaLN in float32: layer_norm(x) * (1 + scale) + shift.

    All internal arithmetic (mean, variance, normalisation, affine transform)
    runs in float32 to avoid the bfloat16 precision loss that accumulates
    through dozens of transformer blocks.  Inputs are up-cast on entry and the
    result is down-cast back to bfloat16.
    """

    @staticmethod
    def forward(ctx, x, scale, shift):
        x_val = ttnn.typecast(x.get_value(), ttnn.DataType.FLOAT32)
        s_val = ttnn.typecast(scale.get_value(), ttnn.DataType.FLOAT32)
        b_val = ttnn.typecast(shift.get_value(), ttnn.DataType.FLOAT32)

        eps = 1e-6
        mean = ttnn.mean(
            x_val,
            dim=-1,
            keepdim=True,
            compute_kernel_config=_ADALN_REDUCE_COMPUTE_KERNEL_CONFIG,
        )
        x_centered = ttnn.subtract(x_val, mean)
        var = ttnn.mean(
            ttnn.mul(x_centered, x_centered),
            dim=-1,
            keepdim=True,
            compute_kernel_config=_ADALN_REDUCE_COMPUTE_KERNEL_CONFIG,
        )
        rstd = ttnn.rsqrt(ttnn.add(var, eps))
        x_hat = ttnn.mul(x_centered, rstd)

        weight = ttnn.add(s_val, 1.0)
        out = ttnn.add(ttnn.mul(x_hat, weight), b_val)

        ctx.save_for_backward(x_hat, rstd, weight)
        return ttml.autograd.create_tensor(ttnn.typecast(out, ttnn.DataType.BFLOAT16))

    @staticmethod
    def backward(ctx, grad_output):
        x_hat, rstd, weight = ctx.saved_tensors
        g = ttnn.typecast(
            grad_output.get_value() if hasattr(grad_output, "get_value") else grad_output,
            ttnn.DataType.FLOAT32,
        )

        # d(out)/d(shift) = 1  →  grad_shift = sum over batch/seq dims
        grad_shift = ttnn.sum(
            g,
            dim=[1, 2],
            keepdim=True,
            compute_kernel_config=_ADALN_REDUCE_COMPUTE_KERNEL_CONFIG,
        )

        # d(out)/d(scale): out = x_hat*(1+s) + b  →  d/ds = x_hat
        grad_scale = ttnn.mul(g, x_hat)
        grad_scale = ttnn.sum(
            grad_scale,
            dim=[1, 2],
            keepdim=True,
            compute_kernel_config=_ADALN_REDUCE_COMPUTE_KERNEL_CONFIG,
        )

        # d(out)/d(x) via layernorm backward
        gw = ttnn.mul(g, weight)
        dot = ttnn.mul(gw, x_hat)
        dot_mean = ttnn.mean(
            dot,
            dim=-1,
            keepdim=True,
            compute_kernel_config=_ADALN_REDUCE_COMPUTE_KERNEL_CONFIG,
        )
        gw_mean = ttnn.mean(
            gw,
            dim=-1,
            keepdim=True,
            compute_kernel_config=_ADALN_REDUCE_COMPUTE_KERNEL_CONFIG,
        )
        grad_x = ttnn.mul(ttnn.subtract(ttnn.subtract(gw, gw_mean), ttnn.mul(x_hat, dot_mean)), rstd)

        return (
            ttnn.typecast(grad_x, ttnn.DataType.BFLOAT16),
            ttnn.typecast(grad_scale, ttnn.DataType.BFLOAT16),
            ttnn.typecast(grad_shift, ttnn.DataType.BFLOAT16),
        )


def _adaln(x, scale, shift):
    """AdaLN: layer_norm(x) * (1 + scale) + shift, computed in float32."""
    return AdaLNFunction.apply(x, scale, shift)


class Flux1TransformerBlock(AbstractModuleBase):
    """Double (joint) transformer block with AdaLN modulation."""

    def __init__(self, config: Flux1Config, shard_dim):
        super().__init__()
        dim = config.inner_dim

        self.norm1_linear = ColumnParallelLinear(dim, 6 * dim, has_bias=True, gather_output=True, shard_dim=shard_dim)
        self.norm1_context_linear = ColumnParallelLinear(
            dim, 6 * dim, has_bias=True, gather_output=True, shard_dim=shard_dim
        )
        self.attn = Flux1JointAttention(config, shard_dim)
        self.ff = Flux1FeedForward(dim, shard_dim)
        self.ff_context = Flux1FeedForward(dim, shard_dim)

    def forward(self, spatial, prompt, time_embed, spatial_rope_cos, spatial_rope_sin, prompt_rope_cos, prompt_rope_sin):
        spatial_time = self.norm1_linear(time_embed)
        prompt_time = self.norm1_context_linear(time_embed)

        s_shift_a, s_scale_a, s_gate_a, s_shift_f, s_scale_f, s_gate_f = _chunk4d(spatial_time, 6)
        p_shift_a, p_scale_a, p_gate_a, p_shift_f, p_scale_f, p_gate_f = _chunk4d(prompt_time, 6)

        spatial_normed = _adaln(spatial, s_scale_a, s_shift_a)
        prompt_normed = _adaln(prompt, p_scale_a, p_shift_a)

        spatial_attn, prompt_attn = self.attn(
            spatial_normed, prompt_normed, spatial_rope_cos, spatial_rope_sin, prompt_rope_cos, prompt_rope_sin
        )
        spatial_attn = ttml.ops.binary.mul(spatial_attn, s_gate_a)
        prompt_attn = ttml.ops.binary.mul(prompt_attn, p_gate_a)

        spatial = ttml.ops.binary.add(spatial, spatial_attn)
        prompt = ttml.ops.binary.add(prompt, prompt_attn)

        spatial_ff_in = _adaln(spatial, s_scale_f, s_shift_f)
        spatial_ff = self.ff(spatial_ff_in)
        spatial_ff = ttml.ops.binary.mul(spatial_ff, s_gate_f)
        spatial = ttml.ops.binary.add(spatial, spatial_ff)

        prompt_ff_in = _adaln(prompt, p_scale_f, p_shift_f)
        prompt_ff = self.ff_context(prompt_ff_in)
        prompt_ff = ttml.ops.binary.mul(prompt_ff, p_gate_f)
        prompt = ttml.ops.binary.add(prompt, prompt_ff)

        return spatial, prompt


class Flux1SingleTransformerBlock(AbstractModuleBase):
    """Single transformer block: shared attention + MLP, fused output projection."""

    def __init__(self, config: Flux1Config, shard_dim):
        super().__init__()
        dim = config.inner_dim
        mlp_dim = config.mlp_hidden_dim
        self.dim = dim
        self.mlp_dim = mlp_dim

        self.time_embed = ColumnParallelLinear(dim, 3 * dim, has_bias=True, gather_output=True, shard_dim=shard_dim)
        self.proj_mlp = ColumnParallelLinear(dim, mlp_dim, has_bias=True, shard_dim=shard_dim)
        self.attn = Flux1SingleAttention(config, shard_dim)
        self.proj_out = RowParallelLinear(dim + mlp_dim, dim, has_bias=True, input_is_parallel=True, shard_dim=shard_dim)

    def forward(self, spatial, prompt, time_embed, spatial_rope_cos, spatial_rope_sin, prompt_rope_cos, prompt_rope_sin):
        time_out = self.time_embed(time_embed)
        shift_msa, scale_msa, gate_msa = _chunk4d(time_out, 3)

        spatial_normed = _adaln(spatial, scale_msa, shift_msa)
        prompt_normed = _adaln(prompt, scale_msa, shift_msa)

        mlp_s = GeluTanhApproximate.apply(self.proj_mlp(spatial_normed))
        mlp_p = GeluTanhApproximate.apply(self.proj_mlp(prompt_normed))

        attn_s, attn_p = self.attn(
            spatial_normed, prompt_normed,
            spatial_rope_cos, spatial_rope_sin, prompt_rope_cos, prompt_rope_sin,
        )
        

        fused_s = ConcatLastDim.apply(attn_s, mlp_s)
        fused_p = ConcatLastDim.apply(attn_p, mlp_p)
        out_s = self.proj_out(fused_s)
        out_p = self.proj_out(fused_p)
        out_s = ttml.ops.binary.mul(out_s, gate_msa)
        out_p = ttml.ops.binary.mul(out_p, gate_msa)

        spatial = ttml.ops.binary.add(spatial, out_s)
        prompt = ttml.ops.binary.add(prompt, out_p)
        return spatial, prompt


# =====================================================================
# Full Transformer
# =====================================================================


class CheckpointFunction(ttml.autograd.Function):
    """Gradient checkpointing: detach ALL tensor inputs, recompute forward in backward.

    Forward: run forward_fn with grad disabled, return raw output.
    Backward: detach all tensor args (creating an isolated subgraph),
    rerun forward with grad enabled, backward through the isolated subgraph,
    collect and return gradients for every tensor input.
    """

    @staticmethod
    def forward(ctx, forward_fn, *args):
        ctx.forward_fn = forward_fn
        ctx.args = args
        ctx.tensor_indices = [i for i, a in enumerate(args) if hasattr(a, "get_requires_grad")]

        auto_ctx = ttml.autograd.AutoContext.get_instance()
        prev_mode = auto_ctx.get_gradient_mode()
        auto_ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
        try:
            out = forward_fn(*args)
        finally:
            auto_ctx.set_gradient_mode(prev_mode)

        return out.get_value()

    @staticmethod
    def backward(ctx, grad_output):
        new_args = list(ctx.args)
        detached = {}
        for i in ctx.tensor_indices:
            d = ttml.autograd.create_tensor(ctx.args[i].get_value())
            new_args[i] = d
            detached[i] = d

        recomputed = ctx.forward_fn(*new_args)
        recomputed.set_grad(grad_output)
        recomputed.backward(False)

        grads = []
        for i in ctx.tensor_indices:
            d = detached[i]
            grads.append(d.get_grad() if d.is_grad_initialized() else None)
        return tuple(grads)


def checkpoint(forward_fn, *args):
    """Gradient checkpoint: recompute forward during backward to save memory."""
    return CheckpointFunction.apply(forward_fn, *args)


def _checkpoint_dual_block(block, spatial, prompt, *extra):
    """Gradient-checkpoint a block that returns (spatial, prompt)."""
    spatial_seq = spatial.shape()[2]

    def _wrapped(s, p, *ex):
        rs, rp = block(s, p, *ex)
        return ConcatSeqDim.apply(rs, rp)

    combined = checkpoint(_wrapped, spatial, prompt, *extra)

    return SplitSeqDim.apply(combined, spatial_seq)


class DistributedFlux1Transformer(AbstractModuleBase):
    """Full Flux1 DiT: embeddings → double blocks → single blocks → output."""

    def __init__(self, config: Flux1Config, shard_dim=None, use_checkpoint=False):
        super().__init__()
        self.create_name("DistributedFlux1Transformer")
        self.config = config
        self.shard_dim = shard_dim
        self.use_checkpoint = use_checkpoint
        dim = config.inner_dim

        self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
            embedding_dim=dim,
            pooled_projection_dim=config.pooled_projection_dim,
            with_guidance=config.guidance_embeds,
        )
        self.x_embedder = ColumnParallelLinear(config.in_channels, dim, has_bias=True, gather_output=True, shard_dim=shard_dim)
        self.context_embedder = ColumnParallelLinear(
            config.joint_attention_dim, dim, has_bias=True, gather_output=True, shard_dim=shard_dim
        )

        self.transformer_blocks = ModuleList(
            [Flux1TransformerBlock(config, shard_dim) for _ in range(config.num_layers)]
        )
        self.single_transformer_blocks = ModuleList(
            [Flux1SingleTransformerBlock(config, shard_dim) for _ in range(config.num_single_layers)]
        )

        self.norm_out = ReplicatedLinear(dim, 2 * dim, has_bias=True)
        self.proj_out = ReplicatedLinear(dim, config.patch_size * config.patch_size * config.out_channels, has_bias=True)

    def forward(
        self,
        spatial,
        prompt,
        timestep_proj,
        guidance_proj,
        pooled,
        spatial_rope_cos,
        spatial_rope_sin,
        prompt_rope_cos,
        prompt_rope_sin,
    ):
        time_embed = self.time_text_embed(timestep_proj, guidance_proj, pooled)
        time_embed = _silu(time_embed)

        spatial = self.x_embedder(spatial)
        prompt = self.context_embedder(prompt)

        extra = (time_embed, spatial_rope_cos, spatial_rope_sin, prompt_rope_cos, prompt_rope_sin)
        for block in self.transformer_blocks:
            if self.use_checkpoint:
                spatial, prompt = _checkpoint_dual_block(block, spatial, prompt, *extra)
            else:
                spatial, prompt = block(spatial, prompt, *extra)

        for block in self.single_transformer_blocks:
            if self.use_checkpoint:
                spatial, prompt = _checkpoint_dual_block(block, spatial, prompt, *extra)
            else:
                spatial, prompt = block(spatial, prompt, *extra)

        spatial_time = self.norm_out(time_embed)
        scale, shift = _chunk4d(spatial_time, 2)
        spatial = _adaln(spatial, scale, shift)
        return self.proj_out(spatial)


# =====================================================================
# Weight loading
# =====================================================================


def _load_tensor_distributed(weight_np, shard_type, shard_dim):
    """Create a bfloat16 ttml tensor with appropriate sharding."""
    device = _get_device()
    if shard_type == "col_w":
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 2, shard_dim)
        return ttml.autograd.Tensor.from_numpy(weight_np, ttnn.Layout.TILE, ttnn.bfloat16, mapper)
    elif shard_type in ("col_b", "row_w"):
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 3, shard_dim)
        return ttml.autograd.Tensor.from_numpy(weight_np, ttnn.Layout.TILE, ttnn.bfloat16, mapper)
    else:
        return ttml.autograd.Tensor.from_numpy(weight_np, ttnn.Layout.TILE, ttnn.bfloat16)


def _refuse_proj_out_weight(weight: torch.Tensor, attn_dim: int, mlp_dim: int, tp_size: int) -> torch.Tensor:
    """Re-order proj_out weight columns so TP sharding matches the concat layout."""
    if tp_size <= 1:
        return weight
    w_attn = weight[:, :attn_dim].reshape(weight.shape[0], tp_size, attn_dim // tp_size)
    w_mlp = weight[:, attn_dim:].reshape(weight.shape[0], tp_size, mlp_dim // tp_size)
    return torch.cat([w_attn, w_mlp], dim=2).reshape(weight.shape[0], -1)


def _build_weight_mapping(config: Flux1Config, root: str, tp_size: int):
    """Build {hf_name → (ttml_name, shard_type)} mapping."""
    mapping = {}
    dim = config.inner_dim
    mlp_dim = config.mlp_hidden_dim

    def _add(hf, ttml_name, st):
        mapping[hf] = (f"{root}/{ttml_name}", st)

    # ---- global embeddings (replicated) ----
    for sub in ["timestep_embedder", "text_embedder"]:
        for layer in ["linear_1", "linear_2"]:
            _add(f"time_text_embed.{sub}.{layer}.weight", f"time_text_embed/{sub}/{layer}/weight", None)
            _add(f"time_text_embed.{sub}.{layer}.bias", f"time_text_embed/{sub}/{layer}/bias", None)
    if config.guidance_embeds:
        for layer in ["linear_1", "linear_2"]:
            _add(
                f"time_text_embed.guidance_embedder.{layer}.weight",
                f"time_text_embed/guidance_embedder/{layer}/weight",
                None,
            )
            _add(
                f"time_text_embed.guidance_embedder.{layer}.bias",
                f"time_text_embed/guidance_embedder/{layer}/bias",
                None,
            )

    _add("x_embedder.weight", "x_embedder/weight", "col_w")
    _add("x_embedder.bias", "x_embedder/col_bias", "col_b")
    _add("context_embedder.weight", "context_embedder/weight", "col_w")
    _add("context_embedder.bias", "context_embedder/col_bias", "col_b")

    # ---- double (joint) transformer blocks ----
    for i in range(config.num_layers):
        pfx = f"transformer_blocks.{i}"
        tpfx = f"transformer_blocks/{i}"

        _add(f"{pfx}.norm1.linear.weight", f"{tpfx}/norm1_linear/weight", "col_w")
        _add(f"{pfx}.norm1.linear.bias", f"{tpfx}/norm1_linear/col_bias", "col_b")
        _add(f"{pfx}.norm1_context.linear.weight", f"{tpfx}/norm1_context_linear/weight", "col_w")
        _add(f"{pfx}.norm1_context.linear.bias", f"{tpfx}/norm1_context_linear/col_bias", "col_b")

        _add(f"{pfx}.attn.to_qkv.weight", f"{tpfx}/attn/to_qkv/weight", "col_w")
        _add(f"{pfx}.attn.to_qkv.bias", f"{tpfx}/attn/to_qkv/col_bias", "col_b")
        _add(f"{pfx}.attn.to_out.0.weight", f"{tpfx}/attn/to_out/weight", "row_w")
        _add(f"{pfx}.attn.to_out.0.bias", f"{tpfx}/attn/to_out/row_bias", None)
        for norm_name in ["norm_q", "norm_k"]:
            _add(f"{pfx}.attn.{norm_name}.weight", f"{tpfx}/attn/{norm_name}/weight", None)
        _add(f"{pfx}.attn.add_qkv_proj.weight", f"{tpfx}/attn/add_qkv_proj/weight", "col_w")
        _add(f"{pfx}.attn.add_qkv_proj.bias", f"{tpfx}/attn/add_qkv_proj/col_bias", "col_b")
        _add(f"{pfx}.attn.to_add_out.weight", f"{tpfx}/attn/to_add_out/weight", "row_w")
        _add(f"{pfx}.attn.to_add_out.bias", f"{tpfx}/attn/to_add_out/row_bias", None)
        for norm_name in ["norm_added_q", "norm_added_k"]:
            _add(f"{pfx}.attn.{norm_name}.weight", f"{tpfx}/attn/{norm_name}/weight", None)

        _add(f"{pfx}.ff.net.0.proj.weight", f"{tpfx}/ff/ff1/weight", "col_w")
        _add(f"{pfx}.ff.net.0.proj.bias", f"{tpfx}/ff/ff1/col_bias", "col_b")
        _add(f"{pfx}.ff.net.2.weight", f"{tpfx}/ff/ff2/weight", "row_w")
        _add(f"{pfx}.ff.net.2.bias", f"{tpfx}/ff/ff2/row_bias", None)
        _add(f"{pfx}.ff_context.net.0.proj.weight", f"{tpfx}/ff_context/ff1/weight", "col_w")
        _add(f"{pfx}.ff_context.net.0.proj.bias", f"{tpfx}/ff_context/ff1/col_bias", "col_b")
        _add(f"{pfx}.ff_context.net.2.weight", f"{tpfx}/ff_context/ff2/weight", "row_w")
        _add(f"{pfx}.ff_context.net.2.bias", f"{tpfx}/ff_context/ff2/row_bias", None)

    # ---- single transformer blocks ----
    for i in range(config.num_single_layers):
        pfx = f"single_transformer_blocks.{i}"
        tpfx = f"single_transformer_blocks/{i}"

        _add(f"{pfx}.norm.linear.weight", f"{tpfx}/time_embed/weight", "col_w")
        _add(f"{pfx}.norm.linear.bias", f"{tpfx}/time_embed/col_bias", "col_b")

        _add(f"{pfx}.attn.to_qkv.weight", f"{tpfx}/attn/to_qkv/weight", "col_w")
        _add(f"{pfx}.attn.to_qkv.bias", f"{tpfx}/attn/to_qkv/col_bias", "col_b")
        for norm_name in ["norm_q", "norm_k"]:
            _add(f"{pfx}.attn.{norm_name}.weight", f"{tpfx}/attn/{norm_name}/weight", None)

        _add(f"{pfx}.proj_mlp.weight", f"{tpfx}/proj_mlp/weight", "col_w")
        _add(f"{pfx}.proj_mlp.bias", f"{tpfx}/proj_mlp/col_bias", "col_b")
        _add(f"{pfx}.proj_out.weight", f"{tpfx}/proj_out/weight", "row_w")
        _add(f"{pfx}.proj_out.bias", f"{tpfx}/proj_out/row_bias", None)

    # ---- output ----
    _add("norm_out.linear.weight", "norm_out/weight", None)
    _add("norm_out.linear.bias", "norm_out/bias", None)
    _add("proj_out.weight", "proj_out/weight", None)
    _add("proj_out.bias", "proj_out/bias", None)

    return mapping


def _fuse_qkv_in_state_dict(sd: dict, config: Flux1Config, tp_size: int) -> dict:
    """Interleave Q/K/V weights so contiguous TP sharding keeps [q,k,v] per device."""
    fused = dict(sd)
    dim = config.inner_dim
    shard = dim // tp_size

    def _interleave(q_w, k_w, v_w):
        chunks = []
        for i in range(tp_size):
            chunks.extend([q_w[i * shard : (i + 1) * shard],
                           k_w[i * shard : (i + 1) * shard],
                           v_w[i * shard : (i + 1) * shard]])
        return torch.cat(chunks, dim=0)

    for i in range(config.num_layers):
        pfx = f"transformer_blocks.{i}.attn"
        for (q, k, v, out) in [
            ("to_q", "to_k", "to_v", "to_qkv"),
            ("add_q_proj", "add_k_proj", "add_v_proj", "add_qkv_proj"),
        ]:
            for suffix in ["weight", "bias"]:
                qn, kn, vn = f"{pfx}.{q}.{suffix}", f"{pfx}.{k}.{suffix}", f"{pfx}.{v}.{suffix}"
                if qn in fused:
                    fused[f"{pfx}.{out}.{suffix}"] = _interleave(fused.pop(qn), fused.pop(kn), fused.pop(vn))

    for i in range(config.num_single_layers):
        pfx = f"single_transformer_blocks.{i}.attn"
        for suffix in ["weight", "bias"]:
            qn = f"{pfx}.to_q.{suffix}"
            kn = f"{pfx}.to_k.{suffix}"
            vn = f"{pfx}.to_v.{suffix}"
            if qn in fused:
                fused[f"{pfx}.to_qkv.{suffix}"] = _interleave(fused.pop(qn), fused.pop(kn), fused.pop(vn))
    return fused


def load_weights_from_hf_distributed(
    ttml_model: DistributedFlux1Transformer,
    hf_state_dict: dict,
    config: Flux1Config,
    shard_dim: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """Load HuggingFace FluxTransformer2DModel weights into distributed ttml model."""
    tp_size = _get_tp_size()
    hf_state_dict = _fuse_qkv_in_state_dict(hf_state_dict, config, tp_size)
    ttml_params = ttml_model.parameters()
    root = next(iter(ttml_params)).split("/")[0]

    if verbose:
        print("\n  TTML parameter names:")
        for name in sorted(ttml_params.keys()):
            print(f"    {name}: {list(ttml_params[name].shape())}")

    mapping = _build_weight_mapping(config, root, tp_size)
    ttml_shapes = {name: list(ttml_params[name].shape()) for name in ttml_params}
    dim = config.inner_dim
    mlp_dim = config.mlp_hidden_dim

    def _prepare(hf_name, ttml_name, shard_type):
        if hf_name not in hf_state_dict:
            return None
        if ttml_name not in ttml_shapes:
            return None

        weight = hf_state_dict[hf_name].float()

        if "proj_out/weight" in ttml_name and "single_transformer_blocks" in ttml_name:
            weight = _refuse_proj_out_weight(weight, dim, mlp_dim, tp_size)

        ttml_shape = ttml_shapes[ttml_name]

        if weight.dim() == 2:
            rows, cols = weight.shape
            tgt_rows, tgt_cols = ttml_shape[2], ttml_shape[3]
            if shard_type == "col_w":
                tgt_rows *= tp_size
            elif shard_type == "row_w":
                tgt_cols *= tp_size
            if rows != tgt_rows or cols != tgt_cols:
                padded = torch.zeros(tgt_rows, tgt_cols, dtype=weight.dtype)
                padded[: min(rows, tgt_rows), : min(cols, tgt_cols)] = weight[: min(rows, tgt_rows), : min(cols, tgt_cols)]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0)
        elif weight.dim() == 1:
            d = weight.shape[0]
            tgt_d = ttml_shape[-1]
            if shard_type == "col_b":
                tgt_d *= tp_size
            if d != tgt_d:
                padded = torch.zeros(tgt_d, dtype=weight.dtype)
                padded[: min(d, tgt_d)] = weight[: min(d, tgt_d)]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected weight dim {weight.dim()} for {hf_name}")

        return _load_tensor_distributed(weight.contiguous().numpy(), shard_type, shard_dim)

    from concurrent.futures import ThreadPoolExecutor

    items = [(hf, *v) for hf, v in mapping.items()]
    loaded, skipped = 0, []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(hf, ttml_n, pool.submit(_prepare, hf, ttml_n, st)) for hf, ttml_n, st in items]
        for hf_name, ttml_name, future in tqdm(futures, total=len(items), desc="  Loading weights", unit="w"):
            new_tensor = future.result()
            if new_tensor is None:
                skipped.append(hf_name)
                continue
            ttml_params[ttml_name].assign(new_tensor)
            loaded += 1

    print(f"\n  Weight loading: {loaded} loaded, {len(skipped)} skipped")
    if verbose and skipped:
        print(f"  Skipped: {skipped}")
