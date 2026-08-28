# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of `attention` (FLUX.2 VAE mid-block `Attention`).

Reference module: `encoder.mid_block.attentions.0` /
`decoder.mid_block.attentions.0` of `AutoencoderKLFlux2` — a diffusers
`Attention` driven by `AttnProcessor2_0`, i.e. *spatial self-attention over a
feature map*, not a transformer GQA block:

    residual = x
    h        = group_norm(x^T)^T                 # GroupNorm(32 groups, 512 ch, eps=1e-6)
    q,k,v    = h @ Wq^T+bq, h @ Wk^T+bk, h @ Wv^T+bv
    y        = softmax(q @ k^T / sqrt(head_dim)) @ v
    out      = (y @ Wo^T + bo) + residual        # residual_connection=True, rescale=1

`heads == 1`, `inner_dim == 512`, so `head_dim == inner_dim == 512`.

Tensor-parallel scheme (TP=8) and why
-------------------------------------
The textbook Megatron attention split is *head-parallel*: qkv are
column-parallel over the head axis and out_proj is row-parallel, so the only
collective is one all_reduce after out_proj. That split is unavailable here —
this module has a SINGLE head, so the head axis cannot absorb the 8-way split.

So the split lands one axis down, on the qkv *channel* axis (512 = 8 x 64):

  * `to_q` / `to_k` / `to_v` are COLUMN-parallel: `W` split on its output
    features (`ShardTensorToMesh(dim=-1)` after transposing to `[in, out]`),
    with each bias split the same way so it stays attached to its columns.
  * The score matmul `q @ k^T` contracts over exactly that split axis, so a
    device holding only `q[:, 64d:64d+64]` cannot form a score on its own.
    Column-parallel is therefore followed by its natural collective — an
    `all_gather` on the feature dim — rebuilding full `q` and `k` on every
    device. `scores`, `softmax` and the resulting probabilities are then
    replicated (they are `[N, N]`, independent of the channel split, and cheap).
  * `v` is deliberately NOT gathered: `probs @ v_shard` keeps the context
    channel-sharded, which is exactly the layout `to_out` wants.
  * `to_out` is ROW-parallel: `W` split on its *input* features
    (`ShardTensorToMesh(dim=0)` after the same transpose), so each device
    produces a partial sum over the full model dim, followed by `all_reduce`.
    Its bias is REPLICATED and added AFTER the reduction — sharding it would
    add it 8 times.
  * GroupNorm (gamma/beta + the group-membership matrices), the out bias and
    the residual are REPLICATED: elementwise/lookup tensors never shard, and
    the residual has to be the full model dim on every device anyway.

The math is unchanged by the placement: gathered `q`/`k` are bit-for-bit the
concatenation of what a single device would have computed, and the all_reduced
`to_out` partials sum to the single-device product. Gathered PCC is the judge.

GroupNorm is computed natively rather than via `ttnn.group_norm` so it needs no
sharded-memory/core-grid/input-mask staging: the per-group sums are obtained by
a matmul against a constant 0/1 group-membership matrix `[C, G]`, and the group
statistics are scattered back to channels with its transpose `[G, C]`. Mean and
variance are two-pass (`E[(x-mu)^2]`, not `E[x^2]-E[x]^2`) so the bfloat16
variance never depends on a cancellation between two large numbers.
"""
from __future__ import annotations

import math

import torch

import ttnn


def _mesh_width(device) -> int:
    """Number of devices `device` spans (1 for a single-chip device)."""
    try:
        n = int(device.get_num_devices())
    except (AttributeError, TypeError):
        return 1
    return n if n > 0 else 1


class TtVaeAttention:
    """Tensor-parallel native-ttnn `Attention` for the FLUX.2 VAE mid block."""

    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("attention stub needs the torch reference module to stage its weights")

        self.device = device
        self.tp = _mesh_width(device)

        gn = torch_module.group_norm
        self.groups = int(gn.num_groups)
        self.channels = int(gn.num_channels)
        self.eps = float(gn.eps)
        self.group_size = self.channels // self.groups

        self.heads = int(torch_module.heads)
        self.inner_dim = int(torch_module.to_q.out_features)
        self.residual_connection = bool(torch_module.residual_connection)
        self.rescale_output_factor = float(torch_module.rescale_output_factor)
        self.scale = 1.0 / math.sqrt(self.inner_dim // self.heads)

        if self.heads != 1:
            raise RuntimeError(
                f"attention stub implements the single-head VAE spatial attention "
                f"(heads=1); this module has heads={self.heads}, which wants the "
                f"head-parallel split instead of the channel-parallel one."
            )
        if self.tp > 1 and self.inner_dim % self.tp != 0:
            raise RuntimeError(
                f"inner_dim={self.inner_dim} is not divisible by TP={self.tp}; "
                f"the qkv channel split needs an even division."
            )
        if self.tp > 1 and (self.inner_dim // self.tp) % 32 != 0:
            raise RuntimeError(
                f"per-device channel shard {self.inner_dim // self.tp} is not a "
                f"multiple of the 32-wide tile; pick a TP degree that keeps it tiled."
            )

        state = torch_module.state_dict()

        # ---- weight staging (setup, not the compute path) --------------------
        def _replicated(tensor):
            mapper = ttnn.ReplicateTensorToMesh(device) if self.tp > 1 else None
            return ttnn.from_torch(
                tensor.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=mapper,
            )

        def _sharded(tensor, dim):
            mapper = ttnn.ShardTensorToMesh(device, dim=dim) if self.tp > 1 else None
            return ttnn.from_torch(
                tensor.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=mapper,
            )

        # GroupNorm group-membership matrix: `[C, G]`, one-hot per channel.
        # `x @ group_sum` -> per-group sums; `stat @ group_scatter` -> per-channel.
        membership = torch.zeros(self.channels, self.groups, dtype=torch.float32)
        for g in range(self.groups):
            membership[g * self.group_size : (g + 1) * self.group_size, g] = 1.0
        self.group_sum = _replicated(membership)
        self.group_scatter = _replicated(membership.t().contiguous())

        # Norm affine params stay replicated (elementwise, not a matmul weight).
        self.gn_weight = _replicated(state["group_norm.weight"].reshape(1, 1, self.channels))
        self.gn_bias = _replicated(state["group_norm.bias"].reshape(1, 1, self.channels))

        # Column-parallel qkv: transpose to `[in, out]`, split the OUT features.
        self.wq = _sharded(state["to_q.weight"].t().contiguous(), dim=-1)
        self.wk = _sharded(state["to_k.weight"].t().contiguous(), dim=-1)
        self.wv = _sharded(state["to_v.weight"].t().contiguous(), dim=-1)
        self.bq = _sharded(state["to_q.bias"].reshape(1, self.inner_dim), dim=-1)
        self.bk = _sharded(state["to_k.bias"].reshape(1, self.inner_dim), dim=-1)
        self.bv = _sharded(state["to_v.bias"].reshape(1, self.inner_dim), dim=-1)

        # Row-parallel out proj: split the IN features; bias replicated and
        # applied once, after the all_reduce.
        self.wo = _sharded(state["to_out.0.weight"].t().contiguous(), dim=0)
        self.bo = _replicated(state["to_out.0.bias"].reshape(1, 1, self.inner_dim))

        try:
            self.compute_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        except (AttributeError, RuntimeError, TypeError):
            self.compute_config = None

    # ---- collectives ---------------------------------------------------------
    def _all_gather_channels(self, tensor):
        """Rebuild the full channel dim of a column-parallel result on every device."""
        if self.tp <= 1:
            return tensor
        try:
            return ttnn.all_gather(tensor, dim=2, topology=ttnn.Topology.Linear)
        except TypeError:
            return ttnn.all_gather(tensor, dim=2)

    def _all_reduce_partials(self, tensor):
        """Sum the row-parallel partial products across devices, result replicated."""
        if self.tp <= 1:
            return tensor
        try:
            return ttnn.all_reduce(tensor, topology=ttnn.Topology.Linear)
        except (TypeError, AttributeError):
            return ttnn.all_reduce(tensor)

    # ---- forward -------------------------------------------------------------
    def _group_norm(self, x, positions):
        """GroupNorm over `[B, N, C]` with groups formed on the channel dim."""
        inv_count = 1.0 / float(positions * self.group_size)

        group_totals = ttnn.matmul(x, self.group_sum, compute_kernel_config=self.compute_config)
        group_totals = ttnn.sum(group_totals, dim=1, keepdim=True)
        mean = ttnn.multiply(group_totals, inv_count)
        mean_per_channel = ttnn.matmul(mean, self.group_scatter, compute_kernel_config=self.compute_config)

        centered = ttnn.subtract(x, mean_per_channel)
        squared = ttnn.multiply(centered, centered)
        var_totals = ttnn.matmul(squared, self.group_sum, compute_kernel_config=self.compute_config)
        var_totals = ttnn.sum(var_totals, dim=1, keepdim=True)
        variance = ttnn.multiply(var_totals, inv_count)
        inv_std = ttnn.rsqrt(ttnn.add(variance, self.eps))
        inv_std_per_channel = ttnn.matmul(inv_std, self.group_scatter, compute_kernel_config=self.compute_config)

        normed = ttnn.multiply(centered, inv_std_per_channel)
        normed = ttnn.multiply(normed, self.gn_weight)
        return ttnn.add(normed, self.gn_bias)

    def __call__(self, hidden_states, *args, **kwargs):
        if kwargs.get("encoder_hidden_states") is not None:
            raise RuntimeError("the VAE mid-block attention is self-attention only")

        x = hidden_states
        original_shape = tuple(int(v) for v in x.shape)

        # Normalise to `[B, N, C]`; `[B, 1, N, C]` and `[B, H, W, C]` both fold in.
        if len(original_shape) == 4:
            batch = original_shape[0]
            positions = original_shape[1] * original_shape[2]
            x = ttnn.reshape(x, (batch, positions, original_shape[3]))
        elif len(original_shape) == 3:
            positions = original_shape[1]
        else:
            raise RuntimeError(f"attention expects a rank-3 or rank-4 activation, got {original_shape}")

        residual = x
        normed = self._group_norm(x, positions)

        # Column-parallel qkv on the channel axis.
        query_shard = ttnn.linear(normed, self.wq, bias=self.bq, compute_kernel_config=self.compute_config)
        key_shard = ttnn.linear(normed, self.wk, bias=self.bk, compute_kernel_config=self.compute_config)
        value_shard = ttnn.linear(normed, self.wv, bias=self.bv, compute_kernel_config=self.compute_config)

        # The score matmul contracts over the split axis -> gather q and k back.
        query = self._all_gather_channels(query_shard)
        key = self._all_gather_channels(key_shard)

        scores = ttnn.matmul(
            query,
            ttnn.transpose(key, -2, -1),
            compute_kernel_config=self.compute_config,
        )
        scores = ttnn.multiply(scores, self.scale)
        probs = ttnn.softmax(scores, dim=-1)

        # v stays sharded, so the context comes out channel-sharded — the layout
        # the row-parallel out projection consumes.
        context_shard = ttnn.matmul(probs, value_shard, compute_kernel_config=self.compute_config)

        # Row-parallel out proj -> partial sums -> all_reduce -> replicated bias.
        out = ttnn.matmul(context_shard, self.wo, compute_kernel_config=self.compute_config)
        out = self._all_reduce_partials(out)
        out = ttnn.add(out, self.bo)

        if self.residual_connection:
            out = ttnn.add(out, residual)
        if self.rescale_output_factor != 1.0:
            out = ttnn.multiply(out, 1.0 / self.rescale_output_factor)

        if len(original_shape) == 4:
            out = ttnn.reshape(out, original_shape)
        return out


def build(device, torch_module=None):
    return TtVaeAttention(device, torch_module)


def attention(device, torch_module=None):
    return TtVaeAttention(device, torch_module)
