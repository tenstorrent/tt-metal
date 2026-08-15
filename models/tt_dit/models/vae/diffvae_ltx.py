# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 DiffVAE video decoder: deterministic stages.

The 2.5 video VAE replaces the convolutional decoder with a diffusion one, and it is not a
convnet — every block is 3D neighborhood attention over a local window (:mod:`layers.na3d`).
Stages 1-4 here deterministically upsample the latent into a context volume; the diffusion
stage that turns noise plus that context into pixels lives alongside.

Submodules are named to mirror the checkpoint's own keys (``attn.qkv``, ``mlp.w_gate``, ...)
so loading needs no key remapping beyond the two transformations the values themselves
require: splitting the fused QKV, and the RoPE reordering described in :func:`rope_permutation`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

import ttnn

from ...layers.linear import Linear
from ...layers.module import Module, ModuleList
from ...layers.na3d import (
    AxisShard,
    NA3DDevicePlan,
    build_device_plan,
    build_mesh_device_plan,
    neighborhood_attention_3d,
    plan_na3d,
    plan_na3d_mesh,
    uniform_halo,
)
from ...layers.normalization import RMSNorm

ROPE_BASE = 10000.0


def decoder_config(path) -> dict:
    """The decoder's architecture block, read from the checkpoint's safetensors metadata.

    Read rather than hardcoded: the shapes here (stage channels, depths, kernels, upsample
    strides) drive module construction, and a checkpoint whose config disagreed with baked-in
    constants would otherwise fail as a confusing shape error at load.
    """
    import json
    import struct

    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    vae = json.loads(header["__metadata__"]["config"])["vae"]
    config = dict(vae["decoder"])
    for key in ("in_channels", "out_channels", "patch_size", "head_dim", "model_output_type"):
        if key in vae:
            config[key] = vae[key]
    for key in ("stage_kernels", "upsamples", "stage5_kernel"):
        if key in config:
            config[key] = _tuplify(config[key])
    for key in ("stage_channels", "stage_depths"):
        if key in config:
            config[key] = tuple(config[key])
    return config


def _tuplify(value):
    return tuple(_tuplify(v) if isinstance(v, list) else v for v in value)


def default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    """Split of ``head_dim`` across the T, H and W RoPE chunks (64 -> (16, 24, 24))."""
    assert head_dim % 8 == 0, f"head_dim={head_dim} must be a multiple of 8"
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 != 0:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    return (d_t, d_hw, d_hw)


def rope_permutation(rope_dim_split: tuple[int, int, int]) -> torch.Tensor:
    """``head_dim`` reordering that turns upstream's interleaved pairs into two halves.

    Upstream rotates adjacent dim pairs ``(d0,d1), (d2,d3), ...`` within each axis chunk,
    which on device would need a stride-2 gather per rotation. Attention only sees ``q·k``, so
    permuting ``head_dim`` identically in q and k is invisible in the output — and reordering
    to ``[all first-of-pair, all second-of-pair]`` makes RoPE the contiguous
    ``(x1*cos - x2*sin, x1*sin + x2*cos)``. Verified bit-identical to upstream.

    Folded into the q/k projection rows and the q_norm/k_norm weights at load time, so it is
    free at runtime. RMSNorm tolerates it because its scale is over all dims, hence
    permutation-invariant, provided its learned weight is permuted the same way.
    """
    evens, odds, offset = [], [], 0
    for width in rope_dim_split:
        evens.extend(range(offset, offset + width, 2))
        odds.extend(range(offset + 1, offset + width, 2))
        offset += width
    return torch.tensor(evens + odds)


def rope_tables(
    dims: tuple[int, int, int],
    rope_dim_split: tuple[int, int, int],
    *,
    mesh_device,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """``(cos, sin)`` of shape ``(1, T, H, W, 1, head_dim // 2)`` for the permuted layout.

    Each axis contributes ``width // 2`` columns in T, H, W order, matching
    :func:`rope_permutation`. Positions are 0-based and local to the volume: a neighborhood
    window never crosses a tile, and a global phase cancels inside the window's softmax, so
    local and absolute positions give the same attention.
    """
    t, h, w = dims
    cos_columns, sin_columns = [], []
    for axis, (length, width) in enumerate(zip(dims, rope_dim_split)):
        exponents = torch.arange(0, width, 2, dtype=torch.float64) / width
        inv_freq = (1.0 / torch.pow(torch.tensor(ROPE_BASE, dtype=torch.float64), exponents)).to(torch.float32)
        angle = torch.arange(length, dtype=torch.float32)[:, None] * inv_freq[None, :]
        shape = [1, 1, 1, angle.shape[-1]]
        shape[axis] = length
        cos_columns.append(angle.cos().reshape(shape).expand(t, h, w, angle.shape[-1]))
        sin_columns.append(angle.sin().reshape(shape).expand(t, h, w, angle.shape[-1]))

    def upload(columns):
        table = torch.cat(columns, dim=-1).reshape(1, t, h, w, 1, -1)
        return ttnn.from_torch(table, device=mesh_device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    return upload(cos_columns), upload(sin_columns)


def apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    """Rotate a permuted ``(1, T, H, W, heads, head_dim)`` tensor using contiguous halves."""
    shape = list(x.shape)
    half = shape[-1] // 2
    low = ttnn.slice(x, [0] * len(shape), shape[:-1] + [half])
    high = ttnn.slice(x, [0] * (len(shape) - 1) + [half], shape[:-1] + [2 * half])
    rotated = ttnn.concat(
        [
            ttnn.subtract(ttnn.multiply(low, cos), ttnn.multiply(high, sin)),
            ttnn.add(ttnn.multiply(low, sin), ttnn.multiply(high, cos)),
        ],
        dim=-1,
    )
    ttnn.deallocate(low)
    ttnn.deallocate(high)
    return rotated


class NeighborhoodAttention(Module):
    """3D neighborhood attention with absolute RoPE, matching upstream's parameter shell."""

    def __init__(self, dim: int, kernel_size: tuple[int, int, int], *, head_dim: int = 64, mesh_device=None):
        super().__init__()
        assert dim % head_dim == 0, f"dim={dim} not divisible by head_dim={head_dim}"
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5
        self.rope_dim_split = default_rope_dim_split(head_dim)

        self.qkv = Linear(dim, 3 * dim, bias=True, mesh_device=mesh_device)
        self.proj = Linear(dim, dim, bias=True, mesh_device=mesh_device)
        self.q_norm = RMSNorm(head_dim, norm_eps=1e-6, bias=False, mesh_device=mesh_device)
        self.k_norm = RMSNorm(head_dim, norm_eps=1e-6, bias=False, mesh_device=mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Fold the RoPE reordering into q/k weights; v and the output projection are untouched."""
        perm = rope_permutation(self.rope_dim_split)

        def reorder_head_dim(tensor: torch.Tensor) -> torch.Tensor:
            # Rows are head-major: (heads, head_dim, ...) — permute within each head.
            return tensor.reshape(self.num_heads, self.head_dim, *tensor.shape[1:])[:, perm].reshape(tensor.shape)

        for leaf in ("weight", "bias"):
            key = f"qkv.{leaf}"
            if key in state:
                q, k, v = state[key].chunk(3, dim=0)
                state[key] = torch.cat([reorder_head_dim(q), reorder_head_dim(k), v], dim=0)
        for key in ("q_norm.weight", "k_norm.weight"):
            if key in state:
                state[key] = state[key][perm]

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        dims: tuple[int, int, int],
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        device_plan: NA3DDevicePlan,
        exchange: "Callable[[ttnn.Tensor], ttnn.Tensor] | None" = None,
    ) -> ttnn.Tensor:
        """``x`` is ``(tokens, dim)`` in TILE layout; returns ``(queries, dim)``.

        Sharded, ``x`` holds only this device's tokens and ``exchange`` returns them with the
        halo the plan reads — so ``dims`` is the padded buffer, and the output is narrower than
        the input by exactly the halo. Projecting q/k/v on the halo duplicates a slice of the
        neighbour's work, which is the cheaper trade: exchanging x moves one tensor where
        exchanging q/k/v would move three.
        """
        if exchange is not None:
            x = exchange(x)
        t, h, w = dims
        tokens = t * h * w
        fused = self.qkv(x)

        parts = []
        for index in range(3):
            part = ttnn.slice(fused, [0, index * self.dim], [tokens, (index + 1) * self.dim])
            part = ttnn.reshape(part, (tokens * self.num_heads, self.head_dim))
            parts.append(part)
        ttnn.deallocate(fused)
        q, k, v = parts

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = ttnn.multiply(q, self.scale)

        shape = (1, t, h, w, self.num_heads, self.head_dim)
        q, k, v = (ttnn.to_layout(ttnn.reshape(part, shape), ttnn.ROW_MAJOR_LAYOUT) for part in (q, k, v))
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attended = neighborhood_attention_3d(q, k, v, kernel_size=self.kernel_size, scale=1.0, device_plan=device_plan)
        queries = math.prod(device_plan.plan.output_dims)
        attended = ttnn.to_layout(ttnn.reshape(attended, (queries, self.dim)), ttnn.TILE_LAYOUT)
        out = self.proj(attended)
        ttnn.deallocate(attended)
        return out


class SwiGLU(Module):
    """``w_down(silu(w_gate(x)) * w_up(x))``, biasless, as upstream ships it."""

    def __init__(self, dim: int, hidden_dim: int, *, mesh_device=None):
        super().__init__()
        self.w_gate = Linear(dim, hidden_dim, bias=False, mesh_device=mesh_device)
        self.w_up = Linear(dim, hidden_dim, bias=False, mesh_device=mesh_device)
        self.w_down = Linear(hidden_dim, dim, bias=False, mesh_device=mesh_device)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.silu(self.w_gate(x))
        up = self.w_up(x)
        product = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = self.w_down(product)
        ttnn.deallocate(product)
        return out


class NABlock(Module):
    """Pre-norm block: neighborhood attention then SwiGLU, both with residual adds."""

    def __init__(self, dim: int, kernel_size: tuple[int, int, int], *, head_dim: int = 64, mesh_device=None):
        super().__init__()
        # Upstream rounds the 4x MLP ratio up to a multiple of 16.
        hidden = (int(dim * 4.0) + 15) // 16 * 16
        self.norm1 = RMSNorm(dim, norm_eps=1e-6, bias=False, mesh_device=mesh_device)
        self.attn = NeighborhoodAttention(dim, kernel_size, head_dim=head_dim, mesh_device=mesh_device)
        self.norm2 = RMSNorm(dim, norm_eps=1e-6, bias=False, mesh_device=mesh_device)
        self.mlp = SwiGLU(dim, hidden, mesh_device=mesh_device)

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        dims: tuple[int, int, int],
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        device_plan: NA3DDevicePlan,
        exchange: "Callable[[ttnn.Tensor], ttnn.Tensor] | None" = None,
    ) -> ttnn.Tensor:
        """Sharded, ``x`` is this device's tokens throughout: only the attention sees the halo.

        The residual and the MLP are per-token, so they stay on the shard and never need a
        neighbour. One exchange per block is the whole communication cost.
        """
        attended = self.attn(self.norm1(x), dims=dims, cos=cos, sin=sin, device_plan=device_plan, exchange=exchange)
        x = ttnn.add(x, attended)
        ttnn.deallocate(attended)
        projected = self.mlp(self.norm2(x))
        x = ttnn.add(x, projected)
        ttnn.deallocate(projected)
        return x


@dataclass(frozen=True)
class MeshShardConfig:
    """Split the deterministic stages' volume over a 2D mesh: H on axis 0, W on axis 1.

    ``enter_stage`` is the first stage that runs split. The stages before it stay replicated,
    which is what lets the latent grid keep a height the mesh does not divide — 34 over 4 —
    while every grid from the first upsample on divides cleanly.
    """

    ccl: object
    mesh: tuple[int, int]
    enter_stage: int = 1


def _even_shards(length: int, parts: int) -> list[AxisShard]:
    assert length % parts == 0, f"axis {length} does not split evenly over {parts} devices"
    step = length // parts
    return [AxisShard(length=length, start=i * step, stop=(i + 1) * step) for i in range(parts)]


class LinearPixelShuffleUpsample(Module):
    """Channel-expanding Linear then a channels-last 3D pixel shuffle.

    The checkpoint packs the projection's output channels as ``(c p1 p2 p3)`` with the output
    channel outermost, so the shuffle is a reshape-and-transpose rather than a plain view.

    The projection's rows are reordered at load to ``(p1 p2 p3 c)`` — see
    ``channel_permutation`` — which lets the shuffle keep the channel axis innermost.
    """

    def __init__(
        self,
        in_channels: int,
        stride: tuple[int, int, int],
        out_channels_reduction_factor: int = 1,
        *,
        mesh_device=None,
    ):
        super().__init__()
        self.stride = tuple(stride)
        span = self.stride[0] * self.stride[1] * self.stride[2]
        self.proj_out_channels = span * in_channels // out_channels_reduction_factor
        self.out_channels = self.proj_out_channels // span
        self.proj = Linear(in_channels, self.proj_out_channels, bias=True, mesh_device=mesh_device)

    def output_dims(self, dims: tuple[int, int, int], *, drop_leading_frame: bool = True) -> tuple[int, int, int]:
        """Grid this upsample produces, without running it.

        A sharded run still has to track the whole volume's grid alongside its own, because the
        window bounds every shard plans against are global.
        """
        p1, p2, p3 = self.stride
        t, h, w = dims[0] * p1, dims[1] * p2, dims[2] * p3
        if p1 == 2 and drop_leading_frame:
            t -= 1
        return t, h, w

    def channel_permutation(self) -> torch.Tensor:
        """Row order taking the checkpoint's ``(c p1 p2 p3)`` output channels to ``(p1 p2 p3 c)``.

        Applied to the projection's weight and bias at load, so it costs nothing at runtime and
        the shuffle's reshape can keep the channel axis innermost. That placement is not
        cosmetic: with a stride factor of 2 innermost, ROW_MAJOR rounds that extent up to a full
        32-element face, so the tensor occupies 16x its own size — a 2.6 GB activation asks for
        45 GB at 1920x1088, which is exactly how this was found.
        """
        p1, p2, p3 = self.stride
        index = torch.arange(self.proj_out_channels).reshape(self.out_channels, p1, p2, p3)
        return index.permute(1, 2, 3, 0).reshape(-1)

    def forward(
        self, x: ttnn.Tensor, *, dims: tuple[int, int, int], drop_leading_frame: bool = True
    ) -> tuple[ttnn.Tensor, tuple[int, int, int]]:
        """``x`` is ``(tokens, in_channels)``; returns ``(tokens', out_channels)`` and new dims."""
        t, h, w = dims
        p1, p2, p3 = self.stride
        projected = self.proj(x)
        projected = ttnn.to_layout(projected, ttnn.ROW_MAJOR_LAYOUT)

        # (t, h, w, p1, p2, p3, c) -> (t, p1, h, p2, w, p3, c), channels innermost throughout.
        projected = ttnn.reshape(projected, (t, h, w, p1, p2, p3, self.out_channels))
        projected = ttnn.permute(projected, (0, 3, 1, 4, 2, 5, 6))
        out_dims = (t * p1, h * p2, w * p3)
        projected = ttnn.reshape(projected, (out_dims[0], out_dims[1] * out_dims[2], self.out_channels))

        if p1 == 2 and drop_leading_frame:
            # The temporal shuffle emits a duplicate first frame; dropping it preserves the
            # causal 1:2 (composed 1:8) mapping. Only the chunk holding the true t=0 has one.
            projected = ttnn.slice(projected, [1, 0, 0], [out_dims[0], out_dims[1] * out_dims[2], self.out_channels])
            out_dims = (out_dims[0] - 1, out_dims[1], out_dims[2])

        tokens = out_dims[0] * out_dims[1] * out_dims[2]
        projected = ttnn.to_layout(ttnn.reshape(projected, (tokens, self.out_channels)), ttnn.TILE_LAYOUT)
        return projected, out_dims


class DeterministicStages(Module):
    """Stages 1-4: NA blocks and upsamples that turn the latent into the stage-5 context.

    ``conv_in`` lives here so the latent's per-channel denormalization can be folded into it:
    the decoder's first act is ``conv_in(x * std + mean)``, which is exactly a Linear with
    ``std`` scaled into the weight columns and ``W @ mean`` added to the bias. Free and exact.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        stage_channels: tuple[int, ...],
        stage_depths: tuple[int, ...],
        stage_kernels: tuple[tuple[int, int, int], ...],
        upsamples: tuple[tuple[tuple[int, int, int], int], ...],
        head_dim: int = 64,
        mesh_device=None,
    ):
        super().__init__()
        assert len(upsamples) == len(stage_channels) - 1, "one upsample between consecutive stages"
        self.stage_kernels = stage_kernels
        self.head_dim = head_dim
        self.mesh_device = mesh_device
        self.conv_in = Linear(in_channels, stage_channels[0], bias=True, mesh_device=mesh_device)

        self.det_stages = ModuleList(
            [
                ModuleList(
                    [
                        NABlock(
                            stage_channels[stage],
                            stage_kernels[stage],
                            head_dim=head_dim,
                            mesh_device=mesh_device,
                        )
                        for _ in range(stage_depths[stage])
                    ]
                )
                for stage in range(len(upsamples))
            ]
        )
        self.upsamples = ModuleList(
            [
                LinearPixelShuffleUpsample(
                    stage_channels[stage], upsamples[stage][0], upsamples[stage][1], mesh_device=mesh_device
                )
                for stage in range(len(upsamples))
            ]
        )
        self._plan_cache: dict[tuple, NA3DDevicePlan] = {}
        self._rope_cache: dict[tuple, tuple[ttnn.Tensor, ttnn.Tensor]] = {}
        self._mesh_plan_cache: dict[tuple, tuple[NA3DDevicePlan, tuple[int, int, int], tuple[int, int, int]]] = {}

    def _plan(self, dims: tuple[int, int, int], kernel: tuple[int, int, int]) -> NA3DDevicePlan:
        key = (dims, kernel)
        if key not in self._plan_cache:
            self._plan_cache[key] = build_device_plan(plan_na3d(dims, kernel), mesh_device=self.mesh_device)
        return self._plan_cache[key]

    def _mesh_plan(
        self, dims: tuple[int, int, int], kernel: tuple[int, int, int], shard: "MeshShardConfig"
    ) -> tuple[NA3DDevicePlan, tuple[int, int, int], tuple[int, int, int]]:
        """Device plan, buffer grid and halo for one stage of a mesh-split volume.

        ``dims`` is the whole grid; the split is H over cluster axis 0 and W over axis 1, with
        T left alone because a mesh has two axes and T is the one that divides worst.
        """
        key = (dims, kernel, shard.mesh)
        if key not in self._mesh_plan_cache:
            halo = (0,) + tuple(uniform_halo(dims[i], shard.mesh[i - 1], kernel[i]) for i in (1, 2))
            grid = [
                (AxisShard(dims[0], 0, dims[0]), h, w)
                for h in _even_shards(dims[1], shard.mesh[0])
                for w in _even_shards(dims[2], shard.mesh[1])
            ]
            plans = plan_na3d_mesh(grid, kernel, halo=halo)
            self._mesh_plan_cache[key] = (
                build_mesh_device_plan(plans, mesh_device=self.mesh_device),
                plans[0].dims,
                halo,
            )
        return self._mesh_plan_cache[key]

    def _enter_shard(
        self, x: ttnn.Tensor, dims: tuple[int, int, int], channels: int, shard: "MeshShardConfig"
    ) -> tuple[ttnn.Tensor, tuple[int, int, int]]:
        """Replicated volume to one shard per device, via host.

        Each device keeps a different sub-box, which no device-side op expresses: a ttnn program
        carries the same arguments everywhere. The round trip is paid once, and the entry stage
        is chosen so the volume is still small when it happens.
        """
        local = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).reshape(1, *dims, channels)
        sharded = ttnn.from_torch(
            local,
            device=self.mesh_device,
            dtype=x.get_dtype(),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=shard.mesh, dims=[2, 3]),
        )
        query = (dims[0], dims[1] // shard.mesh[0], dims[2] // shard.mesh[1])
        return ttnn.to_layout(ttnn.reshape(sharded, (-1, channels)), ttnn.TILE_LAYOUT), query

    def _rope(self, dims: tuple[int, int, int]) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if dims not in self._rope_cache:
            self._rope_cache[dims] = rope_tables(
                dims, default_rope_dim_split(self.head_dim), mesh_device=self.mesh_device
            )
        return self._rope_cache[dims]

    def state_from_checkpoint(self, path, *, statistics: bool = True) -> dict[str, torch.Tensor]:
        """Load ``decoder.*`` tensors from an LTX-2.5 video-VAE safetensors file.

        Also folds ``per_channel_statistics`` into ``conv_in`` when present, so the caller
        hands us the same normalized latent the conv decoder takes.
        """
        from safetensors import safe_open

        state: dict[str, torch.Tensor] = {}
        stats: dict[str, torch.Tensor] = {}
        with safe_open(str(path), "pt") as handle:
            for key in handle.keys():
                if key.startswith("per_channel_statistics."):
                    stats[key[len("per_channel_statistics.") :]] = handle.get_tensor(key).float()
                    continue
                if not key.startswith("decoder."):
                    continue
                name = key[len("decoder.") :]
                if name.startswith(("conv_in.", "det_stages.", "upsamples.")):
                    stage = name.split(".")[1] if name.startswith("det_stages.") else None
                    # det_stages holds the diffusion blocks' config slot as its last entry in
                    # the config, but on disk only the deterministic stages appear here.
                    if stage is not None and int(stage) >= len(self.det_stages):
                        continue
                    state[name] = handle.get_tensor(key).float()

        if statistics and "std-of-means" in stats:
            weight, bias = state["conv_in.weight"], state["conv_in.bias"]
            std, mean = stats["std-of-means"], stats["mean-of-means"]
            state["conv_in.weight"] = weight * std[None, :]
            state["conv_in.bias"] = bias + weight @ mean

        for index, upsample in enumerate(self.upsamples):
            order = upsample.channel_permutation()
            for leaf in ("weight", "bias"):
                key = f"upsamples.{index}.proj.{leaf}"
                state[key] = state[key][order]

        return state

    def load_checkpoint(self, path, *, statistics: bool = True) -> None:
        self.load_state_dict(self.state_from_checkpoint(path, statistics=statistics))

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        dims: tuple[int, int, int],
        drop_leading_frame: bool = True,
        stages: int | None = None,
        shard: MeshShardConfig | None = None,
    ) -> tuple[ttnn.Tensor, tuple[int, int, int]]:
        """``x`` is ``(tokens, in_channels)`` channels-last in TILE layout, normalized latent.

        With ``shard``, the returned volume is this device's share and ``dims`` describes it;
        the caller reassembles. The upsamples need no special handling — a pixel shuffle only
        redistributes a token's own channels, so a shard's boundaries scale with the stride and
        stay aligned with its neighbours'.
        """
        x = self.conv_in(x)
        count = len(self.upsamples) if stages is None else stages
        whole = dims
        for stage in range(count):
            kernel = self.stage_kernels[stage]
            if shard is not None and stage == shard.enter_stage:
                x, dims = self._enter_shard(x, dims, x.shape[-1], shard)

            if shard is not None and stage >= shard.enter_stage:
                plan, buffer_dims, halo = self._mesh_plan(whole, kernel, shard)
                cos, sin = self._rope(buffer_dims)
                exchange = self._halo_exchange(shard, halo, dims, x.shape[-1])
                grid = buffer_dims
            else:
                plan = self._plan(dims, kernel)
                cos, sin = self._rope(dims)
                exchange, grid = None, dims

            for block in self.det_stages[stage]:
                x = block(x, dims=grid, cos=cos, sin=sin, device_plan=plan, exchange=exchange)
            x, dims = self.upsamples[stage](x, dims=dims, drop_leading_frame=drop_leading_frame)
            whole = self.upsamples[stage].output_dims(whole, drop_leading_frame=drop_leading_frame)
        return x, dims

    def _halo_exchange(self, shard: MeshShardConfig, halo, query_dims, channels):
        """Wrap a shard's tokens back into a volume, pad from the neighbours, flatten again."""

        def exchange(tokens: ttnn.Tensor) -> ttnn.Tensor:
            volume = ttnn.reshape(ttnn.to_layout(tokens, ttnn.ROW_MAJOR_LAYOUT), (1, *query_dims, channels))
            padded = shard.ccl.neighbor_pad(
                volume,
                dims=[2, 3],
                pad_left=[halo[1], halo[2]],
                pad_right=[halo[1], halo[2]],
                padding_mode="replicate",
                axes=[0, 1],
                neighbor_sems=[
                    shard.ccl.get_np_ping_pong_semaphore(0),
                    shard.ccl.get_np_ping_pong_semaphore(1),
                ],
                num_links=[1, 1],
            )
            return ttnn.to_layout(ttnn.reshape(padded, (-1, channels)), ttnn.TILE_LAYOUT)

        return exchange


class DiffVAEDecoder(Module):
    """The whole LTX-2.5 diffusion video decoder: deterministic stages then the diffusion stage.

    Takes the same normalized latent the conv decoder takes and returns pixels, so it can drop
    into the pipeline in its place.

    Two temporal adjustments frame the deterministic stages and are easy to mistake for noise:
    before stage 1 the last latent frame is replicated (:attr:`ghost_latent_frames` times) to
    give NATTEN a trailing border, and after stage 4 that appendix is cropped back off. They
    are inverse halves of one workaround — apply the pad without the crop and the video grows
    16 spurious frames.
    """

    def __init__(self, config: dict, *, mesh_device, dtype: ttnn.DataType = ttnn.bfloat16):
        super().__init__()
        from .diffvae_ltx_stage5 import DiffVAEStage5, DiffVAEStage5Config

        self.config = config
        self.mesh_device = mesh_device
        self.patch_size = config["patch_size"]
        self.out_channels = config["out_channels"]
        self.in_channels = config["in_channels"]
        self.stage5_kernel = config["stage5_kernel"]
        # Upstream: (stage_kernels[0][0] // 2) * 2 latent frames of trailing replication.
        self.ghost_latent_frames = (config["stage_kernels"][0][0] // 2) * 2
        # Composed temporal upscale of the four upsamples, which is also the ghost's pixel cost.
        self.time_scale = math.prod(stride[0] for stride, _ in config["upsamples"])

        self.stages = DeterministicStages(
            in_channels=config["in_channels"],
            stage_channels=config["stage_channels"],
            stage_depths=config["stage_depths"],
            stage_kernels=config["stage_kernels"],
            upsamples=config["upsamples"],
            head_dim=config["head_dim"],
            mesh_device=mesh_device,
        )
        self.stage5 = DiffVAEStage5(
            DiffVAEStage5Config(
                dim=config["stage_channels"][-1],
                head_dim=config["head_dim"],
                kernel_size=config["stage5_kernel"],
                context_channels=config["stage_channels"][-1],
                mlp_hidden=4 * config["stage_channels"][-1],
                num_blocks=config["stage_depths"][-1],
                patch_size=config["patch_size"],
                out_channels=config["out_channels"],
                timestep_scale_multiplier=config["timestep_scale_multiplier"],
            ),
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.dtype = dtype

    def load_checkpoint(self, path) -> None:
        """Load both halves from one LTX-2.5 video-VAE safetensors file."""
        from safetensors import safe_open

        self.stages.load_checkpoint(path)

        prefixes = ("diff_blocks.", "shared_adaln.", "t_embedder.", "conv_in_x_t.", "conv_out.", "norm_out.")
        state: dict[str, torch.Tensor] = {}
        with safe_open(str(path), "pt") as handle:
            for key in handle.keys():
                if not key.startswith("decoder."):
                    continue
                name = key[len("decoder.") :]
                if name.startswith(prefixes):
                    state[name] = handle.get_tensor(key).float()
        self.stage5.load_torch_state_dict(state)

    def context_frames(self, latent_frames: int) -> int:
        """Stage-5 temporal extent for a latent of ``latent_frames``, after pad and crop."""
        padded = latent_frames + self.ghost_latent_frames
        # Each temporal upsample doubles then drops its duplicate leading frame, so the
        # composed map is causal: n -> time_scale * (n - 1) + 1.
        grown = self.time_scale * (padded - 1) + 1
        return max(grown - self.ghost_latent_frames * self.time_scale, self.stage5_kernel[0])

    def forward_context(
        self, latent: torch.Tensor, *, shard: MeshShardConfig | None = None
    ) -> tuple[ttnn.Tensor, tuple[int, int, int]]:
        """Deterministic stages on a ``(B, C, T, H, W)`` normalized latent, ghost cropped.

        The returned dims are the whole context volume even when the tensor is one shard of it:
        stage 5 plans against the global grid and derives its own share, so the whole extent is
        the useful thing to hand back. The ghost crop is temporal and T is never split, so it
        applies unchanged to a shard.
        """
        batch, channels, t, h, w = latent.shape
        assert batch == 1, f"batched decode is not implemented; got batch={batch}"
        assert channels == self.in_channels, f"latent has {channels} channels, expected {self.in_channels}"

        padded = torch.cat([latent, latent[:, :, -1:].expand(-1, -1, self.ghost_latent_frames, -1, -1)], dim=2)
        tokens = padded.permute(0, 2, 3, 4, 1).reshape(-1, channels).contiguous()
        x = ttnn.from_torch(tokens, device=self.mesh_device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)

        x, dims = self.stages(x, dims=(padded.shape[2], h, w), shard=shard)
        keep = self.context_frames(t)
        if keep < dims[0]:
            channels_out = self.config["stage_channels"][-1]
            x = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (dims[0], dims[1] * dims[2], channels_out))
            x = ttnn.slice(x, [0, 0, 0], [keep, dims[1] * dims[2], channels_out])
            x = ttnn.to_layout(ttnn.reshape(x, (keep * dims[1] * dims[2], channels_out)), ttnn.TILE_LAYOUT)
            dims = (keep, dims[1], dims[2])
        if shard is not None:
            dims = (dims[0], dims[1] * shard.mesh[0], dims[2] * shard.mesh[1])
        return x, dims

    def decode(
        self,
        latent: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        seed: int = 0,
        shard: MeshShardConfig | None = None,
    ) -> torch.Tensor:
        """Normalized ``(B, C, T, H, W)`` latent to ``(B, 3, T', H', W')`` pixels.

        ``noise`` is an input, not an implementation detail: stage 5 predicts x0 from it in a
        single step. Pass it to compare against a reference that drew its own.

        ``shard`` splits the volume over a 2D mesh from ``shard.enter_stage`` onward and returns
        the same pixels, gathered. Whether a grid divides is checked stage by stage, so an entry
        chosen too early fails at the split rather than silently rebalancing.
        """
        from .diffvae_ltx_stage5 import Grid

        context, dims = self.forward_context(latent, shard=shard)
        grid = Grid(batch=1, t=dims[0], h=dims[1], w=dims[2])
        sites = grid.sites if shard is None else self.stage5.shard_grid(grid, shard).sites
        context = ttnn.reshape(context, (1, 1, sites, self.config["stage_channels"][-1]))

        if noise is None:
            shape = (1, self.out_channels, grid.t, grid.h * self.patch_size, grid.w * self.patch_size)
            noise = torch.randn(shape, generator=torch.Generator().manual_seed(seed))

        # default_num_inference_steps is 1 on this checkpoint, so linspace(1, 1, 1) = [1.0].
        timestep = ttnn.from_torch(
            torch.tensor([[[[1.0]]]]), device=self.mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
        )
        return self.stage5.forward(context, noise, timestep, grid, shard=shard)

    def forward(
        self,
        latent: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        seed: int = 0,
        shard: MeshShardConfig | None = None,
    ) -> torch.Tensor:
        return self.decode(latent, noise=noise, seed=seed, shard=shard)
