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
import os

import torch

import ttnn

from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear
from ...layers.module import Module, ModuleList
from ...layers.na3d import (
    NA3DDevicePlan,
    build_device_plan,
    neighborhood_attention_3d,
    neighborhood_attention_3d_op_sp_sharded,
    neighborhood_attention_3d_op_sp_w_sharded,
    plan_na3d,
)
from ...layers.normalization import RMSNorm
from ...utils import decode_tree
from .diffvae_ltx_stage5 import TILE, block_prof, deep_prof, log_dram, stage_timer

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


#: Bytes one pointwise chunk may hold. Everything in a deterministic block except the attention
#: gather is pointwise in the site axis, and the widths involved are multiples of the activation:
#: the SwiGLU's hidden width is 4x, so its three intermediates come to 30 GiB where the block's own
#: activation is 2.5 GiB at 6s 1920x1088. Chunking bounds that by the chunk instead of the volume.
#: A GiB keeps per-chunk dispatch negligible against a 2.6M-row volume.
CHUNK_BYTES = 1 << 30


def _chunk_rows(width: int, *, dtype_bytes: int = 2) -> int:
    """Rows whose ``width``-wide intermediate fits :data:`CHUNK_BYTES`, tile-aligned."""
    rows = CHUNK_BYTES // (width * dtype_bytes)
    return max(TILE, rows // TILE * TILE)


def _pointwise_in_chunks(x: ttnn.Tensor, fn, *, width: int) -> ttnn.Tensor:
    """Apply the pointwise ``fn`` to row chunks of ``(rows, ·)`` ``x``. **Consumes** ``x``.

    ``width`` is the widest intermediate ``fn`` builds per row, which is what sets the chunk size.
    Running whole is kept as a distinct path so short videos pay nothing: a single chunk would
    otherwise add a concat of the entire output.
    """
    rows, columns = int(x.shape[-2]), int(x.shape[-1])
    step = _chunk_rows(width)
    if step >= rows:
        out = fn(x)
        ttnn.deallocate(x)
        return out

    parts = []
    for start in range(0, rows, step):
        chunk = ttnn.slice(x, [start, 0], [min(start + step, rows), columns])
        parts.append(fn(chunk))
        ttnn.deallocate(chunk)
    ttnn.deallocate(x)
    joined = ttnn.concat(parts, dim=-2)
    for part in parts:
        ttnn.deallocate(part)
    return joined


def _consume(x: ttnn.Tensor, op, *args) -> ttnn.Tensor:
    """``op(x, *args)``, freeing ``x`` unless ``op`` handed it straight back.

    Only for ops that copy — retiling, permuting, slicing. Leaving ``x`` to Python's next rebind
    keeps a second copy of the volume live across the following allocation, and at 6s 1920x1088
    those copies are 10 GiB each.
    """
    out = op(x, *args)
    if out is not x:
        ttnn.deallocate(x)
    return out


def _consume_pair(op, a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
    """``op(a, b)``, freeing both operands. These are half-volume tensors at full resolution."""
    out = op(a, b)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return out


def apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    """Rotate a permuted ``(1, T, H, W, heads, head_dim)`` tensor using contiguous halves.

    **Consumes** ``x``, and builds the two output halves one after the other rather than as one
    expression. The single-expression form keeps all four products live at once, which at 6s
    1920x1088 is 5 GiB of temporaries on top of the halves and the result.
    """
    shape = list(x.shape)
    half = shape[-1] // 2
    low = ttnn.slice(x, [0] * len(shape), shape[:-1] + [half])
    high = ttnn.slice(x, [0] * (len(shape) - 1) + [half], shape[:-1] + [2 * half])
    ttnn.deallocate(x)

    first = _consume_pair(ttnn.subtract, ttnn.multiply(low, cos), ttnn.multiply(high, sin))
    second = _consume_pair(ttnn.add, ttnn.multiply(low, sin), ttnn.multiply(high, cos))
    ttnn.deallocate(low)
    ttnn.deallocate(high)

    rotated = ttnn.concat([first, second], dim=-1)
    ttnn.deallocate(first)
    ttnn.deallocate(second)
    return rotated


class NeighborhoodAttention(Module):
    """3D neighborhood attention with absolute RoPE, matching upstream's parameter shell."""

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        *,
        head_dim: int = 64,
        mesh_device=None,
        na3d_backend: str = "gather",
        ccl_manager=None,
        sp_axis: int | None = None,
        tp_axis: int | None = None,
    ):
        super().__init__()
        # No ccl_manager here for the gather path: the deterministic stages build their device plans
        # up front and pass them in, and a sharded plan carries the manager that reassembles it. The
        # "op_sp" backend does need one (plus sp_axis), so it can be supplied here.
        assert dim % head_dim == 0, f"dim={dim} not divisible by head_dim={head_dim}"
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5
        self.mesh_device = mesh_device  # only for the deep-profile spans; nothing else reads it here
        # NA3D executor: "gather" (grouped gather + dense masked attention with the passed-in
        # device_plan), "op" (the SDPA op's on-device neighborhood_3d mask, replicated), or "op_sp"
        # (the op path with the attention split over T across sp_axis via ccl_manager).
        self.na3d_backend = na3d_backend
        self.ccl_manager = ccl_manager
        self.sp_axis = sp_axis
        # TP-over-heads on the orthogonal mesh axis (only under op_sp_w_sharded): the attention head-
        # slices q/k/v over tp_axis and gathers back before the out-proj -- makes the det stages 2-D
        # (W-SP x TP-heads), tapping the axis they'd otherwise run replicated over. num_heads varies
        # per stage (32/16/8/8) but all divide 4. The det RoPE tables broadcast over heads, so no rebuild.
        self.tp_axis = tp_axis
        self.rope_dim_split = default_rope_dim_split(head_dim)

        # DIFFVAE_DET_FUSED_QKV=1: keep the checkpoint's fused qkv as one matmul and split it with
        # nlp_create_qkv_heads, partitioning the heads over tp_axis FIRST. The head split otherwise
        # happens inside the attention, so q/k norm, the scale and both RoPEs run on all the heads
        # and three quarters of that is then discarded; partitioning up front makes them 1/tp.
        # DIFFVAE_DET_COLPAR_QKV=1 goes further: shard the fused weight on its output axis so each
        # chip's matmul only ever computes its own heads. The partition then has nothing to do --
        # the output is born 3*dim/tp wide -- and nlp_create_qkv_heads splits it locally.
        self.colpar_qkv = tp_axis is not None and os.environ.get("DIFFVAE_DET_COLPAR_QKV") == "1"
        # Deliberately not gated on tp_axis. A TP axis is what the head *partition* needs; what
        # fused RoPE needs is the (B, NH, S, HD) TILE layout nlp_create_qkv_heads emits, which a
        # replicated stage wants just as much. At tp=1 the partition is skipped and the layout
        # benefit stands alone -- this is what lets stage 1 reach the RoPE and qkv fast paths.
        self.fused_qkv = self.colpar_qkv or os.environ.get("DIFFVAE_DET_FUSED_QKV") == "1"
        # DIFFVAE_DET_FUSED_ROPE=1: rotate with one rotary_embedding_hf per lane instead of the
        # nine-op halves form. The weight fold already puts q/k in HF's rotate_half convention, so
        # this is the same arithmetic; what it needs is a full-width cos/sin rather than the half
        # tables that form implies. Applied while still TILE, which is also what lets q and k skip
        # the ROW_MAJOR trip the hand-rolled version required.
        self.fused_rope = self.fused_qkv and os.environ.get("DIFFVAE_DET_FUSED_ROPE") == "1"
        # DIFFVAE_DET_FLAT_SEQ=1: hand the attention the (B, NH, S, HD) that create_heads emits and
        # take (tokens, dim) back, instead of routing through the 6-D volume its other backends
        # take. The volume is an interface, not a computation: to_seq tears down what to_volume just
        # built, and the round trip copies q/k/v three times where the reorder needs one.
        # Needs the fused RoPE, which is the only form that leaves q/k in TILE (B, NH, S, HD).
        # Backend-gated because only the W-sharded attention implements the flat path; the gather
        # backend takes the 6-D volume, and reaching it with flat_seq set would hand a replicated
        # stage the sharded kernel.
        self.flat_seq = (
            self.fused_rope and self.na3d_backend == "op_sp_w_sharded" and os.environ.get("DIFFVAE_DET_FLAT_SEQ") == "1"
        )
        self._fused_rope_cache: dict = {}
        self.tp = int(list(mesh_device.shape)[tp_axis]) if tp_axis is not None else 1
        if self.fused_qkv:
            assert self.num_heads % self.tp == 0, f"num_heads={self.num_heads} not divisible by tp={self.tp}"
        self.heads_local = self.num_heads // self.tp if self.fused_qkv else self.num_heads

        # Three projections rather than the checkpoint's fused one. Fused, the (rows, 3*dim) output
        # and the three slices taken from it are live together -- six activations, 15 GiB at 6s
        # 1920x1088 -- where separate matmuls hold three. The weight is split at load instead.
        # Under fused_qkv the partition lands immediately after the matmul, so what is held is
        # (rows, 3*dim) plus (rows, 3*dim/tp), not six full-width activations.
        if self.fused_qkv:
            qkv_linear = {"bias": True, "mesh_device": mesh_device}
            if self.colpar_qkv:
                # device_major already orders the weight [dev][q|k|v][heads/tp], which is exactly
                # what a contiguous column shard needs.
                qkv_linear["weight_mesh_axes"] = [None, tp_axis]
                qkv_linear["bias_mesh_axes"] = [None, tp_axis]
            self.qkv = Linear(dim, 3 * dim, **qkv_linear)
        else:
            self.to_q = Linear(dim, dim, bias=True, mesh_device=mesh_device)
            self.to_k = Linear(dim, dim, bias=True, mesh_device=mesh_device)
            self.to_v = Linear(dim, dim, bias=True, mesh_device=mesh_device)
        self.proj = Linear(dim, dim, bias=True, mesh_device=mesh_device)
        self.q_norm = RMSNorm(head_dim, norm_eps=1e-6, bias=False, mesh_device=mesh_device)
        self.k_norm = RMSNorm(head_dim, norm_eps=1e-6, bias=False, mesh_device=mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Split the shipped fused ``qkv`` into three projections and fold the RoPE reordering
        into the q/k halves; v and the output projection are untouched."""
        perm = rope_permutation(self.rope_dim_split)

        def reorder_head_dim(tensor: torch.Tensor) -> torch.Tensor:
            # Rows are head-major: (heads, head_dim, ...) — permute within each head.
            return tensor.reshape(self.num_heads, self.head_dim, *tensor.shape[1:])[:, perm].reshape(tensor.shape)

        def device_major(fused: torch.Tensor) -> torch.Tensor:
            """Reorder ``[q_all | k_all | v_all]`` rows to ``[dev][q|k|v][heads/tp]``.

            ``mesh_partition`` takes a contiguous slice of the output dim, so this is what makes
            that slice this chip's own ``[q | k | v]`` — the layout ``nlp_create_qkv_heads`` then
            expects. Sharding the shipped order instead would hand chip 0 all of q and part of k.
            """
            rest = fused.shape[1:]
            grouped = fused.reshape(3, self.tp, self.dim // self.tp, *rest)
            axes = (1, 0, 2, *range(3, 3 + len(rest)))
            return grouped.permute(*axes).reshape(3 * self.dim, *rest)

        for leaf in ("weight", "bias"):
            key = f"qkv.{leaf}"
            if key in state:
                q, k, v = state.pop(key).chunk(3, dim=0)
                q, k = reorder_head_dim(q), reorder_head_dim(k)
                if self.fused_qkv:
                    state[key] = device_major(torch.cat([q, k, v], dim=0))
                else:
                    state[f"to_q.{leaf}"] = q
                    state[f"to_k.{leaf}"] = k
                    state[f"to_v.{leaf}"] = v
        for key in ("q_norm.weight", "k_norm.weight"):
            if key in state:
                state[key] = state[key][perm]

    def _fused_rope_tables(self, cos: ttnn.Tensor, sin: ttnn.Tensor, tokens: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``(cos, sin)`` as ``rotary_embedding_hf`` wants them: full head_dim, flat, TILE.

        The halves form only needs ``head_dim/2`` columns; HF's carries each frequency twice, so
        the table is its own halves concatenated. Cached because the stage hands the same tables to
        every block, and this is otherwise three ops per block for a tensor that never changes.
        """
        key = (tokens, int(cos.shape[-1]))
        cached = self._fused_rope_cache.get(key)
        if cached is None:

            def prepare(table: ttnn.Tensor) -> ttnn.Tensor:
                doubled = ttnn.concat([table, table], dim=-1)
                flat = ttnn.reshape(doubled, (1, 1, tokens, self.head_dim))
                out = ttnn.to_layout(flat, ttnn.TILE_LAYOUT)
                ttnn.deallocate(doubled)
                return out

            cached = (prepare(cos), prepare(sin))
            self._fused_rope_cache[key] = cached
        return cached

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        dims: tuple[int, int, int],
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        device_plan: NA3DDevicePlan,
    ) -> ttnn.Tensor:
        t, h, w = dims
        tokens = t * h * w
        heads = self.heads_local
        with deep_prof(self.mesh_device, "qkv-proj", category=decode_tree.PROJ):
            if self.fused_qkv:
                flat = self.qkv(x)
                ttnn.deallocate(x)
                qkv = ttnn.reshape(flat, (1, 1, tokens, int(flat.shape[-1])))
                if not self.colpar_qkv and self.tp > 1:
                    # The head partition lands here rather than inside the attention, so the norms, the
                    # scale and both RoPEs below see heads/tp instead of computing every head and
                    # letting the attention discard all but this chip's. Under colpar_qkv the matmul
                    # already emitted only this chip's heads, so there is nothing to slice.
                    partitioned = ttnn.mesh_partition(qkv, dim=3, cluster_axis=self.tp_axis)
                    ttnn.deallocate(qkv)
                    qkv = partitioned
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    qkv, num_heads=heads, num_kv_heads=heads, transpose_k_heads=False
                )
                ttnn.deallocate(qkv)
            else:
                heads_shape = (tokens * self.num_heads, self.head_dim)
                q, k, v = (ttnn.reshape(project(x), heads_shape) for project in (self.to_q, self.to_k, self.to_v))
                ttnn.deallocate(x)

        with deep_prof(self.mesh_device, "qkv-norm", category=decode_tree.NORM_ROPE):
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = ttnn.multiply(q, self.scale)

        if self.fused_rope:
            with deep_prof(self.mesh_device, "qkv-rope (fused op)", category=decode_tree.NORM_ROPE):
                # Rotate here, before the volume reshape: the op wants TILE, and q/k are still in the
                # (1, heads, tokens, head_dim) form create_heads emitted.
                cos_full, sin_full = self._fused_rope_tables(cos, sin, tokens)
                q = _consume(q, ttnn.experimental.rotary_embedding_hf, cos_full, sin_full)
                k = _consume(k, ttnn.experimental.rotary_embedding_hf, cos_full, sin_full)

        # Untilize before splitting out the head axis, never after. TILE rounds both of the last
        # two dims up to 32, so a trailing (num_heads, head_dim) of (4, 64) is padded 8x: at
        # 1920x1088 that turns a 1.35 GB activation into a 10.83 GB allocation, three times over
        # for q/k/v, and it was what stopped a 6s decode. In ROW_MAJOR the same split is a pure
        # stride change, and the attention gathers rows in ROW_MAJOR anyway.
        shape = (1, t, h, w, heads, self.head_dim)

        def to_volume(part: ttnn.Tensor) -> ttnn.Tensor:
            part = _consume(part, ttnn.to_layout, ttnn.ROW_MAJOR_LAYOUT)
            if self.fused_qkv:
                # create_heads emits (1, heads, tokens, head_dim); the volume wants heads innermost.
                part = _consume(part, ttnn.permute, (0, 2, 1, 3))
            return ttnn.reshape(part, shape)

        if self.flat_seq:
            # The attention takes the (B, NH, S, HD) it was handed and returns (tokens, dim), so the
            # volume never exists: building it costs a permute here and two more to undo, for a
            # reorder of the sequence axis that one permute inside does on its own.
            attended = neighborhood_attention_3d_op_sp_w_sharded(
                q,
                k,
                v,
                dims=(t, h, w * int(list(q.device().shape)[self.sp_axis])),
                kernel_size=self.kernel_size,
                sp_axis=self.sp_axis,
                ccl_manager=self.ccl_manager,
                scale=1.0,
                tp_axis=self.tp_axis,
                heads_presharded=True,
                flat_seq=True,
            )
            for part in (q, k, v):
                ttnn.deallocate(part)
            out = self.proj(attended)
            ttnn.deallocate(attended)
            return out

        with deep_prof(self.mesh_device, "qkv-to-volume", category=decode_tree.RESHAPE):
            q, k, v = (to_volume(part) for part in (q, k, v))
        if not self.fused_rope:
            with deep_prof(self.mesh_device, "qkv-rope (unfused)", category=decode_tree.NORM_ROPE):
                q = apply_rope(q, cos, sin)
                k = apply_rope(k, cos, sin)

        if self.na3d_backend == "op_sp_sharded":
            # Full-stage SP: x/q/k/v are this chip's T-slice, so `dims` here is local. The attention
            # needs the full grid + this shard's global origin; K/V are gathered inside. cos/sin were
            # already sharded the same way, so the RoPE above used each frame's global position.
            sp = int(list(q.device().shape)[self.sp_axis])
            attended = neighborhood_attention_3d_op_sp_sharded(
                q,
                k,
                v,
                dims=(t * sp, h, w),
                kernel_size=self.kernel_size,
                sp_axis=self.sp_axis,
                ccl_manager=self.ccl_manager,
                scale=1.0,
            )
        elif self.na3d_backend == "op_sp_w_sharded":
            # Full-stage spatial-W SP: x/q/k/v are this chip's W-slice, so `dims` here is local (w is
            # W/sp). The attention needs the full W; K/V are gathered inside over the W-outer flatten.
            # cos/sin were W-sharded the same way, so the RoPE above used each column's global W.
            sp = int(list(q.device().shape)[self.sp_axis])
            attended = neighborhood_attention_3d_op_sp_w_sharded(
                q,
                k,
                v,
                dims=(t, h, w * sp),
                kernel_size=self.kernel_size,
                sp_axis=self.sp_axis,
                ccl_manager=self.ccl_manager,
                scale=1.0,
                tp_axis=self.tp_axis,
                heads_presharded=self.fused_qkv,
            )
        else:
            # Stage 0's path: the grouped-gather executor. Named so stage 0 stops being one opaque row.
            with deep_prof(self.mesh_device, f"na3d {self.na3d_backend}", category=decode_tree.SDPA):
                attended = neighborhood_attention_3d(
                    q,
                    k,
                    v,
                    kernel_size=self.kernel_size,
                    scale=1.0,
                    device_plan=device_plan,
                    backend=self.na3d_backend,
                    ccl_manager=self.ccl_manager,
                    sp_axis=self.sp_axis,
                )
        with deep_prof(self.mesh_device, "out-proj", category=decode_tree.PROJ):
            attended = ttnn.to_layout(ttnn.reshape(attended, (tokens, self.dim)), ttnn.TILE_LAYOUT)
            out = self.proj(attended)
            ttnn.deallocate(attended)
        return out


class SwiGLU(Module):
    """``w_down(silu(w_gate(x)) * w_up(x))``, biasless, as upstream ships it."""

    def __init__(self, dim: int, hidden_dim: int, *, mesh_device=None, tp_axis=None, ccl_manager=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.tp_axis = tp_axis
        self.ccl_manager = ccl_manager
        # DIFFVAE_DET_TP_MLP=1 shards hidden_dim over tp_axis. Every hidden column is independent
        # until w_down contracts over it, so gate/up are column-parallel and w_down row-parallel;
        # its reduce_scatter plus the all_gather below is an all-reduce, which is the one collective
        # the split costs. Implies the packed weight, since the fused kernel is what consumes it.
        self.tp_mlp = tp_axis is not None and os.environ.get("DIFFVAE_DET_TP_MLP") == "1"
        # DIFFVAE_DET_FUSED_SWIGLU=1 packs [up | gate] into one GEMM whose epilogue emits
        # silu(gate) * up, so the separate silu and multiply passes over the 4x-wide hidden
        # activation disappear along with their buffers.
        self.fused = self.tp_mlp or os.environ.get("DIFFVAE_DET_FUSED_SWIGLU") == "1"

        if self.tp_mlp:
            self.gate_up = ColParallelLinear(
                dim,
                hidden_dim,
                bias=False,
                activation_fn="swiglu",
                mesh_device=mesh_device,
                mesh_axis=tp_axis,
                ccl_manager=ccl_manager,
            )
            self.w_down = RowParallelLinear(
                hidden_dim, dim, bias=False, mesh_device=mesh_device, mesh_axis=tp_axis, ccl_manager=ccl_manager
            )
        elif self.fused:
            self.gate_up = Linear(dim, hidden_dim, bias=False, activation_fn="swiglu", mesh_device=mesh_device)
            self.w_down = Linear(hidden_dim, dim, bias=False, mesh_device=mesh_device)
        else:
            self.w_gate = Linear(dim, hidden_dim, bias=False, mesh_device=mesh_device)
            self.w_up = Linear(dim, hidden_dim, bias=False, mesh_device=mesh_device)
            self.w_down = Linear(hidden_dim, dim, bias=False, mesh_device=mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Pack the shipped ``w_gate``/``w_up`` into the fused ``[up | gate]`` weight.

        ``up`` first: ``Linear`` transposes before handing the packed weight to
        ``prepare_for_fused_swiglu``, whose default ordering is ``[up (N) | gate (N)]``.
        """
        if not self.fused:
            return
        gate = state.pop("w_gate.weight", None)
        up = state.pop("w_up.weight", None)
        if gate is not None and up is not None:
            state["gate_up.weight"] = torch.cat([up, gate], dim=0)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """**Consumes** ``x``.

        Run in row chunks because this is where a block peaks: ``hidden_dim`` is 4x the
        activation, so ``gate``, ``up`` and their product together are 12 activations, which at 6s
        1920x1088 is 30 GiB against 31 GiB of usable DRAM. It is pointwise in the site axis, so
        chunking is exact and needs no halo.
        """
        width = self.hidden_dim // self.mlp_shards
        return _pointwise_in_chunks(x, self._project, width=width)

    @property
    def mlp_shards(self) -> int:
        return int(list(self.gate_up.mesh_device.shape)[self.tp_axis]) if self.tp_mlp else 1

    def _project(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.fused:
            hidden = self.gate_up(x)
            if self.tp_mlp:
                # w_down preserves its input rank, and the all_gather below names dim 3 absolutely,
                # which only holds for a rank-4 tensor. Widen here so the gather stays valid. The
                # reshape aliases its input, so the original must outlive it rather than be consumed.
                hidden = ttnn.reshape(hidden, (1, 1, hidden.shape[-2], hidden.shape[-1]))
            # use_persistent_buffer=False: RowParallelLinear otherwise returns the CCL manager's
            # cached reduce-scatter buffer, which the deallocate below would destroy under it.
            out = self.w_down(hidden, use_persistent_buffer=False) if self.tp_mlp else self.w_down(hidden)
            ttnn.deallocate(hidden)
            if self.tp_mlp:
                # w_down reduce_scatters, so restore the full width the residual add expects.
                gathered = self.ccl_manager.all_gather(out, dim=3, mesh_axis=self.tp_axis, use_hyperparams=False)
                ttnn.deallocate(out)
                out = ttnn.reshape(gathered, (gathered.shape[-2], self.dim))
            return out

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

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        *,
        head_dim: int = 64,
        mesh_device=None,
        na3d_backend: str = "gather",
        ccl_manager=None,
        sp_axis: int | None = None,
        tp_axis: int | None = None,
    ):
        super().__init__()
        # Upstream rounds the 4x MLP ratio up to a multiple of 16.
        hidden = (int(dim * 4.0) + 15) // 16 * 16
        self.norm1 = RMSNorm(dim, norm_eps=1e-6, bias=False, mesh_device=mesh_device)
        self.attn = NeighborhoodAttention(
            dim,
            kernel_size,
            head_dim=head_dim,
            mesh_device=mesh_device,
            na3d_backend=na3d_backend,
            ccl_manager=ccl_manager,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
        )
        self.norm2 = RMSNorm(dim, norm_eps=1e-6, bias=False, mesh_device=mesh_device)
        self.mlp = SwiGLU(dim, hidden, mesh_device=mesh_device, tp_axis=tp_axis, ccl_manager=ccl_manager)
        self.mesh_device = mesh_device  # the deep-profile spans below need it to sync

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        dims: tuple[int, int, int],
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        device_plan: NA3DDevicePlan,
    ) -> ttnn.Tensor:
        # DIFFVAE_BLOCK_PROF=1 only. Each span costs two device syncs, and there are 16 det blocks,
        # so leaving these on by default would inflate the very stage totals they explain. The norms
        # sit inside their span (stage 5 keeps its modulate outside) rather than paying two more.
        if decode_tree.DEEP:
            with block_prof(self.mesh_device, "attention", category=decode_tree.ATTENTION):
                attended = self.attn(self.norm1(x), dims=dims, cos=cos, sin=sin, device_plan=device_plan)
            x = ttnn.add(x, attended)
            ttnn.deallocate(attended)
            with block_prof(self.mesh_device, "mlp", category=decode_tree.MLP):
                projected = self.mlp(self.norm2(x))
            x = ttnn.add(x, projected)
            ttnn.deallocate(projected)
            return x

        attended = self.attn(self.norm1(x), dims=dims, cos=cos, sin=sin, device_plan=device_plan)
        x = ttnn.add(x, attended)
        ttnn.deallocate(attended)
        projected = self.mlp(self.norm2(x))
        x = ttnn.add(x, projected)
        ttnn.deallocate(projected)
        return x


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

    def _shuffle(self, projected: ttnn.Tensor, t: int, h: int, w: int, drop_leading_frame: bool) -> ttnn.Tensor:
        """Pixel-shuffle a ROW_MAJOR ``(t*h*w, proj_out_channels)`` projection into ``(t', h*p2, w*p3,
        c)`` flattened to ``(t'*..., c)`` in TILE. **Consumes** ``projected``; returns the tile tensor
        and the output frame count. Each step allocates a second copy of the widened volume, so each
        is freed the moment it dies.
        """
        p1, p2, p3 = self.stride
        c = self.out_channels
        # (t, h, w, p1, p2, p3, c) -> (t, p1, h, p2, w, p3, c), channels innermost throughout.
        projected = _consume(
            projected, lambda v: ttnn.permute(ttnn.reshape(v, (t, h, w, p1, p2, p3, c)), (0, 3, 1, 4, 2, 5, 6))
        )
        out_t = t * p1
        rows = (h * p2) * (w * p3)
        if p1 == 2 and drop_leading_frame:
            # The temporal shuffle emits a duplicate first frame; dropping it preserves the causal
            # 1:2 (composed 1:8) mapping. Only the slab holding the true t=0 has one.
            projected = _consume(
                projected, lambda v: ttnn.slice(ttnn.reshape(v, (out_t, rows, c)), [1, 0, 0], [out_t, rows, c])
            )
            out_t -= 1
        projected = _consume(projected, lambda v: ttnn.to_layout(ttnn.reshape(v, (out_t * rows, c)), ttnn.TILE_LAYOUT))
        return projected, out_t

    def forward(
        self, x: ttnn.Tensor, *, dims: tuple[int, int, int], drop_leading_frame: bool = True
    ) -> tuple[ttnn.Tensor, tuple[int, int, int]]:
        """``x`` is ``(tokens, in_channels)``; returns ``(tokens', out_channels)`` and new dims.

        The projection widens the channels by the stride span, which at the last 6s 1920x1088
        upsample is a single 10 GiB buffer (plus the permute's copy) -- past what fits, and what
        OOMs a 6s decode. Since the shuffle maps each source frame to its own output frames
        ``[t*p1, (t+1)*p1)`` independently, the volume is processed in source-frame slabs whenever
        the projection would blow :data:`CHUNK_BYTES`, bounding the widened copy to one slab. Slabbing
        needs a frame boundary to be tile-aligned (``h*w`` a multiple of TILE); the large upsamples
        that need it satisfy that, and the small ones run whole.
        """
        t, h, w = dims
        p1, p2, p3 = self.stride
        hw = h * w
        slab = max(1, CHUNK_BYTES // (hw * self.proj_out_channels * 2))

        if slab >= t or hw % TILE != 0:
            projected = _consume(self.proj(x), ttnn.to_layout, ttnn.ROW_MAJOR_LAYOUT)
            out, out_t = self._shuffle(projected, t, h, w, drop_leading_frame)
            return out, (out_t, h * p2, w * p3)

        in_channels = int(x.shape[-1])
        parts: list[ttnn.Tensor] = []
        out_t_total = 0
        for start in range(0, t, slab):
            st = min(start + slab, t) - start
            x_slab = ttnn.slice(x, [start * hw, 0], [(start + st) * hw, in_channels])
            projected = _consume(self.proj(x_slab), ttnn.to_layout, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(x_slab)
            part, part_t = self._shuffle(projected, st, h, w, drop_leading_frame and start == 0)
            parts.append(part)
            out_t_total += part_t
        joined = ttnn.concat(parts, dim=-2)
        for part in parts:
            ttnn.deallocate(part)
        return joined, (out_t_total, h * p2, w * p3)


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
        ccl_manager=None,
        na3d_backend: str | None = None,
        sp_axis: int | None = None,
        tp_axis: int | None = None,
    ):
        super().__init__()
        assert len(upsamples) == len(stage_channels) - 1, "one upsample between consecutive stages"
        self.stage_kernels = stage_kernels
        self.head_dim = head_dim
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        # Spatial-W SP for the deterministic stages: when the backend is "op_sp_w_sharded" the
        # activation is W-sharded from stage 1 on (stage 0's W is not divisible by the mesh axis, so
        # it stays replicated), then gathered back to a replicated context at the end -- so the
        # handoff to stage 5 is unchanged. Defaults to the env override for the OOM diagnostic.
        #
        # DET_ in the name because this reaches stages 1-4 ONLY, alongside the other DIFFVAE_DET_*
        # knobs. Stage 5 is selected separately by DIFFVAE_STAGE5_BACKEND; the two used to share the
        # name DIFFVAE_NA3D_BACKEND, so setting it moved both halves of the decode at once.
        self.na3d_backend = na3d_backend or os.environ.get("DIFFVAE_DET_NA3D_BACKEND", "gather")
        self.sp_axis = sp_axis
        # TP-over-heads on the orthogonal axis for the W-sharded stages (2-D SP x TP): the det stages
        # otherwise run replicated over this axis. Only applied to the op_sp_w_sharded stages (1+).
        self.tp_axis = tp_axis
        self._w_sharded = self.na3d_backend == "op_sp_w_sharded"
        self.sp = int(list(mesh_device.shape)[sp_axis]) if self._w_sharded else 1
        if self._w_sharded:
            assert sp_axis is not None and ccl_manager is not None, "op_sp_w_sharded needs sp_axis + ccl_manager"
        self.conv_in = Linear(in_channels, stage_channels[0], bias=True, mesh_device=mesh_device)

        def block_backend(stage: int) -> str:
            # Stage 0 runs replicated (its W is not shardable), but on the fast gather backend (its
            # plan query-shards across the mesh) -- NOT op, which is ~5x slower and would swamp the
            # W-shard win on stages 1+. The rest shard over W.
            if self._w_sharded:
                return "gather" if stage == 0 else "op_sp_w_sharded"
            return self.na3d_backend

        self.det_stages = ModuleList(
            [
                ModuleList(
                    [
                        NABlock(
                            stage_channels[stage],
                            stage_kernels[stage],
                            head_dim=head_dim,
                            mesh_device=mesh_device,
                            na3d_backend=block_backend(stage),
                            ccl_manager=ccl_manager if self._w_sharded and stage > 0 else None,
                            sp_axis=sp_axis if self._w_sharded and stage > 0 else None,
                            tp_axis=tp_axis if self._w_sharded and stage > 0 else None,
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

    def _plan(self, dims: tuple[int, int, int], kernel: tuple[int, int, int]) -> NA3DDevicePlan:
        key = (dims, kernel)
        if key not in self._plan_cache:
            self._plan_cache[key] = build_device_plan(
                plan_na3d(dims, kernel), mesh_device=self.mesh_device, ccl_manager=self.ccl_manager
            )
        return self._plan_cache[key]

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

    def _wshard(self, x: ttnn.Tensor, dims: tuple[int, int, int]) -> ttnn.Tensor:
        """Reshard a replicated ``(T*H*W, ch)`` volume into this chip's W-band ``(T*H*(W/sp), ch)``.

        A W-band is not a contiguous slice of the T-outer sequence, so the flat rows are viewed as a
        ``(T, H, W, ch)`` volume, ``mesh_partition``ed on W, and flattened back in (t, h, w_local)
        order. **Consumes** ``x``.
        """
        t, h, w = dims
        ch = int(x.shape[-1])
        rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(x)
        vol = ttnn.reshape(rm, (t, h, w, ch))
        band = ttnn.mesh_partition(vol, dim=2, cluster_axis=self.sp_axis)  # (T, H, W/sp, ch)
        ttnn.deallocate(rm)
        flat = ttnn.reshape(band, (t * h * (w // self.sp), ch))
        return ttnn.to_layout(flat, ttnn.TILE_LAYOUT)

    def _wgather(self, x: ttnn.Tensor, dims: tuple[int, int, int]) -> ttnn.Tensor:
        """Gather a W-sharded ``(T*H*(W/sp), ch)`` band back to the replicated ``(T*H*W, ch)`` volume.

        Chip order along ``sp_axis`` is W order, so the all-gather over the W dim rebuilds the full
        volume with no reshuffle. **Consumes** ``x``.
        """
        t, h, w = dims
        ch = int(x.shape[-1])
        w_local = w // self.sp
        vol = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (t, h, w_local, ch))
        ttnn.deallocate(x)
        full = self.ccl_manager.all_gather(vol, dim=2, mesh_axis=self.sp_axis, use_hyperparams=False)  # (T, H, W, ch)
        return ttnn.to_layout(ttnn.reshape(full, (t * h * w, ch)), ttnn.TILE_LAYOUT)

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        dims: tuple[int, int, int],
        drop_leading_frame: bool = True,
        stages: int | None = None,
        gather_output: bool = True,
    ) -> tuple[ttnn.Tensor, tuple[int, int, int]]:
        """``x`` is ``(tokens, in_channels)`` channels-last in TILE layout, normalized latent.

        Under spatial-W SP the activation is W-sharded from stage 1 on and gathered back to a
        replicated volume before returning, so ``dims`` here is always the FULL ``(T, H, W)``.

        ``gather_output=False`` skips that final all-gather and returns this chip's W-band
        ``(T*H*(W/sp), ch)`` instead -- for the W-sharded det->stage-5 handoff, where stage 5 consumes
        the band directly (same ``sp_axis``, same W order) rather than re-sharding a replicated context.
        ``dims`` is still the FULL ``(T, H, W)``; the caller derives ``W/sp`` from ``self.sp``.
        """
        with stage_timer(self.mesh_device, "conv_in (denorm folded)", category=decode_tree.MLP):
            x = self.conv_in(x)
        count = len(self.upsamples) if stages is None else stages
        sharded = False
        for stage in range(count):
            # Labelled with the dims going IN: the out-dims do not exist until the upsample below
            # runs, and a span names itself at open so a leaked one is still identifiable.
            with stage_timer(self.mesh_device, f"det stage {stage} (in {dims[0]},{dims[1]},{dims[2]})"):
                t, h, w = dims
                stage_sharded = self._w_sharded and stage > 0
                if stage_sharded:
                    assert w % self.sp == 0, f"stage {stage} W={w} not divisible by sp={self.sp}"
                    if not sharded:
                        with stage_timer(
                            self.mesh_device, "reshard: replicated -> W-sharded", category=decode_tree.RESHAPE
                        ):
                            x = self._wshard(x, dims)  # replicated -> W-sharded at the stage-0 -> 1 boundary
                        sharded = True
                local_dims = (t, h, w // self.sp) if stage_sharded else dims

                # Tables and plan are per-stage setup, not block work; timed apart so a stage's number is
                # its blocks rather than its blocks plus whatever it had to build first.
                with stage_timer(
                    self.mesh_device, f"stage {stage + 1} setup: rope tables + plan", category=decode_tree.SETUP
                ):
                    cos, sin = self._rope(dims)
                    if stage_sharded:
                        cos = ttnn.mesh_partition(cos, dim=3, cluster_axis=self.sp_axis)
                        sin = ttnn.mesh_partition(sin, dim=3, cluster_axis=self.sp_axis)
                    plan = None if stage_sharded else self._plan(dims, self.stage_kernels[stage])

                # dim read here rather than after the loop: NABlock is residual, so its channel count
                # is unchanged by the blocks.
                with stage_timer(
                    self.mesh_device, f"STAGE {stage + 1}: {len(self.det_stages[stage])}x NABlock dim {x.shape[-1]}"
                ):
                    for index, block in enumerate(self.det_stages[stage]):
                        x = block(x, dims=local_dims, cos=cos, sin=sin, device_plan=plan)
                        log_dram(
                            self.mesh_device,
                            f"det stage {stage} block {index} dims={local_dims} sharded={stage_sharded}",
                        )

                with stage_timer(self.mesh_device, f"upsample {stage + 1}", category=decode_tree.UPSAMPLE):
                    x, out_dims = self.upsamples[stage](x, dims=local_dims, drop_leading_frame=drop_leading_frame)
                if stage_sharded:
                    out_dims = (out_dims[0], out_dims[1], out_dims[2] * self.sp)  # local W -> full W
                dims = out_dims
                log_dram(self.mesh_device, f"det stage {stage} upsampled to {dims} sharded={stage_sharded}")

        if sharded and gather_output:
            with stage_timer(self.mesh_device, "det -> replicated context gather", category=decode_tree.ALLGATHER):
                x = self._wgather(x, dims)  # W-sharded -> replicated context; stage-5 handoff unchanged
            log_dram(self.mesh_device, f"det gathered to replicated {dims}")
        return x, dims


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

    #: The conv decoder can hand back YUV 4:2:0 straight off the mesh; this one returns host
    #: pixels, so the pipeline keeps the float export path when this decoder is installed.
    supports_yuv = True

    #: Unlike the conv decoder, this one cannot share the mesh with the transformer: its stage-5
    #: activations are ~8.5 GB at 1920x1088, against the ~2.6 GB a resident 22B DiT leaves free.
    #: The pipeline therefore evicts the DiT before decode even under static loading.
    requires_exclusive_residency = True

    def __init__(
        self,
        config: dict,
        *,
        mesh_device,
        dtype: ttnn.DataType = ttnn.bfloat16,
        ccl_manager=None,
        stage5_na3d_backend: str | None = None,
        stage5_sp_axis: int | None = None,
        stage5_tp_axis: int | None = None,
        stages_na3d_backend: str | None = None,
        stages_sp_axis: int | None = None,
        stages_tp_axis: int | None = None,
    ):
        super().__init__()
        from .diffvae_ltx_stage5 import DiffVAEStage5, DiffVAEStage5Config

        # The ~8.5 GB that makes this decoder demand the mesh to itself is the REPLICATED figure:
        # 78x272x480 sites x 256 channels x 2 B is ~5.2 GB for ONE stage-5 activation with every
        # chip holding the whole volume. W-sharded over sp (and TP over heads) each chip holds
        # about an eighth of that, which may well fit beside a resident 22B DiT.
        #
        # This matters beyond memory: eviction is what invalidates the transformer's CAPTURED
        # TRACE, which is why traced + DiffVAE is refused outright under static loading. A sharded
        # decoder that does not need the mesh to itself never evicts, so both can stay resident and
        # traced -- the shape a serving loop wants. DIFFVAE_EXCLUSIVE=1 forces the old behaviour.
        if stage5_sp_axis is not None:
            self.requires_exclusive_residency = os.environ.get("DIFFVAE_EXCLUSIVE") == "1"

        self.config = config
        self.mesh_device = mesh_device
        self._timestep = None
        # Without a manager every chip decodes the whole volume, which is correct but costs the
        # mesh: memory per chip then scales with frame count rather than with frames/mesh size.
        self.ccl_manager = ccl_manager
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
            ccl_manager=ccl_manager,
            na3d_backend=stages_na3d_backend,
            sp_axis=stages_sp_axis,
            tp_axis=stages_tp_axis,
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
            ccl_manager=ccl_manager,
            na3d_backend=stage5_na3d_backend,
            sp_axis=stage5_sp_axis,
            tp_axis=stage5_tp_axis,
        )
        # W-sharded det->stage-5 handoff: when BOTH halves W-shard on the same axis, the det stages
        # hand their context over W-sharded instead of all-gathering it to a replicated volume that
        # stage 5 would immediately re-shard. Both the det _wgather and stage-5 _wshard_context go away.
        self._wsharded_handoff = (
            self.stages._w_sharded and self.stage5._w_sharded and self.stages.sp_axis == self.stage5.sp_axis
        )
        self.dtype = dtype

    def torch_state_from_checkpoint(self, path, *, statistics: bool = True) -> dict[str, torch.Tensor]:
        """One state dict for the whole decoder, keyed for :meth:`load_torch_state_dict`.

        Kept separate from applying it so the weights can go through the tt_dit disk cache, which
        wants a provider it can skip calling on a cache hit. Both halves come from the same file
        but need different remapping (folded statistics, permuted upsample projections), so each
        half maps its own keys and they are prefixed by attribute name here.
        """
        from safetensors import safe_open

        state = {f"stages.{k}": v for k, v in self.stages.state_from_checkpoint(path, statistics=statistics).items()}

        prefixes = ("diff_blocks.", "shared_adaln.", "t_embedder.", "conv_in_x_t.", "conv_out.", "norm_out.")
        with safe_open(str(path), "pt") as handle:
            for key in handle.keys():
                if not key.startswith("decoder."):
                    continue
                name = key[len("decoder.") :]
                if name.startswith(prefixes):
                    state[f"stage5.{name}"] = handle.get_tensor(key).float()
        return state

    def load_checkpoint(self, path, *, statistics: bool = True) -> None:
        """Load both halves from one LTX-2.5 video-VAE safetensors file, bypassing the cache."""
        self.load_torch_state_dict(self.torch_state_from_checkpoint(path, statistics=statistics))

    def context_frames(self, latent_frames: int) -> int:
        """Stage-5 temporal extent for a latent of ``latent_frames``, after pad and crop."""
        padded = latent_frames + self.ghost_latent_frames
        # Each temporal upsample doubles then drops its duplicate leading frame, so the
        # composed map is causal: n -> time_scale * (n - 1) + 1.
        grown = self.time_scale * (padded - 1) + 1
        return max(grown - self.ghost_latent_frames * self.time_scale, self.stage5_kernel[0])

    def forward_context(
        self, latent: torch.Tensor, *, gather_output: bool = True, latent_tt: ttnn.Tensor | None = None
    ) -> tuple[ttnn.Tensor, tuple[int, int, int]]:
        """Deterministic stages on a ``(B, C, T, H, W)`` normalized latent, ghost cropped.

        ``gather_output=False`` returns the context W-sharded (this chip's band), for the W-sharded
        det->stage-5 handoff. ``dims`` is always the FULL ``(T, H, W)``; the ghost crop is on ``T``,
        which is orthogonal to the W-shard, so it just uses ``W/sp`` columns per chip when sharded.

        ``latent_tt`` supplies the raw latent already on device, in the ROW_MAJOR
        ``(1, C, T, H*W)`` form the device-preproc path uploads. It exists for tracing: a trace
        refuses host-to-device writes during capture, so the upload has to happen outside the
        captured region and the buffer be handed in. ``latent`` is then read for its shape only,
        and the caller keeps ownership of the buffer.
        """
        batch, channels, t, h, w = latent.shape
        assert batch == 1, f"batched decode is not implemented; got batch={batch}"
        assert channels == self.in_channels, f"latent has {channels} channels, expected {self.in_channels}"

        ghost = self.ghost_latent_frames
        if os.environ.get("DIFFVAE_DEVICE_PREPROC") == "1":
            # Upload the latent as-is and do the ghost pad and channels-last flatten on device, so the
            # only host step left before the pipeline is the transfer itself. The pad replicates the
            # last frame, which is a slice plus a concat on the T axis.
            with stage_timer(self.mesh_device, "host->mesh: upload latent (raw)", category=decode_tree.HOST_XFER):
                raw = latent_tt
                if raw is None:
                    raw = ttnn.from_torch(
                        latent.reshape(1, channels, t, h * w).contiguous(),
                        device=self.mesh_device,
                        dtype=self.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )

            with stage_timer(self.mesh_device, "device: ghost pad + flatten", category=decode_tree.RESHAPE):
                last = ttnn.slice(raw, [0, 0, t - 1, 0], [1, channels, t, h * w])
                parts = [raw] + [last] * ghost
                padded_tt = ttnn.concat(parts, dim=2)
                if latent_tt is None:
                    ttnn.deallocate(raw)
                ttnn.deallocate(last)
                moved = ttnn.permute(padded_tt, (0, 2, 3, 1))  # (1, T+ghost, H*W, C)
                ttnn.deallocate(padded_tt)
                x = ttnn.to_layout(ttnn.reshape(moved, ((t + ghost) * h * w, channels)), ttnn.TILE_LAYOUT)
                ttnn.deallocate(moved)
        else:
            with stage_timer(self.mesh_device, "host: ghost pad + permute/flatten", category=decode_tree.HOST_COMPUTE):
                padded = torch.cat([latent, latent[:, :, -1:].expand(-1, -1, ghost, -1, -1)], dim=2)
                tokens = padded.permute(0, 2, 3, 4, 1).reshape(-1, channels).contiguous()
            with stage_timer(self.mesh_device, "host->mesh: upload TILE", category=decode_tree.HOST_XFER):
                x = ttnn.from_torch(tokens, device=self.mesh_device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)

        x, dims = self.stages(x, dims=(t + ghost, h, w), gather_output=gather_output)
        sharded_out = self.stages._w_sharded and not gather_output
        w_eff = dims[2] // self.stages.sp if sharded_out else dims[2]  # local W columns per chip
        keep = self.context_frames(t)
        with stage_timer(self.mesh_device, "ghost crop on T", category=decode_tree.RESHAPE):
            if keep < dims[0]:
                channels_out = self.config["stage_channels"][-1]
                # Each step here allocates a full copy of the uncropped volume, which at 1920x1088 is
                # 2.7 GB against 31.8 GB of DRAM, so they are freed as they die rather than at GC. The
                # crop is on T (frame axis), which the W-shard leaves untouched -- just use w_eff columns.
                frames = ttnn.reshape(
                    ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (dims[0], dims[1] * w_eff, channels_out)
                )
                ttnn.deallocate(x)
                cropped = ttnn.slice(frames, [0, 0, 0], [keep, dims[1] * w_eff, channels_out])
                ttnn.deallocate(frames)
                x = ttnn.to_layout(ttnn.reshape(cropped, (keep * dims[1] * w_eff, channels_out)), ttnn.TILE_LAYOUT)
                ttnn.deallocate(cropped)
                dims = (keep, dims[1], dims[2])
        return x, dims

    def decode(
        self,
        latent: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        seed: int = 0,
        latent_tt: ttnn.Tensor | None = None,
        device_out: bool = False,
        output_type: str = "float",
    ) -> torch.Tensor | ttnn.Tensor:
        """Normalized ``(B, C, T, H, W)`` latent to ``(B, 3, T', H', W')`` pixels.

        ``noise`` is an input, not an implementation detail: stage 5 predicts x0 from it in a
        single step. Pass it to compare against a reference that drew its own.

        ``latent_tt`` and ``device_out`` move the two host boundaries out of the way so the whole
        decode can be captured as one trace: the upload happens before the region and the PCIe pull
        after it. See :meth:`forward_context` and :meth:`DiffVAEStage5.forward`.
        """
        from .diffvae_ltx_stage5 import Grid

        # The whole decode under one node: the tree hangs off this, and its total is an honesty
        # check against the caller's own wall-clock measurement of the same call.
        with stage_timer(self.mesh_device, "decode TOTAL", root=True):
            with stage_timer(self.mesh_device, "det stages TOTAL (forward_context)"):
                context, dims = self.forward_context(
                    latent, gather_output=not self._wsharded_handoff, latent_tt=latent_tt
                )
            grid = Grid(batch=1, t=dims[0], h=dims[1], w=dims[2])
            channels_out = self.config["stage_channels"][-1]
            # W-sharded handoff: context is this chip's band; reshape to the local site count stage 5's
            # W-sharded path expects and skip its re-shard (the det->stage-5 all-gather + re-shard both go).
            with stage_timer(self.mesh_device, "context reshape for stage 5", category=decode_tree.RESHAPE):
                if self._wsharded_handoff:
                    w_local = grid.w // self.stages.sp
                    context = ttnn.reshape(context, (1, 1, grid.t * grid.h * w_local, channels_out))
                else:
                    context = ttnn.reshape(context, (1, 1, grid.sites, channels_out))

            # DIFFVAE_DEVICE_NOISE=1 leaves noise as None so stage 5 draws it on device in the patchified
            # layout. Host generation is proportional to the OUTPUT volume, not the latent -- 908M floats at
            # 1080p 6s -- and a caller supplying its own noise pays neither path.
            if noise is None and os.environ.get("DIFFVAE_DEVICE_NOISE") != "1":
                shape = (1, self.out_channels, grid.t, grid.h * self.patch_size, grid.w * self.patch_size)
                with stage_timer(
                    self.mesh_device, f"host: noise randn {tuple(shape)}", category=decode_tree.HOST_COMPUTE
                ):
                    noise = torch.randn(shape, generator=torch.Generator().manual_seed(seed))

            # default_num_inference_steps is 1 on this checkpoint, so linspace(1, 1, 1) = [1.0]. Uploaded
            # once and kept: a constant on the per-decode path is still a host-to-device write, which a
            # trace refuses during capture.
            if self._timestep is None:
                self._timestep = ttnn.from_torch(
                    torch.tensor([[[[1.0]]]]), device=self.mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
                )
            timestep = self._timestep
            with stage_timer(self.mesh_device, "stage5 TOTAL (forward)"):
                pixels = self.stage5.forward(
                    context,
                    noise,
                    timestep,
                    grid,
                    context_sharded=self._wsharded_handoff,
                    seed=seed,
                    device_out=device_out,
                    output_type=output_type,
                )
            return pixels

    def forward(
        self,
        latent: torch.Tensor,
        *,
        output_type: str = "float",
        noise: torch.Tensor | None = None,
        seed: int = 0,
    ) -> torch.Tensor:
        """``decode`` behind the pipeline's decoder signature, so this can stand in for the
        conv decoder in ``decode_latents``.

        ``output_type`` matches ``LTXVideoDecoder.forward``: ``float`` keeps ``[-1, 1]``, ``rgb``
        maps it to planar uint8 the same way ``utils.tensor.float_to_uint8`` does on device.
        ``yuv`` converts and gathers YUV 4:2:0 on device, which needs
        ``DIFFVAE_DEVICE_UNPATCHIFY=1`` so the pixels exist on device in the first place.
        """
        if output_type == "yuv":
            return self.decode(latent, noise=noise, seed=seed, output_type="yuv")
        pixels = self.decode(latent, noise=noise, seed=seed)
        if output_type == "float":
            return pixels
        if output_type == "rgb":
            return pixels.add(1.0).mul(0.5 * 255.0).clamp(0.0, 255.0).to(torch.uint8)
        raise ValueError(f"unknown output_type {output_type!r}")

    def release_trace(self) -> None:
        """No-op: this decoder is not traced yet, but the pipeline releases traces blindly."""
