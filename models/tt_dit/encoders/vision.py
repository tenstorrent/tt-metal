# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Building blocks shared by the Qwen-VL vision towers (Qwen2.5-VL and Qwen3-VL).

Both towers embed flattened patches with a `Linear`, rotate q/k with a 2-D `(row, col)` rotary
embedding, and run plain MHA with biases over blocks named by `cu_seqlens`. They differ in their
norms, MLPs, mergers and position handling, which live in the per-model modules.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import torch

import ttnn
from models.tt_dit.layers.linear import ColParallelLinear, Linear, RowParallelLinear
from models.tt_dit.layers.module import Module
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.mochi import get_rot_transformation_mat
from models.tt_dit.utils.tensor import bf16_tensor, typed_tensor

# `ttnn` SDPA requires a tile-aligned head dimension.
_TILE = 32


class VisionParallel(NamedTuple):
    """Resolved tensor- and sequence-parallel placement for the tower.

    Separate from `EncoderParallelConfig` because that carries mesh axes while every module here needs
    the *factors* too, and because the replicated case (both factors 1) must stay a plain `Linear`
    path: gating on the factor keeps an unparallelized run bit-identical to the single-device module.
    """

    tp_axis: int | None = None
    tp_factor: int = 1
    sp_axis: int | None = None
    sp_factor: int = 1
    ccl_manager: CCLManager | None = None

    @property
    def tp(self) -> bool:
        return self.tp_factor > 1

    @property
    def sp(self) -> bool:
        return self.sp_factor > 1


def resolve_vision_parallel(
    mesh_device: ttnn.MeshDevice,
    parallel_config: EncoderParallelConfig | None,
    ccl_manager: CCLManager | None,
) -> VisionParallel:
    """`VisionParallel` from the pipeline's `EncoderParallelConfig`.

    `sequence_parallel` is optional on that config; the tower is its only encoder-side consumer. A
    factor above 1 on either axis requires a `ccl_manager`, since both paths reduce or gather across
    devices.
    """
    if parallel_config is None:
        return VisionParallel(ccl_manager=ccl_manager)

    tp, sp = parallel_config.tensor_parallel, parallel_config.sequence_parallel
    resolved = VisionParallel(
        tp_axis=tp.mesh_axis if tp is not None else None,
        tp_factor=tp.factor if tp is not None else 1,
        sp_axis=sp.mesh_axis if sp is not None else None,
        sp_factor=sp.factor if sp is not None else 1,
        ccl_manager=ccl_manager,
    )
    if (resolved.tp or resolved.sp) and ccl_manager is None:
        msg = "ccl_manager is required when tensor- or sequence-parallel factor is greater than 1"
        raise ValueError(msg)
    if resolved.tp and resolved.sp and resolved.tp_axis == resolved.sp_axis:
        msg = f"tensor and sequence parallelism cannot share mesh axis {resolved.tp_axis}"
        raise ValueError(msg)
    for name, axis, factor in (
        ("tensor", resolved.tp_axis, resolved.tp_factor),
        ("sequence", resolved.sp_axis, resolved.sp_factor),
    ):
        if factor > 1 and tuple(mesh_device.shape)[axis] != factor:
            msg = (
                f"{name}-parallel factor {factor} does not match mesh axis {axis} "
                f"of shape {tuple(mesh_device.shape)}"
            )
            raise ValueError(msg)
    return resolved


class VisionPatchEmbed(Module):
    """Patch embedding of the vision tower.

    The reference is a `Conv3d` whose kernel equals its stride over patches that the processor has
    already flattened, which makes it a plain matmul: each `(in_channels, temporal_patch_size,
    patch_size, patch_size)` patch is one independent row. Verified equal to the reference `Conv3d` to
    3.8e-6 in fp32 (accumulation order only), which is far below bf16 resolution. So this is a
    `Linear` and `_prepare_torch_state` flattens the convolution weight into it.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        hidden_size: int = 1152,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        bias: bool = True,
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()
        self.patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
        self.hidden_size = hidden_size
        self.proj = Linear(self.patch_dim, hidden_size, bias=bias, mesh_device=mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # `proj.weight` arrives as the Conv3d kernel (hidden, in_ch, t, p, p); flatten the spatial and
        # channel axes so it multiplies the already-flattened patch rows.
        weight = state.get("proj.weight")
        if weight is not None and weight.ndim > 2:
            state["proj.weight"] = weight.reshape(self.hidden_size, self.patch_dim)

    def forward(self, patches: ttnn.Tensor) -> ttnn.Tensor:
        """`(total_patches, patch_dim)` -> `(total_patches, hidden_size)`."""
        return self.proj.forward(patches)


def vision_rope_position_ids(grid_thw: torch.Tensor, *, spatial_merge_size: int) -> torch.Tensor:
    """`(total_patches, 2)` `(row, col)` positions for the tower's 2-D rotary embedding.

    Note this is a *different* grid from the decoder's 3-axis M-RoPE: the tower rotates over spatial
    position within one image only, and has no temporal axis -- frames of a video repeat the same
    `(row, col)` grid, since each frame is its own attention block.

    The reshape/transpose puts positions into `spatial_merge_size` blocks, matching the patch order the
    merger consumes.
    """
    out = []
    for t, h, w in grid_thw.int().tolist():
        m = spatial_merge_size
        hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos = hpos.reshape(h // m, m, w // m, m).transpose(1, 2).flatten()
        wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos = wpos.reshape(h // m, m, w // m, m).transpose(1, 2).flatten()
        out.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(out, dim=0)


def vision_cu_seqlens(grid_thw: torch.Tensor) -> tuple[int, ...]:
    """The attention block boundaries of a batch of images and videos.

    `cu_seqlens = repeat_interleave(h * w, t).cumsum()` padded with a leading 0, matching the
    reference. One image is one block; a video is one block *per frame*, since `t` repeats its spatial
    extent. Attention never crosses a boundary, so a single image collapses to plain full attention.
    """
    bounds = [0]
    for t, h, w in grid_thw.tolist():
        for _ in range(int(t)):
            bounds.append(bounds[-1] + int(h) * int(w))
    return tuple(bounds)


def vision_rope_tensors(
    grid_thw: torch.Tensor,
    *,
    head_dim: int,
    spatial_merge_size: int,
    rope_theta: float = 10000.0,
    padded_head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(cos, sin)` of shape `(total_patches, padded_head_dim)` for the tower's rotary embedding.

    Frequencies are **interleaved** for `rotary_embedding_llama` (matching `rope_permute_qk`) and padded
    with an identity tail (cos=1, sin=0). Built on the host and uploaded.
    """
    padded_head_dim = padded_head_dim or math.ceil(head_dim / _TILE) * _TILE
    position_ids = vision_rope_position_ids(grid_thw, spatial_merge_size=spatial_merge_size)
    dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    freqs = (position_ids.unsqueeze(-1) * inv_freq).flatten(1)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)
    cos, sin = emb.cos(), emb.sin()
    pad = padded_head_dim - head_dim
    if pad:
        cos = torch.nn.functional.pad(cos, (0, pad), value=1.0)
        sin = torch.nn.functional.pad(sin, (0, pad), value=0.0)
    return cos, sin


def pad_head_dim(weight: torch.Tensor, *, num_heads: int, head_dim: int, padded: int, axis: int) -> torch.Tensor:
    """Zero-pad a per-head weight or bias from `head_dim` to `padded` along `axis`.

    `axis=0` for the output rows of q/k/v and for biases, `axis=1` for the input columns of the output
    projection. The zeros contribute nothing to the dot products, which is why the softmax `scale` must
    still be computed from the *true* head_dim -- see [`VisionAttention`].
    """
    if head_dim == padded:
        return weight
    if axis == 0:
        shaped = weight.reshape(num_heads, head_dim, *weight.shape[1:])
        pad = [0] * (2 * (shaped.ndim - 2)) + [0, padded - head_dim]
        return torch.nn.functional.pad(shaped, pad).reshape(num_heads * padded, *weight.shape[1:])
    shaped = weight.reshape(weight.shape[0], num_heads, head_dim)
    return torch.nn.functional.pad(shaped, (0, padded - head_dim)).reshape(weight.shape[0], num_heads * padded)


def rope_permute_qk(t: torch.Tensor, *, num_heads: int, head_dim: int) -> torch.Tensor:
    """Reorder each head's output channels SPLIT-rotation -> INTERLEAVED for `rotary_embedding_llama`.

    Applied to q/k only, before `pad_head_dim`.
    """
    half = head_dim // 2
    perm = torch.empty(head_dim, dtype=torch.long)
    perm[0::2] = torch.arange(half)
    perm[1::2] = torch.arange(half, head_dim)
    rest = t.shape[1:]
    return t.reshape(num_heads, head_dim, *rest).index_select(1, perm).reshape(num_heads * head_dim, *rest)


def with_batch_axis(x: ttnn.Tensor) -> tuple[ttnn.Tensor, bool]:
    """`(x, added)` with a leading batch axis, because the CCL paths require rank >= 3.

    The towers carry 2-D `(rows, width)` activations end to end, unlike the decoder's
    `(batch, seq, hidden)`. `RowParallelLinear` reduce-scatters on `dim=3` after a SINGLE unsqueeze, so
    a 2-D input leaves it indexing past the end (`IndexError` in `get_rs_ping_pong_buffer`). Adding the
    axis here is local to the towers rather than a change to shared `layers/linear.py`.
    """
    if len(x.shape) >= 3:
        return x, False
    return ttnn.reshape(x, (1, x.shape[0], x.shape[1])), True


def drop_batch_axis(x: ttnn.Tensor, *, added: bool) -> ttnn.Tensor:
    return ttnn.reshape(x, (x.shape[-2], x.shape[-1])) if added else x


def row_parallel_forward(linear: Linear | RowParallelLinear, x: ttnn.Tensor, p: VisionParallel) -> ttnn.Tensor:
    """Run a row-parallel linear on a 2-D activation and gather its fractured result to full width.

    Reduce-scatters on the hidden dim; used by the mergers (see `row_parallel_seq_forward`).
    """
    if not p.tp:
        return linear.forward(x)
    x, added = with_batch_axis(x)
    out = p.ccl_manager.all_gather(linear.forward(x), dim=-1, mesh_axis=p.tp_axis, use_hyperparams=True)
    return drop_batch_axis(out, added=added)


def row_parallel_seq_forward(linear: Linear | RowParallelLinear, x: ttnn.Tensor, p: VisionParallel) -> ttnn.Tensor:
    """Row-parallel linear whose all-reduce is split on the SEQUENCE dim, not the hidden dim.

    Same result as `row_parallel_forward`; rows are zero-padded internally for the collective only.
    """
    if not p.tp:
        return linear.forward(x)
    x, added = with_batch_axis(x)
    rows = x.shape[1]
    npad = (-rows) % (p.tp_factor * _TILE)
    if npad:
        x = ttnn.pad(x, [(0, 0), (0, npad), (0, 0)], value=0.0)
    out = linear.forward(x, reduce_scatter_dim=-2)
    out = p.ccl_manager.all_gather(out, dim=1, mesh_axis=p.tp_axis, use_hyperparams=True)
    if npad:
        out = out[:, :rows, :]
    return drop_batch_axis(out, added=added)


def interleave_for_col_parallel(t: torch.Tensor, *, parts: int, tp_factor: int) -> torch.Tensor:
    """Reorder a `parts`-way concatenated weight/bias so column fracturing keeps each part per device.

    `qkv` is stored as `[q | k | v]` over the row axis. `ColParallelLinear` fractures `out_features`
    into `tp_factor` CONTIGUOUS chunks, which would hand device 0 the first eighth of `q` alone and no
    `k` or `v` at all. Grouping by device first -- `[q0 k0 v0][q1 k1 v1]...` -- makes the contiguous
    chunk the correct `(q, k, v)` triple for that device's heads.

    Operates on the torch layout (rows = out_features); `ColParallelLinear._prepare_torch_state`
    transposes afterwards.
    """
    if tp_factor == 1:
        return t
    rows = t.shape[0]
    if rows % (parts * tp_factor) != 0:
        msg = f"cannot interleave {rows} rows into {parts} parts across {tp_factor} devices"
        raise ValueError(msg)
    trailing = t.shape[1:]
    return t.reshape(parts, tp_factor, rows // (parts * tp_factor), *trailing).transpose(0, 1).reshape(rows, *trailing)


class VisionAttention(Module):
    """Plain MHA over one attention block, with biases and no QK-norm.

    `head_dim` is `hidden_size // num_heads`, which is not tile-aligned for either tower (72 for
    Qwen3-VL, 80 for Qwen2.5-VL). `ttnn` SDPA rejects it outright (`TT_FATAL logical_shape[3] ==
    legacy_shape[3]`), so q/k/v/o are padded to 96 at load time and `scale` is passed explicitly from
    the true head_dim. Leaving `scale` to SDPA would make it `96 ** -0.5` and silently change the
    softmax temperature -- wrong output, not a crash.

    Attention is confined to the blocks `cu_seqlens` names (see [`vision_cu_seqlens`]): one image, one
    frame of a video, or one window of Qwen2.5-VL's windowed layers. A single block reduces to plain
    full attention. Multiple blocks are handled by SDPA's windowed mode, which builds the
    block-diagonal mask on the device rather than materializing an `s x s` mask -- 17 GiB for a full
    MiniMax-H3 `ref2va` request of nine images and three videos.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        mesh_device: ttnn.MeshDevice,
        parallel: VisionParallel | None = None,
        linear_compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
    ) -> None:
        super().__init__()
        self._p = parallel or VisionParallel()
        self.mesh_device = mesh_device
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.padded_head_dim = math.ceil(self.head_dim / _TILE) * _TILE
        self.scale = self.head_dim**-0.5

        self.inner = num_heads * self.padded_head_dim
        # TP shards HEADS (not head_dim): each device owns `num_heads // tp_factor` whole heads, so
        # SDPA needs no cross-device traffic for TP and the rotary tensors stay unsharded.
        if num_heads % self._p.tp_factor != 0:
            msg = f"num_heads {num_heads} is not divisible by TP factor {self._p.tp_factor}"
            raise ValueError(msg)
        self.num_local_heads = num_heads // self._p.tp_factor
        self.local_inner = self.num_local_heads * self.padded_head_dim

        # Ring SDPA runs its CCL workers on cores the compute kernel must not also own, so the last
        # column is reserved for them and `ccl_core_grid_offset` points at it (as in attention_wan.py).
        full_grid = mesh_device.compute_with_storage_grid_size()
        self._sdpa_worker_grid = (full_grid.x - 1, full_grid.y)

        if self._p.tp:
            kw = dict(
                mesh_device=mesh_device,
                mesh_axis=self._p.tp_axis,
                ccl_manager=self._p.ccl_manager,
                compute_kernel_config=linear_compute_kernel_config,
            )
            self.qkv = ColParallelLinear(hidden_size, 3 * self.inner, bias=True, **kw)
            self.proj = RowParallelLinear(self.inner, hidden_size, bias=True, **kw)
        else:
            kw = dict(mesh_device=mesh_device, compute_kernel_config=linear_compute_kernel_config)
            self.qkv = Linear(hidden_size, 3 * self.inner, bias=True, **kw)
            self.proj = Linear(self.inner, hidden_size, bias=True, **kw)

        self._rope_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

        # Match the decoder attention's SDPA precision (`model_qwen3vl.py::Qwen3VlAttention`):
        # HiFi4 with fp32 accumulation. The default is lower precision, and while a single block still
        # scores ~99.99% either way, 27 of them compound -- the tower is deep enough that the
        # difference is visible in the merged output where one block hides it.
        self._sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def _ring_program_config(self, local_seq_len: int) -> ttnn.SDPAProgramConfig:
        """Flash tiling for the ring, sized from the LOCAL shard.

        Built per call rather than in `__init__` like the replicated path's config, because the shard
        length is a property of the request (patch count / sp_factor) and a chunk larger than the shard
        over-reads it. Mirrors the decoder's `_sdpa_program_config` clamp.
        """
        chunk = min(-(-local_seq_len // _TILE) * _TILE, 128)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self._sdpa_worker_grid,
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=False,  # False is the more accurate softmax
        )

    def _windowed_program_config(self, seq_len: int) -> ttnn.SDPAProgramConfig:
        """Flash tiling for windowed (block-diagonal) attention.

        Large K chunk is a fidelity choice (fewer streaming-softmax rescales); (64, 960) is the largest
        that fits the pipeline's L1.
        """
        tiles = -(-seq_len // _TILE) * _TILE
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=min(tiles, 64),
            k_chunk_size=min(tiles, 960),
            exp_approx_mode=False,
        )

    def _rope_interleave_qk(self, parts: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Permute only the q and k thirds of a `[q|k|v]` pack SPLIT->INTERLEAVED; leave v as is."""
        q, k, v = parts
        pk = dict(num_heads=self.num_heads, head_dim=self.head_dim)
        return rope_permute_qk(q, **pk), rope_permute_qk(k, **pk), v

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        kw = dict(num_heads=self.num_heads, head_dim=self.head_dim, padded=self.padded_head_dim)
        tp = self._p.tp_factor
        if (w := state.get("qkv.weight")) is not None:
            # the reference packs [q|k|v] on the output axis; pad each third's heads independently
            parts = self._rope_interleave_qk(w.chunk(3, dim=0))
            padded = torch.cat([pad_head_dim(part, axis=0, **kw) for part in parts])
            state["qkv.weight"] = interleave_for_col_parallel(padded, parts=3, tp_factor=tp)
        if (b := state.get("qkv.bias")) is not None:
            parts = self._rope_interleave_qk(b.chunk(3, dim=0))
            padded = torch.cat([pad_head_dim(part, axis=0, **kw) for part in parts])
            state["qkv.bias"] = interleave_for_col_parallel(padded, parts=3, tp_factor=tp)
        if (w := state.get("proj.weight")) is not None:
            # `proj` is row-parallel: its INPUT axis is the one that fractures, and it fractures
            # contiguously into the same per-device head groups `qkv` produces. So padding the head
            # dimension is all that is needed -- no interleave, and no transpose (RowParallelLinear
            # does its own).
            state["proj.weight"] = pad_head_dim(w, axis=1, **kw)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
        cu_seqlens: Sequence[int] | None = None,
    ) -> ttnn.Tensor:
        seq_len = hidden_states.shape[-2]
        qkv = self.qkv.forward(hidden_states)

        qkv = ttnn.reshape(qkv, (1, 1, seq_len, 3 * self.local_inner))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_local_heads,
            num_kv_heads=self.num_local_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cos, sin = pos_embeds
        cos = ttnn.reshape(cos, (1, 1, cos.shape[-2], cos.shape[-1]))
        sin = ttnn.reshape(sin, (1, 1, sin.shape[-2], sin.shape[-1]))
        q = ttnn.experimental.rotary_embedding_llama(
            q,
            cos,
            sin,
            self._rope_trans_mat,
            is_decode_mode=False,
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )
        k = ttnn.experimental.rotary_embedding_llama(
            k,
            cos,
            sin,
            self._rope_trans_mat,
            is_decode_mode=False,
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )

        single_block = cu_seqlens is None or len(cu_seqlens) <= 2
        if self._p.sp:
            # Ring SDPA rejects a non-tile-aligned shard deep in the device op ("Per-device Q seq
            # length must be divisible by TILE_HEIGHT"); check it here so the constraint is legible.
            # This is stricter than, and therefore subsumes, the merger's merge-group alignment.
            if seq_len % _TILE != 0:
                msg = (
                    f"sequence-parallel shard has {seq_len} rows, which is not a multiple of {_TILE}; "
                    f"the patch count must be divisible by sp_factor * {_TILE}"
                )
                raise ValueError(msg)
            if single_block:
                attn = self._ring_attention(q, k, v, seq_len)
            else:
                attn = self._windowed_sp_attention(q, k, v, seq_len, cu_seqlens)
        elif single_block:
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                scale=self.scale,
                program_config=self._windowed_program_config(seq_len),
                compute_kernel_config=self._sdpa_compute_kernel_config,
            )
        else:
            if cu_seqlens[0] != 0 or cu_seqlens[-1] != seq_len:
                msg = f"cu_seqlens must span [0, {seq_len}], got {cu_seqlens[0]}..{cu_seqlens[-1]}"
                raise ValueError(msg)
            cu_window = ttnn.from_torch(
                torch.tensor(cu_seqlens, dtype=torch.int32),
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
            )
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                scale=self.scale,
                program_config=self._windowed_program_config(seq_len),
                compute_kernel_config=self._sdpa_compute_kernel_config,
                cu_window_seqlens=cu_window,
            )
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.reshape(attn, (seq_len, self.local_inner))
        return row_parallel_seq_forward(self.proj, attn, self._p)

    def _ring_attention(self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, local_seq_len: int) -> ttnn.Tensor:
        """Full attention over a sequence sharded on the SP axis.

        `ring_joint_scaled_dot_product_attention` is the same primitive the Wan denoiser drives
        (`wan2_2/attention_wan.py`); it gathers k/v around the ring while streaming the softmax, so no
        device ever materializes the whole `s x s` score matrix. The joint inputs are the API's
        cross-attention slots and are unused here -- zero-width tensors keep it pure self-attention.
        """
        sp_axis, ccl = self._p.sp_axis, self._p.ccl_manager
        empty_joint = bf16_tensor(
            torch.zeros((1, self.num_local_heads, 0, self.padded_head_dim)), device=self.mesh_device
        )
        attn, _joint, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            k,
            v,
            empty_joint,
            empty_joint,
            empty_joint,
            persistent_output_buffer_k=ccl.get_ag_ping_pong_buffer(k.shape, 2, sp_axis, dtype=k.dtype),
            persistent_output_buffer_v=ccl.get_ag_ping_pong_buffer(v.shape, 2, sp_axis, dtype=v.dtype),
            joint_strategy="rear",
            logical_n=local_seq_len * self._p.sp_factor,
            program_config=self._ring_program_config(local_seq_len),
            compute_kernel_config=self._sdpa_compute_kernel_config,
            dim=2,
            scale=self.scale,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(sp_axis),
            num_links=ccl.num_links,
            cluster_axis=sp_axis,
            mesh_device=self.mesh_device,
            topology=ccl.topology,
            subdevice_id=ccl.ccl_sub_device_id,
            ccl_core_grid_offset=(self._sdpa_worker_grid[0], 0),
            use_column_major_ccl=True,
        )
        return attn

    def _windowed_sp_attention(
        self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, local_seq_len: int, cu_seqlens: Sequence[int]
    ) -> ttnn.Tensor:
        """Windowed (block-diagonal) attention over a sequence sharded on the SP axis.

        Q stays local, K/V are all-gathered; each shard's global row offset is passed as a sharded tensor.
        """
        sp_axis, ccl = self._p.sp_axis, self._p.ccl_manager
        global_seq_len = local_seq_len * self._p.sp_factor
        if cu_seqlens[0] != 0 or cu_seqlens[-1] != global_seq_len:
            msg = f"cu_seqlens must span [0, {global_seq_len}], got {cu_seqlens[0]}..{cu_seqlens[-1]}"
            raise ValueError(msg)
        k = ccl.all_gather(k, dim=-2, mesh_axis=sp_axis, use_hyperparams=True)
        v = ccl.all_gather(v, dim=-2, mesh_axis=sp_axis, use_hyperparams=True)
        cu_window = ttnn.from_torch(
            torch.tensor(cu_seqlens, dtype=torch.int32),
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )
        q_offsets = typed_tensor(
            torch.arange(self._p.sp_factor, dtype=torch.int32) * local_seq_len,
            ttnn.uint32,
            device=self.mesh_device,
            mesh_axis=sp_axis,
            shard_dim=0,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            program_config=self._windowed_program_config(local_seq_len),
            compute_kernel_config=self._sdpa_compute_kernel_config,
            cu_window_seqlens=cu_window,
            windowed_q_token_offset_tensor=q_offsets,
        )
