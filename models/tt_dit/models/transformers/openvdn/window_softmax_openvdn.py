# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""OpenVDN window softmax on device: exact softmax in which (video, video) pairs see only the query frame's window
(plus the anchor frames), and every pair involving a global (text/audio/reference) row stays dense.

Reference: OpenVDN/vdn-minimax-h3 @e262cb57, `window_softmax_reference` in
`src/models/softmax_attention/window.py` (mirrored in `reference_openvdn.py`).

Correct first, not fast: the sequence is cut into query groups that share one key set (global rows, each anchor
frame, each run of frames with the same window; c1 windows are whole chunks), and each group is one dense
`ttnn.transformer.scaled_dot_product_attention` call over its gathered keys. Lengths that are not a multiple of the
tile are padded, and padded keys are masked out.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import torch

import ttnn

from ....utils import tensor
from . import linear_ops_openvdn as ops

Range = tuple[int, int]

_PAD = ttnn.TILE_SIZE


@dataclasses.dataclass(frozen=True)
class WindowLayout:
    """Global rows outside `[video_start, video_end)`, video rows frame-major."""

    total: int
    video_start: int
    num_frames: int
    tokens_per_frame: int

    @property
    def video_end(self) -> int:
        return self.video_start + self.num_frames * self.tokens_per_frame


def window_groups(
    layout: WindowLayout, bounds: Sequence[Range], anchor_frames: str = "both"
) -> list[tuple[Range, list[Range]]]:
    """`(query rows, key row ranges)` per group; the query ranges partition the sequence. Key order follows the
    reference: globals, the window's frames, then anchor columns outside the window."""
    vs, ve, f_count, s = layout.video_start, layout.video_end, layout.num_frames, layout.tokens_per_frame
    full = [(0, layout.total)]
    global_keys = [r for r in ((0, vs), (ve, layout.total)) if r[1] > r[0]]

    def frame_rows(f0: int, f1: int) -> Range:
        return (vs + f0 * s, vs + (f1 + 1) * s)

    def keys_for(frame: int) -> list[Range]:
        lo, hi = max(bounds[frame][0], 0), min(bounds[frame][1], f_count - 1)
        if anchor_frames in ("rows", "both") and frame in (0, f_count - 1):
            return full
        extra = [a for a in ((0, f_count - 1) if anchor_frames in ("columns", "both") else ()) if not lo <= a <= hi]
        return global_keys + [frame_rows(lo, hi)] + [frame_rows(a, a) for a in extra]

    groups = [((r0, r1), full) for r0, r1 in global_keys]
    start = 0
    for frame in range(1, f_count + 1):
        if frame == f_count or keys_for(frame) != keys_for(start):
            groups.append((frame_rows(start, frame - 1), keys_for(start)))
            start = frame
    return groups


def _rows(x: ttnn.Tensor, ranges: Sequence[Range]) -> ttnn.Tensor:
    _, heads, _, dim = x.shape
    parts = [ttnn.slice(x, [0, 0, r0, 0], [1, heads, r1, dim]) for r0, r1 in ranges]
    return parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)


def _pad_rows(x: ttnn.Tensor, multiple: int) -> ttnn.Tensor:
    rows = x.shape[2]
    pad = -rows % multiple
    return x if pad == 0 else ttnn.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0)


class WindowSoftmax:
    """Window softmax for one layout. Inputs `[1, H, N, d]` bf16, QK-normed and RoPE'd.

    The padding masks depend only on the group shapes, so one set serves every instance (every layer) on a device:
    at a 768p, 5 s sequence they are ~1 GB, which per layer would not fit."""

    _shared_masks: dict[tuple[int, int, int], ttnn.Tensor] = {}

    def __init__(
        self,
        device: ttnn.MeshDevice,
        layout: WindowLayout,
        bounds: Sequence[Range],
        *,
        scale: float,
        anchor_frames: str = "both",
        sp_axis: int | None = None,
        sp_factor: int = 1,
        gather: Callable[[ttnn.Tensor], ttnn.Tensor] | None = None,
        q_chunk: int = _PAD,
        k_chunk: int = _PAD,
        aligned: bool = False,
    ) -> None:
        """With `sp_factor > 1` (q, k, v replicated on the `sp_axis` devices), each group's query rows are split
        across those devices (`mesh_partition`), each attends its share against the group's keys, and `gather`
        (an all-gather on dim 2 over `sp_axis`) reassembles the group: every device then holds the whole output, as
        without the split, for 1/sp of the attention work. Exact: a query row's softmax does not depend on which
        device computes it."""
        self.device = device
        self.scale = scale
        self.groups = window_groups(layout, bounds, anchor_frames)
        self.sp_axis, self.sp_factor, self.gather = sp_axis, sp_factor, gather
        assert sp_factor == 1 or (sp_axis is not None and gather is not None)
        self.q_chunk, self.k_chunk = q_chunk, k_chunk
        self.program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        self._masks = self._shared_masks
        #: `aligned=True`: run on the frame-padded layout, where every group slice is whole tiles.
        self.aligned = _AlignedLayout(layout) if aligned else None

    def _mask(self, q_pad: int, k_rows: int) -> ttnn.Tensor | None:
        """Additive mask `[q_pad, k_pad]` hiding the padded keys, or None when nothing is padded."""
        if k_rows % self.k_chunk == 0:
            return None
        key = (id(self.device), q_pad, k_rows, self.k_chunk)
        if key not in self._masks:
            k_pad = k_rows + (-k_rows % self.k_chunk)
            mask = torch.zeros(1, 1, q_pad, k_pad)
            mask[..., k_rows:] = float("-inf")
            self._masks[key] = tensor.from_torch(mask, device=self.device, dtype=ttnn.bfloat16)
        return self._masks[key]

    def __call__(self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor) -> ttnn.Tensor:
        if self.aligned is not None:
            return self._call_aligned(q, k, v)
        heads, dim = q.shape[1], q.shape[3]
        outs = []
        for (q0, q1), key_ranges in self.groups:
            k_rows = sum(r1 - r0 for r0, r1 in key_ranges)
            qg = _pad_rows(_rows(q, [(q0, q1)]), self.q_chunk * self.sp_factor)
            if self.sp_factor > 1:
                qg = ttnn.mesh_partition(qg, dim=2, cluster_axis=self.sp_axis)
            kg = _pad_rows(_rows(k, key_ranges), self.k_chunk)
            vg = _pad_rows(_rows(v, key_ranges), self.k_chunk)
            out = ttnn.transformer.scaled_dot_product_attention(
                qg,
                kg,
                vg,
                attn_mask=self._mask(qg.shape[2], k_rows),
                is_causal=False,
                scale=self.scale,
                program_config=self.program_config,
                compute_kernel_config=self.compute_config,
            )
            if self.sp_factor > 1:
                out = self.gather(out)
            if out.shape[2] != q1 - q0:
                out = ttnn.slice(out, [0, 0, 0, 0], [1, heads, q1 - q0, dim])
            outs.append((q0, out))
        outs.sort(key=lambda item: item[0])
        return ttnn.concat([o for _, o in outs], dim=2)

    # ------------------------------------------------------------------ tile-aligned path

    def _call_aligned(self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor) -> ttnn.Tensor:
        """The same groups on the frame-padded layout (`_AlignedLayout`): q, k, v are converted once, every group
        slice is then tile-aligned (no untilize/tilize per slice), padded keys are masked, and the output is
        converted back once. Same softmax over the same keys in the same order; padded query rows are dropped."""
        geo = self.aligned
        q, k, v = (geo.to_aligned(t) for t in (q, k, v))
        heads, dim = q.shape[1], q.shape[3]
        outs = []
        for index, ((q0, q1), key_ranges) in enumerate(self.groups):
            a0, a1 = geo.map(q0, q1)
            k_ranges = [geo.map(r0, r1) for r0, r1 in key_ranges]
            qg = _pad_rows(_rows(q, [(a0, a1)]), self.q_chunk * self.sp_factor)
            if self.sp_factor > 1:
                qg = ttnn.mesh_partition(qg, dim=2, cluster_axis=self.sp_axis)
            kg = _pad_rows(_rows(k, k_ranges), self.k_chunk)
            vg = _pad_rows(_rows(v, k_ranges), self.k_chunk)
            out = ttnn.transformer.scaled_dot_product_attention(
                qg,
                kg,
                vg,
                attn_mask=self._aligned_mask(index, qg.shape[2], k_ranges, kg.shape[2]),
                is_causal=False,
                scale=self.scale,
                program_config=self.program_config,
                compute_kernel_config=self.compute_config,
            )
            if self.sp_factor > 1:
                out = self.gather(out)
            if out.shape[2] != a1 - a0:
                out = ttnn.slice(out, [0, 0, 0, 0], [1, heads, a1 - a0, dim])
            outs.append((a0, out))
        outs.sort(key=lambda item: item[0])
        out = ttnn.concat([o for _, o in outs], dim=2)
        return geo.from_aligned(out)

    def _aligned_mask(self, index: int, q_rows: int, k_ranges: list[Range], k_rows: int) -> ttnn.Tensor | None:
        valid = torch.cat([self.aligned.valid[r0:r1] for r0, r1 in k_ranges])
        valid = torch.cat([valid, torch.zeros(k_rows - valid.numel(), dtype=torch.bool)])
        if bool(valid.all()):
            return None
        key = (id(self.device), self.aligned.key, index, q_rows, k_rows)
        if key not in self._masks:
            mask = torch.zeros(1, 1, q_rows, k_rows)
            mask[..., ~valid] = float("-inf")
            self._masks[key] = tensor.from_torch(mask, device=self.device, dtype=ttnn.bfloat16)
        return self._masks[key]


def _ceil(n: int, multiple: int = _PAD) -> int:
    return -(-n // multiple) * multiple


class _AlignedLayout:
    """The packed sequence with every segment padded to whole tiles (`multiple` rows): head globals `[0, vs)` ->
    `[0, A0)`, frame f -> `[A0 + f*Sp, A0 + f*Sp + S)` of a `Sp`-row slot, tail globals after the frames, then zero
    rows up to a multiple of `total_multiple`. `valid` marks the real rows."""

    def __init__(self, layout: WindowLayout, multiple: int = _PAD, total_multiple: int = 1) -> None:
        self.layout = layout
        self.key = (*dataclasses.astuple(layout), multiple, total_multiple)
        vs, ve, s, f = layout.video_start, layout.video_end, layout.tokens_per_frame, layout.num_frames
        self.head, self.slot = _ceil(vs, multiple), _ceil(s, multiple)
        self.tail_start = self.head + f * self.slot
        self.tail_len = layout.total - ve
        self.total = _ceil(self.tail_start + _ceil(self.tail_len, multiple), total_multiple)
        valid = torch.zeros(self.total, dtype=torch.bool)
        valid[:vs] = True
        for i in range(f):
            valid[self.head + i * self.slot : self.head + i * self.slot + s] = True
        valid[self.tail_start : self.tail_start + self.tail_len] = True
        self.valid = valid

    def map(self, r0: int, r1: int) -> Range:
        """A logical group range (globals, whole frames, or everything) -> its aligned range."""
        lay = self.layout
        vs, ve, s = lay.video_start, lay.video_end, lay.tokens_per_frame
        if (r0, r1) == (0, lay.total):
            return 0, self.total
        if r1 <= vs:
            assert (r0, r1) == (0, vs)
            return 0, self.head
        if r0 >= ve:
            assert (r0, r1) == (ve, lay.total)
            return self.tail_start, self.total
        assert (r0 - vs) % s == 0 and (r1 - vs) % s == 0 and vs <= r0 < r1 <= ve, (r0, r1)
        return self.head + (r0 - vs) // s * self.slot, self.head + (r1 - vs) // s * self.slot

    @staticmethod
    def _zeros_rows(x: ttnn.Tensor, rows: int) -> ttnn.Tensor:
        shape = list(x.shape)
        shape[-2] = rows
        return ops.zeros(x.device(), shape, x.dtype, ttnn.ROW_MAJOR_LAYOUT)

    def to_aligned(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """`[1, H, N, d]` logical rows -> `[1, H, total, d]` aligned (zero pads), via one row-major round trip."""
        lay = self.layout
        _, h, _, d = x.shape
        vs, ve, s, f = lay.video_start, lay.video_end, lay.tokens_per_frame, lay.num_frames
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        parts = []
        if vs:
            parts.append(ttnn.slice(x, [0, 0, 0, 0], [1, h, vs, d]))
            if self.head > vs:
                parts.append(self._zeros_rows(parts[-1], self.head - vs))
        video = ttnn.reshape(ttnn.slice(x, [0, 0, vs, 0], [1, h, ve, d]), [h, f, s, d])
        if self.slot > s:
            video = ttnn.concat([video, self._zeros_rows(video, self.slot - s)], dim=2)
        parts.append(ttnn.reshape(video, [1, h, f * self.slot, d]))
        if self.tail_len:
            parts.append(ttnn.slice(x, [0, 0, ve, 0], [1, h, lay.total, d]))
        if self.total - self.tail_start > self.tail_len:
            parts.append(self._zeros_rows(parts[-1], self.total - self.tail_start - self.tail_len))
        return ttnn.to_layout(ttnn.concat(parts, dim=2), ttnn.TILE_LAYOUT)

    def from_aligned(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Inverse of `to_aligned`: drop the pad rows."""
        lay = self.layout
        _, h, _, d = x.shape
        vs, s, f = lay.video_start, lay.tokens_per_frame, lay.num_frames
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        parts = [ttnn.slice(x, [0, 0, 0, 0], [1, h, vs, d])] if vs else []
        video = ttnn.slice(x, [0, 0, self.head, 0], [1, h, self.tail_start, d])
        video = ttnn.slice(ttnn.reshape(video, [h, f, self.slot, d]), [0, 0, 0, 0], [h, f, s, d])
        parts.append(ttnn.reshape(video, [1, h, f * s, d]))
        if self.tail_len:
            parts.append(ttnn.slice(x, [0, 0, self.tail_start, 0], [1, h, self.tail_start + self.tail_len, d]))
        out = ttnn.concat(parts, dim=2) if len(parts) > 1 else parts[0]
        return ttnn.to_layout(out, ttnn.TILE_LAYOUT)


class SparseWindowSoftmax:
    """The window softmax as one block-sparse ring attention (`ttnn.transformer.sparse_ring_sdpa`, the op H3's VSA
    runs) instead of one dense SDPA per group (stage 4).

    The sequence goes to the frame-padded layout with 64-row segments (`_AlignedLayout(multiple=64)`), so every
    64-row block holds rows of one segment (head globals, one frame, tail globals), real rows first. The groups'
    (query, key) pairs then become a static block mask, and `block_counts` (real rows per block) keeps the pad rows
    out of every softmax. Exact: each query row's softmax runs over the same keys (in another order). Inputs and
    output `[1, H, N, d]` replicated over SP, as `WindowSoftmax`; inside, q, k, v are split over SP and the op
    gathers K/V around the ring itself.

    `ring_sdpa(q, k, v, block_mask, block_counts)` runs the op with the caller's CCL setup."""

    BLOCK = 64

    def __init__(
        self,
        device: ttnn.MeshDevice,
        layout: WindowLayout,
        bounds: Sequence[Range],
        *,
        anchor_frames: str,
        sp_axis: int,
        sp_factor: int,
        gather: Callable[[ttnn.Tensor], ttnn.Tensor],
        ring_sdpa: Callable[..., ttnn.Tensor],
        num_heads: int,
    ) -> None:
        block = self.BLOCK
        self.device = device
        self.sp_axis, self.gather, self.ring_sdpa = sp_axis, gather, ring_sdpa
        self.geo = geo = _AlignedLayout(layout, multiple=block, total_multiple=block * sp_factor)
        num_blocks = geo.total // block
        per_shard = num_blocks // sp_factor
        pad = per_shard + (-per_shard % 32)

        valid = geo.valid.reshape(num_blocks, block)
        counts = valid.sum(dim=1)
        # real rows first in every block
        assert bool((valid == (torch.arange(block)[None, :] < counts[:, None])).all())

        def blocks(r0: int, r1: int) -> slice:
            a0, a1 = geo.map(r0, r1)
            return slice(a0 // block, -(-a1 // block))

        keep = torch.zeros(num_blocks, num_blocks, dtype=torch.int32)
        for (q0, q1), key_ranges in window_groups(layout, bounds, anchor_frames):
            for r0, r1 in key_ranges:
                keep[blocks(q0, q1), blocks(r0, r1)] = 1

        def shard_cols(t: torch.Tensor) -> torch.Tensor:  # [..., num_blocks] -> [..., sp * pad], each shard padded
            t = t.reshape(*t.shape[:-1], sp_factor, per_shard)
            t = torch.nn.functional.pad(t, (0, pad - per_shard))
            return t.reshape(*t.shape[:-2], sp_factor * pad)

        mask = shard_cols(keep).reshape(sp_factor, per_shard, sp_factor * pad)
        mask = torch.nn.functional.pad(mask, (0, 0, 0, pad - per_shard)).reshape(1, 1, sp_factor * pad, -1)
        self.block_mask = tensor.from_torch(
            mask.expand(1, num_heads, -1, -1).contiguous(),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axes=[None, None, sp_axis, None],
        )
        self.block_counts = tensor.from_torch(
            shard_cols(counts.to(torch.int32)).reshape(1, 1, 1, -1),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def __call__(self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor) -> ttnn.Tensor:
        q, k, v = (ttnn.mesh_partition(self.geo.to_aligned(t), dim=2, cluster_axis=self.sp_axis) for t in (q, k, v))
        out = self.ring_sdpa(q, k, v, self.block_mask, self.block_counts)
        return self.geo.from_aligned(self.gather(out))
