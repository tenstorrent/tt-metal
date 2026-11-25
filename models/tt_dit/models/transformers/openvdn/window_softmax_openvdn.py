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
from collections.abc import Sequence

import torch

import ttnn

from ....utils import tensor

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
    """Window softmax for one layout. Inputs `[1, H, N, d]` bf16, QK-normed and RoPE'd."""

    def __init__(
        self,
        device: ttnn.MeshDevice,
        layout: WindowLayout,
        bounds: Sequence[Range],
        *,
        scale: float,
        anchor_frames: str = "both",
    ) -> None:
        self.device = device
        self.scale = scale
        self.groups = window_groups(layout, bounds, anchor_frames)
        self.program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=_PAD,
            k_chunk_size=_PAD,
            exp_approx_mode=False,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        self._masks: dict[tuple[int, int], ttnn.Tensor] = {}

    def _mask(self, q_rows: int, k_rows: int) -> ttnn.Tensor | None:
        """Additive mask hiding the padded keys, or None when nothing is padded."""
        if k_rows % _PAD == 0:
            return None
        key = (q_rows, k_rows)
        if key not in self._masks:
            q_pad, k_pad = q_rows + (-q_rows % _PAD), k_rows + (-k_rows % _PAD)
            mask = torch.zeros(1, 1, q_pad, k_pad)
            mask[..., k_rows:] = float("-inf")
            self._masks[key] = tensor.from_torch(mask, device=self.device, dtype=ttnn.bfloat16)
        return self._masks[key]

    def __call__(self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor) -> ttnn.Tensor:
        heads, dim = q.shape[1], q.shape[3]
        outs = []
        for (q0, q1), key_ranges in self.groups:
            k_rows = sum(r1 - r0 for r0, r1 in key_ranges)
            qg = _pad_rows(_rows(q, [(q0, q1)]), _PAD)
            kg = _pad_rows(_rows(k, key_ranges), _PAD)
            vg = _pad_rows(_rows(v, key_ranges), _PAD)
            out = ttnn.transformer.scaled_dot_product_attention(
                qg,
                kg,
                vg,
                attn_mask=self._mask(q1 - q0, k_rows),
                is_causal=False,
                scale=self.scale,
                program_config=self.program_config,
                compute_kernel_config=self.compute_config,
            )
            if out.shape[2] != q1 - q0:
                out = ttnn.slice(out, [0, 0, 0, 0], [1, heads, q1 - q0, dim])
            outs.append((q0, out))
        outs.sort(key=lambda item: item[0])
        return ttnn.concat([o for _, o in outs], dim=2)
