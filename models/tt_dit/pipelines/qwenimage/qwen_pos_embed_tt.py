# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device-resident builder for Qwen image RoPE tensors.

This module mirrors the logic of ``diffusers.QwenEmbedRope.forward`` but assembles
the per-call cos/sin tables on the Tenstorrent submesh instead of the host CPU.

Only replicated outputs are supported today. Sequence-parallel callers (``sp_factor > 1``)
must continue to use the host-side ``QwenEmbedRope.forward`` path and shard at upload
time via ``from_torch(..., mesh_axes=[sp_axis, None])``; there is currently no ttnn
primitive for replicated-to-sharded resharding along the sequence axis.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch

import ttnn

from ...utils import tensor as tensor_utils


class QwenPosEmbedTT:
    """Pre-uploads Qwen RoPE frequency tables and assembles cos/sin on device.

    The class holds the following per-submesh buffers (all replicated):

    * ``_pos_real_full`` / ``_pos_imag_full``: full ``[max_pos, A]`` tables in TILE
      layout, used for the prompt (text) slice where the last dim is tile aligned.
    * ``_pos_real_axes[i]`` / ``_pos_imag_axes[i]`` / ``_neg_real_axes[i]`` /
      ``_neg_imag_axes[i]``: per-axis pieces of size ``[max_pos, axes_dim[i] // 2]``
      in ROW_MAJOR layout (per-axis last dims are not tile aligned on Qwen, e.g.
      ``[8, 28, 28]``).

    ``build`` returns the four half-dim tensors (``spatial_rope_{cos,sin}_half`` and
    ``prompt_rope_{cos,sin}_half``) in TILE layout. The caller is responsible for the
    subsequent ``rope_double_last_dim_device`` doubling of the last dim.
    """

    def __init__(self, torch_pos_embed, submesh_devices: Sequence[ttnn.MeshDevice]):
        self._axes_dim: List[int] = list(torch_pos_embed.axes_dim)
        self._scale_rope: bool = bool(torch_pos_embed.scale_rope)
        self._pos_table_len: int = int(torch_pos_embed.pos_freqs.shape[0])
        self._submesh_devices = list(submesh_devices)

        half_splits = [d // 2 for d in self._axes_dim]
        if sum(half_splits) * 2 != sum(self._axes_dim):
            msg = f"axes_dim must all be even, got {self._axes_dim}"
            raise ValueError(msg)

        pos_real = torch.split(torch_pos_embed.pos_freqs.real.contiguous(), half_splits, dim=1)
        pos_imag = torch.split(torch_pos_embed.pos_freqs.imag.contiguous(), half_splits, dim=1)
        neg_real = torch.split(torch_pos_embed.neg_freqs.real.contiguous(), half_splits, dim=1)
        neg_imag = torch.split(torch_pos_embed.neg_freqs.imag.contiguous(), half_splits, dim=1)

        pos_real_full_cpu = torch_pos_embed.pos_freqs.real.contiguous()
        pos_imag_full_cpu = torch_pos_embed.pos_freqs.imag.contiguous()

        self._pos_real_full: List[ttnn.Tensor] = []
        self._pos_imag_full: List[ttnn.Tensor] = []
        self._pos_real_axes: List[List[ttnn.Tensor]] = [[], [], []]
        self._pos_imag_axes: List[List[ttnn.Tensor]] = [[], [], []]
        self._neg_real_axes: List[List[ttnn.Tensor]] = [[], [], []]
        self._neg_imag_axes: List[List[ttnn.Tensor]] = [[], [], []]

        for submesh in self._submesh_devices:
            self._pos_real_full.append(tensor_utils.from_torch(pos_real_full_cpu, device=submesh, on_host=False))
            self._pos_imag_full.append(tensor_utils.from_torch(pos_imag_full_cpu, device=submesh, on_host=False))
            for ax in range(3):
                self._pos_real_axes[ax].append(
                    tensor_utils.from_torch(
                        pos_real[ax].contiguous(),
                        device=submesh,
                        on_host=False,
                        layout=ttnn.Layout.ROW_MAJOR,
                    )
                )
                self._pos_imag_axes[ax].append(
                    tensor_utils.from_torch(
                        pos_imag[ax].contiguous(),
                        device=submesh,
                        on_host=False,
                        layout=ttnn.Layout.ROW_MAJOR,
                    )
                )
                self._neg_real_axes[ax].append(
                    tensor_utils.from_torch(
                        neg_real[ax].contiguous(),
                        device=submesh,
                        on_host=False,
                        layout=ttnn.Layout.ROW_MAJOR,
                    )
                )
                self._neg_imag_axes[ax].append(
                    tensor_utils.from_torch(
                        neg_imag[ax].contiguous(),
                        device=submesh,
                        on_host=False,
                        layout=ttnn.Layout.ROW_MAJOR,
                    )
                )

    @staticmethod
    def _slice_dim0(t: ttnn.Tensor, begin: int, end: int) -> ttnn.Tensor:
        rank = len(t.shape)
        begins = [0] * rank
        ends = list(t.shape)
        begins[0] = int(begin)
        ends[0] = int(end)
        return ttnn.slice(t, begins, ends)

    def build(
        self,
        submesh_index: int,
        img_shapes_per_batch: Iterable[Tuple[int, int, int]],
        max_txt_seq_len: int,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Assemble the four half-dim RoPE tensors on the chosen submesh.

        Parameters
        ----------
        submesh_index:
            Index into the submesh list passed at construction.
        img_shapes_per_batch:
            Iterable of ``(frame, height, width)`` tuples. For ``QwenImageEdit`` this
            contains one entry per latent (noise + reference images); for ``QwenImage``
            it contains a single entry.
        max_txt_seq_len:
            Maximum prompt sequence length (used to slice the text RoPE).

        Returns
        -------
        Tuple of four device tensors (spatial cos, spatial sin, prompt cos, prompt sin).
        Each has shape ``[seq_len, sum(axes_dim) // 2]`` and is in TILE layout.
        """
        img_shapes = list(img_shapes_per_batch)
        if not img_shapes:
            msg = "img_shapes_per_batch must contain at least one entry"
            raise ValueError(msg)

        a0, a1, a2 = [d // 2 for d in self._axes_dim]
        a_sum = a0 + a1 + a2

        vid_cos_pieces: List[ttnn.Tensor] = []
        vid_sin_pieces: List[ttnn.Tensor] = []
        max_vid_index = 0

        pos_real_ax = [self._pos_real_axes[ax][submesh_index] for ax in range(3)]
        pos_imag_ax = [self._pos_imag_axes[ax][submesh_index] for ax in range(3)]
        neg_real_ax = [self._neg_real_axes[ax][submesh_index] for ax in range(3)]
        neg_imag_ax = [self._neg_imag_axes[ax][submesh_index] for ax in range(3)]

        for idx, fhw in enumerate(img_shapes):
            frame, height, width = int(fhw[0]), int(fhw[1]), int(fhw[2])

            frame_cos = self._slice_dim0(pos_real_ax[0], idx, idx + frame)
            frame_sin = self._slice_dim0(pos_imag_ax[0], idx, idx + frame)

            if self._scale_rope:
                h_tail = height - height // 2
                h_head = height // 2
                w_tail = width - width // 2
                w_head = width // 2

                neg_h_cos = self._slice_dim0(neg_real_ax[1], self._pos_table_len - h_tail, self._pos_table_len)
                neg_h_sin = self._slice_dim0(neg_imag_ax[1], self._pos_table_len - h_tail, self._pos_table_len)
                pos_h_cos = self._slice_dim0(pos_real_ax[1], 0, h_head)
                pos_h_sin = self._slice_dim0(pos_imag_ax[1], 0, h_head)
                height_cos = ttnn.concat([neg_h_cos, pos_h_cos], dim=0)
                height_sin = ttnn.concat([neg_h_sin, pos_h_sin], dim=0)

                neg_w_cos = self._slice_dim0(neg_real_ax[2], self._pos_table_len - w_tail, self._pos_table_len)
                neg_w_sin = self._slice_dim0(neg_imag_ax[2], self._pos_table_len - w_tail, self._pos_table_len)
                pos_w_cos = self._slice_dim0(pos_real_ax[2], 0, w_head)
                pos_w_sin = self._slice_dim0(pos_imag_ax[2], 0, w_head)
                width_cos = ttnn.concat([neg_w_cos, pos_w_cos], dim=0)
                width_sin = ttnn.concat([neg_w_sin, pos_w_sin], dim=0)
            else:
                height_cos = self._slice_dim0(pos_real_ax[1], 0, height)
                height_sin = self._slice_dim0(pos_imag_ax[1], 0, height)
                width_cos = self._slice_dim0(pos_real_ax[2], 0, width)
                width_sin = self._slice_dim0(pos_imag_ax[2], 0, width)

            frame_b_cos = ttnn.reshape(frame_cos, [frame, 1, 1, a0])
            frame_b_sin = ttnn.reshape(frame_sin, [frame, 1, 1, a0])
            if height != 1 or width != 1:
                frame_b_cos = ttnn.repeat(frame_b_cos, (1, height, width, 1))
                frame_b_sin = ttnn.repeat(frame_b_sin, (1, height, width, 1))

            height_b_cos = ttnn.reshape(height_cos, [1, height, 1, a1])
            height_b_sin = ttnn.reshape(height_sin, [1, height, 1, a1])
            if frame != 1 or width != 1:
                height_b_cos = ttnn.repeat(height_b_cos, (frame, 1, width, 1))
                height_b_sin = ttnn.repeat(height_b_sin, (frame, 1, width, 1))

            width_b_cos = ttnn.reshape(width_cos, [1, 1, width, a2])
            width_b_sin = ttnn.reshape(width_sin, [1, 1, width, a2])
            if frame != 1 or height != 1:
                width_b_cos = ttnn.repeat(width_b_cos, (frame, height, 1, 1))
                width_b_sin = ttnn.repeat(width_b_sin, (frame, height, 1, 1))

            vid_cos = ttnn.concat([frame_b_cos, height_b_cos, width_b_cos], dim=-1)
            vid_sin = ttnn.concat([frame_b_sin, height_b_sin, width_b_sin], dim=-1)

            seq_len = frame * height * width
            vid_cos = ttnn.reshape(vid_cos, [seq_len, a_sum])
            vid_sin = ttnn.reshape(vid_sin, [seq_len, a_sum])

            vid_cos_pieces.append(vid_cos)
            vid_sin_pieces.append(vid_sin)

            if self._scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        if len(vid_cos_pieces) > 1:
            spatial_rope_cos_half = ttnn.concat(vid_cos_pieces, dim=0)
            spatial_rope_sin_half = ttnn.concat(vid_sin_pieces, dim=0)
        else:
            spatial_rope_cos_half = vid_cos_pieces[0]
            spatial_rope_sin_half = vid_sin_pieces[0]

        if spatial_rope_cos_half.layout != ttnn.TILE_LAYOUT:
            spatial_rope_cos_half = ttnn.to_layout(spatial_rope_cos_half, ttnn.TILE_LAYOUT)
            spatial_rope_sin_half = ttnn.to_layout(spatial_rope_sin_half, ttnn.TILE_LAYOUT)

        prompt_rope_cos_half = ttnn.slice(
            self._pos_real_full[submesh_index],
            [int(max_vid_index), 0],
            [int(max_vid_index) + int(max_txt_seq_len), a_sum],
        )
        prompt_rope_sin_half = ttnn.slice(
            self._pos_imag_full[submesh_index],
            [int(max_vid_index), 0],
            [int(max_vid_index) + int(max_txt_seq_len), a_sum],
        )

        return spatial_rope_cos_half, spatial_rope_sin_half, prompt_rope_cos_half, prompt_rope_sin_half
