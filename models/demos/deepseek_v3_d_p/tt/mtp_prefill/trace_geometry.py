# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The MTP levels in a form one captured trace can replay for every chunk.

The eager path decides per chunk, on the host, whether a chip is split, which levels generate and
which row the LM head reads. A replay cannot re-decide, so here every decision is a persistent
device tensor: a mask or one-hot that degenerates to identity (ones, zeros) when unused. The program
is then the same for every chunk and only the tensors change; :meth:`MTPTraceGeometry.write`
refreshes them before each replay.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import global_to_local_token_id, rotated_row_of_position
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.device_windows import MTPSplitChipLookahead, MTPUnionEmbedding
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import mtp_generation_keep_rows, mtp_generation_select_rows

# The one-hot matmuls stand in for slices, so they must copy rows bit-exactly: HiFi4 keeps every
# mantissa bit of the copied operand and the fp32 dest holds it until the bf16 pack.
ONE_HOT_COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


class MTPTraceGeometry:
    """One chunk's MTP geometry as persistent ``[sp, 1, rows, cols]`` tensors, one row block per SP chip.

    - ``keep``: zero on the rows generation writes, applied once before level 0. Clearing them that early
      matches the eager clear at level ``provided_levels``: a level below it never reads those rows.
    - ``selects[k]``: level ``k``'s one-hot patch selector, all zero for a provided level.
    - ``lm_rows``: one-hot ``[32, W]`` picking the tile that holds the chunk's last real row.
    - ``win_keep[k]`` / ``win_place[k]``: the window at shift ``k + 1`` is ``plain * win_keep + win_place @ L``,
      which on the split chip swaps the rows before ``split_row`` for its lookahead ``L``. Identity elsewhere.
    - ``next_select``, ``take_tail``, ``take_next``: ``L`` is the union's tile at row W (the split chip's own
      lookahead slots) or the next chip's first tile.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        *,
        sp_factor: int,
        chunk_size: int,
        mesh_shape: tuple,
        sp_axis: int,
        num_mtp_tokens: int,
        num_levels: int,
    ):
        self.mesh_device = mesh_device
        self.sp_factor = int(sp_factor)
        self.chunk_size = int(chunk_size)
        self.mesh_shape = tuple(mesh_shape)
        self.sp_axis = int(sp_axis)
        self.num_mtp_tokens = int(num_mtp_tokens)
        self.num_levels = int(num_levels)
        assert self.chunk_size % self.sp_factor == 0, f"chunk {chunk_size} not divisible by sp_factor {sp_factor}"
        self.window_len = self.chunk_size // self.sp_factor
        self.union_len = self.window_len + self.num_mtp_tokens
        assert not self.has_split_chip or self.num_mtp_tokens >= ttnn.TILE_SIZE, (
            f"the split-chip lookahead reads a {ttnn.TILE_SIZE}-row tile past the trunk; the union only has "
            f"{self.num_mtp_tokens} lookahead rows"
        )
        self._tensors: dict = {}
        for name, host in self._host(0, self.chunk_size, self.num_levels).items():
            self._tensors[name] = ttnn.from_torch(
                host,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self._mapper(),
            )

    @property
    def has_split_chip(self) -> bool:
        return self.sp_factor > 1

    @property
    def keep(self) -> ttnn.Tensor:
        return self._tensors["keep"]

    @property
    def selects(self) -> list:
        return [self._tensors[f"select{k}"] for k in range(self.num_levels)]

    @property
    def lm_rows(self) -> ttnn.Tensor:
        return self._tensors["lm_rows"]

    @property
    def win_keep(self) -> list:
        return [self._tensors[f"win_keep{k}"] for k in range(self.num_levels)]

    @property
    def win_place(self) -> list:
        return [self._tensors[f"win_place{k}"] for k in range(self.num_levels)]

    @property
    def next_select(self) -> ttnn.Tensor:
        return self._tensors["next_select"]

    @property
    def take_tail(self) -> ttnn.Tensor:
        return self._tensors["take_tail"]

    @property
    def take_next(self) -> ttnn.Tensor:
        return self._tensors["take_next"]

    def write(self, actual_start: int, actual_end: int, provided_levels: int) -> None:
        """Refresh every tensor in place for the chunk ``[actual_start, actual_end)``. Never inside a capture."""
        for name, host in self._host(actual_start, actual_end, provided_levels).items():
            src = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._mapper())
            ttnn.copy_host_to_device_tensor(src, self._tensors[name])

    def deallocate(self) -> None:
        for t in self._tensors.values():
            ttnn.deallocate(t)
        self._tensors = {}

    def _mapper(self):
        return ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=self.mesh_shape, dims=(self.sp_axis, None))

    def _host(self, actual_start: int, actual_end: int, provided_levels: int) -> dict:
        sp, k_levels, tile = self.sp_factor, self.num_levels, ttnn.TILE_SIZE
        w, width = self.window_len, ttnn.TILE_SIZE * self.sp_factor
        assert 0 <= provided_levels <= k_levels, f"provided_levels {provided_levels} outside [0, {k_levels}]"
        geom = dict(
            num_mtp_tokens=self.num_mtp_tokens,
            num_levels=k_levels,
            chunk_start=actual_start,
            actual_end=actual_end,
        )
        generated = range(provided_levels, k_levels)
        out = {"keep": mtp_generation_keep_rows(sp, self.chunk_size, **geom, levels=generated)}

        tile_start, source_row = 0, 0
        if generated:
            last_row = rotated_row_of_position(actual_start, sp, w, actual_end - 1)
            assert (
                last_row is not None
            ), f"the chunk at {actual_start} does not carry its own last real position {actual_end - 1}"
            device_id, local = global_to_local_token_id(last_row, sp, self.chunk_size, is_balanced=False)
            tile_start = (local // tile) * tile
            source_row = device_id * tile + local % tile
        for k in range(k_levels):
            out[f"select{k}"] = (
                mtp_generation_select_rows(sp, self.chunk_size, **geom, level=k, source_row=source_row)
                if k in generated
                else torch.zeros(sp, 1, self.union_len, width)
            )
        lm_rows = torch.zeros(sp, 1, tile, w)
        lm_rows[:, 0, torch.arange(tile), tile_start + torch.arange(tile)] = 1.0
        out["lm_rows"] = lm_rows

        win_keep = [torch.ones(sp, 1, w, 1) for _ in range(k_levels)]
        win_place = [torch.zeros(sp, 1, w, tile) for _ in range(k_levels)]
        next_select = torch.zeros(sp, 1, tile, width)
        take_tail, take_next = torch.zeros(sp, 1, 1, 1), torch.zeros(sp, 1, 1, 1)
        split = MTPSplitChipLookahead.geometry(actual_start, w, sp, chunk_end=actual_end, num_levels=k_levels)
        if split is not None:
            split_row, split_chip, next_chip = split
            assert k_levels < split_row < w and k_levels <= tile <= self.num_mtp_tokens, (
                f"a split at row {split_row} of {w} cannot take a lookahead for K={k_levels}: it needs "
                f"K < split_row < window_len and K <= {tile} <= the union's lookahead rows"
            )
            for k in range(k_levels):
                shift = k + 1
                rows = torch.arange(split_row - shift, split_row)
                win_keep[k][split_chip, 0, rows, 0] = 0.0
                win_place[k][split_chip, 0, rows, torch.arange(shift)] = 1.0
            if next_chip is None:
                take_tail[:] = 1.0
            else:
                take_next[:] = 1.0
                next_select[:, 0, torch.arange(tile), next_chip * tile + torch.arange(tile)] = 1.0
        if self.has_split_chip:
            for k in range(k_levels):
                out[f"win_keep{k}"] = win_keep[k]
                out[f"win_place{k}"] = win_place[k]
            out.update(next_select=next_select, take_tail=take_tail, take_next=take_next)
        return out


class MTPTraceEmbedSource:
    """``TtMTPPredictor.forward``'s ``embeds`` callable with no per-chunk branch.

    Every level runs the generation chain and patches the union; ``geometry`` zeroes the patch of a
    provided level, so the eager result is reproduced with one program for every chunk.
    """

    def __init__(self, union: MTPUnionEmbedding, geometry: MTPTraceGeometry, embed_fn, all_gather_sp=None):
        assert union.num_levels == geometry.num_levels, (
            f"union carries {union.num_levels} levels, geometry {geometry.num_levels}; "
            "the runner and the runtime disagree on PREFILL_MTP_LEVELS"
        )
        assert not geometry.has_split_chip or all_gather_sp is not None, "the split-chip lookahead needs all_gather_sp"
        self.union = union
        self.geometry = geometry
        self.embed_fn = embed_fn
        self.all_gather_sp = all_gather_sp
        self._next_level = 0

    @property
    def generated_tokens(self) -> list:
        """Always empty: the generated ids never leave the device. Kept for interface parity."""
        return []

    def __call__(self, k: int, prev_normed) -> ttnn.Tensor:
        assert k == self._next_level, f"levels must run in order; expected {self._next_level}, got {k}"
        self._next_level += 1
        if k == 0:
            self.union.clear_rows(self.geometry.keep)
        gathered = self.embed_fn(prev_normed)
        self.union.add_patch(self.geometry.selects[k], gathered)
        ttnn.deallocate(gathered)
        return self.union.blended_window(k + 1, self.geometry, self.all_gather_sp, ONE_HOT_COMPUTE_CONFIG)


def select_rows(row_select: ttnn.Tensor, x: ttnn.Tensor) -> ttnn.Tensor:
    """``row_select @ x`` with a one-hot ``row_select``: the selected rows of ``x``, bit-exact."""
    return ttnn.matmul(row_select, x, compute_kernel_config=ONE_HOT_COMPUTE_CONFIG)


__all__ = ["MTPTraceGeometry", "MTPTraceEmbedSource", "ONE_HOT_COMPUTE_CONFIG", "select_rows"]
