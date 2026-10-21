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
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    global_to_local_token_id,
    mtp_lookahead_positions,
    rotated_chip_row,
    rotated_row_of_position,
)
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.device_windows import MTPSplitChipLookahead, MTPUnionEmbedding

# The one-hot matmuls stand in for slices, so they must copy rows bit-exactly: HiFi4 keeps every
# mantissa bit of the copied operand and the fp32 dest holds it until the bf16 pack.
ONE_HOT_COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def _generate_only(name: str) -> bool:
    return name in ("keep", "rows", "col", "lm_rows")


class MTPTraceGeometry:
    """One chunk's MTP geometry as persistent ``[sp, 1, rows, cols]`` tensors, one row block per SP chip.

    - ``keep``: zero on the rows generation writes, applied once before level 0. Clearing them that early
      matches the eager clear at level ``provided_levels``: a level below it never reads those rows.
    - ``selects[k]``: level ``k``'s one-hot patch selector, all zero for a provided level.
    - ``lm_rows``: one-hot ``[32, W]`` copying the chunk's last real row into all 32 rows.
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
        # Split-chip lookahead of the last written chunk, or None.
        self.split = None
        # Plan key per tensor; write() skips a tensor whose key is unchanged.
        self._keys: dict = {}
        plan = self._plan(0, self.chunk_size, self.num_levels)
        for name in self._names():
            self._keys[name] = self._key(name, plan)
            self._tensors[name] = self._device(self._build(name, plan, {}))
        width = ttnn.TILE_SIZE * self.sp_factor
        self._selects = [
            self._device(torch.zeros(self.sp_factor, 1, self.union_len, width)) for _ in range(self.num_levels)
        ]
        # Constant: level_pick[k] keeps row k of "col".
        tile = ttnn.TILE_SIZE
        self._level_pick, self._level_cols = [], []
        for k in range(self.num_levels):
            pick = torch.zeros(self.sp_factor, 1, tile, tile)
            pick[:, 0, k, k] = 1.0
            self._level_pick.append(self._device(pick))
            self._level_cols.append(self._device(torch.zeros(self.sp_factor, 1, tile, width)))
        # Compiles the select programs before capture.
        self._build_selects(range(self.num_levels))

    @property
    def has_split_chip(self) -> bool:
        return self.sp_factor > 1

    @property
    def keep(self) -> ttnn.Tensor:
        return self._tensors["keep"]

    @property
    def selects(self) -> list:
        return self._selects

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
        """Rewrite the tensors whose plan key changed for ``[actual_start, actual_end)``. Never inside a capture.

        A provided chunk leaves the generate-only tensors and the selects stale; its replay does not read them.
        """
        plan = self._plan(actual_start, actual_end, provided_levels)
        self.split = plan["split"]
        generates = provided_levels < self.num_levels
        built, written = {}, set()
        for name in self._names():
            if not generates and _generate_only(name):
                continue
            key = self._key(name, plan)
            if key == self._keys[name]:
                continue
            ttnn.copy_host_to_device_tensor(self._host_tensor(self._build(name, plan, built)), self._tensors[name])
            self._keys[name] = key
            written.add(name)
        self._build_selects(range(self.num_levels) if written & {"rows", "col"} else ())

    def deallocate(self) -> None:
        for t in list(self._tensors.values()) + self._selects + self._level_pick + self._level_cols:
            ttnn.deallocate(t)
        self._tensors, self._selects, self._level_pick, self._level_cols = {}, [], [], []

    def _mapper(self):
        return ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=self.mesh_shape, dims=(self.sp_axis, None))

    def _host_tensor(self, host: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._mapper())

    def _device(self, host: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            host,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._mapper(),
        )

    def _build_selects(self, levels) -> None:
        # select[k] = rows @ (level_pick[k] @ col); one-hot operands keep it exact.
        for k in levels:
            ttnn.matmul(
                self._level_pick[k],
                self._tensors["col"],
                compute_kernel_config=ONE_HOT_COMPUTE_CONFIG,
                optional_output_tensor=self._level_cols[k],
            )
            ttnn.matmul(
                self._tensors["rows"],
                self._level_cols[k],
                compute_kernel_config=ONE_HOT_COMPUTE_CONFIG,
                optional_output_tensor=self._selects[k],
            )

    def _names(self) -> list:
        names = ["keep", "rows", "col", "lm_rows"]
        if self.has_split_chip:
            names += [f"win_keep{k}" for k in range(self.num_levels)]
            names += [f"win_place{k}" for k in range(self.num_levels)]
            names += ["next_select", "take_tail", "take_next"]
        return names

    def _key(self, name: str, plan: dict):
        if name in ("keep", "rows"):
            return plan["landing_key"]
        if name == "col":
            return (tuple(sorted(plan["landing"])), plan["source_row"])
        if name == "lm_rows":
            return plan["last_local_row"]
        return plan["split"]

    def _generation_rows(self, actual_start: int, actual_end: int, levels) -> dict:
        """Per level, the (chips, union rows) position ``actual_end + level`` lands on (closed form)."""
        w = self.window_len
        slots = mtp_lookahead_positions(actual_start, self.sp_factor, w, actual_end, self.num_levels)
        assert all(len(s) <= self.num_mtp_tokens for s in slots), f"lookahead slots do not fit in {self.num_mtp_tokens}"
        out = {}
        for level in levels:
            pos, chips, rows = actual_end + level, [], []
            for c, chip_slots in enumerate(slots):
                row = rotated_chip_row(actual_start, self.sp_factor, w, c, pos)
                if row is not None:
                    chips.append(c), rows.append(row)
                elif chip_slots[0] <= pos < chip_slots[0] + len(chip_slots):
                    chips.append(c), rows.append(w + pos - chip_slots[0])
            assert chips, f"no chip holds global position {pos}: chunk_start={actual_start} level={level}"
            out[level] = (torch.tensor(chips), torch.tensor(rows))
        return out

    def _plan(self, actual_start: int, actual_end: int, provided_levels: int) -> dict:
        """The chunk's decisions; every tensor is a function of one of them."""
        k_levels, tile, w = self.num_levels, ttnn.TILE_SIZE, self.window_len
        assert 0 <= provided_levels <= k_levels, f"provided_levels {provided_levels} outside [0, {k_levels}]"
        assert k_levels <= tile, f"rows packs one level per column of a {tile}-wide tile; K={k_levels}"
        generated = range(provided_levels, k_levels)
        landing = self._generation_rows(actual_start, actual_end, generated) if generated else {}
        last_local_row, source_row = 0, 0
        if generated:
            last_row = rotated_row_of_position(actual_start, self.sp_factor, w, actual_end - 1)
            assert (
                last_row is not None
            ), f"the chunk at {actual_start} does not carry its own last real position {actual_end - 1}"
            device_id, local = global_to_local_token_id(last_row, self.sp_factor, self.chunk_size, is_balanced=False)
            last_local_row = local
            source_row = device_id * tile + local % tile
        split = MTPSplitChipLookahead.geometry(
            actual_start, w, self.sp_factor, chunk_end=actual_end, num_levels=k_levels
        )
        return dict(
            landing=landing,
            landing_key=tuple((k, tuple(c.tolist()), tuple(r.tolist())) for k, (c, r) in sorted(landing.items())),
            last_local_row=last_local_row,
            source_row=source_row,
            split=split,
        )

    def _build(self, name: str, plan: dict, built: dict) -> torch.Tensor:
        """Host tensor ``name`` for ``plan``; ``built`` caches the window group, which shares one key."""
        sp, k_levels, tile, w = self.sp_factor, self.num_levels, ttnn.TILE_SIZE, self.window_len
        width = tile * sp
        if name == "keep":
            keep = torch.ones(sp, 1, self.union_len, 1)
            for chips, rows in plan["landing"].values():
                keep[chips, 0, rows, 0] = 0.0
            return keep
        if name == "rows":
            rows_t = torch.zeros(sp, 1, self.union_len, tile)
            for k, (chips, rows) in plan["landing"].items():
                rows_t[chips, 0, rows, k] = 1.0
            return rows_t
        if name == "col":
            col = torch.zeros(sp, 1, tile, width)
            col[:, 0, list(plan["landing"]), plan["source_row"]] = 1.0
            return col
        if name == "lm_rows":
            lm_rows = torch.zeros(sp, 1, tile, w)
            lm_rows[:, 0, :, plan["last_local_row"]] = 1.0
            return lm_rows
        if not built:
            built.update(self._build_window(plan["split"]))
        return built[name]

    def _build_window(self, split) -> dict:
        sp, k_levels, tile, w = self.sp_factor, self.num_levels, ttnn.TILE_SIZE, self.window_len
        width = tile * sp
        win_keep = [torch.ones(sp, 1, w, 1) for _ in range(k_levels)]
        win_place = [torch.zeros(sp, 1, w, tile) for _ in range(k_levels)]
        next_select = torch.zeros(sp, 1, tile, width)
        take_tail, take_next = torch.zeros(sp, 1, 1, 1), torch.zeros(sp, 1, 1, 1)
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
        out = {f"win_keep{k}": win_keep[k] for k in range(k_levels)}
        out.update({f"win_place{k}": win_place[k] for k in range(k_levels)})
        out.update(next_select=next_select, take_tail=take_tail, take_next=take_next)
        return out

    def _host(self, actual_start: int, actual_end: int, provided_levels: int) -> dict:
        """Every tensor the replay reads, built on host (the test reference)."""
        plan = self._plan(actual_start, actual_end, provided_levels)
        built: dict = {}
        out = {name: self._build(name, plan, built) for name in self._names()}
        for k in range(self.num_levels):
            pick = torch.zeros(self.sp_factor, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE)
            pick[:, 0, k, k] = 1.0
            out[f"select{k}"] = out["rows"] @ (pick @ out["col"])
        return out


class MTPTraceEmbedSource:
    """``TtMTPPredictor.forward``'s ``embeds`` callable with no per-chunk branch.

    Every level runs the generation chain and patches the union; ``geometry`` zeroes the patch of a
    provided level, so the eager result is reproduced with one program for every chunk.

    With ``embed_fn=None`` it serves chunks whose levels are all provided: generation and the patch are
    skipped, which is what the masks would reduce them to. ``blend=False`` also skips the split-chip
    blend (the identity on chip-aligned chunks).
    """

    def __init__(
        self, union: MTPUnionEmbedding, geometry: MTPTraceGeometry, embed_fn, all_gather_sp=None, blend: bool = True
    ):
        assert union.num_levels == geometry.num_levels, (
            f"union carries {union.num_levels} levels, geometry {geometry.num_levels}; "
            "the runner and the runtime disagree on PREFILL_MTP_LEVELS"
        )
        assert not geometry.has_split_chip or all_gather_sp is not None, "the split-chip lookahead needs all_gather_sp"
        self.union = union
        self.geometry = geometry
        self.embed_fn = embed_fn
        self.all_gather_sp = all_gather_sp
        self.blend = blend
        self._next_level = 0

    @property
    def generated_tokens(self) -> list:
        """Always empty: the generated ids never leave the device. Kept for interface parity."""
        return []

    def __call__(self, k: int, prev_normed) -> ttnn.Tensor:
        assert k == self._next_level, f"levels must run in order; expected {self._next_level}, got {k}"
        self._next_level += 1
        if self.embed_fn is not None:
            if k == 0:
                self.union.clear_rows(self.geometry.keep)
            gathered = self.embed_fn(prev_normed)
            self.union.add_patch(self.geometry.selects[k], gathered)
            ttnn.deallocate(gathered)
        if not self.blend:
            return self.union.window(k + 1)
        return self.union.blended_window(k + 1, self.geometry, self.all_gather_sp, ONE_HOT_COMPUTE_CONFIG)


def select_rows(row_select: ttnn.Tensor, x: ttnn.Tensor) -> ttnn.Tensor:
    """``row_select @ x`` with a one-hot ``row_select``: the selected rows of ``x``, bit-exact."""
    return ttnn.matmul(row_select, x, compute_kernel_config=ONE_HOT_COMPUTE_CONFIG)


__all__ = ["MTPTraceGeometry", "MTPTraceEmbedSource", "ONE_HOT_COMPUTE_CONFIG", "select_rows"]
