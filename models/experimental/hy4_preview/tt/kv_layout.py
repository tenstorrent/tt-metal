# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where a sequence position lives in the context-parallel KV cache (tenstorrent/tt-metal#65886).

The cache is split by sequence, not by head: positions are cut into blocks of 128 tokens and the
blocks are dealt round-robin to the chips (block-cyclic), so every chip holds 1/32 of any context
and a growing context fills the chips evenly.

    block = position // 128
    chip  = block % num_chips                     chips counted row-major over the mesh, the order
                                                  ttnn.get_device_tensors returns them in
    slot  = (block // num_chips) * 128 + position % 128      the row inside that chip's stripe

This is the stripe prefill writes (tenstorrent/tt-metal#65886, KvStripe: block 128, 32 chips, linear
chip = sp_coord * tp + tp_coord): on the 8x4 mesh, with the sequence on axis 0 and the hidden on
axis 1 (tt/common.py), row-major is that linear chip, so both sides address the same rows.

Pure Python and torch, no device: this is the one contract the cache, the indexer and sparse
attention share. The methods take ints or integer tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BlockCyclicLayout:
    num_chips: int = 32
    block: int = 128

    def chip(self, position):
        """The chip (row-major index) that holds ``position``."""
        return position // self.block % self.num_chips

    def slot(self, position):
        """The row of ``position`` inside its chip's stripe."""
        return position // self.block // self.num_chips * self.block + position % self.block

    def position(self, chip, slot):
        """The inverse of (chip, slot)."""
        return (slot // self.block * self.num_chips + chip) * self.block + slot % self.block

    def slots(self, length: int) -> int:
        """Rows a chip's stripe needs to hold a context of ``length`` positions: a whole number of blocks."""
        blocks = -(-length // self.block)
        return -(-blocks // self.num_chips) * self.block

    def split(self, x: torch.Tensor, dim: int = 0, slots: int | None = None) -> torch.Tensor:
        """Stripe ``x`` along ``dim`` (its sequence dim): returns [num_chips, ...] with ``dim`` (shifted
        by the new leading dim) holding each chip's slots, zero where a slot has no position yet."""
        length = x.shape[dim]
        slots = self.slots(length) if slots is None else slots
        assert slots >= self.slots(length), (slots, length)
        positions = torch.arange(length)
        x = x.movedim(dim, 0)
        out = x.new_zeros(self.num_chips, slots, *x.shape[1:])
        out[self.chip(positions), self.slot(positions)] = x
        return out.movedim(1, dim + 1)

    def merge(self, stripes: torch.Tensor, length: int, dim: int = 0) -> torch.Tensor:
        """The inverse of split: [num_chips, ...] with the slots on ``dim + 1`` -> the first ``length`` positions."""
        positions = torch.arange(length)
        return stripes.movedim(dim + 1, 1)[self.chip(positions), self.slot(positions)].movedim(0, dim)

    def filled(self, length: int, slots: int | None = None) -> torch.Tensor:
        """[num_chips, slots] bool: which slots hold one of the first ``length`` positions."""
        slots = self.slots(length) if slots is None else slots
        return self.split(torch.ones(length, dtype=torch.bool), slots=slots)
