# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-side row bookkeeping of one FLUX 3 Image Klein sample on an SP x TP mesh (pure torch, no ``ttnn``).

The reference DiT (``reference/flux3_klein/dit.py:216-228``) runs each mode's early blocks on the mode's own
rows and then concatenates the non-empty modes in ``MODES`` order ``[image | image_cond | text]`` into the
single-stage sequence. On device every mode is a separate SP-sharded tensor ``[1, L_m_pad, 3072]`` with
``L_m_pad = packing.padded_length(L_m, SP)`` (a multiple of ``32 * SP``; zero alignment rows at its end) and
``R_m = L_m_pad / SP`` tile-aligned rows per device. The single stage is the same concatenation with ALL
alignment rows moved to the end::

    G = [image_0 .. image_{L_img-1} | cond_0 .. | text_0 .. text_{L_txt-1} | pad ... pad]      # L_pad = R * SP rows
    R = sum_m R_m, L_pad = R * SP, L_real = sum_m L_m, logical_length = L_real

which is reached in one data movement (``stacks_flux3_klein.assemble_single_stage``): the per-device local
concat ``[img_d | cond_d | txt_d]`` of the mode shards is all-gathered over SP into the ring-ordered buffer::

    P = [img_0 cond_0 txt_0 | img_1 cond_1 txt_1 | ... ]        # device d's shard occupies rows [d * R, (d + 1) * R)

and device ``d`` gathers its ``R`` rows of ``G`` from ``P`` with the constant index map :meth:`assembly_indices`
(``G row (mode m, row i) -> P row (i // R_m) * R + offset_m + (i % R_m)``, ``offset_m`` = rows of the earlier
modes on one device). The ``L_pad - L_real`` pad rows of ``G`` copy one of the early stacks' zero pad rows
(``source_row`` of the FIRST pad row of the first padded mode), so the assembly needs every pad row of every
part to be exactly zero (the early stacks guarantee it with per-token ``(0, 1, 0)`` modulation).
:meth:`image_out_indices` is the reverse map after the final layer: image row ``i`` is ``G`` row ``i`` because the
image comes first; the image tail copies row 0 and is discarded by the caller.

An empty mode (text-to-image has no ``image_cond``) contributes no rows anywhere: the reference's phantom token
(``dit.py:192-198``) never enters the single stage and is never attended, so the device neither projects nor
runs a stack for it (``docs/M2_M3_DESIGN.md`` decision 5).

Mode ids (``FLUX3_KLEIN_PAD_MODE_ID`` for an alignment row) name the row kind for the per-token modulation
gathers: ``sequence_mode_ids`` in ``G`` order, ``mode_tail_ids`` per early-stack tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ....pipelines.flux3_klein.packing import FLUX3_KLEIN_MODES, Flux3KleinPacked, padded_length

# Row kinds: 0 image, 1 image_cond, 2 text (the index in FLUX3_KLEIN_MODES), 3 an alignment (pad) row.
FLUX3_KLEIN_PAD_MODE_ID = 3


def mode_id(mode: str) -> int:
    """Index of ``mode`` in ``FLUX3_KLEIN_MODES``: 0 image, 1 image_cond, 2 text (the row kind of its rows)."""
    return FLUX3_KLEIN_MODES.index(mode)


@dataclass(frozen=True)
class Flux3KleinSequenceLayout:
    """Row layout of ONE sample at sequence-parallel factor ``sp_factor``. ``lengths`` names all three
    ``FLUX3_KLEIN_MODES`` (0 = empty); everything else is derived (see the module docstring)."""

    sp_factor: int
    lengths: dict[str, int]
    # Derived in __post_init__.
    padded: dict[str, int] = None  # L_m_pad = padded_length(L_m, sp_factor)
    rows: dict[str, int] = None  # R_m = L_m_pad // sp_factor
    rows_total: int = None  # R
    total_padded: int = None  # L_pad = R * sp_factor
    total: int = None  # L_real
    modes: tuple[str, ...] = None  # non-empty modes in MODES order

    def __post_init__(self) -> None:
        if self.sp_factor < 1:
            raise ValueError(f"sp_factor must be positive, got {self.sp_factor}")
        if set(self.lengths) != set(FLUX3_KLEIN_MODES):
            raise ValueError(f"lengths must name exactly {FLUX3_KLEIN_MODES}, got {sorted(self.lengths)}")
        if any(n < 0 for n in self.lengths.values()):
            raise ValueError(f"negative length in {self.lengths}")
        lengths = {m: int(self.lengths[m]) for m in FLUX3_KLEIN_MODES}
        padded = {m: padded_length(lengths[m], self.sp_factor) for m in FLUX3_KLEIN_MODES}
        rows = {m: padded[m] // self.sp_factor for m in FLUX3_KLEIN_MODES}
        rows_total = sum(rows.values())
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "padded", padded)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "rows_total", rows_total)
        object.__setattr__(self, "total_padded", rows_total * self.sp_factor)
        object.__setattr__(self, "total", sum(lengths.values()))
        object.__setattr__(self, "modes", tuple(m for m in FLUX3_KLEIN_MODES if lengths[m] > 0))

    @classmethod
    def from_lengths(cls, lengths: dict[str, int], sp_factor: int) -> Flux3KleinSequenceLayout:
        """The layout of real lengths ``lengths`` (all three modes, 0 = empty) at sequence-parallel factor ``sp_factor``."""
        return cls(sp_factor=sp_factor, lengths=dict(lengths))

    @classmethod
    def from_packed(cls, packed: Flux3KleinPacked, sample: int, sp_factor: int) -> Flux3KleinSequenceLayout:
        """Sample ``sample`` of a :func:`packing.pack` result: lengths from ``mode_indices[m][sample]``."""
        lengths = {m: packed.mode_indices[m][sample][1] - packed.mode_indices[m][sample][0] for m in FLUX3_KLEIN_MODES}
        return cls(sp_factor=sp_factor, lengths=lengths)

    # --- the ring-ordered all-gather buffer P -----------------------------------------------------------------

    def row_offset(self, mode: str) -> int:
        """Rows of the earlier modes on one device: where ``mode``'s shard starts inside a device's local concat."""
        return sum(self.rows[m] for m in FLUX3_KLEIN_MODES[: mode_id(mode)])

    def source_row(self, mode: str, i: int) -> int:
        """``P`` row of row ``i`` (``0 <= i < padded[mode]``) of the mode's early-stack tensor."""
        if not 0 <= i < self.padded[mode]:
            raise IndexError(f"{mode} row {i} outside [0, {self.padded[mode]})")
        device, local = divmod(i, self.rows[mode])
        return device * self.rows_total + self.row_offset(mode) + local

    def _first_pad_source(self) -> int:
        """``P`` row of the first alignment row of the first padded mode (exactly zero by the stacks' contract)."""
        for m in self.modes:
            if self.padded[m] > self.lengths[m]:
                return self.source_row(m, self.lengths[m])
        raise AssertionError(f"no padded mode although total_padded {self.total_padded} > total {self.total}")

    def assembly_indices(self) -> torch.LongTensor:
        """``[total_padded]`` int64: ``P`` row of every ``G`` row (real rows in ``MODES`` order, then the pad rows)."""
        indices = [self.source_row(m, i) for m in self.modes for i in range(self.lengths[m])]
        pad = self.total_padded - self.total
        if pad > 0:
            indices.extend([self._first_pad_source()] * pad)
        return torch.tensor(indices, dtype=torch.int64)

    def image_out_indices(self) -> torch.LongTensor:
        """``[padded['image']]`` int64: ``G`` row of every image-layout row (``i -> i``; the tail copies row 0, any real
        row would do, the caller discards it)."""
        length, padded = self.lengths["image"], self.padded["image"]
        if length == 0:
            raise ValueError("a sample always has image rows")
        return torch.tensor(list(range(length)) + [0] * (padded - length), dtype=torch.int64)

    # --- row kinds --------------------------------------------------------------------------------------------

    def sequence_mode_ids(self) -> torch.LongTensor:
        """``[total_padded]`` int64 mode id per ``G`` row; ``FLUX3_KLEIN_PAD_MODE_ID`` for the pad rows."""
        ids = [mode_id(m) for m in self.modes for _ in range(self.lengths[m])]
        ids.extend([FLUX3_KLEIN_PAD_MODE_ID] * (self.total_padded - self.total))
        return torch.tensor(ids, dtype=torch.int64)

    def mode_tail_ids(self, mode: str) -> torch.LongTensor:
        """``[padded[mode]]`` int64: the mode's id for its ``L_m`` real rows, ``FLUX3_KLEIN_PAD_MODE_ID`` for the tail."""
        ids = [mode_id(mode)] * self.lengths[mode] + [FLUX3_KLEIN_PAD_MODE_ID] * (
            self.padded[mode] - self.lengths[mode]
        )
        return torch.tensor(ids, dtype=torch.int64)

    def sequence_ids(self, ids: dict[str, torch.Tensor]) -> torch.Tensor:
        """``[total, 4]`` fp32 RoPE ids of the real rows in ``G`` order: ``ids[m][:L_m]`` of the non-empty modes (equal to
        ``Flux3KleinPacked.sequence_ids()`` for a one-sample pack; pass this sample's rows for a multi-sample pack)."""
        parts = []
        for m in self.modes:
            rows = ids[m]
            if rows.shape[0] < self.lengths[m] or rows.shape[-1] != 4:
                raise ValueError(f"{m}: ids {tuple(rows.shape)} do not cover {self.lengths[m]} rows of 4")
            parts.append(rows[: self.lengths[m]].to(torch.float32))
        return torch.cat(parts)

    def tail_confined(self) -> bool:
        """True when every non-empty mode's alignment tail fits inside its LAST device's shard, the condition under
        which a per-device local concat (no all-gather) already gives a correct permuted layout on every device
        (documentation of the M10 CCL-free fast path; not used by the M3 forward). Text 80 at SP4 breaks it:
        ``padded_length(80, 4) = 128`` leaves a 48-row tail over 32-row shards."""
        return all(self.padded[m] - self.lengths[m] <= self.rows[m] for m in self.modes)
