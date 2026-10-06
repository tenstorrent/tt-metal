# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Packer side-effects applied between Dest and the L1 write.

The packer does more than convert format. Three knobs change the values that
reach L1, and the hardware applies them in this order:

    Dest -> ReLU -> round -> edge mask -> format convert -> L1

Rounding sits between ReLU and the mask in hardware, but the mask *replaces* a
datum outright, so masking before the format conversion gives the same bytes.

Two of the three are exactly modellable. Stochastic rounding is not — see
:func:`is_deterministic`.
"""

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Sequence, Union

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import PackerReluType, StochasticRounding
from helpers.tile_constants import MAX_FACE_R_DIM

#: Datums per row of the edge-mask geometry — each mask is 16 bits wide.
EDGE_MASK_WIDTH = 16

#: Number of selectable edge masks the hardware holds.
EDGE_MASK_COUNT = 4


class EdgeMaskMode(IntEnum):
    """What a masked datum becomes.

    An ``IntEnum`` because these are hardware field values, so a caller holding
    a raw 0 or 1 still compares and packs correctly, while the names carry the
    meaning. Matches how the other mode knobs in :mod:`helpers.llk_params` are
    declared.
    """

    #: Masked datums are zeroed.
    ZERO = 0
    #: Masked datums saturate negative, so they lose a following max-reduce.
    NEG_SATURATE = 1


@dataclass(frozen=True)
class PackEdgeMask:
    """Edge masking at tile edges, as the packer's configuration registers hold it.

    Args:
        masks: up to ``EDGE_MASK_COUNT`` masks of ``EDGE_MASK_WIDTH`` bits, the raw
            register values. Bit *j* covers datum *j* of a 16-datum row. Whether a
            set bit keeps or masks that datum depends on the architecture, so the
            caller says which -- see ``masked_when_set`` on :meth:`keep`.
            Required, with no pass-everything default, because there isn't one:
            0xFFFF passes a row through on Wormhole/Blackhole but masks it on
            Quasar, whose pass-through is ``EDGE_MASK_ROW_DATUMS_NONE`` (0x0000).
            A default would read as "no masking" and mean the opposite on one of
            the two.
        select: which mask each 16-datum row uses. An int uses one mask for every
            row; a sequence gives one index per row, in the order the rows sit in
            the data. The hardware selector is 2 bits **per row**, never per datum
            (Quasar's ``EDGE_MASK_SELECT_FACE*``, Wormhole/Blackhole's
            ``TILE_ROW_SET_MAPPING``), so a row cannot mix masks.
        mode: :class:`EdgeMaskMode` -- zero or negative-saturate.
    """

    masks: Sequence[int]
    select: Union[int, Sequence[int]] = 0
    mode: EdgeMaskMode = EdgeMaskMode.ZERO
    #: Rows the selector pattern covers before it repeats, i.e. one tile's
    #: worth. The packer reuses EDGE_MASK_SELECT_FACE0..3 for every tile, so a
    #: multi-tile buffer applies the same pattern again rather than running off
    #: the end of the list. ``None`` means the pattern is the whole list.
    select_period: Optional[int] = None

    def __post_init__(self):
        if not 1 <= len(self.masks) <= EDGE_MASK_COUNT:
            raise ValueError(
                f"expected 1..{EDGE_MASK_COUNT} masks, got {len(self.masks)}"
            )
        if any(not 0 <= m < (1 << EDGE_MASK_WIDTH) for m in self.masks):
            raise ValueError(f"each mask must fit {EDGE_MASK_WIDTH} bits")
        selectors = [self.select] if isinstance(self.select, int) else self.select
        if any(not 0 <= s < len(self.masks) for s in selectors):
            raise ValueError(
                f"each selector must index one of the {len(self.masks)} masks"
            )
        # Raises for anything outside the enum, and normalises a raw int to the
        # member, so the stored value matches the annotation. object.__setattr__
        # because the dataclass is frozen.
        object.__setattr__(self, "mode", EdgeMaskMode(self.mode))

    @classmethod
    def from_face_select_words(
        cls,
        masks: Sequence[int],
        face_select_words: Sequence[int],
        mode: EdgeMaskMode = EdgeMaskMode.ZERO,
    ) -> "PackEdgeMask":
        """Build from Quasar's ``EDGE_MASK_SELECT_FACE0..3`` register words.

        Each 32-bit word holds a 2-bit selector for each of a face's 16 rows,
        row 0 in the low bits, which is the row-major order this class indexes.
        """
        select = [
            (word >> (2 * row)) & 0x3
            for word in face_select_words
            for row in range(MAX_FACE_R_DIM)
        ]
        # One word per face, so the pattern covers exactly one tile and repeats
        # for the next -- which is what the packer does with these registers.
        return cls(
            masks=masks,
            select=select,
            mode=mode,
            select_period=len(face_select_words) * MAX_FACE_R_DIM,
        )

    def keep(self, count: int, *, masked_when_set: bool = False) -> torch.Tensor:
        """Bool tensor, True where datum *i* survives the mask.

        `masked_when_set` is the architecture's polarity: True on Quasar, whose
        packer inverts the register before the gasket applies it, False on
        Wormhole/Blackhole, where a set bit passes the datum through.
        """
        if count % EDGE_MASK_WIDTH:
            raise ValueError(
                f"{count} datums is not a whole number of {EDGE_MASK_WIDTH}-datum rows"
            )
        rows = count // EDGE_MASK_WIDTH
        if isinstance(self.select, int):
            selectors = [self.select] * rows
        else:
            selectors = list(self.select)
            if self.select_period is None:
                # A hand-built list is taken literally: too short is a mistake,
                # not an invitation to repeat it.
                if len(selectors) < rows:
                    raise ValueError(
                        f"select has {len(selectors)} entries for {rows} rows"
                    )
            elif self.select_period <= 0 or rows % self.select_period:
                raise ValueError(
                    f"select covers {self.select_period} rows, which does not "
                    f"tile {rows} rows evenly"
                )
            else:
                selectors = [selectors[row % self.select_period] for row in range(rows)]
        words = torch.tensor([self.masks[s] for s in selectors[:rows]])
        columns = torch.arange(EDGE_MASK_WIDTH)
        bit_set = ((words[:, None] >> columns) & 1).bool().reshape(-1)
        return ~bit_set if masked_when_set else bit_set

    def apply(
        self, values: torch.Tensor, *, masked_when_set: bool = False
    ) -> torch.Tensor:
        """Replace masked datums with zero, or with negative saturation."""
        flat = values.reshape(-1)
        keep = self.keep(flat.numel(), masked_when_set=masked_when_set)
        if self.mode == EdgeMaskMode.NEG_SATURATE:
            if not flat.is_floating_point():
                # -inf has no integer encoding, and what the hardware writes
                # for an integer Dest is not modelled here. Nothing programs
                # this mode yet, so say so rather than guess at INT_MIN.
                raise ValueError(
                    f"{EdgeMaskMode.NEG_SATURATE.name} is not modelled for an "
                    f"integer Dest ({flat.dtype}); only {EdgeMaskMode.ZERO.name} is"
                )
            replacement = torch.full_like(flat, float("-inf"))
        else:
            replacement = torch.zeros_like(flat)
        return torch.where(keep, flat, replacement).reshape(values.shape)


#: Formats whose ReLU threshold field is read as fp16 rather than bf16.
#: The 8-bit and sub-8-bit block formats (Fp8, Bfp4a, Bfp2a) belong here too
#: once the harness supports them.
FP16_THRESHOLD_FORMATS = (DataFormat.Float16, DataFormat.Bfp8)


def _encode_threshold_bits(threshold: float, dest_format: DataFormat) -> int:
    """The 16-bit threshold field as the packer's config register stores it.

    The field is 16 bits wide whatever the pack format, so the threshold is
    narrowed to reach it -- and the two branches narrow *differently*. The
    Float16 family reads the field as fp16, so the value is **rounded** to
    fp16. Everything else reads it as bf16, which the register takes as the top
    half of the fp32 and therefore **truncates**; for a 32-bit Dest the
    hardware shifts those bits back up by 16 to rebuild the comparand.
    """
    if dest_format in FP16_THRESHOLD_FORMATS:
        return torch.tensor(threshold, dtype=torch.float16).view(torch.uint16).item()
    fp32_bits = struct.unpack(">I", struct.pack(">f", threshold))[0]
    return (fp32_bits >> 16) & 0xFFFF


def _decode_threshold_bits(bits: int, dest_format: DataFormat) -> float:
    """The value those 16 bits represent -- the inverse of the encode above."""
    if dest_format in FP16_THRESHOLD_FORMATS:
        return torch.tensor(bits, dtype=torch.uint16).view(torch.float16).item()
    return struct.unpack(">f", struct.pack(">I", (bits & 0xFFFF) << 16))[0]


def _encode_threshold(threshold: float, dest_format: DataFormat) -> float:
    """The threshold as the configuration register actually holds it.

    A ReLU compares against the register, not against the float the test asked
    for, so the golden has to narrow first. Narrowing with a plain bf16 cast
    rounds where the register truncates, which shifts the threshold a full bf16
    step for any value with mantissa bits below the cut -- 0.3 becomes 0.30078
    instead of 0.29883, and every datum between the two is then clamped
    differently.

    This duplicates the encoding in ``PackGolden`` on purpose: heavyweight owns
    its own model of the packer so the old golden can be retired without
    stranding it. The two were bit-identical across every format and threshold
    swept when this was written, and nothing enforces that they stay so -- there
    is no test comparing them, deliberately, since one would reintroduce the
    dependency. Treat the agreement as a fact about that moment, not a
    guarantee, and re-check before relying on either matching the other.
    """
    return _decode_threshold_bits(
        _encode_threshold_bits(threshold, dest_format), dest_format
    )


def apply_relu(
    values: torch.Tensor,
    relu_type: PackerReluType,
    threshold: float = 0.0,
    dest_format: DataFormat = DataFormat.Float16_b,
) -> torch.Tensor:
    """
    Takes the type and threshold directly rather than the packed 32-bit
    ``relu_config`` word, and narrows the threshold to what the register holds
    before comparing -- see :func:`_encode_threshold`.
    """
    if relu_type is PackerReluType.NoRelu:
        return values
    if relu_type is PackerReluType.ZeroRelu:
        return torch.relu(values)

    limit = _encode_threshold(threshold, dest_format)
    if relu_type is PackerReluType.MinThresholdRelu:
        # Below the threshold is flushed; above it passes through untouched.
        return torch.where(values <= limit, torch.zeros_like(values), values)
    if relu_type is PackerReluType.MaxThresholdRelu:
        return torch.clamp(values, min=0.0, max=limit)
    raise ValueError(f"unknown relu type {relu_type}")


#: Which stage each stochastic-rounding mode randomises, and therefore why the
#: golden cannot follow it. ``No`` is absent: nothing is randomised.
STOCH_RND_EFFECTS = {
    StochasticRounding.Fpu: (
        "the FPU rounds stochastically when it writes Dest, where this golden's "
        "Dest write rounds to nearest"
    ),
    StochasticRounding.Pack: (
        "the packer rounds stochastically on the way to L1, where this golden's "
        "pack rounds to nearest"
    ),
    StochasticRounding.All: (
        "both the FPU's Dest write and the packer's L1 write round "
        "stochastically, where this golden rounds to nearest at each"
    ),
}


def is_deterministic(stoch_rnd: StochasticRounding) -> bool:
    """Whether the golden can reproduce this rounding configuration exactly.

    Only ``No`` can be. Every other mode draws from a pseudo-random sequence
    seeded on device, so the golden's answer is the round-to-nearest one, which
    hardware matches only in expectation -- each datum may land one ULP either
    side. Compare with PCC rather than exactly, as ``test_unpack_matmul`` does.

    ``Fpu`` counts as nondeterministic even though the packer rounds normally
    under it: the randomness simply lands a stage earlier, in the FPU's write
    to Dest, which this golden's ``src_to_dest`` rounds to nearest. It only
    *diverges* when that write actually has to round -- a 16-bit Dest, or an
    accumulation long enough for the error to build -- which is why
    ``matmul_sweep.skip_matmul_combination`` skips only the bf16 /
    ``DestAccumulation.No`` / ``kt_dim >= 4`` corner rather than every ``Fpu``
    variant. A yes/no answer has no room for that, so it answers conservatively
    and leaves the narrower judgement to the caller: a test that knows its
    Dest write is exact can use an exact compare under ``Fpu`` anyway.
    """
    return stoch_rnd is StochasticRounding.No


def apply_pack_effects(
    values: torch.Tensor,
    *,
    relu_type: PackerReluType = PackerReluType.NoRelu,
    relu_threshold: float = 0.0,
    dest_format: DataFormat = DataFormat.Float16_b,
    edge_mask: Optional[PackEdgeMask] = None,
    edge_mask_masked_when_set: bool = False,
) -> torch.Tensor:
    """ReLU then edge masking, in the order the packer applies them.

    `edge_mask_masked_when_set` is the architecture's mask polarity -- see
    :meth:`PackEdgeMask.keep`.
    """
    values = apply_relu(values, relu_type, relu_threshold, dest_format)
    if edge_mask is not None:
        values = edge_mask.apply(values, masked_when_set=edge_mask_masked_when_set)
    return values
