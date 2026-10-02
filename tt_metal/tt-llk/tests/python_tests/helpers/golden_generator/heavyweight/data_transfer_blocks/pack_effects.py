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
    """Per-datum edge masking, as the packer applies it at tile edges.

    Args:
        masks: up to ``EDGE_MASK_COUNT`` masks of ``EDGE_MASK_WIDTH`` bits. A set
            bit **keeps** the datum in that column; a clear bit masks it —
            note the polarity, the hardware masks where the bit is clear.
        select: which mask applies. An int uses one mask for every datum;
            a sequence gives a per-datum mask index, matching the hardware's
            2-bit per-datum selector.
        mode: :class:`EdgeMaskMode` — zero or negative-saturate.
    """

    masks: Sequence[int] = (0xFFFF,)
    select: Union[int, Sequence[int]] = 0
    mode: EdgeMaskMode = EdgeMaskMode.ZERO

    def __post_init__(self):
        if not 1 <= len(self.masks) <= EDGE_MASK_COUNT:
            raise ValueError(
                f"expected 1..{EDGE_MASK_COUNT} masks, got {len(self.masks)}"
            )
        if any(not 0 <= m < (1 << EDGE_MASK_WIDTH) for m in self.masks):
            raise ValueError(f"each mask must fit {EDGE_MASK_WIDTH} bits")
        # Raises for anything outside the enum, and normalises a raw int to the
        # member, so the stored value matches the annotation. object.__setattr__
        # because the dataclass is frozen.
        object.__setattr__(self, "mode", EdgeMaskMode(self.mode))

    def keep(self, count: int) -> torch.Tensor:
        """Bool tensor, True where datum *i* survives the mask."""
        if isinstance(self.select, int):
            indices = [self.select] * count
        else:
            indices = list(self.select)
            if len(indices) < count:
                raise ValueError(
                    f"select has {len(indices)} entries for {count} datums"
                )
        return torch.tensor(
            [
                bool((self.masks[indices[i]] >> (i % EDGE_MASK_WIDTH)) & 1)
                for i in range(count)
            ]
        )

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        """Replace masked datums with zero, or with negative saturation."""
        flat = values.reshape(-1)
        keep = self.keep(flat.numel())
        replacement = (
            torch.full_like(flat, float("-inf"))
            if self.mode == EdgeMaskMode.NEG_SATURATE
            else torch.zeros_like(flat)
        )
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
    swept when this was written; keep them so, or retire the other one.
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
) -> torch.Tensor:
    """ReLU then edge masking, in the order the packer applies them."""
    values = apply_relu(values, relu_type, relu_threshold, dest_format)
    if edge_mask is not None:
        values = edge_mask.apply(values)
    return values
