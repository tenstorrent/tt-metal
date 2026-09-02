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

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import PackerReluType, StochasticRounding

#: Datums per row of the edge-mask geometry — each mask is 16 bits wide.
EDGE_MASK_WIDTH = 16

#: Number of selectable edge masks the hardware holds.
EDGE_MASK_COUNT = 4


class EdgeMaskMode:
    """What a masked datum becomes."""

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
    mode: int = EdgeMaskMode.ZERO

    def __post_init__(self):
        if not 1 <= len(self.masks) <= EDGE_MASK_COUNT:
            raise ValueError(
                f"expected 1..{EDGE_MASK_COUNT} masks, got {len(self.masks)}"
            )
        if any(not 0 <= m < (1 << EDGE_MASK_WIDTH) for m in self.masks):
            raise ValueError(f"each mask must fit {EDGE_MASK_WIDTH} bits")
        if self.mode not in (EdgeMaskMode.ZERO, EdgeMaskMode.NEG_SATURATE):
            raise ValueError(f"unknown edge mask mode {self.mode}")

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


def _encode_threshold(threshold: float, dest_format: DataFormat) -> float:
    """Round the ReLU threshold to the 16 bits the packer's register holds.

    The register stores the threshold in the packer's intermediate format, so a
    value the golden compares against must go through the same narrowing. The
    Float16 (exponent-A) family stores it as fp16, everything else as bf16.
    """
    if dest_format in (DataFormat.Float16, DataFormat.Bfp8):
        return float(torch.tensor(threshold, dtype=torch.float16))
    return float(torch.tensor(threshold, dtype=torch.bfloat16))


def apply_relu(
    values: torch.Tensor,
    relu_type: PackerReluType,
    threshold: float = 0.0,
    dest_format: DataFormat = DataFormat.Float16_b,
) -> torch.Tensor:
    """Packer ReLU. Mirrors ``PackGolden.apply_relu``.

    The threshold is narrowed to what the packer's register can hold first.
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


def is_deterministic(stoch_rnd: StochasticRounding) -> bool:
    """Whether the packer's rounding can be reproduced by a golden model.

    Stochastic rounding is driven by a pseudo-random sequence seeded on device,
    so its result is **not** reproducible
    here. When it is enabled the golden returns the round-to-nearest result,
    which the hardware matches only in expectation — each datum may land one ULP
    of the output format either side. Compare with PCC rather than exactly, as
    ``test_unpack_matmul`` already does.
    """
    return stoch_rnd in (StochasticRounding.No, StochasticRounding.Fpu)


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
