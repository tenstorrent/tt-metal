# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The FPU: src registers -> Dest, and Dest back to a src register.

The maths itself belongs to the operation; what belongs here is where the
result lands and what the trip costs. Both directions are modelled because both
lose precision: Dest is narrower than the FPU's internal result, and a src
register is narrower than Dest.
"""

from typing import Optional

import torch
from helpers.format_config import DataFormat

from ..l1_support import L1Support
from ..storage import dest_to_src_storage, to_dest_storage


class Fpu:
    """What the FPU's writes cost, on the way into Dest and back out of it."""

    def __init__(self, l1: L1Support) -> None:
        self.l1 = l1

    def src_to_dest(
        self,
        values: torch.Tensor,
        dest_format: DataFormat = DataFormat.Float32,
        current: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Write a math result into Dest.

        A Dest slot holds only what `dest_format` can represent, and the value
        is rounded on the way *in*, not on the way out to L1.

        `current` is Dest's existing contents, which a multi-pass op passes to
        accumulate rather than replace -- the FPU has an accumulate-enable bit,
        not a second instruction. Because the rounding happens on this write, it
        happens on every pass; accumulating at full precision and rounding once
        at the end would model an accumulator the hardware does not have.
        """
        self.l1.check_dest_format(dest_format)
        if current is not None:
            # An integer Dest accumulates exactly on the device. float32 holds
            # integers only to 2^24, so a Dest value past that loses its low
            # bits here, before the narrow -- 16777217 + 1 would come back as
            # 16777216. float64 covers Int32's whole range exactly.
            wide = torch.float64 if dest_format.is_integer() else torch.float32
            values = current.to(wide) + values.to(wide)
        return to_dest_storage(values, dest_format)

    def dest_to_src(
        self,
        dest_values: torch.Tensor,
        dest_format: DataFormat = DataFormat.Float32,
        src_format: DataFormat = DataFormat.Float16_b,
    ) -> torch.Tensor:
        """Move Dest back into a src register, for an op that reuses its result.

        Dest is wider than a src register, so the round trip is lossy --
        modelling that loss where the hardware takes it is the point of having
        this block. SrcA and SrcB share the datum layout, so one method serves
        both. See :func:`..storage.dest_to_src_storage` for what each Dest
        format costs.
        """
        return dest_to_src_storage(dest_values, dest_format, src_format)
