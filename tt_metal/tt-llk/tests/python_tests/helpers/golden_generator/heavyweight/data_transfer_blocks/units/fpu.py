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
            values = self._accumulate(values, current, dest_format)
        return to_dest_storage(values, dest_format)

    @staticmethod
    def _accumulate(
        values: torch.Tensor, current: torch.Tensor, dest_format: DataFormat
    ) -> torch.Tensor:
        """Add a new result to what Dest holds, in a type exact for both.

        Three cases, because the operands are not always the same kind:

        * **Both integer** -- an integer op, whose operands the unpacker landed
          as integers and whose result stayed integer. Add as integers, which
          is what the device does. ``int64`` because the narrow types *wrap*:
          ``int8`` 100 + 100 is -56, and the saturation below would then clamp
          the wrong number.
        * **A float result into an integer Dest** -- reachable from a matmul on
          integer inputs, and from a caller naming an integer ``dest_format``
          explicitly. ``float64``, because float32 holds integers exactly only
          to 2**24 and an Int32 Dest reaches 2**31; a value past that would
          lose its low bits here, before the narrow, so ``16777217 + 1`` would
          come back 16777216.
        * **A float Dest** -- ``float32``, the width the rest of the float path
          uses.

        The narrowing and the saturation happen once, afterwards, in
        :func:`..storage.to_dest_storage`.
        """
        both_integer = (
            not values.dtype.is_floating_point and not current.dtype.is_floating_point
        )
        if both_integer:
            wide = torch.int64
        elif dest_format.is_integer():
            wide = torch.float64
        else:
            wide = torch.float32
        return current.to(wide) + values.to(wide)

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
