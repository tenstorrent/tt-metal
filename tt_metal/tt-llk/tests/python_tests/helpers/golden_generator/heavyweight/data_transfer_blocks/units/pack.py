# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The packer: Dest -> L1.

Owns the ``Dest -> L1`` legality table, the edge-mask polarity, and the
pipeline of effects the packer applies on the way out.
"""

import warnings
from typing import FrozenSet, Mapping, Optional

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import PackerReluType, StochasticRounding

from ..l1_support import L1Support
from ..pack_effects import (
    STOCH_RND_EFFECTS,
    PackEdgeMask,
    apply_pack_effects,
    is_deterministic,
)
from ..storage import to_dest_storage


class UnmodelledHardwareWarning(RuntimeWarning):
    """A hardware behaviour this golden knowingly does not reproduce.

    A ``RuntimeWarning`` subclass on purpose. ``pytest.ini`` sets
    ``ignore::UserWarning`` for the whole suite, so a plain ``UserWarning``
    reaches nobody running the tests -- and that is the only audience for a
    warning whose message is "this answer is approximate". ``pytest.warns``
    bypasses the filter, so the unit test would keep passing while the suite
    stayed silent.
    """


class Pack:
    """What the packer does to Dest on the way to L1."""

    def __init__(
        self,
        l1: L1Support,
        *,
        to_l1_formats: Optional[Mapping[DataFormat, FrozenSet[DataFormat]]] = None,
        edge_mask_masked_when_set: bool = False,
        promotes_outlier_to_32_bit_dest: bool = True,
    ) -> None:
        #: Legal ``Dest format -> L1 format`` pairs, the pack-side counterpart
        #: of the unpacker's pair table. The packer converts on the way out,
        #: but not between arbitrary pairs: a 32-bit Dest reaches the narrow
        #: floats, a 16-bit float Dest does not reach Float32, and the integer
        #: widths each reach only their own. Empty means unmodelled.
        self.to_l1_formats = to_l1_formats or {}

        #: Whether a set bit in an edge-mask register masks the datum. False is
        #: Wormhole/Blackhole, where ``PCK_EDGE_OFFSET`` mask 0xFFFF passes a
        #: row through and 0x0 clears it; Quasar is the other way round.
        self.edge_mask_masked_when_set = edge_mask_masked_when_set

        #: Whether the harness promotes an exponent-B input with a Float16
        #: output to a 32-bit Dest. True for every architecture except Quasar,
        #: which ``TestConfig`` names explicitly.
        self.promotes_outlier_to_32_bit_dest = promotes_outlier_to_32_bit_dest

        self.l1 = l1

    def check_pack_pair(self, dest_format: DataFormat, l1_format: DataFormat) -> None:
        """Refuse a ``Dest -> L1`` pair this architecture's packer cannot do.

        The unpack side has ``Unpack.is_valid_src_format`` for the mirror of
        this. Without it a Float32 Dest packed to Float16_b, or an Int32 Dest
        packed to a float, returns bytes that decode to believable numbers for
        a conversion the hardware refuses to perform.
        """
        if not self.to_l1_formats:
            return
        allowed = self.to_l1_formats.get(dest_format, frozenset())
        if l1_format not in allowed:
            raise ValueError(
                f"{self.l1.name} cannot pack a {dest_format} Dest to "
                f"{l1_format}. From {dest_format} the packer reaches "
                f"{sorted(str(f) for f in allowed) or 'nothing'}."
            )

    def check_outlier_promotion(
        self, dest_format: DataFormat, l1_format: DataFormat
    ) -> None:
        """Refuse the 16-bit Dest the harness would have promoted away.

        An exponent-B input packed to Float16 is the combination
        ``is_format_combination_outlier`` names: the hardware cannot convert an
        8-bit-exponent datum straight to Float16, so ``TestConfig`` turns
        ``dest_acc`` on and runs a 32-bit Dest instead. A Float16_b Dest is the
        tell, since that is what an exponent-B input resolves to without
        accumulation.

        Without this the golden models a bf16 Dest while the device ran fp32 --
        a real precision difference, presented as an arithmetic disagreement.
        """
        if (
            self.promotes_outlier_to_32_bit_dest
            and dest_format is DataFormat.Float16_b
            and l1_format is DataFormat.Float16
        ):
            raise ValueError(
                f"{self.l1.name} never runs a {dest_format} Dest packed "
                f"to {l1_format}: an 8-bit-exponent input with a Float16 output "
                f"is the outlier TestConfig promotes to dest_acc=Yes, so the "
                f"device uses a 32-bit Dest. Pass dest_acc=True and a Float32 "
                f"dest_format to match it."
            )

    def dest_to_l1(
        self,
        dest_values: torch.Tensor,
        l1_format: DataFormat,
        dest_format: DataFormat = DataFormat.Float32,
        *,
        relu_type: PackerReluType = PackerReluType.NoRelu,
        relu_threshold: float = 0.0,
        edge_mask: Optional[PackEdgeMask] = None,
        stoch_rnd: StochasticRounding = StochasticRounding.No,
        **geometry,
    ) -> list:
        """Pack Dest into L1 -- the T2 half, mirroring the unpacker.

        The packer's pipeline, in hardware order:

        1. **Dest storage precision.** What a Dest slot can hold, which is
           32-bit under accumulation and 16-bit otherwise. Idempotent if the
           FPU block upstream already applied it, so it is safe to hand this a
           plain fp32 result from a torch golden.
        2. **ReLU**. The threshold is narrowed to the 16 bits the packer's
           configuration register holds before it is compared against.
        3. **Edge mask**, zeroing or negative-saturating datums at tile edges.
        4. **The packer.** Rounding and requantization into `l1_format` are
           whatever :mod:`helpers.pack` implements -- the same codec the test
           harness writes L1 with.

        `dest_format` cannot be inferred from `l1_format`: Dest follows the
        *unpacker's* output, i.e. the input format, never the pack format. Use
        ``Unpack.dest_format_for`` to derive it from the input side.

        `stoch_rnd` is accepted so a caller can describe the hardware
        configuration it is comparing against, but any mode other than ``No``
        **warns** rather than being silently ignored -- including ``Fpu``, which
        randomises the Dest write rather than this block's. See
        :func:`..pack_effects.is_deterministic`.
        """
        self.l1.check_dest_format(dest_format)
        self.check_pack_pair(dest_format, l1_format)
        if not is_deterministic(stoch_rnd):
            # The golden cannot follow a device-seeded random sequence, so say
            # so at the call site rather than hand back round-to-nearest bytes
            # that look like a reproducible answer. Name the stage that is
            # actually randomised: under Fpu the packer behaves normally and it
            # is the Dest write that drifts, so blaming the packer would send a
            # reader looking in the wrong block. warnings dedupes per location,
            # so a multi-tile run reports this once.
            warnings.warn(
                f"{stoch_rnd} cannot be reproduced -- "
                f"{STOCH_RND_EFFECTS[stoch_rnd]}, driven by a device-seeded "
                f"pseudo-random sequence. These bytes are the round-to-nearest "
                f"result, which hardware matches only in expectation: each datum "
                f"may land one ULP of {l1_format} either side. Compare with PCC, "
                f"not exactly.",
                UnmodelledHardwareWarning,
                stacklevel=2,
            )
        self.check_outlier_promotion(dest_format, l1_format)
        # Normally already at Dest precision (src_to_dest wrote it there); this
        # is a no-op then, and a safety net for a caller that bypassed Dest.
        values = to_dest_storage(dest_values, dest_format)
        values = apply_pack_effects(
            values,
            relu_type=relu_type,
            relu_threshold=relu_threshold,
            dest_format=dest_format,
            edge_mask=edge_mask,
            edge_mask_masked_when_set=self.edge_mask_masked_when_set,
        )
        return self.l1.write(values, l1_format, **geometry)
