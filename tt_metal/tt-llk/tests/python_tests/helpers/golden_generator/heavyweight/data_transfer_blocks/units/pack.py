# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The packer: Dest -> L1.

Owns the ``Dest -> L1`` legality table, the edge-mask polarity, and the
pipeline of effects the packer applies on the way out.

See `README.md`, "units/pack.py".
"""

import warnings
from typing import FrozenSet, Mapping, Optional

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import PackerReluType, StochasticRounding

from ..l1_support import L1Support
from ..storage import (
    DEST_32_BIT_FORMATS,
    to_dest_storage,
    truncate_to_l1_lattice,
)
from .pack_effects import (
    STOCH_RND_EFFECTS,
    PackEdgeMask,
    apply_pack_effects,
    is_deterministic,
)


class UnmodelledHardwareWarning(RuntimeWarning):
    """A hardware behaviour this golden knowingly does not reproduce.

    A ``RuntimeWarning`` on purpose: ``pytest.ini`` sets ``ignore::UserWarning``
    for the whole suite, so a plain ``UserWarning`` would reach nobody running
    the tests, while ``pytest.warns`` bypasses the filter and the unit test
    would keep passing.
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
        gasket_converts_wide_dest: bool = True,
        truncating_outputs: FrozenSet[DataFormat] = frozenset(),
    ) -> None:
        #: Legal ``Dest -> L1`` pairs, the pack-side counterpart of the
        #: unpacker's table. Empty means unmodelled.
        self.to_l1_formats = to_l1_formats or {}

        #: Whether a set bit in an edge-mask register masks the datum.
        #: False on Wormhole/Blackhole, True on Quasar.
        self.edge_mask_masked_when_set = edge_mask_masked_when_set

        #: Whether the harness promotes an exponent-B input with a Float16
        #: output to a 32-bit Dest. Every architecture except Quasar.
        self.promotes_outlier_to_32_bit_dest = promotes_outlier_to_32_bit_dest

        #: Whether the gasket converts a 32-bit Dest before the packer sees it,
        #: so the packer reads the *output* format rather than Dest's. True on
        #: Wormhole and Blackhole, False on Quasar, whose packer converts
        #: directly. It decides which format the ReLU threshold is encoded in.
        self.gasket_converts_wide_dest = gasket_converts_wide_dest

        #: Output formats this packer truncates toward zero. The codecs in
        #: :mod:`helpers.pack` round, because they were written to lay down
        #: stimuli rather than to model this conversion, so a format listed
        #: here is quantized before they see it.
        self.truncating_outputs = truncating_outputs

        self.l1 = l1

    def check_pack_pair(self, dest_format: DataFormat, l1_format: DataFormat) -> None:
        """Refuse a ``Dest -> L1`` pair this architecture's packer cannot do.

        Without it a Float16 Dest packed to Float32, or an Int32 Dest packed to a
        float, returns bytes that decode to believable numbers for a conversion the
        hardware refuses. The rows are worth reading rather than inferring. See
        `README.md`, "What the packer will and will not convert".
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

    def pack_src_format(
        self, dest_format: DataFormat, l1_format: DataFormat
    ) -> DataFormat:
        """The format the packer reads out of Dest.

        Usually Dest's own. With a 32-bit Dest on an architecture whose gasket
        converts first, the packer instead reads the output format, which is
        what the ReLU threshold is then encoded in. Mirrors ``infer_pack_in``.
        """
        if self.gasket_converts_wide_dest and dest_format in DEST_32_BIT_FORMATS:
            return l1_format
        return dest_format

    def check_outlier_promotion(
        self, dest_format: DataFormat, l1_format: DataFormat
    ) -> None:
        """Refuse the 16-bit Dest the harness would have promoted away.

        ``TestConfig`` turns ``dest_acc`` on for an exponent-B input with a Float16
        output, so the device runs a 32-bit Dest. A Float16_b Dest is the tell.
        Without this the golden models bf16 while the device ran fp32, and the
        precision gap shows up as an arithmetic disagreement.
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
        """Pack Dest into L1, the T2 half that mirrors the unpacker.

        ReLU, then rounding, then the edge mask, then the format conversion.

        `dest_format` cannot be inferred from `l1_format`: Dest follows the
        *unpacker's* output, never the pack format. Use ``Unpack.dest_format_for``
        to derive it from the input side.

        `stoch_rnd` is accepted so a caller can describe the hardware it is
        comparing against, but any mode other than ``No`` **warns**, including
        ``Fpu``, which randomises the Dest write rather than this block's.
        """
        self.l1.check_dest_format(dest_format)
        self.check_pack_pair(dest_format, l1_format)
        if not is_deterministic(stoch_rnd):
            # Name the stage actually randomised: under Fpu the packer
            # behaves normally and it is the Dest write that drifts, so
            # blaming the packer would send a reader to the wrong block.
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
        # Normally already at Dest precision (src_to_dest wrote it there), so this
        # is a no-op then, and a safety net for a caller that bypassed Dest.
        values = to_dest_storage(dest_values, dest_format)
        values = apply_pack_effects(
            values,
            relu_type=relu_type,
            relu_threshold=relu_threshold,
            dest_format=self.pack_src_format(dest_format, l1_format),
            edge_mask=edge_mask,
            edge_mask_masked_when_set=self.edge_mask_masked_when_set,
        )
        if l1_format in self.truncating_outputs:
            truncated = truncate_to_l1_lattice(values, l1_format)
            if truncated is None:
                warnings.warn(
                    f"{self.l1.name} truncates when packing to {l1_format}, "
                    f"but this golden has no lattice for it, so the rounding "
                    f"codec is used instead. Those bytes are unverified.",
                    UnmodelledHardwareWarning,
                    stacklevel=3,
                )
            else:
                values = truncated
        return self.l1.write(values, l1_format, **geometry)
