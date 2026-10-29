# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The unpacker: L1 -> SrcA/SrcB/SrcS/Dest.

Owns the two legality tables that vary by architecture, the L1 -> src format
mapping, and the transfers themselves.

It also derives the **Dest format**, which belongs here because Dest takes the
format the unpacker landed in the src register, never the pack format.

See `README.md`, "units/unpack.py".
"""

from typing import FrozenSet, Mapping, Optional, Union

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation

from ..l1_support import L1Buffer, L1Support
from ..storage import (
    DEST_32_BIT_FORMATS,
    DEST_MANTISSA_BITS,
    INT32_DEST_COMPATIBLE,
    SRC_MANT_BITS,
    SRC_MANTISSA_BITS,
    SRC_STORAGE_FORMATS,
    SRCS_STORAGE_FORMATS,
    as_dest_acc,
    to_dest_storage,
    to_src_storage,
    truncate_mantissa,
)


class Unpack:
    """Values as each register sees them after the unpacker has moved them."""

    def __init__(
        self,
        l1: L1Support,
        *,
        to_src_formats: Optional[Mapping[DataFormat, FrozenSet[DataFormat]]] = None,
        to_dest_formats: Optional[Mapping[DataFormat, FrozenSet[DataFormat]]] = None,
        has_srcs: bool = False,
        src_mantissa_bits: Optional[Mapping[DataFormat, int]] = None,
    ) -> None:
        #: Legal L1 -> src-register pairs. Empty means unmodelled at this
        #: level of detail, not that everything is legal.
        self.to_src_formats = to_src_formats or {}

        #: Legal L1 -> Dest pairs, a different table. The unpacker does not
        #: widen, so a narrow input has no 32-bit Dest target.
        self.to_dest_formats = to_dest_formats or {}

        #: Whether this architecture has a SrcS register. Quasar only.
        self.has_srcs = has_srcs

        #: Explicit mantissa bits a src register keeps per format. Overrides
        #: the defaults in :data:`..storage.SRC_MANTISSA_BITS`, which Quasar
        #: needs because its unpacker aliases Float16_b onto Tf32.
        self.src_mantissa_bits = {**SRC_MANTISSA_BITS, **(src_mantissa_bits or {})}

        self.l1 = l1

    # ------------------------------------------------------------------
    # Format mapping
    # ------------------------------------------------------------------

    def src_mantissa(self, src_format: DataFormat) -> int:
        """Explicit mantissa bits a src register keeps for `src_format` here."""
        return self.src_mantissa_bits.get(src_format, SRC_MANT_BITS)

    def src_format(self, l1_format: DataFormat) -> DataFormat:
        """The src-register storage format the unpacker lands `l1_format` in.

        Raises if this architecture has no such L1 format: the mapping is derived
        from exponent-family predicates that answer for every ``DataFormat``, so
        without the check it would answer for a format the hardware cannot read.
        """
        self.l1.check(l1_format)
        return self._src_format(l1_format)

    def _src_format(self, l1_format: DataFormat) -> DataFormat:
        """The mapping itself, unchecked.

        A format reaching the final pass-through stays as itself, which is only
        right for integers. A float landing there needs a case adding. See
        ``Fp8_e4m3``.
        """
        if l1_format in (DataFormat.Float32, DataFormat.Tf32):
            return DataFormat.Tf32
        if l1_format.is_mx_format():
            # The unpacker converts MX into the 8-bit-exponent family whatever
            # the pack format, so math and Dest see bf16.
            return DataFormat.Float16_b
        if l1_format is DataFormat.Fp8_e4m3:
            # L1-only, widened to Float16 in the register. Named explicitly
            # because it reports neither exponent family, so it would
            # otherwise fall through and stay Fp8_e4m3 in a src register.
            return DataFormat.Float16
        if l1_format.is_exponent_A():
            return DataFormat.Float16
        if l1_format.is_exponent_B():
            return DataFormat.Float16_b
        return l1_format

    def srcs_format(
        self, l1_format: DataFormat, dest_acc: Union[bool, DestAccumulation] = False
    ) -> DataFormat:
        """The storage format the unpacker lands `l1_format` in for SrcS.

        SrcS does not narrow Float32 to Tf32 the way SrcA/SrcB must, and with
        accumulation on it widens instead. Integers pass through unchanged, which
        is a deliberate divergence from ``infer_unpack_out``. See `README.md`,
        "SrcS and the integer divergence".
        """
        self.check_has_srcs()
        self.l1.check(l1_format)
        dest_acc = as_dest_acc(dest_acc)
        if l1_format.is_mx_format():
            return DataFormat.Float16_b
        if l1_format is DataFormat.Float32:
            return DataFormat.Float32
        if l1_format.is_integer():
            return l1_format
        if dest_acc:
            if l1_format in (DataFormat.Float16, DataFormat.Float16_b):
                return l1_format
            return DataFormat.Float32
        return self._src_format(l1_format)

    # ------------------------------------------------------------------
    # Legality
    # ------------------------------------------------------------------

    def check_has_srcs(self) -> None:
        if not self.has_srcs:
            raise ValueError(
                f"{self.l1.name} has no SrcS register, so there is no such "
                f"transfer to model. SrcS is Quasar-only. Use "
                f"l1_to_srcA/l1_to_srcB, or l1_to_dest for the unpack-to-Dest "
                f"path."
            )

    def is_valid_src_format(
        self, l1_format: DataFormat, src_format: DataFormat
    ) -> bool:
        """Whether this unpacker can land `l1_format` as `src_format`.

        The pair matters, not just the target: a real src format and a readable L1
        format do not make the conversion between them something the unpacker can
        do. An architecture with an empty table gets the older, weaker check. SrcS
        has its own rules in :meth:`is_valid_srcs_format`.
        """
        # No 32-bit datum reaches SrcA/SrcB on any architecture here, table
        # or no table. Without this the fallback's `src_format == l1_format`
        # arm would hand back int32 values for a register that cannot hold
        # them.
        if src_format in (DataFormat.Float32, DataFormat.Int32, DataFormat.UInt32):
            return False
        if self.to_src_formats:
            return src_format in self.to_src_formats.get(l1_format, frozenset())
        return src_format in SRC_STORAGE_FORMATS or src_format == l1_format

    @staticmethod
    def is_valid_srcs_format(
        l1_format: DataFormat, src_format: DataFormat, dest_acc: bool
    ) -> bool:
        """Whether an explicit SrcS storage format is one the unpacker can produce."""
        if src_format not in SRCS_STORAGE_FORMATS and src_format != l1_format:
            return False
        if dest_acc and l1_format in (DataFormat.Float16, DataFormat.Float16_b):
            return src_format == l1_format
        return True

    # ------------------------------------------------------------------
    # Transfers
    # ------------------------------------------------------------------

    def l1_to_src(
        self,
        l1_bytes: L1Buffer,
        l1_format: DataFormat,
        src_format: Optional[DataFormat] = None,
        **geometry,
    ) -> torch.Tensor:
        """Read L1, then apply src-register storage precision.

        Which src formats an L1 format can land in is per-architecture and
        enforced. Within that, `src_format` defaults to the one the LLK would pick.
        """
        # Validate whichever format we end up with. Checking only the
        # explicit one would exempt the common path, and the default comes
        # from the input rather than from what the unpacker can do.
        defaulted = src_format is None
        if defaulted:
            # Same ambiguity dest_format_for refuses: with dest_acc off the
            # family comes from the *output* format, and both choices clip
            # where Tf32 would not. With dest_acc on it really is Tf32.
            if l1_format in (
                DataFormat.Float32,
                DataFormat.Tf32,
            ) and not as_dest_acc(geometry.get("dest_acc", False)):
                raise ValueError(
                    f"{l1_format} input with dest_acc off does not determine "
                    f"the src register family: the unpacker lands it as "
                    f"Float16 or Float16_b depending on the *output* format "
                    f"(infer_unpack_out), and both clip where Tf32 would not. "
                    f"Pass the src_format the harness configured, or "
                    f"dest_acc=True, where the device uses Tf32."
                )
            src_format = self.src_format(l1_format)
        if not self.is_valid_src_format(l1_format, src_format):
            if src_format is DataFormat.Float32:
                raise ValueError(
                    "Float32 reaches a src register only as Tf32, Float16 or "
                    "Float16_b -- a 19-bit datum cannot hold it. Use "
                    "l1_to_dest for the unpack-to-Dest path, or l1_to_srcS, "
                    "which are the two destinations that keep full fp32."
                )
            legal = sorted(
                str(f) for f in self.to_src_formats.get(l1_format, frozenset())
            )
            if defaulted:
                raise ValueError(
                    f"{self.l1.name} cannot unpack {l1_format} into SrcA/SrcB"
                    + (
                        f"; the legal src formats for it are {legal}."
                        if legal
                        else " at all -- it reaches Dest or SrcS only. Use "
                        "l1_to_dest, or l1_to_srcS."
                    )
                )
            raise ValueError(
                f"{self.l1.name} cannot unpack {l1_format} as {src_format}. "
                + (
                    f"Legal src formats for {l1_format}: {legal}."
                    if legal
                    else f"A src register can hold "
                    f"{sorted(str(f) for f in SRC_STORAGE_FORMATS)}."
                )
            )
        values = self.l1.read(l1_bytes, l1_format, **geometry)
        return to_src_storage(values, src_format, self.src_mantissa(src_format))

    def l1_to_srcs(
        self,
        l1_bytes: L1Buffer,
        l1_format: DataFormat,
        src_format: Optional[DataFormat] = None,
        *,
        dest_acc: Union[bool, DestAccumulation] = False,
        **geometry,
    ) -> torch.Tensor:
        """Values visible in SrcS.

        SrcS uses a per-slice L1 layout, so the buffer must have been packed with
        ``use_srcs=True``. `dest_acc` selects the storage format, while the slice layout
        follows the resolved src format's width instead.

        Not a delegate of :meth:`l1_to_src`: under SrcA's rules a Float32 input
        would lose the 13 mantissa bits SrcS keeps.
        """
        self.check_has_srcs()
        self.l1.check(l1_format)
        dest_acc = as_dest_acc(dest_acc)
        if src_format is None:
            src_format = self.srcs_format(l1_format, dest_acc)
        elif not self.is_valid_srcs_format(l1_format, src_format, dest_acc):
            raise ValueError(
                f"{self.l1.name} cannot unpack {l1_format} into SrcS as "
                f"{src_format} with dest_acc={dest_acc}. SrcS holds "
                f"{sorted(str(f) for f in SRCS_STORAGE_FORMATS)} or the input "
                f"format itself, and with dest_acc a Float16/Float16_b input "
                f"stays as itself."
            )
        geometry.setdefault("use_srcs", True)
        # Slice layout follows the SrcS element width, not dest_acc: 32-bit
        # only for Float32, Int32 and Tf32. MX lands as Float16_b, so an MX
        # buffer keeps 144-byte slices whatever dest_acc says.
        values = self.l1.read(
            l1_bytes, l1_format, dest_acc=src_format.is_32_bit(), **geometry
        )
        return to_src_storage(values, src_format, self.src_mantissa(src_format))

    def l1_to_dest(
        self,
        l1_bytes: L1Buffer,
        l1_format: DataFormat,
        dest_format: Optional[DataFormat] = None,
        *,
        dest_acc: Union[bool, DestAccumulation] = False,
        **geometry,
    ) -> torch.Tensor:
        """Load an L1 buffer straight into Dest, bypassing the src registers.

        The path an op takes to seed Dest before a feedback loop. The value lands
        at Dest precision, so it keeps more mantissa than the same buffer read
        through :meth:`l1_to_src` would.
        """
        self.l1.check(l1_format)
        dest_format = self.resolve_dest_format(dest_format, l1_format, dest_acc)
        if self.to_dest_formats:
            legal = self.to_dest_formats.get(l1_format, frozenset())
            if dest_format not in legal:
                raise ValueError(
                    f"{self.l1.name} cannot unpack {l1_format} into a "
                    f"{dest_format} Dest. The unpacker does not widen, so a "
                    f"narrow input has no 32-bit Dest target. Legal Dest "
                    f"formats for it: {sorted(str(f) for f in legal)}."
                )
        values = self.l1.read(l1_bytes, l1_format, **geometry)
        # The unpacker truncates here, since there is no rounding stage on this
        # path, unlike the FPU's write to Dest. A plain cast would round, and
        # a Float16 Dest drops its low bits the same way a Float16_b one does.
        keep = DEST_MANTISSA_BITS.get(dest_format)
        if keep is not None:
            values = truncate_mantissa(values, keep)
        return to_dest_storage(values, dest_format)

    # ------------------------------------------------------------------
    # Dest format, derived from this unpacker's output
    # ------------------------------------------------------------------

    def dest_format_for(
        self,
        l1_input_format: DataFormat,
        dest_acc: Union[bool, DestAccumulation] = False,
    ) -> DataFormat:
        """The Dest format for an op whose *input* was `l1_input_format`.

        Dest takes the src register's format, so the input decides the family and
        `dest_acc` picks the 32- or 16-bit member.

        **Float32 and Tf32 input raise**, because the input format alone does not
        decide the family and guessing would model a different kernel. See
        `README.md`, "The two families a wide float could land in".
        """
        self.l1.check(l1_input_format)
        dest_acc = as_dest_acc(dest_acc)
        if l1_input_format.is_integer():
            # Same pairing resolve_dest_format enforces: deriving the format
            # is not a reason to skip the check.
            if dest_acc:
                if l1_input_format not in INT32_DEST_COMPATIBLE:
                    raise ValueError(
                        f"{l1_input_format} cannot drive a 32-bit Dest, so "
                        f"dest_acc has no valid Dest format for it. The FPU "
                        f"takes {sorted(f.name for f in INT32_DEST_COMPATIBLE)} "
                        f"into Int32."
                    )
                return DataFormat.Int32
            if l1_input_format in DEST_32_BIT_FORMATS:
                raise ValueError(
                    f"{l1_input_format} is a 32-bit format, so its Dest is "
                    f"32-bit and dest_acc has to be enabled. With it off there "
                    f"is no narrower Dest to put it in."
                )
            if l1_input_format not in self.l1.dest_formats:
                raise ValueError(
                    f"{self.l1.name} can read {l1_input_format} from L1 "
                    f"but cannot hold it in Dest, and this golden models no "
                    f"conversion for it. Dest can hold "
                    f"{sorted(str(f) for f in self.l1.dest_formats)}."
                )
            return l1_input_format
        if dest_acc:
            return DataFormat.Float32
        if l1_input_format in (DataFormat.Float32, DataFormat.Tf32):
            raise ValueError(
                f"{l1_input_format} input with a 16-bit Dest does not determine "
                f"the register family on its own: the unpacker may land it as "
                f"Float16 (10 mantissa bits, 5-bit exponent, saturates outside "
                f"~[6.1e-05, 65504]) or as Float16_b/Tf32 (full fp32 range, "
                f"bf16's 7 mantissa bits in Dest). The harness chooses from the "
                f"*output* format in infer_unpack_out, so pass the dest_format "
                f"and src_format it configured, or enable dest_acc, where both "
                f"families give a Float32 Dest."
            )
        # Narrow Dest keeps the exponent family the unpacker put in the src
        # register.
        src = self._src_format(l1_input_format)
        return DataFormat.Float16 if src is DataFormat.Float16 else DataFormat.Float16_b

    def resolve_dest_format(
        self,
        dest_format: Optional[DataFormat],
        l1_input_format: DataFormat,
        dest_acc: Union[bool, DestAccumulation],
    ) -> DataFormat:
        """The Dest format to run with: derived from the input, or validated.

        An explicit `dest_format` is checked against `dest_acc` rather than taken
        on trust. Dest is 32-bit exactly when accumulation is on, so ``Float32``
        with ``dest_acc=False`` describes a state that cannot exist, and left
        unchecked it runs at the wrong precision and looks like a maths bug.
        """
        dest_acc = as_dest_acc(dest_acc)
        if dest_format is None:
            return self.dest_format_for(l1_input_format, dest_acc)
        self.l1.check_dest_format(dest_format)
        if (dest_format in DEST_32_BIT_FORMATS) != dest_acc:
            wide = dest_format in DEST_32_BIT_FORMATS
            raise ValueError(
                f"dest_format={dest_format} is a {32 if wide else 16}-bit Dest "
                f"format but dest_acc={dest_acc}, and Dest is 32-bit exactly "
                f"when accumulation is enabled. Pass "
                f"dest_acc={'True' if wide else 'False'}, or leave dest_format "
                f"as None to derive it from the input format."
            )
        return dest_format
