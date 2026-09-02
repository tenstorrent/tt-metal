# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Value-level model of the L1 <-> register-file data-transfer blocks.

Each block answers one question: given a buffer sitting in L1, what does the
consumer see after the hardware has moved it? Blocks take and return the things
the hardware takes and returns — **L1 takes bytes** — so a golden op can chain
them the way the pipeline does:

    l1_bytes -> l1_to_srcA -> [math] -> dest_to_l1 -> l1_bytes

There is deliberately no quantization step. Storing a tensor as MxFp8R *is* the
quantization; by the time the L1 bytes exist the loss has happened, and reading
them back just reports it. Use :func:`.l1_codec.pack_to_l1` to build a buffer.

That leaves one real conversion for these blocks to model: the unpacker lands
every L1 format in one of a small number of src-register storage families. The
family, not the L1 format, is what the FPU reads.

Architectures differ in which L1 formats exist and in that mapping, so the
machinery lives here and each architecture supplies the differences. Notably
Quasar has the MX family and **no block float**; Wormhole/Blackhole the reverse.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, FrozenSet, Optional, Sequence, Union

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import PackerReluType, StochasticRounding, format_dict

from .l1_codec import pack_to_l1, unpack_from_l1
from .pack_effects import PackEdgeMask, apply_pack_effects

#: Explicit mantissa bits a src-register datum holds. The datum is 19 bits:
#: 1 sign + 8 exponent + 10 mantissa, whatever format is stored in it.
SRC_MANT_BITS = 10

#: Mantissa bits dropped converting Float32 (23 explicit bits) into a src datum.
FP32_TO_SRC_MANT_TRUNC = 23 - SRC_MANT_BITS

#: Formats the unpacker can land in a src register.
#:
#: Only two families exist in the datapath. ``Float16_b`` is an **alias for
#: Tf32** — the hardware converts it to Tf32 and the distinction survives only
#: in the row/column mask path, which is not on the datum path.
#: Nothing truncates the mantissa to bf16's 7 bits, so precision in a src
#: register is ``min(input mantissa bits, 10)`` — the src format sets the
#: exponent range, not the mantissa width.
SRC_STORAGE_FORMATS = frozenset(
    {DataFormat.Float16, DataFormat.Float16_b, DataFormat.Tf32}
)

#: Formats a Dest register can hold.
#:
#: Dest is 32-bit when accumulation is enabled and 16-bit otherwise — that is
#: the whole of what ``DestAccumulation`` controls. Tf32 has no separate Dest
#: encoding; it lives in a Float32 container.
DEST_STORAGE_FORMATS = frozenset(
    {
        DataFormat.Float32,
        DataFormat.Int32,
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.Int16,
        DataFormat.Int8,
    }
)

#: 32-bit Dest formats — valid exactly when ``DestAccumulation.Yes``.
DEST_32_BIT_FORMATS = frozenset({DataFormat.Float32, DataFormat.Int32})

L1Buffer = Union[Sequence[int], bytes]


class DataTransferBlocks(ABC):
    """Base for the per-architecture data-transfer blocks."""

    #: L1 formats this architecture's unpacker can read.
    SUPPORTED_L1_FORMATS: ClassVar[FrozenSet[DataFormat]] = frozenset()

    # ------------------------------------------------------------------
    # Blocks
    # ------------------------------------------------------------------

    @abstractmethod
    def l1_to_srcA(
        self,
        l1_bytes: L1Buffer,
        l1_format: DataFormat,
        src_format: Optional[DataFormat] = None,
        **geometry,
    ) -> torch.Tensor:
        """Values visible in SrcA after unpacking `l1_bytes` from L1."""

    def l1_to_srcB(
        self,
        l1_bytes: L1Buffer,
        l1_format: DataFormat,
        src_format: Optional[DataFormat] = None,
        **geometry,
    ) -> torch.Tensor:
        """Values visible in SrcB. SrcA and SrcB share the datum layout."""
        return self.l1_to_srcA(l1_bytes, l1_format, src_format, **geometry)

    def l1_to_srcS(
        self,
        l1_bytes: L1Buffer,
        l1_format: DataFormat,
        src_format: Optional[DataFormat] = None,
        **geometry,
    ) -> torch.Tensor:
        """Values visible in SrcS.

        SrcS uses a per-slice L1 layout rather than one flat block list, so the
        buffer must have been packed with ``use_srcs=True``; the values it
        decodes to are the same.
        """
        geometry.setdefault("use_srcs", True)
        return self.l1_to_srcA(l1_bytes, l1_format, src_format, **geometry)

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
        """Pack Dest into L1 — the T2 half, mirroring :meth:`l1_to_srcA`.

        The packer's pipeline, in hardware order:

        1. **Dest storage precision.** What a Dest slot can hold, which is 32-bit
           under accumulation and 16-bit otherwise. Idempotent if the math block
           upstream already applied it, so it is safe to hand this a plain fp32
           result from a torch golden.
        2. **ReLU**. The threshold is narrowed to the 16 bits the packer's
           configuration register holds before it is compared against.
        3. **Edge mask**, zeroing or negative-saturating datums at tile edges.
        4. **The packer.** Rounding and requantization into `l1_format` are
           whatever :mod:`helpers.pack` implements — the same codec the test
           harness writes L1 with.

        `dest_format` cannot be inferred from `l1_format`: Dest follows the
        *unpacker's* output, i.e. the input format, never the pack format. Use
        :meth:`dest_format_for` to derive it from the input side.

        `stoch_rnd` is accepted for completeness but **cannot be reproduced** —
        it is driven by a pseudo-random sequence on device. With it enabled the bytes returned here are
        the round-to-nearest result, which hardware matches only in expectation.
        Check :func:`.pack_effects.is_deterministic` and compare with PCC.
        """
        self._check_dest_format(dest_format)
        values = self._to_dest_storage(dest_values, dest_format)
        values = apply_pack_effects(
            values,
            relu_type=relu_type,
            relu_threshold=relu_threshold,
            dest_format=dest_format,
            edge_mask=edge_mask,
        )
        return self.pack_to_l1(values, l1_format, **geometry)

    # ------------------------------------------------------------------
    # L1 access
    # ------------------------------------------------------------------

    def supports(self, l1_format: DataFormat) -> bool:
        """Whether this architecture's unpacker can read `l1_format` from L1."""
        return l1_format in self.SUPPORTED_L1_FORMATS

    @property
    def supported_dest_formats(self) -> FrozenSet[DataFormat]:
        """Dest formats available on this architecture."""
        return DEST_STORAGE_FORMATS & self.SUPPORTED_L1_FORMATS

    def dest_format_for(
        self, l1_input_format: DataFormat, dest_acc: bool = False
    ) -> DataFormat:
        """The Dest format for an op whose *input* was `l1_input_format`.

        Dest follows the unpacker's src output, so it is the input format that
        decides the family — never the pack format. `dest_acc` then picks the
        32- or 16-bit member of that family.
        """
        self._check_supported(l1_input_format)
        if l1_input_format.is_integer():
            return DataFormat.Int32 if dest_acc else l1_input_format
        if dest_acc:
            return DataFormat.Float32
        # Narrow Dest keeps the exponent family the unpacker put in the src register.
        src = self._src_format(l1_input_format)
        return DataFormat.Float16 if src is DataFormat.Float16 else DataFormat.Float16_b

    def _check_dest_format(self, dest_format: DataFormat) -> None:
        if dest_format not in self.supported_dest_formats:
            raise ValueError(
                f"{dest_format} is not a Dest format on {type(self).__name__}. "
                f"Dest can hold {sorted(str(f) for f in self.supported_dest_formats)}."
            )

    @staticmethod
    def _to_dest_storage(values: torch.Tensor, dest_format: DataFormat) -> torch.Tensor:
        """Round `values` to what a Dest slot can hold."""
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values)
        return values.to(format_dict[dest_format])

    def pack_to_l1(
        self, tensor: torch.Tensor, l1_format: DataFormat, **geometry
    ) -> list:
        """Lay `tensor` out in L1 as `l1_format`. Where precision is lost."""
        self._check_supported(l1_format)
        return pack_to_l1(tensor, l1_format, **geometry)

    def unpack_from_l1(
        self, l1_bytes: L1Buffer, l1_format: DataFormat, **geometry
    ) -> torch.Tensor:
        """Read `l1_bytes` back as values, before any src-register conversion."""
        self._check_supported(l1_format)
        if isinstance(l1_bytes, torch.Tensor):
            raise TypeError(
                "L1 holds bytes, not a tensor. Build a buffer with "
                f"pack_to_l1(tensor, {l1_format}) and pass that instead."
            )
        return unpack_from_l1(l1_bytes, l1_format, **geometry)

    def _check_supported(self, l1_format: DataFormat) -> None:
        if not self.supports(l1_format):
            raise ValueError(
                f"{type(self).__name__} cannot read {l1_format} from L1 on this "
                f"architecture."
            )

    # ------------------------------------------------------------------
    # Format conversion
    # ------------------------------------------------------------------

    def src_format(self, l1_format: DataFormat) -> DataFormat:
        """The src-register storage format the unpacker lands `l1_format` in.

        Raises if this architecture has no such L1 format — the mapping is
        derived from exponent-family predicates that answer for every
        ``DataFormat``, so without this check it would return a plausible
        answer for a format the hardware cannot read.
        """
        self._check_supported(l1_format)
        return self._src_format(l1_format)

    def _src_format(self, l1_format: DataFormat) -> DataFormat:
        """Architecture's L1 -> src-register format mapping.

        Float32 and Tf32 land in Tf32 (8-bit exponent, 10-bit mantissa).
        Everything else resolves to one of the two 16-bit exponent families, and
        integer formats pass through unchanged. Override where an architecture
        diverges; :meth:`src_format` does the support check.
        """
        if l1_format in (DataFormat.Float32, DataFormat.Tf32):
            return DataFormat.Tf32
        if l1_format.is_mx_format():
            # The unpacker converts MX into the 8-bit-exponent family regardless of
            # the pack format, so math and Dest see bf16.
            return DataFormat.Float16_b
        if l1_format.is_exponent_A():
            return DataFormat.Float16
        if l1_format.is_exponent_B():
            return DataFormat.Float16_b
        return l1_format

    def _l1_to_src(
        self,
        l1_bytes: L1Buffer,
        l1_format: DataFormat,
        src_format: Optional[DataFormat] = None,
        **geometry,
    ) -> torch.Tensor:
        """Read L1, then apply src-register storage precision.

        Any L1 format can be unpacked into any src storage format; `src_format`
        defaults to the one the LLK would pick for `l1_format`.
        """
        if src_format is None:
            src_format = self.src_format(l1_format)
        elif not self._is_valid_src_format(l1_format, src_format):
            raise ValueError(
                f"{src_format} is not a src-register storage format. "
                f"A src register can hold {sorted(str(f) for f in SRC_STORAGE_FORMATS)}"
                f"{' or pass ' + str(l1_format) + ' through unconverted' if l1_format == src_format else ''}."
            )
        values = self.unpack_from_l1(l1_bytes, l1_format, **geometry)
        return self._to_src_storage(values, src_format)

    @staticmethod
    def _is_valid_src_format(l1_format: DataFormat, src_format: DataFormat) -> bool:
        """A src storage format, or the input format passed through unconverted."""
        return src_format in SRC_STORAGE_FORMATS or src_format == l1_format

    @staticmethod
    def _to_src_storage(values: torch.Tensor, src_format: DataFormat) -> torch.Tensor:
        """Apply the precision a src-register datum can hold.

        Two losses, both from the unpacker:

        * **Mantissa** — a src datum keeps 10 explicit bits, truncated (not
          rounded). Only Float32/Tf32 data carries more.
        * **Exponent range** — the Float16 family has a 5-bit range, so values
          above it saturate to infinity and values below the smallest normal
          flush to zero. The Tf32 family has fp32's range and clips nothing.

        Float32 is *not* a conversion target: a 19-bit datum cannot hold it, so
        the unpacker splits it across two lanes (mantissa MSBs low, LSBs high)
        and every bit survives. Integer formats pass through unchanged.
        """
        if src_format is DataFormat.Float16:
            # 1+5+10 is IEEE fp16 exactly. Truncate first so the cast only has
            # to apply the range clamp -- casting straight to fp16 would round.
            truncated = DataTransferBlocks._truncate_src_mantissa(values)
            narrowed = truncated.to(torch.float16)
            # The unpacker flushes to zero once the rebiased exponent hits 0,
            # so fp16 subnormals never reach the register.
            return torch.where(
                narrowed.abs() < torch.finfo(torch.float16).smallest_normal,
                torch.zeros_like(narrowed),
                narrowed,
            )
        if src_format in (DataFormat.Float16_b, DataFormat.Tf32):
            # Float16_b is an alias for Tf32 here -- see SRC_STORAGE_FORMATS.
            return DataTransferBlocks._truncate_src_mantissa(values)
        if src_format is DataFormat.Float32:
            return values.to(torch.float32)
        return values.to(format_dict[src_format])

    @staticmethod
    def _truncate_src_mantissa(values: torch.Tensor) -> torch.Tensor:
        """Keep the top 10 mantissa bits, truncating as the unpacker does."""
        raw = values.to(torch.float32).contiguous().view(torch.int32)
        return (raw & ~((1 << FP32_TO_SRC_MANT_TRUNC) - 1)).view(torch.float32)
