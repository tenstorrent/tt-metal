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

What is left is the loss each boundary takes: the unpacker landing an L1 datum
in a src register, the FPU writing a Dest slot, Dest fed back into a src
register, and the packer's ReLU, edge mask and requantization on the way out.

One of those varies by architecture — the unpacker lands every L1 format in one
of a small number of src-register storage families, and the family, not the L1
format, is what the FPU reads. Architectures differ in which L1 formats exist
and in that mapping, so the machinery lives here and each architecture supplies
the differences. Notably Quasar has the MX family and **no block float**;
Wormhole and Blackhole the reverse.
"""

import warnings
from abc import ABC
from typing import ClassVar, FrozenSet, Mapping, Optional, Sequence, Union

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    PackerReluType,
    StochasticRounding,
    format_dict,
)

from .l1_codec import (
    MODELLED_L1_FORMATS,
    _no_codec_message,
    pack_to_l1,
    unpack_from_l1,
)
from .pack_effects import (
    STOCH_RND_EFFECTS,
    PackEdgeMask,
    apply_pack_effects,
    is_deterministic,
)

#: Explicit mantissa bits an IEEE float32 holds, the width every mask below is
#: cut from.
FP32_MANTISSA_BITS = 23

#: Explicit mantissa bits a src-register datum holds. The datum is 19 bits:
#: 1 sign + 8 exponent + 10 mantissa, whatever format is stored in it.
SRC_MANT_BITS = 10

#: Explicit mantissa bits a bf16 datum holds, and so what survives a Float16_b
#: Dest on the way back to a src register.
BF16_MANT_BITS = 7

#: Symmetric magnitude an INT8-shaped src datum reaches: 7 magnitude bits plus
#: a sign, so -128 has no representation on this path.
INT8_MAX_MAGNITUDE = 127

#: Formats the unpacker can land in a src register.
#:
#: Only two families exist in the datapath. ``Float16_b`` is an **alias for
#: Tf32** — the hardware converts it to Tf32 and the distinction survives only
#: in the row/column mask path, which is not on the datum path.
#: Nothing truncates the mantissa to bf16's 7 bits, so precision in a src
#: register is ``min(input mantissa bits, 10)`` — the src format sets the
#: exponent range, not the mantissa width.
#: The two format families an architecture has one of, derived from the
#: predicates in :mod:`helpers.format_config` rather than relisted here, so a
#: new member reaches every architecture without a second copy of the list.
#: Quasar has MX and no block float; Wormhole and Blackhole the reverse.
BLOCK_FLOAT_FORMATS = frozenset(f for f in DataFormat if f.is_block_float())
MX_FORMATS = frozenset(f for f in DataFormat if f.is_mx_format())

SRC_STORAGE_FORMATS = frozenset(
    {DataFormat.Float16, DataFormat.Float16_b, DataFormat.Tf32}
)

#: Formats the unpacker can land in SrcS: the src-register formats plus full
#: Float32, which SrcA/SrcB cannot hold.
SRCS_STORAGE_FORMATS = SRC_STORAGE_FORMATS | {DataFormat.Float32}

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
        DataFormat.UInt8,
    }
)

#: 32-bit Dest formats — valid exactly when ``DestAccumulation.Yes``.
#: Integer L1 formats that can drive a 32-bit Int32 Dest. Int16 is absent on
#: purpose: the FPU supports it for MOV only and not in 32-bit Dest mode.
INT32_DEST_COMPATIBLE = frozenset({DataFormat.Int8, DataFormat.UInt8, DataFormat.Int32})

DEST_32_BIT_FORMATS = frozenset({DataFormat.Float32, DataFormat.Int32})

L1Buffer = Union[Sequence[int], bytes]


def truncate_mantissa(values: torch.Tensor, keep_bits: int) -> torch.Tensor:
    """Keep the top `keep_bits` explicit mantissa bits, zeroing the rest.

    Truncation, not rounding: every narrowing on the register paths drops the
    low bits rather than folding them in, so this is the one primitive behind
    all of them -- the unpacker's 10-bit src datum, a Float16_b Dest's 7 bits
    on the way back, and the high half of a fidelity phase.
    """
    raw = values.to(torch.float32).contiguous().view(torch.int32)
    return (raw & ~((1 << (FP32_MANTISSA_BITS - keep_bits)) - 1)).view(torch.float32)


def saturate_to_integer(
    values: torch.Tensor, dtype: torch.dtype, dest_format: DataFormat
) -> torch.Tensor:
    """Clamp into `dtype`'s range, then narrow -- the hardware saturates.

    A plain cast wraps instead: a Dest result of 200 narrowed to Int8 lands as
    -56, so a later clamp in the packer sees -56 where it should have seen 127.
    A float beyond int32's range is worse, because the cast is undefined.

    Signed formats clamp to ``[min + 1, max]``: the register is sign-magnitude,
    so the most negative two's-complement value has no encoding. Unsigned
    formats use the full range.

    The same rule as ``golden_generators.saturate_integer``, duplicated for the
    reason the rest of this package duplicates -- heavyweight owns its model of
    the hardware so the old golden can be retired without stranding it.
    """
    info = torch.iinfo(dtype)
    low = info.min if dest_format.name.startswith("U") else info.min + 1
    if values.is_floating_point():
        # float64 for the clamp: int32's max is not representable in float32,
        # so clamping there rounds the bound *up* to 2^31 and the cast is
        # undefined again -- the very thing being fixed. float64 holds every
        # bound these formats use exactly. NaN survives a clamp and makes the
        # cast undefined too, so map it first.
        values = torch.nan_to_num(
            values.double(), nan=0.0, posinf=float(info.max), neginf=float(low)
        )
    return values.clamp(low, info.max).to(dtype)


class UnmodelledHardwareWarning(RuntimeWarning):
    """A hardware behaviour this golden knowingly does not reproduce.

    A ``RuntimeWarning`` subclass on purpose. ``pytest.ini`` sets
    ``ignore::UserWarning`` for the whole suite, so a plain ``UserWarning``
    reaches nobody running the tests -- and that is the only audience for a
    warning whose message is "this answer is approximate". ``pytest.warns``
    bypasses the filter, so the unit test would keep passing while the suite
    stayed silent.
    """


def flush_subnormals(
    values: torch.Tensor, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Zero anything below the smallest normal of `dtype`, or of `values` itself.

    Neither the FPU nor the unpacker writes a denormal: a slot whose exponent
    field is zero has a zero mantissa too. Keeping the subnormal instead lets a
    value the hardware zeroed survive to the packer, which rounds it *up* onto
    the output lattice -- so a datum the device reports as 0 comes back as the
    output format's smallest representable value.

    By default the threshold comes from the tensor's own dtype, which suits a
    flush applied *after* the narrowing cast. Pass `dtype` to flush against a
    narrower format while still holding the wide value -- needed wherever the
    cast rounds, because rounding can carry the top of the subnormal range up
    to the smallest normal and so past a flush applied afterwards.
    """
    threshold = torch.finfo(dtype or values.dtype).smallest_normal
    return torch.where(
        values.abs() < threshold,
        torch.zeros_like(values),
        values,
    )


def as_dest_acc(dest_acc: Union[bool, DestAccumulation]) -> bool:
    """Normalise an accumulation setting to a plain bool.

    Accepts ``DestAccumulation`` as well as a bool, because the rest of the
    harness passes the enum and a bare ``bool`` parameter cannot be trusted to
    notice: ``DestAccumulation`` defines no ``__bool__``, so ``DestAccumulation.No``
    is **truthy**. Passed to a bool parameter it reads as "accumulation on" and
    silently selects a 32-bit Dest -- the model then runs at fp32 where the
    device had 16 bits, which looks like an arithmetic discrepancy rather than
    a wiring mistake.

    Taking the enum here and reducing to a bool once keeps the codec boundary
    (``helpers.pack`` / ``helpers.unpack``, which size tiles from a bool) plain
    while the ops speak the harness's own vocabulary.
    """
    if isinstance(dest_acc, DestAccumulation):
        return dest_acc is DestAccumulation.Yes
    return bool(dest_acc)


class DataTransferBlocks(ABC):
    """Base for the per-architecture data-transfer blocks."""

    #: Legal ``L1 format -> src-register format`` pairs for this architecture's
    #: unpacker. Empty means the architecture is not modelled at this level of
    #: detail, and the weaker "is it a src storage format at all" check applies
    #: instead; it is not a claim that everything is legal.
    UNPACK_TO_SRC_FORMATS: ClassVar[Mapping[DataFormat, FrozenSet[DataFormat]]] = {}

    #: Legal ``L1 format -> Dest format`` pairs for the unpack-to-Dest path,
    #: which is a different table from :attr:`UNPACK_TO_SRC_FORMATS`. Notably
    #: the unpacker does not widen: a narrow input cannot land in a 32-bit
    #: Dest. Empty means unmodelled, and the weaker "is it a Dest format at
    #: all" check applies instead.
    UNPACK_TO_DEST_FORMATS: ClassVar[Mapping[DataFormat, FrozenSet[DataFormat]]] = {}

    #: L1 formats this architecture's unpacker can read. Empty in the base: a
    #: subclass has to declare it, and that -- not any method -- is what makes
    #: this class abstract in practice. An instance that reached here with the
    #: empty set would reject every format, so :meth:`__init__` says so up
    #: front instead of failing one call later.
    SUPPORTED_L1_FORMATS: ClassVar[FrozenSet[DataFormat]] = frozenset()

    #: Edge-mask polarity: whether a set bit in an edge-mask register masks the
    #: datum. False is Wormhole/Blackhole, where ``PCK_EDGE_OFFSET`` mask 0xFFFF
    #: passes a row through and 0x0 clears it; Quasar overrides it.
    EDGE_MASK_MASKED_WHEN_SET: ClassVar[bool] = False

    #: Legal ``Dest format -> L1 format`` pairs for the packer, the pack-side
    #: counterpart of :attr:`UNPACK_TO_SRC_FORMATS`. The packer converts on the
    #: way out, but not between arbitrary pairs: a 32-bit Dest reaches the
    #: narrow floats, a 16-bit float Dest does not reach Float32, and the
    #: integer widths each reach only their own. Empty means unmodelled, and
    #: only the weaker "can this architecture hold the L1 format, and can Dest
    #: hold the Dest format" checks apply; it is not a claim that every pair is
    #: legal.
    PACK_TO_L1_FORMATS: ClassVar[Mapping[DataFormat, FrozenSet[DataFormat]]] = {}

    #: Whether this architecture has a SrcS register at all. False in the base,
    #: and only Quasar sets it: Wormhole and Blackhole have no SrcS, no
    #: ``UNP_S`` unpacker and no ``_is_srcs_32bit_mode_`` -- every apparent
    #: match in their headers is ``SrcSelector`` or ``UNP_SEL``. Without this
    #: the SrcS blocks sit on the base class and answer for a register that
    #: does not exist, returning full-fp32 "SrcS" values on Wormhole.
    HAS_SRCS: ClassVar[bool] = False

    #: Whether the harness promotes an exponent-B input with a Float16 output
    #: to a 32-bit Dest. True here because ``TestConfig`` does exactly that on
    #: every architecture except Quasar, which it names explicitly
    #: (``is_format_combination_outlier`` + ``CHIP_ARCH != QUASAR``). The
    #: combination therefore never runs with a 16-bit Dest on those devices, so
    #: modelling it as one would compare a bf16 Dest against silicon running
    #: fp32. Quasar clears it.
    PROMOTES_OUTLIER_TO_32_BIT_DEST: ClassVar[bool] = True

    def __init__(self) -> None:
        if not self.SUPPORTED_L1_FORMATS:
            raise TypeError(
                f"{type(self).__name__} declares no SUPPORTED_L1_FORMATS, so it "
                f"can read nothing from L1. Instantiate an architecture's "
                f"subclass, or give this one its format set."
            )

    # ------------------------------------------------------------------
    # Blocks
    # ------------------------------------------------------------------

    def l1_to_srcA(
        self,
        l1_bytes: L1Buffer,
        l1_format: DataFormat,
        src_format: Optional[DataFormat] = None,
        **geometry,
    ) -> torch.Tensor:
        """Values visible in SrcA after unpacking `l1_bytes` from L1.

        Concrete, because the unpack itself does not vary by architecture --
        what varies is :meth:`_src_format`, which decides the storage format,
        and ``SUPPORTED_L1_FORMATS``. An architecture whose unpack genuinely
        differs can still override this.
        """
        return self._l1_to_src(l1_bytes, l1_format, src_format, **geometry)

    def l1_to_srcB(
        self,
        l1_bytes: L1Buffer,
        l1_format: DataFormat,
        src_format: Optional[DataFormat] = None,
        **geometry,
    ) -> torch.Tensor:
        """Values visible in SrcB. SrcA and SrcB share the datum layout."""
        return self.l1_to_srcA(l1_bytes, l1_format, src_format, **geometry)

    def _check_pack_pair(self, dest_format: DataFormat, l1_format: DataFormat) -> None:
        """Refuse a ``Dest -> L1`` pair this architecture's packer cannot do.

        The unpack side has :meth:`_is_valid_src_format` for the mirror of
        this. Without it a Float32 Dest packed to Float16_b, or an Int32 Dest
        packed to a float, returns bytes that decode to believable numbers for
        a conversion the hardware refuses to perform.
        """
        if not self.PACK_TO_L1_FORMATS:
            return
        allowed = self.PACK_TO_L1_FORMATS.get(dest_format, frozenset())
        if l1_format not in allowed:
            raise ValueError(
                f"{type(self).__name__} cannot pack a {dest_format} Dest to "
                f"{l1_format}. From {dest_format} the packer reaches "
                f"{sorted(str(f) for f in allowed) or 'nothing'}."
            )

    def _check_has_srcs(self) -> None:
        if not self.HAS_SRCS:
            raise ValueError(
                f"{type(self).__name__} has no SrcS register, so there is no "
                f"such transfer to model. SrcS is Quasar-only: Wormhole and "
                f"Blackhole have no UNP_S unpacker and no "
                f"_is_srcs_32bit_mode_. Use l1_to_srcA/l1_to_srcB, or "
                f"l1_to_dest for the unpack-to-Dest path."
            )

    def l1_to_srcS(
        self,
        l1_bytes: L1Buffer,
        l1_format: DataFormat,
        src_format: Optional[DataFormat] = None,
        *,
        dest_acc: Union[bool, DestAccumulation] = False,
        **geometry,
    ) -> torch.Tensor:
        """Values visible in SrcS.

        SrcS uses a per-slice L1 layout rather than one flat block list, so the
        buffer must have been packed with ``use_srcs=True``. `dest_acc` selects
        the storage format; the slice layout follows the resolved src format's
        width instead, which is what the hardware keys on.

        Not a delegate of :meth:`l1_to_srcA`: SrcS has its own format rules, and
        under SrcA's a Float32 input would lose the 13 mantissa bits SrcS keeps.
        See :meth:`srcs_format`.
        """
        self._check_has_srcs()
        self._check_supported(l1_format)
        dest_acc = as_dest_acc(dest_acc)
        if src_format is None:
            src_format = self.srcs_format(l1_format, dest_acc)
        elif not self._is_valid_srcs_format(l1_format, src_format, dest_acc):
            raise ValueError(
                f"{type(self).__name__} cannot unpack {l1_format} into SrcS as "
                f"{src_format} with dest_acc={dest_acc}. SrcS holds "
                f"{sorted(str(f) for f in SRCS_STORAGE_FORMATS)} or the input "
                f"format itself, and with dest_acc a Float16/Float16_b input "
                f"stays as itself."
            )
        geometry.setdefault("use_srcs", True)
        # The slice layout follows the SrcS element width, not dest_acc. On
        # device ``_is_srcs_32bit_mode_`` keys on the UNP_S destination format
        # -- 32-bit only for Float32, Int32 and Tf32 -- and the harness derives
        # it the same way, from ``unpack_S_dst.is_32_bit()``. MX lands in SrcS
        # as Float16_b, so an MX buffer holds 144-byte slices whatever dest_acc
        # says; reading it at 80 would take the wrong stride through data the
        # device wrote.
        values = self.unpack_from_l1(
            l1_bytes, l1_format, dest_acc=src_format.is_32_bit(), **geometry
        )
        return self._to_src_storage(values, src_format)

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

        The path an op takes to seed Dest before a feedback loop. The value
        lands at Dest precision, not src-register precision, so it keeps more
        mantissa than the same buffer read through ``l1_to_srcA`` would.
        """
        self._check_supported(l1_format)
        dest_format = self.resolve_dest_format(dest_format, l1_format, dest_acc)
        if self.UNPACK_TO_DEST_FORMATS:
            legal = self.UNPACK_TO_DEST_FORMATS.get(l1_format, frozenset())
            if dest_format not in legal:
                raise ValueError(
                    f"{type(self).__name__} cannot unpack {l1_format} into a "
                    f"{dest_format} Dest. The unpacker does not widen, so a "
                    f"narrow input has no 32-bit Dest target. Legal Dest "
                    f"formats for it: {sorted(str(f) for f in legal)}."
                )
        values = self.unpack_from_l1(l1_bytes, l1_format, **geometry)
        # The unpacker truncates on this path -- there is no rounding stage in
        # the unpack datapath, unlike the FPU's write to Dest that
        # :meth:`src_to_dest` models. Narrowing with a plain cast would round.
        if dest_format in (DataFormat.Float16_b, DataFormat.Tf32):
            values = truncate_mantissa(values, BF16_MANT_BITS)
        return self._to_dest_storage(values, dest_format)

    def dest_to_srcA(
        self,
        dest_values: torch.Tensor,
        dest_format: DataFormat = DataFormat.Float32,
        src_format: DataFormat = DataFormat.Float16_b,
    ) -> torch.Tensor:
        """Move Dest back into SrcA, so an op can use its own result as an operand.

        Dest is wider than a src register, so the round trip is lossy — modelling
        that loss where the hardware takes it is the point of having this block.
        See :meth:`_dest_to_src_storage` for what each Dest format costs.
        """
        return self._dest_to_src_storage(dest_values, dest_format, src_format)

    def dest_to_srcB(
        self,
        dest_values: torch.Tensor,
        dest_format: DataFormat = DataFormat.Float32,
        src_format: DataFormat = DataFormat.Float16_b,
    ) -> torch.Tensor:
        """Move Dest back into SrcB. See :meth:`dest_to_srcA`."""
        return self._dest_to_src_storage(dest_values, dest_format, src_format)

    @staticmethod
    def _dest_to_src_storage(
        values: torch.Tensor, dest_format: DataFormat, src_format: DataFormat
    ) -> torch.Tensor:
        """Re-quantize a Dest datum into a 19-bit src datum.

        The conversion is driven by the **Dest** format, not the src format —
        the src format is consulted only to decide whether a wide Dest needs its
        exponent rebiasing into the 5-bit range. Every case is bit-slicing, so
        nothing rounds: bits below the target width are dropped, not folded in.

        * **Float16_b** keeps 7 mantissa bits and zero-fills the low 3, so a
          bf16 Dest does **not** come back with a src register's full 10.
        * **Int16 is carried unchanged.** The hardware routes it through the
          same path by labelling it bf16, but that is transport, not
          conversion: the routing redistributes all 16 bits of the datum --
          sign, then bits 14:8, three padding zeros, then bits 7:0 -- so every
          bit survives and the move back reassembles them. Treating the label
          as a conversion and masking in the float domain would quantize a
          genuine integer instead, turning 257 into 256 and 1001 into 1000.
        * **Float16** passes as fp16 — 10 mantissa bits, 5-bit exponent.
        * **Int32 saturates to INT8**, clamping to +/-127. A wide integer Dest
          cannot survive the trip, and the clamp is silent.
        * **Float32 / Tf32** keep 10 mantissa bits and the 8-bit exponent, unless
          the src register is Float16, in which case the exponent is rebiased and
          values below the fp16 normal range flush to zero.
        """
        if dest_format is DataFormat.Int16:
            # Bit-preserving: see the Int16 note above. The hardware borrows the
            # bf16 label only to get the datum across intact.
            return values
        if dest_format is DataFormat.Float16_b:
            # 7 explicit mantissa bits survive; the low 3 arrive as zeros.
            return truncate_mantissa(values, BF16_MANT_BITS)
        if dest_format is DataFormat.Float16:
            return values.to(torch.float16).to(torch.float32)
        if dest_format is DataFormat.Int32:
            # Only 7 magnitude bits plus a sign reach the src register.
            return (
                values.to(torch.float32)
                .clamp(-INT8_MAX_MAGNITUDE, INT8_MAX_MAGNITUDE)
                .trunc()
            )
        if dest_format in (DataFormat.Float32, DataFormat.Tf32):
            if src_format is DataFormat.Float16:
                return DataTransferBlocks._to_src_storage(values, DataFormat.Float16)
            return DataTransferBlocks._truncate_src_mantissa(values)
        return DataTransferBlocks._to_src_storage(values, src_format)

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

        `stoch_rnd` is accepted so a caller can describe the hardware
        configuration it is comparing against, but any mode other than ``No``
        **warns** rather than being silently ignored — including ``Fpu``, which
        randomises the Dest write rather than this block's. See
        :func:`.pack_effects.is_deterministic`.
        """
        self._check_dest_format(dest_format)
        self._check_pack_pair(dest_format, l1_format)
        if not is_deterministic(stoch_rnd):
            # The golden cannot follow a device-seeded random sequence, so say so
            # at the call site rather than hand back round-to-nearest bytes that
            # look like a reproducible answer. Name the stage that is actually
            # randomised: under Fpu the packer behaves normally and it is the
            # Dest write that drifts, so blaming the packer would send a reader
            # looking in the wrong block. warnings dedupes per location, so a
            # multi-tile run reports this once.
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
        self._check_outlier_promotion(dest_format, l1_format)
        # Normally already at Dest precision (src_to_dest wrote it there); this
        # is a no-op then, and a safety net for a caller that bypassed Dest.
        values = self._to_dest_storage(dest_values, dest_format)
        values = apply_pack_effects(
            values,
            relu_type=relu_type,
            relu_threshold=relu_threshold,
            dest_format=dest_format,
            edge_mask=edge_mask,
            edge_mask_masked_when_set=self.EDGE_MASK_MASKED_WHEN_SET,
        )
        return self.pack_to_l1(values, l1_format, **geometry)

    # ------------------------------------------------------------------
    # L1 access
    # ------------------------------------------------------------------

    def supports(self, l1_format: DataFormat) -> bool:
        """Whether this golden can move `l1_format` through L1 on this arch.

        Two conditions, and they are not the same question: the architecture
        has to be able to hold the format in L1, *and* this golden has to have
        a codec for it. ``Tf32`` is a real L1 format everywhere and ``Bfp8`` on
        Wormhole/Blackhole, but neither has a codec here, so answering on
        architecture support alone would promise bytes that cannot be written.
        :meth:`_check_supported` says which of the two failed.
        """
        return (
            l1_format in self.SUPPORTED_L1_FORMATS and l1_format in MODELLED_L1_FORMATS
        )

    @property
    def supported_dest_formats(self) -> FrozenSet[DataFormat]:
        """Dest formats available on this architecture."""
        return DEST_STORAGE_FORMATS & self.SUPPORTED_L1_FORMATS

    # ------------------------------------------------------------------
    # Src registers -> Dest
    # ------------------------------------------------------------------

    def src_to_dest(
        self,
        values: torch.Tensor,
        dest_format: DataFormat = DataFormat.Float32,
        current: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Write a math result into Dest.

        The maths belongs to the operation; what belongs here is where it lands.
        A Dest slot holds only what `dest_format` can represent, and the value is
        rounded on the way *in*, not on the way out to L1.

        `current` is Dest's existing contents, which a multi-pass op passes to
        accumulate rather than replace — the FPU has an accumulate-enable bit,
        not a second instruction. Because the rounding happens on this write, it
        happens on every pass; accumulating at full precision and rounding once
        at the end would model an accumulator the hardware does not have.
        """
        self._check_dest_format(dest_format)
        if current is not None:
            # An integer Dest accumulates exactly on the device. float32 holds
            # integers only to 2^24, so a Dest value past that loses its low
            # bits here, before the narrow -- 16777217 + 1 would come back as
            # 16777216. float64 covers Int32's whole range exactly.
            wide = torch.float64 if dest_format.is_integer() else torch.float32
            values = current.to(wide) + values.to(wide)
        return self._to_dest_storage(values, dest_format)

    def dest_format_for(
        self,
        l1_input_format: DataFormat,
        dest_acc: Union[bool, DestAccumulation] = False,
    ) -> DataFormat:
        """The Dest format for an op whose *input* was `l1_input_format`.

        Dest takes the src register's format, so the input format decides the
        family and `dest_acc` picks the 32- or 16-bit member of it.

        **Except for Float32 and Tf32 input, where the input format is not
        enough to decide.** A 19-bit src datum cannot hold an fp32, and both
        families are legal targets: Float16 keeps 10 mantissa bits but only a
        5-bit exponent, so values outside roughly [6.1e-05, 65504] saturate to
        zero or infinity, while Float16_b/Tf32 keeps fp32's range and narrows
        Dest to bf16's 7 mantissa bits instead. Which one the kernel gets is a
        harness decision -- ``infer_unpack_out`` picks the family from the
        *output* format, to limit exponent mixing -- and guessing it here would
        model a different kernel than the one under test. So raise, and let the
        caller pass the formats it configured.

        `dest_acc` resolves it: a 32-bit Dest is Float32 for either family, and
        the harness lands the src in Tf32, which is what ``_src_format`` returns.
        """
        self._check_supported(l1_input_format)
        dest_acc = as_dest_acc(dest_acc)
        if l1_input_format.is_integer():
            # The same pairing resolve_dest_format enforces for an explicit
            # format: a 32-bit Dest exists exactly when accumulation is on.
            # Deriving the format is not a reason to skip the check.
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
            if l1_input_format not in self.supported_dest_formats:
                raise ValueError(
                    f"{type(self).__name__} can read {l1_input_format} from L1 "
                    f"but cannot hold it in Dest, and this golden models no "
                    f"conversion for it. Dest can hold "
                    f"{sorted(str(f) for f in self.supported_dest_formats)}."
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
        # Narrow Dest keeps the exponent family the unpacker put in the src register.
        src = self._src_format(l1_input_format)
        return DataFormat.Float16 if src is DataFormat.Float16 else DataFormat.Float16_b

    def resolve_dest_format(
        self,
        dest_format: Optional[DataFormat],
        l1_input_format: DataFormat,
        dest_acc: Union[bool, DestAccumulation],
    ) -> DataFormat:
        """The Dest format to run with: derived from the input, or validated.

        With `dest_format` left as ``None`` this is :meth:`dest_format_for`.
        Given one explicitly, it is checked against `dest_acc` instead of taken
        on trust, because the two are not independent settings: a Dest slot is
        32-bit exactly when accumulation is on, so ``Float32`` with
        ``dest_acc=False`` describes a machine state that cannot exist. Left
        unchecked that combination runs and produces a plausible answer at the
        wrong precision -- full fp32 where the device had 16 bits, or the
        reverse -- which is indistinguishable from a maths bug downstream.
        """
        dest_acc = as_dest_acc(dest_acc)
        if dest_format is None:
            return self.dest_format_for(l1_input_format, dest_acc)
        self._check_dest_format(dest_format)
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

    def _check_outlier_promotion(
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
            self.PROMOTES_OUTLIER_TO_32_BIT_DEST
            and dest_format is DataFormat.Float16_b
            and l1_format is DataFormat.Float16
        ):
            raise ValueError(
                f"{type(self).__name__} never runs a {dest_format} Dest packed "
                f"to {l1_format}: an 8-bit-exponent input with a Float16 output "
                f"is the outlier TestConfig promotes to dest_acc=Yes, so the "
                f"device uses a 32-bit Dest. Pass dest_acc=True and a Float32 "
                f"dest_format to match it."
            )

    def _check_dest_format(self, dest_format: DataFormat) -> None:
        if dest_format not in self.supported_dest_formats:
            raise ValueError(
                f"{dest_format} is not a Dest format on {type(self).__name__}. "
                f"Dest can hold {sorted(str(f) for f in self.supported_dest_formats)}."
            )

    @staticmethod
    def _to_dest_storage(values: torch.Tensor, dest_format: DataFormat) -> torch.Tensor:
        """Round `values` to what a Dest slot can hold.

        The FPU produces no denormal result: a float Dest slot with a zero
        exponent has a zero mantissa too, so anything below the format's
        smallest normal lands as zero. Keeping the subnormal instead lets a
        value the hardware zeroed survive to the packer, which rounds it *up*
        onto the output lattice — a datum the device reports as 0 comes back as
        the output format's smallest representable value. Only Float16 Dest
        meets the threshold in practice; the wider formats bottom out near
        2**-126.
        """
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values)
        target = format_dict[dest_format]
        if not target.is_floating_point:
            return saturate_to_integer(values, target, dest_format)
        return flush_subnormals(values.to(target))

    @staticmethod
    def _normalised_geometry(geometry: dict) -> dict:
        """`geometry` with any ``dest_acc`` reduced to a bool.

        The codec tests it with a plain ``if``, and ``DestAccumulation`` has no
        ``__bool__``, so ``DestAccumulation.No`` passed straight through would
        select the 32-bit SrcS slice layout -- 1280 bytes where 1152 is
        correct. Nothing downstream notices: the read back finds one tile's
        worth of datums in the wrong layout and returns them.

        Every block reaches L1 through :meth:`pack_to_l1` or
        :meth:`unpack_from_l1`, so normalising in both covers the lot.
        """
        if "dest_acc" not in geometry:
            return geometry
        return {**geometry, "dest_acc": as_dest_acc(geometry["dest_acc"])}

    def pack_to_l1(
        self, tensor: torch.Tensor, l1_format: DataFormat, **geometry
    ) -> list:
        """Lay `tensor` out in L1 as `l1_format`. Where precision is lost."""
        self._check_supported(l1_format)
        return pack_to_l1(tensor, l1_format, **self._normalised_geometry(geometry))

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
        return unpack_from_l1(
            l1_bytes, l1_format, **self._normalised_geometry(geometry)
        )

    def _check_supported(self, l1_format: DataFormat) -> None:
        # Two different failures, kept apart because they call for opposite
        # responses: a format the hardware cannot hold means the test is asking
        # for something impossible, while a missing codec means the test is
        # reasonable and the model has a hole.
        if l1_format not in self.SUPPORTED_L1_FORMATS:
            raise ValueError(
                f"{type(self).__name__} cannot read {l1_format} from L1 on this "
                f"architecture."
            )
        if l1_format not in MODELLED_L1_FORMATS:
            raise ValueError(_no_codec_message(l1_format))

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

    def srcs_format(
        self, l1_format: DataFormat, dest_acc: Union[bool, DestAccumulation] = False
    ) -> DataFormat:
        """The storage format the unpacker lands `l1_format` in when the target is SrcS.

        SrcS does not narrow Float32 to Tf32 the way SrcA/SrcB must, and with
        accumulation on it widens instead:

        * MX lands in Float16_b, as it does in SrcA.
        * Float32 stays Float32, with or without `dest_acc`.
        * With `dest_acc`, Float16 and Float16_b stay themselves -- the
          unpacker cannot convert fp16 to a 32-bit SrcS datum -- and every
          other float widens to Float32.
        * Otherwise a format lands as it would in SrcA.

        Integer formats pass through unchanged, and that is a **deliberate
        divergence** from ``infer_unpack_out`` in
        :mod:`helpers.data_format_inference`, which the float cases above do
        follow. Its SrcS branch returns Float32 for everything except
        Float16/Float16_b once ``dest_acc`` is on, so an Int8 input is programmed
        as a Float32 SrcS -- a conversion the unpacker does not perform, and one
        that would make a golden report 1.0 where the device holds the integer
        1. The blanket widening reads as a rule written for the float path;
        integer SrcS with accumulation is not swept, so nothing has forced the
        question. Left as-is here rather than copied, and noted so the next
        reader does not "fix" the divergence by aligning with it.
        """
        self._check_has_srcs()
        self._check_supported(l1_format)
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

    @staticmethod
    def _is_valid_srcs_format(
        l1_format: DataFormat, src_format: DataFormat, dest_acc: bool
    ) -> bool:
        """Whether an explicit SrcS storage format is one the unpacker can produce."""
        if src_format not in SRCS_STORAGE_FORMATS and src_format != l1_format:
            return False
        if dest_acc and l1_format in (DataFormat.Float16, DataFormat.Float16_b):
            return src_format == l1_format
        return True

    def _src_format(self, l1_format: DataFormat) -> DataFormat:
        """Architecture's L1 -> src-register format mapping.

        Float32 and Tf32 land in Tf32 (8-bit exponent, 10-bit mantissa).
        Everything else resolves to one of the two 16-bit exponent families, and
        integer formats pass through unchanged. Override where an architecture
        diverges; :meth:`src_format` does the support check.

        A format reaching the final pass-through stays in the src register as
        itself, which is only right for integers. A float that lands there is a
        format whose exponent family is unknown to :mod:`helpers.format_config`
        and which therefore needs a case here — see ``Fp8_e4m3``.
        """
        if l1_format in (DataFormat.Float32, DataFormat.Tf32):
            return DataFormat.Tf32
        if l1_format.is_mx_format():
            # The unpacker converts MX into the 8-bit-exponent family regardless of
            # the pack format, so math and Dest see bf16.
            return DataFormat.Float16_b
        if l1_format is DataFormat.Fp8_e4m3:
            # An L1-only encoding: every architecture that has it widens it to
            # Float16 in the register, alongside Float16 and Lf8 in the A-format
            # exponent family. It needs naming explicitly because it reports
            # neither exponent family -- is_exponent_A() and is_exponent_B() are
            # both False for it -- so without this it would fall through to the
            # pass-through below and stay Fp8_e4m3 in a src register, which no
            # architecture does.
            return DataFormat.Float16
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

        Which src formats an L1 format can land in is per-architecture and
        enforced -- see :meth:`_is_valid_src_format`. Within that, `src_format`
        defaults to the one the LLK would pick for `l1_format`.
        """
        # Validate whichever format we end up with, defaulted or not. Checking
        # only the explicit one would exempt exactly the common path, and the
        # default is derived from the input rather than from what the unpacker
        # can actually do -- which is how an Int32 input reached SrcA.
        defaulted = src_format is None
        if defaulted:
            # A wide float input does not determine the src family on its own,
            # the same ambiguity dest_format_for refuses. With dest_acc off the
            # device picks Float16 or Float16_b from the *output* format
            # (infer_unpack_out), and both clip: Float16 saturates above 65504
            # and flushes below 2**-14. Defaulting to Tf32 here would keep the
            # full fp32 range and silently disagree with silicon at the edges.
            # With dest_acc on the device really does use Tf32, so that case
            # needs no caller input.
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
        if not self._is_valid_src_format(l1_format, src_format):
            if src_format is DataFormat.Float32:
                raise ValueError(
                    "Float32 reaches a src register only as Tf32, Float16 or "
                    "Float16_b -- a 19-bit datum cannot hold it. Use "
                    "l1_to_dest for the unpack-to-Dest path, or l1_to_srcS, "
                    "which are the two destinations that keep full fp32."
                )
            legal = sorted(
                str(f) for f in self.UNPACK_TO_SRC_FORMATS.get(l1_format, frozenset())
            )
            if defaulted:
                raise ValueError(
                    f"{type(self).__name__} cannot unpack {l1_format} into "
                    f"SrcA/SrcB"
                    + (
                        f"; the legal src formats for it are {legal}."
                        if legal
                        else " at all -- it reaches Dest or SrcS only. Use "
                        "l1_to_dest, or l1_to_srcS."
                    )
                )
            raise ValueError(
                f"{type(self).__name__} cannot unpack {l1_format} as "
                f"{src_format}. "
                + (
                    f"Legal src formats for {l1_format}: {legal}."
                    if legal
                    else f"A src register can hold "
                    f"{sorted(str(f) for f in SRC_STORAGE_FORMATS)}."
                )
            )
        values = self.unpack_from_l1(l1_bytes, l1_format, **geometry)
        return self._to_src_storage(values, src_format)

    def _is_valid_src_format(
        self, l1_format: DataFormat, src_format: DataFormat
    ) -> bool:
        """Whether this architecture's unpacker can land `l1_format` as `src_format`.

        The pair matters, not just the target. A src-register format being
        real, and an L1 format being readable, does not make the conversion
        between them something the unpacker can do -- Int32 reaches Dest and
        SrcS but never SrcA/SrcB, and an integer input does not become a float
        in a src register.

        ``UNPACK_TO_SRC_FORMATS`` carries the legal pairs per architecture, and
        an architecture that leaves it empty is simply unmodelled here and gets
        the older, weaker check. SrcS is a different destination with its own
        rules, including the Float32 that SrcA/SrcB cannot hold, and is checked
        by :meth:`_is_valid_srcs_format` instead.
        """
        # No 32-bit datum reaches SrcA/SrcB on any architecture modelled here,
        # so this holds whether or not the pair table is populated. Wormhole's
        # cunpack_common.h is explicit for the integers -- Int32's "SrcA/SrcB
        # path: NOT possible (ISA doc explicitly states 'Not possible')", and
        # UInt32 valid only when targeting Dest -- and without this the
        # fallback's `src_format == l1_format` arm would hand back full int32
        # values for a register that cannot hold them.
        if src_format in (DataFormat.Float32, DataFormat.Int32, DataFormat.UInt32):
            return False
        if self.UNPACK_TO_SRC_FORMATS:
            return src_format in self.UNPACK_TO_SRC_FORMATS.get(l1_format, frozenset())
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

        **Float32 into a src register loses its low 13 mantissa bits.** A 19-bit
        datum cannot hold fp32, and the unpacker's format table allows a
        Float32 source to land only in Tf32, Float16 or Float16_b when the
        destination is a src register -- all of which keep 10. Full fp32
        survives only on the unpack-to-Dest path, which :meth:`l1_to_dest`
        models, and in SrcS; the ``Float32`` case below is reachable only from
        :meth:`l1_to_srcS`. Integer formats pass through unchanged.
        """
        if src_format is DataFormat.Float16:
            # 1+5+10 is IEEE fp16 exactly. Truncate first so the cast only has
            # to apply the range clamp -- casting straight to fp16 would round.
            truncated = DataTransferBlocks._truncate_src_mantissa(values)
            # The unpacker flushes to zero once the rebiased exponent hits 0,
            # so fp16 subnormals never reach the register. Flush *before* the
            # cast: the cast rounds to nearest, and the top subnormal band
            # (within half an ulp of 2**-14) would round up to the smallest
            # normal and sail through a flush applied afterwards.
            return flush_subnormals(truncated, torch.float16).to(torch.float16)
        if src_format in (DataFormat.Float16_b, DataFormat.Tf32):
            # Float16_b is an alias for Tf32 here -- see SRC_STORAGE_FORMATS.
            return DataTransferBlocks._truncate_src_mantissa(values)
        if src_format is DataFormat.Float32:
            return values.to(torch.float32)
        return values.to(format_dict[src_format])

    @staticmethod
    def _truncate_src_mantissa(values: torch.Tensor) -> torch.Tensor:
        """Keep the top 10 mantissa bits, truncating as the unpacker does."""
        return truncate_mantissa(values, SRC_MANT_BITS)
