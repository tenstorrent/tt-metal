# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import math
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import List, Optional, Tuple, Union

import ml_dtypes
import numpy as np

from .tile_constants import SRCS_SLICE_32B_ELEMENT_COUNT, SRCS_SLICE_ELEMENT_COUNT

# ============================================================================
# Data Format Classes
# ============================================================================


class DataFormatInfo:
    """
    A helper class that encapsulates metadata for a data format.

    Attributes:
        name (str): A human-readable name for the data format.
        byte_size (Fraction): The size in bytes of one unit of the data format.
    """

    def __init__(self, name: str, byte_size: Union[int, float, Fraction]):
        self.name = name
        self.byte_size = (
            byte_size if isinstance(byte_size, Fraction) else Fraction(byte_size)
        )

    def _byte_size_str(self) -> str:
        if self.byte_size.denominator == 1:
            return str(self.byte_size.numerator)
        return str(float(self.byte_size))

    def __str__(self) -> str:
        """Returns the string representation of the data format info."""
        return f"{self.name}/{self._byte_size_str()}B"

    def __repr__(self) -> str:
        """Returns the representation of the data format info."""
        return self.__str__()


class DataFormat(Enum):
    """
    An enumeration of data formats supported by the LLKs.
    Holds format name and byte size, and is extendable.
    """

    Float16 = DataFormatInfo("Float16", 2)
    Float16_b = DataFormatInfo("Float16_b", 2)
    Bfp8 = DataFormatInfo("Bfp8", 1)  # WH/BH specific
    Bfp8_b = DataFormatInfo("Bfp8_b", 1)  # WH/BH specific
    Bfp4_b = DataFormatInfo("Bfp4_b", 1)  # WH/BH specific
    Bfp2_b = DataFormatInfo("Bfp2_b", 1)  # WH/BH specific
    Float32 = DataFormatInfo("Float32", 4)
    Int32 = DataFormatInfo("Int32", 4)
    Tf32 = DataFormatInfo("Tf32", 3)
    UInt32 = DataFormatInfo("UInt32", 4)  # WH/BH specific
    Int16 = DataFormatInfo("Int16", 2)  # QSR specific
    UInt16 = DataFormatInfo("UInt16", 2)  # WH/BH specific
    Int8 = DataFormatInfo("Int8", 1)
    UInt8 = DataFormatInfo("UInt8", 1)
    MxFp8R = DataFormatInfo("MxFp8R", 1)  # QSR specific
    MxFp8P = DataFormatInfo("MxFp8P", 1)  # QSR specific
    MxFp6R = DataFormatInfo(
        "MxFp6R", 1
    )  # QSR specific - E3M2, 6 bits used in an 8-bit L1 container
    MxFp6P = DataFormatInfo(
        "MxFp6P", 1
    )  # QSR specific - E2M3, 6 bits used in an 8-bit L1 container
    MxFp4 = DataFormatInfo(
        "MxFp4", Fraction(1, 2)
    )  # QSR specific - 4 bits (0.5 bytes) per element
    MxFp4_2x_A = DataFormatInfo(
        "MxFp4_2x_A", Fraction(1, 2)
    )  # QSR specific - 2x-packed FP4 in Src Reg, 5-bit exp (FP16 family) - 10 bits per element in one double-datumed element in src reg.
    MxFp4_2x_B = DataFormatInfo(
        "MxFp4_2x_B", Fraction(1, 2)
    )  # QSR specific - 2x-packed FP4 in Src Reg, 8-bit exp (FP16_b family) - 10 bits per element in one double-datumed element in src reg.
    MxInt8 = DataFormatInfo("MxInt8", 1)  # QSR specific - S1.6, 8 bits per element
    MxInt4 = DataFormatInfo(
        "MxInt4", Fraction(1, 2)
    )  # QSR specific - S1.2, 4 bits (0.5 bytes) per element
    MxInt2 = DataFormatInfo(
        "MxInt2", Fraction(1, 4)
    )  # QSR specific - S1.0, 2 bits (0.25 bytes) per element
    Fp8_e4m3 = DataFormatInfo("Fp8_e4m3", 1)

    @property
    def cpp_enum_value(self) -> str:
        return f"DataFormat::{self.name}"

    @property
    def cpp_underlying_value(self) -> str:
        return f"ckernel::to_underlying(DataFormat::{self.name})"

    @property
    def size(self) -> Fraction:
        """Returns the byte size of the data format."""
        return self.value.byte_size

    def __str__(self) -> str:
        """Returns the string representation of the data format."""
        return self.value.name

    def is_integer(self) -> bool:
        """Checks if the data format is an integer type."""
        return self in {
            DataFormat.Int32,
            DataFormat.UInt32,
            DataFormat.Int16,
            DataFormat.UInt16,
            DataFormat.Int8,
            DataFormat.UInt8,
        }

    def needs_int8_math_config(self) -> bool:
        """Checks if the format requires int8 math mode in the ALU."""
        return self in {DataFormat.Int8, DataFormat.UInt8, DataFormat.Int32}

    def is_32_bit(self) -> bool:
        """Checks if the data format is a 32-bit type."""
        return self in {DataFormat.Float32, DataFormat.Int32, DataFormat.UInt32}

    def is_exponent_A(self) -> bool:
        """Checks if the data format is an exponent A format."""

        return self in {
            DataFormat.Float16,
            DataFormat.Bfp8,
            DataFormat.MxFp4_2x_A,
        }

    def is_exponent_B(self) -> bool:
        """Checks if the data format is an exponent B format."""
        return self in {
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
            DataFormat.Bfp4_b,
            DataFormat.Bfp2_b,
            DataFormat.Tf32,
            DataFormat.Float32,
            DataFormat.MxFp4_2x_B,
        }

    def num_bytes_per_tile(self, num_datums: int = 1024) -> int:
        """Returns the number of bytes per tile for the data format."""
        num_exponents = 0
        if self in {DataFormat.Bfp8, DataFormat.Bfp8_b}:
            num_exponents = num_datums // 16
        elif self in {DataFormat.Bfp4_b}:
            num_exponents = num_datums // 16
            return (num_datums // 2) + num_exponents
        elif self in {DataFormat.Bfp2_b}:
            num_exponents = num_datums // 16
            return (num_datums // 4) + num_exponents
        elif self.is_mx_format():
            # MX formats: 1 scale (E8M0, 8 bits) per 32 elements
            num_scales = num_datums // MX_FORMAT_BLOCK_SIZE
            # For MxFp4, self.size = 0.5, so convert to int to avoid float in l1_align
            element_bytes = int(self.size * num_datums)
            return l1_align(num_scales) + l1_align(element_bytes)
        # For formats with fractional byte sizes (e.g., hypothetically), ensure int result
        return int(self.size * num_datums) + num_exponents

    def is_float32(self) -> bool:
        """Checks if the data format is a Float32 type."""
        return self == DataFormat.Float32

    def is_mx_format(self) -> bool:
        """Checks if the data format is an MX (Microscaling) format."""
        return self in {
            DataFormat.MxFp8R,
            DataFormat.MxFp8P,
            DataFormat.MxFp6R,
            DataFormat.MxFp6P,
            DataFormat.MxFp4,
            DataFormat.MxInt8,
            DataFormat.MxInt4,
            DataFormat.MxInt2,
        }

    def is_mx_int_format(self) -> bool:
        """Checks if the data format is an MX integer format."""
        return self in {
            DataFormat.MxInt8,
            DataFormat.MxInt4,
            DataFormat.MxInt2,
        }

    def is_mx_fp_format(self) -> bool:
        """Checks if the data format is an MX floating-point format."""
        return self in {
            DataFormat.MxFp8R,
            DataFormat.MxFp8P,
            DataFormat.MxFp6R,
            DataFormat.MxFp6P,
            DataFormat.MxFp4,
        }

    def supports_l1_accumulation(self) -> bool:
        """Checks if the data format supports L1 accumulation"""
        return self in {
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.Float16,
            DataFormat.Float16_b,
        }


# ============================================================================
# MX (Microscaling) Format Value Maps
# ============================================================================

# Map of MX formats to their block sizes (all use 32-element blocks per OCP spec)
# This is a size of a basic block with one scale factor in MX formats.
# Bare in mind that MX formats can have multiple contiguous scales with corresponding values after.
MX_FORMAT_BLOCK_SIZE = 32


@dataclass(frozen=True)
class MxFpFormatSpec:
    """Canonical per-format parameters for an MX floating-point element format.

    Single source of truth shared by the pack element-quantizer, the unpack
    decoder, the block-aware compare, and the finfo-derived value maps below.
    The element layout is SxEyMz (sign, ``exp_bits`` exponent, ``man_bits``
    mantissa, no hidden bit); ``ml_dtype`` is the matching ml_dtypes scalar type.
    """

    ml_dtype: type
    exp_bits: int
    man_bits: int
    exp_bias: int
    exp_min_unbiased: int  # min-normal unbiased exponent
    exp_max_unbiased: int  # max-normal unbiased exponent
    has_nan: bool  # only E5M2 / E4M3 represent NaN
    # Largest *normal* mantissa at the max exponent. Defaults to the full field;
    # E4M3 reserves {exp=all-ones, man=all-ones} for NaN, so its man_max is one less.
    man_max_override: Optional[int] = None
    # Accepted adjacent-representable steps for the block-aware compare.
    max_compare_steps: int = 2

    @property
    def man_max(self) -> int:
        if self.man_max_override is not None:
            return self.man_max_override
        return (1 << self.man_bits) - 1

    @property
    def nan_code(self) -> Optional[int]:
        # +NaN = {sign 0, exp all-ones, man all-ones}; 0x7F for both E5M2 and E4M3.
        if not self.has_nan:
            return None
        return (1 << (self.exp_bits + self.man_bits)) - 1

    def element_quantizer_kwargs(self) -> dict:
        """Keyword args consumed by pack._quantize_to_mx_fp_element_codes."""
        return dict(
            exp_bits=self.exp_bits,
            man_bits=self.man_bits,
            exp_bias=self.exp_bias,
            exp_max_unbiased=self.exp_max_unbiased,
            exp_min_unbiased=self.exp_min_unbiased,
            man_max=self.man_max,
            nan_code=self.nan_code,
        )


# Canonical MX floating-point format parameters, keyed by DataFormat. Every other
# MX-FP table (pack/unpack element params, the block-aware compare params, and the
# finfo value maps below) derives from this so the per-format constants live in
# exactly one place.
MX_FP_SPECS: dict[DataFormat, MxFpFormatSpec] = {
    DataFormat.MxFp8R: MxFpFormatSpec(  # E5M2
        ml_dtypes.float8_e5m2,
        exp_bits=5,
        man_bits=2,
        exp_bias=15,
        exp_min_unbiased=-14,
        exp_max_unbiased=15,
        has_nan=True,
    ),
    DataFormat.MxFp8P: MxFpFormatSpec(  # E4M3 (top mantissa at max exp reserved for NaN)
        ml_dtypes.float8_e4m3fn,
        exp_bits=4,
        man_bits=3,
        exp_bias=7,
        exp_min_unbiased=-6,
        exp_max_unbiased=8,
        has_nan=True,
        man_max_override=6,
    ),
    DataFormat.MxFp6R: MxFpFormatSpec(  # E3M2
        ml_dtypes.float6_e3m2fn,
        exp_bits=3,
        man_bits=2,
        exp_bias=3,
        exp_min_unbiased=-2,
        exp_max_unbiased=4,
        has_nan=False,
    ),
    DataFormat.MxFp6P: MxFpFormatSpec(  # E2M3
        ml_dtypes.float6_e2m3fn,
        exp_bits=2,
        man_bits=3,
        exp_bias=1,
        exp_min_unbiased=0,
        exp_max_unbiased=2,
        has_nan=False,
    ),
    DataFormat.MxFp4: MxFpFormatSpec(  # E2M1
        ml_dtypes.float4_e2m1fn,
        exp_bits=2,
        man_bits=1,
        exp_bias=1,
        exp_min_unbiased=0,
        exp_max_unbiased=2,
        has_nan=False,
    ),
}

# The value maps below all derive from MX_FP_SPECS so each format's dtype is
# named exactly once.

# Maximum normal magnitude per format (OCP MX spec):
#   E5M2=57344, E4M3=448, E3M2=28.0, E2M3=7.5, E2M1=6.0.
MX_FORMAT_MAX_NORMAL = {
    fmt: float(ml_dtypes.finfo(spec.ml_dtype).max) for fmt, spec in MX_FP_SPECS.items()
}

# Minimum normal magnitude per format (OCP MX spec):
#   E5M2=2^-14, E4M3=2^-6, E3M2=0.25, E2M3=1.0, E2M1=1.0.
MX_FORMAT_MIN_NORMAL = {
    fmt: float(ml_dtypes.finfo(spec.ml_dtype).smallest_normal)
    for fmt, spec in MX_FP_SPECS.items()
}

# Maximum subnormal = smallest_normal × (2^man_bits − 1)/2^man_bits (largest
# mantissa, hidden bit 0): E5M2/E3M2 ×0.75, E4M3/E2M3 ×0.875, E2M1 ×0.5.
MX_FORMAT_MAX_SUBNORMAL = {
    fmt: float(ml_dtypes.finfo(spec.ml_dtype).smallest_normal)
    * (((1 << spec.man_bits) - 1) / (1 << spec.man_bits))
    for fmt, spec in MX_FP_SPECS.items()
}

# Minimum (smallest) subnormal magnitude per format, straight from the dtype.
MX_FORMAT_MIN_SUBNORMAL = {
    fmt: float(ml_dtypes.finfo(spec.ml_dtype).smallest_subnormal)
    for fmt, spec in MX_FP_SPECS.items()
}

# Map of MX formats to safe minimum magnitudes for stimulus generation.
MX_FORMAT_MIN_MAGNITUDE = {
    DataFormat.MxFp8R: 2.44e-4,
    DataFormat.MxFp8P: 0.0625,
    DataFormat.MxFp6R: 0.25,  # min normal (E3M2)
    DataFormat.MxFp6P: 1.0,  # min normal (E2M3)
    DataFormat.MxFp4: 1.0,
}


@dataclass(frozen=True)
class MxIntFormatSpec:
    """Canonical per-format parameters for an MX integer (MxInt) element format.

    Single source of truth for the symmetric fixed-point S1.k MxInt elements:
    2's-complement with an implicit 2^-k scale, one E8M0 block scale per
    32-element OCP block, no normal/subnormal split. Shared by the pack
    quantizer, the block-aware compare, and the max-magnitude map below.
    Sibling of MxFpFormatSpec (different encoding family, no exp/man split).
    """

    elem_scale: int  # 2^k implicit scale: int_val = round(scaled * elem_scale)
    elem_max: int  # symmetric clamp magnitude (the -(elem_max+1) code is left unused)
    max_ulp_steps: int  # accepted lattice steps for the block-aware compare

    @property
    def max_normal(self) -> float:
        """Max representable element magnitude = elem_max / elem_scale."""
        return self.elem_max / self.elem_scale


# Canonical MxInt format parameters, keyed by DataFormat. MX_INT_MAX below derives
# from this so the per-format constants live in exactly one place. Signed
# 2's-complement with an implicit power-of-2 scale (OCP spec); no normal/subnormal
# split.
#   MxInt8 (S1.6, scale 2^-6): symmetric ±127/64; MxInt4 (S1.2, scale 2^-2):
#   symmetric ±7/4; MxInt2 (S1.0, scale 2^0): symmetric ±1 (only -1/0/+1
#   representable; the -2 encoding 0b10 is left unused for symmetry).
#   (ws-tensix metadata says 15/8 for MxInt4, but its decode formula at
#   storage.py:419-434 actually yields 7/4 = raw_7 × 2^-2.)
MX_INT_SPECS: dict[DataFormat, MxIntFormatSpec] = {
    DataFormat.MxInt8: MxIntFormatSpec(elem_scale=64, elem_max=127, max_ulp_steps=3),
    DataFormat.MxInt4: MxIntFormatSpec(elem_scale=4, elem_max=7, max_ulp_steps=2),
    DataFormat.MxInt2: MxIntFormatSpec(elem_scale=1, elem_max=1, max_ulp_steps=1),
}

# Max representable element magnitude per MxInt format (= elem_max / elem_scale).
MX_INT_MAX = {fmt: spec.max_normal for fmt, spec in MX_INT_SPECS.items()}

# ============================================================================
# MX SrcS Slice L1 Layout
# ============================================================================
# Each SrcS slice is 8×16 = 128 elements.  In L1 a slice is stored as
# [scales padded to 16 B][elements padded to 16 B].
#
# These constants cover every MX float format whose SrcS elements occupy one
# byte each in L1 (MxFp8 and MxFp6 alike — MxFp6 is byte-padded, so the layout
# is identical). MxFp4 is excluded: it packs two elements per byte.


def l1_align(size: int) -> int:
    """Align *size* to the next 16B boundary."""
    l1_alignment = 16
    return (size + l1_alignment - 1) // l1_alignment * l1_alignment


# Per SrcS slice (8×16 = 128 elements, each 8-bit in L1):
#   scales:   128 / 32 = 4 bytes   → padded to 16 B
#   elements: 128 × 1 = 128 bytes  → already 16 B-aligned
#   total: 16 + 128 = 144 bytes per slice
MXFP_SRCS_SLICE_SCALE_BYTES = SRCS_SLICE_ELEMENT_COUNT // MX_FORMAT_BLOCK_SIZE  # 4
MXFP_SRCS_SLICE_ELEMENT_BYTES = SRCS_SLICE_ELEMENT_COUNT  # 128
MXFP_SRCS_SLICE_PACKED_BYTE_LEN = l1_align(MXFP_SRCS_SLICE_SCALE_BYTES) + l1_align(
    MXFP_SRCS_SLICE_ELEMENT_BYTES
)

# 32-bit SrcS mode (dest_acc): 4x16 = 64 elements per slice
#   scales:   64 / 32 = 2 bytes   -> padded to 16 B
#   elements: 64 x 1  = 64 bytes  -> already 16 B-aligned
#   total: 16 + 64 = 80 bytes per slice
MXFP_SRCS_SLICE_32B_SCALE_BYTES = (
    SRCS_SLICE_32B_ELEMENT_COUNT // MX_FORMAT_BLOCK_SIZE
)  # 2
MXFP_SRCS_SLICE_32B_ELEMENT_BYTES = SRCS_SLICE_32B_ELEMENT_COUNT  # 64
MXFP_SRCS_SLICE_32B_PACKED_BYTE_LEN = l1_align(
    MXFP_SRCS_SLICE_32B_SCALE_BYTES
) + l1_align(
    MXFP_SRCS_SLICE_32B_ELEMENT_BYTES
)  # 80

# ============================================================================
# MX (Microscaling) Format Utilities
# ============================================================================


def encode_e8m0_scale(max_abs_value, element_max_normal):
    """
    Encode a scale factor as E8M0 (8-bit exponent, no mantissa, bias=127).

    Per OCP MX spec Section 2.B (Gorodecky et al., 5 Nov 2024):
    e = ⌈log₂(amax/destmax)⌉ (round up to ensure no overflow)
    E8M0 = clamp(e, -127, 127) + bias

    This "round up" approach ensures post-scaling values do not exceed
    representable FP8 range, minimizing quantization error.

    Args:
        max_abs_value: Maximum absolute value in the block
        element_max_normal: Maximum normal value for element format (e.g., 448 for E4M3, 57344 for E5M2)

    Returns:
        E8M0 encoded scale (0-255), where 255 = NaN
    """
    # Handle special cases
    if max_abs_value == 0 or np.isnan(max_abs_value):
        return 127  # Scale = 2^0 = 1 (neutral scale)
    if np.isinf(max_abs_value):
        return 254  # Max representable scale

    # Calculate exponent: ceil(log2(max_value / element_max)) per OCP spec
    scale_ratio = max_abs_value / element_max_normal
    exponent = math.ceil(math.log2(scale_ratio))

    # Clamp to E8M0 range and add bias
    return int(max(-127, min(127, exponent)) + 127)


def decode_e8m0_scale(e8m0_value):
    """
    Decode E8M0 scale factor to float.

    Args:
        e8m0_value: E8M0 encoded scale (0-255)

    Returns:
        Scale factor as float (2**exponent), or NaN if e8m0_value = 255
    """
    if e8m0_value == 255:
        return float("nan")  # NaN encoding per OCP spec

    exponent = int(e8m0_value) - 127  # Remove bias
    return 2.0**exponent


@dataclass
class FormatConfig:
    """
    A data class that holds configuration details for formats passed to LLKs

    Attributes:
    unpack_A_src (DataFormat): The source format for source register A in the Unpacker, which is the format of our data in L1.
    unpack_A_dst (DataFormat): The destination format for source register A in the Unpacker, which is the format of our data in the source register.
    unpack_B_src (Optional[DataFormat]): The source format for source register B in the Unpacker, which is the format of our data in L1. Optional; defaults to `unpack_A_src` if `same_src_format=True`.
    unpack_B_dst (Optional[DataFormat]): The destination format for source register B in the Unpacker, which is the format of our data in the source register. Optional; defaults to `unpack_A_dst` if `same_src_format=True`.
    unpack_S_src (DataFormat): The source format for source register S in the Unpacker (L1). Defaults to `unpack_A_src` when omitted.
    unpack_S_dst (DataFormat): The destination format for source register S in the Unpacker (register). Defaults to `unpack_A_dst` when omitted.
    pack_src (DataFormat): The source format for the Packer.
    pack_dst (DataFormat): The destination format for the Packer.
    pack_S_src (DataFormat): The source format for the S path in the Packer. Defaults to `pack_src` when omitted.
    pack_S_dst (DataFormat): The destination format for the S path in the Packer. Defaults to `pack_dst` when omitted.
    math (DataFormat): The format used for _llk_math_ functions.

    Optional Parameters:
    same_src_format (bool): If `True`, the formats for source registers A and B will be the same for unpack operations.
    If `False`, source registers A and B have different formats formats must be specified. Defaults to `True`.

    unpack_B_src (Optional[DataFormat]): The source format for source register B in the Unpacker which is the format of our data in L1, used only if `same_src_format=False` i.e when source registers don't share the same formats we distinguish source register A and B formats.
    unpack_B_dst (Optional[DataFormat]): The destination format for source register B in the Unpacker, which is the format of our data in src register used only if `same_src_format=False` i.e when source registers don't share the same formats we distinguish source register A and B formats.
    unpack_S_src (Optional[DataFormat]): Optional override for `unpack_S_src`; otherwise defaults to `unpack_A_src`.
    unpack_S_dst (Optional[DataFormat]): Optional override for `unpack_S_dst`; otherwise defaults to `unpack_A_dst`.
    pack_S_src (Optional[DataFormat]): Optional override for `pack_S_src`; otherwise defaults to `pack_src`.
    pack_S_dst (Optional[DataFormat]): Optional override for `pack_S_dst`; otherwise defaults to `pack_dst`.

    Example:
    >>> formats = FormatConfig(
    >>>     unpack_A_src=DataFormat.Float32,
    >>>     unpack_A_dst=DataFormat.Float16,
    >>>     pack_src=DataFormat.Float16,
    >>>     pack_dst=DataFormat.Float32,
    >>>     math=DataFormat.Float32    # same_src_format defaults to True, thus our source registers have same formats and we don't need to define formats for source register B
    >>> )
    >>> print(formats.unpack_A_src)
    DataFormat.Float32
    >>> print(formats.unpack_B_src)
    DataFormat.Float32                 # B formats match A if same_src_format=True
    """

    unpack_A_src: DataFormat
    unpack_A_dst: DataFormat
    unpack_B_src: Optional[DataFormat]
    unpack_B_dst: Optional[DataFormat]
    unpack_S_src: DataFormat
    unpack_S_dst: DataFormat
    pack_src: DataFormat
    pack_dst: DataFormat
    pack_S_src: DataFormat
    pack_S_dst: DataFormat
    math: DataFormat
    sfpu_math: DataFormat

    def __init__(
        self,
        unpack_A_src: DataFormat,
        unpack_A_dst: DataFormat,
        pack_src: DataFormat,
        pack_dst: DataFormat,
        math: DataFormat,
        sfpu_math: Optional[
            DataFormat
        ] = None,  # SFPU-side math format; defaults to `math`. Differs only when the SFPU operates in a format with no native register/dest representation (e.g. UInt16 on Quasar, routed through an Int16 data path).
        same_src_format: bool = True,  # If True, A and B share unpack formats; omit unpack_B_src / unpack_B_dst (they are set from A).
        # Optional unpack_S_* and pack_S_* default to the A and main pack paths when omitted (mirrors common "S same as A" usage).
        unpack_B_src: Optional[DataFormat] = None,
        unpack_B_dst: Optional[DataFormat] = None,
        unpack_S_src: Optional[DataFormat] = None,
        unpack_S_dst: Optional[DataFormat] = None,
        pack_S_src: Optional[DataFormat] = None,
        pack_S_dst: Optional[DataFormat] = None,
    ):

        self.unpack_A_src = unpack_A_src
        self.unpack_A_dst = unpack_A_dst
        self.pack_src = pack_src
        self.pack_dst = pack_dst
        self.math = math
        self.sfpu_math = sfpu_math if sfpu_math is not None else math
        if same_src_format:
            self.unpack_B_src = unpack_A_src
            self.unpack_B_dst = unpack_A_dst
        else:
            if unpack_B_src is None or unpack_B_dst is None:
                raise ValueError(
                    "When same_src_format is False, both unpack_B_src and unpack_B_dst must be provided."
                )
            self.unpack_B_src = unpack_B_src
            self.unpack_B_dst = unpack_B_dst
        self.unpack_S_src = (
            unpack_S_src if unpack_S_src is not None else self.unpack_A_src
        )
        if unpack_S_dst is not None:
            self.unpack_S_dst = unpack_S_dst
        elif self.unpack_A_dst == DataFormat.MxFp4_2x_A:
            self.unpack_S_dst = DataFormat.Float16
        elif self.unpack_A_dst == DataFormat.MxFp4_2x_B:
            self.unpack_S_dst = DataFormat.Float16_b
        else:
            self.unpack_S_dst = self.unpack_A_dst
        self.pack_S_src = pack_S_src if pack_S_src is not None else self.pack_src
        self.pack_S_dst = pack_S_dst if pack_S_dst is not None else self.pack_dst

        # MxFp4_2x_A/B are 2x-packed Src Register formats. They have no L1, math, or pack
        # representation — only unpack_A_dst / unpack_B_dst (the in-register format) may use them.
        srcab_only = {DataFormat.MxFp4_2x_A, DataFormat.MxFp4_2x_B}
        for field_name, value in (
            ("unpack_A_src", self.unpack_A_src),
            ("unpack_B_src", self.unpack_B_src),
            ("unpack_S_src", self.unpack_S_src),
            ("unpack_S_dst", self.unpack_S_dst),
            ("math", self.math),
            ("sfpu_math", self.sfpu_math),
            ("pack_src", self.pack_src),
            ("pack_dst", self.pack_dst),
            ("pack_S_src", self.pack_S_src),
            ("pack_S_dst", self.pack_S_dst),
        ):
            if value in srcab_only:
                raise ValueError(
                    f"{value.name} is a 2x-packed SrcA/SrcB-only format and cannot be used "
                    f"as {field_name}. It is only valid for unpack_A_dst / unpack_B_dst. "
                    f"For L1 input use DataFormat.MxFp4."
                )

    @property
    def output_format(self) -> DataFormat:
        return self.pack_dst

    @property
    def input_format(self) -> DataFormat:
        return self.unpack_A_src

    @property
    def input_format_B(self) -> DataFormat:
        return self.unpack_B_src


FORMATS_CONFIG_STRUCT_RUNTIME = [
    """
struct FormatConfig
{
    std::uint32_t unpack_A_src = 0;
    std::uint32_t unpack_B_src = 0;
    std::uint32_t unpack_S_src = 0;
    std::uint32_t unpack_A_dst = 0;
    std::uint32_t unpack_B_dst = 0;
    std::uint32_t unpack_S_dst = 0;
    std::uint32_t math = 0;
    std::uint32_t sfpu_math = 0;
    std::uint32_t pack_src = 0;
    std::uint32_t pack_dst = 0;
    std::uint32_t pack_S_src = 0;
    std::uint32_t pack_S_dst = 0;
};
"""
]

FORMATS_CONFIG_STRUCT_COMPILETIME = [
    "// Formats struct",
    "struct FormatConfig",
    "{",
    "    const std::uint32_t unpack_A_src;",
    "    const std::uint32_t unpack_B_src;",
    "    const std::uint32_t unpack_S_src;",
    "    const std::uint32_t unpack_A_dst;",
    "    const std::uint32_t unpack_B_dst;",
    "    const std::uint32_t unpack_S_dst;",
    "    const std::uint32_t math;",
    "    const std::uint32_t sfpu_math;",
    "    const std::uint32_t pack_src;",
    "    const std::uint32_t pack_dst;",
    "    const std::uint32_t pack_S_src;",
    "    const std::uint32_t pack_S_dst;",
    "",
    "    constexpr FormatConfig(",
    "        std::uint32_t unpack_A_src_,",
    "        std::uint32_t unpack_B_src_,",
    "        std::uint32_t unpack_S_src_,",
    "        std::uint32_t unpack_A_dst_,",
    "        std::uint32_t unpack_B_dst_,",
    "        std::uint32_t unpack_S_dst_,",
    "        std::uint32_t math_,",
    "        std::uint32_t sfpu_math_,",
    "        std::uint32_t pack_src_,",
    "        std::uint32_t pack_dst_,",
    "        std::uint32_t pack_S_src_,",
    "        std::uint32_t pack_S_dst_) :",
    "        unpack_A_src(unpack_A_src_),",
    "        unpack_B_src(unpack_B_src_),",
    "        unpack_S_src(unpack_S_src_),",
    "        unpack_A_dst(unpack_A_dst_),",
    "        unpack_B_dst(unpack_B_dst_),",
    "        unpack_S_dst(unpack_S_dst_),",
    "        math(math_),",
    "        sfpu_math(sfpu_math_),",
    "        pack_src(pack_src_),",
    "        pack_dst(pack_dst_),",
    "        pack_S_src(pack_S_src_),",
    "        pack_S_dst(pack_S_dst_)",
    "    {",
    "    }",
    "};",
    "",
]

WORMHOLE_DATA_FORMAT_ENUM_VALUES = {
    DataFormat.Float32: 0,
    DataFormat.Float16: 1,
    DataFormat.Bfp8: 2,
    DataFormat.Tf32: 4,
    DataFormat.Float16_b: 5,
    DataFormat.Bfp8_b: 6,
    DataFormat.Bfp4_b: 7,
    DataFormat.Bfp2_b: 15,
    DataFormat.Int32: 8,
    DataFormat.UInt16: 9,
    DataFormat.Int8: 14,
    DataFormat.UInt32: 24,
    DataFormat.UInt8: 30,
}

BLACKHOLE_DATA_FORMAT_ENUM_VALUES = {
    DataFormat.Float32: 0,
    DataFormat.Float16: 1,
    DataFormat.Bfp8: 2,
    DataFormat.Tf32: 4,
    DataFormat.Float16_b: 5,
    DataFormat.Bfp8_b: 6,
    DataFormat.Bfp4_b: 7,
    DataFormat.Bfp2_b: 15,
    DataFormat.Int32: 8,
    DataFormat.UInt16: 9,
    DataFormat.Int8: 14,
    DataFormat.UInt32: 24,
    DataFormat.Fp8_e4m3: 26,
    DataFormat.UInt8: 30,
}

QUASAR_DATA_FORMAT_ENUM_VALUES = {
    DataFormat.Float32: 0,
    DataFormat.Tf32: 4,
    DataFormat.Float16: 1,
    DataFormat.Float16_b: 5,
    DataFormat.MxFp8R: 18,
    DataFormat.MxFp8P: 20,
    DataFormat.MxFp6R: 19,
    DataFormat.MxFp6P: 21,
    DataFormat.MxFp4: 22,
    DataFormat.MxInt8: 2,
    DataFormat.MxInt4: 3,
    DataFormat.MxInt2: 11,
    DataFormat.MxFp4_2x_A: 27,
    DataFormat.MxFp4_2x_B: 24,
    DataFormat.Int32: 8,
    DataFormat.Int8: 14,
    DataFormat.UInt8: 17,
    DataFormat.UInt16: 130,
    DataFormat.Int16: 9,
}


@dataclass
class InputOutputFormat:
    """
    A data class that holds configuration details for formats passed to LLKs.
    This class is used to hold input and output DataFormat that the client wants to test.
    They are used for format inference model to infer the rest of the formats for the LLk pipeline, instead of the user.

    If input_B is not specified, it defaults to the same as input (input_A).

    register_format_hint: optional opt-in for SrcA/SrcB-only register formats (e.g. MxFp4_2x_A / MxFp4_2x_B).
    When set, the inferred unpack_A_dst / unpack_B_dst become the hint instead of the default
    (e.g. for MxFp4 input, defaults to Float16_b). Only valid when input == MxFp4 today.
    """

    input: DataFormat
    output: DataFormat
    input_B: Optional[DataFormat] = None
    register_format_hint: Optional[DataFormat] = None

    def __init__(
        self,
        input_format: DataFormat,
        output_format: DataFormat,
        input_format_B: Optional[DataFormat] = None,
        register_format_hint: Optional[DataFormat] = None,
    ):
        self.input = input_format
        self.output = output_format
        self.input_B = input_format_B if input_format_B is not None else input_format
        self.register_format_hint = register_format_hint

    @property
    def output_format(self) -> DataFormat:
        return self.output

    @property
    def input_format(self) -> DataFormat:
        return self.input

    @property
    def input_format_B(self) -> DataFormat:
        return self.input_B

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        if self.register_format_hint is not None:
            return f"InputOutputFormat[L1_Input:{self.input},A(reg_hint):{self.register_format_hint},B(reg_hint):{self.register_format_hint},out:{self.output}]"
        return f"InputOutputFormat[L1_Input_A:{self.input},L1_Input_B:{self.input_B},out:{self.output}]"


def create_formats_for_testing(formats: List[Tuple[DataFormat]]) -> List[FormatConfig]:
    """
    A function that creates a list of FormatConfig objects from a list of DataFormat objects that client wants to test.
    This function is useful for creating a list of FormatConfig objects for testing multiple formats combinations
    and cases which the user has specifically defined and wants to particularly test instead of a full format flush.

    Args:
    formats (List[Tuple[DataFormat]]): A list of tuples of DataFormat objects for which FormatConfig objects need to be created.

    Returns:
    List[FormatConfig]: A list of FormatConfig objects created from the list of DataFormat objects passed as input.

    Example:
    >>> formats = [(DataFormat.Float16, DataFormat.Float32, DataFormat.Float16, DataFormat.Float32, DataFormat.Float32)]
    >>> format_configs = create_formats_for_testing(formats)
    >>> print(format_configs[0].unpack_A_src)
    DataFormat.Float16
    >>> print(format_configs[0].unpack_B_src)
    DataFormat.Float16
    """
    format_configs = []
    for format_tuple in formats:
        if len(format_tuple) == 5:
            format_configs.append(
                FormatConfig(
                    unpack_A_src=format_tuple[0],
                    unpack_A_dst=format_tuple[1],
                    pack_src=format_tuple[2],
                    pack_dst=format_tuple[3],
                    math=format_tuple[4],
                )
            )
        else:
            format_configs.append(
                FormatConfig(
                    unpack_A_src=format_tuple[0],
                    unpack_A_dst=format_tuple[1],
                    unpack_B_src=format_tuple[2],
                    unpack_B_dst=format_tuple[3],
                    pack_src=format_tuple[4],
                    pack_dst=format_tuple[5],
                    math=format_tuple[6],
                    same_src_format=False,
                )
            )
    return format_configs


def is_dest_acc_needed(format: InputOutputFormat) -> bool:
    """
    This function is called when a format configuration for input and output is called without dest accumulation.
    If the input-output combination is an outlier that is not supported when dest accumulation is on
    then the data format inference model will turn dest accumulation off for this combination to work.

    We must notify the user that this has happened and change the test output to reflect this.
    """
    return (
        format.input_format
        in [
            DataFormat.Bfp8_b,
            DataFormat.Bfp4_b,
            DataFormat.Bfp2_b,
            DataFormat.Float16_b,
        ]
        and format.output_format == DataFormat.Float16
    )
