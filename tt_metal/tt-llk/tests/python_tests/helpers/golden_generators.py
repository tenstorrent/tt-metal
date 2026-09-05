# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
import math
import os
import struct
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import ClassVar, Optional

import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    PackerReluType,
    ReduceDimension,
    ReducePool,
    SdpaOp,
    TopKSortDirection,
    format_dict,
    pack_relu_config,
)
from helpers.pack import (
    pack_mxfp4,
    pack_mxfp8p,
    pack_mxfp8r,
    pack_mxint2,
    pack_mxint4,
    pack_mxint8,
)
from helpers.sfpu_dispatch_constants import (
    CLAMP_MAX,
    CLAMP_MIN,
    HARDSHRINK_LAMBDA,
    INT_MAXMIN_SCALAR,
    LRELU_NEGATIVE_SLOPE,
    PRELU_SLOPE,
    RELU_MAX_THRESHOLD,
    RELU_MIN_THRESHOLD,
    SOFTPLUS_BETA,
    SOFTPLUS_THRESHOLD,
    SOFTSHRINK_LAMBDA,
    THRESHOLD_T,
    THRESHOLD_V,
    UNARY_COMP_THRESHOLD,
    UNARY_MAX_MIN_VALUE,
)
from helpers.sfpu_domains import dest_truncation_mask, nan_survives_to_l1
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.unpack import (
    unpack_mxfp4,
    unpack_mxfp8p,
    unpack_mxfp8r,
    unpack_mxint2,
    unpack_mxint4,
    unpack_mxint8,
)

from .bfp_format_utils import bfp2b_to_float16b as _bfp2b_to_float16b
from .bfp_format_utils import bfp4b_to_float16b as _bfp4b_to_float16b
from .bfp_format_utils import bfp8b_to_float16b as _bfp8b_to_float16b
from .logger import logger
from .tile_shape import construct_tile_shape

# Tile and face dimension constants
FACE_DIM = 16
ELEMENTS_PER_FACE = 256  # 16x16 = 256 elements per face
FACES_PER_TILE = 4
ELEMENTS_PER_TILE = 1024  # 4 faces × 256 elements
TILE_SIZE = 32
TILE_DIM = 32
TILE_DIMENSIONS = (32, 32)  # Tile dimensions as tuple

# Destination register capacity (in tiles)
MAX_TILES_16_BIT_DEST = 8
MAX_TILES_32_BIT_DEST = 4

golden_registry = {}


# Flush-to-zero (FTZ): values below a threshold get snapped to 0, matching what
# the hardware keeps as nonzero. bf16/fp32 flush subnormals, so they flush below
# their smallest normal; fp16 keeps subnormals, so it flushes below its smallest
# subnormal (i.e. effectively nothing). Other formats (BFP/MX) use 1e-37.
_FTZ_THRESHOLD = {
    DataFormat.Float32: float(torch.finfo(torch.float32).tiny),  # 2^-126 ~ 1.18e-38
    DataFormat.Float16_b: float(torch.finfo(torch.bfloat16).tiny),  # 2^-126 ~ 1.18e-38
    DataFormat.Float16: 2.0**-24,  # smallest fp16 subnormal ~ 5.96e-8
}


def _apply_ftz(result: torch.Tensor, data_format: DataFormat) -> torch.Tensor:
    """Flush subnormal-magnitude values in *result* to zero, matching hardware FTZ.

    The threshold is format-specific (see _FTZ_THRESHOLD above). Integer formats
    have no subnormals, so they are returned unchanged.
    """
    if data_format.is_integer():
        return result
    threshold = _FTZ_THRESHOLD.get(data_format, 1e-37)
    result_f32 = result.float()
    return torch.where(
        result_f32.abs() < threshold,
        torch.zeros_like(result_f32),
        result_f32,
    ).to(result.dtype)


def saturate_integer(result: torch.Tensor, data_format, torch_format) -> torch.Tensor:
    """Apply integer saturation during format conversion.

    Hardware saturates (clamps) values instead of wrapping on overflow.
    This handles downsizing (Int32->Int8), signed/unsigned conversions (UInt8->Int8),
    and any case where source values might exceed destination range.
    """
    iinfo = torch.iinfo(torch_format)
    is_unsigned = str(data_format).startswith("U")
    if is_unsigned:
        min_val, max_val = iinfo.min, iinfo.max
    else:
        # +1 because hardware uses sign-magnitude representation
        min_val, max_val = iinfo.min + 1, iinfo.max

    # Convert to intermediate type (int64 or int32) to avoid overflow during clamping
    # Use int64 when source can hold values outside int32 range (e.g. UInt32 is torch.int64)
    intermediate_type = (
        torch.int64 if result.dtype in (torch.uint32, torch.int64) else torch.int32
    )
    result = result.to(intermediate_type)
    result = torch.clamp(result, min_val, max_val)
    return result.to(torch_format)


def apply_l1_accumulation(
    partials: list[torch.Tensor],
    data_format: DataFormat,
) -> torch.Tensor:
    """
    Simulate L1 accumulation by summing partial results.

    With L1 acc enabled, the packer accumulates into the same output tile
    slots across multiple passes. For integer formats the hardware
    saturates at every step instead of wrapping, so the golden must
    clamp the running sum to the output range after each addition.

    Args:
        partials: List of tensors, one per accumulation pass, all of the
                  same shape. Each tensor represents the contribution
                  packed into the output tiles during that pass.
        data_format: DataFormat of the pack output.  When the format is integer,
               each accumulation step is saturated to the format's representable range.
    Returns:
        Element-wise sum of all partials (saturated per-step for integers).
    """
    needs_saturation = data_format.is_integer()

    accumulated = partials[0].clone()
    for partial in partials[1:]:
        if needs_saturation:
            wide = accumulated.to(torch.int64) + partial.to(torch.int64)
            accumulated = saturate_integer(wide, data_format, format_dict[data_format])
        else:
            accumulated += partial
    return accumulated


BFP_BLOCK_ELEMENTS = 16


def truncate_to_dest_width(
    tensor: torch.Tensor, dst_format: DataFormat
) -> torch.Tensor:
    """*tensor*'s low mantissa bits dropped, as a 16-bit Dest drops them on the unpack.

    One helper for both goldens, so a Dest-width change cannot update one call site and miss the
    other. test_sfpu_domains pins the masks against sfpu_domains' mantissa-width table.
    """
    masked = tensor.contiguous().view(torch.int32) & dest_truncation_mask(dst_format)
    return masked.view(torch.float32)


def _bfp_zero_nonfinite_blocks(operand):
    """Zero every finite element sharing a block with a non-finite one, in place.

    A block-float block shares one exponent across BFP_BLOCK_ELEMENTS elements, so a
    non-finite anywhere in the block takes that exponent -- and every finite neighbour --
    with it. Shared by Bfp8_b/Bfp4_b/Bfp2_b: the mantissa width differs, the destroyed
    shared exponent does not. Non-finite lanes keep their values.

    Mutates *operand* and returns it; callers use the in-place half.
    """
    if isinstance(operand, torch.Tensor):
        values = operand
    else:
        values = torch.as_tensor(operand, dtype=torch.float32)

    # reshape() on the boolean temporary, never on `values` itself -- a copy here is
    # harmless, whereas reshaping `values` could silently detach the in-place write.
    non_finite = ~torch.isfinite(values)
    blocks = non_finite.reshape(-1, BFP_BLOCK_ELEMENTS)
    block_is_tainted = (
        blocks.any(dim=1, keepdim=True).expand_as(blocks).reshape(non_finite.shape)
    )

    to_zero = block_is_tainted & ~non_finite
    if isinstance(operand, torch.Tensor):
        operand[to_zero] = 0.0
    else:
        for index in to_zero.flatten().nonzero().flatten().tolist():
            operand[index] = 0.0
    return operand


def convert_nan_to_inf(operand):
    """Replace every NaN with an infinity *of the same sign*, preserving the input type.

    Takes a Tensor or a list of floats and returns the same type, so downstream
    `result.to(...)` does not break on a tensor.

    The sign models the pack path, which rewrites exponent/mantissa and leaves the sign bit
    alone, so a signed NaN packs to -inf -- Neg(NaN) -> -inf is the case that measured it.
    Sound only because cast_to_dest_dtype keeps that sign across the Dest write and
    UnarySFPUGolden canonicalises a *generated* NaN's sign, which IEEE leaves unspecified;
    without both, the sign read here comes from the cast or the host libm, not the datum.
    """
    if isinstance(operand, torch.Tensor):
        return torch.where(
            torch.isnan(operand),
            torch.copysign(torch.full_like(operand, float("inf")), operand),
            operand,
        )
    return [math.copysign(math.inf, x) if math.isnan(x) else x for x in operand]


def sfpu_total_order_key(value: float) -> int:
    """Rank *value* under the total order the SFPU compares FP32 with.

    IEEE would compare these operands with the false-on-NaN rule, and with -0 == +0.

    The SFPU does not: `SFPGT`, `SFPLE` and `SFPSWAP` route through the ISA's
    `SignMagIsSmaller()`, a sign-magnitude bit-pattern compare that documents

        -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN

    on both arches (tt-isa-documentation, {BlackholeA0,WormholeB0}/TensixTile/
    TensixCoprocessor/SFPSWAP.md), so +NaN outranks every finite value.

    Limits of this model: Wormhole is measured rather than read off the ISA, since sfpi expands
    the compare in the backend.
    """
    bits = struct.unpack("<i", struct.pack("<f", value))[0]
    # The remap `SignMagIsSmaller()` performs: xor with the sign bit smeared down over the
    # magnitude, which is a mask of 0x7FFFFFFF where the sign bit is set and 0 where it is not.
    # So a negative with magnitude m ranks at -1 - m, putting -0.0 at -1 and strictly below +0.0
    # at 0. Ranking it at -m instead ties the two zeros and makes min/max operand-order-dependent.
    return bits ^ 0x7FFFFFFF if bits < 0 else bits


_order = sfpu_total_order_key


def sfpu_min(a: float, b: float) -> float:
    """min(a, b) under the SFPU's total order -- see sfpu_total_order_key."""
    return a if sfpu_total_order_key(a) <= sfpu_total_order_key(b) else b


def sfpu_max(a: float, b: float) -> float:
    """max(a, b) under the SFPU's total order -- see sfpu_total_order_key."""
    return a if sfpu_total_order_key(a) >= sfpu_total_order_key(b) else b


def sfpu_order_key_elementwise(tensor: torch.Tensor) -> torch.Tensor:
    """sfpu_total_order_key over a float tensor, elementwise.

    The vectorised twin of the scalar version, for the binary and reduce goldens: they hold
    whole tensors, where a Python loop per element is measurable across a sweep this size.
    """
    bits = tensor.to(torch.float32).contiguous().view(torch.int32)
    # Same remap as the scalar version, and it cannot overflow int32: the largest magnitude is
    # 0x7FFFFFFF, which XORs to INT32_MIN.
    return torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


def sfpu_min_elementwise(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """min(a, b) under the SFPU's total order, elementwise -- see sfpu_order_key_elementwise."""
    return torch.where(
        sfpu_order_key_elementwise(a) <= sfpu_order_key_elementwise(b), a, b
    )


def sfpu_max_elementwise(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """max(a, b) under the SFPU's total order, elementwise -- see sfpu_order_key_elementwise."""
    return torch.where(
        sfpu_order_key_elementwise(a) >= sfpu_order_key_elementwise(b), a, b
    )


def sfpu_relu_max(value: float, threshold: float) -> float:
    """The golden twin of `_relu_max_body_`, which several kernels share verbatim.

        v_if (result > threshold) result = threshold;
        v_if (result < 0.0f)      result = 0.0f;

    The first compare is against a vector and so uses the total order, which puts a NaN above
    the threshold and replaces it; the relu clamp then sees a finite value. The order is not
    interchangeable -- relu first would leave the NaN in place.

    The relu clamp reads the order key's sign, so it fires for -0.0 as well and this returns +0.0
    there. Unspecified on hardware either way: that branch is SFPSETCC, whose contract holds only
    "provided that VC is neither negative zero nor any kind of NaN".
    """
    clamped = sfpu_min(value, threshold)
    return 0.0 if sfpu_total_order_key(clamped) < 0 else clamped


def sfpu_clamp(value: float, low: float, high: float) -> float:
    """clamp under the SFPU's total order, in the kernel's order of operations.

    Metal `calculate_clamp` and `calculate_hardtanh` (`sfpi::clamp`) are both this same
    max-then-min composition of SFPSWAP min/max, so one golden models both. A +NaN
    outranks every value: the max leaves it in place and the min lands it on *high*,
    where torch.clamp would keep IEEE semantics and return NaN.
    """
    return sfpu_min(sfpu_max(value, low), high)


def cast_to_dest_dtype(values: torch.Tensor, dtype) -> torch.Tensor:
    """Cast fp32 *values* to a Dest *dtype*, keeping the sign of any NaN.

    torch's fp32 -> bfloat16 cast canonicalises every NaN to 0xFFFF, sign set, turning a
    positive NaN negative. Hardware does not: a 16-bit Dest holds the top half of the fp32
    pattern, so the sign survives and the pack path's NaN -> inf substitution reads it
    (convert_nan_to_inf). Unrepaired, the golden gives -inf for *every* NaN reaching a
    Float16_b Dest -- right for Neg(NaN), wrong for the tranche's other four, one accident.

    Only bfloat16 is affected; torch's fp16 cast carries the sign through. It must be fixed
    here, not in convert_nan_to_inf, because untilize_block reorders lanes in between,
    leaving no lane-aligned fp32 sign source.
    """
    out = values.to(dtype)
    if dtype is not torch.bfloat16:
        return out
    nan = torch.isnan(values)
    if not bool(nan.any()):
        return out
    # Repair the NaN lanes by the bit pattern rather than by value: torch offers no way to
    # build a negative bfloat16 NaN from a float, because .to(), full_like() and neg() all
    # normalise the sign. Taking the top 16 bits of the fp32 pattern is what a 16-bit Dest
    # does anyway, and it carries the sign and the NaN-ness across together.
    top_half = (values.view(torch.int32) >> 16).to(torch.int16)
    return torch.where(nan, top_half, out.view(torch.int16)).view(torch.bfloat16)


def convert_inf_to_value(operand, inf_value: float):
    """Replace every +inf with *inf_value*, preserving the input type.

    Accepts a torch.Tensor or a plain list of floats and returns the same
    type so that downstream code (e.g. `result.to(...)`) does not break
    when the caller passes a tensor.
    """
    if isinstance(operand, torch.Tensor):
        return torch.where(
            operand == math.inf,
            torch.full_like(operand, inf_value),
            operand,
        )

    return [inf_value if x == math.inf else x for x in operand]


def calculate_fractional_part(mantissa_value):
    fraction_value = 0.0
    divisor = 1.0  # Start with 2^0 = 1
    for bit in mantissa_value:
        if bit == "1":
            fraction_value += 1 / divisor
        divisor *= 2
    return fraction_value


def reassemble_float_after_fidelity(data_format, sgn1, sgn2, exp1, exp2, mant1, mant2):

    exponent1 = exp1.to(torch.int16)
    exponent2 = exp2.to(torch.int16)

    if data_format in [
        DataFormat.Float16_b,
        DataFormat.Bfp8_b,
        DataFormat.Bfp4_b,
        DataFormat.Bfp2_b,
        DataFormat.Float32,
    ]:
        exponent1 = exponent1 - 127
        exponent2 = exponent2 - 127
    elif data_format == DataFormat.Float16:
        exponent1 = exponent1 - 15
        exponent2 = exponent2 - 15
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

    mantissa1 = []
    mantissa2 = []

    # Convert mantissa tensor values to binary strings before passing to calculate_fractional_part
    for m1 in mant1:
        mantissa1.append(calculate_fractional_part(format(int(m1.item()), "011b")))
    for m2 in mant2:
        mantissa2.append(calculate_fractional_part(format(int(m2.item()), "011b")))

    reconstructed1 = ((-1.0) ** sgn1) * (2.0**exponent1) * torch.tensor(mantissa1)
    reconstructed2 = ((-1.0) ** sgn2) * (2.0**exponent2) * torch.tensor(mantissa2)

    torch_format = format_dict.get(data_format, format_dict[DataFormat.Float16_b])

    return reconstructed1.to(torch_format), reconstructed2.to(torch_format)


def register_golden(cls):
    """Register a golden class by its type."""
    golden_registry[cls] = cls()
    return cls


def get_golden_generator(cls):
    """Retrieve the registered golden class instance."""
    if cls not in golden_registry:
        raise KeyError(f"Golden class {cls.__name__} is not registered.")
    return golden_registry[cls]


def _dummy_zeros(*operands, **kwargs):
    # Size the dummy tensor from the caller's tile-shape kwargs when they are
    # all present (num_faces * face_r_dim * FACE_DIM per tile, times tile_cnt),
    # so callers that strictly size-check the result (e.g. untilize_block on a
    # 16x32 tiny tile) get a correctly-sized tensor.
    num_faces = kwargs.get("num_faces")
    face_r_dim = kwargs.get("face_r_dim")
    tile_cnt = kwargs.get("tile_cnt")
    # MatmulGolden callers pass operand shapes instead of a tile geometry; its
    # result is rows(A) x cols(B).
    dims_a = kwargs.get("input_A_dimensions")
    dims_b = kwargs.get("input_B_dimensions")
    if num_faces is not None and face_r_dim is not None and tile_cnt is not None:
        size = tile_cnt * num_faces * face_r_dim * FACE_DIM
    elif dims_a is not None and dims_b is not None:
        size = dims_a[0] * dims_b[1]
    else:
        # Nothing named the geometry, so fall back to the first tensor operand:
        # for the elementwise goldens the result is exactly as long as its
        # inputs. This is the historical ELEMENTS_PER_TILE (1024) whenever the
        # operand is a whole tile, and it is what keeps callers that size-check
        # a PARTIAL-tile result (an 8x32 SDPA tile, say) working -- a fixed 1024
        # would blow up when they combine the result with their own operands.
        operand = next((arg for arg in operands if isinstance(arg, torch.Tensor)), None)
        size = ELEMENTS_PER_TILE if operand is None else operand.numel()
    return torch.zeros(size, dtype=torch.bfloat16)


class DummyGoldenGenerator:
    def __call__(self, *args, **kwargs):
        return _dummy_zeros(*args, **kwargs)

    def transpose_faces_multi_tile(self, *args, **kwargs):
        return _dummy_zeros(*args, **kwargs)

    def transpose_within_faces_multi_tile(self, *args, **kwargs):
        return _dummy_zeros(*args, **kwargs)

    def accumulate_l1(self, *args, **kwargs):
        return _dummy_zeros(*args, **kwargs)


def dummy_golden_generator(cls):
    return DummyGoldenGenerator()


class ProxyMode(Enum):
    LOAD_GOLDEN = 1
    CACHE_GOLDEN = 2


# Proxy is used to allow test infra to only generate stimuli
class GeneratorProxy:
    TEMP_RESULT: ClassVar
    MODE: ClassVar[ProxyMode]

    STIMULI_CACHE_ROOT: ClassVar[Path]

    def __init__(self, wrapped_generator):
        self.wrapped_generator = wrapped_generator

    def __call__(self, *args, **kwds):
        logger.debug(f"Generator object call with mode {GeneratorProxy.MODE}")

        if os.environ.get("PYTEST_CURRENT_TEST", "").startswith(
            "test_fused"
        ) or os.environ.get("PYTEST_CURRENT_TEST", "").startswith("test_zzz_pack"):
            return self.wrapped_generator(*args, **kwds)

        if GeneratorProxy.MODE == ProxyMode.LOAD_GOLDEN:
            stimuli_id = sha256(
                os.environ.get("PYTEST_CURRENT_TEST", "").encode()
            ).hexdigest()
            golden_path = GeneratorProxy.STIMULI_CACHE_ROOT / stimuli_id / "golden.pt"
            result = torch.load(golden_path)
        elif GeneratorProxy.MODE == ProxyMode.CACHE_GOLDEN:
            result = self.wrapped_generator(*args, **kwds)
            # We cache tensor value in TEMP_RESULT when we call Stimuli_Config.save_to_caches
            GeneratorProxy.TEMP_RESULT = result
        else:
            raise ValueError("GeneratorProxy mode not set to a valid value!")

        return result

    def __getattr__(self, name):
        attr = getattr(self.wrapped_generator, name)

        if callable(attr):

            def wrapper(*args, **kwargs):
                logger.debug(f"Wrapper call with mode {GeneratorProxy.MODE}")
                if os.environ.get("PYTEST_CURRENT_TEST", "").startswith(
                    "test_fused"
                ) or os.environ.get("PYTEST_CURRENT_TEST", "").startswith(
                    "test_zzz_pack"
                ):
                    return attr(*args, **kwargs)

                if GeneratorProxy.MODE == ProxyMode.LOAD_GOLDEN:
                    stimuli_id = sha256(
                        os.environ.get("PYTEST_CURRENT_TEST", "").encode()
                    ).hexdigest()
                    golden_path = (
                        GeneratorProxy.STIMULI_CACHE_ROOT / stimuli_id / "golden.pt"
                    )
                    result = torch.load(golden_path)

                elif GeneratorProxy.MODE == ProxyMode.CACHE_GOLDEN:
                    result = attr(*args, **kwargs)
                    # We cache tensor value in TEMP_RESULT when we call Stimuli_Config.save_to_caches
                    GeneratorProxy.TEMP_RESULT = result

                return result

            return wrapper

        return attr

    def __str__(self):
        return str(self.wrapped_generator)

    def __repr__(self):
        return repr(self.wrapped_generator)


def get_golden_proxied(cls):
    """Retrieve the registered golden class instance."""
    if cls not in golden_registry:
        raise KeyError(f"Golden class {cls.__name__} is not registered.")
    return GeneratorProxy(golden_registry[cls])


def quantize_mx_stimuli(
    tensor: torch.Tensor, data_format: DataFormat, num_faces: int = 4
) -> torch.Tensor:
    """
    Quantize MX format stimuli by performing pack→unpack roundtrip.

    This simulates the quantization that occurs when data is stored in MX format
    in L1 memory and then unpacked by hardware. The golden model should use
    quantized values to match what hardware actually sees.

    The L1 layout (flat vs SrcS) does not affect quantization — the same
    scales and FP8 elements are produced regardless of byte arrangement.

    Args:
        tensor: Input tensor (bfloat16 values)
        data_format: MX format (MxFp8R, MxFp8P, or MxFp4)
        num_faces: Number of faces (1, 2, or 4)

    Returns:
        Quantized tensor (bfloat16 values after pack→unpack roundtrip)

    Raises:
        ValueError: If data_format is not an MX format, num_faces is invalid, or tensor size is incorrect
    """
    # Validate data format
    if not data_format.is_mx_format():
        raise ValueError(
            f"quantize_mx_stimuli only supports MX formats, got {data_format}"
        )

    # Validate num_faces
    if num_faces not in [1, 2, 4]:
        raise ValueError(f"num_faces must be 1, 2, or 4, got {num_faces}")

    # Validate tensor size matches expected for num_faces
    elements_per_face = 256
    expected_elements = elements_per_face * num_faces
    actual_elements = tensor.numel()

    if actual_elements < expected_elements:
        raise ValueError(
            f"Tensor has {actual_elements} elements, but need at least {expected_elements} "
            f"for {num_faces} face(s)"
        )

    # Quantize based on format
    match data_format:
        case DataFormat.MxFp8R:
            packed = pack_mxfp8r(tensor, num_faces=num_faces)
            return unpack_mxfp8r(packed, num_faces=num_faces)
        case DataFormat.MxFp8P:
            packed = pack_mxfp8p(tensor, num_faces=num_faces)
            return unpack_mxfp8p(packed, num_faces=num_faces)
        case DataFormat.MxFp4:
            packed = pack_mxfp4(tensor, num_faces=num_faces)
            return unpack_mxfp4(packed, num_faces=num_faces)
        case DataFormat.MxInt8:
            packed = pack_mxint8(tensor, num_faces=num_faces)
            return unpack_mxint8(packed, num_faces=num_faces)
        case DataFormat.MxInt4:
            packed = pack_mxint4(tensor, num_faces=num_faces)
            return unpack_mxint4(packed, num_faces=num_faces)
        case DataFormat.MxInt2:
            packed = pack_mxint2(tensor, num_faces=num_faces)
            return unpack_mxint2(packed, num_faces=num_faces)
        case _:
            # This should never happen due to validation above, but kept for safety
            raise ValueError(f"Unsupported MX format: {data_format}")


def quantize_mx_tensor_chunked(
    tensor: torch.Tensor, data_format: DataFormat
) -> torch.Tensor:
    """
    Quantize MX format tensor by processing in chunks.

    Args:
        tensor: Input tensor (bfloat16 values)
        data_format: MX format (MxFp8R, MxFp8P, or MxFp4)

    Returns:
        Quantized tensor (bfloat16 values)
    """
    tensor = tensor if isinstance(tensor, torch.Tensor) else torch.tensor(tensor)
    tensor_size = tensor.numel()

    if tensor_size == 0:
        return tensor

    # Pre-allocate output tensor for better performance
    quantized = torch.zeros_like(tensor)
    idx = 0
    # FACE_R_DIM support is not implemented yet, so we only support 4 faces for now.
    # Chunk size lookup: (min_size, chunk_size, num_faces)
    chunk_configs = [(1024, 1024, 4), (512, 512, 2), (256, 256, 1)]

    while idx < tensor_size:
        remaining = tensor_size - idx

        # Select chunk configuration based on remaining elements
        for min_size, chunk_size, num_faces in chunk_configs:
            if remaining >= min_size:
                break
        else:
            # Handle case where remaining < 256
            chunk_size, num_faces = 256, 1

        actual_chunk_size = min(chunk_size, remaining)
        chunk = tensor[idx : idx + actual_chunk_size]

        # Pad only if necessary
        if actual_chunk_size < chunk_size:
            padding = torch.zeros(
                chunk_size - actual_chunk_size, dtype=chunk.dtype, device=chunk.device
            )
            chunk = torch.cat([chunk, padding])

        # Quantize chunk
        quantized_chunk = quantize_mx_stimuli(chunk, data_format, num_faces=num_faces)

        # Write directly to output tensor (avoid list append + concat)
        quantized[idx : idx + actual_chunk_size] = quantized_chunk[:actual_chunk_size]
        idx += actual_chunk_size

    return quantized


def quantize_input_to_unpack_format(
    operand: torch.Tensor,
    input_format: Optional[DataFormat],
    *,
    all_mx_formats: bool = False,
) -> torch.Tensor:
    """
    Quantize input stimuli to match the values visible after hardware unpack.

    Some callers only model MXFP4 today; keep that as the default and let broader
    MX golden paths opt in explicitly.
    """
    if input_format == DataFormat.Bfp2_b:
        return _bfp2b_to_float16b(operand)
    if input_format == DataFormat.Bfp4_b:
        return _bfp4b_to_float16b(operand)
    if input_format == DataFormat.Bfp8_b:
        return _bfp8b_to_float16b(operand)
    if input_format is not None and input_format.is_mx_format():
        if all_mx_formats or input_format == DataFormat.MxFp4:
            return quantize_mx_tensor_chunked(operand, input_format)
        return operand
    return operand


class SrcFormatModel:
    """
    Source register holds data in TF32 format.

    This class is supposed to model how input data is converted to the source register format.
    """

    @staticmethod
    def to_src_format(format_from: DataFormat, tensor: torch.Tensor) -> torch.Tensor:
        """Returns tuple (matrix_sign, matrix_exponent, matrix_mantissa)"""
        CONVERSION_MAP = {
            DataFormat.Bfp8_b: SrcFormatModel._bfp8b_to_tf32,
            DataFormat.Bfp4_b: SrcFormatModel._bfp8b_to_tf32,
            DataFormat.Bfp2_b: SrcFormatModel._bfp8b_to_tf32,
            DataFormat.Float16_b: SrcFormatModel._fp16b_to_tf32,
            DataFormat.Float16: SrcFormatModel._fp16_to_tf32,
            DataFormat.Tf32: SrcFormatModel._fp32_to_tf32,
            DataFormat.Float32: SrcFormatModel._fp32_to_tf32,
            DataFormat.MxFp8R: SrcFormatModel._mxfp8r_to_tf32,
            DataFormat.MxFp8P: SrcFormatModel._mxfp8p_to_tf32,
            DataFormat.MxFp4: SrcFormatModel._mxfp4_to_tf32,
            DataFormat.MxInt8: SrcFormatModel._mxint8_to_tf32,
            DataFormat.MxInt4: SrcFormatModel._mxint4_to_tf32,
            DataFormat.MxInt2: SrcFormatModel._mxint2_to_tf32,
            DataFormat.Fp8_e4m3: SrcFormatModel._fp8_e4m3_to_tf32,
        }

        # todo: value error

        return CONVERSION_MAP[format_from](tensor)

    @staticmethod
    def _exponent_bias(exponent_width: int) -> int:
        return (1 << (exponent_width - 1)) - 1

    @staticmethod
    def _bfp8b_to_tf32(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PyTorch doesn't natively support bfp8, so it's implemented as bfloat16 in test infra"""

        return SrcFormatModel._fp16b_to_tf32(tensor)

    @staticmethod
    def _fp16b_to_tf32(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Handles Float16_b (and Bfp8_b)"""

        tensor_raw = tensor.to(torch.bfloat16).view(torch.uint16).to(torch.int64)

        BFP16_MANT_WIDTH = 7
        BFP16_EXP_WIDTH = 8
        BFP16_SIGN_WIDTH = 1

        BFP16_MANT_SHAMT = 0
        BFP16_EXP_SHAMT = BFP16_MANT_WIDTH
        BFP16_SIGN_SHAMT = BFP16_MANT_WIDTH + BFP16_EXP_WIDTH

        BFP16_MANT_MASK = ((1 << BFP16_MANT_WIDTH) - 1) << BFP16_MANT_SHAMT
        BFP16_EXP_MASK = ((1 << BFP16_EXP_WIDTH) - 1) << BFP16_EXP_SHAMT
        BFP16_SIGN_MASK = ((1 << BFP16_SIGN_WIDTH) - 1) << BFP16_SIGN_SHAMT

        sign = (tensor_raw & BFP16_SIGN_MASK) >> BFP16_SIGN_SHAMT
        exp = (tensor_raw & BFP16_EXP_MASK) >> BFP16_EXP_SHAMT
        mant = (tensor_raw & BFP16_MANT_MASK) >> BFP16_MANT_SHAMT

        # apply exponent bias
        exp = exp - SrcFormatModel._exponent_bias(BFP16_EXP_WIDTH)

        # when converting BFPx -> TF32, 3 LSBs are implied 0
        BFP16_TF32_MANT_RIGHT_PAD = 3
        mant = mant << BFP16_TF32_MANT_RIGHT_PAD

        # handle MSB is implied 1
        mant = mant | (1 << (BFP16_MANT_WIDTH + BFP16_TF32_MANT_RIGHT_PAD))

        return (sign, exp, mant)

    @staticmethod
    def _fp16_to_tf32(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Handles Float16"""

        tensor_raw = tensor.to(torch.float16).view(torch.uint16).to(torch.int64)

        FP16_MANT_WIDTH = 10
        FP16_EXP_WIDTH = 5
        FP16_SIGN_WIDTH = 1

        FP16_MANT_SHAMT = 0
        FP16_EXP_SHAMT = FP16_MANT_WIDTH
        FP16_SIGN_SHAMT = FP16_MANT_WIDTH + FP16_EXP_WIDTH

        FP16_MANT_MASK = ((1 << FP16_MANT_WIDTH) - 1) << FP16_MANT_SHAMT
        FP16_EXP_MASK = ((1 << FP16_EXP_WIDTH) - 1) << FP16_EXP_SHAMT
        FP16_SIGN_MASK = ((1 << FP16_SIGN_WIDTH) - 1) << FP16_SIGN_SHAMT

        sign = (tensor_raw & FP16_SIGN_MASK) >> FP16_SIGN_SHAMT
        exp = (tensor_raw & FP16_EXP_MASK) >> FP16_EXP_SHAMT
        mant = (tensor_raw & FP16_MANT_MASK) >> FP16_MANT_SHAMT

        # apply exponent bias
        exp = exp - SrcFormatModel._exponent_bias(FP16_EXP_WIDTH)

        # handle MSB is implied 1
        mant = mant | (1 << FP16_MANT_WIDTH)

        return (sign, exp, mant)

    @staticmethod
    def _fp32_to_tf32(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Handles Float32"""

        tensor_raw = tensor.to(torch.float32).view(torch.uint32).to(torch.int64)

        FP32_MANT_WIDTH = 23
        FP32_EXP_WIDTH = 8
        FP32_SIGN_WIDTH = 1

        FP32_MANT_SHAMT = 0
        FP32_EXP_SHAMT = FP32_MANT_WIDTH
        FP32_SIGN_SHAMT = FP32_MANT_WIDTH + FP32_EXP_WIDTH

        FP32_MANT_MASK = ((1 << FP32_MANT_WIDTH) - 1) << FP32_MANT_SHAMT
        FP32_EXP_MASK = ((1 << FP32_EXP_WIDTH) - 1) << FP32_EXP_SHAMT
        FP32_SIGN_MASK = ((1 << FP32_SIGN_WIDTH) - 1) << FP32_SIGN_SHAMT

        sign = (tensor_raw & FP32_SIGN_MASK) >> FP32_SIGN_SHAMT
        exp = (tensor_raw & FP32_EXP_MASK) >> FP32_EXP_SHAMT
        mant = (tensor_raw & FP32_MANT_MASK) >> FP32_MANT_SHAMT

        FP32_TF32_MANT_RIGHT_TRUNC = 13

        # apply exponent bias
        exp = exp - SrcFormatModel._exponent_bias(FP32_EXP_WIDTH)

        # when converting FP32 -> TF32, 13 LSBs are truncated
        mant = mant >> FP32_TF32_MANT_RIGHT_TRUNC

        # handle MSB is implied 1
        mant = mant | (1 << (FP32_MANT_WIDTH - FP32_TF32_MANT_RIGHT_TRUNC))

        return (sign, exp, mant)

    @staticmethod
    def _mxfp8r_to_tf32(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Handles MXFP8R format (MXFP8 E5M2 variant).

        Golden generators work on the original stimuli data (before compression).
        MXFP8R stimuli are generated as torch.bfloat16, so we delegate to Float16_b conversion.
        The pack/unpack functions handle the MXFP8 compression/decompression separately.
        """
        return SrcFormatModel._fp16b_to_tf32(tensor)

    @staticmethod
    def _mxfp8p_to_tf32(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Handles MXFP8P format (MXFP8 E4M3 variant).

        Golden generators work on the original stimuli data (before compression).
        MXFP8P stimuli are generated as torch.bfloat16, so we delegate to Float16_b conversion.
        The pack/unpack functions handle the MXFP8 compression/decompression separately.
        """
        return SrcFormatModel._fp16b_to_tf32(tensor)

    @staticmethod
    def _mxfp4_to_tf32(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Handles MxFp4 format (MX E2M1 variant).

        Golden generators work on the original stimuli data (before compression).
        MxFp4 stimuli are generated as torch.bfloat16, so we delegate to Float16_b conversion.
        The pack/unpack functions handle the MxFp4 compression/decompression separately.
        """
        return SrcFormatModel._fp16b_to_tf32(tensor)

    @staticmethod
    def _mxint8_to_tf32(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Handles MxInt8 format (signed S1.6 elements with E8M0 block exponent).

        MxInt8 is an L1-only storage format; hardware unpacks it into Float16/Float16_b/TF32
        in the source registers. Golden generators work on the original stimuli stored as
        torch.bfloat16, so we delegate to Float16_b conversion. The pack/unpack functions
        handle the MxInt8 integer-quantization roundtrip separately.
        """
        return SrcFormatModel._fp16b_to_tf32(tensor)

    @staticmethod
    def _mxint4_to_tf32(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Handles MxInt4 format (signed S1.2 elements with E8M0 block exponent).

        L1-only storage format like MxInt8; hardware unpacks to Float16/Float16_b/TF32 in
        the source registers. Stimuli are stored as torch.bfloat16, so we delegate to
        Float16_b conversion. The pack/unpack functions handle the MxInt4 quantization
        roundtrip (2 nibbles per byte) separately.
        """
        return SrcFormatModel._fp16b_to_tf32(tensor)

    @staticmethod
    def _mxint2_to_tf32(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Handles MxInt2 format (signed S1.0 elements with E8M0 block exponent).

        L1-only storage format like MxInt8/MxInt4; hardware unpacks to
        Float16/Float16_b/TF32 in the source registers. Stimuli are stored as
        torch.bfloat16, so we delegate to Float16_b conversion. The
        pack/unpack functions handle the MxInt2 quantization roundtrip (4
        crumbs per byte) separately.
        """
        return SrcFormatModel._fp16b_to_tf32(tensor)

    @staticmethod
    def _fp8_e4m3_to_tf32(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Handles Fp8_e4m3 format.

        Fp8_e4m3 is an L1-only encoding; hardware converts it to Float16 in source registers.
        Golden generators work on stimuli stored as torch.bfloat16, so we delegate to
        Float16_b conversion for fidelity masking.
        """
        return SrcFormatModel._fp16b_to_tf32(tensor)

    @staticmethod
    def from_src_format(
        data_format: DataFormat,
        tensor: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # int64, int64, int64 tensors
        sign, exp, mant = tensor

        # Convert mantissa with non-implied 1 to fractional value
        TF32_MANT_WIDTH = 10
        frac = mant.to(torch.float32) / (1 << TF32_MANT_WIDTH)

        reassembled = ((-1.0) ** sign) * (2.0**exp) * frac

        torch_format = format_dict.get(data_format, format_dict[DataFormat.Float16_b])
        return reassembled.to(torch_format)


class FidelityMasking:

    def _apply_fidelity_masking(
        self,
        data_format: DataFormat,
        operand_a: torch.Tensor,
        operand_b: torch.Tensor,
        fidelity_iteration: int,
    ):
        if (fidelity_iteration < 0) or (fidelity_iteration > 3):
            raise ValueError(f"Invalid fidelity iteration: {fidelity_iteration}")

        if get_chip_architecture() == ChipArchitecture.QUASAR:
            FP_FIDELITY_ITER_MASK = [
                (0b11111111000, 0b11111111000),
                (0b00000000111, 0b11111111000),
                (0b11111111000, 0b00000000111),
                (0b00000000111, 0b00000000111),
            ]
        else:
            FP_FIDELITY_ITER_MASK = [
                (0b11111000000, 0b11111110000),
                (0b00000111110, 0b11111110000),
                (0b11111000000, 0b00000001111),
                (0b00000111110, 0b00000001111),
            ]

        sign_a, exp_a, mant_a = SrcFormatModel.to_src_format(data_format, operand_a)
        sign_b, exp_b, mant_b = SrcFormatModel.to_src_format(data_format, operand_b)

        fidelity_mask_a, fidelity_mask_b = FP_FIDELITY_ITER_MASK[fidelity_iteration]

        mant_a = mant_a & fidelity_mask_a
        mant_b = mant_b & fidelity_mask_b

        repack_a = SrcFormatModel.from_src_format(data_format, (sign_a, exp_a, mant_a))
        repack_b = SrcFormatModel.from_src_format(data_format, (sign_b, exp_b, mant_b))

        return repack_a, repack_b


def to_tensor(operand, data_format):
    torch_format = format_dict.get(data_format)
    return operand.clone().detach().to(torch_format)


def transpose_tensor(tensor):
    """Transpose a PyTorch tensor.
    Args:
        tensor: Input PyTorch tensor to transpose
    Returns:
        torch.Tensor: Transposed tensor
    """
    return tensor.T


@register_golden
class TransposeGolden:
    def __init__(self):
        pass

    def _quantize_transpose_input(self, operand, data_format):
        """Quantize input before transposing to match hardware unpack behavior.

        Hardware unpacks BFP data using the original (pre-transpose) block structure,
        then transposes. So quantization must happen before the transpose.
        """
        if data_format == DataFormat.Bfp4_b:
            return _bfp4b_to_float16b(operand)
        elif data_format == DataFormat.Bfp8_b:
            return _bfp8b_to_float16b(operand)
        return operand

    def transpose_within_faces(
        self,
        operand,
        data_format: DataFormat,
        input_dimensions: list[int] = [32, 32],
        num_faces: int = 4,
    ):
        """Transpose a tile tensor by transposing within each face.
        A tile tensor consists of faces, each face is always 16x16 = 256 elements.
        For num_faces < 4, we process only the first num_faces faces from the tensor.
        Args:
            operand: Input tensor to transpose
            data_format: Data format for the result
            input_dimensions: Input tensor dimensions (for compatibility)
            num_faces: Number of faces in the tile (1, 2, or 4)
        Returns:
            torch.Tensor: Tensor with each face transposed, result size = num_faces * 256
        """
        if num_faces not in [1, 2, 4]:
            raise ValueError(f"num_faces must be 1, 2, or 4, got {num_faces}")

        tensor = to_tensor(operand, data_format)
        tensor = self._quantize_transpose_input(tensor, data_format)
        torch_format = format_dict[data_format]

        # Each face is always 16x16 = 256 elements
        face_size = ELEMENTS_PER_FACE
        face_dim = FACE_DIM
        elements_per_tile_needed = face_size * num_faces

        # Select first N faces
        tensor_to_process = tensor[:elements_per_tile_needed]

        # Split into faces and transpose each face individually
        faces = tensor_to_process.view(num_faces, face_dim, face_dim)
        transposed_faces = faces.transpose(-2, -1)
        result = transposed_faces.flatten().to(torch_format)

        return result

    def transpose_faces(
        self,
        operand,
        data_format: DataFormat,
        input_dimensions: list[int] = [32, 32],
        num_faces: int = 4,
    ):
        """Transpose the arrangement of faces in a tile tensor.
        Treats each face as a single element and transposes their arrangement.

        For 4 faces arranged as:
        f0 f1
        f2 f3
        After transposition:
        f0 f2
        f1 f3

        For 2 faces: f0, f1 -> f0, f1 (no change in linear arrangement)
        For 1 face: f0 -> f0 (identity operation)

        Args:
            operand: Input tensor to transpose
            data_format: Data format for the result
            input_dimensions: Input tensor dimensions (for compatibility)
            num_faces: Number of faces in the tile (1, 2, or 4)
        Returns:
            torch.Tensor: Tensor with faces rearranged in transposed order
        """
        if num_faces not in [1, 2, 4]:
            raise ValueError(f"num_faces must be 1, 2, or 4, got {num_faces}")

        torch_format = format_dict[data_format]
        tensor = to_tensor(operand, data_format)
        tensor = self._quantize_transpose_input(tensor, data_format)

        total_elements = ELEMENTS_PER_FACE * num_faces
        tensor = tensor[:total_elements]

        if num_faces == 4:
            # Reorder faces: f0, f1, f2, f3 -> f0, f2, f1, f3
            faces = torch.tensor_split(tensor, 4)
            tensor = torch.cat([faces[0], faces[2], faces[1], faces[3]])

        return tensor.to(torch_format)

    def _apply_tile_operation_multi_tile(
        self,
        operand: torch.Tensor,
        data_format: DataFormat,
        num_tiles: int,
        operation_func: callable,
        tilize: bool = False,
        untilize: bool = False,
        input_dimensions: tuple[int, int] = (32, 32),
    ) -> torch.Tensor:
        """
        Apply a tile-level operation across multiple tiles in a tensor.

        This is a generic helper function that applies any single-tile operation
        to each tile in a multi-tile tensor, handling common preprocessing and
        postprocessing steps.

        Args:
            operand: Input tensor containing concatenated tiles to process
            data_format: Target data format for the result tensor
            num_tiles: Number of 32×32 tiles in the input tensor (must be positive)
            operation_func: Function to apply to each tile (e.g., self.transpose_faces)
            tilize: If True, applies tilization preprocessing to the input
            untilize: If True, applies untilization postprocessing to the result
            input_dimensions: Overall input matrix dimensions as (rows, cols)

        Returns:
            Tensor with the operation applied to all tiles

        Raises:
            ValueError: If tensor size doesn't match expected size for num_tiles
            ValueError: If num_tiles is not positive
        """
        # Input validation
        if num_tiles <= 0:
            raise ValueError(f"num_tiles must be positive, got {num_tiles}")

        if not callable(operation_func):
            raise ValueError("operation_func must be callable")

        # Convert and prepare tensor
        tensor = to_tensor(operand, data_format)

        # Apply tilization if requested
        if tilize:
            tilize_fn = get_golden_generator(TilizeGolden)
            tensor = tilize_fn(tensor, input_dimensions, data_format).flatten()

        # Validate tensor dimensions
        total_elements = tensor.numel()
        expected_elements = num_tiles * ELEMENTS_PER_TILE
        if total_elements != expected_elements:
            raise ValueError(
                f"Tensor size mismatch: got {total_elements} elements for {num_tiles} tiles. "
                f"Expected {expected_elements} elements "
                f"({num_tiles} tiles × {ELEMENTS_PER_TILE} elements/tile)"
            )

        # Reshape tensor for efficient batch processing
        tile_tensors = tensor.view(num_tiles, ELEMENTS_PER_TILE)

        # Apply operation to all tiles
        processed_tiles = [
            operation_func(
                tile_tensor,
                data_format,
                input_dimensions=TILE_DIMENSIONS,
            )
            for tile_tensor in tile_tensors
        ]

        # Concatenate results
        result = torch.cat(processed_tiles)

        # Apply untilization if requested
        if untilize:
            untilize_fn = get_golden_generator(UntilizeGolden)
            result = untilize_fn(result, data_format, input_dimensions).flatten()

        return result.to(format_dict[data_format])

    def transpose_faces_multi_tile(
        self,
        operand: torch.Tensor,
        data_format: DataFormat,
        num_tiles: int,
        tilize: bool = False,
        untilize: bool = False,
        input_dimensions: tuple[int, int] = (32, 32),
    ) -> torch.Tensor:
        """
        Transpose face arrangements across multiple tiles in a tensor.

        This function applies face transposition to each 32×32 tile in a multi-tile tensor.
        Each tile contains 1024 elements arranged as 4 faces of 256 elements each.
        The operation rearranges the faces within each tile.

        Args:
            operand: Input tensor containing concatenated tiles to transpose
            data_format: Target data format for the result tensor
            num_tiles: Number of 32×32 tiles in the input tensor (must be positive)
            tilize: If True, applies tilization preprocessing to the input
            untilize: If True, applies untilization postprocessing to the result
            input_dimensions: Overall input matrix dimensions as (rows, cols)

        Returns:
            Tensor with face arrangements transposed for all tiles

        Raises:
            ValueError: If tensor size doesn't match expected size for num_tiles
            ValueError: If num_tiles is not positive

        Example:
            >>> # Process 4 tiles with face transposition
            >>> result = obj.transpose_faces_multi_tile(
            ...     tensor, "bfloat16", num_tiles=4, tilize=True
            ... )
        """
        return self._apply_tile_operation_multi_tile(
            operand=operand,
            data_format=data_format,
            num_tiles=num_tiles,
            operation_func=self.transpose_faces,
            tilize=tilize,
            untilize=untilize,
            input_dimensions=input_dimensions,
        )

    def transpose_within_faces_multi_tile(
        self,
        operand: torch.Tensor,
        data_format: DataFormat,
        num_tiles: int,
        tilize: bool = False,
        untilize: bool = False,
        input_dimensions: tuple[int, int] = (32, 32),
    ) -> torch.Tensor:
        """
        Transpose elements within each face across multiple tiles.

        This function applies within-face transposition to each 32×32 tile in a multi-tile tensor.
        Each tile contains 4 faces of 256 elements each, and the transposition is applied
        independently within each face of every tile, preserving face boundaries.

        Args:
            operand: Input tensor containing concatenated tiles to process
            data_format: Target data format for the result tensor
            num_tiles: Number of 32×32 tiles in the input tensor (must be positive)
            tilize: If True, applies tilization preprocessing to the input
            untilize: If True, applies untilization postprocessing to the result
            input_dimensions: Overall input matrix dimensions as (rows, cols)

        Returns:
            Tensor with elements transposed within each face of all tiles

        Raises:
            ValueError: If tensor size doesn't match expected size for num_tiles
            ValueError: If num_tiles is not positive

        Example:
            >>> # Process 2 tiles with within-face transposition
            >>> result = obj.transpose_within_faces_multi_tile(
            ...     tensor, "float32", num_tiles=2, untilize=True
            ... )

        Note:
            The transposition occurs within each of the 4 faces per tile, preserving
            the face boundaries but reordering elements within each face.
        """
        return self._apply_tile_operation_multi_tile(
            operand=operand,
            data_format=data_format,
            num_tiles=num_tiles,
            operation_func=self.transpose_within_faces,
            tilize=tilize,
            untilize=untilize,
            input_dimensions=input_dimensions,
        )


@register_golden
class MatmulGolden(FidelityMasking):

    MATH_FIDELITY_TO_ITER_COUNT = {
        MathFidelity.LoFi: 0,
        MathFidelity.HiFi2: 1,
        MathFidelity.HiFi3: 2,
        MathFidelity.HiFi4: 3,
    }

    @staticmethod
    def _convert_block_float_inputs(
        operand1,
        operand2,
        input_A_format: DataFormat,
        input_B_format: DataFormat,
        input_A_dimensions,
        input_B_dimensions,
        tilize: bool,
    ):
        if input_A_format == DataFormat.Bfp8_b:
            dims = input_A_dimensions if tilize else None
            operand1 = _bfp8b_to_float16b(operand1, dims)
        if input_B_format == DataFormat.Bfp8_b:
            dims = input_B_dimensions if tilize else None
            operand2 = _bfp8b_to_float16b(operand2, dims)
        if input_A_format == DataFormat.Bfp4_b:
            dims = input_A_dimensions if tilize else None
            operand1 = _bfp4b_to_float16b(operand1, dims)
        if input_B_format == DataFormat.Bfp4_b:
            dims = input_B_dimensions if tilize else None
            operand2 = _bfp4b_to_float16b(operand2, dims)
        if input_A_format == DataFormat.Bfp2_b:
            dims = input_A_dimensions if tilize else None
            operand1 = _bfp2b_to_float16b(operand1, dims)
        if input_B_format == DataFormat.Bfp2_b:
            dims = input_B_dimensions if tilize else None
            operand2 = _bfp2b_to_float16b(operand2, dims)

        return operand1, operand2

    @staticmethod
    def _resolve_matmul_dimensions(input_A_dimensions, input_B_dimensions):
        M, K1 = input_A_dimensions[0], input_A_dimensions[1]
        K2, N = input_B_dimensions[0], input_B_dimensions[1]

        if K1 != K2:
            raise AssertionError(
                f"Matrix dimensions incompatible: A[{M},{K1}] × B[{K2},{N}]"
            )

        return M, K1, K2, N, [M, N]

    def _get_fidelity_iters(self, math_fidelity):
        fidelity_iter_count = self.MATH_FIDELITY_TO_ITER_COUNT[math_fidelity]
        if fidelity_iter_count == 3:
            return [None]
        return list(range(fidelity_iter_count + 1))

    def _prepare_fidelity_operands(
        self,
        operand1,
        operand2,
        fidelity_format: DataFormat,
        fidelity_iter: Optional[int],
    ):
        t1 = to_tensor(operand1, fidelity_format)
        t2 = to_tensor(operand2, fidelity_format)
        if fidelity_iter is not None:
            # The Tensix matmul swaps its operands through the source registers:
            # the lhs is unpacked into SrcB and the rhs into SrcA. The fidelity
            # masks are per-source (mask_a -> SrcA, mask_b -> SrcB) and asymmetric
            # (e.g. LoFi keeps the top 4 of SrcA's mantissa but the top 6 of
            # SrcB's), so the lhs must take the SrcB mask and the rhs the SrcA mask.
            # Feed (rhs, lhs) into the masking and unswap the result so each operand
            # is masked as the source register it actually lands in.
            t2, t1 = self._apply_fidelity_masking(
                fidelity_format, t2, t1, fidelity_iter
            )
        return t1, t2

    def __call__(
        self,
        operand1,
        operand2,
        data_format,
        math_fidelity,
        input_A_dimensions=None,
        input_B_dimensions=None,
        tilize: bool = False,
        input_A_format: DataFormat = None,
        input_B_format: DataFormat = None,
        math_format: Optional[DataFormat] = None,
        dest_acc: Optional[DestAccumulation] = None,
    ):
        # Route MX outputs through the KT-chunked path that honors math_format
        # and dest_acc. The default path does a single fp32-internal torch.matmul
        # which can disagree with HW's apparent per-KT-tile rounding once results land
        # near MxInt2/MxInt4 quantization bin boundaries.
        if data_format.is_mx_format():
            return self._matmul_mx(
                operand1,
                operand2,
                data_format,
                math_fidelity,
                input_A_dimensions,
                input_B_dimensions,
                tilize,
                input_A_format,
                input_B_format,
                math_format,
                dest_acc,
            )

        if data_format.is_integer():
            return self._matmul_integer(
                operand1,
                operand2,
                data_format,
                input_A_dimensions,
                input_B_dimensions,
                tilize,
                input_A_format,
                input_B_format,
            )

        return self._matmul_default(
            operand1,
            operand2,
            data_format,
            math_fidelity,
            input_A_dimensions,
            input_B_dimensions,
            tilize,
            input_A_format,
            input_B_format,
        )

    # Integer matmul is LoFi-only on Quasar.
    def _matmul_integer(
        self,
        operand1,
        operand2,
        data_format,
        input_A_dimensions,
        input_B_dimensions,
        tilize: bool,
        input_A_format: DataFormat = None,
        input_B_format: DataFormat = None,
    ):
        torch_format = format_dict[data_format]

        M, K1, K2, N, _ = self._resolve_matmul_dimensions(
            input_A_dimensions, input_B_dimensions
        )

        t1 = to_tensor(operand1, input_A_format).view(M, K1)
        t2 = to_tensor(operand2, input_B_format).view(K2, N)

        res = saturate_integer(
            torch.matmul(t1.to(torch.int64), t2.to(torch.int64)).view(M * N),
            data_format,
            torch_format,
        )

        if tilize:
            res = tilize_block(
                res,
                dimensions=(input_A_dimensions[0], input_B_dimensions[1]),
                stimuli_format=data_format,
            ).flatten()
        return res

    def _matmul_default(
        self,
        operand1,
        operand2,
        data_format,
        math_fidelity,
        input_A_dimensions,
        input_B_dimensions,
        tilize: bool,
        input_A_format: DataFormat,
        input_B_format: DataFormat,
    ):
        torch_format = format_dict[data_format]

        operand1, operand2 = self._convert_block_float_inputs(
            operand1,
            operand2,
            input_A_format,
            input_B_format,
            input_A_dimensions,
            input_B_dimensions,
            tilize,
        )

        # Handle multi-tile matmul with different operand dimensions
        if input_A_dimensions is not None and input_B_dimensions is not None:
            # Multi-tile matmul: A[M,K] × B[K,N] = C[M,N]
            M, K1, K2, N, output_dimensions = self._resolve_matmul_dimensions(
                input_A_dimensions, input_B_dimensions
            )

        fidelity_iters = self._get_fidelity_iters(math_fidelity)
        res: Optional[torch.Tensor] = None

        for fidelity_iter in fidelity_iters:
            t1, t2 = self._prepare_fidelity_operands(
                operand1, operand2, data_format, fidelity_iter
            )
            t1, t2 = t1.view(M, K1), t2.view(K2, N)
            partial = (
                torch.matmul(t1, t2)
                .view(output_dimensions[0] * output_dimensions[1])
                .to(torch_format)
            )
            if res is None:
                res = partial
            else:
                res += partial

        if tilize:
            res = tilize_block(
                res,
                dimensions=(input_A_dimensions[0], input_B_dimensions[1]),
                stimuli_format=data_format,
            ).flatten()
        return res

    def _matmul_mx(
        self,
        operand1,
        operand2,
        data_format,
        math_fidelity,
        input_A_dimensions,
        input_B_dimensions,
        tilize: bool,
        input_A_format: DataFormat,
        input_B_format: DataFormat,
        math_format: Optional[DataFormat],
        dest_acc: Optional[DestAccumulation],
    ):
        math_format = math_format or data_format
        result_format = math_format if data_format.is_mx_format() else data_format
        result_torch_format = format_dict[result_format]

        operand1, operand2 = self._convert_block_float_inputs(
            operand1,
            operand2,
            input_A_format,
            input_B_format,
            input_A_dimensions,
            input_B_dimensions,
            tilize,
        )

        # Handle multi-tile matmul with different operand dimensions
        if input_A_dimensions is not None and input_B_dimensions is not None:
            # Multi-tile matmul: A[M,K] × B[K,N] = C[M,N]
            M, K1, K2, N, output_dimensions = self._resolve_matmul_dimensions(
                input_A_dimensions, input_B_dimensions
            )

        fidelity_iters = self._get_fidelity_iters(math_fidelity)

        math_torch_format = format_dict[math_format]
        use_fp16_acc = dest_acc == DestAccumulation.No and math_format in (
            DataFormat.Float16,
            DataFormat.Float16_b,
        )

        def _matmul_blocked(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            if not use_fp16_acc:
                return torch.matmul(a, b).to(result_torch_format)

            # Quasar matmul probably accumulates per KT tile (32 elements), so match KT_DIM chunking
            # to avoid extra rounding compared to HW.
            k_tile = TILE_DIM if K1 >= TILE_DIM else K1

            acc = torch.zeros((M, N), dtype=math_torch_format)
            for k0 in range(0, K1, k_tile):
                a_block = a[:, k0 : k0 + k_tile].to(math_torch_format)
                b_block = b[k0 : k0 + k_tile, :].to(math_torch_format)
                acc += torch.matmul(a_block, b_block)
            return acc.to(result_torch_format)

        def _accumulate(
            acc: Optional[torch.Tensor], value: torch.Tensor
        ) -> torch.Tensor:
            if acc is None:
                return value
            if use_fp16_acc:
                return (acc + value).to(result_torch_format)
            return acc + value

        res: Optional[torch.Tensor] = None

        for fidelity_iter in fidelity_iters:
            t1, t2 = self._prepare_fidelity_operands(
                operand1, operand2, math_format, fidelity_iter
            )
            t1, t2 = t1.view(M, K1), t2.view(K2, N)
            res = _accumulate(
                res,
                _matmul_blocked(t1, t2).view(
                    output_dimensions[0] * output_dimensions[1]
                ),
            )

        if tilize:
            tilize_format = result_format if data_format.is_mx_format() else data_format
            res = tilize_block(
                res,
                dimensions=(input_A_dimensions[0], input_B_dimensions[1]),
                stimuli_format=tilize_format,
            ).flatten()
        return res


@register_golden
class BroadcastGolden:
    """
    Golden generator for broadcast operations (Scalar, Column, Row).

    Broadcasts operand values according to the specified broadcast type:
    - Scalar: Takes first element of each tile and broadcasts it across entire output tile
    - Column: Broadcasts column values across rows (Faces 0-1 use Face 0's column, Faces 2-3 use Face 2's column)
    - Row: Broadcasts row values down columns (first row of Face 0/1)

    Output size = tile_cnt * num_faces * (face_r_dim * 16) elements.
    """

    def __init__(self):
        self.broadcast_handlers = {
            BroadcastType.Scalar: self._broadcast_scalar,
            BroadcastType.Column: self._broadcast_column,
            BroadcastType.Row: self._broadcast_row,
        }

    def __call__(
        self,
        broadcast_type,
        operand,
        data_format,
        num_faces: int = 4,
        tile_cnt: int = 1,
        face_r_dim: int = 16,
        input_format: DataFormat = None,
    ):
        if broadcast_type not in self.broadcast_handlers:
            raise ValueError(f"Unsupported broadcast type: {broadcast_type}")

        torch_format = format_dict[data_format]

        # Convert input to tensor
        if isinstance(operand, torch.Tensor):
            input_flat = operand.flatten().to(torch_format)
        else:
            input_flat = torch.tensor(operand, dtype=torch_format).flatten()

        # Quantize input tile-by-tile BEFORE extracting broadcast values.
        # The hardware unpacks src_B from its L1 encoding before applying the broadcast,
        # so quantization must be based on the original (non-broadcast) tile rows.
        format_for_quant = input_format or data_format
        if format_for_quant == DataFormat.Bfp2_b:
            input_flat = _bfp2b_to_float16b(input_flat)
        elif format_for_quant == DataFormat.Bfp4_b:
            input_flat = _bfp4b_to_float16b(input_flat)
        elif format_for_quant == DataFormat.Bfp8_b:
            input_flat = _bfp8b_to_float16b(input_flat)
        elif format_for_quant.is_mx_format():
            input_flat = quantize_mx_tensor_chunked(input_flat, format_for_quant)

        # Calculate output size based on variable face dimensions
        elements_per_tile = face_r_dim * FACE_DIM * num_faces

        results = []
        for tile_idx in range(tile_cnt):
            tile_start = tile_idx * elements_per_tile
            tile_end = tile_start + elements_per_tile
            tile_data = input_flat[tile_start:tile_end]

            tile_result = self.broadcast_handlers[broadcast_type](
                tile_data, num_faces=num_faces, face_r_dim=face_r_dim
            )
            results.append(tile_result)

        return torch.cat(results)

    def _broadcast_scalar(self, tile_data, **kwargs):
        """Broadcast first element of each tile across the entire output tile."""
        scalar_value = tile_data[0]

        return torch.full_like(tile_data, scalar_value)

    def _broadcast_column(
        self,
        tile_data,
        num_faces: int,
        face_r_dim: int,
    ):
        """
        Process a single tile for column broadcast.

        For a face_r_dim x 16 face: input has face_r_dim unique values (one per row),
        each value is replicated 16 times across its row.
        Output pattern: [row0_val]*16, [row1_val]*16, ..., [row(face_r_dim-1)_val]*16
        """
        face_size = face_r_dim * FACE_DIM

        # Process face 0 (used by faces 0-1)
        source_face_0 = tile_data[:face_size]
        col_values_0 = source_face_0[::FACE_DIM]
        face_0_broadcast = col_values_0.repeat_interleave(FACE_DIM)

        # Handle different face counts: 1, 2, 4
        if num_faces == 1:
            return face_0_broadcast
        elif num_faces == 2:
            # Both faces use face 0 - use repeat instead of cat
            return face_0_broadcast.repeat(2)
        else:  # num_faces == 4
            # Process face 2 (used by faces 2-3)
            source_face_2 = tile_data[2 * face_size : 3 * face_size]
            col_values_2 = source_face_2[::FACE_DIM]
            face_2_broadcast = col_values_2.repeat_interleave(FACE_DIM)

            return torch.cat(
                [face_0_broadcast, face_0_broadcast, face_2_broadcast, face_2_broadcast]
            )

    def _broadcast_row(
        self,
        tile_data,
        num_faces: int,
        face_r_dim: int,
    ):
        """Process a single tile for row broadcast."""
        face_size = face_r_dim * FACE_DIM

        # Process face 0: take first row and repeat to fill face
        face_0_row = tile_data[:FACE_DIM]
        face_0_broadcast = face_0_row.repeat(face_r_dim)

        if num_faces == 1:
            return face_0_broadcast
        elif num_faces in (2, 4):
            # Extract and repeat face 1 row
            face_1_row = tile_data[face_size : face_size + FACE_DIM]
            face_1_broadcast = face_1_row.repeat(face_r_dim)

            if num_faces == 2:
                return torch.cat([face_0_broadcast, face_1_broadcast])
            else:  # num_faces == 4
                return torch.cat(
                    [
                        face_0_broadcast,
                        face_1_broadcast,
                        face_0_broadcast,
                        face_1_broadcast,
                    ]
                )


@register_golden
class DataCopyGolden:
    def __call__(
        self,
        operand1,
        data_format,
        num_faces: int = 4,
        input_dimensions: list[int] = [32, 32],
        face_r_dim: int = 16,  # Default to 16 for backward compatibility
        input_format=None,
        tile_shape=None,
    ):
        torch_format = format_dict[data_format]

        # Quantize input to match what hardware actually sees after unpack from L1.
        operand1 = quantize_input_to_unpack_format(
            operand1, input_format, all_mx_formats=True
        )

        height, width = input_dimensions[0], input_dimensions[1]

        # Tile count selection:
        # - tile_shape given: derive directly from the real tile geometry. This
        #   is required for full-width tiny tiles (e.g. 16x32, num_faces=2) where
        #   face_r_dim is still 16 but a tensor packs into more, smaller tiles than
        #   the 32x32 assumption below would compute.
        # - face_r_dim < 16: legacy partial-face path treats the input as one tile.
        # - otherwise: assume standard 32x32 tiles (backward compatible).
        if tile_shape is not None:
            tile_rows = tile_shape.total_row_dim()
            tile_cols = tile_shape.total_col_dim()
            tile_cnt = (height // tile_rows) * (width // tile_cols)
        elif face_r_dim < 16:
            tile_cnt = 1
        else:
            tile_cnt = (height // 32) * (width // 32)

        # Calculate elements based on variable face dimensions
        # Each face is face_r_dim × 16, and we have num_faces
        elements_per_tile_needed = face_r_dim * FACE_DIM * num_faces

        # Convert input to tensor if needed
        if not isinstance(operand1, torch.Tensor):
            operand1 = torch.tensor(operand1, dtype=torch_format)

        # Determine actual tile size from input:
        # If input is sized for partial faces (num_faces < 4), use elements_per_tile_needed
        # Otherwise use full tile size
        total_elements = operand1.numel()
        expected_partial_size = tile_cnt * elements_per_tile_needed

        if total_elements == expected_partial_size:
            # Input is already sized for num_faces, just pass through
            tile_size = elements_per_tile_needed
        else:
            # Input has full tiles, need to select elements
            tile_size = height * width // tile_cnt if tile_cnt > 0 else height * width

        reshaped = operand1.view(tile_cnt, tile_size)
        selected = reshaped[:, :elements_per_tile_needed]
        result = selected.flatten()

        # Ensure result is in correct format if not already
        if result.dtype != torch_format:
            if data_format.is_integer():
                result = saturate_integer(result, data_format, torch_format)
            else:
                result = result.to(torch_format)

        # Apply BFP output quantization round-trip to match hardware behaviour
        if data_format in (DataFormat.Bfp4_b, DataFormat.Bfp8_b, DataFormat.Bfp2_b):
            result_t = (
                result.float()
                if isinstance(result, torch.Tensor)
                else torch.tensor(result, dtype=torch.float32)
            )
            flat = result_t.ravel()
            if num_faces == 4:
                data = tilize_block(
                    flat, input_dimensions, DataFormat.Float16_b
                ).ravel()
                dims = input_dimensions
            else:
                data = flat
                dims = None
            if data_format == DataFormat.Bfp4_b:
                result = _bfp4b_to_float16b(data, dims)
            elif data_format == DataFormat.Bfp2_b:
                result = _bfp2b_to_float16b(data, dims)
            else:
                result = _bfp8b_to_float16b(data, dims)

        # Final FTZ pass: hardware always flushes subnormals to zero. The BFP
        # helpers no longer FTZ internally, so funnel every output (BFP, MX,
        # plain FP) through the centralised FTZ to match silicon behaviour.
        return _apply_ftz(result, data_format)


@register_golden
class TypecastGolden:
    """Golden generator for the SFPU typecast operation.

    Models the production flow (copy_tile -> typecast_tile -> pack): the tile loads into Dest,
    the SFPU converts each datum in place, the packer writes it to L1. Purely elementwise and
    read back row-major over the same unpack->Dest->pack path as DataCopyGolden, so the
    conversion applies no tilization. Covers the full ttnn matrix over float, integer and
    block-float (Bfp8_b / Bfp4_b) source/destination dtypes:
      * block-float input round-trips through ``quantize_input_to_unpack_format``, matching
        what the SFPU sees;
      * float/int -> integer: truncate toward zero for int32/uint32, round-to-nearest for
        uint16/uint8 (whole-number stimuli make both exact); UInt8 keeps the low byte, others
        clamp to the dest range;
      * -> plain float: value-preserving cast;
      * -> block-float: tilized into 16-element BFP blocks, through the packer's
        shared-exponent quantization, untilized back to row-major (as DataCopyGolden).
    """

    _BLOCK_FLOAT_FORMATS = (
        DataFormat.Bfp8_b,
        DataFormat.Bfp4_b,
        DataFormat.Bfp2_b,
    )

    def __call__(
        self,
        operand,
        input_format: DataFormat,
        output_format: DataFormat,
        input_dimensions: list[int] = [32, 32],
    ):
        operand = quantize_input_to_unpack_format(
            operand, input_format, all_mx_formats=True
        )
        if not isinstance(operand, torch.Tensor):
            operand = torch.tensor(operand)

        operand = operand.flatten()

        if output_format.is_integer():
            if input_format.is_integer():
                values = operand.to(torch.int64)
            else:
                # int32/uint32 truncate; uint16/uint8 round. Whole-number
                # stimuli make trunc == round, so trunc models both exactly.
                values = torch.trunc(operand.float()).to(torch.int64)
            result = self._to_integer(values, output_format)
        elif output_format in self._BLOCK_FLOAT_FORMATS:
            result = self._to_block_float(
                operand.float(), output_format, input_dimensions
            )
        else:
            result = self._to_float(operand.float(), output_format)

        return _apply_ftz(result, output_format).flatten()

    @staticmethod
    def _to_integer(values: torch.Tensor, output_format: DataFormat) -> torch.Tensor:
        out_torch = format_dict[output_format]
        if output_format == DataFormat.UInt8:
            # Hardware keeps the low 8 bits (two's complement wrap).
            return (values % 256).to(out_torch)
        if output_format == DataFormat.UInt16:
            return torch.clamp(values, 0, 65535).to(out_torch)
        if output_format == DataFormat.UInt32:
            return torch.clamp(values, 0, 2**32 - 1).to(out_torch)
        if output_format == DataFormat.Int32:
            # +1 on the min: hardware uses sign-magnitude representation.
            return torch.clamp(values, -(2**31 - 1), 2**31 - 1).to(out_torch)
        return saturate_integer(values, output_format, out_torch)

    @staticmethod
    def _to_float(values: torch.Tensor, output_format: DataFormat) -> torch.Tensor:
        return values.to(format_dict[output_format])

    @staticmethod
    def _to_block_float(
        values: torch.Tensor,
        output_format: DataFormat,
        input_dimensions: list[int],
    ) -> torch.Tensor:
        """Quantize fp values to a block-float output, matching the packer.

        The packer computes one shared exponent per 16 contiguous datums in
        Dest (i.e. per face row), so the values are first tilized into that
        layout, quantized to the target BFP width, then untilized back to the
        row-major order the device result is read back in.
        """
        data = tilize_block(
            values.ravel(), input_dimensions, DataFormat.Float16_b
        ).ravel()
        if output_format == DataFormat.Bfp4_b:
            return _bfp4b_to_float16b(data, input_dimensions)
        if output_format == DataFormat.Bfp2_b:
            return _bfp2b_to_float16b(data, input_dimensions)
        return _bfp8b_to_float16b(data, input_dimensions)


@register_golden
class PackGolden:
    """
    Golden generator for pack operations with optional ReLU activation.
    This is similar to DataCopyGolden but includes support for ReLU configuration.
    It's implemented as a separate class to allow future pack testing extensions
    without affecting DataCopyGolden.
    """

    def __call__(
        self,
        operand1,
        data_format,
        num_faces: int = 4,
        input_dimensions: list[int] = [32, 32],
        face_r_dim: int = 16,
    ):
        if num_faces not in [1, 2, 4]:
            raise ValueError(f"num_faces must be 1, 2, or 4, got {num_faces}")

        torch_format = format_dict[data_format]

        height, width = input_dimensions[0], input_dimensions[1]

        tile_cnt = (height // 32) * (width // 32)
        tile_size = height * width // tile_cnt

        # Calculate elements based on variable face dimensions
        # Each face is face_r_dim × 16, and we have num_faces
        elements_per_tile_needed = face_r_dim * FACE_DIM * num_faces

        if not isinstance(operand1, torch.Tensor):
            operand1 = torch.tensor(operand1, dtype=torch_format)

        result = operand1.view(tile_cnt, tile_size)[
            :, :elements_per_tile_needed
        ].reshape(-1)

        if result.dtype != torch_format:
            if data_format.is_integer():
                result = saturate_integer(result, data_format, torch_format)
            else:
                result = result.to(torch_format)

        return result

    @staticmethod
    def get_relu_mode_and_threshold_bits(
        relu_type: PackerReluType,
        relu_threshold: float,
        intermediate_format: DataFormat,
    ) -> tuple[PackerReluType, int]:
        """
        Return (mode, threshold_bits) for use with RELU_CONFIG and golden.
        threshold_bits is 0 for NO_RELU and ZERO_RELU.
        """
        if relu_type in [
            PackerReluType.MinThresholdRelu,
            PackerReluType.MaxThresholdRelu,
        ]:
            threshold_bits = PackGolden._encode_threshold_to_bits(
                relu_threshold, intermediate_format
            )
            return (relu_type, threshold_bits)
        return (relu_type, 0)

    @staticmethod
    def generate_relu_config(
        relu_type: PackerReluType,
        relu_threshold: float,
        intermediate_format: DataFormat,
    ) -> int:
        """
        Generate a 32-bit ReLU configuration value.
        Args:
            relu_type: The ReLU type (NO_RELU, ZERO_RELU, MIN_THRESHOLD_RELU, MAX_THRESHOLD_RELU)
            relu_threshold: The threshold value (default 0.0, ignored for NO_RELU and ZERO_RELU)
            intermediate_format: The intermediate data format (determines FP16 vs BF16 encoding)
        Returns:
            int: 32-bit ReLU configuration value with type in lower 2 bits and threshold in upper 16 bits
        """
        mode, threshold_bits = PackGolden.get_relu_mode_and_threshold_bits(
            relu_type, relu_threshold, intermediate_format
        )
        return pack_relu_config(mode, threshold_bits)

    @staticmethod
    def _encode_threshold_to_bits(
        threshold: float, intermediate_format: DataFormat
    ) -> int:
        # FP16, FP8, BFP8a (Bfp8), BFP4a, BFP2a use FP16 interpretation
        # TODO: Add more formats once available
        fp16_formats = [DataFormat.Float16, DataFormat.Bfp8]

        if intermediate_format in fp16_formats:
            return (
                torch.tensor(threshold, dtype=torch.float16).view(torch.uint16).item()
            )
        else:
            # For Float32, Float16_b, and other BF16-compatible formats:
            # Encode as BF16 (upper 16 bits of FP32)
            # HW requires BF16/FP16 threshold in 16-bit field for both FP16 and FP32 pack_src
            # For FP32 dest (EN_32BIT_DEST), C++ shifts this left by 16 to reconstruct full FP32 value
            fp32_bits = struct.unpack(">I", struct.pack(">f", threshold))[0]
            return (fp32_bits >> 16) & 0xFFFF

    @staticmethod
    def _decode_threshold_from_bits(
        threshold_bits: int, intermediate_format: DataFormat
    ) -> float:
        # FP16, FP8, BFP8a (Bfp8), BFP4a, BFP2a use FP16 interpretation
        # TODO: Add more formats once supported
        fp16_formats = [DataFormat.Float16, DataFormat.Bfp8]

        if intermediate_format in fp16_formats:
            return (
                torch.tensor(threshold_bits, dtype=torch.uint16)
                .view(torch.float16)
                .item()
            )
        else:
            fp32_bits = threshold_bits << 16
            return struct.unpack(">f", struct.pack(">I", fp32_bits))[0]

    @staticmethod
    def _extract_threshold_from_config(
        relu_config: int, intermediate_format: DataFormat
    ) -> float:
        threshold_bits = (relu_config >> 16) & 0xFFFF
        return PackGolden._decode_threshold_from_bits(
            threshold_bits, intermediate_format
        )

    @staticmethod
    def get_relu_type(relu_config):
        """
        Get the ReLU type from the configuration.
        """
        relu_type = PackerReluType.from_bits(relu_config & 0x3)
        return relu_type

    @staticmethod
    def get_relu_threshold(relu_config, intermediate_format):
        """
        Get the ReLU threshold value based on configuration.
        The relu_config is a 32-bit value where:
        - Lowest 2 bits: ReLU type
        - Upper 16 bits: ReLU threshold value (as FP16 or BF16)
        - Remaining bits: unknown/reserved
        Args:
            relu_config: 32-bit ReLU configuration value
            intermediate_format: The intermediate data format that acts as an input format for Packer engine.
        Returns:
            float: The threshold value, or None if ReLU is disabled
        """
        relu_type = PackerReluType.from_bits(relu_config & 0x3)

        match relu_type:
            case PackerReluType.NoRelu:
                return None

            case PackerReluType.ZeroRelu:
                return 0.0

            case PackerReluType.MinThresholdRelu | PackerReluType.MaxThresholdRelu:
                threshold_bits = (relu_config >> 16) & 0xFFFF

                # Parse threshold based on intermediate format.
                # FP16, FP8, BFP8a (Bfp8), BFP4a, BFP2a use FP16 interpretation.
                # TODO: add other formats once supported.
                parse_fp16_formats = [DataFormat.Float16, DataFormat.Bfp8]

                if intermediate_format in parse_fp16_formats:
                    threshold_tensor = torch.tensor(
                        [threshold_bits], dtype=torch.uint16
                    ).view(torch.float16)
                    threshold = float(threshold_tensor.item())
                else:
                    # BF16 interpretation (FP32 and other formats).
                    # BF16 is essentially just the upper 16 bits of FP32, so shift left by 16.
                    threshold_as_fp32_bits = threshold_bits << 16
                    threshold_tensor = torch.tensor(
                        [threshold_as_fp32_bits], dtype=torch.uint32
                    ).view(torch.float32)
                    threshold = float(threshold_tensor.item())

                return threshold

    @staticmethod
    def apply_relu(result, relu_config, intermediate_format):
        """
        Apply ReLU operation based on configuration.
        Args:
            result: Input tensor
        relu_config: 32-bit ReLU configuration (lowest 2 bits = type, bits 16–31 = threshold, bits 2–15 reserved)
        intermediate_format: The intermediate data format (DataFormat enum)
        Returns:
            Tensor with ReLU applied
        """

        relu_type = PackGolden.get_relu_type(relu_config)

        match relu_type:
            case PackerReluType.NoRelu:
                return result

            case PackerReluType.ZeroRelu:
                return torch.relu(result)

            case PackerReluType.MinThresholdRelu:
                threshold = PackGolden._extract_threshold_from_config(
                    relu_config, intermediate_format
                )
                # Return 0 if x <= threshold, else x
                return torch.where(
                    result <= threshold, torch.tensor(0.0, dtype=result.dtype), result
                )

            case PackerReluType.MaxThresholdRelu:
                threshold = PackGolden._extract_threshold_from_config(
                    relu_config, intermediate_format
                )
                # Clamp between 0 and threshold
                return torch.clamp(result, min=0.0, max=threshold)

    @staticmethod
    def accumulate_l1(
        partials: list[torch.Tensor],
        data_format: DataFormat,
    ) -> torch.Tensor:
        return apply_l1_accumulation(partials, data_format)

    @staticmethod
    def is_relu_threshold_tolerance_issue(
        golden_tensor,
        result_tensor,
        relu_config,
        intermediate_format,
        rtol=0.01,
        atol=0.01,
    ) -> bool:
        """Are all golden/result mismatches explained by ReLU near-threshold rounding?

        Near the threshold, golden (Python) and hardware (Tensix) can clamp differently -- one
        to zero, the other to a small non-zero -- via FP16/BF16 precision, format-conversion
        rounding, or threshold encode/decode loss.

        Args:
            golden_tensor: Expected output tensor
            result_tensor: Actual hardware output tensor
            relu_config: The ReLU configuration value
            rtol: Relative tolerance for threshold proximity checks (default 0.01)
            atol: Absolute tolerance for threshold proximity checks (default 0.01)
        Returns:
            bool: True if every mismatch is a near-threshold rounding issue, else False
        """
        relu_type = PackGolden.get_relu_type(relu_config)
        threshold = PackGolden.get_relu_threshold(relu_config, intermediate_format)

        # Only applicable for threshold-based ReLU modes
        # Zero relu is exact because of the sign bit, so no tolerance issues there.
        if relu_type not in [
            PackerReluType.MinThresholdRelu,
            PackerReluType.MaxThresholdRelu,
        ]:
            return False

        is_close = torch.isclose(golden_tensor, result_tensor, rtol=rtol, atol=atol)
        mismatches = ~is_close

        if is_close.all():
            return False

        # Check if values are within tolerance of the threshold
        golden_near_threshold = torch.isclose(
            golden_tensor[mismatches],
            torch.full_like(golden_tensor[mismatches], threshold),
            rtol=rtol,
            atol=atol,
        )
        result_near_threshold = torch.isclose(
            result_tensor[mismatches],
            torch.full_like(result_tensor[mismatches], threshold),
            rtol=rtol,
            atol=atol,
        )

        acceptable = False
        if relu_type == PackerReluType.MinThresholdRelu:
            # One side should be 0, other should be near threshold
            golden_is_zero = golden_tensor[mismatches] == 0.0
            result_is_zero = result_tensor[mismatches] == 0.0
            acceptable = (golden_is_zero & result_near_threshold) | (
                result_is_zero & golden_near_threshold
            )
        else:  # For MAX_THRESHOLD_RELU: Check if both values are near the threshold
            acceptable = golden_near_threshold & result_near_threshold

        return acceptable.all().item()


@register_golden
class UnarySFPUGolden:
    # Ops whose NaN result carries a *real* sign, because the kernel moves the sign bit
    # rather than generating a NaN: Neg flips it, Abs clears it, Identity passes it through.
    #
    # For every other op a NaN result is an invalid-operation default, whose sign IEEE 754
    # leaves unspecified. The SFPU emits a positive one; torch inherits the host libm and
    # picks either, inconsistently -- cos(inf) gives 0xFFC00000 where sqrt(-1) gives
    # 0x7FC00000. That is invisible until the NaN crosses a 16-bit Dest, where the pack path
    # substitutes a *signed* infinity (convert_nan_to_inf) and turns the arbitrary sign bit
    # into a +inf/-inf disagreement. Canonicalise, so the golden asserts the sign only where
    # it means something.
    _NAN_SIGN_TRANSPARENT_OPS = frozenset(
        {
            MathOperation.Neg,
            MathOperation.Abs,
            MathOperation.Identity,
        }
    )

    def __init__(self):
        self.ops = {
            MathOperation.Abs: self._abs,
            MathOperation.EqualZero: self._equal_zero,
            MathOperation.NotEqualZero: self._not_equal_zero,
            MathOperation.LessThanZero: self._less_than_zero,
            MathOperation.GreaterThanZero: self._greater_than_zero,
            MathOperation.LessThanEqualZero: self._less_than_equal_zero,
            MathOperation.GreaterThanEqualZero: self._greater_than_equal_zero,
            MathOperation.Atanh: self._atanh,
            MathOperation.Asinh: self._asinh,
            MathOperation.Acosh: self._acosh,
            MathOperation.Cos: self._cos,
            MathOperation.Log: self._log,
            MathOperation.Log1p: self._log1p,
            MathOperation.Reciprocal: self._reciprocal,
            MathOperation.Relu: self._relu,
            MathOperation.Rsqrt: self._rsqrt,
            MathOperation.Sin: self._sin,
            MathOperation.Signbit: self._signbit,
            MathOperation.Sqrt: self._sqrt,
            MathOperation.Square: self._square,
            MathOperation.Tanh: self._tanh,
            MathOperation.Celu: self._celu,
            MathOperation.Silu: self._silu,
            MathOperation.Erfinv: self._erfinv,
            MathOperation.Heaviside: self._heaviside,
            MathOperation.Softshrink: self._softshrink,
            MathOperation.Softsign: self._softsign,
            MathOperation.Mish: self._mish,
            MathOperation.Selu: self._selu,
            MathOperation.I0: self._i0,
            MathOperation.Rdiv: self._rdiv,
            MathOperation.Clamp: self._clamp,
            MathOperation.Hardtanh: self._hardtanh,
            MathOperation.Tanhshrink: self._tanhshrink,
            MathOperation.Floor: self._floor,
            MathOperation.Ceil: self._ceil,
            MathOperation.Trunc: self._trunc,
            MathOperation.Frac: self._frac,
            MathOperation.Round: self._round,
            MathOperation.Tan: self._tan,
            MathOperation.Atan: self._atan,
            MathOperation.Asin: self._asin,
            MathOperation.Acos: self._acos,
            MathOperation.Sinh: self._sinh,
            MathOperation.Cosh: self._cosh,
            MathOperation.Gelu: self._gelu,
            MathOperation.GeluAppx: self._gelu,
            MathOperation.GeluTanh: self._gelu_tanh,
            MathOperation.GeluDerivative: self._gelu_derivative,
            MathOperation.LogWithBase: self._log_with_base,
            MathOperation.ExpWithBase: self._exp_with_base,
            MathOperation.Neg: self._neg,
            MathOperation.Tanh: self._tanh,
            MathOperation.Fill: self._fill,
            MathOperation.Elu: self._elu,
            MathOperation.Exp: self._exp,
            MathOperation.Exp2: self._exp2,
            MathOperation.Hardsigmoid: self._hardsigmoid,
            MathOperation.Sigmoid: self._sigmoid,
            MathOperation.Threshold: self._threshold,
            MathOperation.ReluMax: self._relu_max,
            MathOperation.ReluMin: self._relu_min,
            MathOperation.Lrelu: self._lrelu,
            MathOperation.Erf: self._erf,
            MathOperation.Erfc: self._erfc,
            MathOperation.Expm1: self._expm1,
            MathOperation.Cbrt: self._cbrt,
            MathOperation.I1: self._i1_bessel,
            MathOperation.Sign: self._sign,
            MathOperation.TanhDerivative: self._tanh_derivative,
            MathOperation.TanhDerivativeLut: self._tanh_derivative_lut,
            MathOperation.RsqrtCompat: self._rsqrt,
            MathOperation.ReciprocalCompat: self._reciprocal,
            MathOperation.Expm1Cw: self._expm1,
            MathOperation.Hardmish: self._hardmish,
            MathOperation.Lgamma: self._lgamma,
            MathOperation.Digamma: self._digamma,
            MathOperation.Identity: self._identity,
            MathOperation.Prelu: self._prelu,
            MathOperation.Rpow: self._rpow,
            MathOperation.UnaryPower: self._unary_power,
            MathOperation.Fmod: self._fmod,
            MathOperation.Remainder: self._remainder,
            MathOperation.UnaryGt: self._unary_gt,
            MathOperation.UnaryLt: self._unary_lt,
            MathOperation.UnaryGe: self._unary_ge,
            MathOperation.UnaryLe: self._unary_le,
            MathOperation.UnaryNe: self._unary_ne,
            MathOperation.UnaryEq: self._unary_eq,
            MathOperation.UnaryMax: self._unary_max,
            MathOperation.UnaryMin: self._unary_min,
            MathOperation.Polygamma: self._polygamma,
            MathOperation.Xielu: self._xielu,
            MathOperation.Hardshrink: self._hardshrink,
            MathOperation.Softplus: self._softplus,
            MathOperation.SigmoidAppx: self._sigmoid_appx,
            MathOperation.SqrtCustom: self._sqrt,
            MathOperation.Add1: self._add1,
            MathOperation.CastFp32ToFp16a: self._cast_fp32_to_fp16a,
            MathOperation.Isinf: self._isinf,
            MathOperation.Isposinf: self._isposinf,
            MathOperation.Isneginf: self._isneginf,
            MathOperation.Isnan: self._isnan,
            MathOperation.Isfinite: self._isfinite,
            MathOperation.LogicalNotUnary: self._logical_not,
            MathOperation.ReduceColumn: self._reduce_columns,
            MathOperation.ReduceRow: self._reduce_rows,
            MathOperation.Cumsum: self._cumsum,
            MathOperation.Typecast: self._typecast,
            # Integer unary ops (routed through the integer path in __call__).
            MathOperation.LeftShift: self._left_shift,
            MathOperation.RightShift: self._right_shift,
            MathOperation.UnaryMaxInt32: self._unary_max_int32,
            MathOperation.UnaryMinInt32: self._unary_min_int32,
            MathOperation.UnaryMaxUint32: self._unary_max_int32,
            MathOperation.UnaryMinUint32: self._unary_min_int32,
        }
        # Elementwise integer unary ops that use the dedicated exact-int path in
        # __call__. Only these ops are routed there; other integer-capable ops
        # (e.g. ReduceColumn/ReduceRow, Typecast) keep their own layout handling.
        self._integer_unary_ops = {
            MathOperation.LeftShift,
            MathOperation.RightShift,
            MathOperation.UnaryMaxInt32,
            MathOperation.UnaryMinInt32,
            MathOperation.UnaryMaxUint32,
            MathOperation.UnaryMinUint32,
        }
        # Fixed dispatch constants shared with sfpu_operations.h: unary shift by 3
        # bits, integer unary max/min against the scalar 1000.
        self._int_shift_amount = 3
        self._int_maxmin_scalar = INT_MAXMIN_SCALAR
        self.data_format = None
        # Precision the SFPU actually evaluates at, which is Dest's and not the output
        # format's. The per-element ops below read this rather than data_format: no
        # output format can restore precision the value has already lost, and none can
        # take away precision Dest still holds.
        self.dst_format = None
        self.dest_acc = DestAccumulation.No

    def __call__(
        self,
        operation,
        operand1,
        data_format,
        dest_acc,
        input_format,
        dimensions: tuple[int, int],
        iterations: int = None,
        dest_idx: int = 0,
        fill_const_value: float = 5,
        reduce_pool: Optional[ReducePool] = None,
        skip_tilize: bool = False,
        unpack_to_srcs: bool = False,
        shift_amount: int = 3,
    ):
        self.data_format = data_format
        self.dst_format = data_format
        self.dest_acc = dest_acc
        # Mirrors the SFPU_SHIFT_AMOUNT template parameter; only the unary shift ops read it.
        self._int_shift_amount = shift_amount

        if operation not in self.ops:
            raise ValueError(f"Unsupported operation: {operation}")

        # Elementwise integer unary ops run on a dedicated exact-int path: tilize ->
        # per-element op -> untilize, staying in the integer dtype (no float dst
        # coercion / FTZ). Gated on the op set so integer-capable non-elementwise ops
        # (ReduceColumn/ReduceRow, Typecast) keep their own layout handling below.
        if (
            operation in self._integer_unary_ops
            and input_format is not None
            and input_format.is_integer()
        ):
            return self._call_integer(operation, operand1, input_format, dimensions)

        # Quantize input to match what hardware actually sees after unpack from L1.
        # Matters most for discontinuous ops (floor/ceil/trunc/frac), where a sub-ULP
        # quantization step across an integer becomes a full 1.0 error.
        operand1 = quantize_input_to_unpack_format(
            operand1, input_format, all_mx_formats=True
        )

        # Special handling for Column and Row reduction which needs to process the entire tensor.
        #
        # This returns before the dst_format derivation below, and so before cast_to_dest_dtype
        # and convert_nan_to_inf -- unobservable while every lane is finite, decisive as soon as
        # one is not, because a reduction propagates its special to the single output element.
        # Both steps are applied here rather than by falling through: the reduce path has already
        # collapsed the tensor, and the code below assumes an element-wise result.
        if operation in [MathOperation.ReduceColumn, MathOperation.ReduceRow]:
            reduced = self.ops[operation](operand1, reduce_pool)
            return self._model_reduce_dest_and_pack(
                reduced, input_format, data_format, self.dest_acc, reduce_pool
            )

        # determine the data format for dst
        if input_format.is_mx_format():
            # MX in L1 always unpacks to Float16_b even if dest_acc=Yes.
            dst_format = DataFormat.Float16_b
        elif unpack_to_srcs and input_format in (
            DataFormat.Float16,
            DataFormat.Float16_b,
        ):
            # SrcS: fp16 stays 16-bit; dest_acc does not widen.
            dst_format = input_format
        elif self.dest_acc == DestAccumulation.Yes:
            dst_format = DataFormat.Float32
        elif DataFormat.Float16 in (input_format, data_format):
            dst_format = DataFormat.Float16
        else:
            dst_format = DataFormat.Float16_b

        self.dst_format = dst_format

        if self.dest_acc == DestAccumulation.No and input_format == DataFormat.Float32:
            # dst in 16-bit mode and 32-bit input: truncation may occur when unpacked to dst
            operand1 = truncate_to_dest_width(
                operand1,
                (
                    DataFormat.Float16
                    if dst_format == DataFormat.Float16
                    else DataFormat.Float16_b
                ),
            )

        # Not to_tensor(): its plain .to() canonicalises every NaN to a *negative* bfloat16
        # one, which would hand the op a sign the input never had -- and then the sign the
        # pack path reads back out would be an artefact of the cast rather than of the
        # datum. See cast_to_dest_dtype.
        tensor = (
            cast_to_dest_dtype(operand1, format_dict[dst_format])
            if operand1.dtype == torch.float32
            else to_tensor(operand1, dst_format)
        )

        if iterations is None or iterations * TILE_SIZE > tensor.numel():
            iterations = tensor.numel() // TILE_SIZE

        if iterations <= 0:
            raise ValueError(f"Invalid iterations: {iterations}")

        result = tensor.clone().flatten()

        # Cumsum accumulates down each tile's columns, so it cannot go through the
        # per-element map below and is evaluated here on the untilized (row-major) view.
        # The tilize that follows puts it in the layout the element-wise path produces, so
        # every later stage (dest rounding, untilize, output conversion) stays shared.
        whole_tensor_res = (
            self._cumsum(result, dimensions)
            if operation == MathOperation.Cumsum
            else None
        )

        if not skip_tilize:
            result = tilize_block(result, dimensions, input_format).flatten()
            if whole_tensor_res is not None:
                # Tilized as Float32 so this permutation does not round the accumulated
                # values; the single Dest-format rounding is applied below, together with
                # the element-wise path's.
                whole_tensor_res = tilize_block(
                    whole_tensor_res, dimensions, DataFormat.Float32
                ).flatten()

        start = ELEMENTS_PER_TILE * dest_idx
        elements_to_process = TILE_SIZE * iterations

        if start + elements_to_process > tensor.numel():
            raise ValueError(
                f"Processing {iterations} iterations from dest_idx={dest_idx} "
                f"would exceed tensor bounds (trying to access element {start + elements_to_process}, "
                f"but tensor has only {tensor.numel()} elements)"
            )

        window = slice(start, start + elements_to_process)
        if whole_tensor_res is not None:
            op_res = whole_tensor_res.tolist()[window]
        else:
            op_res = [
                (
                    self.ops[operation](x, fill_const_value)
                    if operation == MathOperation.Fill
                    else self.ops[operation](x)
                )
                for x in result.tolist()[window]
            ]

        op_dtype = (
            torch.float32
            if data_format in (DataFormat.Bfp4_b, DataFormat.Bfp2_b)
            else format_dict[dst_format]
        )
        op_tensor = torch.tensor(op_res, dtype=torch.float32)
        if operation not in self._NAN_SIGN_TRANSPARENT_OPS:
            # abs() clears the sign bit without disturbing the NaN payload, which is exactly
            # the canonicalisation wanted here. See _NAN_SIGN_TRANSPARENT_OPS.
            op_tensor = torch.where(torch.isnan(op_tensor), op_tensor.abs(), op_tensor)
        if dst_format == DataFormat.Float16:
            # SFPU arithmetic flushes A-exponent results below the FP16 minimum
            # normal before storing them to Dest/SrcS. Apply this before the
            # FP16 cast so a subnormal does not round up to the minimum normal.
            op_tensor = torch.where(
                op_tensor.abs() < torch.finfo(torch.float16).tiny,
                torch.zeros_like(op_tensor),
                op_tensor,
            )
        # Two casts, both NaN-sign preserving: the Dest write's own rounding, then the store
        # into `result`, whose dtype follows input_format through tilize_block and is not
        # always the Dest dtype. Plain assignment for the second would redo it with torch's
        # canonicalising cast and silently undo the first.
        op_rounded = cast_to_dest_dtype(op_tensor, op_dtype).float()
        result[
            ELEMENTS_PER_TILE * dest_idx : ELEMENTS_PER_TILE * dest_idx
            + TILE_SIZE * iterations
        ] = cast_to_dest_dtype(op_rounded, result.dtype)

        if not skip_tilize:
            result = untilize_block(result, input_format, dimensions).flatten()

        if self.data_format in (
            DataFormat.Bfp8_b,
            DataFormat.Bfp4_b,
            DataFormat.Bfp2_b,
        ):
            _bfp_zero_nonfinite_blocks(result)

        match (dst_format, data_format):
            # in the following cases, nans are preserved
            case (DataFormat.Float16, DataFormat.Float16):
                pass
            case (DataFormat.Float32, DataFormat.Float16):
                pass
            case (DataFormat.Float32, DataFormat.Float32):
                pass
            # otherwise, nans are converted to `inf` or a special value
            case _:
                result = convert_nan_to_inf(result)

        if data_format in (DataFormat.Bfp4_b, DataFormat.Bfp2_b):
            result_t = (
                torch.tensor(result, dtype=torch.float32)
                if not isinstance(result, torch.Tensor)
                else result.float()
            )
            tilized = tilize_block(
                result_t.flatten(), dimensions, DataFormat.Float16_b
            ).flatten()
            converter = (
                _bfp4b_to_float16b
                if data_format == DataFormat.Bfp4_b
                else _bfp2b_to_float16b
            )
            result = converter(tilized, dimensions)

        if data_format.is_mx_format():
            # Quantize from the actual Dest/packer source precision. Casting to
            # BF16 here incorrectly models Float16 and FP32 Dest values as
            # Float16 -> BF16 -> MX or Float32 -> BF16 -> MX.
            result = quantize_mx_tensor_chunked(result, data_format)

        # depending on `data_format`, `inf` values may get converted when unpacked to L1.
        # Cast to the target data_format dtype before replacing inf so that
        # replacement values larger than float16-max (65504) don't overflow for torch tensors.
        if dst_format == DataFormat.Float16:
            target_dtype = format_dict[data_format]
            if isinstance(result, torch.Tensor) and result.dtype != target_dtype:
                result = result.to(target_dtype)
            match data_format:
                case DataFormat.Float16_b:
                    result = convert_inf_to_value(result, 130560.0)
                case DataFormat.Float32:
                    result = convert_inf_to_value(result, 131008.0)
                case DataFormat.Bfp8_b:
                    result = convert_inf_to_value(result, 130048.0)
                case DataFormat.Bfp4_b:
                    result = convert_inf_to_value(result, 130048.0)
                case DataFormat.Bfp2_b:
                    result = convert_inf_to_value(result, 130048.0)

        # Final FTZ pass — see _apply_ftz for rationale. Centralised here
        # because the BFP helpers above no longer FTZ internally.
        return _apply_ftz(
            torch.tensor(result, dtype=format_dict[data_format]), data_format
        )

    # Helper functions
    def handle_infinite_numbers(self, expected: float) -> float:
        """Handle infinite numbers based on the data format.
        Tensix will return inf, -inf for B_exponent formats, and NaN for Float16.
        Returns:
            float: Infinite number
            Depending on our format we either return NaN or +/- inf.
        """
        if self.data_format.is_exponent_B():
            return expected
        else:  # self.data_format == DataFormat.Float16:
            return math.nan

    def _torch_unary(self, x, torch_fn) -> float:
        """Apply torch_fn to scalar x in fp32, then enforce the
        format-aware NaN rule: convert +/-inf to NaN when the dest is
        A-exponent (Float16).
        """
        result = torch_fn(torch.tensor(x, dtype=torch.float32)).item()
        if math.isinf(result) and not self.data_format.is_exponent_B():
            return math.nan
        return result

    # Operation methods
    def _abs(self, x):
        return abs(x)

    def _add1(self, x):
        # add1(x) = x + 1
        return x + 1.0

    # Predicate ops: return 1.0 where the test holds, else 0.0. Each x is a
    # python float, so math.isinf/isnan give the same verdict the SFPU does on
    # the corresponding inf/nan dest bits.
    def _isinf(self, x):
        return 1.0 if math.isinf(x) else 0.0

    def _isposinf(self, x):
        return 1.0 if (math.isinf(x) and x > 0) else 0.0

    def _isneginf(self, x):
        return 1.0 if (math.isinf(x) and x < 0) else 0.0

    def _isnan(self, x):
        return 1.0 if math.isnan(x) else 0.0

    def _isfinite(self, x):
        return 1.0 if math.isfinite(x) else 0.0

    def _logical_not(self, x):
        # logical_not(x) = (x == 0) ? 1 : 0. NaN != 0, so logical_not(nan) = 0.
        return 1.0 if x == 0 else 0.0

    def _cast_fp32_to_fp16a(self, x):
        # cast_fp32_to_fp16a lowers to sfpi::convert<vFloat16a>, which rounds each
        # lane to the fp16a *mantissa* (10 fraction bits, round-to-nearest-even)
        # while the value stays in the fp32-range SFPU LREG. It only reduces
        # mantissa precision; it does NOT clamp the exponent to the fp16 range, so
        # magnitudes above the fp16 max (65504) are preserved (rounded), not
        # overflowed to +/-inf. Model that by rounding the fp32 bit pattern's
        # 23-bit mantissa down to 10 bits (drop 13) with round-half-to-even,
        # keeping the exponent intact.
        bits = struct.unpack("<I", struct.pack("<f", x))[0]
        exponent = (bits >> 23) & 0xFF
        if exponent == 0xFF:
            # Non-finite input (inf/nan): pass the bit pattern through unchanged.
            return struct.unpack("<f", struct.pack("<I", bits))[0]
        drop = 13
        lower_mask = (1 << drop) - 1
        halfway = 1 << (drop - 1)
        remainder = bits & lower_mask
        truncated = bits & ~lower_mask
        # Round-half-to-even: up on >halfway, or ==halfway with an odd kept LSB.
        if remainder > halfway or (remainder == halfway and (truncated >> drop) & 1):
            truncated += 1 << drop  # carry may ripple into the exponent (correct)
        return struct.unpack("<f", struct.pack("<I", truncated & 0xFFFFFFFF))[0]

    # Comparison-to-zero ops. The Quasar kernel builds the strict comparisons from
    # SFPSETCC sign + magnitude tests (ltz = negative AND nonzero, gtz = positive AND
    # nonzero), so ±0.0 is excluded from ltz/gtz and the semantics reduce to plain IEEE:
    #   eqz/nez: magnitude tests (both +0.0 and -0.0 count as zero).
    #   ltz/gtz: strict (x < 0 / x > 0); ltz(-0.0)=gtz(+0.0)=False.
    #   lez/gez: x <= 0 / x >= 0, inclusive of ±0.0.
    def _equal_zero(self, x):
        return 1.0 if x == 0.0 else 0.0

    def _not_equal_zero(self, x):
        return 1.0 if x != 0.0 else 0.0

    def _signbit(self, x):
        # Mirrors the kernel: logical-shift the fp32 bit pattern right by 31,
        # i.e. return 1.0 iff the sign bit is set (negative, incl. -0.0).
        return 1.0 if math.copysign(1.0, x) < 0.0 else 0.0

    def _less_than_zero(self, x):
        return 1.0 if x < 0.0 else 0.0

    def _greater_than_zero(self, x):
        return 1.0 if x > 0.0 else 0.0

    def _less_than_equal_zero(self, x):
        return 1.0 if x <= 0.0 else 0.0

    def _greater_than_equal_zero(self, x):
        return 1.0 if x >= 0.0 else 0.0

    def _atanh(self, x):
        return self._torch_unary(x, torch.atanh)

    def _asinh(self, x):
        return math.asinh(x)

    def _acosh(self, x):
        return self._torch_unary(x, torch.acosh)

    def _cos(self, x):
        # torch rather than math: math.cos raises ValueError("math domain error") on a
        # non-finite input, so a cat-B special reached this as an exception rather than as
        # a result. IEEE gives NaN for cos(+/-inf) and for cos(NaN), which torch.cos does.
        return self._torch_unary(x, torch.cos)

    def _log(self, x):
        return self._torch_unary(x, torch.log)

    # The dispatch is metal calculate_log with IS_BASE_TWO=true and base_scale = fp32
    # 1/ln(2), i.e. log2 with an exact exponent term, so torch.log2 is the golden.
    def _log_with_base(self, x):
        return self._torch_unary(x, torch.log2)

    def _log1p(self, x):
        return self._torch_unary(x, torch.log1p)

    def _reciprocal(self, x):
        return self._torch_unary(x, torch.reciprocal)

    def _sin(self, x):
        # torch rather than math, same reason as _cos: math.sin raises on a non-finite input.
        # IEEE gives NaN for sin(+/-inf) and sin(NaN), which torch.sin does.
        return self._torch_unary(x, torch.sin)

    def _relu(self, x):
        return max(0.0, x)

    def _rsqrt(self, x):
        return self._torch_unary(x, torch.rsqrt)

    def _sqrt(self, x):
        return self._torch_unary(x, torch.sqrt)

    def _tanh(self, x):
        return math.tanh(x)

    def _tanhshrink(self, x):
        # tanhshrink(x) = x - tanh(x)
        return x - math.tanh(x)

    def _floor(self, x):
        return math.floor(x) if math.isfinite(x) else x

    def _ceil(self, x):
        return math.ceil(x) if math.isfinite(x) else x

    def _trunc(self, x):
        return math.trunc(x) if math.isfinite(x) else x

    def _frac(self, x):
        # Fractional part with sign of x: frac(x) = x - trunc(x)
        return (x - math.trunc(x)) if math.isfinite(x) else x

    def _round(self, x):
        # decimals=0, round-half-to-even (matches the kernel's _round_even_ and
        # torch.round / Python's banker's rounding).
        return float(round(x)) if math.isfinite(x) else x

    def _tan(self, x):
        # torch rather than math, same reason as _sin / _cos. IEEE gives NaN for tan(+/-inf);
        # tan(+/-0) = +/-0.
        return self._torch_unary(x, torch.tan)

    def _atan(self, x):
        return math.atan(x)

    def _asin(self, x):
        # Domain restricted to [-1, 1] by the stimuli spec -- but cat B injects specials from
        # outside any domain, and math.asin raises on those rather than returning NaN. See
        # _tan.
        return self._torch_unary(x, torch.asin)

    def _acos(self, x):
        # Domain restricted to [-1, 1] by the stimuli spec -- same caveat as _asin.
        return self._torch_unary(x, torch.acos)

    def _sinh(self, x):
        return math.sinh(x)

    def _cosh(self, x):
        return math.cosh(x)

    def _square(self, x):
        # A *finite* input that overflows saturates, and handle_infinite_numbers picks inf or
        # NaN per format. A non-finite input is not an overflow and must propagate: isfinite(x * x)
        # alone is false for NaN too, so the golden reported square(NaN) = inf.
        if math.isnan(x):
            return x
        if not math.isfinite(x * x):
            return self.handle_infinite_numbers(math.inf)
        return x * x

    def _celu(self, x):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.dst_format])
        )
        return torch.nn.functional.celu(input_tensor, alpha=1.0).item()

    def _silu(self, x):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.dst_format])
        )
        return torch.nn.functional.silu(input_tensor).item()

    def _erfinv(self, x):
        # domain (-1, 1); |x| >= 1 is excluded by the stimuli domain registry.
        return self._torch_unary(x, torch.erfinv)

    def _heaviside(self, x):
        # Matches calculate_heaviside: 0 for x<0, 1 for x>0, and the
        # dispatch-supplied value (0.5) at exactly x==0.
        if x < 0.0:
            return 0.0
        if x > 0.0:
            return 1.0
        return 0.5

    def _softshrink(self, x, lambd=SOFTSHRINK_LAMBDA):
        # Matches calculate_softshrink with lambda fixed to the dispatch constant.
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.dst_format])
        )
        return torch.nn.functional.softshrink(input_tensor, lambd=lambd).item()

    def _softsign(self, x):
        # softsign(x) = x / (1 + |x|).
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.dst_format])
        )
        return torch.nn.functional.softsign(input_tensor).item()

    def _mish(self, x):
        # mish(x) = x * tanh(softplus(x)).
        return self._torch_unary(x, torch.nn.functional.mish)

    def _selu(self, x):
        # selu with default scale/alpha; matches the dispatch constants.
        return self._torch_unary(x, torch.nn.functional.selu)

    def _i0(self, x):
        # modified Bessel I0; kernel uses a poly approx valid on |x| <= 3.75.
        #
        # torch.special.i0 returns NaN at +/-inf. That is a torch limitation, not the
        # mathematics: I0 is even and increases without bound, so I0(+/-inf) = +inf -- which is
        # what the kernel returns. Taking torch's answer made the golden the wrong party.
        if math.isnan(x):
            return x
        if math.isinf(x):
            return self.handle_infinite_numbers(math.inf)
        return self._torch_unary(x, torch.special.i0)

    def _rdiv(self, x, value=2.0):
        # rdiv(x) = value / x; value fixed to the dispatch constant (2.0).
        return self._torch_unary(x, lambda t: value / t)

    def _clamp(self, x, min_val=CLAMP_MIN, max_val=CLAMP_MAX):
        # Metal calculate_clamp is the composition sfpu_clamp models -- see its docstring.
        return sfpu_clamp(x, min_val, max_val)

    def _hardtanh(self, x, min_val=CLAMP_MIN, max_val=CLAMP_MAX):
        # Metal calculate_hardtanh is sfpi::clamp, the same composition sfpu_clamp models,
        # so Hardtanh's golden IS Clamp's. The identity is pinned in test_sfpu_domains
        # (test_hardtanh_golden_matches_the_clamp_golden).
        return sfpu_clamp(x, min_val, max_val)

    def _elu(self, x):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.dst_format])
        )
        return torch.nn.functional.elu(input_tensor, alpha=1.0).item()

    def _exp(self, x):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.dst_format])
        )
        return torch.exp(input_tensor).item()

    def _exp2(self, x):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.dst_format])
        )
        return torch.exp2(input_tensor).item()

    def _exp_with_base(self, x):
        # Matches the dispatch: calculate_exponential with SCALE_EN and a bf16
        # scale of 0.5, i.e. exp(0.5 * x). 0.5 is exact in bf16, so the only error
        # versus this golden is the shared exp approximation itself.
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.dst_format])
        )
        return torch.exp(0.5 * input_tensor).item()

    def _call_integer(self, operation, operand1, input_format, dimensions):
        """Exact integer golden: tilize -> elementwise op -> untilize, in int dtype.

        tilize/untilize is a pure permutation and cancels for elementwise ops, but is
        kept to mirror the float/binary golden layout handling exactly.
        """
        torch_dtype = format_dict[input_format]
        tensor = (
            operand1 if isinstance(operand1, torch.Tensor) else torch.tensor(operand1)
        )
        tensor = tensor.to(torch_dtype).flatten()
        tilized = tilize_block(tensor, dimensions, input_format).flatten()
        op = self.ops[operation]
        op_res = [int(op(int(x))) for x in tilized.tolist()]
        result = torch.tensor(op_res, dtype=torch_dtype)
        result = untilize_block(result, input_format, dimensions).flatten()
        return result

    # The two unary shifts do NOT share an out-of-range rule. calculate_left_shift zeroes the
    # result; calculate_right_shift clamps the amount to 31 and shifts anyway. They agree for a
    # positive operand -- x >> 31 is 0 -- and part company for a negative one, where the clamped
    # arithmetic shift gives -1. This is also where the *unary* right shift differs from the
    # *binary* one, which produces 0 for both signs (BinarySFPUGolden._right_shift).
    #
    # Both kernels take the amount as `const uint`, so a negative Python amount arrives as a
    # large unsigned and is out of range on that path rather than by being negative.
    def _shift_amount(self) -> int:
        return int(self._int_shift_amount)

    def _left_shift(self, x):
        # calculate_left_shift: `out_of_range ? vInt(0) : (v << amt)`.
        n = self._shift_amount()
        if n < 0 or n >= 32:
            return 0
        return int(x) << n

    def _right_shift(self, x):
        # calculate_right_shift: `eff = (shift_amt >= 32) ? 31u : shift_amt`, then a logical
        # shift with the sign bits OR'd back in for a negative operand -- an arithmetic shift
        # at the clamped amount. Python's >> on ints is already sign-propagating, so shifting
        # by eff reproduces it, including the -1 an out-of-range amount gives for a negative.
        n = self._shift_amount()
        eff = 31 if (n < 0 or n >= 32) else n
        return int(x) >> eff

    def _unary_max_int32(self, x):
        return max(int(x), self._int_maxmin_scalar)

    def _unary_min_int32(self, x):
        return min(int(x), self._int_maxmin_scalar)

    def _neg(self, x):
        return -x

    def _typecast(self, x):
        # Typecast is an elementwise identity at the value level; the src->dst format
        # conversion is applied by the golden framework's dst_format / output-format
        # casting (and, for class-1 MXFP8 pairs, by the unpack/pack gasket on hardware).
        return x

    def _gelu(self, x):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.dst_format])
        )
        return torch.nn.functional.gelu(input_tensor).item()

    def _gelu_tanh(self, x):
        # Matches calculate_gelu_tanh: the tanh approximation of GELU,
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.dst_format])
        )
        return torch.nn.functional.gelu(input_tensor, approximate="tanh").item()

    def _gelu_derivative(self, x):
        # d/dx [x * Phi(x)] = Phi(x) + x * phi(x), with the erf-based standard
        # normal CDF/PDF (matches the kernel's exact-gelu derivative, not the
        # tanh approximation): Phi(x) = 0.5*(1+erf(x/sqrt2)), phi(x) = N(0,1) pdf.
        phi = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        return cdf + x * phi

    def _fill(self, x, const_value=5):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.dst_format])
        )
        return input_tensor.fill_(const_value).item()

    # Slope and offset exactly as hardsigmoid_init programs them into vConstFloatPrgm0/1.
    # 0.1666666716337204 is the fp32 value of 1/6, not a rounding of the literal.
    _HARDSIGMOID_SLOPE = 0.1666666716337204
    _HARDSIGMOID_OFFSET = 0.5

    def _hardsigmoid(self, x):
        # The kernel is `_relu_max_body_(x * slope + offset, 1.0)` -- the same helper relu_max
        # uses, which is why both diverged from their goldens at NaN in the same way. Not
        # torch.nn.functional.hardsigmoid: that clamps under IEEE and returns NaN.
        return sfpu_relu_max(
            float(x) * self._HARDSIGMOID_SLOPE + self._HARDSIGMOID_OFFSET, 1.0
        )

    def _sigmoid(self, x):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.dst_format])
        )
        return torch.nn.functional.sigmoid(input_tensor).item()

    def _threshold(self, x, t=THRESHOLD_T, v=THRESHOLD_V):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.dst_format])
        )
        return torch.nn.functional.threshold(input_tensor, t, v).item()

    def _relu_max(self, x, threshold=RELU_MAX_THRESHOLD):
        # _relu_max_body_ is `v_if (val > threshold) val = threshold` then
        # `v_if (val < 0) val = 0`. The first is a two-vector compare and so uses the total
        # order -- a NaN is greater than the threshold and is replaced by it, after which the
        # relu clamp sees a finite value. Order matters: relu-then-min would keep the NaN.
        return sfpu_relu_max(float(x), float(threshold))

    def _relu_min(self, x, threshold=RELU_MIN_THRESHOLD):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.dst_format])
        )
        return torch.max(input_tensor, torch.tensor(threshold)).item()

    def _lrelu(self, x, negative_slope=LRELU_NEGATIVE_SLOPE):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.dst_format])
        )
        return torch.nn.functional.leaky_relu(
            input_tensor, negative_slope=negative_slope
        ).item()

    def _erf(self, x):
        return self._torch_unary(x, torch.erf)

    def _erfc(self, x):
        return self._torch_unary(x, torch.erfc)

    def _expm1(self, x):
        return self._torch_unary(x, torch.expm1)

    def _cbrt(self, x):
        # Cube root preserves sign: cbrt(x) = sign(x) * |x|^(1/3).
        return self._torch_unary(
            x, lambda t: torch.sign(t) * torch.abs(t).pow(1.0 / 3.0)
        )

    def _i1_bessel(self, x):
        # Modified Bessel I1; kernel poly approx is valid on |x| <= ~3.75.
        #
        # Same torch limitation as _i0, with the sign kept: I1 is odd, so I1(+/-inf) = +/-inf.
        # Fixed for correctness rather than to enrol the op -- I1 is still outside
        # SPECIALS_READY_OPS because the *kernel* saturates a non-finite input to
        # +/-1.1547668e37, which is the Log saturation question and not a golden matter.
        if math.isnan(x):
            return x
        if math.isinf(x):
            return math.copysign(self.handle_infinite_numbers(math.inf), x)
        return self._torch_unary(x, torch.special.i1)

    def _sign(self, x):
        # Matches calculate_sign: -1 for x<0, 0 for x==0, +1 otherwise.
        return float(torch.sign(torch.tensor(x, dtype=torch.float32)).item())

    def _tanh_derivative(self, x):
        # tanh'(x) = 1 - tanh(x)^2 = sech^2(x).
        t = math.tanh(x)
        return 1.0 - t * t

    def _tanh_derivative_lut(self, x):
        # Legacy tt-llk _calculate_tanh_derivative_ computes 1 - tanh(x)^2 where
        # tanh comes from the raw 3-region SFPLUT (same LUT as the tanh kernel),
        # NOT an accurate tanh. So the faithful golden models that piecewise-linear
        # LUT, gated by |x| into exponent buckets (breakpoints at 1.0 and 2.0):
        #   |x| < 1 : 0.90625*|x|
        #   |x| < 2 : 0.09375*|x| + 0.8125
        #   else    : 1.0            (saturates, so tanh' -> 0)
        # The LUT is odd, but 1 - t^2 squares away the sign. This is the kernel's
        # true contract; validating it against accurate tanh would fail by design
        # (the header documents catastrophic cancellation for |x| > ~3.4).
        a = abs(x)
        if a < 1.0:
            t = 0.90625 * a
        elif a < 2.0:
            t = 0.09375 * a + 0.8125
        else:
            t = 1.0
        return 1.0 - t * t

    def _hardmish(self, x):
        # hardmish(x) = x * clamp(0.5*x + 1, 0, 1).
        return self._torch_unary(x, lambda t: t * torch.clamp(0.5 * t + 1.0, 0.0, 1.0))

    def _lgamma(self, x):
        # Single-tile Stirling kernel is accurate for x >= ~0.5 (domain-restricted).
        return self._torch_unary(x, torch.lgamma)

    def _digamma(self, x):
        # digamma = d/dx ln(gamma(x)); kernel LUT is fit on [0.01, 102].
        return self._torch_unary(x, torch.digamma)

    def _identity(self, x):
        return x

    # Fixed scalar dispatch constants mirrored from sfpu_operations.h. The ones an edge
    # probe has to land on exactly come from sfpu_dispatch_constants, which both this
    # golden and sfpu_domains._OP_EDGE_POINTS read — see that module for why they are not
    # written twice. The rest are local because nothing probes at them.
    _PRELU_SLOPE = PRELU_SLOPE
    _RPOW_BASE = 2.0
    _UNARY_POWER_EXP = 2.0
    _FMOD_DIVISOR = 2.0
    _REMAINDER_DIVISOR = 2.0
    _UNARY_COMP_THRESHOLD = UNARY_COMP_THRESHOLD
    _UNARY_MAX_MIN_VALUE = UNARY_MAX_MIN_VALUE
    _POLYGAMMA_ORDER = 1
    _XIELU_ALPHA_P = 1.0
    _XIELU_ALPHA_N = 1.0
    _XIELU_BETA = 0.5
    _HARDSHRINK_LAMBDA = HARDSHRINK_LAMBDA
    _SOFTPLUS_BETA = SOFTPLUS_BETA
    _SOFTPLUS_THRESHOLD = SOFTPLUS_THRESHOLD

    def _prelu(self, x):
        return x if x >= 0.0 else self._PRELU_SLOPE * x

    def _rpow(self, x):
        return self._torch_unary(
            x, lambda t: torch.pow(torch.tensor(self._RPOW_BASE), t)
        )

    def _unary_power(self, x):
        return self._torch_unary(x, lambda t: torch.pow(t, self._UNARY_POWER_EXP))

    def _fmod(self, x):
        return self._torch_unary(
            x, lambda t: torch.fmod(t, torch.tensor(self._FMOD_DIVISOR))
        )

    def _remainder(self, x):
        return self._torch_unary(
            x, lambda t: torch.remainder(t, torch.tensor(self._REMAINDER_DIVISOR))
        )

    # The four ordered comparisons rank by the SFPU's total order rather than by IEEE, so a
    # NaN operand compares as larger than every finite value instead of making the result
    # false. See sfpu_total_order_key. Python's own operators are IEEE, so they cannot be
    # used here even though they agree on every finite input.
    def _unary_gt(self, x):
        return 1.0 if _order(x) > _order(self._UNARY_COMP_THRESHOLD) else 0.0

    def _unary_lt(self, x):
        return 1.0 if _order(x) < _order(self._UNARY_COMP_THRESHOLD) else 0.0

    def _unary_ge(self, x):
        return 1.0 if _order(x) >= _order(self._UNARY_COMP_THRESHOLD) else 0.0

    def _unary_le(self, x):
        return 1.0 if _order(x) <= _order(self._UNARY_COMP_THRESHOLD) else 0.0

    def _unary_ne(self, x):
        return 1.0 if x != self._UNARY_COMP_THRESHOLD else 0.0

    def _unary_eq(self, x):
        return 1.0 if x == self._UNARY_COMP_THRESHOLD else 0.0

    def _unary_max(self, x):
        return sfpu_max(x, self._UNARY_MAX_MIN_VALUE)

    def _unary_min(self, x):
        # Under the total order a +NaN is the maximum, so min() returns the *other* operand
        # -- which is why this diverged from a Python min() and _unary_max did not.
        return sfpu_min(x, self._UNARY_MAX_MIN_VALUE)

    def _polygamma(self, x):
        return self._torch_unary(x, lambda t: torch.polygamma(self._POLYGAMMA_ORDER, t))

    def _xielu(self, x):
        # Mirrors calculate_xielu: beta = 0.5, alpha_p/alpha_n learnable params.
        beta_mul_x = self._XIELU_BETA * x
        if x > 0.0:
            return self._XIELU_ALPHA_P * x * x + beta_mul_x
        return self._XIELU_ALPHA_N * (math.expm1(x) - x) + beta_mul_x

    def _hardshrink(self, x):
        # hardshrink(x) = x when |x| > lambda, else 0.
        #
        # NaN is not "inside the shrink band": every comparison against it is false, so
        # abs(x) > lambda failed and the golden returned 0.0. It propagates, which is both what
        # torch.nn.functional.hardshrink does and what the kernel does.
        if math.isnan(x):
            return x
        return x if abs(x) > self._HARDSHRINK_LAMBDA else 0.0

    def _softplus(self, x):
        # softplus(x) = (1/beta) * ln(1 + exp(beta*x)); linear above threshold.
        return self._torch_unary(
            x,
            lambda t: torch.nn.functional.softplus(
                t, beta=self._SOFTPLUS_BETA, threshold=self._SOFTPLUS_THRESHOLD
            ),
        )

    def _sigmoid_appx(self, x):
        # Golden is the exact sigmoid; the kernel is a LUT approximation of it.
        return self._torch_unary(x, torch.sigmoid)

    def _cumsum(self, x, dimensions: tuple[int, int]):
        """Column-wise (top-to-bottom) cumulative sum inside each 32x32 tile.

        Reached through the whole-tensor branch of __call__, so ``x`` is the untilized
        (row-major) view of the [H, W] tensor already in the Dest format. Tiles are
        independent: the kernel runs once per tile with ``first = true``, zeroing the
        cross-tile carry. Accumulated in float32 to match the SFPU's FP32 running total —
        the caller applies the single Dest-format rounding.
        """
        rows, cols = dimensions[0], dimensions[1]
        tiles = x.reshape(rows // TILE_DIM, TILE_DIM, cols // TILE_DIM, TILE_DIM)
        return torch.cumsum(tiles.to(torch.float32), dim=1).flatten()

    # Pools whose NaN result is *emitted by SFPMAD* rather than selected from a lane. Sum and
    # Average accumulate, so a NaN they produce is arithmetic and its sign is the ISA's to choose
    # (canonical 0x7fc00000 on Blackhole, unspecified on Wormhole). Max and Min are a bare
    # SFPSWAP(VEC_MIN_MAX) -- see _reduce_extremum -- so a NaN they return is the lane they picked,
    # sign included, and it stays asserted on both arches.
    _SFPMAD_REDUCE_POOLS = (ReducePool.Sum, ReducePool.Average)

    @classmethod
    def _model_reduce_dest_and_pack(
        cls,
        reduced,
        input_format: DataFormat,
        output_format: DataFormat,
        dest_acc,
        reduce_pool: ReducePool = None,
    ):
        """The Dest write and the pack, for the reduce path that used to skip both.

        Same two steps and the same order as the element-wise path above and as
        BinarySFPUGolden: canonicalise the sign of a NaN the fold *emitted*, round to the width
        Dest holds keeping that sign across the cast (cast_to_dest_dtype, not `.to()`), then
        substitute a signed infinity wherever the packer cannot write a NaN through this pipeline.

        The canonicalisation is what stops the host library deciding the answer. `torch.sum` over a
        column holding `+inf` and `-inf` returns a *negatively* signed NaN, which the substitution
        below would turn into `-inf` -- where Blackhole's SFPMAD emits the canonical 0x7fc00000 and
        packs `+inf`. Same defect the element-wise and binary paths already carry a fix for, in the
        one path that returns before reaching theirs.

        Only the float axis is modelled. Integer reduce operands never reach here -- __call__
        routes them through _call_integer or keeps their own layout handling -- and
        nan_survives_to_l1 makes no claim about integer formats.
        """
        if input_format.is_integer() or output_format.is_integer():
            return reduced

        if reduce_pool in cls._SFPMAD_REDUCE_POOLS:
            reduced = torch.where(torch.isnan(reduced), reduced.abs(), reduced)

        dst_format = (
            DataFormat.Float32
            if dest_acc == DestAccumulation.Yes
            else (
                DataFormat.Float16
                if DataFormat.Float16 in (input_format, output_format)
                else DataFormat.Float16_b
            )
        )
        result = cast_to_dest_dtype(
            reduced.to(torch.float32), format_dict[dst_format]
        ).float()
        if not nan_survives_to_l1(input_format, output_format, dest_acc):
            result = convert_nan_to_inf(result)
        return result.reshape(reduced.shape)

    @staticmethod
    def _reduce_extremum(x, dim: int, want_max: bool):
        """Fold *x* along *dim* with the comparator the reduce kernel actually uses.

        ckernel_sfpu_reduce.h reduces MAX/MIN with a bare `TTI_SFPSWAP(VEC_MIN_MAX)` and no NaN
        guard, so the documented SFPU total order reaches the result: +NaN outranks every finite
        value and -NaN is below -inf. torch.max/torch.min propagate a NaN instead, so a column
        holding one +NaN gives a *finite* minimum on hardware where torch.min returns NaN.

        Read from the kernel, not inferred from SFPSWAP's ISA page: the six binary comparison
        kernels route through the same instruction and wrap it in an explicit NaN rejection, which
        makes them IEEE. Whether the order reaches the result is a property of the guard.

        A fold rather than one vectorised compare, since the order is not expressible as
        torch.max: rank by sfpu_order_key_elementwise, then select.

        **Integer formats keep torch.** ReduceColumn/ReduceRow are not routed through
        _call_integer, so an Int32 or UInt32 reduce arrives here with an integer dtype, and the
        sign-magnitude remap would reinterpret those bits as a float pattern. It would be the
        wrong model anyway -- the Int32 reduce path is _emit_int32_signed_cswap_, which corrects
        SFPSWAP's order back to two's complement, and there is no NaN on that axis.
        """
        if not torch.is_floating_point(x):
            return (
                torch.max(x, dim=dim).values
                if want_max
                else torch.min(x, dim=dim).values
            )

        moved = x.movedim(dim, 0)
        result = moved[0]
        for i in range(1, moved.shape[0]):
            result = (
                sfpu_max_elementwise(result, moved[i])
                if want_max
                else sfpu_min_elementwise(result, moved[i])
            )
        return result

    def _reduce_columns(self, x, reduce_pool: ReducePool):
        """Reduce columns across tiles, computing sum, average, or max."""
        # Reduce columns within this tensor
        # Take max along the height (dim=0) for each column
        if reduce_pool == ReducePool.Max:
            reduced_tile = self._reduce_extremum(x, dim=0, want_max=True)
        elif reduce_pool == ReducePool.Min:
            reduced_tile = self._reduce_extremum(x, dim=0, want_max=False)
        elif reduce_pool == ReducePool.Sum:
            reduced_tile = torch.sum(x, dim=0)
        elif reduce_pool == ReducePool.Average:
            reduced_tile = torch.sum(x, dim=0) / x.shape[0]
        else:
            raise ValueError(f"Unsupported reduce pool type: {reduce_pool}")

        # Construct golden tensor: first row is column max, others are zero
        reduced_tile_tensor = torch.zeros_like(x)
        reduced_tile_tensor[0, :] = reduced_tile
        return reduced_tile_tensor

    def _reduce_rows(self, x, reduce_pool: ReducePool):
        """Reduce rows across tiles, computing sum, average, min, or max."""
        if reduce_pool == ReducePool.Max:
            reduced_tile = self._reduce_extremum(x, dim=1, want_max=True)
        elif reduce_pool == ReducePool.Min:
            reduced_tile = self._reduce_extremum(x, dim=1, want_max=False)
        elif reduce_pool == ReducePool.Sum:
            reduced_tile = torch.sum(x, dim=1)
        elif reduce_pool == ReducePool.Average:
            reduced_tile = torch.sum(x, dim=1) / x.shape[1]
        else:
            raise ValueError(
                f"Unsupported reduce pool type for row reduction: {reduce_pool}"
            )

        # Construct golden tensor: first column is row max, others are zero
        reduced_tile_tensor = torch.zeros_like(x)
        reduced_tile_tensor[:, 0] = reduced_tile
        return reduced_tile_tensor


@register_golden
class EltwiseBinaryGolden(FidelityMasking):
    def __init__(self):
        self.ops = {
            MathOperation.Elwadd: self._add,
            MathOperation.Elwsub: self._sub,
            MathOperation.Elwmul: self._mul,
        }

    def _quantize_input(self, operand, input_fmt, output_fmt):
        """Quantize a single operand to match what hardware sees after unpack."""
        if input_fmt is None:
            return to_tensor(operand, output_fmt)
        if input_fmt == DataFormat.Bfp2_b:
            return _bfp2b_to_float16b(operand)
        if input_fmt == DataFormat.Bfp4_b:
            return _bfp4b_to_float16b(operand)
        if input_fmt == DataFormat.Bfp8_b:
            return _bfp8b_to_float16b(operand)
        if input_fmt.is_mx_format():
            return quantize_mx_tensor_chunked(operand, input_fmt)
        return to_tensor(operand, input_fmt)

    _UNSET = object()

    def _compute_eltwise(
        self, op, t1, t2, math_format_for_fidelity, math_fidelity, keep_float32=False
    ):
        """Compute a single eltwise operation with fidelity masking.

        When ``keep_float32`` is True, the result stays in float32 (used by
        accumulation paths that need extra precision before the final cast).
        """
        MATH_FIDELITY_TO_ITER_COUNT = {
            MathFidelity.LoFi: 0,
            MathFidelity.HiFi2: 1,
            MathFidelity.HiFi3: 2,
            MathFidelity.HiFi4: 3,
        }
        fidelity_iter_count = MATH_FIDELITY_TO_ITER_COUNT[math_fidelity]

        if keep_float32:
            t1 = t1.to(torch.float32)
            t2 = t2.to(torch.float32)

        if op == MathOperation.Elwmul:
            result = None
            for fidelity_iter in range(fidelity_iter_count + 1):
                t1, t2 = self._apply_fidelity_masking(
                    math_format_for_fidelity, t1, t2, fidelity_iter
                )
                phase_result = self.ops[op](t1, t2)
                if fidelity_iter == 0:
                    result = phase_result
                else:
                    result += phase_result
        else:
            result = self.ops[op](t1, t2)

        return result

    def _binary_int_op(self, op, t1, t2, data_format):
        """Integer eltwise op in int32. Int8 operands cannot overflow int32."""
        torch_format = format_dict[data_format]
        t1_int32 = t1.to(torch.int32)
        t2_int32 = t2.to(torch.int32)
        if op == MathOperation.Elwadd:
            res = t1_int32 + t2_int32
        elif op == MathOperation.Elwsub:
            res = t1_int32 - t2_int32
        elif op == MathOperation.Elwmul:
            res = t1_int32 * t2_int32
        else:
            raise ValueError(f"Unsupported integer eltwise operation: {op}")
        return res.to(torch_format)

    def _eltwise_integer(
        self,
        op,
        operand1,
        operand2,
        data_format,
        input_format,
        acc_to_dest,
        tile_shape,
        num_tiles_per_accumulation,
    ):

        t1 = to_tensor(operand1, input_format)
        t2 = to_tensor(operand2, input_format)

        if acc_to_dest:
            tile_size = tile_shape.total_tile_size()
            num_total_tiles = t1.numel() // tile_size
            num_blocks = num_total_tiles // num_tiles_per_accumulation

            t1_tiles = t1.view(num_total_tiles, tile_size)
            t2_tiles = t2.view(num_total_tiles, tile_size)

            accumulated = []
            for block in range(num_blocks):
                partials = [
                    self._binary_int_op(
                        op,
                        t1_tiles[block * num_tiles_per_accumulation + tile],
                        t2_tiles[block * num_tiles_per_accumulation + tile],
                        data_format,
                    )
                    for tile in range(num_tiles_per_accumulation)
                ]
                accumulated.append(apply_l1_accumulation(partials, data_format))
            return torch.cat(accumulated)

        return self._binary_int_op(op, t1, t2, data_format)

    def __call__(
        self,
        op,
        operand1,
        operand2,
        data_format,
        math_fidelity,
        input_format=None,
        input_format_B=_UNSET,
        acc_to_dest=False,
        tile_shape=None,
        num_tiles_per_accumulation=1,
    ):
        if tile_shape is None:
            tile_shape = construct_tile_shape()

        if op not in self.ops:
            raise ValueError(f"Unsupported Eltwise operation: {op}")

        # If input_format_B is not provided at all, default to input_format.
        # If explicitly passed as None, it means "already quantized, skip".
        if input_format_B is EltwiseBinaryGolden._UNSET:
            input_format_B = input_format

        if input_format is not None and input_format.is_integer():
            return self._eltwise_integer(
                op,
                operand1,
                operand2,
                data_format,
                input_format,
                acc_to_dest,
                tile_shape,
                num_tiles_per_accumulation,
            )

        # On Quasar with IMPLIED_MATH_FORMAT=Yes, the HW dest register's
        # physical storage is implied from the SrcA tag: Float16 input →
        # FP16A (S1E5M10); Float16_b and plain MX inputs → BF16 (S1E8M7).
        # For MX-output paths we preserve that precision through the golden
        # so multi-tile accumulation rounds the same way as HW.
        out_is_mx = data_format.is_mx_format()
        hw_dest_dtype = (
            torch.float16
            if (out_is_mx and input_format == DataFormat.Float16)
            else torch.bfloat16
        )
        # Step 1: Quantize each input independently to match what hardware sees
        # after unpacking from L1. Each operand uses its own format.
        operand1 = self._quantize_input(operand1, input_format, data_format)
        operand2 = self._quantize_input(operand2, input_format_B, data_format)

        # Fidelity masking models the source register decomposition, so use
        # the *input* format, not the output format.  Block-float / MX formats
        # are unpacked to Float16_b in the source registers.
        #
        # Consider both operands: if *either* operand is BFP/MX, both unpack
        # to Float16_b in src regs, so the math operates on Float16_b
        # regardless of the other operand's format. Falling back to operand
        # A's format alone (or to data_format when input_format is None,
        # which callers use to signal "already quantized") would mismodel
        # the mixed-format and pre-quantized cases.
        def _src_reg_format(fmt):
            if fmt is None:
                return None
            if (
                fmt in (DataFormat.Bfp4_b, DataFormat.Bfp8_b, DataFormat.Bfp2_b)
                or fmt.is_mx_format()
            ):
                return DataFormat.Float16_b
            return fmt

        src_a_fmt = _src_reg_format(input_format)
        src_b_fmt = _src_reg_format(input_format_B)
        if src_a_fmt == DataFormat.Float16_b or src_b_fmt == DataFormat.Float16_b:
            math_format_for_fidelity = DataFormat.Float16_b
        else:
            math_format_for_fidelity = src_a_fmt or src_b_fmt or data_format

        t1, t2 = operand1, operand2

        # Step 2: Calculate the eltwise result
        if acc_to_dest:
            # The concept of tile should only be used when we have accumulation, otherwise we can use the entire tensor.
            tile_size = tile_shape.total_tile_size()
            num_total_tiles = t1.numel() // tile_size
            num_blocks = num_total_tiles // num_tiles_per_accumulation

            t1_tiles = t1.view(num_total_tiles, tile_size)
            t2_tiles = t2.view(num_total_tiles, tile_size)

            accumulated = []
            for block in range(num_blocks):
                block_acc = None
                for tile in range(num_tiles_per_accumulation):
                    idx = block * num_tiles_per_accumulation + tile
                    tile_result_f32 = self._compute_eltwise(
                        op,
                        t1_tiles[idx],
                        t2_tiles[idx],
                        math_format_for_fidelity,
                        math_fidelity,
                        keep_float32=True,
                    )
                    if block_acc is None:
                        block_acc = tile_result_f32.to(hw_dest_dtype)
                    else:
                        # Add in better precision and then convert to lower precision.
                        block_acc = (block_acc.to(torch.float32) + tile_result_f32).to(
                            hw_dest_dtype
                        )
                accumulated.append(block_acc)

            result = torch.cat(accumulated)
        else:
            result = self._compute_eltwise(
                op,
                t1,
                t2,
                math_format_for_fidelity,
                math_fidelity,
            )

        # Quantize output to match what hardware packs back into L1.
        if data_format == DataFormat.Bfp2_b:
            result = _bfp2b_to_float16b(result.to(torch.bfloat16))
        elif data_format == DataFormat.Bfp4_b:
            result = _bfp4b_to_float16b(result.to(torch.bfloat16))
        elif data_format == DataFormat.Bfp8_b:
            result = _bfp8b_to_float16b(result.to(torch.bfloat16))
        elif data_format.is_mx_format():
            # MX output conversion is performed by the packer gasket. Avoid forcing
            # an extra bfloat16 cast before MX quantization; quantize from the current
            # result dtype so the golden follows the active pack-source path more
            # closely.
            result = quantize_mx_tensor_chunked(result, data_format)
        else:
            if data_format.is_integer():
                torch_format = format_dict[data_format]
                result = saturate_integer(result, data_format, torch_format)
            else:
                result = to_tensor(result, data_format)

        # Final FTZ pass: hardware always flushes subnormals to zero. Do this
        # after all quantization so it covers every output format (including
        # FP, where it's the only FTZ — the BFP helpers no longer FTZ
        # internally, see bfp_format_utils._finalize_bfp_quantized).
        return _apply_ftz(result, data_format)

    # Operation methods
    @staticmethod
    def _wide_dtype(t):
        """Pick a lossless wide type: int64 for integer tensors, float32 otherwise."""
        if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            return torch.int64
        return torch.float32

    def _add(self, t1, t2):
        wide = self._wide_dtype(t1)
        return (t1.to(wide) + t2.to(wide)).to(t1.dtype)

    def _sub(self, t1, t2):
        wide = self._wide_dtype(t1)
        return (t1.to(wide) - t2.to(wide)).to(t1.dtype)

    def _mul(self, t1, t2):
        wide = self._wide_dtype(t1)
        return (t1.to(wide) * t2.to(wide)).to(t1.dtype)

    def _div(self, t1, t2):
        # Compute in float32 to match the SFPU divide path (reciprocal +
        # Newton-Raphson refinement in fp32; the bf16 dest case rounds back
        # via RNE (Round to Nearest) on store, modeled by the final `.to(t1.dtype)` cast).
        # IEEE 754 division naturally produces:
        #   0/0 -> NaN, x/0 -> ±inf, x/x -> 1.0
        # which matches the special-case branches in the SFPU helper.
        return (t1.to(torch.float32) / t2.to(torch.float32)).to(t1.dtype)

    def _gt_int(self, t1, t2):
        return (t1 > t2).to(torch.int32)

    def _lt_int(self, t1, t2):
        return (t1 < t2).to(torch.int32)

    def _le_int(self, t1, t2):
        return (t1 <= t2).to(torch.int32)

    def _ge_int(self, t1, t2):
        return (t1 >= t2).to(torch.int32)


@register_golden
class BinarySFPUGolden(EltwiseBinaryGolden):
    def __init__(self):
        super().__init__()
        self.ops.update(
            {
                MathOperation.SfpuElwadd: self._add,
                MathOperation.SfpuElwsub: self._sub,
                MathOperation.SfpuElwmul: self._mul,
                MathOperation.SfpuElwdiv: self._div,
                MathOperation.SfpuElwmulInt: self._mul,
                MathOperation.SfpuGtInt: self._gt_int,
                MathOperation.SfpuLtInt: self._lt_int,
                MathOperation.SfpuLeInt: self._le_int,
                MathOperation.SfpuGeInt: self._ge_int,
                MathOperation.SfpuXlogy: self._xlogy,
                MathOperation.SfpuLogaddexp: self._logaddexp,
                MathOperation.SfpuLogaddexp2: self._logaddexp2,
                MathOperation.SfpuElwrsub: self._rsub,
                MathOperation.SfpuElwpow: self._pow,
                MathOperation.SfpuElwRightShift: self._right_shift,
                MathOperation.SfpuElwLeftShift: self._left_shift,
                MathOperation.SfpuElwLogicalRightShift: self._logical_right_shift,
                MathOperation.SfpuAddTopRow: self._add_top_row,
                MathOperation.SfpuElwLt: self._lt,
                MathOperation.SfpuElwGt: self._gt,
                MathOperation.SfpuElwLe: self._le,
                MathOperation.SfpuElwGe: self._ge,
                MathOperation.SfpuElwEq: self._eq,
                MathOperation.SfpuElwNe: self._ne,
                MathOperation.SfpuBinaryMax: self._max,
                MathOperation.SfpuBinaryMin: self._min,
                MathOperation.SfpuBinaryFmod: self._fmod,
                MathOperation.SfpuBinaryRemainder: self._remainder,
                MathOperation.SfpuBitwiseAnd: self._bitwise_and,
                MathOperation.SfpuBitwiseOr: self._bitwise_or,
                MathOperation.SfpuBitwiseXor: self._bitwise_xor,
                MathOperation.SfpuDivInt32: self._div_int32,
                MathOperation.SfpuDivInt32Floor: self._div_int32_floor,
                MathOperation.SfpuGcd: self._gcd,
                MathOperation.SfpuLcm: self._lcm,
                MathOperation.SfpuRsubInt32: self._rsub_int32,
                MathOperation.SfpuMask: self._mask,
                MathOperation.SfpuAtan2: self._atan2,
                MathOperation.SfpuMulInt32: self._mul_int32,
                MathOperation.SfpuIsclose: self._isclose,
                MathOperation.SfpuLogsigmoid: self._logsigmoid,
                # Integer / format-typed binary SFPU ops.
                MathOperation.SfpuEqInt: self._eq_int,
                MathOperation.SfpuNeInt: self._ne_int,
                MathOperation.SfpuMaxInt32: self._max,
                MathOperation.SfpuMinInt32: self._min,
                MathOperation.SfpuMaxUint32: self._max,
                MathOperation.SfpuMinUint32: self._min,
                MathOperation.SfpuRemainderInt32: self._remainder_int,
                MathOperation.SfpuRemainderUint32: self._remainder_int,
                MathOperation.SfpuFmodInt32: self._fmod_int,
            }
        )

    def __call__(
        self,
        operation: MathOperation,
        tensor,
        src1_idx: int,
        src2_idx: int,
        dst_idx: int,
        num_iterations: int,
        dimensions: tuple[int, int],
        data_format: DataFormat,
        skip_tilize: bool = False,
        input_format: DataFormat = None,
        dest_acc: DestAccumulation = None,
        output_format: DataFormat = None,
        collect_generated_nan: bool = False,
    ):
        """*dest_acc* and *output_format* enable the Dest-width and pack-path modelling.

        Both default to None, which reproduces the pre-cat-B behaviour: the golden computes in
        *data_format* and models neither the store into Dest nor the pack out of it. Sound only
        while every operand is finite, since both steps are sub-ULP on a finite value and
        decisive on a non-finite one.

        Supply both to get what the hardware does (modelled inline in __call__ below, the same two
        steps UnarySFPUGolden applies): the SFPU evaluates
        in fp32 and stores to a Dest whose width *dest_acc* selects, and the packer substitutes a
        signed infinity for a NaN a 16-bit Dest cannot hold. Same contract UnarySFPUGolden and
        ScalarBinopGolden already model.

        *collect_generated_nan* additionally returns a per-lane mask of the results that were a
        NaN this op *invented*, in the result's layout -- for a caller that has to stop asserting
        the sign of one. See _canonicalise_emitted_nan.
        """
        if operation not in self.ops:
            raise ValueError(f"Unsupported SFPU operation: {operation}")

        if num_iterations < 1:
            raise ValueError(f"num_iterations must be at least 1, got {num_iterations}")

        if (dest_acc is None) != (output_format is None):
            raise ValueError(
                "dest_acc and output_format must be supplied together: the Dest width comes "
                "from dest_acc and whether a NaN survives the pack depends on the output "
                "format, so modelling one without the other gives a golden that is wrong in a "
                "different way than the one it replaces"
            )

        # Quantize MX inputs through pack/unpack round-trip so the golden
        # operates on the same values hardware sees after unpack.
        if input_format is not None and input_format.is_mx_format():
            tensor = quantize_mx_tensor_chunked(tensor, input_format)

        total_elements = dimensions[0] * dimensions[1]
        elements_per_tile = ELEMENTS_PER_TILE
        elements_per_row = 32

        num_tiles = total_elements // elements_per_tile

        src1_start = src1_idx * elements_per_tile
        src2_start = src2_idx * elements_per_tile
        dst_start = dst_idx * elements_per_tile

        if operation == MathOperation.SfpuAddTopRow:
            if collect_generated_nan:
                raise ValueError(
                    "SfpuAddTopRow returns before the Dest modelling that produces the "
                    "generated-NaN mask, so it cannot report one"
                )
            return self._add_top_row(
                tensor.flatten(),
                src1_idx,
                src2_idx,
                dst_idx,
                data_format,
            )

        if not skip_tilize and data_format not in (
            DataFormat.Bfp8_b,
            DataFormat.Bfp4_b,
            DataFormat.Bfp2_b,
        ):
            result = tilize_block(tensor.flatten(), dimensions, data_format).flatten()
        else:
            result = tensor.flatten().clone()

        for name, idx in [
            ("src1_idx", src1_idx),
            ("src2_idx", src2_idx),
            ("dst_idx", dst_idx),
        ]:
            if not 0 <= idx < num_tiles:
                raise ValueError(
                    f"{name} {idx} is out of bounds. Tensor has {num_tiles} tiles."
                )

        elements_to_process = num_iterations * elements_per_row

        for name, start in [
            ("src1_idx", src1_start),
            ("src2_idx", src2_start),
            ("dst_idx", dst_start),
        ]:
            if start + elements_to_process > total_elements:
                raise ValueError(
                    f"Processing {num_iterations} iterations from {name} "
                    f"would exceed tensor bounds (trying to access element {start + elements_to_process}, "
                    f"but tensor has only {total_elements} elements)"
                )

        # Dest modelling applies to the float axis only. On an integer format there is no Dest
        # narrowing to model and no NaN to substitute, and routing int32 through fp32 would cost
        # exactness above 2**24 -- so the integer ops keep the original path outright.
        model_dest = dest_acc is not None and not data_format.is_integer()
        dst_format = (
            self._dest_format(data_format, output_format, dest_acc)
            if model_dest
            else None
        )

        if model_dest and dest_acc == DestAccumulation.No and data_format.is_32_bit():
            # A 32-bit operand landing in a 16-bit Dest drops its low mantissa bits on the way
            # in, before the op ever sees it. Same helper UnarySFPUGolden.__call__ uses, so the
            # two cannot drift on the width.
            result = truncate_to_dest_width(result, dst_format).clone()

        # Same layout as `result`, so it survives the untilize below unchanged.
        generated_nan = torch.zeros(result.numel(), dtype=torch.bool)

        for iteration in range(num_iterations):
            row_offset = iteration * elements_per_row

            src1_row_start = src1_start + row_offset
            src2_row_start = src2_start + row_offset
            dst_row_start = dst_start + row_offset

            src1_row = result[src1_row_start : src1_row_start + elements_per_row]
            src2_row = result[src2_row_start : src2_row_start + elements_per_row]

            if model_dest:
                # Hand the ops fp32 operands. The op methods cast their result back to
                # `t1.dtype`, so a bf16 operand would put torch's canonicalising bf16 cast
                # *inside* the op, before the Dest cast below could preserve anything. Widening
                # first is also what the hardware does -- the SFPU evaluates in fp32 and narrows
                # only on the store to Dest -- and is lossless for a bf16 operand.
                src1_row = src1_row.to(torch.float32)
                src2_row = src2_row.to(torch.float32)

            row_values = [
                self.ops[operation](src1_row[i], src2_row[i])
                for i in range(elements_per_row)
            ]

            if model_dest:
                result_row = torch.tensor(
                    [float(v) for v in row_values], dtype=torch.float32
                )
                result_row, generated_row = self._canonicalise_emitted_nan(
                    operation, result_row
                )
                generated_nan[dst_row_start : dst_row_start + elements_per_row] = (
                    generated_row
                )
                # Two casts, both NaN-sign preserving, for the reason UnarySFPUGolden records:
                # the first is the Dest write's own rounding, the second the store into
                # `result`, whose dtype follows data_format and is not always the Dest dtype.
                # Plain assignment for the second would redo it with torch's canonicalising
                # cast and silently undo the first.
                result_row = cast_to_dest_dtype(
                    result_row, format_dict[dst_format]
                ).float()
                result[dst_row_start : dst_row_start + elements_per_row] = (
                    cast_to_dest_dtype(result_row, result.dtype)
                )
            else:
                result[dst_row_start : dst_row_start + elements_per_row] = torch.tensor(
                    row_values, dtype=format_dict[data_format]
                )

        if not skip_tilize and data_format not in (
            DataFormat.Bfp8_b,
            DataFormat.Bfp4_b,
            DataFormat.Bfp2_b,
        ):
            result = untilize_block(result, data_format, dimensions)
            # The same permutation, so the mask keeps pointing at the lanes it was recorded for.
            # 0.0 and 1.0 are exact in every format this branch runs for, so untilize_block's
            # format cast cannot lose a lane.
            generated_nan = untilize_block(
                generated_nan.to(torch.float32), data_format, dimensions
            ).flatten()

        if model_dest and not nan_survives_to_l1(data_format, output_format, dest_acc):
            # The packer cannot write a NaN through this pipeline, so it substitutes an infinity
            # of the NaN's own sign (SFPSTORE: "NaN is also converted to infinity"). Asked of
            # sfpu_domains rather than restated here, so this golden and the gate that decides
            # where the probe is sent cannot disagree about which cells narrow.
            result = convert_nan_to_inf(result)

        if collect_generated_nan:
            return result, generated_nan.flatten().bool()

        return result

    # The ops whose NaN result is a *selected operand* rather than a computed one -- an exclusion
    # list, because they are the minority and because an allowlist of the arithmetic ops silently
    # drops the composition ops (div, fmod, remainder, xlogy, pow, atan2), whose NaN is every bit
    # as computed as add's.
    #
    # binary_max_min is a bare SFPSWAP(VEC_MIN_MAX): it picks one of its two inputs, so a NaN it
    # returns is the datum it was handed, sign included, and asserting that sign is sound on both
    # arches. Everything else builds its result through the datapath, and `SFPMAD.md` scopes its
    # NaN-sign wording to "if a NaN is emitted" without distinguishing one it computed from one
    # that arrived on an input -- so on Wormhole any NaN they emit has a sign that "might or might
    # not be set", and on Blackhole it is the canonical 0x7fc00000. Either way it is not the
    # operand's. The six comparisons never return a NaN at all (they store 0.0 or 1.0), so which
    # side of this split they fall on cannot matter.
    _NAN_SIGN_SELECTED_OPS = frozenset(
        {
            MathOperation.SfpuBinaryMax,
            MathOperation.SfpuBinaryMin,
        }
    )

    @classmethod
    def _canonicalise_emitted_nan(cls, operation, result_row):
        """Clear the sign of a NaN the datapath computed; keep the sign of one SFPSWAP selected.

        IEEE 754 leaves the sign of an invalid-operation default unspecified, and the ISA declines
        to promise the operand's sign even for a NaN that merely passed through: `SFPMAD.md` says
        only "if a NaN is emitted", then that Blackhole gives the canonical 0x7fc00000 and Wormhole
        "might or might not" set the sign bit. So for the arithmetic ops the golden must not export
        a sign at all -- not the host libm's invented one, which is what made xlogy(0,0) and
        div(0,0) disagree for no reason either kernel owns, and not the operand's either.

        abs() clears the sign bit without disturbing the payload, as UnarySFPUGolden does at the
        same point. It only becomes observable once the pack path substitutes a *signed* infinity
        for the NaN, where the assertion is sound on Blackhole and gated off on Wormhole.

        Returns the per-lane mask as well as the row, because a caller gating that assertion needs
        to know *which lanes*: this is the last point where a NaN is still legible, the
        substitution downstream leaving none to re-derive it from. Lanes holding a genuine
        infinity are never in it -- `0 - (-inf)` is `+inf` by IEEE and stays asserted.
        """
        if operation in cls._NAN_SIGN_SELECTED_OPS:
            return result_row, torch.zeros_like(result_row, dtype=torch.bool)
        emitted = torch.isnan(result_row)
        return torch.where(emitted, result_row.abs(), result_row), emitted

    @staticmethod
    def _dest_format(
        data_format: DataFormat,
        output_format: DataFormat,
        dest_acc: DestAccumulation,
    ) -> DataFormat:
        """The format Dest holds, which is what the SFPU's precision actually follows.

        Same derivation as UnarySFPUGolden.__call__ and the one nan_survives_to_l1() applies
        internally; test_sfpu_domains pins the three to each other so a change to any one fails
        rather than drifting.
        """
        if dest_acc == DestAccumulation.Yes:
            return DataFormat.Float32
        if DataFormat.Float16 in (data_format, output_format):
            return DataFormat.Float16
        return DataFormat.Float16_b

    # Operation methods are covered by Eltwise Binary Golden
    def _xlogy(self, x, y):
        # xlogy(x, y) = x * log(y). The kernel returns NaN for y < 0 (and for
        # y == NaN); y == 0 yields x * -inf. Non-finite edge cases across
        # formats/dest_acc are not consistently modelled, so xlogy is exercised
        # with strictly-positive stimuli (default [0.1, 1.1]) where the result
        # is always finite. Computed in fp32 to mirror the SFPU log path.
        xf = (
            x.to(torch.float32)
            if isinstance(x, torch.Tensor)
            else torch.tensor(float(x))
        )
        yf = (
            y.to(torch.float32)
            if isinstance(y, torch.Tensor)
            else torch.tensor(float(y))
        )
        res = xf * torch.log(yf)
        return res.to(x.dtype) if isinstance(x, torch.Tensor) else res.item()

    def _logaddexp(self, t1, t2):
        # logaddexp(a, b) = log(exp(a) + exp(b)), finite for any finite pair. Computed
        # in fp32 to mirror the SFPU kernel's fused max(a,b) + log1p(exp(-|a-b|)) form,
        # which never overflows an intermediate.
        wide = self._wide_dtype(t1)
        return torch.logaddexp(t1.to(wide), t2.to(wide)).to(t1.dtype)

    def _logaddexp2(self, t1, t2):
        # logaddexp2(a, b) = log2(2**a + 2**b), finite for any finite pair. Computed
        # in fp32 to mirror the SFPU kernel's fused max(a,b) + log2(1 + 2**-|a-b|)
        # form, which never overflows an intermediate.
        wide = self._wide_dtype(t1)
        return torch.logaddexp2(t1.to(wide), t2.to(wide)).to(t1.dtype)

    def _rsub(self, t1, t2):
        # rsub(a, b) = b - a. The kernel computes in1 - in0, i.e. src2 - src1.
        wide = self._wide_dtype(t1)
        return (t2.to(wide) - t1.to(wide)).to(t1.dtype)

    def _pow(self, t1, t2):
        # pow(a, b) = a ** b, computed in fp32 to mirror the SFPU exp(b*log(a))
        # path. Default stimuli are positive ([0.1, 1.1]), so no NaN handling is
        # required for the base here.
        return (t1.to(torch.float32) ** t2.to(torch.float32)).to(t1.dtype)

    def _right_shift(self, t1, t2):
        # The kernel defines shift amounts outside [0, 31] as producing 0 (see
        # ckernel_sfpu_shift.h). torch.bitwise_right_shift is an *arithmetic* shift,
        # so for a negative value an out-of-range shift would sign-extend to -1 rather
        # than 0; guard here so the golden matches the hardware contract.
        if int(t2) < 0 or int(t2) >= 32:
            return 0
        return torch.bitwise_right_shift(t1, t2).item()

    def _left_shift(self, t1, t2):
        # Shift amounts outside [0, 31] produce 0 to match the kernel contract.
        if int(t2) < 0 or int(t2) >= 32:
            return 0
        return torch.bitwise_left_shift(t1, t2).item()

    def _logical_right_shift(self, t1, t2):
        # Shift amounts outside [0, 31] produce 0 to match the kernel contract.
        if int(t2) < 0 or int(t2) >= 32:
            return 0
        # Perform logical right shift by treating t1 as unsigned 32-bit
        t1_uint = t1.to(torch.int64) & 0xFFFFFFFF
        result = (t1_uint >> t2).to(torch.int32)
        return result

    # **The comparison family splits, and the kernels say where.** Both halves route through
    # SFPSWAP, which the ISA specifies with SignMagIsSmaller() and its total order
    # (-NaN < -Inf < ... < +Inf < +NaN) -- so the ISA page alone predicts the total order for all
    # eight of the entries below. For six of them that prediction is wrong, because the kernel
    # wraps the swap in an explicit NaN rejection:
    #
    #   lt/gt  calculate_binary_comp_fp32_strict_ordered -- pre-stores 0, then guards the store
    #          with SFPIADD(inf, |a|+|b|, CC_GTE0), commented "rejects NaN". A NaN operand makes
    #          |a|+|b| a NaN, the predicate fails, and the pre-stored 0 stands.
    #   le/ge  calculate_binary_comp_fp32_weak_ordered -- pre-stores 1, rejects if false, then
    #          stores 0 under "if abs(a) + abs(b) > inf; a or b is NaN".
    #   eq/ne  calculate_binary_comp_fp32_equal -- same "rejects NaN" guard; the default result
    #          (0 for eq, 1 for ne) stands.
    #
    #   max/min  binary_max_min is a bare TTI_SFPSWAP(VEC_MIN_MAX) with **no NaN guard at all**,
    #            so for these two the total order does reach the result.
    #
    # So the six comparisons implement IEEE's unordered semantics deliberately, and max/min do
    # not. Measured on a Wormhole n150: with all eight modelled on the total order the six
    # comparisons failed 4 cells each and max/min passed everywhere. The unary comparisons keep
    # the total order -- different kernels, no such guard.
    def _lt(self, t1, t2):
        return float(t1 < t2)

    def _gt(self, t1, t2):
        return float(t1 > t2)

    def _le(self, t1, t2):
        return float(t1 <= t2)

    def _ge(self, t1, t2):
        return float(t1 >= t2)

    def _eq(self, t1, t2):
        return float(t1 == t2)

    def _ne(self, t1, t2):
        return float(t1 != t2)

    @staticmethod
    def _is_float(t):
        """Does this operand carry a float dtype (as opposed to Int32/UInt32)?

        max/min serve both axes -- SfpuBinaryMax/Min on float and SfpuMaxInt32/MinInt32/
        MaxUint32/MinUint32 on integers -- and only the float one has a total order to follow.
        """
        return torch.is_floating_point(
            t if isinstance(t, torch.Tensor) else torch.tensor(t)
        )

    def _max(self, t1, t2):
        # torch.maximum agrees with the total order for a *positive* NaN by coincidence and
        # disagrees for a negative one, where -NaN is the order's smallest value and torch
        # propagates it -- so a one-sided NaN probe would certify torch.maximum as correct.
        if self._is_float(t1):
            return sfpu_max_elementwise(t1, t2).to(t1.dtype)
        wide = self._wide_dtype(t1)
        return torch.maximum(t1.to(wide), t2.to(wide)).to(t1.dtype)

    def _min(self, t1, t2):
        # torch.minimum propagates a NaN; the total order makes +NaN the largest value, so a
        # min against it returns the *other* operand. This one diverges on a positive NaN too.
        if self._is_float(t1):
            return sfpu_min_elementwise(t1, t2).to(t1.dtype)
        wide = self._wide_dtype(t1)
        return torch.minimum(t1.to(wide), t2.to(wide)).to(t1.dtype)

    def _fmod(self, t1, t2):
        # fmod(a, b) = a - trunc(a/b) * b (result takes the sign of a). Computed in
        # fp32 to mirror the SFPU reciprocal path; the final cast models the bf16 store.
        return torch.fmod(t1.to(torch.float32), t2.to(torch.float32)).to(t1.dtype)

    def _remainder(self, t1, t2):
        # remainder(a, b) = a - floor(a/b) * b (result takes the sign of b), matching
        # torch.remainder and the SFPU floor-based kernel.
        return torch.remainder(t1.to(torch.float32), t2.to(torch.float32)).to(t1.dtype)

    def _bitwise_and(self, t1, t2):
        return torch.bitwise_and(t1.to(torch.int32), t2.to(torch.int32)).to(torch.int32)

    def _bitwise_or(self, t1, t2):
        return torch.bitwise_or(t1.to(torch.int32), t2.to(torch.int32)).to(torch.int32)

    def _bitwise_xor(self, t1, t2):
        return torch.bitwise_xor(t1.to(torch.int32), t2.to(torch.int32)).to(torch.int32)

    def _div_int32(self, t1, t2):
        # int32 truncating division (rounds toward zero), matching calculate_div_int32_trunc.
        return torch.div(
            t1.to(torch.int64), t2.to(torch.int64), rounding_mode="trunc"
        ).to(torch.int32)

    def _div_int32_floor(self, t1, t2):
        # int32 floor division (rounds toward -inf), matching calculate_div_int32_floor.
        return torch.div(
            t1.to(torch.int64), t2.to(torch.int64), rounding_mode="floor"
        ).to(torch.int32)

    def _gcd(self, t1, t2):
        return torch.gcd(t1.to(torch.int32), t2.to(torch.int32)).to(torch.int32)

    def _lcm(self, t1, t2):
        # lcm(a, b) = |a / gcd(a, b) * b|. The kernel takes abs() of both operands
        # and assumes |a|, |b| < 2^15; goldens mirror torch.lcm (non-negative).
        return torch.lcm(t1.to(torch.int32), t2.to(torch.int32)).to(torch.int32)

    def _rsub_int32(self, t1, t2):
        # rsub_int32 computes out = in1 - in0 = t2 - t1. Exact integer subtraction
        # (widen to int64 so the intermediate can't overflow before the int32 cast).
        return (t2.to(torch.int64) - t1.to(torch.int64)).to(torch.int32)

    def _mask(self, t1, t2):
        # mask: data (t1) is zeroed wherever the mask (t2) is zero, else passed
        # through. Matches calculate_mask (v_if(is_fp16_zero(mask)) data = 0).
        return t1 if float(t2) != 0.0 else t1 * 0

    def _atan2(self, t1, t2):
        # calculate_sfpu_atan2 computes atan2(in0, in1) = atan2(y, x) with y=t1
        # (src1) and x=t2 (src2). Evaluated in fp32 to mirror the SFPU minimax path;
        # the kernel is an approximation, so the match relies on the PCC tolerance.
        return torch.atan2(t1.to(torch.float32), t2.to(torch.float32))

    def _eq_int(self, t1, t2):
        # Integer equality, exact 0/1 (calculate_binary_eq_int over Int32 dest bits).
        return int(int(t1) == int(t2))

    def _ne_int(self, t1, t2):
        return int(int(t1) != int(t2))

    def _remainder_int(self, t1, t2):
        # Integer remainder. Stimuli are non-negative with divisor >= 1, so the result is
        # convention-agnostic (trunc/floor/unsigned all agree) and equals Python's a % b.
        return int(int(t1) % int(t2))

    def _fmod_int(self, t1, t2):
        # int32 fmod (sign follows dividend). Non-negative stimuli make it equal to a % b,
        # matching the internal unsigned-remainder kernel.
        return int(int(t1) % int(t2))

    def _mul_int32(self, t1, t2):
        # int32 multiply, low 32 bits. The kernel stores two's-complement bits via
        # plain INT32, so only non-negative products round-trip through the harness'
        # sign-magnitude packer; the test keeps operands positive with product < 2^31.
        # Widen to int64 for the multiply so the intermediate can't overflow.
        return (t1.to(torch.int64) * t2.to(torch.int64)).to(torch.int32)

    def _isclose(self, t1, t2):
        # isclose(a, b) = |a - b| <= atol + rtol * |b|, returned as 1.0 / 0.0. Uses
        # torch's default tolerances (rtol=1e-5, atol=1e-8), matching the fp32 bit
        # patterns hard-coded in the ISCLOSE dispatch. equal_nan=False (torch default).
        # Evaluated in fp32; the test's large-margin stimuli keep the result robust to
        # tolerance precision.
        close = torch.isclose(
            t1.to(torch.float32),
            t2.to(torch.float32),
            rtol=1e-5,
            atol=1e-8,
            equal_nan=False,
        )
        return 1.0 if bool(close) else 0.0

    def _logsigmoid(self, t1, t2):
        # logsigmoid(x) = log(sigmoid(x)) = -softplus(-x), with x = t1. The kernel takes
        # exp(-x) as its second operand (t2), which the test bakes into the paired
        # stimuli; the golden only needs x. It is a piecewise (poly + exp) approximation,
        # so it is matched under the PCC tolerance. Evaluated in fp32.
        return torch.nn.functional.logsigmoid(t1.to(torch.float32))

    def _add_top_row(
        self,
        tensor,
        src1_idx,
        src2_idx,
        dst_idx,
        data_format=DataFormat.Float32,
    ):
        """
        Add top row operation for tile pairs in untilized format.

        For UInt32, masks results to 32 bits to match hardware's unsigned wraparound.
        """
        src1_idx_start = src1_idx * ELEMENTS_PER_TILE
        src2_idx_start = src2_idx * ELEMENTS_PER_TILE
        dst_idx_start = dst_idx * ELEMENTS_PER_TILE

        result = tensor.clone()

        # Untilized format: row-wise layout
        ROWS_0_1_OFFSET = 0  # Rows 0-1 start at element 0
        ROWS_8_9_OFFSET = 256  # Rows 8-9 start at element 256
        # Two consecutive rows = 2 rows × 32 columns = 64 elements
        TWO_ROWS_ELEMENTS = 64

        # Add rows 0-1 (elements 0-63)
        rows_0_1_dst_start = dst_idx_start + ROWS_0_1_OFFSET
        rows_0_1_dst_end = rows_0_1_dst_start + TWO_ROWS_ELEMENTS
        rows_0_1_src1_start = src1_idx_start + ROWS_0_1_OFFSET
        rows_0_1_src1_end = rows_0_1_src1_start + TWO_ROWS_ELEMENTS
        rows_0_1_src2_start = src2_idx_start + ROWS_0_1_OFFSET
        rows_0_1_src2_end = rows_0_1_src2_start + TWO_ROWS_ELEMENTS

        added_0_1 = (
            tensor[rows_0_1_src1_start:rows_0_1_src1_end]
            + tensor[rows_0_1_src2_start:rows_0_1_src2_end]
        )
        if data_format == DataFormat.UInt32:
            added_0_1 = added_0_1 & 0xFFFFFFFF
        result[rows_0_1_dst_start:rows_0_1_dst_end] = added_0_1

        # Add rows 8-9 (elements 256-319)
        rows_8_9_dst_start = dst_idx_start + ROWS_8_9_OFFSET
        rows_8_9_dst_end = rows_8_9_dst_start + TWO_ROWS_ELEMENTS
        rows_8_9_src1_start = src1_idx_start + ROWS_8_9_OFFSET
        rows_8_9_src1_end = rows_8_9_src1_start + TWO_ROWS_ELEMENTS
        rows_8_9_src2_start = src2_idx_start + ROWS_8_9_OFFSET
        rows_8_9_src2_end = rows_8_9_src2_start + TWO_ROWS_ELEMENTS

        added_8_9 = (
            tensor[rows_8_9_src1_start:rows_8_9_src1_end]
            + tensor[rows_8_9_src2_start:rows_8_9_src2_end]
        )
        if data_format == DataFormat.UInt32:
            added_8_9 = added_8_9 & 0xFFFFFFFF
        result[rows_8_9_dst_start:rows_8_9_dst_end] = added_8_9

        return result


@register_golden
class ReduceGolden:
    """Golden for reduce operations (Max/Average/Sum pooling).

    Reduce dimensions:
        Column: f0+f2 (left), f1+f3 (right) → row 0
        Row:    f0+f1 (upper), f2+f3 (lower) → col 0
        Scalar: all elements → single value at [0]
    """

    def __init__(self):
        self.dim_handlers = {
            ReduceDimension.Column: self._reduce_column,
            ReduceDimension.Row: self._reduce_row,
            ReduceDimension.Scalar: self._reduce_scalar,
        }

    def _quantize_reduce_input(self, operand, fmt, data_format):
        """Quantize input to match what hardware sees after unpack (same as EltwiseBinaryGolden)."""
        if fmt is None:
            return to_tensor(operand, data_format)
        if fmt == DataFormat.Bfp2_b:
            return _bfp2b_to_float16b(operand)
        if fmt == DataFormat.Bfp4_b:
            return _bfp4b_to_float16b(operand)
        if fmt == DataFormat.Bfp8_b:
            return _bfp8b_to_float16b(operand)
        if fmt.is_mx_format():
            return quantize_mx_tensor_chunked(operand, fmt)
        return to_tensor(operand, data_format)

    def __call__(
        self,
        operand,
        reduce_dim,
        pool_type,
        data_format,
        tile_cnt=1,
        reduce_to_one=False,
        tile_shape=None,
        input_format=None,
    ):
        if tile_shape is None:
            tile_shape = construct_tile_shape()

        if reduce_dim not in self.dim_handlers:
            raise ValueError(f"Unsupported reduce dimension: {reduce_dim}")

        # Same convention as EltwiseBinaryGolden: plain cast uses input format when set,
        # else output format (callers that omit input_format typically use matching I/O).
        fmt_for_plain = input_format if input_format is not None else data_format
        operand = self._quantize_reduce_input(operand, input_format, fmt_for_plain)

        if reduce_to_one:
            # Accumulate all tiles into a single result
            result = self._reduce_all_tiles(
                operand, reduce_dim, pool_type, data_format, tile_cnt, tile_shape
            )
        else:
            # Process each tile independently; quantize output like eltwise binary
            result = torch.cat(
                [
                    self._quantize_reduce_output(
                        self._process_tile(
                            operand,
                            reduce_dim,
                            pool_type,
                            data_format,
                            tile,
                            tile_shape,
                        ),
                        data_format,
                    )
                    for tile in range(tile_cnt)
                ]
            )
        # MX-quantize the golden to match what HW physically packs into L1.
        # Low-bit outputs (e.g. MxInt2: 2 bits per element → only {-1, 0, -0 (not recommended), +1}
        # scaled by the block's shared E8M0 exponent) snap aggressively to
        # the block lattice at pack time; without this the golden carries
        # raw input values that miss the target bins.
        if data_format.is_mx_format():
            result = quantize_mx_tensor_chunked(result, data_format)

        # Final FTZ pass: hardware always flushes subnormals to zero. Same
        # rationale as EltwiseBinaryGolden — covers both BFP and FP outputs
        # now that the BFP helpers no longer FTZ internally.
        return _apply_ftz(result, data_format)

    def _quantize_reduce_output(self, tensor: torch.Tensor, data_format: DataFormat):
        """Quantize output to match what hardware packs into L1 (same as EltwiseBinaryGolden)."""
        if data_format == DataFormat.Bfp2_b:
            return _bfp2b_to_float16b(tensor.to(torch.bfloat16))
        elif data_format == DataFormat.Bfp4_b:
            return _bfp4b_to_float16b(tensor.to(torch.bfloat16))
        elif data_format == DataFormat.Bfp8_b:
            return _bfp8b_to_float16b(tensor.to(torch.bfloat16))
        elif data_format.is_mx_format():
            return quantize_mx_tensor_chunked(tensor.to(torch.bfloat16), data_format)
        elif data_format.is_integer():
            return saturate_integer(tensor, data_format, format_dict[data_format])
        else:
            return to_tensor(tensor, data_format)

    def _reduce_all_tiles(
        self, operand, reduce_dim, pool_type, data_format, tile_cnt, tile_shape
    ):
        """Accumulate reduction across all tiles into a single result."""
        accumulated = None

        for tile_idx in range(tile_cnt):
            tile_result = self._process_tile(
                operand, reduce_dim, pool_type, data_format, tile_idx, tile_shape
            )

            if accumulated is None:
                # First tile - store it in float32 for high precision accumulation
                accumulated = tile_result.to(torch.float32)
            else:
                # Subsequent tiles - pool with previous accumulation in float32
                tile_result_f32 = tile_result.to(torch.float32)
                if pool_type == ReducePool.Max:
                    accumulated = torch.maximum(accumulated, tile_result_f32)
                elif pool_type == ReducePool.Sum:
                    accumulated = torch.add(accumulated, tile_result_f32)
                elif pool_type == ReducePool.Average:
                    # Average reduce operation performs dest += avg(curr_tile) when reducing to populated dest locations.
                    # Result should simply be the accumulation of averages.
                    accumulated = torch.add(accumulated, tile_result_f32)
                else:
                    raise ValueError(f"Unsupported pool type: {pool_type}")

        # Convert back to target data format at the end (same as eltwise output path)
        return self._quantize_reduce_output(accumulated, data_format)

    def _make_tile_result(self, data_format, tile_shape):
        """Create a zero-filled tile result in a dtype wide enough to avoid integer overflow."""
        torch_format = format_dict[data_format]
        dtype = torch.int64 if data_format.is_integer() else torch_format
        return torch.zeros(tile_shape.total_tile_size(), dtype=dtype)

    def _process_tile(
        self, operand, reduce_dim, pool_type, data_format, tile_idx, tile_shape
    ):

        tile_start = tile_idx * tile_shape.total_tile_size()
        tile_data = operand[tile_start : tile_start + tile_shape.total_tile_size()]

        # Extract 4 faces as 16x16 matrices
        faces = tile_data.view(
            tile_shape.total_num_faces(), tile_shape.face_r_dim, tile_shape.face_c_dim
        )

        return self.dim_handlers[reduce_dim](faces, pool_type, data_format, tile_shape)

    def _reduce_column(self, faces, pool_type, data_format, tile_shape):
        result = self._make_tile_result(data_format, tile_shape)

        # For each column of faces, concatenate vertically and pool along rows
        for col_idx in range(tile_shape.num_faces_c_dim):
            # Gather all faces in this column (vertically stacked)
            face_indices = [
                col_idx + row_idx * tile_shape.num_faces_c_dim
                for row_idx in range(tile_shape.num_faces_r_dim)
            ]
            column_faces = torch.cat([faces[i] for i in face_indices], dim=0)

            # Pool along rows (dim=0) to get one value per column
            pooled_values = self._apply_pooling(column_faces, pool_type, dim=0)

            # Place in the first row of this face column (in tilized layout)
            # Face col_idx starts at position col_idx * face_r_dim * face_c_dim
            result_start = col_idx * tile_shape.face_r_dim * tile_shape.face_c_dim
            result[result_start : result_start + tile_shape.face_c_dim] = pooled_values

        return result

    def _reduce_row(self, faces, pool_type, data_format, tile_shape):
        result = self._make_tile_result(data_format, tile_shape)

        # For each row of faces, concatenate horizontally and pool along columns
        for row_idx in range(tile_shape.num_faces_r_dim):
            # Gather all faces in this row (horizontally stacked)
            face_indices = [
                row_idx * tile_shape.num_faces_c_dim + col_idx
                for col_idx in range(tile_shape.num_faces_c_dim)
            ]
            row_faces = torch.cat([faces[i] for i in face_indices], dim=1)

            # Pool along columns (dim=1) to get one value per row
            pooled_values = self._apply_pooling(row_faces, pool_type, dim=1)

            # Place in the first column of this face row (in tilized layout)
            # Face row starts at position row_idx * num_faces_c_dim * face_r_dim * face_c_dim
            # Within each face, we need to place values at the start of each row (stride = face_c_dim)
            face_row_start = (
                row_idx
                * tile_shape.num_faces_c_dim
                * tile_shape.face_r_dim
                * tile_shape.face_c_dim
            )
            for i, val in enumerate(pooled_values):
                result[face_row_start + i * tile_shape.face_c_dim] = val

        return result

    def _reduce_scalar(self, faces, pool_type, data_format, tile_shape):
        result = self._make_tile_result(data_format, tile_shape)
        result[0] = self._apply_pooling(faces.flatten(), pool_type, dim=0)
        return result

    def _apply_pooling(self, tensor, pool_type, dim):
        if pool_type == ReducePool.Max:
            return torch.max(tensor, dim=dim).values
        elif pool_type == ReducePool.Average:
            return torch.mean(tensor.float(), dim=dim)
        elif pool_type == ReducePool.Sum:
            return torch.sum(tensor, dim=dim)
        else:
            raise ValueError(f"Unsupported pool type: {pool_type}")


@register_golden
class ReduceBlockMaxRowGolden:
    # Row-wise block max reduce. Works for both 32x32 (num_faces=4) and 16x32 (num_faces=2,
    # one face-row) tiles: the reduce is per-row across the full block width and the caller
    # passes block dims sized by the actual tile row/col dims. Column width per tile is 32 in
    # both cases (only the row count shrinks for tiny tiles), so the reduce span is ct_dim * 32.
    def __call__(self, operand, ct_dim, data_format, dimensions):
        operand = operand.reshape(dimensions)
        output = torch.zeros(dimensions)
        reduce_width = min(ct_dim * 32, dimensions[1])
        for i in range(dimensions[0]):
            output[i, 0] = torch.max(operand[i, :reduce_width])

        return output


@register_golden
class ReduceGapoolGolden(FidelityMasking):
    """Golden for GAPOOL reduce (Sum/Average pooling) with fidelity masking.

    Hardware computes matmul (D = srcB @ srcA) per face, accumulating across fidelity iterations.

    Reduce dimensions:
        Column: f0+f2 (left), f1+f3 (right) → row 0
        Row:    f0+f1 (upper), f2+f3 (lower) → col 0, (srcA transposed by unpacker)
        Scalar: all faces summed → transpose → pool again → single value at [0]
    """

    MATH_FIDELITY_TO_ITER_COUNT = {
        MathFidelity.LoFi: 0,
        MathFidelity.HiFi2: 1,
        MathFidelity.HiFi3: 2,
        MathFidelity.HiFi4: 3,
    }

    def __call__(
        self,
        operand1,
        operand2,
        data_format,
        reduce_dim,
        math_fidelity=MathFidelity.LoFi,
        tile_cnt=1,
        tile_shape=None,
        input_format=None,
        dest_acc: Optional[DestAccumulation] = None,
    ):
        if tile_shape is None:
            tile_shape = construct_tile_shape()

        # Integer reduce (e.g. Int8 -> Int32) is exact-integer accumulation.
        # Int8 reduce is LoFi-only: there is no mantissa multi-pass, so fidelity
        # masking does not apply, and the gapool matmul is an exact integer sum.
        # Full 32x32 tiles only
        if input_format is not None and input_format.is_integer():
            return self._reduce_integer(
                operand1, operand2, data_format, reduce_dim, tile_cnt, input_format
            )

        # Quantize MX format inputs to match hardware behavior
        if input_format is not None and input_format.is_mx_format():
            operand1 = quantize_mx_tensor_chunked(operand1, input_format)

        fidelity_iter_count = self.MATH_FIDELITY_TO_ITER_COUNT[math_fidelity]

        # On Quasar with implied_math_format, HW dest precision is implied by
        # the SrcA tag: Float16 → FP16A; Float16_b / MX inputs → BF16. For
        # MX-output paths we preserve that precision through the gapool +
        # face-accumulation chain rather than collapsing inputs to the output
        # dtype (which would force fp16 → bf16 before any math).
        # When dest_acc=Yes, HW accumulates in fp32 regardless of input —
        # so the inter-face / inter-fidelity accumulators must follow.
        out_is_mx = data_format.is_mx_format()
        fp32_acc = dest_acc == DestAccumulation.Yes
        if out_is_mx and input_format is not None:
            compute_format = (
                DataFormat.Float16_b if input_format.is_mx_format() else input_format
            )
        else:
            compute_format = data_format

        result = torch.cat(
            [
                self._process_tile(
                    operand1,
                    operand2,
                    compute_format,
                    reduce_dim,
                    fidelity_iter_count,
                    tile,
                    tile_shape,
                    fp32_acc=fp32_acc,
                )
                for tile in range(tile_cnt)
            ]
        )

        # MX output conversion is performed by the packer gasket. Avoid forcing
        # an extra bfloat16 cast before MX quantization; quantize from the current
        # result dtype so the golden follows the active pack-source path more
        # closely.
        if out_is_mx:
            result = quantize_mx_tensor_chunked(result, data_format)

        return result

    def _process_tile(
        self,
        operand1,
        operand2,
        data_format,
        reduce_dim,
        fidelity_iter_count,
        tile_idx,
        tile_shape,
        fp32_acc=False,
    ):
        # Extract srcA tile and srcB face0 (only f0 unpacked for srcB)
        tile_size = tile_shape.total_tile_size()
        face_r_dim = tile_shape.face_r_dim
        face_c_dim = tile_shape.face_c_dim
        num_faces = tile_shape.total_num_faces()
        face_size = face_r_dim * face_c_dim

        tile_start = tile_idx * tile_size
        src_a = to_tensor(operand1[tile_start : tile_start + tile_size], data_format)
        src_b = to_tensor(operand2[:face_size], data_format)

        # HW performs 16x16 matmul in FPU; pad rows beyond face_r_dim with zeros
        # so the golden models the padded register state.
        row_pad = face_c_dim - face_r_dim
        a_padded = src_a.view(num_faces, face_r_dim, face_c_dim)
        b_padded = src_b.view(1, face_r_dim, face_c_dim)
        if row_pad > 0:
            a_padded = torch.nn.functional.pad(a_padded, (0, 0, 0, row_pad))
            b_padded = torch.nn.functional.pad(b_padded, (0, 0, 0, row_pad))

        # Row reduce: transpose within each (padded) face of SrcA (models unpacker behavior)
        if reduce_dim == ReduceDimension.Row:
            a_padded = a_padded.transpose(1, 2).contiguous()

        src_a_flat = a_padded.flatten()
        src_b_flat = b_padded.flatten()

        # Compute gapool for each face across all fidelity iterations
        face_results = self._compute_gapool(
            src_a_flat,
            src_b_flat,
            data_format,
            fidelity_iter_count,
            num_faces,
            face_c_dim,
            fp32_acc=fp32_acc,
        )

        # Combine results based on reduce dimension
        return self._accumulate_gapool_results(
            face_results,
            src_b_flat,
            data_format,
            reduce_dim,
            fidelity_iter_count,
            tile_shape,
            fp32_acc=fp32_acc,
        )

    def _compute_gapool(
        self,
        src_a,
        src_b,
        data_format,
        fidelity_iter_count,
        num_faces,
        dim,
        fp32_acc=False,
    ):
        """Compute D = srcB @ srcA per face (dim x dim), accumulating across fidelity iterations."""
        acc_dtype = torch.float32 if fp32_acc else src_a.dtype
        face_results = torch.zeros(
            num_faces, dim * dim, dtype=acc_dtype, device=src_a.device
        )

        for fidelity_iter in range(fidelity_iter_count + 1):
            a_masked, b_masked = self._apply_fidelity_masking(
                data_format, src_a, src_b, fidelity_iter
            )

            a_faces = a_masked.view(num_faces, dim, dim)
            b_face = b_masked.view(1, dim, dim)
            # When dest_acc=Yes HW dest is fp32; promote operands to fp32 so
            # the per-iter dot product is not rounded to bf16/fp16 before
            # accumulating. Operand precision was already set by fidelity
            # masking; this only affects the matmul's internal rounding.
            if fp32_acc:
                result = torch.matmul(b_face.float(), a_faces.float())
            else:
                result = torch.matmul(b_face, a_faces)

            face_results += result.view(num_faces, -1).to(acc_dtype)

        return face_results

    def _accumulate_gapool_results(
        self,
        face_results,
        src_b,
        data_format,
        reduce_dim,
        fidelity_iter_count,
        tile_shape,
        fp32_acc=False,
    ):
        """Place pooled results in output tile based on reduce dimension."""
        face_r_dim = tile_shape.face_r_dim
        face_c_dim = tile_shape.face_c_dim
        face_size = face_r_dim * face_c_dim
        result = torch.zeros(tile_shape.total_tile_size(), dtype=face_results.dtype)

        if reduce_dim == ReduceDimension.Column:
            # For each face column, sum faces vertically → row 0 of first face in that column
            for col_idx in range(tile_shape.num_faces_c_dim):
                face_indices = [
                    col_idx + row_idx * tile_shape.num_faces_c_dim
                    for row_idx in range(tile_shape.num_faces_r_dim)
                ]
                summed = sum(face_results[i] for i in face_indices)
                result_start = col_idx * face_size
                result[result_start : result_start + face_c_dim] = summed[:face_c_dim]

        elif reduce_dim == ReduceDimension.Row:
            # For each face row, sum faces horizontally → col 0 of first face in that row
            for row_idx in range(tile_shape.num_faces_r_dim):
                face_indices = [
                    row_idx * tile_shape.num_faces_c_dim + col_idx
                    for col_idx in range(tile_shape.num_faces_c_dim)
                ]
                summed = sum(face_results[i] for i in face_indices)
                face_row_start = row_idx * tile_shape.num_faces_c_dim * face_size
                for i in range(face_r_dim):
                    result[face_row_start + i * face_c_dim] = summed[i]

        elif reduce_dim == ReduceDimension.Scalar:
            # Sum all (padded 16x16) faces, transpose, pool again to get single scalar
            all_faces_sum = sum(
                face_results[i] for i in range(tile_shape.total_num_faces())
            )
            all_faces = all_faces_sum.view(face_c_dim, face_c_dim).T.flatten()
            pool_result = self._compute_gapool(
                all_faces,
                src_b,
                data_format,
                fidelity_iter_count,
                1,
                face_c_dim,
                fp32_acc=fp32_acc,
            )
            result[0] = pool_result[0][0]

        return result

    def _reduce_integer(
        self, operand1, operand2, data_format, reduce_dim, tile_cnt, input_format
    ):
        return torch.cat(
            [
                self._process_tile_integer(
                    operand1, operand2, data_format, reduce_dim, tile, input_format
                )
                for tile in range(tile_cnt)
            ]
        )

    def _process_tile_integer(
        self, operand1, operand2, data_format, reduce_dim, tile_idx, input_format
    ):
        tile_start = tile_idx * ELEMENTS_PER_TILE
        src_a = to_tensor(
            operand1[tile_start : tile_start + ELEMENTS_PER_TILE], input_format
        ).to(torch.int64)
        src_b = to_tensor(operand2[:ELEMENTS_PER_FACE], input_format).to(torch.int64)

        # Row reduce: transpose within each face of SrcA (models unpacker behavior)
        if reduce_dim == ReduceDimension.Row:
            src_a = (
                src_a.view(FACES_PER_TILE, FACE_DIM, FACE_DIM).transpose(1, 2).flatten()
            )

        face_results = self._compute_gapool_integer(src_a, src_b)
        return self._accumulate_gapool_results_integer(
            face_results, src_b, data_format, reduce_dim
        )

    def _compute_gapool_integer(self, src_a, src_b, num_faces=FACES_PER_TILE):
        """Exact integer D = srcB @ srcA per face (single LoFi pass)."""
        a_faces = src_a.view(num_faces, FACE_DIM, FACE_DIM)
        b_face = src_b.view(1, FACE_DIM, FACE_DIM)
        return torch.matmul(b_face, a_faces).view(num_faces, -1)

    def _accumulate_gapool_results_integer(
        self, face_results, src_b, data_format, reduce_dim
    ):
        """Place pooled integer results in the output tile"""
        torch_format = format_dict[data_format]
        face_shape = (FACE_DIM, FACE_DIM)
        f0, f1, f2, f3 = face_results
        result = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.int64)

        if reduce_dim == ReduceDimension.Column:
            # Sum left faces (f0+f2) → face0 row 0, right faces (f1+f3) → face1 row 0
            result[:FACE_DIM] = (f0 + f2)[:FACE_DIM]
            result[ELEMENTS_PER_FACE : ELEMENTS_PER_FACE + FACE_DIM] = (f1 + f3)[
                :FACE_DIM
            ]

        elif reduce_dim == ReduceDimension.Row:
            # Sum top faces (f0+f1) → face0 col 0, bottom faces (f2+f3) → face2 col 0
            result[0:ELEMENTS_PER_FACE:FACE_DIM] = (f0 + f1)[:FACE_DIM]
            result[2 * ELEMENTS_PER_FACE : 3 * ELEMENTS_PER_FACE : FACE_DIM] = (
                f2 + f3
            )[:FACE_DIM]

        elif reduce_dim == ReduceDimension.Scalar:
            # Sum all faces, transpose, pool again to get single scalar
            all_faces = (f0 + f1 + f2 + f3).view(face_shape).T.flatten()
            pool_result = self._compute_gapool_integer(all_faces, src_b, num_faces=1)
            result[0] = pool_result[0][0]

        return saturate_integer(result, data_format, torch_format)


@register_golden
class UntilizeGolden:
    def __call__(
        self,
        operand,
        data_format,
        dimensions=[32, 32],
        input_format: Optional[DataFormat] = None,
        tile_dimensions=None,
    ):
        from helpers.tilize_untilize import untilize_block

        operand = quantize_input_to_unpack_format(
            operand, input_format, all_mx_formats=True
        )

        result = untilize_block(
            operand,
            stimuli_format=data_format,
            dimensions=dimensions,
            tile_dimensions=tile_dimensions,
        )
        result = result.flatten()

        return result


@register_golden
class TilizeGolden:
    def __call__(
        self, operand, dimensions, data_format, num_faces=4, tile_dimensions=None
    ):
        from helpers.llk_params import format_dict
        from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
        from helpers.tilize_untilize import tilize_block

        # Validate the number of faces
        if not (1 <= num_faces <= 4):
            raise ValueError(f"`num_faces` must be between 1 and 4, got {num_faces}")

        # Always do full tilization first. When tile_dimensions is provided, tilize_block
        # derives the face layout (face_r_dim / num_faces) from the tile geometry.
        result = tilize_block(
            operand, dimensions, data_format, tile_dimensions=tile_dimensions
        )
        torch_format = format_dict[data_format]

        # Legacy partial-face selection only applies to standard 32x32 tiles; tiny-tile
        # layouts already carry the correct face count from tile_dimensions.
        is_standard_tile = tile_dimensions is None or tuple(tile_dimensions) == (
            DEFAULT_TILE_R_DIM,
            DEFAULT_TILE_C_DIM,
        )
        if is_standard_tile and num_faces < FACES_PER_TILE:
            elements_per_tile_needed = num_faces * ELEMENTS_PER_FACE
            tile_cnt = result.numel() // ELEMENTS_PER_TILE
            result = result.reshape(tile_cnt, ELEMENTS_PER_TILE)[
                :, :elements_per_tile_needed
            ]

        return result.flatten().to(torch_format)


@register_golden
class PackRowsGolden:
    def __call__(
        self,
        operand,
        data_format,
        dimensions=[32, 32],
        num_rows_to_pack=1,
        tile_count=1,
    ):
        row_num_datums = 16

        if not isinstance(operand, torch.Tensor):
            operand = torch.tensor(operand, dtype=format_dict[data_format])

        operand_flat = operand.flatten()

        # Extract first num_rows_to_pack * row_num_datums elements from each tile
        num_elements_per_tile = num_rows_to_pack * row_num_datums

        # Calculate total number of elements we need
        total_elements = tile_count * ELEMENTS_PER_TILE

        operand_flat = operand_flat[:total_elements]

        # Reshape the data: (total_elements,) -> (tile_count, ELEMENTS_PER_TILE)
        tiles_reshaped = operand_flat.view(tile_count, ELEMENTS_PER_TILE)

        # Extract first num_elements_per_tile elements from each tile
        extracted_elements = tiles_reshaped[:, :num_elements_per_tile]

        result = extracted_elements.flatten()

        return result.to(format_dict[data_format])


@register_golden
class TopKGolden:
    """
    Golden generator for TopK operation.

    """

    def __call__(
        self,
        operand,
        data_format,
        K=32,
        sort_direction: TopKSortDirection = TopKSortDirection.Descending,
        input_dimensions=[32, 128],
    ):
        """
        Perform per-row topk on a tensor.

        Args:
            operand: Input tensor (flattened).
            data_format: Data format for the result.
            k: Number of top elements to extract per row (default: 32).
            sort_direction: Direction to sort top-k values (default: Descending).
            input_dimensions: Input dimensions [rows, cols] (default: [32, 128]).
        Constraint:
            In LLK api, we perform topk with k >= 32, so input tensor must always have at least 2 tiles for values and
            of course 2 tiles for indices since we need to reorder them based on the topk results.
            Therefore input_dimensions must contain at least 4 tiles.
        Returns:
            Tensor with topk applied per row. One Tile of values followed by one tile of indices.
        """
        torch_format = format_dict[data_format]

        num_stages = 2  # One stage for values and one stage for indices.

        minimal_number_of_tiles_required = 4

        # Convert to tensor if needed.
        if not isinstance(operand, torch.Tensor):
            operand = torch.tensor(operand, dtype=torch_format)
        else:
            operand = operand.to(torch_format)

        num_rows_tensor, num_cols_tensor = input_dimensions
        num_tiles_in_input = (num_rows_tensor * num_cols_tensor) // ELEMENTS_PER_TILE

        if num_tiles_in_input < minimal_number_of_tiles_required:
            raise ValueError(
                f"Expected at least 2 tiles for values and 2 tiles for indices (total 4 tiles), but got {num_tiles_in_input} tiles."
            )

        # Create a new zeroed tensor with dimensions [num_rows_tensor, num_stages * K cols].
        result_num_cols = num_stages * K
        result_num_rows = num_rows_tensor

        result = torch.zeros(result_num_rows * result_num_cols, dtype=torch_format)

        for row in range(num_rows_tensor):
            # Create uint16 indices and view as the operand's dtype to preserve bits.
            uint16_indices = torch.arange(
                0, num_cols_tensor // num_stages, dtype=torch.int16
            ).to(torch.uint16)

            # Perform Topk On Row - use operand indices
            operand_values_start_idx = row * num_cols_tensor
            operand_values_end_idx = (
                operand_values_start_idx + num_cols_tensor // num_stages
            )

            values = operand[operand_values_start_idx:operand_values_end_idx]

            # Get top-k values and their positions in the original array.
            # Use stable argsort so ties preserve original order.
            # We always do stable sort, and within the test we can check that ties are handled correctly based on the original order of indices.
            topk_positions = torch.argsort(
                values,
                descending=(sort_direction == TopKSortDirection.Descending),
                stable=True,
            )[:K]

            topk_values = values[topk_positions]

            # Convert uint16 to int32 for indexing (PyTorch doesn't support uint16 indexing)
            topk_indices = uint16_indices.to(torch.int32)[topk_positions].to(
                torch.uint16
            )

            # Write to result tensor - use result indices
            result_values_start_idx = row * result_num_cols
            result_indices_start_idx = (
                result_values_start_idx + result_num_cols // num_stages
            )

            result[result_values_start_idx : result_values_start_idx + K] = topk_values
            result[result_indices_start_idx : result_indices_start_idx + K] = (
                topk_indices.view(operand.dtype)
            )

        # Tilize to match hardware layout.
        result_tilizer = TilizeGolden()
        result = result_tilizer(
            result,
            dimensions=[result_num_rows, result_num_cols],
            data_format=data_format,
        )

        return result


@register_golden
class SdpaSfpuGolden:
    # Columns the kernel writes to.
    TRANSFORMED_COLS = (0, 2, 4, 6, 8, 10, 12, 14)

    def __call__(
        self,
        input_2d,
        op,
        exp_scale: float = 1.0,
        softplus_beta: float = 1.0,
        softplus_threshold: float = 20.0,
    ):
        x = input_2d.to(torch.float32).clone()
        out = x.clone()

        if op == SdpaOp.RecipLegacy:
            transformed = torch.reciprocal(x.abs())
        elif op == SdpaOp.RecipIter:
            transformed = torch.reciprocal(x)
        elif op in (SdpaOp.ExpAccurate, SdpaOp.ExpPoly):
            # Both fold the scale, so the reference is exp(scale * x).
            transformed = torch.exp(exp_scale * x)
        elif op == SdpaOp.Softplus:
            transformed = torch.nn.functional.softplus(
                x, beta=softplus_beta, threshold=softplus_threshold
            )
        else:
            raise ValueError(f"SdpaSfpuGolden: unhandled op {op}")

        cols = torch.tensor(self.TRANSFORMED_COLS, dtype=torch.long)
        out[:, cols] = transformed[:, cols]
        return out


@register_golden
class SdpaCorrectionGolden:
    """Golden for calculate_fused_max_sub_exp_add_tile in ckernel_sfpu_sdpa.h."""

    def __call__(self, tiles, scale: float):
        prev_max, worker_max, cur_max_seed, prev_sum, worker_sum = (
            t.to(torch.float32) for t in tiles
        )

        cur_max = torch.maximum(prev_max, worker_max)
        exp_prev = torch.exp(scale * (prev_max - cur_max))
        exp_worker = torch.exp(scale * (worker_max - cur_max))
        corrected_worker_sum = exp_worker * worker_sum
        corrected_prev_sum = exp_prev * prev_sum

        computed = [
            exp_prev,
            exp_worker,
            cur_max,
            corrected_worker_sum + corrected_prev_sum,
            corrected_worker_sum,
        ]

        cols = torch.tensor(SdpaSfpuGolden.TRANSFORMED_COLS, dtype=torch.long)
        seeds = [prev_max, worker_max, cur_max_seed, prev_sum, worker_sum]
        out = []
        for seed, value in zip(seeds, computed):
            tile = seed.clone()
            tile[:, cols] = value[:, cols]
            out.append(tile)
        return out


@register_golden
class HadamardH128Golden:
    """
    Hadamard computes Y = H_128 @ x (* 1/sqrt(128) when normalizing), where
    H_128 is the Sylvester matrix H_128 = kron(H_8, H_16) and, for row a < 8,
    H_16[a, r < 8] == H_8[a, r], so the first 8 rows of H_16 @ X_pad are H_8 @ X.
    """

    @staticmethod
    def sylvester(order: int) -> torch.Tensor:
        """The Sylvester Hadamard matrix of the given pow2 order."""
        if order <= 0 or order & (order - 1):
            raise ValueError(f"Hadamard order must be a power of two, got {order}")
        matrix = torch.ones(1, 1, dtype=torch.float32)
        while matrix.shape[0] < order:
            matrix = torch.cat(
                [
                    torch.cat([matrix, matrix], dim=1),
                    torch.cat([matrix, -matrix], dim=1),
                ],
                dim=0,
            )
        return matrix

    def __call__(self, x, normalize: bool = True):
        x = x.reshape(-1).to(torch.float32)
        if x.numel() != 128:
            raise ValueError(f"H128 takes a 128-element input, got {x.numel()}")
        result = HadamardH128Golden.sylvester(128) @ x
        if normalize:
            result = result * (1.0 / math.sqrt(128.0))
        return result


@register_golden
class GeneralizedMoeGateGolden:
    """Golden generator for the generalized_moe_gate LLK.

    The gate ranks experts by the score+bias sort key but emits the unbiased payload score of the
    winners, normalized over the kept ranks and scaled. Grouped keeps the four column-pair groups
    {2g, 2g+1} with the largest top-2 key sum, takes the top 8 of their 128 members, and pins topk
    to 8 with linear output.

    Args:
        keys, payload, ids: [16, 16] DEST faces of the sort key, the emitted score and the id.
        topk: how many ranks survive; ranks beyond it emit weight 0 and id 0.
        eps: added to the normalization sum before the reciprocal. scale multiplies the reciprocal.
    Returns:
        float tensor [2, 8], the weight and id per rank, descending by key. One tensor, as the
        golden cache requires.
    """

    def __call__(
        self,
        keys,
        payload,
        ids,
        topk=8,
        output_softmax=False,
        eps=0.0,
        scale=1.0,
        grouped=False,
    ):
        keys = keys.reshape(-1).to(torch.float32)
        payload = payload.reshape(-1).to(torch.float32)
        ids = ids.reshape(-1).to(torch.float32)

        candidates = torch.arange(keys.numel())
        if grouped:
            groups = candidates.reshape(16, 16).t().reshape(8, 32)
            top2 = keys[groups].sort(dim=-1, descending=True).values[:, :2].sum(dim=-1)
            candidates = groups[top2.argsort(descending=True)[:4]].reshape(-1)
            topk, output_softmax = 8, False

        order = candidates[keys[candidates].argsort(descending=True)[:8]]
        weights = payload[order]
        if output_softmax:
            # The kernel subtracts rank 0's payload before the exp, not the largest payload.
            # Softmax is shift invariant so the normalized result is the same either way.
            weights = torch.exp(weights - weights[0])
        weights[topk:] = 0.0

        weights = weights * (scale / (weights[:topk].sum() + eps))
        selected_ids = ids[order]
        selected_ids[topk:] = 0.0
        return torch.stack([weights, selected_ids])


@register_golden
class TopKXLGolden:
    """Golden generator for the TopK-XL LLKs (K = 512/1024/2048).

    TopK-XL takes row-major values and returns row-major indices, so the golden
    is a plain per-row torch.topk.

    Args:
        rows: float tensor [num_rows, search_len] of the per-row values.
        K: number of top elements per row.
    Returns:
        indices: sorted int64 tensor [num_rows, K] of the top-K row-major positions.
    """

    def __call__(self, rows, K):
        _, indices = torch.topk(rows.float(), K, dim=-1, largest=True, sorted=True)
        return indices


@register_golden
class Top32RmGolden:
    """Golden generator for the DeepSeek top32_rm LLKs (row-major top-32, K=32).

    Mirrors the on-silicon gtest reference verify_top32_outputs(): rank the
    (score, original_index) pairs of a single row by score DESCENDING, ties broken
    by the smaller original index, and take the first K. The index paired with each
    surviving score is that score's original row-major position, because the kernel's
    index stream is index[i] = i and index tracking carries it through the sort.

    Every stimulus is exactly representable in bf16, so the fp32 score order equals
    the bf16 compare order the SFPU SFPSWAP uses.

    Args:
        row: 1-D float tensor [row_elements] of the row's scores.
        K:   number of top elements (32 for top32_rm).
    Returns:
        (values, indices): float tensor [K] of the top-K scores and int64 tensor [K]
        of their original row-major positions, in descending-score order.
    """

    def __call__(self, row, K=32):
        row = row.flatten().float()
        n = row.numel()
        # Stable descending sort with an index tiebreak: torch.sort is stable, so
        # sorting by -score keeps the smaller original index first among equal
        # scores, matching the gtest comparator (score desc, orig_idx asc).
        order = torch.argsort(-row, stable=True)[:K]
        return row[order], order.to(torch.int64)


@register_golden
class WhereGolden:
    def __call__(self, operand1, true_value, false_value):
        # Element-wise select matching the C++ sfpu_ternary_function:
        #   result[i] = (cond[i] == 0) ? false_value[i] : true_value[i]
        cond = operand1.flatten().to(torch.float32)
        mask = cond != 0.0
        return torch.where(mask, true_value.flatten(), false_value.flatten())


@register_golden
class TernarySFPUGolden:
    """Golden for the ternary SFPU kernels (addcmul / addcdiv / lerp / snake_beta).

    All operate element-wise on three same-shaped operands (a, b, c) — and, for
    the addc kernels, a scalar constant — so, like where, the result at each
    position depends only on the same-position inputs. No tilize is needed: the
    kernel copies each input tile into a Dest tile (layout-preserving) and the
    SFPU processes rows in place, so a row-major element-wise reference matches
    the packed result.

        addcmul:    out = a + (value * b * c)
        addcdiv:    out = a + (value * b / c)
        lerp:       out = a + c * (b - a)
        snake_beta: out = a + sin(b * a)^2 / c    (a=x, b=alpha, c=beta)

    Known limitation: this reference computes in fp32 with a single final cast and
    is dest-accumulation-agnostic. The kernels, however, branch on
    is_fp32_dest_acc_en for their intermediate rounding (addcmul emits an
    SFP_STOCH_RND fp32->fp16b before the store; addcdiv/lerp round via
    float32_to_bf16_rne; snake_beta drops to a lower-degree sin polynomial when it
    is off), so both dest_acc arms are checked against this one golden and are
    distinguished only by the (looser, for Bfp8_b) PCC/atol tolerance rather than
    by a bit-exact reference. Tightening this into a dest_acc-aware golden that
    models the intermediate bf16 rounding is tracked as follow-up.
    """

    def __call__(
        self,
        operation: MathOperation,
        operand_a,
        operand_b,
        operand_c,
        value_bits: int,
        data_format: DataFormat,
    ):
        # value is passed to the kernel as a raw fp32 bit pattern; decode it the
        # same way (Converter::as_float / SFPLOADI) so the reference agrees.
        value = struct.unpack("<f", struct.pack("<I", value_bits & 0xFFFFFFFF))[0]

        a = operand_a.flatten().to(torch.float32)
        b = operand_b.flatten().to(torch.float32)
        c = operand_c.flatten().to(torch.float32)

        if operation == MathOperation.SfpuAddcmul:
            result = a + (value * b * c)
        elif operation == MathOperation.SfpuAddcdiv:
            result = a + (value * b / c)
        elif operation == MathOperation.SfpuLerp:
            result = a + c * (b - a)
        elif operation == MathOperation.SfpuSnakeBeta:
            result = a + torch.sin(b * a) ** 2 / c
        else:
            raise ValueError(f"Unsupported ternary SFPU operation: {operation}")

        return result.to(format_dict[data_format]).flatten()


@register_golden
class ScalarBinopGolden:
    """Golden for the float unary-with-scalar binops in binop_with_unary.h
    (add / sub / mul / div / rsub).

    Each is element-wise on a single operand plus a scalar, so — like the
    other SFPU element-wise goldens — no tilize is needed: the kernel copies
    the input tile into Dest and processes rows in place, so a row-major
    reference matches the packed result.

        add:  out = x + s
        sub:  out = x - s
        mul:  out = x * s
        div:  out = x * s      (s is the host-inverted divisor 1/d; the kernel
                                multiplies, so this reproduces x / d exactly)
        rsub: out = s - x      (kernel rounds fp32->bf16 RNE for a 16-bit dest;
                                the final cast to a bf16 output format is RNE
                                and reproduces it)
    """

    def __call__(
        self,
        operation: MathOperation,
        operand_a,
        value_bits: int,
        data_format: DataFormat,
        dest_acc: DestAccumulation,
    ):
        # Decode the scalar the same way the kernel does (Converter::as_float).
        value = struct.unpack("<f", struct.pack("<I", value_bits & 0xFFFFFFFF))[0]

        a = operand_a.flatten().to(torch.float32)

        if operation == MathOperation.ScalarAdd:
            result = a + value
        elif operation == MathOperation.ScalarSub:
            result = a - value
        elif operation in (MathOperation.ScalarMul, MathOperation.ScalarDiv):
            result = a * value
        elif operation == MathOperation.ScalarRsub:
            result = value - a
        else:
            raise ValueError(f"Unsupported scalar binop operation: {operation}")

        # Dest, then the pack path -- the same two steps UnarySFPUGolden models. dest_acc decides
        # the Dest width: Yes gives a 32-bit Dest that holds a NaN, No a 16-bit one that does not,
        # and the packer then substitutes an infinity of the NaN's own sign. This suite reaches
        # only Float32 at dest_acc=Yes and Float16_b at dest_acc=No (_skip_unsupported excludes
        # the rest), so the width follows dest_acc alone; a third combination would need the
        # dst_format derivation UnarySFPUGolden.__call__ does.
        #
        # cast_to_dest_dtype rather than .to(): torch's bfloat16 cast forces every NaN's sign bit
        # to 1, which would then decide the substituted infinity's sign by accident.
        result = cast_to_dest_dtype(result, format_dict[data_format]).flatten()
        if dest_acc == DestAccumulation.No:
            result = convert_nan_to_inf(result)
        return result


def truncate_to_bfloat16(values: torch.Tensor) -> torch.Tensor:
    """SFPSTORE into a 16-bit Dest truncates rather than rounds.

    Clearing the low 16 bits of the IEEE-754 pattern drops mantissa bits without
    touching sign or exponent, i.e. moves toward zero for either sign.
    """
    return (
        (values.to(torch.float32).contiguous().view(torch.int32) & ~0xFFFF)
        .view(torch.float32)
        .clone()
    )


def round_to_dest_width(
    values: torch.Tensor, dest_acc: DestAccumulation
) -> torch.Tensor:
    """A value as it sits in Dest, at the width dest_acc selects."""
    if dest_acc == DestAccumulation.Yes:
        return values.to(torch.float32)
    return values.to(torch.bfloat16).to(torch.float32)


@register_golden
class SoftmaxKGolden:
    """Golden for the softmax_k SFPU entry (experimental/ckernel_sfpu_softmax_k.h).

    Per-row softmax over the 16 columns of face 0's first row band, with the row
    maximum supplied by the caller instead of being reduced on the fly:

        out[r][c] = exp(x[r][c] - m[r]) / sum_{c' < k} exp(x[r][c'] - m[r])   c < k
        out[r][c] = 0                                                        c >= k

    Columns >= k have to be exactly 0.0 in the input: the kernel takes a condition
    code from |even-column value| before subtracting the max and only re-enables all
    lanes after the exponential, so those lanes stay 0 and are then multiplied by the
    reciprocal. Rows outside the processed band come back untouched.

    The kernel round-trips through Dest three times -- x - max, exp(), and the
    normalized result -- so the golden quantizes at each of those stores to the width
    dest_acc selects. Plain SFPSTORE truncates; only the exp kernels round, via the
    SFP_STOCH_RND(RND_EVEN, FP32_TO_FP16B) they do before their own store.
    """

    def __call__(self, input_tile, logits, k, dest_acc, rows=4, face_dim=16):
        golden = input_tile.to(torch.float32).clone()

        def stored(values):
            """A plain SFPSTORE: truncating on a 16-bit Dest, exact on a 32-bit one."""
            if dest_acc == DestAccumulation.Yes:
                return values.to(torch.float32)
            return truncate_to_bfloat16(values)

        for row in range(rows):
            shifted = stored(logits[row] - logits[row].max())
            exponentials = round_to_dest_width(torch.exp(shifted), dest_acc)
            golden[row, :k] = stored(exponentials / exponentials.sum())
            golden[row, k:face_dim] = 0.0
        return golden


@register_golden
class MoeGateTopkGolden:
    """Golden for the generic MoE-gate top-k SFPU entry
    (experimental/ckernel_sfpu_generic_moe_gate_topk.h).

    The kernel sorts on the *biased* keys but carries the raw score as the payload, so
    the winners are chosen by `sort_keys` and the expected scores are looked up from
    `scores` by the winning id. That split is the whole point: reporting the key instead
    of the score, or pairing a score with the wrong id, both show up here.

    With normalize the kernel divides each winner score by the sum of the winners' raw
    scores (plus eps) and multiplies by scale; without it the scores pass through.

    Returns (winner_ids, expected_scores) where winner_ids is in descending-key order
    and expected_scores[i] corresponds to winner_ids[i]. Callers that only know the
    winners as an unordered set should reorder via `scores_for_ids`.
    """

    def __call__(
        self,
        sort_keys,
        scores,
        num_winners: int,
        normalize: bool,
        eps: float = 0.0,
        scale: float = 1.0,
    ):
        winner_ids = torch.argsort(sort_keys, descending=True)[:num_winners].tolist()
        return winner_ids, self.scores_for_ids(
            winner_ids, winner_ids, scores, normalize, eps, scale
        )

    def scores_for_ids(
        self,
        ids,
        winner_ids,
        scores,
        normalize: bool,
        eps: float = 0.0,
        scale: float = 1.0,
    ):
        """Expected scores for `ids`, normalized over the `winner_ids` set.

        The normalization denominator is fixed by the winner set, not by `ids`, so a
        caller can ask for the scores in the order the device returned them while
        still dividing by the same total.
        """
        factor = 1.0
        if normalize:
            total = scores[winner_ids].to(torch.float32).sum()
            factor = scale / (total + eps)
        return torch.tensor(
            [scores[i].item() * factor for i in ids], dtype=torch.float32
        )


@register_golden
class SdpaExpUnclampedGolden:
    """Golden for the upper-unclamped exp helpers
    (experimental/ckernel_sfpu_sdpa_exp_unclamped.h).

    The kernel is the accurate 21f exp with the upper input clamp removed, so across
    the domain its caller uses -- val <= 0, and anywhere well below the clamp point
    at val ~= 88.7 -- it is just exp(val * scale). The scale arrives as a *bfloat16*
    bit pattern, which is what sfpi::sFloat16b() consumes, not an fp32 one.

    The kernel static_asserts !is_fp32_dest_acc_en and then does
    `convert<vFloat16b>(y, NearestEven)` unconditionally before the store, so the value
    that reaches Dest is always bf16 regardless of the pack format -- hence the
    round_to_dest_width(DestAccumulation.No) below. The pack format only decides
    whether that value is converted a second time (a no-op) on the way to L1.
    """

    def __call__(self, operand, scale_bits: int, data_format: DataFormat):
        scale = (
            torch.tensor([scale_bits << 16], dtype=torch.int32)
            .view(torch.float32)
            .item()
        )
        result = torch.exp(operand.flatten().to(torch.float32) * scale)
        return round_to_dest_width(result, DestAccumulation.No).to(
            format_dict[data_format]
        )


@register_golden
class SamplingGolden:
    """Golden for the sampling SFPU helpers
    (experimental/llk_sfpu/ckernel_sfpu_sampling.h).

    Element-wise reference per op, followed by the store each one actually performs.
    SFPSTORE into a 16-bit Dest truncates, and only recip_scalar compensates for that
    (convert<vFloat16b>(Nearest), and only when !(DST_ACCUM_MODE || APPROX)); with a
    32-bit Dest nothing rounds at all. Callers pass the scalar operand as a raw fp32
    bit pattern so it decodes exactly the way Converter::as_float does on device.
    """

    # The only helper that converts before storing, so the only one that rounds
    # rather than truncates on a 16-bit Dest.
    ROUND_TO_NEAREST_OPS = {"recip_scalar"}

    def __call__(
        self,
        op: str,
        operand_a,
        operand_b,
        scalar_bits: int,
        dest_acc: DestAccumulation,
    ):
        scalar = struct.unpack("<f", struct.pack("<I", scalar_bits & 0xFFFFFFFF))[0]
        a = operand_a.to(torch.float32)
        b = operand_b.to(torch.float32)

        if op == "recip_scalar":
            result = 1.0 / a
        elif op == "clamp_max_scalar":
            result = torch.minimum(a, torch.full_like(a, scalar))
        elif op == "mul_unary_scalar":
            result = a * scalar
        elif op == "le":
            result = (a <= b).to(torch.float32)
        elif op == "lt":
            result = (a < b).to(torch.float32)
        elif op == "ge":
            result = (a >= b).to(torch.float32)
        elif op == "add":
            result = a + b
        elif op == "sub":
            result = a - b
        elif op == "mul":
            result = a * b
        else:
            raise ValueError(f"Unsupported sampling operation: {op}")

        return self._store(result, op, dest_acc)

    def _store(self, values, op: str, dest_acc: DestAccumulation):
        if dest_acc == DestAccumulation.Yes:
            return values.to(torch.float32)  # the whole fp32 LReg lands in Dest
        if op in self.ROUND_TO_NEAREST_OPS:
            return values.to(torch.bfloat16).to(torch.float32)
        return truncate_to_bfloat16(values)


def rope_bands(
    ht: int,
    wt: int,
    x_base: int,
    x_stride: int,
    cos_base: int,
    sin_base: int,
    cs_stride: int,
):
    """(x_row, cos_row, sin_row) for every vector sfpu_rope_all_rows issues."""
    for w in range(wt):
        for face in range(2):
            cs_offset = w * cs_stride + face * FACE_DIM
            for head in range(ht):
                x_row = x_base + w * x_stride + face * FACE_DIM + head * wt * x_stride
                yield x_row, cos_base + cs_offset, sin_base + cs_offset


def rope_rotated_rows(**geometry) -> list[int]:
    """Every Dest row the rotation writes, ascending."""
    return sorted(
        x_row + i for x_row, _, _ in rope_bands(**geometry) for i in range(4)  # rows
    )


@register_golden
class RopeGolden:
    """
    Adjacent columns of a Dest row contain these pairs:
        x'_even = cos*x_even - sin*x_odd
        x'_odd  = sin*x_even + cos*x_odd
    The golden returns the entire Dest register.
    """

    def __call__(
        self, dest: torch.Tensor, scale: float = None, **geometry
    ) -> torch.Tensor:
        source = dest.to(torch.float32)
        golden = source.clone()

        even = torch.arange(0, source.shape[1], 2)
        odd = even + 1
        factor = 1.0 if scale is None else scale

        for x_row, cos_row, sin_row in rope_bands(**geometry):
            for i in range(4):  # rows
                cos = source[cos_row + i, even] * factor
                sin = source[sin_row + i, even] * factor
                x_even = source[x_row + i, even]
                x_odd = source[x_row + i, odd]
                golden[x_row + i, even] = truncate_to_bfloat16(
                    cos * x_even - sin * x_odd
                )
                golden[x_row + i, odd] = truncate_to_bfloat16(
                    sin * x_even + cos * x_odd
                )
        return golden
