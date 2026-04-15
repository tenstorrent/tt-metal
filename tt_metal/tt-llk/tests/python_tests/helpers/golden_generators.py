# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
import math
import struct
from typing import Optional

import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    BroadcastType,
    DestAccumulation,
    FastMode,
    MathFidelity,
    MathOperation,
    PackerReluType,
    ReduceDimension,
    ReducePool,
    TopKSortDirection,
    format_dict,
    pack_relu_config,
)
from helpers.pack import pack_mxfp8p, pack_mxfp8r
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.unpack import unpack_mxfp8p, unpack_mxfp8r

from .bfp_format_utils import bfp4b_to_float16b as _bfp4b_to_float16b
from .bfp_format_utils import bfp8b_to_float16b as _bfp8b_to_float16b
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


def check_bfp8_b(operand: list) -> list:
    """Check if datum is BFP8_B there is a +/- inf then zero out entire row of 16 elements because they inherit the same exponent and therefore get zeroed out in tensix."""
    # tensor_bytes = pack_bfp8_b(torch.tensor(operand, dtype=torch.bfloat16))
    # tensor = unpack_bfp8_b(tensor_bytes)
    # return tensor

    not_finite = [math.inf, -math.inf]
    for i, x in enumerate(operand):
        if x in not_finite or math.isnan(x):
            # Zero out the entire row of 16 elements
            for col in range(16):
                row = i // 16
                index = row * 16 + col
                if not (operand[index] in not_finite or math.isnan(operand[index])):
                    operand[index] = 0.0

    return operand


def check_bfp4_b(operand: list) -> list:
    """Check if datum is BFP4_B: if there is a +/- inf then zero out entire row of 16 elements because they share the same exponent and therefore get zeroed out in tensix."""
    not_finite = [math.inf, -math.inf]
    for i, x in enumerate(operand):
        if x in not_finite or math.isnan(x):
            # Zero out the entire row of 16 elements
            for col in range(16):
                row = i // 16
                index = row * 16 + col
                if not (operand[index] in not_finite or math.isnan(operand[index])):
                    operand[index] = 0.0

    return operand


def convert_nan_to_inf(operand: list) -> list:
    return [math.inf if math.isnan(x) else x for x in operand]


def convert_inf_to_value(operand: list, inf_value: float) -> list:
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


def quantize_mx_stimuli(
    tensor: torch.Tensor, data_format: DataFormat, num_faces: int = 4
) -> torch.Tensor:
    """
    Quantize MX format stimuli by performing pack→unpack roundtrip.

    This simulates the quantization that occurs when data is stored in MX format
    in L1 memory and then unpacked by hardware. The golden model should use
    quantized values to match what hardware actually sees.

    Args:
        tensor: Input tensor (bfloat16 values)
        data_format: MX format (MxFp8R or MxFp8P)
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
    if data_format == DataFormat.MxFp8R:
        packed = pack_mxfp8r(tensor, num_faces=num_faces)
        return unpack_mxfp8r(packed, num_faces=num_faces)
    elif data_format == DataFormat.MxFp8P:
        packed = pack_mxfp8p(tensor, num_faces=num_faces)
        return unpack_mxfp8p(packed, num_faces=num_faces)
    else:
        # This should never happen due to validation above, but kept for safety
        raise ValueError(f"Unsupported MX format: {data_format}")


def quantize_mx_tensor_chunked(
    tensor: torch.Tensor, data_format: DataFormat
) -> torch.Tensor:
    """
    Quantize MX format tensor by processing in chunks.

    Args:
        tensor: Input tensor (bfloat16 values)
        data_format: MX format (MxFp8R or MxFp8P)

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
            DataFormat.Float16_b: SrcFormatModel._fp16b_to_tf32,
            DataFormat.Float16: SrcFormatModel._fp16_to_tf32,
            DataFormat.Float32: SrcFormatModel._fp32_to_tf32,
            DataFormat.MxFp8R: SrcFormatModel._mxfp8r_to_tf32,
            DataFormat.MxFp8P: SrcFormatModel._mxfp8p_to_tf32,
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


FTZ_THRESHOLD = 1e-37


def quantize_input(operand, fmt, data_format):
    """Quantize a single operand to match what hardware sees after unpack.

    When data is stored in BFP or MX format in L1 memory and unpacked by
    hardware, precision is lost.  This function simulates that loss so the
    golden operates on the same values that hardware sees.

    Args:
        operand: Input tensor.
        fmt: The L1 storage format of this operand (``None`` to skip quantization).
        data_format: Fallback format used for a plain dtype cast when *fmt*
            is ``None`` or does not require special quantization.
    """
    if fmt is None:
        return to_tensor(operand, data_format)
    if fmt == DataFormat.Bfp4_b:
        return _bfp4b_to_float16b(operand)
    if fmt == DataFormat.Bfp8_b:
        return _bfp8b_to_float16b(operand)
    if fmt.is_mx_format():
        return quantize_mx_tensor_chunked(operand, fmt)
    return to_tensor(operand, fmt)


def quantize_output(tensor, data_format):
    """Quantize output and apply flush-to-zero to match hardware packing.

    Hardware packs results back into L1 in the target ``data_format``,
    losing precision.  After quantization, values with magnitude below
    ``FTZ_THRESHOLD`` are flushed to zero (hardware always runs with FTZ).

    Args:
        tensor: Result tensor to quantize.
        data_format: Target L1 storage format.
    """
    if data_format == DataFormat.Bfp4_b:
        tensor = _bfp4b_to_float16b(tensor.to(torch.bfloat16))
    elif data_format == DataFormat.Bfp8_b:
        tensor = _bfp8b_to_float16b(tensor.to(torch.bfloat16))
    elif data_format.is_mx_format():
        tensor = quantize_mx_tensor_chunked(tensor.to(torch.bfloat16), data_format)
    else:
        tensor = to_tensor(tensor, data_format)

    t_f32 = tensor.float()
    return torch.where(
        t_f32.abs() < FTZ_THRESHOLD,
        torch.zeros_like(t_f32),
        t_f32,
    ).to(tensor.dtype)


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

        tensor = quantize_input(operand, data_format, data_format)
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
        tensor = quantize_input(operand, data_format, data_format)

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
    ):
        torch_format = format_dict[data_format]

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

        t1 = to_tensor(operand1, data_format)
        t2 = to_tensor(operand2, data_format)

        # Handle multi-tile matmul with different operand dimensions
        if input_A_dimensions is not None and input_B_dimensions is not None:
            # Multi-tile matmul: A[M,K] × B[K,N] = C[M,N]
            M, K1 = input_A_dimensions[0], input_A_dimensions[1]
            K2, N = input_B_dimensions[0], input_B_dimensions[1]

            # Verify K dimensions match for valid matmul
            if K1 != K2:
                raise AssertionError(
                    f"Matrix dimensions incompatible: A[{M},{K1}] × B[{K2},{N}]"
                )

            output_dimensions = [M, N]

        MATH_FIDELITY_TO_ITER_COUNT = {
            MathFidelity.LoFi: 0,
            MathFidelity.HiFi2: 1,
            MathFidelity.HiFi3: 2,
            MathFidelity.HiFi4: 3,
        }

        fidelity_iter_count = MATH_FIDELITY_TO_ITER_COUNT[math_fidelity]

        res = 0

        if fidelity_iter_count == 0:

            t1, t2 = self._apply_fidelity_masking(data_format, t1, t2, 0)
            t1, t2 = t1.view(M, K1), t2.view(K2, N)
            res = (
                torch.matmul(t1, t2)
                .view(output_dimensions[0] * output_dimensions[1])
                .to(torch_format)
            )

        elif fidelity_iter_count == 1:

            t1, t2 = self._apply_fidelity_masking(data_format, t1, t2, 0)
            t1, t2 = t1.view(M, K1), t2.view(K2, N)
            res = (
                torch.matmul(t1, t2)
                .view(output_dimensions[0] * output_dimensions[1])
                .to(torch_format)
            )

            t1 = to_tensor(operand1, data_format)
            t2 = to_tensor(operand2, data_format)
            t1, t2 = self._apply_fidelity_masking(data_format, t1, t2, 1)
            t1, t2 = t1.view(M, K1), t2.view(K2, N)
            res += (
                torch.matmul(t1, t2)
                .view(output_dimensions[0] * output_dimensions[1])
                .to(torch_format)
            )

        elif fidelity_iter_count == 2:

            t1, t2 = self._apply_fidelity_masking(data_format, t1, t2, 0)
            t1, t2 = t1.view(M, K1), t2.view(K2, N)
            res = (
                torch.matmul(t1, t2)
                .view(output_dimensions[0] * output_dimensions[1])
                .to(torch_format)
            )

            t1 = to_tensor(operand1, data_format)
            t2 = to_tensor(operand2, data_format)
            t1, t2 = self._apply_fidelity_masking(data_format, t1, t2, 1)
            t1, t2 = t1.view(M, K1), t2.view(K2, N)
            res += (
                torch.matmul(t1, t2)
                .view(output_dimensions[0] * output_dimensions[1])
                .to(torch_format)
            )

            t1 = to_tensor(operand1, data_format)
            t2 = to_tensor(operand2, data_format)
            t1, t2 = self._apply_fidelity_masking(data_format, t1, t2, 2)
            t1, t2 = t1.view(M, K1), t2.view(K2, N)
            res += (
                torch.matmul(t1, t2)
                .view(output_dimensions[0] * output_dimensions[1])
                .to(torch_format)
            )

        elif fidelity_iter_count == 3:

            t1, t2 = t1.view(M, K1), t2.view(K2, N)
            res = (
                torch.matmul(t1, t2)
                .view(output_dimensions[0] * output_dimensions[1])
                .to(torch_format)
            )

        res = quantize_output(res, data_format)

        if tilize:
            res = tilize_block(
                res,
                dimensions=(input_A_dimensions[0], input_B_dimensions[1]),
                stimuli_format=data_format,
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
    ):
        if broadcast_type not in self.broadcast_handlers:
            raise ValueError(f"Unsupported broadcast type: {broadcast_type}")

        torch_format = format_dict[data_format]

        # Convert input to tensor
        if isinstance(operand, torch.Tensor):
            input_flat = operand.flatten().to(torch_format)
        else:
            input_flat = torch.tensor(operand, dtype=torch_format).flatten()

        # Quantize the input tile-by-tile BEFORE extracting broadcast values.
        # Hardware unpacks src_B from its block-float encoding before applying
        # the broadcast, so the shared-exponent quantization is determined by
        # the original (non-broadcast) 16-element rows of the tile.
        input_flat = quantize_input(input_flat, data_format, data_format)

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
    ):
        torch_format = format_dict[data_format]

        operand1 = quantize_input(operand1, input_format, data_format)

        height, width = input_dimensions[0], input_dimensions[1]

        # Handle partial faces (face_r_dim < 16) as single tiles
        if face_r_dim < 16:
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
            # Apply saturation for integer format conversions to match hardware behavior
            # Hardware saturates (clamps) values instead of wrapping around
            if data_format.is_integer():
                iinfo = torch.iinfo(torch_format)
                is_unsigned = str(data_format).startswith("U")
                if is_unsigned:
                    min_val, max_val = iinfo.min, iinfo.max
                else:
                    min_val, max_val = iinfo.min + 1, iinfo.max

                # Convert to intermediate type (int64 or int32) to avoid overflow during clamping
                # Use int64 when source can hold values outside int32 range (e.g. UInt32 is torch.int64)
                intermediate_type = (
                    torch.int64
                    if result.dtype in (torch.uint32, torch.int64)
                    else torch.int32
                )
                result = result.to(intermediate_type)

                # Apply saturation to clamp values to destination range
                # This handles downsizing (Int32->Int8), signed/unsigned conversions (UInt8->Int8),
                # and any case where source values might exceed destination range
                result = torch.clamp(result, min_val, max_val)

            result = result.to(torch_format)

        # Apply BFP output quantization round-trip to match hardware behaviour
        if data_format in (DataFormat.Bfp4_b, DataFormat.Bfp8_b):
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
            else:
                result = _bfp8b_to_float16b(data, dims)

        return result


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
        elif operand1.dtype != torch_format:
            operand1 = operand1.to(torch_format)

        result = operand1.view(tile_cnt, tile_size)[
            :, :elements_per_tile_needed
        ].reshape(-1)

        return quantize_output(result, data_format)

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
            # Encode as BF16 (upper 16 bits of FP32)
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
        relu_type = PackerReluType(relu_config & 0x3)
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
        relu_type = PackerReluType(relu_config & 0x3)

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


@register_golden
class UnarySFPUGolden:
    def __init__(self):
        self.ops = {
            MathOperation.Abs: self._abs,
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
            MathOperation.Sqrt: self._sqrt,
            MathOperation.Square: self._square,
            MathOperation.Tanh: self._tanh,
            MathOperation.Celu: self._celu,
            MathOperation.Silu: self._silu,
            MathOperation.Gelu: self._gelu,
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
            MathOperation.ReduceColumn: self._reduce_columns,
            MathOperation.ReduceRow: self._reduce_rows,
        }
        self.data_format = None
        self.dest_acc = DestAccumulation.No
        self.approx_mode = ApproximationMode.No
        self.fast_mode = FastMode.No

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
        approx_mode: ApproximationMode = ApproximationMode.No,
        fast_mode: FastMode = FastMode.No,
    ):
        self.data_format = data_format
        self.dest_acc = dest_acc
        self.approx_mode = approx_mode
        self.fast_mode = fast_mode

        if operation not in self.ops:
            raise ValueError(f"Unsupported operation: {operation}")

        operand1 = quantize_input(operand1, input_format, data_format)

        # Special handling for Column and Row reduction which needs to process the entire tensor
        if operation in [MathOperation.ReduceColumn, MathOperation.ReduceRow]:
            return self.ops[operation](operand1, reduce_pool)

        # determine the data format for dst
        if self.dest_acc == DestAccumulation.Yes:
            dst_format = DataFormat.Float32
        elif DataFormat.Float16 in (input_format, data_format):
            dst_format = DataFormat.Float16
        else:
            dst_format = DataFormat.Float16_b

        if self.dest_acc == DestAccumulation.No and input_format == DataFormat.Float32:
            # dst in 16-bit mode and 32-bit input: truncation may occur when unpacked to dst
            if dst_format == DataFormat.Float16:
                # truncate to float16
                operand1 = (operand1.view(torch.int32) & 0xFFFFE000).view(torch.float32)
            else:
                # truncate to float16_b
                operand1 = (operand1.view(torch.int32) & 0xFFFF0000).view(torch.float32)

        tensor = to_tensor(operand1, dst_format)

        if iterations is None or iterations * TILE_SIZE > tensor.numel():
            iterations = tensor.numel() // TILE_SIZE

        if iterations <= 0:
            raise ValueError(f"Invalid iterations: {iterations}")

        result = tensor.clone().flatten()

        if not skip_tilize:
            result = tilize_block(result, dimensions, input_format).flatten()

        start = ELEMENTS_PER_TILE * dest_idx
        elements_to_process = TILE_SIZE * iterations

        if start + elements_to_process > tensor.numel():
            raise ValueError(
                f"Processing {iterations} iterations from dest_idx={dest_idx} "
                f"would exceed tensor bounds (trying to access element {start + elements_to_process}, "
                f"but tensor has only {tensor.numel()} elements)"
            )

        op_res = [
            (
                self.ops[operation](x, fill_const_value)
                if operation == MathOperation.Fill
                else self.ops[operation](x)
            )
            for x in result.tolist()[
                ELEMENTS_PER_TILE * dest_idx : ELEMENTS_PER_TILE * dest_idx
                + TILE_SIZE * iterations
            ]
        ]

        op_dtype = (
            torch.float32
            if data_format == DataFormat.Bfp4_b
            else format_dict[dst_format]
        )
        result[
            ELEMENTS_PER_TILE * dest_idx : ELEMENTS_PER_TILE * dest_idx
            + TILE_SIZE * iterations
        ] = torch.tensor(op_res, dtype=op_dtype)

        if not skip_tilize:
            result = untilize_block(result, input_format, dimensions).flatten()

        if self.data_format == DataFormat.Bfp8_b:
            check_bfp8_b(result)

        if self.data_format == DataFormat.Bfp4_b:
            check_bfp4_b(result)

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

        if data_format == DataFormat.Bfp4_b:
            result_t = (
                torch.tensor(result, dtype=torch.float32)
                if not isinstance(result, torch.Tensor)
                else result.float()
            )
            tilized = tilize_block(
                result_t.flatten(), dimensions, DataFormat.Float16_b
            ).flatten()
            result = _bfp4b_to_float16b(tilized, dimensions)

        # depending on `data_format`, `inf` values may get converted when unpacked to L1.
        if dst_format == DataFormat.Float16:
            match data_format:
                case DataFormat.Float16_b:
                    result = convert_inf_to_value(result, 130560.0)
                case DataFormat.Float32:
                    result = convert_inf_to_value(result, 131008.0)
                case DataFormat.Bfp8_b:
                    result = convert_inf_to_value(result, 130048.0)
                case DataFormat.Bfp4_b:
                    result = convert_inf_to_value(result, 130048.0)

        return torch.tensor(result, dtype=format_dict[data_format])

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

    # Operation methods
    def _abs(self, x):
        return abs(x)

    def _atanh(self, x):
        if x < -1.0 or x > 1.0:
            return math.nan
        if x == -1.0:
            return self.handle_infinite_numbers(-math.inf)
        if x == 1.0:
            return self.handle_infinite_numbers(math.inf)
        return math.atanh(x)

    def _asinh(self, x):
        return math.asinh(x)

    def _acosh(self, x):
        if x < 1.0:
            return math.nan
        return math.acosh(x)

    def _cos(self, x):
        return math.cos(x)

    def _log(self, x):
        if x == 0.0:
            return self.handle_infinite_numbers(-math.inf)
        return math.log(x)

    def _log1p(self, x):
        if x == -1.0:
            return self.handle_infinite_numbers(-math.inf)
        return math.log1p(x)

    def _reciprocal(self, x):
        if x == 0.0:
            return self.handle_infinite_numbers(float("inf"))
        return 1 / x

    def _sin(self, x):
        # Never not finite, values range from [-1, 1]
        return math.sin(x)

    def _relu(self, x):
        return max(0.0, x)

    def _rsqrt(self, x):
        if x < 0.0:
            return self.handle_infinite_numbers(float("nan"))
        if x == 0.0:
            return self.handle_infinite_numbers(float("inf"))
        return 1 / math.sqrt(x)

    def _sqrt(self, x):
        if x < 0.0:
            return math.nan
        return math.sqrt(x)

    def _tanh(self, x):
        return math.tanh(x)

    def _square(self, x):
        if not math.isfinite(x * x):
            return self.handle_infinite_numbers(math.inf)
        return x * x

    def _tanh(self, x):
        return math.tanh(x)

    def _celu(self, x):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.data_format])
        )
        return torch.nn.functional.celu(input_tensor, alpha=1.0).item()

    def _silu(self, x):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.data_format])
        )
        return torch.nn.functional.silu(input_tensor).item()

    def _elu(self, x):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.data_format])
        )
        return torch.nn.functional.elu(input_tensor, alpha=1.0).item()

    def _exp(self, x):
        if self.approx_mode == ApproximationMode.Yes:
            if self.fast_mode == FastMode.Yes:
                return self._exp_schraudolph_fast(x, clamp_negative=True)
            else:
                return self._exp_piecewise_approx(x)
        else:
            input_tensor = (
                x
                if isinstance(x, torch.Tensor)
                else torch.tensor(x, dtype=format_dict[self.data_format])
            )
            return torch.exp(input_tensor).item()

    # ===============================================================
    # LLK implementations of exp
    # ===============================================================
    @staticmethod
    def _exp_schraudolph_approx(x):
        """Schraudolph scheme used by _calculate_exponential_approx_.

        Computes exp(x) by exploiting the IEEE 754 float bit layout:
          1. in = x * (1/ln2) + c23_73       (c23_73 = bf16(0x4340) = 192.0)
          2. in_short = adj_exp + reinterpret<int32>(in)
                        (adj_exp = bf16(0xBD3F) reinterpreted as int32)
          3. in_short <<= 7                   (10 - FRAC_BITS where FRAC_BITS=3)
          4. return reinterpret<float>(in_short)
        """
        LN2_RECIP = 1.442695
        C23_73 = 192.0  # bf16 0x4340 → float32 0x43400000
        # bf16 0xBD3F → float32 0xBD3F0000 → reinterpret as int32
        ADJ_EXP_INT = struct.unpack("!i", bytes.fromhex("BD3F0000"))[0]

        val = x * LN2_RECIP + C23_73
        val_int = struct.unpack("i", struct.pack("f", val))[0]
        in_short = ADJ_EXP_INT + val_int
        in_short = (in_short << 7) & 0xFFFFFFFF
        if in_short >= 0x80000000:
            in_short -= 0x100000000
        return struct.unpack("f", struct.pack("i", in_short))[0]

    @staticmethod
    def _exp_piecewise_approx(x):
        """_calculate_exponential_piecewise_ with APPROXIMATION_MODE=true.

        Clamps extreme inputs then delegates to the Schraudolph approximation.
        """
        if x >= 89:
            return float("inf")
        if x < -42:
            return 0.0
        return UnarySFPUGolden._exp_schraudolph_approx(x)

    @staticmethod
    def _exp_schraudolph_fast(x, clamp_negative=True):
        """Schraudolph scheme for FAST_APPROX && APPROXIMATION_MODE.

        When clamp_negative=True: clamps inputs below -88.5 to -88.5, uses
        uint16 rounding and left-shift by 15.

        When clamp_negative=False: uses int16 sign-magnitude rounding, shift,
        and sign-setting. Outputs for x < ~-88 may be negative (need ReLU).
        """
        A = 369.329925537109375  # 256.0 / ln(2)
        B_MINUS_C = 32500.818359375
        THRESHOLD = -88.5

        if clamp_negative:
            x = max(x, THRESHOLD)
            val = A * x + B_MINUS_C
            val_int = int(round(val))
            val_int = max(0, min(val_int, 65535))
            result_bits = (val_int << 15) & 0xFFFFFFFF
        else:
            val = A * x + B_MINUS_C
            val_int = int(round(val))
            val_int = max(-32767, min(val_int, 32767))
            sign = 0 if val_int >= 0 else 1
            magnitude = abs(val_int)
            shifted = (magnitude << 15) & 0x7FFFFFFF
            result_bits = shifted | (sign << 31)

        return struct.unpack("f", struct.pack("I", result_bits))[0]

    @staticmethod
    def _exp_horner(x):
        """_sfpu_exp_ Horner-form polynomial used when APPROXIMATION_MODE=false.

        For positive x: extracts exponent, sets it to -1, evaluates a 2nd-order
        Horner polynomial, then squares repeatedly to restore the exponent.
        For negative x: computes exp(|x|) then takes reciprocal.
        """
        if x == 0.0:
            return 1.0

        is_negative = x < 0
        val = abs(x)

        bits = struct.unpack("I", struct.pack("f", val))[0]
        exp_biased = (bits >> 23) & 0xFF
        original_exp = exp_biased - 127

        if original_exp >= 0:
            bits_new = (bits & 0x807FFFFF) | (126 << 23)
            val = struct.unpack("f", struct.pack("I", bits_new))[0]

        tmp = val * 0.8373 + 0.863281
        val = val * tmp + 1.0

        if original_exp >= 0:
            for _ in range(original_exp + 1):
                val = val * val

        if is_negative:
            if val != 0:
                val = 1.0 / val

        return val

    # ===============================================================
    # End of LLK implementations of exp
    # ===============================================================

    def _exp2(self, x):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.data_format])
        )
        return torch.exp2(input_tensor).item()

    def _neg(self, x):
        return -x

    def _gelu(self, x):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.data_format])
        )
        return torch.nn.functional.gelu(input_tensor).item()

    def _fill(self, x, const_value=5):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.data_format])
        )
        return input_tensor.fill_(const_value).item()

    def _hardsigmoid(self, x):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.data_format])
        )
        return torch.nn.functional.hardsigmoid(input_tensor).item()

    def _sigmoid(self, x):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.data_format])
        )
        return torch.nn.functional.sigmoid(input_tensor).item()

    def _threshold(self, x, t=5, v=10):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.data_format])
        )
        return torch.nn.functional.threshold(input_tensor, t, v).item()

    def _relu_max(self, x, threshold=5):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.data_format])
        )
        return torch.relu(torch.min(input_tensor, torch.tensor(threshold))).item()

    def _relu_min(self, x, threshold=5):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.data_format])
        )
        return torch.max(input_tensor, torch.tensor(threshold)).item()

    def _reduce_columns(self, x, reduce_pool: ReducePool):
        """Reduce columns across tiles, computing sum, average, or max."""
        # Reduce columns within this tensor
        # Take max along the height (dim=0) for each column
        if reduce_pool == ReducePool.Max:
            reduced_tile = torch.max(x, dim=0).values
        elif reduce_pool == ReducePool.Min:
            reduced_tile = torch.min(x, dim=0).values
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
        """Reduce rows across tiles, computing sum, average, or max."""
        if reduce_pool == ReducePool.Max:
            reduced_tile = torch.max(x, dim=1).values
        elif reduce_pool == ReducePool.Sum:
            reduced_tile = torch.sum(x, dim=1)
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

    _UNSET = object()

    def __call__(
        self,
        op,
        operand1,
        operand2,
        data_format,
        math_fidelity,
        input_format=None,
        input_format_B=_UNSET,
    ):
        if op not in self.ops:
            raise ValueError(f"Unsupported Eltwise operation: {op}")

        # If input_format_B is not provided at all, default to input_format.
        # If explicitly passed as None, it means "already quantized, skip".
        if input_format_B is EltwiseBinaryGolden._UNSET:
            input_format_B = input_format

        operand1 = quantize_input(operand1, input_format, data_format)
        operand2 = quantize_input(operand2, input_format_B, data_format)

        # Fidelity masking models the source register decomposition, so use
        # the *input* format, not the output format.  Block-float / MX formats
        # are unpacked to Float16_b in the source registers.
        def _src_reg_format(fmt):
            if fmt is None:
                return None
            if fmt in (DataFormat.Bfp4_b, DataFormat.Bfp8_b) or fmt.is_mx_format():
                return DataFormat.Float16_b
            return fmt

        math_format_for_fidelity = _src_reg_format(input_format) or data_format

        t1, t2 = operand1, operand2

        # Step 2: Compute the operation (with fidelity masking for Elwmul).
        MATH_FIDELITY_TO_ITER_COUNT = {
            MathFidelity.LoFi: 0,
            MathFidelity.HiFi2: 1,
            MathFidelity.HiFi3: 2,
            MathFidelity.HiFi4: 3,
        }
        fidelity_iter_count = MATH_FIDELITY_TO_ITER_COUNT[math_fidelity]

        if op == MathOperation.Elwmul:
            result = None
            orig_t1, orig_t2 = t1, t2
            for fidelity_iter in range(fidelity_iter_count + 1):
                if fidelity_iter > 0:
                    t1, t2 = operand1, operand2
                masked_t1, masked_t2 = self._apply_fidelity_masking(
                    math_format_for_fidelity, orig_t1, orig_t2, fidelity_iter
                )
                phase_result = self.ops[op](masked_t1, masked_t2)
                if fidelity_iter == 0:
                    result = phase_result
                else:
                    result += phase_result
        else:
            result = self.ops[op](t1, t2)

        return quantize_output(result, data_format)

    # Operation methods
    def _add(self, t1, t2):
        return (t1.to(torch.float32) + t2.to(torch.float32)).to(t1.dtype)

    def _sub(self, t1, t2):
        return (t1.to(torch.float32) - t2.to(torch.float32)).to(t1.dtype)

    def _mul(self, t1, t2):
        return (t1.to(torch.float32) * t2.to(torch.float32)).to(t1.dtype)


@register_golden
class BinarySFPUGolden(EltwiseBinaryGolden):
    def __init__(self):
        super().__init__()
        self.ops.update(
            {
                MathOperation.SfpuElwadd: self._add,
                MathOperation.SfpuElwsub: self._sub,
                MathOperation.SfpuElwmul: self._mul,
                MathOperation.SfpuXlogy: self._xlogy,
                MathOperation.SfpuElwRightShift: self._right_shift,
                MathOperation.SfpuElwLeftShift: self._left_shift,
                MathOperation.SfpuElwLogicalRightShift: self._logical_right_shift,
                MathOperation.SfpuAddTopRow: self._add_top_row,
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
    ):
        if operation not in self.ops:
            raise ValueError(f"Unsupported SFPU operation: {operation}")

        if num_iterations < 1:
            raise ValueError(f"num_iterations must be at least 1, got {num_iterations}")

        total_elements = dimensions[0] * dimensions[1]
        elements_per_tile = ELEMENTS_PER_TILE
        elements_per_row = 32

        num_tiles = total_elements // elements_per_tile

        src1_start = src1_idx * elements_per_tile
        src2_start = src2_idx * elements_per_tile
        dst_start = dst_idx * elements_per_tile

        if operation == MathOperation.SfpuAddTopRow:
            return self._add_top_row(
                tensor.flatten(),
                src1_idx,
                src2_idx,
                dst_idx,
            )

        if not skip_tilize and data_format not in (
            DataFormat.Bfp8_b,
            DataFormat.Bfp4_b,
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

        for iteration in range(num_iterations):
            row_offset = iteration * elements_per_row

            src1_row_start = src1_start + row_offset
            src2_row_start = src2_start + row_offset
            dst_row_start = dst_start + row_offset

            src1_row = result[src1_row_start : src1_row_start + elements_per_row]
            src2_row = result[src2_row_start : src2_row_start + elements_per_row]

            result_row = torch.tensor(
                [
                    self.ops[operation](src1_row[i], src2_row[i])
                    for i in range(elements_per_row)
                ],
                dtype=format_dict[data_format],
            )

            result[dst_row_start : dst_row_start + elements_per_row] = result_row

        if not skip_tilize and data_format not in (
            DataFormat.Bfp8_b,
            DataFormat.Bfp4_b,
        ):
            result = untilize_block(result, data_format, dimensions)

        return quantize_output(result, data_format)

    # Operation methods are covered by Eltwise Binary Golden
    def _xlogy(self, x, y):
        # Unable to model edge cases for Tensix behavior in golden.
        # Tensix shows inconsistent patterns in handling non-finite results for xlogy, depending on the input,
        # data format (both input and output), and destination accumulation (dest_acc).
        # We need to work with the Tensix team to understand when and why certain results are returned,
        # what configuration dependencies exist, and how to handle them appropriately.
        # Without this understanding, discrepancies will occur between golden and Tensix results due to differing edge case handling.
        pass

    def _right_shift(self, t1, t2):
        return torch.bitwise_right_shift(t1, t2).item()

    def _left_shift(self, t1, t2):
        return torch.bitwise_left_shift(t1, t2).item()

    def _logical_right_shift(self, t1, t2):
        # Perform logical right shift by treating t1 as unsigned 32-bit
        t1_uint = t1.to(torch.int64) & 0xFFFFFFFF
        result = (t1_uint >> t2).to(torch.int32)
        return result

    def _add_top_row(
        self,
        tensor,
        src1_idx,
        src2_idx,
        dst_idx,
    ):
        """
        Add top row operation for tile pairs in untilized format.
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

        result[rows_0_1_dst_start:rows_0_1_dst_end] = (
            tensor[rows_0_1_src1_start:rows_0_1_src1_end]
            + tensor[rows_0_1_src2_start:rows_0_1_src2_end]
        )

        # Add rows 8-9 (elements 256-319)
        rows_8_9_dst_start = dst_idx_start + ROWS_8_9_OFFSET
        rows_8_9_dst_end = rows_8_9_dst_start + TWO_ROWS_ELEMENTS
        rows_8_9_src1_start = src1_idx_start + ROWS_8_9_OFFSET
        rows_8_9_src1_end = rows_8_9_src1_start + TWO_ROWS_ELEMENTS
        rows_8_9_src2_start = src2_idx_start + ROWS_8_9_OFFSET
        rows_8_9_src2_end = rows_8_9_src2_start + TWO_ROWS_ELEMENTS

        result[rows_8_9_dst_start:rows_8_9_dst_end] = (
            tensor[rows_8_9_src1_start:rows_8_9_src1_end]
            + tensor[rows_8_9_src2_start:rows_8_9_src2_end]
        )

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

        fmt_for_plain = input_format if input_format is not None else data_format
        operand = quantize_input(operand, input_format, fmt_for_plain)

        if reduce_to_one:
            # Accumulate all tiles into a single result
            return self._reduce_all_tiles(
                operand, reduce_dim, pool_type, data_format, tile_cnt, tile_shape
            )
        else:
            # Process each tile independently; quantize output like eltwise binary
            return torch.cat(
                [
                    quantize_output(
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

        return quantize_output(accumulated, data_format)

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
        # Pool together columns: reduce along rows (dim=0) for each column
        result = torch.zeros(
            tile_shape.total_tile_size(), dtype=format_dict[data_format]
        )

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
        # Pool together rows: reduce along columns (dim=1) for each row
        result = torch.zeros(
            tile_shape.total_tile_size(), dtype=format_dict[data_format]
        )

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
        # Pool together all faces → single scalar at [0]
        result = torch.zeros(
            tile_shape.total_tile_size(), dtype=format_dict[data_format]
        )
        result[0] = self._apply_pooling(faces.flatten(), pool_type, dim=0)
        return result

    def _apply_pooling(self, tensor, pool_type, dim):
        if pool_type == ReducePool.Max:
            return torch.max(tensor, dim=dim).values
        elif pool_type == ReducePool.Average:
            return torch.mean(tensor, dim=dim)
        elif pool_type == ReducePool.Sum:
            return torch.sum(tensor, dim=dim)
        else:
            raise ValueError(f"Unsupported pool type: {pool_type}")


@register_golden
class ReduceBlockMaxRowGolden:
    def __call__(self, operand, ct_dim, data_format, dimensions, input_format=None):
        operand = quantize_input(operand, input_format, data_format)

        operand = operand.reshape(dimensions)
        output = torch.zeros(dimensions)
        for i in range(dimensions[0]):
            output[i, 0] = torch.max(operand[i, : ct_dim * 32])

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
        input_format=None,
    ):

        fidelity_iter_count = self.MATH_FIDELITY_TO_ITER_COUNT[math_fidelity]

        fmt = input_format if input_format is not None else data_format
        operand1 = quantize_input(operand1, input_format, fmt)
        operand2 = quantize_input(operand2, input_format, fmt)

        result = torch.cat(
            [
                self._process_tile(
                    operand1,
                    operand2,
                    data_format,
                    reduce_dim,
                    fidelity_iter_count,
                    tile,
                )
                for tile in range(tile_cnt)
            ]
        )

        return quantize_output(result, data_format)

    def _process_tile(
        self, operand1, operand2, data_format, reduce_dim, fidelity_iter_count, tile_idx
    ):
        # Extract srcA tile and srcB face0 (only f0 unpacked for srcB)
        tile_start = tile_idx * ELEMENTS_PER_TILE
        src_a = to_tensor(
            operand1[tile_start : tile_start + ELEMENTS_PER_TILE], data_format
        )
        src_b = to_tensor(operand2[:ELEMENTS_PER_FACE], data_format)

        # Row reduce: transpose within each face of SrcA (models unpacker behavior)
        if reduce_dim == ReduceDimension.Row:
            src_a = (
                src_a.view(FACES_PER_TILE, FACE_DIM, FACE_DIM).transpose(1, 2).flatten()
            )

        # Compute gapool for each face across all fidelity iterations
        face_results = self._compute_gapool(
            src_a, src_b, data_format, fidelity_iter_count
        )

        # Combine results based on reduce dimension
        return self._accumulate_gapool_results(
            face_results, src_b, data_format, reduce_dim, fidelity_iter_count
        )

    def _compute_gapool(
        self, src_a, src_b, data_format, fidelity_iter_count, num_faces=FACES_PER_TILE
    ):
        """Compute D = srcB @ srcA for each face, accumulating across fidelity iterations."""
        face_results = torch.zeros(
            num_faces, FACE_DIM * FACE_DIM, dtype=src_a.dtype, device=src_a.device
        )

        for fidelity_iter in range(fidelity_iter_count + 1):
            a_masked, b_masked = self._apply_fidelity_masking(
                data_format, src_a, src_b, fidelity_iter
            )

            a_faces = a_masked.view(num_faces, FACE_DIM, FACE_DIM)
            b_face = b_masked.view(1, FACE_DIM, FACE_DIM)
            result = torch.matmul(b_face, a_faces)

            # Flatten and accumulate in-place
            face_results += result.view(num_faces, -1)

        return face_results

    def _accumulate_gapool_results(
        self, face_results, src_b, data_format, reduce_dim, fidelity_iter_count
    ):
        """Place pooled results in output tile based on reduce dimension."""
        face_shape = (FACE_DIM, FACE_DIM)
        f0, f1, f2, f3 = face_results
        result = torch.zeros(ELEMENTS_PER_TILE, dtype=f0.dtype)

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
            pool_result = self._compute_gapool(
                all_faces, src_b, data_format, fidelity_iter_count, num_faces=1
            )
            result[0] = pool_result[0][0]  # First element of a single face result

        return result


@register_golden
class UntilizeGolden:
    def __call__(self, operand, data_format, dimensions=[32, 32], input_format=None):
        from helpers.tilize_untilize import untilize_block

        operand = quantize_input(operand, input_format, data_format)

        result = untilize_block(
            operand, stimuli_format=data_format, dimensions=dimensions
        )
        return result.flatten()


@register_golden
class TilizeGolden:
    def __call__(
        self, operand, dimensions, data_format, num_faces=4, input_format=None
    ):
        from helpers.llk_params import format_dict
        from helpers.tilize_untilize import tilize_block

        # Validate the number of faces
        if not (1 <= num_faces <= 4):
            raise ValueError(f"`num_faces` must be between 1 and 4, got {num_faces}")

        operand = quantize_input(operand, input_format, data_format)

        # Always do full tilization first
        result = tilize_block(operand, dimensions, data_format)
        torch_format = format_dict[data_format]

        # Then select the appropriate number of faces from the tilized result
        if num_faces < FACES_PER_TILE:
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

        return quantize_output(result, data_format)


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
        input_format=None,
    ):
        """
        Perform per-row topk on a tensor.

        Args:
            operand: Input tensor (flattened).
            data_format: Data format for the result.
            k: Number of top elements to extract per row (default: 32).
            sort_direction: Direction to sort top-k values (default: Descending).
            input_dimensions: Input dimensions [rows, cols] (default: [32, 128]).
            input_format: Input data format for BFP quantization (optional).
        Constraint:
            In LLK api, we perform topk with k >= 32, so input tensor must always have at least 2 tiles for values and
            of course 2 tiles for indices since we need to reorder them based on the topk results.
            Therefore input_dimensions must contain at least 4 tiles.
        Returns:
            Tensor with topk applied per row. One Tile of values followed by one tile of indices.
        """
        torch_format = format_dict[data_format]

        operand = quantize_input(operand, input_format, data_format)

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
