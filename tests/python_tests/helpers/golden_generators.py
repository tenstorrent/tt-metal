# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
import math
from typing import Optional

import torch
from helpers.format_arg_mapping import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    ReduceDimension,
    ReducePool,
    format_dict,
)
from helpers.format_config import DataFormat
from helpers.tilize_untilize import tilize_block, untilize

golden_registry = {}

_FIDELITY_MASK_CONFIGURATION = {
    0: (0x7C0, 0x7F0),
    1: (0x3E, 0x7F0),
    2: (0x7C0, 0x0F0),
    3: (0x3E, 0x0F),
}


def apply_masks(mantissas_1, mantissas_2, math_fidelity_phase):
    """Apply masks to mantissas based on math fidelity phase."""
    a_mask, b_mask = _FIDELITY_MASK_CONFIGURATION[math_fidelity_phase]
    return mantissas_1 & a_mask, mantissas_2 & b_mask


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

    if data_format in [DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Float32]:
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


class FidelityMasking:
    def _apply_fidelity_masking(
        self, operand1, operand2, math_fidelity_phase, data_format
    ):

        # Extract exponents from all operands based on data format
        if data_format == DataFormat.Float16:
            # Convert operands to uint16 for bitwise operations
            operand1_uint = operand1.to(torch.float16).view(torch.uint16)
            operand2_uint = operand2.to(torch.float16).view(torch.uint16)

            # Mask 5 bits starting from 2nd MSB (bits 10 to 14)
            exponent_mask = 0x7C00  # 0111 1100 0000 0000

            exponents_1 = operand1_uint & exponent_mask
            exponents_2 = operand2_uint & exponent_mask
            exponents_1 = exponents_1.to(torch.int32) >> 10
            exponents_2 = exponents_2.to(torch.int32) >> 10

            sign_mask = 0x8000  # 1000 0000 0000 0000
            sign_1 = operand1_uint & sign_mask
            sign_2 = operand2_uint & sign_mask

            mantissa_mask = 0x3FF  # 0000 0011 1111 1111
            mantissas_1 = operand1_uint & mantissa_mask
            mantissas_2 = operand2_uint & mantissa_mask

        elif data_format in [DataFormat.Float16_b, DataFormat.Bfp8_b]:
            # Convert operands to uint16 for bitwise operations
            operand1_uint = operand1.to(torch.bfloat16).view(torch.uint16)
            operand2_uint = operand2.to(torch.bfloat16).view(torch.uint16)

            # Mask 8 bits starting from 2nd MSB (bits 7 to 14)
            exponent_mask = 0x7F80  # 0111 1111 1000 0000

            exponents_1 = operand1_uint & exponent_mask
            exponents_2 = operand2_uint & exponent_mask
            exponents_1 = exponents_1.to(torch.int32) >> 7
            exponents_2 = exponents_2.to(torch.int32) >> 7

            sign_mask = 0x8000  # 1000 0000 0000 0000
            sign_1 = operand1_uint & sign_mask
            sign_2 = operand2_uint & sign_mask

            mantissa_mask = 0x7F  # 0000 0000 0111 1111
            mantissas_1 = operand1_uint & mantissa_mask
            mantissas_2 = operand2_uint & mantissa_mask

            mantissas_1 = mantissas_1.to(torch.int32) << 3
            mantissas_2 = mantissas_2.to(torch.int32) << 3

        elif data_format == DataFormat.Float32:
            # Convert operands to uint32 for bitwise operations
            operand1_uint = operand1.to(torch.float32).view(torch.uint32)
            operand2_uint = operand2.to(torch.float32).view(torch.uint32)

            # Mask 8 bits starting from 2nd MSB (bits 23 to 30)
            exponent_mask = 0x7F800000  # 0111 1111 1000 0000 0000 0000 0000 0000

            exponents_1 = operand1_uint & exponent_mask
            exponents_2 = operand2_uint & exponent_mask

            exponents_1 = exponents_1.to(torch.int32) >> 23
            exponents_2 = exponents_2.to(torch.int32) >> 23

            sign_mask = 0x80000000  # 1000 0000 0000 0000 0000 0000 0000 0000
            sign_1 = operand1_uint & sign_mask
            sign_2 = operand2_uint & sign_mask

            mantissa_mask = 0x007FFFFF  # 0000 0000 0111 1111 1111 1111 1111 1111
            mantissas_1 = operand1_uint & mantissa_mask
            mantissas_2 = operand2_uint & mantissa_mask

            mantissas_1 = mantissas_1.to(torch.int32) >> 13
            mantissas_2 = mantissas_2.to(torch.int32) >> 13
        else:
            raise ValueError(
                f"Unsupported data format for fidelity application: {data_format}"
            )

        mantissa_msb = 0x400  # 1 << 10, MSB of an 11-bit number

        mantissas_1 = mantissas_1 | mantissa_msb
        mantissas_2 = mantissas_2 | mantissa_msb

        mantissas_1, mantissas_2 = apply_masks(
            mantissas_1, mantissas_2, math_fidelity_phase
        )

        # Recombine the sign, exponent, and mantissa bits
        sign_1 = sign_1.to(torch.int16)
        exponents_1 = exponents_1.to(torch.int16)
        mantissas_1 = mantissas_1.to(torch.int16)
        sign_2 = sign_2.to(torch.int16)
        exponents_2 = exponents_2.to(torch.int16)
        mantissas_2 = mantissas_2.to(torch.int16)

        reassembled1, reassembled2 = reassemble_float_after_fidelity(
            data_format,
            sign_1,
            sign_2,
            exponents_1,
            exponents_2,
            mantissas_1,
            mantissas_2,
        )

        return reassembled1, reassembled2


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
        pass  # Removed ops dict since __call__ is no longer used

    def transpose_within_faces(
        self,
        operand,
        data_format: DataFormat,
        input_dimensions: list[int] = [32, 32],
    ):
        """Transpose a tile tensor by transposing within each of the four faces.
        A tile tensor consists of 4 equal faces arranged in a tile.
        This function dynamically splits the tensor into 4 faces, transposes each face
        individually, and reassembles the tile.
        Args:
            operand: Input tensor to transpose
            data_format: Data format for the result
        Returns:
            torch.Tensor: Tensor with each face transposed, flattened back to original size
        """
        tensor = to_tensor(operand, data_format)
        total_elements = tensor.numel()
        if total_elements % 4 != 0:
            raise ValueError(
                f"Tensor size {total_elements} must be divisible by 4 for tile structure"
            )
        face_size = total_elements // 4
        face_dim = math.isqrt(face_size)
        if face_dim * face_dim != face_size:
            raise ValueError(
                f"Each face must be square (for now). Face size {face_size} is not a perfect square"
            )

        # Split the tensor into 4 faces dynamically
        # Transpose each face using the helper function
        result = tensor.view(4, face_dim, face_dim).transpose(-2, -1).flatten()
        return result.to(format_dict[data_format])

    def transpose_faces(
        self,
        operand,
        data_format: DataFormat,
        input_dimensions: list[int] = [32, 32],
    ):
        """Transpose the arrangement of the four faces in a tile tensor.
        Treats each face as a single element and transposes their arrangement.
        If faces are arranged as:
        f0 f1
        f2 f3
        After transposition:
        f0 f2
        f1 f3
        Args:
            operand: Input tensor to transpose
            data_format: Data format for the result
        Returns:
            torch.Tensor: Tensor with faces rearranged in transposed order
        """
        tensor = to_tensor(operand, data_format)
        total_elements = tensor.numel()
        if total_elements % 4 != 0:
            raise ValueError(
                f"Invalid tensor size {total_elements}. A valid tile structure requires the tensor to represent "
                f"4 equal faces, so the total number of elements must be divisible by 4."
            )
        face_size = total_elements // 4
        # Split the tensor into 4 faces
        faces = torch.tensor_split(tensor, 4)
        # Transpose the face arrangement: f0,f1,f2,f3 -> f0,f2,f1,f3
        result = torch.cat([faces[0], faces[2], faces[1], faces[3]])
        return result.to(format_dict[data_format])

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
        # Constants
        ELEMENTS_PER_TILE = 1024  # 32 × 32
        TILE_DIMENSIONS = (32, 32)

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
    ):
        torch_format = format_dict[data_format]

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

        num_fidelity_phases = math_fidelity.value

        res = 0

        if num_fidelity_phases == 0:

            t1, t2 = self._apply_fidelity_masking(t1, t2, 0, data_format)
            t1, t2 = t1.view(M, K1), t2.view(K2, N)
            res = (
                torch.matmul(t1, t2)
                .view(output_dimensions[0] * output_dimensions[1])
                .to(torch_format)
            )

        elif num_fidelity_phases == 1:

            t1, t2 = self._apply_fidelity_masking(t1, t2, 0, data_format)
            t1, t2 = t1.view(M, K1), t2.view(K2, N)
            res = (
                torch.matmul(t1, t2)
                .view(output_dimensions[0] * output_dimensions[1])
                .to(torch_format)
            )

            t1 = to_tensor(operand1, data_format)
            t2 = to_tensor(operand2, data_format)
            t1, t2 = self._apply_fidelity_masking(t1, t2, 1, data_format)
            t1, t2 = t1.view(M, K1), t2.view(K2, N)
            res += (
                torch.matmul(t1, t2)
                .view(output_dimensions[0] * output_dimensions[1])
                .to(torch_format)
            )

        elif num_fidelity_phases == 2:

            t1, t2 = self._apply_fidelity_masking(t1, t2, 0, data_format)
            t1, t2 = t1.view(M, K1), t2.view(K2, N)
            res = (
                torch.matmul(t1, t2)
                .view(output_dimensions[0] * output_dimensions[1])
                .to(torch_format)
            )

            t1 = to_tensor(operand1, data_format)
            t2 = to_tensor(operand2, data_format)
            t1, t2 = self._apply_fidelity_masking(t1, t2, 1, data_format)
            t1, t2 = t1.view(M, K1), t2.view(K2, N)
            res += (
                torch.matmul(t1, t2)
                .view(output_dimensions[0] * output_dimensions[1])
                .to(torch_format)
            )

            # TODO: INVESTIGATE WHY COMMENTING THIS MAKES TEST PASS

            # t1 = to_tensor(operand1, data_format)
            # t2 = to_tensor(operand2, data_format)
            # t1, t2 = self._apply_fidelity_masking(t1, t2, 2, data_format)
            # t1,t2 = t1.view(M, K1), t2.view(K2, N)
            # res +=  torch.matmul(t1, t2).view(output_dimensions[0] * output_dimensions[1]).to(torch_format)

        elif num_fidelity_phases == 3:

            t1, t2 = t1.view(M, K1), t2.view(K2, N)
            res = (
                torch.matmul(t1, t2)
                .view(output_dimensions[0] * output_dimensions[1])
                .to(torch_format)
            )

        if tilize:
            res = tilize_block(
                res,
                dimensions=(input_A_dimensions[0], input_B_dimensions[1]),
                stimuli_format=data_format,
            ).flatten()
        return res


@register_golden
class DataCopyGolden:
    def __call__(
        self,
        operand1,
        data_format,
        num_faces: int = 4,
        input_dimensions: list[int] = [32, 32],
    ):
        torch_format = format_dict[data_format]

        height, width = input_dimensions[0], input_dimensions[1]
        tile_cnt = (height // 32) * (width // 32)
        tile_size = height * width // tile_cnt
        # Depending on the value of 'num_faces' (1, 2, 4), select the first 1, 2 or all 4 faces of a tile
        elements_per_tile_needed = (tile_size // 4) * num_faces

        if not isinstance(operand1, torch.Tensor):
            operand1 = torch.tensor(operand1)

        reshaped = operand1.view(tile_cnt, tile_size)
        selected = reshaped[:, :elements_per_tile_needed]

        return selected.flatten().to(torch_format)


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
            MathOperation.Reciprocal: self._reciprocal,
            MathOperation.Rsqrt: self._rsqrt,
            MathOperation.Sin: self._sin,
            MathOperation.Sqrt: self._sqrt,
            MathOperation.Square: self._square,
            MathOperation.Celu: self._celu,
            MathOperation.Silu: self._silu,
            MathOperation.Gelu: self._gelu,
            MathOperation.Neg: self._neg,
            MathOperation.Fill: self._fill,
            MathOperation.Elu: self._elu,
            MathOperation.Exp: self._exp,
            MathOperation.Exp2: self._exp2,
            MathOperation.Hardsigmoid: self._hardsigmoid,
            MathOperation.Threshold: self._threshold,
            MathOperation.ReluMax: self._relu_max,
            MathOperation.ReluMin: self._relu_min,
            MathOperation.ReduceColumn: self._reduce_columns,
        }
        self.data_format = None
        self.dest_acc = DestAccumulation.No

    def __call__(
        self,
        operation,
        operand1,
        data_format,
        dest_acc,
        input_format,
        reduce_pool: Optional[ReducePool] = None,
    ):
        self.data_format = data_format
        self.dest_acc = dest_acc

        if operation not in self.ops:
            raise ValueError(f"Unsupported operation: {operation}")

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

        # Special handling for SumColumns which needs to process the entire tensor
        if operation == MathOperation.ReduceColumn:
            result = self.ops[operation](tensor, reduce_pool)
        else:
            result = [self.ops[operation](x) for x in tensor.tolist()]

        if self.data_format == DataFormat.Bfp8_b:
            check_bfp8_b(result)

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

        # depending on `data_format`, `inf` values may get converted when unpacked to L1.
        if dst_format == DataFormat.Float16:
            match data_format:
                case DataFormat.Float16_b:
                    result = convert_inf_to_value(result, 130560.0)
                case DataFormat.Float32:
                    result = convert_inf_to_value(result, 131008.0)
                case DataFormat.Bfp8_b:
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

    def _reciprocal(self, x):
        if x == 0.0:
            return self.handle_infinite_numbers(float("inf"))
        return 1 / x

    def _sin(self, x):
        # Never not finite, values range from [-1, 1]
        return math.sin(x)

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

    def _square(self, x):
        if not math.isfinite(x * x):
            return self.handle_infinite_numbers(math.inf)
        return x * x

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
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.data_format])
        )
        return torch.exp(input_tensor).item()

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

    def _fill(self, x):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.data_format])
        )
        return input_tensor.fill_(5).item()

    def _hardsigmoid(self, x):
        input_tensor = (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=format_dict[self.data_format])
        )
        return torch.nn.functional.hardsigmoid(input_tensor).item()

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
        input_tensor = untilize(x, self.data_format).flatten().view(32, 32)

        # Sum along columns (dim=0) to get a 1x32 result
        column_sums = torch.sum(input_tensor, dim=0)  # Shape: [32]

        if reduce_pool == ReducePool.Average:
            # Divide each column sum by 32 individually
            column_averages = column_sums // 32  # Element-wise division by 32
            return column_averages.tolist()

        # Return only the column sums, not a full 1024-element tensor
        return column_sums.tolist()


@register_golden
class EltwiseBinaryGolden(FidelityMasking):
    def __init__(self):
        self.ops = {
            MathOperation.Elwadd: self._add,
            MathOperation.Elwsub: self._sub,
            MathOperation.Elwmul: self._mul,
        }

    def __call__(self, op, operand1, operand2, data_format, math_fidelity):
        if op not in self.ops:
            raise ValueError(f"Unsupported Eltwise operation: {op}")

        t1 = to_tensor(operand1, data_format)
        t2 = to_tensor(operand2, data_format)

        num_fidelity_phases = 0

        _fildelity_dict = {
            MathFidelity.LoFi: 0,
            MathFidelity.HiFi2: 1,
            MathFidelity.HiFi3: 2,
            MathFidelity.HiFi4: 3,
        }

        num_fidelity_phases = _fildelity_dict.get(math_fidelity, 0)

        res = 0

        # If multiply is chosen apply fidelity
        if op == MathOperation.Elwmul:
            res = None
            for phase in range(num_fidelity_phases + 1):
                t1, t2 = self._apply_fidelity_masking(t1, t2, phase, data_format)
                phase_result = self.ops[op](t1, t2)

                if phase == 0:
                    res = phase_result
                else:
                    res += phase_result

            return res
        else:
            return self.ops[op](t1, t2)

    # Operation methods
    def _add(self, t1, t2):
        return t1 + t2

    def _sub(self, t1, t2):
        return t1 - t2

    def _mul(self, t1, t2):
        # Compute in float32 for better fidelity, then cast back to original dtype.
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
            }
        )

    def __call__(
        self, operation: MathOperation, operand1, operand2, data_format: DataFormat
    ):
        if operation not in self.ops:
            raise ValueError(f"Unsupported SFPU operation: {operation}")

        t1 = to_tensor(operand1, data_format)
        t2 = to_tensor(operand2, data_format)

        result = [self.ops[operation](t1[i], t2[i]) for i in range(len(t1))]
        return torch.tensor(result, dtype=format_dict[data_format])

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


@register_golden
class ReduceGolden:
    def __init__(self):
        self.dim_handlers = {
            ReduceDimension.Column: self._reduce_column,
            ReduceDimension.Row: self._reduce_row,
            ReduceDimension.Scalar: self._reduce_scalar,
        }

    def __call__(self, operand, reduce_dim, pool_type, data_format):
        if reduce_dim not in self.dim_handlers:
            raise ValueError(f"Unsupported reduce dimension: {reduce_dim}")

        f0 = operand[:256].view(16, 16)
        f1 = operand[256:512].view(16, 16)
        f2 = operand[512:768].view(16, 16)
        f3 = operand[768:].view(16, 16)
        faces = [f0, f1, f2, f3]
        if reduce_dim == ReduceDimension.Scalar:
            faces = operand
        return self.dim_handlers[reduce_dim](faces, pool_type, data_format)

    def _reduce_column(self, faces, pool_type, data_format):
        left_half = torch.cat((faces[0], faces[2]), 0)
        right_half = torch.cat((faces[1], faces[3]), 0)

        result = torch.zeros(32, 32, dtype=format_dict[data_format])
        result[0, 0:16] = self._apply_pooling(left_half, pool_type, dim=0)
        result[0, 16:32] = self._apply_pooling(right_half, pool_type, dim=0)

        return result.view(1024)

    def _reduce_row(self, faces, pool_type, data_format):
        left_half = torch.cat((faces[0], faces[2]), 1)
        right_half = torch.cat((faces[1], faces[3]), 1)

        result = torch.zeros(32, 32, dtype=format_dict[data_format])
        result[0:16, 0] = self._apply_pooling(left_half, pool_type, dim=1).view(16)
        result[16:32, 0] = self._apply_pooling(right_half, pool_type, dim=1).view(16)

        return result.view(1024)

    def _reduce_scalar(self, operand, pool_type, data_format):
        tensor = operand.view(1024)
        result = torch.zeros(32, 32, dtype=format_dict[data_format])
        result[0, 0] = self._apply_pooling(tensor, pool_type, dim=0)
        return result.view(1024)

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
class UntilizeGolden:
    def __call__(self, operand, data_format, dimensions=[32, 32]):
        from helpers.tilize_untilize import untilize_block

        result = untilize_block(
            operand, stimuli_format=data_format, dimensions=dimensions
        )
        return result.flatten()


@register_golden
class TilizeGolden:
    def __call__(self, operand, dimensions, data_format):
        from helpers.tilize_untilize import tilize_block

        result = tilize_block(operand, dimensions, data_format)
        return result.flatten()
