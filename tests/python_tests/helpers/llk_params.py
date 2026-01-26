# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from enum import Enum, auto

import torch

from .format_config import DataFormat

format_dict = {
    DataFormat.Float32: torch.float32,
    DataFormat.Float16: torch.float16,
    DataFormat.Float16_b: torch.bfloat16,
    DataFormat.Bfp8_b: torch.bfloat16,  # BFP8 not native to PyTorch, is represented as bfloat16
    DataFormat.Int32: torch.int32,
    DataFormat.UInt32: torch.int64,
    DataFormat.UInt16: torch.int32,
    DataFormat.Int8: torch.int8,
    DataFormat.UInt8: torch.uint8,
    DataFormat.MxFp8R: torch.bfloat16,
    DataFormat.MxFp8P: torch.bfloat16,
}


class MathOpType(Enum):
    """Enum for different types of math operations."""

    SFPU_UNARY = auto()
    SFPU_BINARY = auto()
    SFPU_TERNARY = auto()

    FPU_BINARY = auto()
    REDUCE = auto()


# Operation specification
OpSpec = namedtuple("OpSpec", ["cpp_enum_value", "operation_type"])


class MathOperation(Enum):
    """
    An enumeration class that holds all the math operations supported by the LLKs.
    Each enum value is an OpSpec tuple containing (cpp_enum_value, operation_type).

    Operations are organized by type:
    - FPU_BINARY: Floating Point Unit binary operations
    - SFPU_UNARY: Special Function Processing Unit unary operations
    - SFPU_BINARY: Special Function Processing Unit binary operations
    - REDUCE: Reduction operations
    """

    # =============================================================================
    # FPU BINARY OPERATIONS
    # =============================================================================
    Elwadd = OpSpec("ELWADD", MathOpType.FPU_BINARY)
    Elwmul = OpSpec("ELWMUL", MathOpType.FPU_BINARY)
    Elwsub = OpSpec("ELWSUB", MathOpType.FPU_BINARY)

    # =============================================================================
    # SFPU UNARY OPERATIONS
    # =============================================================================
    Abs = OpSpec("abs", MathOpType.SFPU_UNARY)
    Atanh = OpSpec("atanh", MathOpType.SFPU_UNARY)
    Asinh = OpSpec("asinh", MathOpType.SFPU_UNARY)
    Acosh = OpSpec("acosh", MathOpType.SFPU_UNARY)
    Celu = OpSpec("celu", MathOpType.SFPU_UNARY)
    Cos = OpSpec("cosine", MathOpType.SFPU_UNARY)
    Elu = OpSpec("elu", MathOpType.SFPU_UNARY)
    Exp = OpSpec("exponential", MathOpType.SFPU_UNARY)
    Exp2 = OpSpec("exp2", MathOpType.SFPU_UNARY)
    Fill = OpSpec("fill", MathOpType.SFPU_UNARY)
    Gelu = OpSpec("gelu", MathOpType.SFPU_UNARY)
    Hardsigmoid = OpSpec("hardsigmoid", MathOpType.SFPU_UNARY)
    Log = OpSpec("log", MathOpType.SFPU_UNARY)
    Log1p = OpSpec("log1p", MathOpType.SFPU_UNARY)
    Neg = OpSpec("neg", MathOpType.SFPU_UNARY)
    Reciprocal = OpSpec("reciprocal", MathOpType.SFPU_UNARY)
    Relu = OpSpec("relu", MathOpType.SFPU_UNARY)
    Rsqrt = OpSpec("rsqrt", MathOpType.SFPU_UNARY)
    Sin = OpSpec("sine", MathOpType.SFPU_UNARY)
    Silu = OpSpec("silu", MathOpType.SFPU_UNARY)
    Sqrt = OpSpec("sqrt", MathOpType.SFPU_UNARY)
    Square = OpSpec("square", MathOpType.SFPU_UNARY)
    Tanh = OpSpec("tanh", MathOpType.SFPU_UNARY)
    Threshold = OpSpec("threshold", MathOpType.SFPU_UNARY)
    ReluMax = OpSpec(
        "relu_max", MathOpType.SFPU_UNARY
    )  # ReLU_max(x, U) = max(0, min(x, U))
    ReluMin = OpSpec("relu_min", MathOpType.SFPU_UNARY)  # ReLU_min(x, L) = max(x, L)
    TopKLocalSort = OpSpec("topk_local_sort", MathOpType.SFPU_UNARY)
    TopKMerge = OpSpec("topk_merge", MathOpType.SFPU_UNARY)
    TopKRebuild = OpSpec("topk_rebuild", MathOpType.SFPU_UNARY)
    # =============================================================================
    # SFPU BINARY OPERATIONS
    # =============================================================================
    SfpuElwadd = OpSpec("ADD", MathOpType.SFPU_BINARY)
    SfpuElwLeftShift = OpSpec("LSHFT", MathOpType.SFPU_BINARY)
    SfpuElwLogicalRightShift = OpSpec("LOGICAL_RSHFT", MathOpType.SFPU_BINARY)
    SfpuElwmul = OpSpec("MUL", MathOpType.SFPU_BINARY)
    SfpuElwRightShift = OpSpec("RSHFT", MathOpType.SFPU_BINARY)
    SfpuElwsub = OpSpec("SUB", MathOpType.SFPU_BINARY)
    SfpuXlogy = OpSpec("XLOGY", MathOpType.SFPU_BINARY)
    SfpuAddTopRow = OpSpec("ADD_TOP_ROW", MathOpType.SFPU_BINARY)
    SfpuElwdiv = OpSpec("DIV", MathOpType.SFPU_BINARY)
    SfpuElwrsub = OpSpec("RSUB", MathOpType.SFPU_BINARY)
    SfpuElwpow = OpSpec("POW", MathOpType.SFPU_BINARY)

    # =============================================================================
    # SFPU TERNARY OPERATIONS
    # =============================================================================
    SfpuWhere = OpSpec("WHERE", MathOpType.SFPU_TERNARY)
    # Alias maintained for backward compatibility with older test cases
    TTNNWhere = SfpuWhere

    # =============================================================================
    # REDUCE OPERATIONS
    # =============================================================================
    ReduceColumn = OpSpec("REDUCE_COL", MathOpType.REDUCE)
    ReduceRow = OpSpec("REDUCE_ROW", MathOpType.REDUCE)
    ReduceScalar = OpSpec("REDUCE_SCALAR", MathOpType.REDUCE)

    # =============================================================================
    # PROPERTIES AND UTILITY METHODS
    # =============================================================================
    @property
    def cpp_enum_value(self):
        """Get the C++ enum value for this operation."""
        return self.value.cpp_enum_value

    @property
    def operation_type(self):
        """Get the operation type for this operation."""
        return self.value.operation_type

    @classmethod
    def _get_operations_by_type(cls, op_type: MathOpType):
        """Get all operations of a specific type."""
        return {op for op in cls if op.operation_type == op_type}

    @classmethod
    def get_fpu_binary_operations(cls):
        """Get all FPU binary operations."""
        return cls._get_operations_by_type(MathOpType.FPU_BINARY)

    @classmethod
    def get_sfpu_unary_operations(cls):
        """Get all SFPU unary operations."""
        return cls._get_operations_by_type(MathOpType.SFPU_UNARY)

    @classmethod
    def get_sfpu_binary_operations(cls):
        """Get all SFPU binary operations."""
        return cls._get_operations_by_type(MathOpType.SFPU_BINARY)

    @classmethod
    def get_reduce_operations(cls):
        """Get all reduce operations."""
        return cls._get_operations_by_type(MathOpType.REDUCE)


SFPU_UNARY_OPERATIONS = MathOperation.get_sfpu_unary_operations()
SFPU_BINARY_OPERATIONS = MathOperation.get_sfpu_binary_operations()
FPU_BINARY_OPERATIONS = MathOperation.get_fpu_binary_operations()
REDUCE_OPERATIONS = MathOperation.get_reduce_operations()


class ReduceDimension(Enum):
    Column = auto()
    Row = auto()
    Scalar = auto()


class ReducePool(Enum):
    Max = "MAX"
    Min = "MIN"
    Sum = "SUM"
    Average = "AVG"


class DestAccumulation(Enum):
    Yes = "true"
    No = "false"


class StochasticRounding(Enum):
    No = "StochRndType::None"
    Fpu = "StochRndType::Fpu"
    Pack = "StochRndType::Pack"
    All = "StochRndType::All"


class PackerReluType(Enum):
    """
    Relu activation function types for packer operations.
    """

    NoRelu = 0
    ZeroRelu = 1
    MinThresholdRelu = 2
    MaxThresholdRelu = 3

    def __str__(self):
        match self:
            case PackerReluType.NoRelu:
                return "NO_RELU"
            case PackerReluType.ZeroRelu:
                return "ZERO_RELU"
            case PackerReluType.MinThresholdRelu:
                return "MIN_THRESHOLD_RELU"
            case PackerReluType.MaxThresholdRelu:
                return "MAX_THRESHOLD_RELU"
            case _:
                raise ValueError(f"Unsupported PackerReluType: {self!r}")


class Haloize(Enum):
    Yes = "true"
    No = "false"


class ApproximationMode(Enum):
    Yes = "true"
    No = "false"


class Transpose(Enum):
    Yes = True
    No = False


class MathFidelity(Enum):
    LoFi = 0
    HiFi2 = 2
    HiFi3 = 3
    HiFi4 = 4


class NarrowTile(Enum):
    Yes = True
    No = False


class DestSync(Enum):
    Half = 0
    Full = 1


class Tilize(Enum):
    Yes = True
    No = False


class FastMode(Enum):
    Yes = "true"
    No = "false"


class StableSort(Enum):
    Yes = "true"
    No = "false"


class Mailbox(Enum):
    Unpacker = 0x19FFC
    Math = 0x19FF8
    Packer = 0x19FF4


format_tile_sizes = {
    DataFormat.Bfp8_b: 1088,
    DataFormat.Float16: 2048,
    DataFormat.Float16_b: 2048,
    DataFormat.Float32: 4096,
    DataFormat.Int32: 4096,
    DataFormat.Tf32: 3072,  # 3 bytes * 1024 elements
    DataFormat.UInt32: 4096,
    DataFormat.UInt16: 2048,
    DataFormat.Int8: 1024,  # 1 byte * 1024 elements
    DataFormat.UInt8: 1024,  # 1 byte * 1024 elements
    # MX formats: 1 byte per element + 1 scale (8 bits) per 32 elements
    # 1024 elements = 32 blocks × (1 scale + 32 elements) = 1056 bytes
    DataFormat.MxFp8R: 1056,
    DataFormat.MxFp8P: 1056,
}


class DstSync(Enum):
    """Destination synchronization mode for LLK operations."""

    SyncHalf = "SyncHalf"
    SyncFull = "SyncFull"


class BroadcastType(Enum):
    """
    Enum for broadcast types in LLK kernels.
    """

    None_ = "NONE"
    Column = "COL"
    Row = "ROW"
    Scalar = "SCALAR"


class EltwiseBinaryReuseDestType(Enum):
    """
    Enum for destination reuse types in elementwise binary ops.
    """

    NONE = "NONE"
    DEST_TO_SRCA = "DEST_TO_SRCA"
    DEST_TO_SRCB = "DEST_TO_SRCB"


class DataCopyType(Enum):
    A2D = "A2D"
    B2D = "B2D"


class PerfRunType(Enum):
    L1_TO_L1 = 1
    UNPACK_ISOLATE = 2
    MATH_ISOLATE = 3
    PACK_ISOLATE = 4
    L1_CONGESTION = 5


# ******** QUASAR specific ********
class ImpliedMathFormat(Enum):
    No = "false"
    Yes = "true"


class UnpackerEngine(Enum):
    """
    Enum for unpacker engine selection.
    """

    UnpA = "UNP_A"
    UnpB = "UNP_B"
    UnpS = "UNP_S"
    UnpDest = "UNP_DEST"


class ReluConfig(Enum):
    NoRelu = 0
    ZeroRelu = 1


# *********************************
