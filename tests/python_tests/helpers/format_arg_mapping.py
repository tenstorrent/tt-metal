# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

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
}


class MathOperation(Enum):
    """
    An enumeration class that holds all the math operations supported by the LLKs.
    Used to avoid hardcoding the operation strings in the test scripts using strings. This avoid typos and future errors.
    MathOperations(Enum) class instances can be compared via unique values.
    When you have a set of related constants and you want to leverage the benefits of enumeration (unique members, comparisons, introspection, etc.).
    It's a good choice for things like state machines, categories, or settings where values should not be changed or duplicated.
    """

    Elwadd = "ELTWISE_BINARY_ADD"
    Elwsub = "ELTWISE_BINARY_SUB"
    Elwmul = "ELTWISE_BINARY_MUL"
    Abs = "SFPU_OP_ABS"
    Sqrt = "SFPU_OP_SQRT"
    Square = "SFPU_OP_SQUARE"
    Log = "SFPU_OP_LOG"
    Sin = "SFPU_OP_SINE"
    Cos = "SFPU_OP_COSINE"
    Reciprocal = "SFPU_OP_RECIPROCAL"
    Celu = "SFPU_OP_CELU"
    ReduceColumn = "REDUCE_COL_OPERATION"
    ReduceRow = "REDUCE_ROW_OPERATION"
    ReduceScalar = "REDUCE_SCALAR_OPERATION"
    SfpuElwadd = "SFPU_ELWADD"
    SfpuElwsub = "SFPU_ELWSUB"
    SfpuElwmul = "SFPU_ELWMUL"
    SfpuXlogy = "SFPU_OP_XLOGY"
    SfpuElwRightShift = "SFPU_OP_RSHFT"
    SfpuElwLeftShift = "SFPU_OP_LSHFT"
    SfpuElwLogicalRightShift = "SFPU_OP_LOGICAL_RSHFT"
    Silu = "SFPU_OP_SILU"
    Gelu = "SFPU_OP_GELU"
    Neg = "SFPU_OP_NEG"


class ReduceDimension(Enum):
    Column = "ReduceDim::REDUCE_COL"
    Row = "ReduceDim::REDUCE_ROW"
    Scalar = "ReduceDim::REDUCE_SCALAR"
    No = " "


class ReducePool(Enum):
    Max = "PoolType::MAX"
    Sum = "PoolType::SUM"
    Average = "PoolType::AVG"
    No = " "


class DestAccumulation(Enum):
    Yes = "DEST_ACC"
    No = ""


class ApproximationMode(Enum):
    Yes = "true"
    No = "false"


class MathFidelity(Enum):
    LoFi = 0
    HiFi2 = 2
    HiFi3 = 3
    HiFi4 = 4
    Invalid = 5


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
}


class L1BufferLocations(Enum):
    srcA = 0x18FE0
    srcB = 0x18FE4
    Result = 0x18FE8
