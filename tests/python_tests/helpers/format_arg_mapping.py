# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from enum import Enum
from .format_config import DataFormat


format_dict = {
    DataFormat.Float32: torch.float32,
    DataFormat.Float16: torch.float16,
    DataFormat.Float16_b: torch.bfloat16,
    DataFormat.Int32: torch.int32,
}

unpack_A_src_dict = {
    DataFormat.Float32: "UNPACK_A_SRC_FLOAT32",
    DataFormat.Float16: "UNPACK_A_SRC_FLOAT16",
    DataFormat.Float16_b: "UNPACK_A_SRC_FLOAT16_B",
    DataFormat.Bfp8_b: "UNPACK_A_SRC_BFP8_B",
    DataFormat.Int32: "UNPACK_A_SRC_INT32",
}

unpack_A_dst_dict = {
    DataFormat.Float32: "UNPACK_A_DST_FLOAT32",
    DataFormat.Float16: "UNPACK_A_DST_FLOAT16",
    DataFormat.Float16_b: "UNPACK_A_DST_FLOAT16_B",
    DataFormat.Bfp8_b: "UNPACK_A_DST_BFP8_B",
    DataFormat.Int32: "UNPACK_A_DST_INT32",
}

unpack_B_src_dict = {
    DataFormat.Float32: "UNPACK_B_SRC_FLOAT32",
    DataFormat.Float16: "UNPACK_B_SRC_FLOAT16",
    DataFormat.Float16_b: "UNPACK_B_SRC_FLOAT16_B",
    DataFormat.Bfp8_b: "UNPACK_B_SRC_BFP8_B",
    DataFormat.Int32: "UNPACK_B_SRC_INT32",
}

unpack_B_dst_dict = {
    DataFormat.Float32: "UNPACK_B_DST_FLOAT32",
    DataFormat.Float16: "UNPACK_B_DST_FLOAT16",
    DataFormat.Float16_b: "UNPACK_B_DST_FLOAT16_B",
    DataFormat.Bfp8_b: "UNPACK_B_DST_BFP8_B",
    DataFormat.Int32: "UNPACK_B_DST_INT32",
}

math_dict = {
    DataFormat.Float32: "MATH_FLOAT32",
    DataFormat.Float16: "MATH_FLOAT16",
    DataFormat.Float16_b: "MATH_FLOAT16_B",
    DataFormat.Bfp8_b: "MATH_BFP8_B",
    DataFormat.Int32: "MATH_INT32",
}

pack_src_dict = {
    DataFormat.Float32: "PACK_SRC_FLOAT32",
    DataFormat.Float16: "PACK_SRC_FLOAT16",
    DataFormat.Float16_b: "PACK_SRC_FLOAT16_B",
    DataFormat.Bfp8_b: "PACK_SRC_BFP8_B",
    DataFormat.Int32: "PACK_SRC_INT32",
}

pack_dst_dict = {
    DataFormat.Float32: "PACK_DST_FLOAT32",
    DataFormat.Float16: "PACK_DST_FLOAT16",
    DataFormat.Float16_b: "PACK_DST_FLOAT16_B",
    DataFormat.Bfp8_b: "PACK_DST_BFP8_B",
    DataFormat.Int32: "PACK_DST_INT32",
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
    Sqrt = "SFPU_OP_SQRT"
    Square = "SFPU_OP_SQUARE"
    Log = "SFPU_OP_LOG"
    ReduceColumn = "REDUCE_COL_OPERATION"
    ReduceRow = "REDUCE_ROW_OPERATION"
    ReduceScalar = "REDUCE_SCALAR_OPERATION"


class ReduceDimension(Enum):
    Column = "ReduceDim::REDUCE_COL"
    Row = "ReduceDim::REDUCE_ROW"
    Scalar = "ReduceDim::REDUCE_SCALAR"
    No = " "


class ReducePool(Enum):
    Max = "PoolType::MAX"
    Sum = "PoolType::SUM"
    Average = "PoolType::AVG"


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


class TileCount(Enum):
    One = 1
    Two = 2
    Three = 3
    Four = 4


class Mailbox(Enum):
    Unpacker = 0x19FFC
    Math = 0x19FF8
    Packer = 0x19FF4
