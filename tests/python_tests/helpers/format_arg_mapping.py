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


mathop_args_dict = {
    "elwadd": "ELTWISE_BINARY_ADD",
    "elwsub": "ELTWISE_BINARY_SUB",
    "elwmul": "ELTWISE_BINARY_MUL",
    "sqrt": "SFPU_OP_SQRT",
    "square": "SFPU_OP_SQUARE",
    "log": "SFPU_OP_LOG",
    "reduce_col": "REDUCE_COL_OPERATION",
    "reduce_row": "REDUCE_ROW_OPERATION",
    "reduce_scalar": "REDUCE_SCALAR_OPERATION",
}

unpack_src_dict = {
    DataFormat.Float32: "UNPACK_SRC_FLOAT32",
    DataFormat.Float16: "UNPACK_SRC_FLOAT16",
    DataFormat.Float16_b: "UNPACK_SRC_FLOAT16_B",
    DataFormat.Bfp8_b: "UNPACK_SRC_BFP8_B",
    DataFormat.Int32: "UNPACK_SRC_INT32",
}

unpack_dst_dict = {
    DataFormat.Float32: "UNPACK_DST_FLOAT32",
    DataFormat.Float16: "UNPACK_DST_FLOAT16",
    DataFormat.Float16_b: "UNPACK_DST_FLOAT16_B",
    DataFormat.Bfp8_b: "UNPACK_DST_BFP8_B",
    DataFormat.Int32: "UNPACK_DST_INT32",
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

format_sizes = {
    DataFormat.Float32: 1024,
    DataFormat.Float16: 512,
    DataFormat.Float16_b: 512,
    DataFormat.Bfp8_b: 272,
    DataFormat.Int32: 1024,
}

reduce_dim_args = {
    "reduce_col": "ReduceDim::REDUCE_COL",
    "reduce_row": "ReduceDim::REDUCE_ROW",
    "reduce_scalar": "ReduceDim::REDUCE_SCALAR",
    "no_reduce_dim": " ",
}

reduce_pool_args = {
    "max": "PoolType::MAX",
    "sum": "PoolType::SUM",
    "avg": "PoolType::AVG",
}
