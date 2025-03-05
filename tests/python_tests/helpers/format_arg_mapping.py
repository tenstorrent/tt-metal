# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

format_dict = {
    "Float32": torch.float32,
    "Float16": torch.float16,
    "Float16_b": torch.bfloat16,
    "Int32": torch.int32,
}

format_args_dict = {
    "Float32": "FORMAT_FLOAT32",
    "Float16": "FORMAT_FLOAT16",
    "Float16_b": "FORMAT_FLOAT16_B",
    "Bfp8_b": "FORMAT_BFP8_B",
    "Int32": "FORMAT_INT32",
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

format_sizes = {
    "Float16": 512,
    "Float16_b": 512,
    "Bfp8_b": 272,
    "Float32": 1024,
    "Int32": 1024,
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
