# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any, Dict, Type

import yaml
from helpers.format_config import DataFormat
from helpers.fused_fpu import (
    DatacopyFpu,
    EltwiseFpu,
    MatmulFpu,
    ReduceBlockMaxFpu,
    ReduceFpu,
)
from helpers.fused_math import ComputeNode, ComputePipeline
from helpers.fused_operand import OperandRegistry
from helpers.fused_operation import FusedOperation
from helpers.fused_packer import Packer
from helpers.fused_sfpu import BinarySfpu, UnarySfpu
from helpers.fused_unpacker import (
    MatmulUnpacker,
    ReduceBlockMaxUnpacker,
    Unpacker,
    UnpackerA,
    UnpackerAB,
    UnpackerTilizeA,
)
from helpers.llk_params import (
    ApproximationMode,
    BroadcastType,
    DestSync,
    EltwiseBinaryReuseDestType,
    MathOperation,
    ReducePool,
    Transpose,
)

from .fuser_config import FuserConfig, GlobalConfig
from .llk_params import DestAccumulation, MathFidelity

FUSER_CONFIG_DIR = (
    Path(os.environ.get("LLK_HOME")) / "tests" / "python_tests" / "fuser_config"
)

UNPACKER_MAP: Dict[str, Type[Unpacker]] = {
    "UnpackerA": UnpackerA,
    "UnpackerAB": UnpackerAB,
    "UnpackerTilizeA": UnpackerTilizeA,
    "MatmulUnpacker": MatmulUnpacker,
    "ReduceBlockMaxUnpacker": ReduceBlockMaxUnpacker,
}

PACKER_MAP: Dict[str, Type[Packer]] = {
    "Packer": Packer,
}

DATA_FORMAT_MAP: Dict[str, DataFormat] = {
    "Float16_b": DataFormat.Float16_b,
    "Float16": DataFormat.Float16,
    "Float32": DataFormat.Float32,
    "Bfp8_b": DataFormat.Bfp8_b,
}

MATH_FIDELITY_MAP: Dict[str, MathFidelity] = {
    "LoFi": MathFidelity.LoFi,
    "HiFi2": MathFidelity.HiFi2,
    "HiFi3": MathFidelity.HiFi3,
    "HiFi4": MathFidelity.HiFi4,
}

DEST_ACCUMULATION_MAP: Dict[str, DestAccumulation] = {
    "Yes": DestAccumulation.Yes,
    "No": DestAccumulation.No,
}

DEST_SYNC_MAP: Dict[str, DestSync] = {
    "Full": DestSync.Full,
    "Half": DestSync.Half,
}

TRANSPOSE_MAP: Dict[str, Transpose] = {
    "Yes": Transpose.Yes,
    "No": Transpose.No,
}

FPU_OPERATION_MAP: Dict[str, MathOperation] = {
    "Elwadd": MathOperation.Elwadd,
    "Elwmul": MathOperation.Elwmul,
    "Elwsub": MathOperation.Elwsub,
}

SFPU_UNARY_OPERATION_MAP: Dict[str, MathOperation] = {
    "Abs": MathOperation.Abs,
    "Acosh": MathOperation.Acosh,
    "Asinh": MathOperation.Asinh,
    "Atanh": MathOperation.Atanh,
    "Celu": MathOperation.Celu,
    "Cos": MathOperation.Cos,
    "Elu": MathOperation.Elu,
    "Exp": MathOperation.Exp,
    "Exp2": MathOperation.Exp2,
    "Fill": MathOperation.Fill,
    "Gelu": MathOperation.Gelu,
    "Hardsigmoid": MathOperation.Hardsigmoid,
    "Log": MathOperation.Log,
    "Log1p": MathOperation.Log1p,
    "Neg": MathOperation.Neg,
    "Reciprocal": MathOperation.Reciprocal,
    "ReluMax": MathOperation.ReluMax,
    "ReluMin": MathOperation.ReluMin,
    "Rsqrt": MathOperation.Rsqrt,
    "Silu": MathOperation.Silu,
    "Sin": MathOperation.Sin,
    "Sqrt": MathOperation.Sqrt,
    "Square": MathOperation.Square,
    "Tanh": MathOperation.Tanh,
    "Threshold": MathOperation.Threshold,
}

SFPU_BINARY_OPERATION_MAP: Dict[str, MathOperation] = {
    "SfpuElwadd": MathOperation.SfpuElwadd,
    "SfpuElwmul": MathOperation.SfpuElwmul,
    "SfpuElwsub": MathOperation.SfpuElwsub,
    "SfpuElwLeftShift": MathOperation.SfpuElwLeftShift,
    "SfpuElwRightShift": MathOperation.SfpuElwRightShift,
    "SfpuElwLogicalRightShift": MathOperation.SfpuElwLogicalRightShift,
    "SfpuXlogy": MathOperation.SfpuXlogy,
    "SfpuAddTopRow": MathOperation.SfpuAddTopRow,
}

SFPU_TERNARY_OPERATION_MAP: Dict[str, MathOperation] = {
    "SfpuWhere": MathOperation.SfpuWhere,
    "TTNNWhere": MathOperation.TTNNWhere,
}

REDUCE_OPERATION_MAP: Dict[str, MathOperation] = {
    "ReduceColumn": MathOperation.ReduceColumn,
    "ReduceRow": MathOperation.ReduceRow,
    "ReduceScalar": MathOperation.ReduceScalar,
}

REDUCE_POOL_MAP: Dict[str, ReducePool] = {
    "Sum": ReducePool.Sum,
    "Min": ReducePool.Min,
    "Max": ReducePool.Max,
    "Average": ReducePool.Average,
}

APPROXIMATION_MODE_MAP: Dict[str, ApproximationMode] = {
    "Yes": ApproximationMode.Yes,
    "No": ApproximationMode.No,
}

BROADCAST_TYPE_MAP: Dict[str, BroadcastType] = {
    "None": BroadcastType.None_,
    "Row": BroadcastType.Row,
    "Column": BroadcastType.Column,
    "Scalar": BroadcastType.Scalar,
}

ELTWISE_BINARY_REUSE_DEST_TYPE_MAP: Dict[str, EltwiseBinaryReuseDestType] = {
    "NONE": EltwiseBinaryReuseDestType.NONE,
    "DEST_TO_SRCA": EltwiseBinaryReuseDestType.DEST_TO_SRCA,
    "DEST_TO_SRCB": EltwiseBinaryReuseDestType.DEST_TO_SRCB,
}


def parse_math_operation(
    math_config: Dict[str, Any], operands: OperandRegistry
) -> ComputeNode:
    type = math_config["type"]

    if type == "Fpu":
        fpu_type = math_config.get("operation", None)
        if fpu_type is None:
            raise ValueError("Fpu operation can not be None")

        if fpu_type in FPU_OPERATION_MAP:
            math_op = FPU_OPERATION_MAP[fpu_type]
            fpu = EltwiseFpu(math_op)
        elif fpu_type in REDUCE_OPERATION_MAP:
            math_op = REDUCE_OPERATION_MAP[fpu_type]
            pool = REDUCE_POOL_MAP[math_config.get("reduce_pool", "Max")]
            fpu = ReduceFpu(math_op, pool=pool)

        elif fpu_type == "Datacopy":
            fpu = DatacopyFpu()
        elif fpu_type == "Matmul":
            fpu = MatmulFpu()
        elif fpu_type == "ReduceBlockMax":
            fpu = ReduceBlockMaxFpu()
        else:
            raise ValueError(f"Unsupported FPU type: {fpu_type}")

        kwargs = {}

        if "unpacker" in math_config:
            kwargs["unpacker"] = UNPACKER_MAP[math_config["unpacker"]]
        if "unpack_transpose_within_face" in math_config:
            kwargs["unpack_transpose_within_face"] = TRANSPOSE_MAP[
                math_config["unpack_transpose_within_face"]
            ]
        if "unpack_transpose_faces" in math_config:
            kwargs["unpack_transpose_faces"] = TRANSPOSE_MAP[
                math_config["unpack_transpose_faces"]
            ]
        if "broadcast_type" in math_config:
            kwargs["broadcast_type"] = BROADCAST_TYPE_MAP[math_config["broadcast_type"]]

        if "reuse_dest" in math_config:
            kwargs["reuse_dest"] = ELTWISE_BINARY_REUSE_DEST_TYPE_MAP[
                math_config["reuse_dest"]
            ]

        return ComputeNode(
            fpu=fpu,
            sfpu=None,
            **kwargs,
        )

    elif type == "UnarySfpu":
        operation = SFPU_UNARY_OPERATION_MAP[math_config["operation"]]
        approx_mode = APPROXIMATION_MODE_MAP.get(
            math_config.get("approximation_mode", "No"), ApproximationMode.No
        )
        iterations = math_config.get("iterations", 8)
        dest_idx = math_config.get("dst_dest_tile_index", 0)
        fill_const_value = math_config.get("fill_const_value", 1.0)

        return ComputeNode(
            unpacker=None,
            fpu=None,
            sfpu=UnarySfpu(
                operation, approx_mode, iterations, dest_idx, fill_const_value
            ),
        )

    elif type == "BinarySfpu":
        operation = SFPU_BINARY_OPERATION_MAP[math_config["operation"]]
        approx_mode = APPROXIMATION_MODE_MAP.get(
            math_config.get("approximation_mode", "No"), ApproximationMode.No
        )
        iterations = math_config.get("iterations", 8)
        src1_dest_tile_index = math_config.get("src1_dest_tile_index", 0)
        src2_dest_tile_index = math_config.get("src2_dest_tile_index", 0)
        dst_dest_tile_index = math_config.get("dst_dest_tile_index", 0)

        return ComputeNode(
            unpacker=None,
            fpu=None,
            sfpu=BinarySfpu(
                operation,
                approx_mode,
                iterations,
                src1_dest_tile_index,
                src2_dest_tile_index,
                dst_dest_tile_index,
            ),
        )
    else:
        raise ValueError(f"Unsupported math type: {type}")


def parse_operation(
    op_config: Dict[str, Any], operands: OperandRegistry
) -> FusedOperation:
    input_format_name = op_config.get("input_format", "Float16_b")
    input_format = DATA_FORMAT_MAP.get(input_format_name)
    if input_format is None:
        raise ValueError(
            f"Invalid input_format '{input_format_name}'. "
            f"Expected one of: {list(DATA_FORMAT_MAP.keys())}"
        )
    output_format_name = op_config.get("output_format", "Float16_b")
    output_format = DATA_FORMAT_MAP.get(output_format_name)
    if output_format is None:
        raise ValueError(
            f"Invalid output_format '{output_format_name}'. "
            f"Expected one of: {list(DATA_FORMAT_MAP.keys())}"
        )
    operand_mapping = operands.create_mapping(
        src_a=op_config["src_a"],
        src_b=op_config["src_b"],
        output=op_config["output"],
        src_a_dims=op_config.get("src_a_dims", [32, 32]),
        src_b_dims=op_config.get("src_b_dims", [32, 32]),
        output_dims=op_config.get("output_dims", [32, 32]),
        input_format=input_format,
        output_format=output_format,
        src_a_const_value=op_config.get("src_a_const_value"),
        src_b_const_value=op_config.get("src_b_const_value"),
    )

    math_operations = []

    if "math" in op_config:
        for math_config in op_config["math"]:
            math_operations.append(parse_math_operation(math_config, operands))

    packer_name = op_config.get("packer", "Packer")
    packer = PACKER_MAP.get(packer_name)
    if packer is None:
        valid_packers = ", ".join(PACKER_MAP.keys())
        raise ValueError(
            f"Invalid packer {packer_name} in operation config. "
            f"Valid packers are: {valid_packers}"
        )

    kwargs = {}

    if "math_fidelity" in op_config:
        kwargs["math_fidelity"] = MATH_FIDELITY_MAP[op_config["math_fidelity"]]
    if "dest_sync" in op_config:
        kwargs["dest_sync"] = DEST_SYNC_MAP[op_config["dest_sync"]]
    if "batch_size" in op_config:
        kwargs["batch_size"] = op_config["batch_size"]

    return FusedOperation(
        operand_mapping=operand_mapping,
        math=ComputePipeline(math_operations),
        packer=packer,
        **kwargs,
    )


def load_fuser_config(test_name: str) -> FuserConfig:
    yaml_path = FUSER_CONFIG_DIR / f"{test_name}.yaml"
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file does not exist: {yaml_path}")

    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    dest_acc = DEST_ACCUMULATION_MAP[config.get("dest_acc", "No")]
    profiler_enabled = config.get("profiler_enabled", False)
    loop_factor = config.get("loop_factor", 16)

    operands = OperandRegistry()

    pipeline = []
    for op_config in config.get("operations", []):
        operation = parse_operation(op_config, operands)
        pipeline.append(operation)

    fuser_config = FuserConfig(
        pipeline=pipeline,
        global_config=GlobalConfig(
            dest_acc=dest_acc,
            test_name=test_name,
            profiler_enabled=profiler_enabled,
            loop_factor=loop_factor,
        ),
    )

    return fuser_config
