# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict, List, Type

import yaml
from helpers.format_config import DataFormat
from helpers.fused_math import (
    BinarySfpu,
    DatacopyFpu,
    EltwiseFpu,
    Math,
    MatmulFpu,
    UnarySfpu,
)
from helpers.fused_operand import OperandRegistry
from helpers.fused_operation import FusedOperation
from helpers.fused_packer import Packer
from helpers.fused_unpacker import (
    MatmulUnpacker,
    Unpacker,
    UnpackerA,
    UnpackerAB,
    UnpackerTilizeA,
)
from helpers.llk_params import (
    ApproximationMode,
    DestSync,
    MathOperation,
    Transpose,
)

from .llk_params import DestAccumulation, MathFidelity

UNPACKER_MAP: Dict[str, Type[Unpacker]] = {
    "UnpackerA": UnpackerA,
    "UnpackerAB": UnpackerAB,
    "UnpackerTilizeA": UnpackerTilizeA,
    "MatmulUnpacker": MatmulUnpacker,
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
    "Neg": MathOperation.Neg,
    "Reciprocal": MathOperation.Reciprocal,
    "ReluMax": MathOperation.ReluMax,
    "ReluMin": MathOperation.ReluMin,
    "Rsqrt": MathOperation.Rsqrt,
    "Silu": MathOperation.Silu,
    "Sin": MathOperation.Sin,
    "Sqrt": MathOperation.Sqrt,
    "Square": MathOperation.Square,
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

APPROXIMATION_MODE_MAP: Dict[str, ApproximationMode] = {
    "Yes": ApproximationMode.Yes,
    "No": ApproximationMode.No,
}


def parse_math_operation(
    math_config: Dict[str, Any], operands: OperandRegistry
) -> Math:
    fpu_type = math_config.get("fpu", "Datacopy")

    if fpu_type in FPU_OPERATION_MAP:
        math_op = FPU_OPERATION_MAP[fpu_type]
        fpu = EltwiseFpu(math_op)
    elif fpu_type == "Datacopy":
        fpu = DatacopyFpu()
    elif fpu_type == "Matmul":
        fpu = MatmulFpu()
    else:
        raise ValueError(f"Unsupported FPU type: {fpu_type}")

    sfpu_ops = []
    if "sfpu" in math_config:
        for sfpu_config in math_config["sfpu"]:
            sfpu_type = sfpu_config.get("type")

            if sfpu_type == "UnarySfpu":
                operation = SFPU_UNARY_OPERATION_MAP[sfpu_config["operation"]]
                approx_mode = APPROXIMATION_MODE_MAP.get(
                    sfpu_config.get("approximation_mode", "No"), ApproximationMode.No
                )
                iterations = sfpu_config.get("iterations", 32)

                sfpu_ops.append(UnarySfpu(operation, approx_mode, iterations))

            elif sfpu_type == "BinarySfpu":
                operation = SFPU_BINARY_OPERATION_MAP[sfpu_config["operation"]]
                approx_mode = APPROXIMATION_MODE_MAP.get(
                    sfpu_config.get("approximation_mode", "No"), ApproximationMode.No
                )
                iterations = sfpu_config.get("iterations", 32)
                src1_dest_tile_index = sfpu_config.get("src1_dest_tile_index", 0)
                src2_dest_tile_index = sfpu_config.get("src2_dest_tile_index", 0)
                dst_dest_tile_index = sfpu_config.get("dst_dest_tile_index", 0)

                sfpu_ops.append(
                    BinarySfpu(
                        operation,
                        approx_mode,
                        iterations,
                        src1_dest_tile_index,
                        src2_dest_tile_index,
                        dst_dest_tile_index,
                    )
                )
            else:
                raise ValueError(f"Unsupported SFPU type: {sfpu_type}")

    return Math(fpu, sfpu_ops)


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
        input_format=input_format,
        output_format=output_format,
    )

    unpacker_name = op_config.get("unpacker", "UnpackerA")
    unpacker = UNPACKER_MAP.get(unpacker_name)
    if unpacker is None:
        valid_unpackers = ", ".join(UNPACKER_MAP.keys())
        raise ValueError(
            f"Invalid unpacker '{unpacker_name}' in operation config. "
            f"Valid unpackers are: {valid_unpackers}"
        )

    math = parse_math_operation(op_config.get("math", {}), operands)

    packer_name = op_config.get("packer", "Packer")
    packer = PACKER_MAP.get(packer_name)
    if packer is None:
        valid_packers = ", ".join(PACKER_MAP.keys())
        raise ValueError(
            f"Invalid packer {packer_name} in operation config. "
            f"Valid packers are: {valid_packers}"
        )

    kwargs = {}

    if "dest_acc" in op_config:
        kwargs["dest_acc"] = DEST_ACCUMULATION_MAP[op_config["dest_acc"]]
    if "math_fidelity" in op_config:
        kwargs["math_fidelity"] = MATH_FIDELITY_MAP[op_config["math_fidelity"]]
    if "dest_sync" in op_config:
        kwargs["dest_sync"] = DEST_SYNC_MAP[op_config["dest_sync"]]
    if "unpack_transpose_within_face" in op_config:
        kwargs["unpack_transpose_within_face"] = TRANSPOSE_MAP[
            op_config["unpack_transpose_within_face"]
        ]
    if "unpack_transpose_faces" in op_config:
        kwargs["unpack_transpose_faces"] = TRANSPOSE_MAP[
            op_config["unpack_transpose_faces"]
        ]

    return FusedOperation(
        operand_mapping=operand_mapping,
        unpacker=unpacker,
        math=math,
        packer=packer,
        **kwargs,
    )


def create_fuse_pipeline(yaml_path: str) -> List[FusedOperation]:
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file does not exist: {yaml_path}")

    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    operands = OperandRegistry()

    pipeline = []
    for op_config in config.get("operations", []):
        operation = parse_operation(op_config, operands)
        pipeline.append(operation)

    return pipeline
