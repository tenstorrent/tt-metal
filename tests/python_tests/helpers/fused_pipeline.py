# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

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
from helpers.fused_unpacker import MatmulUnpacker, UnpackerAB, UnpackerTilizeA
from helpers.llk_params import (
    ApproximationMode,
    DestSync,
    MathOperation,
)

from .llk_params import DestAccumulation, MathFidelity


def create_fuse_pipeline() -> List[FusedOperation]:
    math_fidelity = MathFidelity.LoFi
    dest_acc = DestAccumulation.Yes
    input_A_dimensions = [64, 64]
    input_B_dimensions = [64, 64]

    operands = OperandRegistry()

    pipeline = [
        FusedOperation(
            operand_mapping=operands.create_mapping(
                src_a="input_A",
                src_b="input_B",
                output="datacopy_output",
                src_a_dims=input_A_dimensions,
                src_b_dims=input_B_dimensions,
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
            ),
            unpacker=UnpackerTilizeA,
            math=Math(
                DatacopyFpu(),
                [
                    UnarySfpu(
                        MathOperation.Exp,
                        ApproximationMode.No,
                        32 * operands.get("datacopy_output").tile_count,
                    ),
                    UnarySfpu(
                        MathOperation.Celu,
                        ApproximationMode.No,
                        32 * operands.get("datacopy_output").tile_count,
                    ),
                    BinarySfpu(
                        MathOperation.SfpuElwadd,
                        ApproximationMode.No,
                        32,
                        0,
                        1,
                        1,
                    ),
                ],
            ),
            packer=Packer,
            dest_acc=dest_acc,
            math_fidelity=math_fidelity,
        ),
        FusedOperation(
            operand_mapping=operands.create_mapping(
                src_a="datacopy_output",
                src_b="input_B",
                output="elwadd1",
                src_a_dims=input_A_dimensions,
                src_b_dims=input_B_dimensions,
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float16_b,
            ),
            unpacker=UnpackerAB,
            math=Math(
                EltwiseFpu(MathOperation.Elwadd),
                [
                    UnarySfpu(
                        MathOperation.Neg,
                        ApproximationMode.No,
                        32 * operands.get("elwadd1").tile_count,
                    ),
                ],
            ),
            packer=Packer,
            dest_acc=dest_acc,
            math_fidelity=math_fidelity,
            dest_sync=DestSync.Full,
        ),
        FusedOperation(
            operand_mapping=operands.create_mapping(
                src_a="elwadd1",
                src_b="input_C",
                output="matmul_result",
                src_a_dims=input_A_dimensions,
                src_b_dims=input_B_dimensions,
                input_format=DataFormat.Float16_b,
                output_format=DataFormat.Float32,
            ),
            unpacker=MatmulUnpacker,
            math=Math(MatmulFpu()),
            packer=Packer,
            dest_acc=dest_acc,
            math_fidelity=math_fidelity,
        ),
        FusedOperation(
            operand_mapping=operands.create_mapping(
                src_a="matmul_result",
                src_b="input_D",
                output="final_output",
                src_b_dims=input_B_dimensions,
                input_format=DataFormat.Float32,
                output_format=DataFormat.Float32,
            ),
            unpacker=MatmulUnpacker,
            math=Math(
                MatmulFpu(),
                [
                    UnarySfpu(
                        MathOperation.Neg,
                        ApproximationMode.No,
                        32 * operands.get("final_output").tile_count,
                    ),
                    UnarySfpu(
                        MathOperation.Sqrt,
                        ApproximationMode.No,
                        32 * operands.get("final_output").tile_count,
                    ),
                    BinarySfpu(
                        MathOperation.SfpuElwadd,
                        ApproximationMode.No,
                        32 * operands.get("final_output").tile_count,
                        0,
                        0,
                        0,
                    ),
                ],
            ),
            packer=Packer,
            dest_acc=dest_acc,
            math_fidelity=math_fidelity,
        ),
    ]

    return pipeline
