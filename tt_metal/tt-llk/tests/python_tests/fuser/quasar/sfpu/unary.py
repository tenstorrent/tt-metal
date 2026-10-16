# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import struct
from typing import List

from fuser.base_sfpu import Sfpu
from fuser.block_data import BlockData
from fuser.fuser_config import GlobalConfig
from fuser.golden.sfpu.unary import unary_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from fuser.sfpu_node import SfpuNode
from helpers.llk_params import (
    ApproximationMode,
    MathOperation,
)
from helpers.sfpu_dispatch_constants import (
    DROPOUT_PROBABILITY_MAX,
    DROPOUT_SCALE,
    DROPOUT_SEED,
)


class UnarySfpu(Sfpu):
    granularity = InvocationGranularity.TILE
    golden_fn = staticmethod(unary_golden)

    def __init__(
        self,
        operation: MathOperation,
        approx_mode: ApproximationMode = ApproximationMode.No,
        iterations: int = 8,
        fill_const_value=5,
    ):
        if not operation in MathOperation.get_sfpu_unary_operations():
            raise ValueError(
                f"Operation {operation} is not a valid SFPU unary operation."
            )
        self.iterations = iterations
        self.approx_mode = approx_mode
        self.operation = operation
        self.fill_const_value = fill_const_value
        # Dropout operands; the defaults match the dispatcher's kDropout* (p = 0, scale 2.0).
        self.dropout_probability = 0
        self.dropout_scale = DROPOUT_SCALE
        self.dropout_seed = DROPOUT_SEED

    @property
    def dropout_probability_operand(self) -> int:
        """p * INT_MAX, as calculate_dropout takes it."""
        return DROPOUT_PROBABILITY_MAX if self.dropout_probability else 0

    @property
    def dropout_scale_bits(self) -> int:
        """fp32 bit pattern of the scale, as calculate_dropout takes it."""
        return struct.unpack("<I", struct.pack("<f", self.dropout_scale))[0]

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_eltwise_unary_sfpu.h",
            "sfpu_operations_quasar.h",
        ]

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: SfpuNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        op = f"SfpuType::{self.operation.cpp_enum_value}"
        en_32bit_dest = config.dest_acc.cpp_enum_value
        approx_mode = self.approx_mode.cpp_enum_value
        if self.operation == MathOperation.Dropout:
            op_init = f"test_utils::init_dropout_quasar<{approx_mode}>({self.dropout_seed:#x} /* seed */);\n"
        else:
            op_init = f"test_utils::init_unary_sfpu_operation_quasar<{op}, {en_32bit_dest}, {approx_mode}>();\n"
        return (
            f"// Operation {stage}: Unary {self.operation.cpp_enum_value} SFPU\n"
            f"_llk_math_eltwise_sfpu_init_();\n" + op_init
        )

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: SfpuNode,
        block: BlockData,
    ) -> str:
        op = f"SfpuType::{self.operation.cpp_enum_value}"
        dest_sync = operation.dest_sync.cpp_enum_value
        en_32bit_dest = config.dest_acc.cpp_enum_value
        sfpu_format = config.sentinel._sfpu_format.cpp_enum_value
        approx_mode = self.approx_mode.cpp_enum_value
        dst_tile_shape = operation.tile_shape.dst_tile_shape
        vector_mode = self._vector_mode(operation)
        quasar_iterations = self.iterations // operation.tile_shape.total_num_faces()
        if self.operation == MathOperation.Dropout:
            return (
                f"test_utils::call_dropout_quasar<"
                f"{dest_sync}, {en_32bit_dest}, {approx_mode}, {quasar_iterations}, {dst_tile_shape}"
                f">({block.tile_id_dest}, {self.dropout_probability_operand:#x} /* probability */, "
                f"{self.dropout_scale_bits:#x} /* scale */, {vector_mode});\n"
            )
        return (
            f"test_utils::call_unary_sfpu_operation_quasar<"
            f"{op}, {dest_sync}, {en_32bit_dest}, {approx_mode}, {quasar_iterations}"
            f", DataFormat::Float32, DataFormat::Float16_b, {dst_tile_shape}"
            f">({block.tile_id_dest}, {sfpu_format}, true, {self.fill_const_value}, 0 /* idx_addr */, {vector_mode});\n"
        )

    def __str__(self) -> str:
        return f"UnarySfpu({self.operation})"
