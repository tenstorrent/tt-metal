# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List, Type

import torch

from .golden_generators import (  # TilizeGolden,
    BinarySFPUGolden,
    DataCopyGolden,
    EltwiseBinaryGolden,
    MatmulGolden,
    ReduceGolden,
    UnarySFPUGolden,
    get_golden_generator,
)

if TYPE_CHECKING:
    from .fused_operation import FusedOperation

from .chip_architecture import ChipArchitecture
from .llk_params import ApproximationMode, MathOperation, ReduceDimension, ReducePool
from .tilize_untilize import tilize_block, untilize_block


class Fpu:
    def exec(self, operation_config: "FusedOperation") -> str:
        return ""

    def golden(self, operation_config: "FusedOperation") -> torch.Tensor:
        return torch.Tensor()

    def get_headers(self) -> List[str]:
        return []

    def __str__(self) -> str:
        return self.__class__.__name__


class MatmulFpu(Fpu):
    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_matmul.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation_config: "FusedOperation",
    ) -> torch.Tensor:
        src_a = operation_config.src_a
        src_b = operation_config.src_b
        output_format = operation_config.output.data_format
        math_fidelity = operation_config.math_fidelity

        generate_golden = get_golden_generator(MatmulGolden)
        golden = generate_golden(
            tensor_a,
            tensor_b,
            output_format,
            math_fidelity,
            input_A_dimensions=src_a.dimensions,
            input_B_dimensions=src_b.dimensions,
            tilize=False,
        )
        return golden

    def exec(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        ct_dim = operation_config.ct_dim
        rt_dim = operation_config.rt_dim
        kt_dim = operation_config.kt_dim
        math_fidelity = operation_config.math_fidelity.value
        transpose = "true" if operation_config.unpack_transpose_faces.value else "false"

        code = (
            f"    // Operation {stage}: Matmul FPU\n"
            f"    _llk_math_matmul_init_<{math_fidelity}>(\n"
            f"        TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, {transpose}, {ct_dim}, {rt_dim}\n"
            f"    );\n"
            f"    _llk_math_wait_for_dest_available_<dest_sync{stage}>();\n"
            f"    for (uint32_t j = 0; j < {kt_dim}; j++)\n"
            f"    {{\n"
            f"        _llk_math_matmul_<{math_fidelity}>(0, {ct_dim}, {rt_dim});\n"
            f"    }}\n"
        )

        return code


class EltwiseFpu(Fpu):
    def __init__(self, operation: MathOperation):
        if not operation in MathOperation.get_fpu_binary_operations():
            raise ValueError(
                f"Operation {operation} is not a valid FPU binary operation."
            )
        self.operation = operation

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_eltwise_binary.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation_config: "FusedOperation",
    ) -> torch.Tensor:
        output_format = operation_config.output.data_format
        math_fidelity = operation_config.math_fidelity

        generate_golden = get_golden_generator(EltwiseBinaryGolden)
        golden_tensor = generate_golden(
            self.operation, tensor_a, tensor_b, output_format, math_fidelity
        ).flatten()

        return golden_tensor

    def exec(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        math_fidelity = operation_config.math_fidelity.value
        dest_acc = operation_config.dest_acc.value
        tile_cnt = operation_config.output.tile_count
        op = self.operation.cpp_enum_value
        num_faces = operation_config.num_faces

        code = (
            f"    // Operation {stage}: Eltwise {op} FPU\n"
            f"    _llk_math_eltwise_binary_init_<ckernel::EltwiseBinaryType::{op}, BroadcastType::NONE, {math_fidelity}>({num_faces}, 0);\n"
            f"    _llk_math_wait_for_dest_available_<dest_sync{stage}>();\n"
            f"    for (int i = 0; i < {tile_cnt}; i++)\n"
            f"    {{\n"
            f"        _llk_math_eltwise_binary_<{op}, BroadcastType::NONE, dest_sync{stage},\n"
            f"            {dest_acc}, {math_fidelity}, EltwiseBinaryReuseDestType::NONE>({num_faces}, i, false);\n"
            f"    }}\n"
        )

        return code

    def __str__(self) -> str:
        return f"EltwiseFpu({self.operation})"


class ReduceFpu(Fpu):
    def __init__(self, operation: MathOperation, pool: ReducePool = ReducePool.Max):
        if operation not in MathOperation.get_reduce_operations():
            raise ValueError(f"Operation {operation} is not a valid REDUCE operation.")
        self.operation = operation
        self.pool = pool

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_reduce.h",
        ]

    def reduce_dim(self) -> str:
        return f"ReduceDim::{self.operation.cpp_enum_value}"

    def reduce_dim_golden(self) -> ReduceDimension:
        if self.operation == MathOperation.ReduceColumn:
            return ReduceDimension.Column
        elif self.operation == MathOperation.ReduceRow:
            return ReduceDimension.Row
        elif self.operation == MathOperation.ReduceScalar:
            return ReduceDimension.Scalar
        else:
            raise ValueError(f"Unsupported reduce operation: {self.operation}")

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation_config: "FusedOperation",
    ) -> torch.Tensor:
        output_format = operation_config.output.data_format
        tile_cnt = operation_config.output.tile_count
        dimensions = operation_config.output.dimensions
        num_faces = operation_config.num_faces

        reduce_dim = self.reduce_dim_golden()
        pool_type = self.pool

        tensor_a = tilize_block(
            tensor_a, dimensions, output_format, num_faces
        ).flatten()

        generate_golden = get_golden_generator(ReduceGolden)
        golden_tensor = generate_golden(
            tensor_a,
            reduce_dim,
            pool_type,
            output_format,
            tile_cnt=tile_cnt,
        ).flatten()

        golden_tensor = untilize_block(golden_tensor, output_format, dimensions)

        return golden_tensor.flatten()

    def exec(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        math_fidelity = operation_config.math_fidelity.value
        dest_acc = operation_config.dest_acc.value
        tile_cnt = operation_config.output.tile_count
        num_faces = operation_config.num_faces

        pool_type_cpp = f"PoolType::{self.pool.value}"
        reduce_dim_cpp = self.reduce_dim()

        unp_a_src_format = f"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{operation_config.src_a.data_format})"

        code = (
            f"    // Operation {stage}: Reduce {self.operation.cpp_enum_value} FPU\n"
            f"    _llk_math_reduce_init_<{pool_type_cpp}, {reduce_dim_cpp}, {dest_acc}, {math_fidelity}, false>();\n"
            f"    _llk_math_wait_for_dest_available_<dest_sync{stage}>();\n"
            f"    for (int i = 0; i < {tile_cnt}; ++i)\n"
            f"    {{\n"
            f"        _llk_math_reduce_<{pool_type_cpp}, {reduce_dim_cpp}, {dest_acc}, {math_fidelity}, false, false>(\n"
            f"            i, false, {num_faces}\n"
            f"        );\n"
            f"    }}\n"
        )

        if operation_config.architecture == ChipArchitecture.WORMHOLE:
            code += f"    _llk_math_reduce_uninit_({unp_a_src_format});\n"

        return code

    def __str__(self) -> str:
        return f"ReduceFpu({self.operation}, {self.pool})"


class DatacopyFpu(Fpu):
    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_eltwise_unary_datacopy.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation_config: "FusedOperation",
    ) -> torch.Tensor:
        golden_generator = get_golden_generator(DataCopyGolden)
        golden_tensor = golden_generator(
            tensor_a,
            operation_config.output.data_format,
            num_faces=operation_config.num_faces,
            input_dimensions=operation_config.src_a.dimensions,
            face_r_dim=operation_config.face_r_dim,
        )

        return golden_tensor

    def exec(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        dest_acc = operation_config.dest_acc.value
        tile_cnt = operation_config.output.tile_count
        tilize_en = "true" if operation_config.bh_tilize.value else "false"
        broadcast_type = "BroadcastType::NONE"
        unpack_to_dest = "true" if operation_config.unpack_to_dest else "false"
        data_copy_type = f"DataCopyType::{operation_config.data_copy_type.name}"
        num_faces = operation_config.num_faces
        is_int_fpu_en = dest_acc
        dst_index = operation_config.dst_index

        code = f"    // Operation {stage}: Datacopy FPU\n"

        if operation_config.architecture == ChipArchitecture.BLACKHOLE:
            code += (
                f"    _llk_math_eltwise_unary_datacopy_init_<{data_copy_type}, {dest_acc}, {broadcast_type}, {tilize_en}, {is_int_fpu_en}>(\n"
                f"        {num_faces}, math_format{stage}\n"
                f"    );\n"
            )
        elif operation_config.architecture == ChipArchitecture.WORMHOLE:
            code += (
                f"    _llk_math_eltwise_unary_datacopy_init_<{data_copy_type}, {dest_acc}, {broadcast_type}, {is_int_fpu_en}>(\n"
                f"        {num_faces}, math_format{stage}\n"
                f"    );\n"
            )
        else:
            raise ValueError("Unsupported architecture for DatacopyFpu")

        code += (
            f"    _llk_math_wait_for_dest_available_<dest_sync{stage}>();\n"
            f"    for (int i = 0; i < {tile_cnt}; ++i)\n"
            f"    {{\n"
        )

        if operation_config.architecture == ChipArchitecture.BLACKHOLE:
            code += (
                f"        _llk_math_eltwise_unary_datacopy_<{data_copy_type}, dest_sync{stage}, {dest_acc}, {broadcast_type}, {unpack_to_dest}>(\n"
                f"            {dst_index} + i, math_format{stage}, math_format{stage}, {num_faces});\n"
            )
        elif operation_config.architecture == ChipArchitecture.WORMHOLE:
            code += (
                f"        _llk_math_eltwise_unary_datacopy_<{data_copy_type}, dest_sync{stage}, {dest_acc}, {broadcast_type}, {unpack_to_dest}>(\n"
                f"            {dst_index} + i, math_format{stage}, math_format{stage});\n"
            )
        else:
            raise ValueError("Unsupported architecture for DatacopyFpu")

        code += f"    }}\n"

        return code


class Sfpu:
    def exec(self, operation_config: "FusedOperation") -> str:
        return ""

    def golden(
        self, tensor: torch.Tensor, operation_config: "FusedOperation"
    ) -> torch.Tensor:
        return tensor

    def get_headers(self) -> List[str]:
        return []

    def __str__(self) -> str:
        return f"{self.__name__}"


class UnarySfpu(Sfpu):
    def __init__(
        self,
        operation: MathOperation,
        approx_mode: ApproximationMode = ApproximationMode.No,
        iterations: int = 32,
    ):
        if not operation in MathOperation.get_sfpu_unary_operations():
            raise ValueError(
                f"Operation {operation} is not a valid SFPU unary operation."
            )
        self.iterations = iterations
        self.approx_mode = approx_mode
        self.operation = operation

    def get_headers(self) -> List[str]:
        return [
            "ckernel_defs.h",
            "ckernel_sfpu.h",
            "llk_math_common.h",
            "llk_math_eltwise_unary_sfpu.h",
            "sfpu_operations.h",
        ]

    def golden(
        self, tensor: torch.Tensor, operation_config: "FusedOperation"
    ) -> torch.Tensor:
        format_input = operation_config.src_a.data_format
        format_output = operation_config.output.data_format
        dest_acc = operation_config.dest_acc
        dimensions = operation_config.output.dimensions

        generate_sfpu_golden = get_golden_generator(UnarySFPUGolden)

        return generate_sfpu_golden(
            self.operation,
            tensor,
            format_output,
            dest_acc,
            format_input,
            dimensions,
            self.iterations,
        )

    def exec(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        dest_acc = operation_config.dest_acc.value
        op = f"SfpuType::{self.operation.cpp_enum_value}"

        code = (
            f"    // Operation {stage}: Unary {self.operation.cpp_enum_value} SFPU\n"
            f"    _llk_math_eltwise_unary_sfpu_init_<SfpuType::{self.operation.cpp_enum_value}>();\n"
            f"    _llk_math_eltwise_unary_sfpu_start_<dest_sync{stage}>(0);\n"
            f"    test_utils::call_sfpu_operation<{self.approx_mode.value}, {dest_acc}, {self.iterations}>({op}, math_format{stage});\n"
            f"    _llk_math_eltwise_unary_sfpu_done_();\n"
        )

        return code

    def __str__(self) -> str:
        return f"UnarySfpu({self.operation})"


class BinarySfpu(Sfpu):
    def __init__(
        self,
        operation: MathOperation,
        approx_mode: ApproximationMode = ApproximationMode.No,
        iterations: int = 32,
        dst_index_in0: int = 0,
        dst_index_in1: int = 1,
        dst_index_out: int = 0,
    ):
        if not operation in MathOperation.get_sfpu_binary_operations():
            raise ValueError(
                f"Operation {operation} is not a valid SFPU binary operation."
            )
        self.operation = operation
        self.approx_mode = approx_mode
        self.iterations = iterations
        self.dst_index_in0 = dst_index_in0
        self.dst_index_in1 = dst_index_in1
        self.dst_index_out = dst_index_out

    def get_headers(self) -> List[str]:
        return [
            "ckernel_defs.h",
            "ckernel_sfpu.h",
            "ckernel_sfpu_binary.h",
            "llk_math_common.h",
            "llk_math_eltwise_binary_sfpu.h",
            "sfpu_operations.h",
        ]

    def golden(
        self, tensor: torch.Tensor, operation_config: "FusedOperation"
    ) -> torch.Tensor:
        math_format = operation_config.output.data_format
        dimensions = operation_config.output.dimensions

        generate_binary_golden = get_golden_generator(BinarySFPUGolden)
        golden_tensor = generate_binary_golden(
            self.operation,
            tensor,
            self.dst_index_in0,
            self.dst_index_in1,
            self.dst_index_out,
            self.iterations,
            dimensions,
            math_format,
        )

        return golden_tensor

    def exec(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        op = f"ckernel::BinaryOp::{self.operation.cpp_enum_value}"
        approx_mode = self.approx_mode.value
        iterations = self.iterations
        src1 = self.dst_index_in0
        src2 = self.dst_index_in1
        dst = self.dst_index_out

        if self.operation == MathOperation.SfpuAddTopRow:
            format = "0"
        else:
            format = f"math_format{stage}"

        code = (
            f"    // Operation {stage}: Binary {self.operation.cpp_enum_value} SFPU\n"
            f"    _llk_math_eltwise_binary_sfpu_init_<SfpuType::add1>();\n"
            f"    _llk_math_eltwise_binary_sfpu_start_<dest_sync{stage}>(0);\n"
            f"    test_utils::call_binary_sfpu_operation<{approx_mode}, {op}, {iterations}, {format}>({src1}, {src2}, {dst});\n"
            f"    _llk_math_eltwise_binary_sfpu_done_();\n"
        )

        return code

    def __str__(self) -> str:
        return f"BinarySfpu({self.operation})"


class SfpuWhere(Sfpu):
    def __init__(
        self,
        approx_mode: ApproximationMode = ApproximationMode.No,
        iterations: int = 32,
        dst_index_in0: int = 0,
        dst_index_in1: int = 1,
        dst_index_in2: int = 2,
        dst_index_out: int = 0,
    ):
        self.operation = MathOperation.SfpuWhere
        self.approx_mode = approx_mode
        self.iterations = iterations
        self.dst_index_in0 = dst_index_in0
        self.dst_index_in1 = dst_index_in1
        self.dst_index_in2 = dst_index_in2
        self.dst_index_out = dst_index_out

    def get_headers(self) -> List[str]:
        return [
            "ckernel_defs.h",
            "ckernel_sfpu.h",
            "ckernel_sfpu_where.h",
            "llk_math_common.h",
            "llk_math_eltwise_ternary_sfpu.h",
        ]

    def golden(
        self, tensor: torch.Tensor, operation_config: "FusedOperation"
    ) -> torch.Tensor:
        return tensor

    def exec(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        src1 = self.dst_index_in0
        src2 = self.dst_index_in1
        src3 = self.dst_index_in2
        dst = self.dst_index_out

        code = (
            f"    // Operation {stage}: Binary {self.operation.cpp_enum_value} SFPU\n"
            f"    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::where>();\n"
            f"    ckernel::sfpu::_init_where_<{self.approx_mode.value}>();\n"
            f"    _llk_math_eltwise_ternary_sfpu_start_<dest_sync{stage}>(0);\n"
            f"    ckernel::sfpu::_calculate_where_<false, math_format{stage}, {self.iterations}>({src1}, {src2}, {src3}, {dst});\n"
            f"    _llk_math_eltwise_ternary_sfpu_done_();\n"
        )

        return code

    def __str__(self) -> str:
        return "SfpuWhere"


class Math:
    fpu: Fpu
    sfpu: List[Sfpu]

    def __init__(self, fpu: Type[Fpu], sfpu: List[Sfpu] = []):
        self.fpu = fpu
        self.sfpu = sfpu

    def get_headers(self) -> List[str]:
        headers = set()

        headers.update(self.fpu.get_headers())

        for sfpu in self.sfpu:
            headers.update(sfpu.get_headers())

        return sorted(list(headers))

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation_config: "FusedOperation",
    ) -> torch.Tensor:
        result = self.fpu.golden(tensor_a, tensor_b, operation_config)

        for sfpu in self.sfpu:
            result = sfpu.golden(result, operation_config)

        return result

    def hw_configure(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        dest_acc = operation_config.dest_acc.value
        if stage == 0:
            code = f"    _llk_math_hw_configure_(math_format{stage}, math_format{stage});\n"
        else:
            code = f"    _llk_math_reconfig_data_format_<{dest_acc}, false>(math_format{stage}, math_format{stage});\n"

        code += f"    _llk_math_pack_sync_init_<dest_sync{stage}, {dest_acc}>();\n\n"

        return code

    def exec(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        format = f"DataFormat::{operation_config.math_format.name}"
        code = (
            f"    // Operation {stage}: Math Setup\n"
            f"    const uint32_t math_format{stage} = static_cast<std::underlying_type_t<DataFormat>>({format});\n"
            f"    const DstSync dest_sync{stage} = DstSync::Sync{operation_config.dest_sync.name};\n"
        )
        code += self.hw_configure(operation_config)
        code += self.fpu.exec(operation_config)

        for sfpu in self.sfpu:
            code += f"\n" f"{sfpu.exec(operation_config)}"

        dest_acc = operation_config.dest_acc.value
        code += (
            f"\n"
            f"    _llk_math_dest_section_done_<dest_sync{stage}, {dest_acc}>();\n"
            f"\n"
        )
        return code
