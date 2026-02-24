# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List, Tuple

import torch

from .golden_generators import (
    DataCopyGolden,
    EltwiseBinaryGolden,
    MatmulGolden,
    ReduceBlockMaxRowGolden,
    ReduceGolden,
    get_golden_generator,
)

if TYPE_CHECKING:
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig
    from .fused_math import ComputeNode

from .chip_architecture import ChipArchitecture
from .llk_params import (
    BroadcastType,
    EltwiseBinaryReuseDestType,
    MathOperation,
    ReduceDimension,
    ReducePool,
)
from .tilize_untilize import tilize_block, untilize_block


class Fpu:
    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        return ""

    def calculate(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        tile_idx: int,
    ) -> str:
        return ""

    def uninit(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        return ""

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (tensor_a, tensor_b, tensor_dst)

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
        tensor_dst: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_format = operation.output.data_format
        math_fidelity = operation.math_fidelity

        generate_golden = get_golden_generator(MatmulGolden)
        golden = generate_golden(
            tensor_a,
            tensor_b,
            output_format,
            math_fidelity,
            input_A_dimensions=operation.src_a.dimensions,
            input_B_dimensions=operation.src_b.dimensions,
            tilize=False,
        )

        return (tensor_a, tensor_b, golden)

    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        stage = operation.stage_id
        batch_size = operation.batch_size
        math_fidelity = operation.math_fidelity.cpp_enum_value
        ct_dim = operation.ct_dim
        transpose = compute_unit.unpack_transpose_faces.cpp_enum_value

        if batch_size == ct_dim:
            rt_dim = 1
        else:
            rt_dim = operation.rt_dim

        return (
            f"// Operation {stage}: Matmul FPU\n"
            f"_llk_math_matmul_init_<{math_fidelity}>(\n"
            f"    TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, {transpose}, {ct_dim}, {rt_dim}\n"
            f");\n"
        )

    def calculate(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        tile_idx: int,
    ) -> str:
        batch_size = operation.batch_size
        ct_dim = operation.ct_dim
        kt_dim = operation.kt_dim
        math_fidelity = operation.math_fidelity.cpp_enum_value

        if batch_size == ct_dim:
            rt_dim = 1
        else:
            rt_dim = operation.rt_dim

        return (
            f"for (std::uint32_t kt = 0; kt < {kt_dim}; kt++)\n"
            f"{{\n"
            f"    _llk_math_matmul_<{math_fidelity}>(0, {ct_dim}, {rt_dim});\n"
            f"}}\n"
        )


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
        tensor_dst: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_format = operation.output.data_format
        math_fidelity = operation.math_fidelity

        if compute_unit.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
            tensor_a = tensor_dst
            tensor_dst = torch.zeros_like(tensor_dst)

        if compute_unit.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCB:
            tensor_b = tensor_dst
            tensor_dst = torch.zeros_like(tensor_dst)

        generate_golden = get_golden_generator(EltwiseBinaryGolden)
        golden_tensor = generate_golden(
            self.operation, tensor_a, tensor_b, output_format, math_fidelity
        ).reshape(operation.max_output_dimensions)

        if (
            config.architecture == ChipArchitecture.WORMHOLE
            and self.operation == MathOperation.Elwmul
        ):
            golden_tensor = golden_tensor + tensor_dst

        return (tensor_a, tensor_b, golden_tensor)

    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        stage = operation.stage_id
        math_fidelity = operation.math_fidelity.cpp_enum_value
        op = self.operation.cpp_enum_value
        num_faces = operation.num_faces
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value

        return (
            f"    // Operation {stage}: Eltwise {op} FPU\n"
            f"    _llk_math_eltwise_binary_init_<ckernel::EltwiseBinaryType::{op}, {broadcast_type}, {math_fidelity}, {reuse_dest}>({num_faces}, 0);\n"
        )

    def calculate(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        tile_idx: int,
    ) -> str:
        stage = operation.stage_id
        math_fidelity = operation.math_fidelity.cpp_enum_value
        dest_acc = config.dest_acc.cpp_enum_value
        op = self.operation.cpp_enum_value
        num_faces = operation.num_faces
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value

        return (
            f"    _llk_math_eltwise_binary_<{op}, {broadcast_type}, dest_sync{stage},\n"
            f"        {dest_acc}, {math_fidelity}, {reuse_dest}>({num_faces}, {tile_idx}, false\n"
            f"    );\n"
        )

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

    def pool_type(self) -> str:
        return f"PoolType::{self.pool.value}"

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
        tensor_dst: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_format = operation.output.data_format
        dimensions = operation.max_output_dimensions
        tile_cnt = (dimensions[0] * dimensions[1]) // 1024
        num_faces = operation.num_faces

        reduce_dim = self.reduce_dim_golden()
        pool_type = self.pool

        src_a_reduced_tensor = tilize_block(
            tensor_a, dimensions, output_format, num_faces
        ).flatten()

        generate_golden = get_golden_generator(ReduceGolden)
        src_a_reduced_tensor = generate_golden(
            src_a_reduced_tensor,
            reduce_dim,
            pool_type,
            output_format,
            tile_cnt=tile_cnt,
        ).flatten()

        dest_golden_tensor = tilize_block(
            tensor_dst, dimensions, output_format, num_faces
        ).flatten()

        generate_golden = get_golden_generator(ReduceGolden)
        dest_golden_tensor = generate_golden(
            dest_golden_tensor,
            reduce_dim,
            pool_type,
            output_format,
            tile_cnt=tile_cnt,
        ).flatten()

        golden_tensor = torch.zeros(tile_cnt * 1024)

        for i in range(tile_cnt * 1024):
            if self.pool == ReducePool.Max:
                golden_tensor[i] = max(src_a_reduced_tensor[i], dest_golden_tensor[i])
            if self.pool == ReducePool.Sum:
                golden_tensor[i] = src_a_reduced_tensor[i] + dest_golden_tensor[i]
            if self.pool == ReducePool.Average:
                golden_tensor[i] = (
                    src_a_reduced_tensor[i] * 32 + dest_golden_tensor[i]
                ) / 32

        golden_tensor = untilize_block(golden_tensor, output_format, dimensions)

        return (tensor_a, tensor_b, golden_tensor)

    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        stage = operation.stage_id
        math_fidelity = operation.math_fidelity.cpp_enum_value
        dest_acc = config.dest_acc.cpp_enum_value
        pool_type_cpp = self.pool.cpp_enum_value
        reduce_dim_cpp = self.reduce_dim()

        return (
            f"    // Operation {stage}: Reduce {self.operation.cpp_enum_value} FPU\n"
            f"    _llk_math_reduce_init_<{pool_type_cpp}, {reduce_dim_cpp}, {dest_acc}, {math_fidelity}, false>();\n"
        )

    def calculate(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        tile_idx: int,
    ) -> str:
        math_fidelity = operation.math_fidelity.cpp_enum_value
        dest_acc = config.dest_acc.cpp_enum_value
        num_faces = operation.num_faces
        pool_type_cpp = self.pool.cpp_enum_value
        reduce_dim_cpp = self.reduce_dim()

        return (
            f"    _llk_math_reduce_<{pool_type_cpp}, {reduce_dim_cpp}, {dest_acc}, {math_fidelity}, false, false>(\n"
            f"        {tile_idx}, false, {num_faces}\n"
            f"    );\n"
        )

    def uninit(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        unp_a_src_format = f"static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{operation.src_a.data_format})"

        if config.architecture == ChipArchitecture.WORMHOLE:
            return f"_llk_math_reduce_uninit_({unp_a_src_format});\n"

        return ""

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
        tensor_dst: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if compute_unit.broadcast_type != BroadcastType.None_:
            source_tensor = tensor_b
        else:
            source_tensor = tensor_a

        golden_generator = get_golden_generator(DataCopyGolden)
        golden_tensor = golden_generator(
            source_tensor,
            operation.output.data_format,
            num_faces=operation.num_faces,
            input_dimensions=operation.src_a.dimensions,
            face_r_dim=operation.face_r_dim,
        )

        return (tensor_a, tensor_b, golden_tensor)

    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        stage = operation.stage_id
        dest_acc = config.dest_acc.cpp_enum_value
        tilize_en = operation.bh_tilize.cpp_enum_value
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        data_copy_type = compute_unit.data_copy_type.cpp_enum_value
        num_faces = operation.num_faces
        is_int_fpu_en = dest_acc

        code = f"    // Operation {stage}: Datacopy FPU\n"
        if config.architecture == ChipArchitecture.BLACKHOLE:
            code += (
                f"    _llk_math_eltwise_unary_datacopy_init_<{data_copy_type}, {dest_acc}, {broadcast_type}, {tilize_en}, {is_int_fpu_en}>(\n"
                f"        {num_faces}, math_format{stage}\n"
                f"    );\n"
            )
        elif config.architecture == ChipArchitecture.WORMHOLE:
            code += (
                f"    _llk_math_eltwise_unary_datacopy_init_<{data_copy_type}, {dest_acc}, {broadcast_type}, {is_int_fpu_en}>(\n"
                f"        {num_faces}, math_format{stage}\n"
                f"    );\n"
            )
        else:
            raise ValueError("Unsupported architecture for DatacopyFpu")

        return code

    def calculate(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        tile_idx: int,
    ) -> str:
        stage = operation.stage_id
        dest_acc = config.dest_acc.cpp_enum_value
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        unpack_to_dest = "true" if operation.unpack_to_dest else "false"
        data_copy_type = f"DataCopyType::{compute_unit.data_copy_type.name}"
        num_faces = operation.num_faces

        if config.architecture == ChipArchitecture.BLACKHOLE:
            code = (
                f"    _llk_math_eltwise_unary_datacopy_<{data_copy_type}, dest_sync{stage}, {dest_acc}, {broadcast_type}, {unpack_to_dest}>(\n"
                f"        {tile_idx}, math_format{stage}, math_format{stage}, {num_faces}\n"
                f"    );\n"
            )
        elif config.architecture == ChipArchitecture.WORMHOLE:
            code = (
                f"    _llk_math_eltwise_unary_datacopy_<{data_copy_type}, dest_sync{stage}, {dest_acc}, {broadcast_type}, {unpack_to_dest}>(\n"
                f"        {tile_idx}, math_format{stage}, math_format{stage}\n"
                f"    );\n"
            )
        else:
            raise ValueError("Unsupported architecture for DatacopyFpu")

        return code

    def uninit(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        return f"_llk_math_eltwise_unary_datacopy_uninit_<{broadcast_type}, false>();\n"


class ReduceBlockMaxFpu(Fpu):
    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        ct_dim = operation.ct_dim
        dest_acc = config.dest_acc.cpp_enum_value
        return f"_llk_math_reduce_block_max_row_init_<{ct_dim}, {dest_acc}>();\n"

    def calculate(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        tile_idx: int,
    ) -> str:
        ct_dim = operation.ct_dim
        dest_acc = config.dest_acc.cpp_enum_value
        if tile_idx % ct_dim != 0:
            return ""
        return f"_llk_math_reduce_block_max_row_<{ct_dim}, {dest_acc}>({tile_idx});\n"

    def uninit(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        return "_llk_math_reduce_block_max_row_uninit_();\n"

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ct_dim = operation.ct_dim
        output_format = operation.output.data_format
        dimensions = operation.max_output_dimensions

        generate_golden = get_golden_generator(ReduceBlockMaxRowGolden)
        src_a_reduced_tensor = generate_golden(
            tensor_a, ct_dim, output_format, dimensions
        ).flatten()

        dest_golden_tensor = generate_golden(
            tensor_dst, ct_dim, output_format, dimensions
        ).flatten()

        numel = min(src_a_reduced_tensor.numel(), dest_golden_tensor.numel())
        golden_tensor = torch.zeros(numel)

        for i in range(numel):
            golden_tensor[i] = max(src_a_reduced_tensor[i], dest_golden_tensor[i])

        return (tensor_a, tensor_b, golden_tensor)

    def get_headers(self) -> List[str]:
        return ["llk_math_reduce_custom.h"]
