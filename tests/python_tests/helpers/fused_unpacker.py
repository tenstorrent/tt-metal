# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List, Tuple

import torch

if TYPE_CHECKING:
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig
    from .fused_math import ComputeNode

from .chip_architecture import ChipArchitecture
from .fused_fpu import ReduceFpu
from .golden_generators import BroadcastGolden, TransposeGolden, get_golden_generator
from .llk_params import BroadcastType, Transpose
from .tilize_untilize import tilize_block, untilize_block


class Unpacker:
    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        return ""

    def unpack(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        tile_idx_expr: str,
    ) -> str:
        return ""

    def uninit(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        return ""

    def perf_set_valid(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        num_faces = operation.num_faces
        return f"_perf_unpack_loop_set_valid<true, true>({num_faces});\n"

    def perf_clear_valid(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        num_faces = operation.num_faces
        return f"_perf_math_loop_clear_valid<true, true>({num_faces});\n"

    def get_headers(self) -> List[str]:
        return ["perf.h"]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode" = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tensor_a, tensor_b


class MatmulUnpacker(Unpacker):
    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_AB_matmul.h",
            "llk_unpack_common.h",
        ]

    def perf_set_valid(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        kt_dim = operation.kt_dim
        ct_dim = operation.ct_dim
        if operation.batch_size == ct_dim:
            rt_dim = 1
        else:
            rt_dim = operation.rt_dim
        return f"_perf_unpack_matmul_mock(1, {rt_dim}, {kt_dim}, {ct_dim});\n"

    def perf_clear_valid(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        kt_dim = operation.kt_dim
        ct_dim = operation.ct_dim
        if operation.batch_size == ct_dim:
            rt_dim = 1
        else:
            rt_dim = operation.rt_dim
        return f"_perf_math_matmul_mock(1, {rt_dim}, {kt_dim}, {ct_dim});\n"

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_matrix = get_golden_generator(TransposeGolden)

        if compute_unit.unpack_transpose_faces == Transpose.Yes:
            tensor_b = t_matrix.transpose_faces_multi_tile(
                tensor_b,
                operation.src_b.data_format,
                operation.src_b.tile_count,
                tilize=True,
                input_dimensions=operation.src_b.dimensions,
            )

        if compute_unit.unpack_transpose_within_face == Transpose.Yes:
            tensor_b = t_matrix.transpose_within_faces_multi_tile(
                tensor_b,
                operation.src_b.data_format,
                operation.src_b.tile_count,
                untilize=True,
                input_dimensions=operation.src_b.dimensions,
            )

        return tensor_a, tensor_b

    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        face_r_dim = operation.face_r_dim
        ct_dim = operation.ct_dim
        kt_dim = operation.kt_dim
        batch_size = operation.batch_size

        transpose_faces = compute_unit.unpack_transpose_faces.cpp_enum_value
        transpose_within_face = compute_unit.unpack_transpose_within_face.cpp_enum_value

        if transpose_within_face != transpose_faces:
            raise ValueError(
                "MatmulUnpacker does not support different values for transpose_faces and transpose_within_face"
            )

        if batch_size == ct_dim:
            rt_dim = 1
        else:
            rt_dim = operation.rt_dim

        return f"    _llk_unpack_AB_matmul_init_<>({transpose_faces}, {ct_dim}, {rt_dim}, {kt_dim}, {face_r_dim}, {face_r_dim});\n"

    def unpack(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        tile_idx_expr: str = None,
    ) -> str:
        stage = operation.stage_id
        ct_dim = operation.ct_dim
        rt_dim = operation.rt_dim
        kt_dim = operation.kt_dim
        batch_size = operation.batch_size
        unpack_tile_size_a = operation.tile_size_unpack_a
        unpack_tile_size_b = operation.tile_size_unpack_b
        full_ct_dim = operation.src_b.dimensions[1] // 32

        if batch_size == ct_dim:
            code = (
                f"    {{\n"
                f"        std::uint32_t mt = batch;\n"
                f"        for (std::uint32_t kt = 0; kt < {kt_dim}; ++kt) {{\n"
                f"            _llk_unpack_AB_matmul_<>(\n"
                f"                L1_ADDRESS(buffer_A{stage}[0]), L1_ADDRESS(buffer_B{stage}[0]),\n"
                f"                mt * {kt_dim} + kt, kt * {full_ct_dim},\n"
                f"                {unpack_tile_size_a}, {unpack_tile_size_b}, false, false, {ct_dim}, 1, {kt_dim}\n"
                f"            );\n"
                f"        }}\n"
                f"    }}\n"
            )
        else:
            code = (
                f"    for (std::uint32_t kt = 0; kt < {kt_dim}; ++kt) {{\n"
                f"        _llk_unpack_AB_matmul_<>(\n"
                f"            L1_ADDRESS(buffer_A{stage}[0]), L1_ADDRESS(buffer_B{stage}[0]),\n"
                f"            kt, kt * {full_ct_dim},\n"
                f"            {unpack_tile_size_a}, {unpack_tile_size_b}, false, false, {ct_dim}, {rt_dim}, {kt_dim}\n"
                f"        );\n"
                f"    }}\n"
            )

        return code


class UnpackerAB(Unpacker):
    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_AB.h",
            "llk_unpack_AB_reduce.h",
            "llk_unpack_common.h",
            "llk_unpack_tilize.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_matrix = get_golden_generator(TransposeGolden)
        if compute_unit.broadcast_type != BroadcastType.None_:
            tilized_b = tilize_block(
                tensor_b, operation.src_b.dimensions, operation.src_b.data_format
            )
            broadcast_golden = get_golden_generator(BroadcastGolden)
            broadcast_result = broadcast_golden(
                compute_unit.broadcast_type,
                tilized_b,
                operation.src_b.data_format,
                operation.num_faces,
                operation.src_b.tile_count,
                operation.face_r_dim,
            )
            tensor_b = untilize_block(
                broadcast_result,
                operation.src_b.data_format,
                operation.src_b.dimensions,
            )

        if compute_unit.unpack_transpose_faces == Transpose.Yes:
            tensor_a = t_matrix.transpose_faces_multi_tile(
                tensor_a,
                operation.src_a.data_format,
                operation.src_a.tile_count,
                tilize=True,
                untilize=True,
                input_dimensions=operation.src_a.dimensions,
            )

        if compute_unit.unpack_transpose_within_face == Transpose.Yes:
            tensor_a = t_matrix.transpose_within_faces_multi_tile(
                tensor_a,
                operation.src_a.data_format,
                operation.src_a.tile_count,
                tilize=True,
                untilize=True,
                input_dimensions=operation.src_a.dimensions,
            )

        return tensor_a.flatten(), tensor_b.flatten()

    def perf_set_valid(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        num_faces = operation.num_faces
        if compute_unit.broadcast_type == BroadcastType.Scalar:
            return (
                f"_perf_unpack_loop_set_valid<false, true>(1);\n"
                f"_perf_unpack_loop_set_valid<true, false>({num_faces});\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Column:
            return (
                f"_perf_unpack_loop_set_valid<false, true>(2);\n"
                f"_perf_unpack_loop_set_valid<true, false>({num_faces});\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Row:
            return f"_perf_unpack_loop_set_valid<true, true>({num_faces});\n"
        else:
            return f"_perf_unpack_loop_set_valid<true, true>({num_faces});\n"

    def perf_clear_valid(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        num_faces = operation.num_faces
        if compute_unit.broadcast_type == BroadcastType.Scalar:
            return (
                f"_perf_math_loop_clear_valid<false, true>(1);\n"
                f"_perf_math_loop_clear_valid<true, false>({num_faces});\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Column:
            return (
                f"_perf_math_loop_clear_valid<false, true>(2);\n"
                f"_perf_math_loop_clear_valid<true, false>({num_faces});\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Row:
            return f"_perf_math_loop_clear_valid<true, true>({num_faces});\n"
        else:
            return f"_perf_math_loop_clear_valid<true, true>({num_faces});\n"

    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        face_r_dim = operation.face_r_dim
        num_faces = operation.num_faces
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value

        if (
            compute_unit.broadcast_type == BroadcastType.Scalar
            and compute_unit.unpack_transpose_faces.value
        ):
            raise ValueError("SrcA transpose is not supported with scalar broadcast")

        transpose_faces = compute_unit.unpack_transpose_faces.cpp_enum_value
        transpose_within_face = compute_unit.unpack_transpose_within_face.cpp_enum_value

        if isinstance(compute_unit.fpu, ReduceFpu):
            if compute_unit.broadcast_type != BroadcastType.None_:
                raise ValueError("ReduceFpu does not support broadcasted inputs.")

            reduce_dim = compute_unit.fpu.reduce_dim()
            pool_type = compute_unit.fpu.pool_type()

            return (
                f"_llk_unpack_AB_reduce_init_<{pool_type}, {reduce_dim}>(\n"
                f"{face_r_dim}, {num_faces});\n"
            )
        else:
            if transpose_within_face != transpose_faces:
                raise ValueError(
                    "UnpackerAB does not support different values for transpose_faces and transpose_within_face"
                )

            face_c_dim = operation.face_c_dim
            num_faces_r_dim = operation.in0_tile_r_dim // face_r_dim
            num_faces_c_dim = operation.in0_tile_c_dim // face_c_dim
            transpose_value = "1" if compute_unit.unpack_transpose_faces.value else "0"
            shape_var = f"tensor_shape_stage_{operation.stage_id}"
            return (
                f"const ckernel::TensorShape {shape_var} = "
                f"{{{face_r_dim}, {face_c_dim}, {num_faces_r_dim}, {num_faces_c_dim}}};\n"
                f"_llk_unpack_AB_init_<{broadcast_type}>({shape_var}, {transpose_value});\n"
            )

    def unpack(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        tile_idx_expr: str,
    ) -> str:
        stage = operation.stage_id
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        return f"_llk_unpack_AB_<{broadcast_type}>(L1_ADDRESS(buffer_A{stage}[{tile_idx_expr}]), L1_ADDRESS(buffer_B{stage}[{tile_idx_expr}]));\n"


class UnpackerA(Unpacker):
    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_A.h",
            "llk_unpack_common.h",
            "llk_unpack_tilize.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_matrix = get_golden_generator(TransposeGolden)

        if compute_unit.broadcast_type != BroadcastType.None_:
            tensor_b = tensor_a
            tensor_a = None
            tensor_b = tilize_block(
                tensor_b, operation.src_a.dimensions, operation.src_a.data_format
            )
            broadcast_golden = get_golden_generator(BroadcastGolden)
            tensor_b = broadcast_golden(
                compute_unit.broadcast_type,
                tensor_b,
                operation.src_a.data_format,
                operation.num_faces,
                operation.src_a.tile_count,
                operation.face_r_dim,
            )
            tensor_b = untilize_block(
                tensor_b,
                operation.src_a.data_format,
                operation.src_a.dimensions,
            )
        else:
            if compute_unit.unpack_transpose_faces == Transpose.Yes:
                tensor_a = t_matrix.transpose_faces_multi_tile(
                    tensor_a,
                    operation.src_a.data_format,
                    operation.src_a.tile_count,
                    tilize=True,
                    untilize=True,
                    input_dimensions=operation.src_a.dimensions,
                )

            if compute_unit.unpack_transpose_within_face == Transpose.Yes:
                tensor_a = t_matrix.transpose_within_faces_multi_tile(
                    tensor_a,
                    operation.src_a.data_format,
                    operation.src_a.tile_count,
                    tilize=True,
                    untilize=True,
                    input_dimensions=operation.src_a.dimensions,
                )
            tensor_b = None

        return tensor_a, tensor_b

    def perf_set_valid(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        if compute_unit.broadcast_type == BroadcastType.Scalar:
            if config.architecture == ChipArchitecture.WORMHOLE:
                return "_perf_unpack_loop_set_valid<true, true>(1);\n"
            else:
                return "_perf_unpack_loop_set_valid<false, true>(1);\n"
        elif compute_unit.broadcast_type == BroadcastType.Column:
            return (
                "_perf_unpack_loop_set_valid<false, true>(2);\n"
                "_perf_unpack_loop_set_valid<true, false>(1);\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Row:
            return "_perf_unpack_loop_set_valid<false, true>(4);\n"
        else:
            num_faces = operation.num_faces
            return f"_perf_unpack_loop_set_valid<true, true>({num_faces});\n"

    def perf_clear_valid(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        if compute_unit.broadcast_type == BroadcastType.Scalar:
            if config.architecture == ChipArchitecture.WORMHOLE:
                return "_perf_math_loop_clear_valid<true, true>(1);\n"
            else:
                return "_perf_math_loop_clear_valid<false, true>(1);\n"
        elif compute_unit.broadcast_type == BroadcastType.Column:
            return (
                "_perf_math_loop_clear_valid<false, true>(2);\n"
                "_perf_math_loop_clear_valid<true, false>(1);\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Row:
            return "_perf_math_loop_clear_valid<false, true>(4);\n"
        else:
            num_faces = operation.num_faces
            return f"_perf_math_loop_clear_valid<true, true>({num_faces});\n"

    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        stage = operation.stage_id
        unpack_to_dest = "true" if operation.unpack_to_dest else "false"
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        face_r_dim = operation.face_r_dim
        num_faces = operation.num_faces
        transpose_faces = compute_unit.unpack_transpose_faces.cpp_enum_value
        transpose_within_face = compute_unit.unpack_transpose_within_face.cpp_enum_value

        return (
            f"    _llk_unpack_A_init_<{broadcast_type}, false, {reuse_dest}, {unpack_to_dest}>(\n"
            f"        {transpose_faces}, {transpose_within_face}, {face_r_dim}, {num_faces}, unpack_a_src_format{stage}, unpack_a_dst_format{stage}\n"
            f"    );\n"
        )

    def unpack(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        tile_idx_expr: str,
    ) -> str:
        stage = operation.stage_id
        unpack_to_dest = "true" if operation.unpack_to_dest else "false"
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value

        return (
            f"_llk_unpack_A_<{broadcast_type}, false, {reuse_dest}, {unpack_to_dest}>(\n"
            f"    L1_ADDRESS(buffer_A{stage}[{tile_idx_expr}]), unpack_a_src_format{stage}, unpack_a_dst_format{stage}\n"
            f");\n"
        )


class UnpackerTilizeA(Unpacker):
    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "llk_unpack_tilize.h",
        ]

    def perf_set_valid(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        valid_cnt = 4 if config.architecture == ChipArchitecture.WORMHOLE else 1
        return f"_perf_unpack_loop_set_valid<true, true>({valid_cnt});\n"

    def perf_clear_valid(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        valid_cnt = 4 if config.architecture == ChipArchitecture.WORMHOLE else 1
        return f"_perf_math_loop_clear_valid<true, true>({valid_cnt});\n"

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tilized_a = tilize_block(
            tensor_a,
            operation.src_a.dimensions,
            operation.src_a.data_format,
            operation.num_faces,
        )

        return tilized_a, None

    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        stage = operation.stage_id
        face_r_dim = operation.face_r_dim
        block_ct_dim = operation.dest_tiles_w
        transpose_faces = compute_unit.unpack_transpose_faces.value
        transpose_within_face = compute_unit.unpack_transpose_within_face.value
        if compute_unit.broadcast_type != BroadcastType.None_:
            raise ValueError("UnpackerTilizeA does not support broadcast")

        if transpose_faces or transpose_within_face:
            raise ValueError("UnpackerTilizeA does not support transpose")

        return f"    _llk_unpack_tilize_init_(unpack_a_src_format{stage}, unpack_a_dst_format{stage}, {block_ct_dim}, {face_r_dim}, false);\n"

    def unpack(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        tile_idx_expr: str,
    ) -> str:
        stage = operation.stage_id
        face_r_dim = operation.face_r_dim
        num_faces = operation.num_faces
        block_ct_dim = operation.dest_tiles_w

        # For tilize, we need to compute row/col from tile_idx
        # Blackhole
        if config.architecture == ChipArchitecture.BLACKHOLE:
            return (
                f"{{\n"
                f"    std::uint32_t row = ({tile_idx_expr}) / {block_ct_dim};\n"
                f"    std::uint32_t col = ({tile_idx_expr}) % {block_ct_dim};\n"
                f"    _llk_unpack_tilize_(L1_ADDRESS(buffer_A{stage}[row * {block_ct_dim}]), col, unpack_a_src_format{stage}, unpack_a_dst_format{stage});\n"
                f"}}\n"
            )

        # Wormhole
        elif config.architecture == ChipArchitecture.WORMHOLE:
            return (
                f"{{\n"
                f"    std::uint32_t row = ({tile_idx_expr}) / {block_ct_dim};\n"
                f"    std::uint32_t col = ({tile_idx_expr}) % {block_ct_dim};\n"
                f"    _llk_unpack_tilize_(L1_ADDRESS(buffer_A{stage}[row * {block_ct_dim}]), col, unpack_a_src_format{stage}, unpack_a_dst_format{stage}, {block_ct_dim}, {face_r_dim}, {num_faces}, false);\n"
                f"}}\n"
            )

        else:
            raise ValueError("Architecture is not supported")

    def uninit(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        stage = operation.stage_id
        face_r_dim = operation.face_r_dim
        num_faces = operation.num_faces

        # Blackhole
        if config.architecture == ChipArchitecture.BLACKHOLE:
            code = f"    _llk_unpack_tilize_uninit_(unpack_a_dst_format{stage}, {num_faces}, {face_r_dim});\n"

        # Wormhole
        elif config.architecture == ChipArchitecture.WORMHOLE:
            code = f"    _llk_unpack_tilize_uninit_(unpack_a_dst_format{stage}, {face_r_dim});\n\n"

        else:
            raise ValueError("Architecture is not supported")

        return code


class ReduceBlockMaxUnpacker(Unpacker):
    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        ct_dim = operation.ct_dim
        dest_acc = config.dest_acc.cpp_enum_value
        if ct_dim > 4:
            raise ValueError("ct_dim must be at most 4 when using Reduce Block Max")
        return f"_llk_unpack_AB_reduce_block_max_row_init_<{ct_dim}, {dest_acc}>();\n"

    def unpack(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        tile_idx_expr: str,
    ) -> str:
        stage = operation.stage_id
        ct_dim = operation.ct_dim

        return (
            f"if (({tile_idx_expr}) % {ct_dim} == 0 ) {{\n"
            f"_llk_unpack_AB_reduce_block_max_row_(L1_ADDRESS(buffer_A{stage}[{tile_idx_expr}]), L1_ADDRESS(buffer_B{stage}[{tile_idx_expr}]));\n"
            f"}}\n"
        )

    def uninit(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        face_r_dim = operation.face_r_dim
        return f"_llk_unpack_AB_reduce_block_max_row_uninit_({face_r_dim}, {face_r_dim});\n"

    def perf_set_valid(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        ct_dim = operation.ct_dim
        return (
            f"_perf_unpack_loop_set_valid<true, false>({ct_dim});\n"
            f"_perf_unpack_loop_set_valid<false, true>(1);\n"
        )

    def perf_clear_valid(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
    ) -> str:
        ct_dim = operation.ct_dim
        return (
            f"_perf_math_loop_clear_valid<true, false>({ct_dim});\n"
            f"_perf_math_loop_clear_valid<false, true>(1);\n"
        )

    def get_headers(self) -> List[str]:
        return ["llk_unpack_AB_reduce_custom.h"]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode" = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tensor_a, tensor_b
