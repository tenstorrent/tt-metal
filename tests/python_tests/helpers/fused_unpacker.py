# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List, Tuple

import torch

if TYPE_CHECKING:
    from .fused_operation import FusedOperation

from .chip_architecture import ChipArchitecture
from .fused_math import ReduceFpu
from .golden_generators import TransposeGolden, get_golden_generator
from .llk_params import Transpose
from .tilize_untilize import tilize_block


class Unpacker:
    def sync(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id

        code = ""
        if stage > 0:
            code = (
                "    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);\n"
                "    t6_semaphore_get<>(semaphore::PACK_DONE);\n\n"
            )

        return code

    def hw_configure(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        unpa_tile_size = operation_config.tile_size_unpack_a
        unpb_tile_size = operation_config.tile_size_unpack_b
        dest_acc = operation_config.dest_acc.value
        unpa_face_r_dim = operation_config.face_r_dim
        unpb_face_r_dim = operation_config.face_r_dim
        unpa_num_faces = operation_config.num_faces_A
        unpb_num_faces = operation_config.num_faces_B

        if stage == 0:
            code = (
                f"    _llk_unpack_hw_configure_<{dest_acc}, false>(\n"
                f"        unpack_a_src_format{stage}, unpack_b_src_format{stage}, unpack_a_dst_format{stage}, unpack_b_dst_format{stage},\n"
                f"        {unpa_face_r_dim}, {unpb_face_r_dim}, {unpa_num_faces}, {unpb_num_faces}, {unpa_tile_size}, {unpb_tile_size}\n"
                f"    );\n"
            )
        else:
            code = (
                f"    _llk_unpack_reconfig_data_format_srca_impl_<{dest_acc}, false>(\n"
                f"        unpack_a_src_format{stage}, unpack_a_dst_format{stage}, {unpa_tile_size}\n"
                f"    );\n"
                f"    _llk_unpack_reconfig_data_format_srcb_impl_<{dest_acc}, false>(\n"
                f"        unpack_b_src_format{stage}, unpack_b_dst_format{stage}, {unpb_tile_size}\n"
                f"    );\n"
            )
        return code

    def unpack(self, operation_config: "FusedOperation") -> str:
        return ""

    def uninit(self, operation_config: "FusedOperation") -> str:
        return ""

    def exec(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        buffer_A_address = operation_config.src_a.l1_address
        buffer_B_address = operation_config.src_b.l1_address
        buffer_A_tile_size = operation_config.buffer_A_tile_size
        buffer_B_tile_size = operation_config.buffer_B_tile_size
        unpack_a_src = operation_config.unpack_a_in
        unpack_a_dst = operation_config.unpack_a_out
        unpack_b_src = operation_config.unpack_a_in
        unpack_b_dst = operation_config.unpack_a_out

        code = self.sync(operation_config)

        code += (
            f"    // Operation {stage}: {self.__class__.__name__}\n"
            f"    UNUSED const Operand buffer_A{stage}({hex(buffer_A_address)}, {buffer_A_tile_size});\n"
            f"    UNUSED const Operand buffer_B{stage}({hex(buffer_B_address)}, {buffer_B_tile_size});\n"
            f"    UNUSED const uint32_t unpack_a_src_format{stage} = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{unpack_a_src.name});\n"
            f"    UNUSED const uint32_t unpack_a_dst_format{stage} = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{unpack_a_dst.name});\n"
            f"    UNUSED const uint32_t unpack_b_src_format{stage} = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{unpack_b_src.name});\n"
            f"    UNUSED const uint32_t unpack_b_dst_format{stage} = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{unpack_b_dst.name});\n"
        )

        code += self.hw_configure(operation_config)
        code += self.unpack(operation_config)
        code += self.uninit(operation_config)

        return code

    def get_headers(self) -> List[str]:
        return []

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation_config: "FusedOperation",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tensor_a, tensor_b


class MatmulUnpacker(Unpacker):
    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_AB_matmul.h",
            "llk_unpack_common.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation_config: "FusedOperation",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_matrix = get_golden_generator(TransposeGolden)

        if operation_config.unpack_transpose_faces == Transpose.Yes:
            tensor_b = t_matrix.transpose_faces_multi_tile(
                tensor_b,
                operation_config.src_b.data_format,
                operation_config.src_b.tile_count,
                tilize=True,
                input_dimensions=operation_config.src_b.dimensions,
            )

        if operation_config.unpack_transpose_within_face == Transpose.Yes:
            tensor_b = t_matrix.transpose_within_faces_multi_tile(
                tensor_b,
                operation_config.src_b.data_format,
                operation_config.src_b.tile_count,
                untilize=True,
                input_dimensions=operation_config.src_b.dimensions,
            )

        return tensor_a, tensor_b

    def unpack(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        face_r_dim = operation_config.face_r_dim
        ct_dim = operation_config.ct_dim
        rt_dim = operation_config.rt_dim
        kt_dim = operation_config.kt_dim
        unpack_tile_size_a = operation_config.tile_size_unpack_a
        unpack_tile_size_b = operation_config.tile_size_unpack_b
        transpose_faces = (
            "true" if operation_config.unpack_transpose_faces.value else "false"
        )
        transpose_within_face = (
            "true" if operation_config.unpack_transpose_within_face.value else "false"
        )

        if transpose_within_face != transpose_faces:
            raise ValueError(
                "MatmulUnpacker does not support different values for transpose_faces and transpose_within_face"
            )

        code = (
            f"    _llk_unpack_AB_matmul_init_<>({transpose_faces}, {ct_dim}, {rt_dim}, {kt_dim}, {face_r_dim}, {face_r_dim});\n"
            f"    for (uint32_t j = 0; j < {kt_dim}; j++)\n"
            f"    {{\n"
            f"        _llk_unpack_AB_matmul_<>(\n"
            f"            L1_ADDRESS(buffer_A{stage}[0]), L1_ADDRESS(buffer_B{stage}[0]),\n"
            f"            j, j * {ct_dim}, {unpack_tile_size_a}, {unpack_tile_size_b}, false, false, {ct_dim}, {rt_dim}, {kt_dim}\n"
            f"        );\n"
            f"    }}\n"
            f"\n"
        )

        return code


class UnpackerAB(Unpacker):
    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_AB.h",
            "llk_unpack_common.h",
            "llk_unpack_tilize.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation_config: "FusedOperation",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_matrix = get_golden_generator(TransposeGolden)

        if operation_config.unpack_transpose_faces == Transpose.Yes:
            tensor_a = t_matrix.transpose_faces_multi_tile(
                tensor_a,
                operation_config.src_a.data_format,
                operation_config.src_a.tile_count,
                tilize=True,
                untilize=True,
                input_dimensions=operation_config.src_a.dimensions,
            )

        if operation_config.unpack_transpose_within_face == Transpose.Yes:
            tensor_a = t_matrix.transpose_within_faces_multi_tile(
                tensor_a,
                operation_config.src_a.data_format,
                operation_config.src_a.tile_count,
                tilize=True,
                untilize=True,
                input_dimensions=operation_config.src_a.dimensions,
            )

        return tensor_a.flatten(), tensor_b.flatten()

    def unpack(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        face_r_dim = operation_config.face_r_dim
        num_faces = operation_config.num_faces
        tile_cnt = operation_config.output.tile_count
        broadcast_type = "BroadcastType::NONE"
        transpose_faces = (
            "true" if operation_config.unpack_transpose_faces.value else "false"
        )
        transpose_within_face = (
            "true" if operation_config.unpack_transpose_within_face.value else "false"
        )

        if transpose_within_face != transpose_faces:
            raise ValueError(
                "UnpackerAB does not support different values for transpose_faces and transpose_within_face"
            )

        if isinstance(operation_config.math.fpu, ReduceFpu):
            reduce_dim = operation_config.math.fpu.reduce_dim()
            within_face_16x16_transpose = (
                1 if reduce_dim == "ReduceDim::REDUCE_ROW" else 0
            )
            code = (
                f"    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>({within_face_16x16_transpose});\n"
                f"    constexpr std::uint32_t UNP_SEL = p_setadc::UNP_AB;\n"
                f"    config_unpacker_x_end<UNP_SEL>({face_r_dim});\n"
                f"    _llk_unpack_AB_mop_config_<BroadcastType::NONE>(false, 4, false);\n"
            )
        else:
            code = f"    _llk_unpack_AB_init_<{broadcast_type}>({face_r_dim}, {num_faces}, false, {transpose_faces});\n"

        code += (
            f"    for (int i = 0; i < {tile_cnt}; i++)\n"
            f"    {{\n"
            f"        _llk_unpack_AB_<>(L1_ADDRESS(buffer_A{stage}[i]), L1_ADDRESS(buffer_B{stage}[i]));\n"
            f"    }}\n"
            f"\n"
        )

        return code


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
        operation_config: "FusedOperation",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_matrix = get_golden_generator(TransposeGolden)

        if operation_config.unpack_transpose_faces == Transpose.Yes:
            tensor_a = t_matrix.transpose_faces_multi_tile(
                tensor_a,
                operation_config.src_a.data_format,
                operation_config.src_a.tile_count,
                tilize=True,
                untilize=True,
                input_dimensions=operation_config.src_a.dimensions,
            )

        if operation_config.unpack_transpose_within_face == Transpose.Yes:
            tensor_a = t_matrix.transpose_within_faces_multi_tile(
                tensor_a,
                operation_config.src_a.data_format,
                operation_config.src_a.tile_count,
                tilize=True,
                untilize=True,
                input_dimensions=operation_config.src_a.dimensions,
            )

        return tensor_a.flatten(), tensor_b.flatten()

    def unpack(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        tile_cnt = operation_config.output.tile_count
        unpack_to_dest = "true" if operation_config.unpack_to_dest else "false"
        broadcast_type = "BroadcastType::NONE"
        eltwise_reuse_type = "NONE"
        face_r_dim = operation_config.face_r_dim
        num_faces = operation_config.num_faces
        transpose_faces = (
            "true" if operation_config.unpack_transpose_faces.value else "false"
        )
        transpose_within_face = (
            "true" if operation_config.unpack_transpose_within_face.value else "false"
        )

        code = (
            f"    _llk_unpack_A_init_<{broadcast_type}, false, EltwiseBinaryReuseDestType::{eltwise_reuse_type}, {unpack_to_dest}>(\n"
            f"        {transpose_faces}, {transpose_within_face}, {face_r_dim}, {num_faces}, unpack_a_src_format{stage}, unpack_a_dst_format{stage}\n"
            f"    );\n"
            f"    for (int i = 0; i < {tile_cnt}; ++i)\n"
            f"    {{\n"
            f"        _llk_unpack_A_<{broadcast_type}, false, EltwiseBinaryReuseDestType::NONE, {unpack_to_dest}>(\n"
            f"            L1_ADDRESS(buffer_A{stage}[i]), unpack_a_src_format{stage}, unpack_a_dst_format{stage}\n"
            f"        );\n"
            f"    }}\n\n"
        )

        return code


class UnpackerTilizeA(Unpacker):
    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "llk_unpack_tilize.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation_config: "FusedOperation",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tilized_a = tilize_block(
            tensor_a,
            operation_config.src_a.dimensions,
            operation_config.src_a.data_format,
            operation_config.num_faces,
        )

        return tilized_a, tensor_b

    def uninit(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        face_r_dim = operation_config.face_r_dim
        num_faces = operation_config.num_faces

        # Blackhole
        if operation_config.architecture == ChipArchitecture.BLACKHOLE:
            code = f"    _llk_unpack_tilize_uninit_(unpack_a_dst_format{stage}, {num_faces}, {face_r_dim});\n"

        # Wormhole
        elif operation_config.architecture == ChipArchitecture.WORMHOLE:
            code = f"    _llk_unpack_tilize_uninit_(unpack_a_dst_format{stage}, {face_r_dim});\n\n"

        else:
            raise ValueError("Architecture is not supported")

        return code

    def unpack(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        face_r_dim = operation_config.face_r_dim
        num_faces = operation_config.num_faces
        block_rt_dim = operation_config.block_rt_dim
        block_ct_dim = operation_config.block_ct_dim
        transpose_faces = operation_config.unpack_transpose_faces.value
        transpose_within_face = operation_config.unpack_transpose_within_face.value

        if transpose_faces or transpose_within_face:
            raise ValueError("UnpackerTilizeA does not support transpose")

        code = f"    _llk_unpack_tilize_init_(unpack_a_src_format{stage}, unpack_a_dst_format{stage}, {block_ct_dim}, {face_r_dim}, false);\n"

        # Blackhole
        if operation_config.architecture == ChipArchitecture.BLACKHOLE:
            code += (
                f"    for (uint32_t i = 0; i < {block_rt_dim}; i++)\n"
                f"    {{\n"
                f"        for (uint32_t j = 0; j < {block_ct_dim}; j++)\n"
                f"        {{\n"
                f"            _llk_unpack_tilize_(L1_ADDRESS(buffer_A{stage}[i * {block_rt_dim}]), j, unpack_a_src_format{stage});\n"
                f"        }}\n"
                f"    }}\n"
            )

        # Wormhole
        elif operation_config.architecture == ChipArchitecture.WORMHOLE:
            code += (
                f"    for (uint32_t i = 0; i < {block_rt_dim}; i++)\n"
                f"    {{\n"
                f"        for (uint32_t j = 0; j < {block_ct_dim}; j++)\n"
                f"        {{\n"
                f"            _llk_unpack_tilize_(L1_ADDRESS(buffer_A{stage}[i * {block_rt_dim}]), j, unpack_a_src_format{stage}, {block_ct_dim}, {face_r_dim}, {num_faces}, false);\n"
                f"        }}\n"
                f"    }}\n"
            )

        else:
            raise ValueError("Architecture is not supported")

        return code
