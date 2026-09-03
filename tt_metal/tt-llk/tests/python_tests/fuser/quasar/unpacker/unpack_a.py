# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.base_unpacker import Unpacker
from fuser.block_data import BlockData, InvocationGranularity
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation
from helpers.llk_params import DestAccumulation, EltwiseBinaryReuseDestType


def _uses_upk_to_dest_semaphores(config: GlobalConfig) -> bool:
    from helpers.llk_params import PerfRunType

    return not config.quasar_use_dvalid and config.perf_run_type in (
        None,
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    )


def upk_to_dest_math_ack() -> str:
    return (
        "_llk_sync_wait_<p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO>(semaphore::UNPACK_MATH);\n"
        "_llk_sync_get_<p_stall::MATH, p_stall::WAIT_SFPU>(semaphore::UNPACK_MATH);\n"
    )


def _unp_sel(compute_unit: FpuNode) -> str:
    if compute_unit.unpack_to_dest.value:
        return "p_unpacr::UNP_DEST"
    if compute_unit.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
        return "p_unpacr::UNP_B"
    return "p_unpacr::UNP_A"


class UnpackerA(Unpacker):
    granularity = InvocationGranularity.ROW
    per_block_init = True

    def __init__(
        self, reuse_dest: EltwiseBinaryReuseDestType = EltwiseBinaryReuseDestType.NONE
    ):
        self.reuse_dest = reuse_dest
        if reuse_dest != EltwiseBinaryReuseDestType.NONE:
            self.granularity = InvocationGranularity.TILE

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "llk_unpack_unary_operand.h",
            "llk_math_common.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor_a = self.transpose_golden(tensor_a, config, operation, compute_unit)

        tensor_a, tensor_b = self.reuse_dest_golden(
            tensor_a, tensor_b, config, operation, compute_unit
        )

        return tensor_a, tensor_b

    def _perf_valid_args(
        self,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> Tuple[str, str, int]:
        if compute_unit.reuse_dest != EltwiseBinaryReuseDestType.NONE:
            num_faces = compute_unit.src_a.tile_shape.total_num_faces()
            return "true", "true", num_faces
        if config.dest_acc == DestAccumulation.Yes:
            return "true", "true", block.block_tiles_x
        return "true", "false", block.block_tiles_x

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        if compute_unit.unpack_to_dest.value:
            return ""
        set_a, set_b, count = self._perf_valid_args(config, compute_unit, block)
        return f"_perf_unpack_loop_set_valid<{set_a}, {set_b}>({count});\n"

    def perf_clear_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        if compute_unit.unpack_to_dest.value:
            return upk_to_dest_math_ack()
        clear_a, clear_b, count = self._perf_valid_args(config, compute_unit, block)
        return f"_perf_math_loop_clear_valid<{clear_a}, {clear_b}>({count});\n"

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        buf_desc_id = compute_unit.src_a.buf_desc_id
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        en_32bit_dest = config.dest_acc.cpp_enum_value
        unpack_to_dest = compute_unit.unpack_to_dest.cpp_enum_value
        transpose_en = compute_unit.transpose_faces.cpp_enum_value
        unp_sel = _unp_sel(compute_unit)
        num_tiles = (
            1
            if compute_unit.reuse_dest != EltwiseBinaryReuseDestType.NONE
            else block.block_tiles_x
        )

        return (
            f"_llk_unpack_unary_operand_init_<{unp_sel}, {transpose_en}, {en_32bit_dest}, {reuse_dest}, {unpack_to_dest}>"
            f"({buf_desc_id}, {tensor_shape}, {num_tiles});\n"
        )

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        unp_sel = _unp_sel(compute_unit)
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        unpack_to_dest = compute_unit.unpack_to_dest.cpp_enum_value
        dest_sync = operation.dest_sync.cpp_enum_value

        return (
            f"_llk_unpack_unary_operand_<{unp_sel}, {reuse_dest}, {unpack_to_dest}, {dest_sync}>"
            f"({block.tile_id_global}, {tensor_shape});\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
