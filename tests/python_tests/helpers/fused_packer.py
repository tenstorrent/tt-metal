# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

import torch

from .chip_architecture import ChipArchitecture

if TYPE_CHECKING:
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig

from .fused_fpu import ReduceBlockMaxFpu, ReduceFpu
from .llk_params import PerfRunType


class Packer:
    def get_headers(self) -> List[str]:
        return [
            "llk_pack.h",
            "llk_pack_common.h",
            "perf.h",
        ]

    def golden(
        self,
        tensor: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ) -> torch.Tensor:
        return tensor

    def _wait_for_math(self) -> str:
        return "_llk_packer_wait_for_math_done_();\n"

    def _dest_section_done(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        dest_sync = f"DstSync::Sync{operation.dest_sync.name}"
        return (
            f"_llk_pack_dest_section_done_<{dest_sync}, {config.dest_acc.value}>();\n"
        )

    def _batch_loop(self, operation: "FusedOperation", config: "GlobalConfig") -> str:
        batch_size = operation.batch_size
        tile_cnt = operation.output.tile_count

        num_full_batches = tile_cnt // batch_size
        remaining_tiles = tile_cnt % batch_size

        code = ""

        if num_full_batches > 0:
            code += f"for (std::uint32_t batch = 0; batch < {num_full_batches}; ++batch) {{\n"
            code += self._wait_for_math()
            code += f"for (std::uint32_t i = 0; i < {batch_size}; ++i) {{\n"
            code += f"std::uint32_t tile_idx = batch * {batch_size} + i;\n"
            code += self.pack(operation, config, "i", "tile_idx")
            code += "}\n"
            code += self._dest_section_done(operation, config)
            code += "}\n"

        if remaining_tiles > 0:
            code += self._wait_for_math()
            code += f"for (std::uint32_t i = 0; i < {remaining_tiles}; ++i) {{\n"
            code += f"std::uint32_t tile_idx = {num_full_batches * batch_size} + i;\n"
            code += self.pack(operation, config, "i", "tile_idx")
            code += "}\n"
            code += self._dest_section_done(operation, config)

        return code

    def pack_with_perf(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        if config.perf_run_type in (
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
        ):
            return ""
        if config.perf_run_type in (
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ):
            return self.pack(operation, config, 0, 0)
        return self._batch_loop(operation, config)

    def exec_perf(self, operation: "FusedOperation", config: "GlobalConfig") -> str:
        code = "{\n"
        code += '    ZONE_SCOPED("INIT")\n'
        code += self.hw_configure(operation, config)
        code += self.init(operation, config)
        code += "    PROFILER_SYNC();\n"
        code += "}\n"

        code += "{\n"
        code += '    ZONE_SCOPED("TILE_LOOP")\n'
        code += f"    for(int loop = 0; loop < {config.loop_factor}; loop++)\n"
        code += "    {\n"
        code += self.pack_with_perf(operation, config)
        code += "    }\n"

        code += self.unpacker_sync(operation, config)

        code += "    PROFILER_SYNC();\n"
        code += "}\n"

        code += "{\n"
        code += '    ZONE_SCOPED("INIT")\n'
        code += self.uninit(operation, config)
        code += "    PROFILER_SYNC();\n"
        code += "}\n"

        return code

    def exec(self, operation: "FusedOperation", config: "GlobalConfig") -> str:
        stage = operation.stage_id
        buffer_Res_tile_size = operation.buffer_Res_tile_size
        pack_src = operation.pack_in
        pack_dst = operation.pack_out
        result_buffer_address = operation.output.l1_address

        code = (
            f"    // Operation {stage}: Packer\n"
            f"    const Operand buffer_Res{stage}({hex(result_buffer_address)}, {buffer_Res_tile_size});\n"
            f"    const std::uint32_t pack_src_format{stage} = ckernel::to_underlying(DataFormat::{pack_src.name});\n"
            f"    const std::uint32_t pack_dst_format{stage} = ckernel::to_underlying(DataFormat::{pack_dst.name});\n"
        )

        if config.profiler_enabled:
            code += self.exec_perf(operation, config)
        else:
            code += self.hw_configure(operation, config)
            code += self.init(operation, config)
            code += self._batch_loop(operation, config)
            code += self.unpacker_sync(operation, config)
            code += self.uninit(operation, config)

        return code

    def hw_configure(self, operation: "FusedOperation", config: "GlobalConfig") -> str:
        stage = operation.stage_id
        bh_tilize = "true" if operation.bh_tilize.value else "false"
        dest_acc = config.dest_acc.value
        pack_size = operation.tile_size_pack
        face_r_dim = operation.face_r_dim
        num_faces = operation.num_faces

        if stage == 0:
            if config.architecture == ChipArchitecture.BLACKHOLE:
                code = (
                    f"    _llk_pack_hw_configure_<{dest_acc}, false, {bh_tilize}>(\n"
                    f"        pack_src_format{stage}, pack_dst_format{stage}, {pack_size}, {face_r_dim}, TILE_C_DIM, {num_faces}\n"
                    f"    );\n"
                )
            elif config.architecture == ChipArchitecture.WORMHOLE:
                code = (
                    f"    _llk_pack_hw_configure_<{dest_acc}, false>(\n"
                    f"        pack_src_format{stage}, pack_dst_format{stage}, {pack_size}, {face_r_dim}, {num_faces}\n"
                    f"    );\n"
                )
        else:
            code = (
                f"    _llk_pack_reconfig_data_format_<{dest_acc}, false>(\n"
                f"        pack_src_format{stage}, pack_dst_format{stage}, {pack_size}\n"
                f"    );\n"
            )

        return code

    def init(self, operation: "FusedOperation", config: "GlobalConfig") -> str:
        stage = operation.stage_id
        dest_acc = config.dest_acc.value
        bh_tilize = "true" if operation.bh_tilize.value else "false"
        face_r_dim = operation.face_r_dim
        num_faces = operation.num_faces
        dest_sync = f"DstSync::Sync{operation.dest_sync.name}"
        if config.architecture == ChipArchitecture.BLACKHOLE:
            code = (
                f"    _llk_pack_init_<false, false, {bh_tilize}>(\n"
                f"        pack_dst_format{stage}, pack_dst_format{stage}, {face_r_dim}, TILE_C_DIM, {num_faces}, false, false\n"
                f"    );\n"
                f"    _llk_pack_dest_init_<{dest_sync}, {dest_acc}>();\n"
            )
        elif config.architecture == ChipArchitecture.WORMHOLE:
            code = (
                f"    _llk_pack_init_<false, false>(\n"
                f"        pack_dst_format{stage}, {face_r_dim}, {num_faces}\n"
                f"    );\n"
                f"    _llk_pack_dest_init_<{dest_sync}, {dest_acc}, false>();\n"
            )
        else:
            raise ValueError("Unsupported architecture for packer")

        if operation.math.has_fpu(ReduceFpu):
            reduce_dim = operation.math.get_reduce_pack_mask()
            code += f"    _llk_pack_reduce_mask_config_<false, {reduce_dim}>();\n"
        elif operation.math.has_fpu(ReduceBlockMaxFpu):
            reduce_dim = "ckernel::ReduceDim::REDUCE_ROW"
            code += f"    _llk_pack_reduce_mask_config_<false, {reduce_dim}>();\n"

        return code

    def pack(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        dest_idx,
        l1_idx,
    ) -> str:
        stage = operation.stage_id
        dest_acc = config.dest_acc.value
        dest_sync = f"DstSync::Sync{operation.dest_sync.name}"
        return f"_llk_pack_<{dest_sync}, {dest_acc}, false>({dest_idx}, L1_ADDRESS(buffer_Res{stage}[{l1_idx}]));\n"

    def uninit(self, operation: "FusedOperation", config: "GlobalConfig") -> str:
        code = ""

        if operation.math.has_fpu(ReduceFpu) or operation.math.has_fpu(
            ReduceBlockMaxFpu
        ):
            code = "    _llk_pack_reduce_mask_clear_();\n"

        return code

    def unpacker_sync(self, operation: "FusedOperation", config: "GlobalConfig") -> str:
        stage = operation.stage_id
        num_stages = operation.num_stages
        code = ""

        if stage < num_stages - 1:
            code += "    t6_semaphore_post<>(semaphore::PACK_DONE);\n\n"

        return code
