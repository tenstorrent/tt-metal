# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

import torch

from .chip_architecture import ChipArchitecture

if TYPE_CHECKING:
    from .fused_operation import FusedOperation


class Packer:
    def get_headers(self) -> List[str]:
        return [
            "llk_pack.h",
            "llk_pack_common.h",
        ]

    def golden(
        self,
        tensor: torch.Tensor,
        operation_config: "FusedOperation",
    ) -> torch.Tensor:
        return tensor

    def hw_configure(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        bh_tilize = "true" if operation_config.bh_tilize.value else "false"
        dest_acc = operation_config.dest_acc.value
        pack_size = operation_config.tile_size_pack

        if stage == 0:
            if operation_config.architecture == ChipArchitecture.BLACKHOLE:
                code = (
                    f"    _llk_pack_hw_configure_<{dest_acc}, false, {bh_tilize}>(\n"
                    f"        pack_in_format{stage}, pack_out_format{stage}, {pack_size}\n"
                    f"    );\n"
                )
            elif operation_config.architecture == ChipArchitecture.WORMHOLE:
                code = (
                    f"    _llk_pack_hw_configure_<{dest_acc}, false>(\n"
                    f"        pack_in_format{stage}, pack_out_format{stage}, {pack_size}\n"
                    f"    );\n"
                )
        else:
            code = (
                f"    _llk_pack_reconfig_data_format_<{dest_acc}, false>(\n"
                f"        pack_in_format{stage}, pack_out_format{stage}, {pack_size}\n"
                f"    );\n"
            )

        return code

    def pack(self, operation_config: "FusedOperation") -> str:
        stage = operation_config.stage_id
        num_stages = operation_config.num_stages
        pack_src = operation_config.pack_in
        pack_dst = operation_config.pack_out
        result_buffer_address = operation_config.output.l1_address
        tile_cnt = operation_config.output.tile_count
        dest_acc = operation_config.dest_acc
        dest_acc_value = dest_acc.value
        buffer_Res_tile_size = operation_config.buffer_Res_tile_size
        bh_tilize = "true" if operation_config.bh_tilize.value else "false"

        code = (
            f"    // Operation {stage}: Packer\n"
            f"    const Operand buffer_Res{stage}({hex(result_buffer_address)}, {buffer_Res_tile_size});\n"
            f"    const uint32_t pack_in_format{stage} = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{pack_src.name});\n"
            f"    const uint32_t pack_out_format{stage} = static_cast<std::underlying_type_t<DataFormat>>(DataFormat::{pack_dst.name});\n"
        )

        code += self.hw_configure(operation_config)

        if operation_config.architecture == ChipArchitecture.BLACKHOLE:
            code += (
                f"    _llk_pack_init_<false, false, {bh_tilize}>(\n"
                f"        pack_out_format{stage}\n"
                f"    );\n"
                f"    _llk_pack_dest_init_<DstSync::SyncHalf, {dest_acc_value}>();\n"
            )
        elif operation_config.architecture == ChipArchitecture.WORMHOLE:
            code += (
                f"    _llk_pack_init_<false, false>(\n"
                f"        pack_out_format{stage}\n"
                f"    );\n"
                f"    _llk_pack_dest_init_<DstSync::SyncHalf, {dest_acc_value}, false>();\n"
            )
        else:
            raise ValueError("Unsupported architecture for packer")

        code += (
            f"    _llk_packer_wait_for_math_done_();\n"
            f"    for (int i = 0; i < {tile_cnt}; i++)\n"
            f"    {{\n"
            f"        _llk_pack_<DstSync::SyncHalf, {dest_acc_value}, false>(i, L1_ADDRESS(buffer_Res{stage}[i]));\n"
            f"    }}\n"
            f"    _llk_pack_dest_section_done_<DstSync::SyncHalf, {dest_acc_value}>();\n"
        )

        if stage < num_stages - 1:
            code += f"    t6_semaphore_post<>(semaphore::PACK_DONE);\n\n"

        return code
