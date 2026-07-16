# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from fuser.fused_operand import Operand
from helpers.format_config import DataFormat
from helpers.llk_params import L1Accumulation


def hw_configure_pack(
    output: Operand,
    dest_acc: str,
    pack_src: DataFormat,
    pack_dst: DataFormat,
    pack_mode: str = "PackMode::Default",
) -> str:
    return (
        f"_llk_pack_hw_configure_<{dest_acc}, PackMode::Default>(\n"
        f"    {pack_src.cpp_underlying_value}, {pack_dst.cpp_underlying_value}, {output.tile_size}, {output.tile_shape.face_r_dim}, {output.tile_shape.total_num_faces()}\n"
        f");\n"
    )


def configure_pack(
    output: Operand,
    dest_acc: str,
    pack_src: DataFormat,
    pack_dst: DataFormat,
) -> str:
    return (
        f"_llk_pack_reconfig_data_format_<{dest_acc}>(\n"
        f"    {pack_src.cpp_underlying_value}, {pack_dst.cpp_underlying_value}, {output.tile_size}\n"
        f");\n"
    )


def relu_config(relu_config_val: int, dest_acc: str) -> str:
    return f"_llk_pack_relu_config_(ReluConfig::from_packed({relu_config_val}));\n"


def l1_accumulation_config(pack_l1_accumulation: L1Accumulation) -> str:
    l1_acc = pack_l1_accumulation.cpp_enum_value
    return f"_llk_pack_reconfig_l1_acc_({l1_acc});\n"


def pack_dest_init(dest_sync: str, dest_acc: str, **kwargs) -> str:
    return f"_llk_pack_dest_init_<{dest_sync}, {dest_acc}>();\n"


def packer_wait_for_math() -> str:
    return "_llk_packer_wait_for_math_done_();\n"


def packer_dest_section_done(dest_sync: str, dest_acc: str) -> str:
    return f"_llk_pack_dest_section_done_<{dest_sync}, {dest_acc}>();\n"


def packer_sync_with_unpacker(has_pack_consumer: bool) -> str:
    if has_pack_consumer:
        return "t6_semaphore_post<>(semaphore::PACK_DONE);\n\n"
    return ""


def pack_reduce_mask_config(operation: "FusedOperation") -> str:
    reduce_dim = operation.reduce_dim.cpp_enum_value
    return f"_llk_pack_reduce_mask_config_<{reduce_dim}>();\n"


def pack_reduce_mask_clear() -> str:
    return "_llk_pack_reduce_mask_clear_();\n"
