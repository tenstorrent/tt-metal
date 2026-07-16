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
    desc = output.cpp_desc_name
    return (
        f"{desc}.reg_data_format = static_cast<std::uint8_t>({pack_src.cpp_underlying_value});\n"
        f"_llk_pack_hw_configure_<p_pacr::PACK0>({desc});\n"
    )


def configure_pack(
    output: Operand,
    dest_acc: str,
    pack_src: DataFormat,
    pack_dst: DataFormat,
) -> str:
    desc = output.cpp_desc_name
    return (
        f"{desc}.reg_data_format = static_cast<std::uint8_t>({pack_src.cpp_underlying_value});\n"
        f"_llk_pack_hw_configure_<p_pacr::PACK0>({desc});\n"
    )


def relu_config(relu_config_val: int, dest_acc: str) -> str:
    return f"_llk_pack_relu_config_<p_pacr::PACK0, {dest_acc}>(ReluConfig::from_packed({relu_config_val}));\n"


def l1_accumulation_config(pack_l1_accumulation: L1Accumulation) -> str:
    l1_acc = "true" if pack_l1_accumulation == L1Accumulation.Yes else "false"
    return f"_llk_pack_set_l1_acc_<p_pacr::PACK0>({l1_acc});\n"


def pack_dest_init(
    dest_sync: str,
    dest_acc: str,
    sfpu_on_dest: bool = False,
) -> str:
    chain = "DestChain::FPU_SFPU_PACK" if sfpu_on_dest else "DestChain::FPU_PACK"
    return f"setup_dest_dvalid<{chain}>();\n"


def packer_wait_for_math() -> str:
    return ""


def packer_dest_section_done(dest_sync: str, dest_acc: str) -> str:
    return f"pack_section_done<{dest_sync}, {dest_acc}>();\n"


def packer_sync_with_unpacker(has_pack_consumer: bool) -> str:
    if has_pack_consumer:
        return "t6_semaphore_post<>(semaphore::PACK_DONE);\n"
    return ""


def pack_reduce_mask_config(operation: "FusedOperation") -> str:
    reduce_dim = operation.reduce_dim.cpp_enum_value
    tensor_shape = operation.tile_shape.cpp_value
    return f"_llk_pack_reduce_mask_config_<{reduce_dim}>({tensor_shape});\n"


def pack_reduce_mask_clear() -> str:
    return "_llk_pack_reduce_mask_clear_();\n"
