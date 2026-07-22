# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from fuser.fused_operand import Operand
from helpers.format_config import DataFormat
from helpers.llk_params import L1Accumulation


def _emit_pack_buf_desc(
    output: Operand,
    pack_src: DataFormat,
    pack_dst: DataFormat,
) -> str:
    return (
        f"tdma_descriptor_t td_pack = ckernel::trisc::construct_tdma_desc("
        f"{output.tile_shape.cpp_value}, "
        f"{output.cpp_name}[0] / 16, "
        f"{pack_dst.cpp_underlying_value}, "
        f"{output.buf_desc_id}, "
        f"{pack_src.cpp_underlying_value});\n"
        f"_configure_buf_desc_table_(td_pack.buf_desc_id, td_pack.buf_desc);\n"
    )


def hw_configure_pack(
    output: Operand,
    dest_acc: str,
    pack_src: DataFormat,
    pack_dst: DataFormat,
    pack_mode: str = "PackMode::Default",
) -> str:
    code = _emit_pack_buf_desc(output, pack_src, pack_dst)
    code += f"_llk_pack_hw_configure_<p_pacr::PACK0>(td_pack);\n"
    return code


def configure_pack(
    output: Operand,
    dest_acc: str,
    pack_src: DataFormat,
    pack_dst: DataFormat,
) -> str:
    code = "{\n"
    code += _emit_pack_buf_desc(output, pack_src, pack_dst)
    code += f"_llk_pack_hw_configure_<p_pacr::PACK0>(td_pack);\n"
    code += "}\n"
    return code


def relu_config(relu_config_val: int, dest_acc: str) -> str:
    return f"_llk_pack_relu_config_<p_pacr::PACK0, {dest_acc}>(ReluConfig::from_packed({relu_config_val}));\n"


def l1_accumulation_config(pack_l1_accumulation: L1Accumulation) -> str:
    l1_acc = "true" if pack_l1_accumulation == L1Accumulation.Yes else "false"
    return f"_llk_pack_set_l1_acc_<p_pacr::PACK0>({l1_acc});\n"


def pack_dest_init(dest_sync: str, dest_acc: str) -> str:
    return "set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});\n"


def packer_wait_for_math() -> str:
    return ""


def packer_dest_section_done(dest_sync: str, dest_acc: str) -> str:
    return f"_llk_pack_dest_dvalid_section_done_<{dest_sync}, {dest_acc}>();\n"


def packer_sync_with_unpacker(stage_id: int, num_stages: int) -> str:
    return ""
