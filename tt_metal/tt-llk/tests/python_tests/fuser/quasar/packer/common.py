# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from fuser.fused_operand import Operand
from helpers.format_config import DataFormat
from helpers.llk_params import L1Accumulation


def _fmt(data_format: DataFormat) -> str:
    return f"ckernel::to_underlying(DataFormat::{data_format.name})"


def hw_configure_pack(
    output: Operand,
    dest_acc: str,
    pack_src: DataFormat,
    pack_dst: DataFormat,
    pack_mode: str = "PackMode::Default",
) -> str:
    face_r_dim = output.tile_shape.face_r_dim
    face_c_dim = output.tile_shape.face_c_dim
    num_faces_r = output.tile_shape.total_row_dim() // face_r_dim
    num_faces_c = output.tile_shape.total_col_dim() // face_c_dim

    code = ""
    code += f"buffer_descriptor_u bd_pack {{}};\n"
    code += f"bd_pack.f.l1_addr_16B = {output.cpp_name}[0] / 16;\n"
    code += f"bd_pack.f.format = static_cast<std::uint8_t>({_fmt(pack_dst)});\n"
    code += f"bd_pack.f.x_dim = {face_c_dim};\n"
    code += f"bd_pack.f.y_dim = {face_r_dim};\n"
    code += f"bd_pack.f.z_dim = {num_faces_r * num_faces_c};\n"
    code += f"tdma_descriptor_t td_pack;\n"
    code += f"td_pack.buf_desc = bd_pack;\n"
    code += f"td_pack.buf_desc_id = {output.buf_desc_id};\n"
    code += f"td_pack.reg_data_format = static_cast<std::uint8_t>({_fmt(pack_src)});\n"
    code += f"_configure_buf_desc_table_(td_pack.buf_desc_id, td_pack.buf_desc);\n"
    code += f"_llk_pack_hw_configure_<p_pacr::PACK0>(td_pack);\n"
    return code


def configure_pack(
    output: Operand,
    dest_acc: str,
    pack_src: DataFormat,
    pack_dst: DataFormat,
) -> str:
    face_r_dim = output.tile_shape.face_r_dim
    face_c_dim = output.tile_shape.face_c_dim
    num_faces = output.tile_shape.total_num_faces()

    code = "{\n"
    code += f"buffer_descriptor_u bd_pack {{}};\n"
    code += f"bd_pack.f.l1_addr_16B = {output.cpp_name}[0] / 16;\n"
    code += f"bd_pack.f.format = static_cast<std::uint8_t>({_fmt(pack_dst)});\n"
    code += f"bd_pack.f.x_dim = {face_c_dim};\n"
    code += f"bd_pack.f.y_dim = {face_r_dim};\n"
    code += f"bd_pack.f.z_dim = {num_faces};\n"
    code += f"tdma_descriptor_t td_pack;\n"
    code += f"td_pack.buf_desc = bd_pack;\n"
    code += f"td_pack.buf_desc_id = {output.buf_desc_id};\n"
    code += f"td_pack.reg_data_format = static_cast<std::uint8_t>({_fmt(pack_src)});\n"
    code += f"_configure_buf_desc_table_(td_pack.buf_desc_id, td_pack.buf_desc);\n"
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
