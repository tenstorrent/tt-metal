// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../../../../../tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_A_top32_rm.h"
#include "llk_unpack_common_api.h"

/*******************************************
 * LLK UNPACK A — Top32 row-major transpose
 *******************************************/

inline void llk_unpack_A_top32_rm_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t src_format = get_operand_src_format(operand_id);
    const std::uint32_t dst_format = get_operand_dst_format(operand_id);
    const bool is_int32 = (src_format & 0xf) == (std::uint32_t)DataFormat::Int32;
    if (is_int32) {
        _llk_unpack_A_top32_rm_init_<UnpackToDestEn>(false, src_format, dst_format);
    } else {
        _llk_unpack_A_top32_rm_init_(true, src_format, dst_format);
    }
}

inline void llk_unpack_A_top32_rm(
    const std::uint32_t operand, const std::uint32_t tile_index, const std::uint32_t num_faces) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t src_format = get_operand_src_format(operand_id);
    const std::uint32_t dst_format = get_operand_dst_format(operand_id);
    const std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    const DataFormat dst_format_masked = static_cast<DataFormat>(dst_format & 0x3);
    const std::uint32_t datum_size = dst_format_masked == DataFormat::Float32   ? 4
                                     : dst_format_masked == DataFormat::Float16 ? 2
                                                                                : 1;
    const std::uint32_t offset_address = (64 >> 4) * datum_size * tile_index;  // 64 elements per "tile"
    const std::uint32_t address = base_address + offset_address;

    const bool is_int32 = (src_format & 0xf) == (std::uint32_t)DataFormat::Int32;
    if (is_int32) {
        _llk_unpack_A_top32_rm_<UnpackToDestEn>(num_faces, address, src_format, dst_format);
        llk_unpack_set_srcb_dummy_valid();
    } else {
        _llk_unpack_A_top32_rm_(num_faces, address, src_format, dst_format);
    }
}
