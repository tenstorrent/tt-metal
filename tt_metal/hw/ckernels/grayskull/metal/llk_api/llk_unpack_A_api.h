// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_A.h"
#include "llk_unpack_common_api.h"

// /*************************************************************************
//  * LLK UNPACK A
//  *************************************************************************/

template <bool is_fp32_dest_acc_en = false /*not used*/, StochRndType stoch_rnd_mode = StochRndType::None /*not used*/>
inline void llk_unpack_A_hw_configure(
    const llk_unpack_A_params_t *unpack_A_params, const int within_face_16x16_transpose = 0) {
    const uint32_t unpA_operand_id = get_operand_id(unpack_A_params->unpA_operand);

    _llk_unpack_A_hw_configure_(
        unpack_src_format[unpA_operand_id],
        unpack_dst_format[unpA_operand_id],
        within_face_16x16_transpose);
}

template <bool is_fp32_dest_acc_en = false /*not used*/, StochRndType stoch_rnd_mode = StochRndType::None /*not used*/>
inline void llk_unpack_A_hw_configure_disaggregated(
    const std::uint32_t unpA_operand, const int within_face_16x16_transpose = 0) {
    const llk_unpack_A_params_t unpack_A_params = {.unpA_operand = unpA_operand};
    llk_unpack_A_hw_configure<is_fp32_dest_acc_en, stoch_rnd_mode>(&unpack_A_params, within_face_16x16_transpose);
}

template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE /*not used*/,
    bool unpack_to_dest = false /*not used*/>
inline void llk_unpack_A_mop_config(
    const bool transpose_of_faces,
    const std::uint32_t operand_id /*not used*/,
    const std::uint32_t unpack_src_format = 0 /*not used*/ ,
    std::uint32_t unpack_dst_format = 0 /*not used*/) {

    _llk_unpack_A_mop_config_<BType, acc_to_dest>(transpose_of_faces > 0);
}

template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false /*not used*/>
inline void llk_unpack_A_init(
    const std::uint32_t transpose_of_faces = 0 /*not used*/,
    const std::uint32_t within_face_16x16_transpose = 0 /*not used*/,
    const std::uint32_t operand = 0 /*not used*/) {

    _llk_unpack_A_init_<BType, acc_to_dest, binary_reuse_dest>(
        transpose_of_faces,
        within_face_16x16_transpose);
}

template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE /*not used*/,
    bool unpack_to_dest = false /*not used*/>
inline void llk_unpack_A(
    const std::uint32_t operand, const std::uint32_t tile_index, const bool transpose_of_faces = 0) {
    // note: unpacker is programmed to automatically skip the tile header (+1)
    // since there is no tile header, we need to -1 the address (in terms of 16B words), to offet unpacker's automatic +1
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = cb_interface[operand_id].fifo_rd_ptr - 1;
    std::uint32_t offset_address = MUL_TILE_SIZE_AND_INDEX<true>((uint)unpack_src_format[operand_id], tile_index);
    std::uint32_t address = base_address + offset_address;

    DEBUG_STATUS("UPAW");
    _llk_unpack_A_<BType, acc_to_dest, binary_reuse_dest>(address, transpose_of_faces > 0);
    DEBUG_STATUS("UPAD");
}


template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false /*not used*/>
inline void llk_unpack_A_block(
    const std::uint32_t operand, const std::uint32_t start_tile_index, const std::uint32_t ntiles, const bool transpose_of_faces = 0) {
    // note: unpacker is programmed to automatically skip the tile header (+1)
    // since there is no tile header, we need to -1 the address (in terms of 16B words), to offet unpacker's automatic +1
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = cb_interface[operand_id].fifo_rd_ptr - 1;
    std::uint32_t offset_address = MUL_TILE_SIZE_AND_INDEX<true>((uint)unpack_src_format[operand_id], 1);
    std::uint32_t address = base_address;

    for (uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        DEBUG_STATUS("UPAW");
        _llk_unpack_A_<BType, acc_to_dest, binary_reuse_dest>(address, transpose_of_faces>0);
        address += offset_address;
        DEBUG_STATUS("UPAD");
    }
}
