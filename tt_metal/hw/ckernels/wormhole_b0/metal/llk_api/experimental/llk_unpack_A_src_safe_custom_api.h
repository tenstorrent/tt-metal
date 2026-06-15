// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "experimental/llk_unpack_A_src_safe_custom.h"
#include "llk_defs.h"
#include "llk_io.h"
#include "llk_operands.h"
#include "llk_param_structs.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

/*************************************************************************
 * LLK UNPACK A - SRC-SAFE CUSTOM (Wormhole B0)
 *
 * Drop-in replacement for llk_unpack_A / llk_unpack_A_block that swaps in the
 * src-safe variant of unpack_to_dest_tile_done, preventing the TEN-3868
 * workaround from issuing an unpack-to-SrcA with INT32 / UInt32 formats
 * (UndefinedBehavior per the FormatConversion spec). The init path is
 * unchanged and is intentionally not duplicated here; consumers call the
 * production llk_unpack_A_init.
 *************************************************************************/

template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_src_safe_custom(const std::uint32_t operand, const std::uint32_t tile_index) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    const std::uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size * tile_index;
    const std::uint32_t address = base_address + offset_address;

    LLK_ASSERT(cb_access_within_bounds(operand_id, tile_index, 1), "Indexed tile read exceeds CB boundary");

    LLK_ASSERT_BLOCK((is_unpacker_A_configured_correctly<
                      UnpackerProgramType::ProgramByTile,
                      (BType != BroadcastType::NONE && !unpack_to_dest)>(
        unpack_src_format[operand_id],
        unpack_dst_format[operand_id],
        get_operand_face_r_dim(operand_id),
        get_operand_num_faces(operand_id))));

    WAYPOINT("UPAW");
    _llk_unpack_A_src_safe_custom_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        address, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
    WAYPOINT("UPAD");
}

template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_block_src_safe_custom(
    const std::uint32_t operand, const std::uint32_t start_tile_index, const std::uint32_t ntiles) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    const std::uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size;
    std::uint32_t address = base_address + start_tile_index * offset_address;

    LLK_ASSERT(cb_access_within_bounds(operand_id, start_tile_index, ntiles), "Block tile read exceeds CB boundary");

    for (std::uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        WAYPOINT("UPAW");
        _llk_unpack_A_src_safe_custom_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
            address, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
        address += offset_address;
        WAYPOINT("UPAD");
    }
}
