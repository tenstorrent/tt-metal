// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_bfd_alloc.h"
#include "llk_unpack_common_api.h"
#include "llk_unpack_unary_broadcast_operands.h"
#include "llk_unpack_unary_operand.h"
#include "llk_unpack_unary_operand_to_dest.h"
#include "tensor_shape.h"

/*************************************************************************
 * LLK UNPACK A
 *************************************************************************/

/**
 *
 * @brief Initialize unpacker for unary / unary-broadcast / binary-dest-reuse paths.
 *
 * Overload matching Blackhole/Wormhole API signature `(transpose_of_faces, within_face_16x16_transpose, operand)`.
 *
 * When `binary_reuse_dest != NONE`, uses the eltwise-binary dest-reuse init path (UNP_A, default tile/face counts).
 * Otherwise uses the unary / unary-broadcast path. For the non-broadcast path the UNP_DEST routing
 * decision is made solely from the `unpack_to_dest` template parameter (no format inspection); when true,
 * all operands (including 16-bit) are routed to DEST through `_llk_unpack_unary_operand_to_dest_init_`, the
 * semaphore-synchronized unpack-to-dest primitive; otherwise `_llk_unpack_unary_operand_init_` targets UNP_A.
 *
 * @tparam BType: Broadcast type; BroadcastType::NONE selects the plain unary path
 * @tparam acc_to_dest: Unused on Quasar in dest-reuse path; kept for API parity
 * @tparam binary_reuse_dest: Dest reuse mode; when not NONE, selects the dest-reuse sub-path
 * @tparam unpack_to_dest: When true, the (non-broadcast) primitive routes the operand through UNP_DEST regardless of
 * format
 * @param transpose_of_faces: Non-zero enables transpose of 16x16 faces (unary/broadcast NONE path only)
 * @param within_face_16x16_transpose: Unused on Quasar; kept for API parity with Blackhole / other arches
 * @param operand: The input operand logical dataflow buffer / CB id
 */
template <
    BroadcastType BType = BroadcastType::NONE,
    [[maybe_unused]] bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_init(
    const std::uint32_t transpose_of_faces,
    const std::uint32_t within_face_16x16_transpose,
    const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    // Unused on the unpack_to_dest path: UNP_DEST does not support tiny tiles, so the primitive takes no shape.
    [[maybe_unused]] const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
    if constexpr (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) {
        static_assert(unpack_to_dest == false, "unpack_to_dest is not yet supported on Quasar");
        static_assert(acc_to_dest == false, "acc_to_dest is not yet supported on Quasar");
        static_assert(BType == BroadcastType::NONE, "On Quasar, only BroadcastType::NONE is supported for dest reuse");

        // For Quasar, the unp_sel field is ignored if binary_reuse_dest != EltwiseBinaryReuseDestType::NONE
        // CB_UNP in the reuse-dest MOP is UNP_B for DEST_TO_SRCA, UNP_A otherwise — program Unp1/Unp0 accordingly
        constexpr ckernel::trisc::BfdResource engine = binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA
                                                           ? ckernel::trisc::BfdResource::Unp1
                                                           : ckernel::trisc::BfdResource::Unp0;
        llk_unpack_program_bfd<engine>(operand_id);
        _llk_unpack_unary_operand_init_<
            p_unpacr::UNP_A,
            false /* TRANSPOSE_EN */,
            false /* IS_32b_DEST_EN */,
            binary_reuse_dest>(ckernel::trisc::bfd_current<engine>(), tensor_shape, 1);
    } else {
        if constexpr (BType == BroadcastType::NONE) {
            LLK_ASSERT(
                transpose_of_faces == within_face_16x16_transpose,
                "Quasar unpack unary supports only full transpose (transpose_of_faces and within_face_16x16_transpose "
                "must match)");
            llk_unpack_program_bfd<ckernel::trisc::BfdResource::Unp0>(operand_id);
            // Route to UNP_DEST purely on the op-writer flag (no format inspection). A 16-bit
            // operand is unpacked to DEST here too when the op writer requested it.
            if constexpr (unpack_to_dest) {
                // One tile per DEST bank section (block_ct_dim = 1); llk_unpack_A / llk_unpack_A_block loop per
                // tile. Transpose is not supported on UNP_DEST and is forced off inside the primitive.
                _llk_unpack_unary_operand_to_dest_init_(
                    ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), 1 /*block_ct_dim*/);
            } else {
                if (transpose_of_faces && within_face_16x16_transpose) {
                    _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, true, DST_ACCUM_MODE, binary_reuse_dest>(
                        ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), tensor_shape, 1);
                } else {
                    _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false, DST_ACCUM_MODE, binary_reuse_dest>(
                        ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), tensor_shape, 1);
                }
            }
        } else {
            static_assert(
                !(DST_ACCUM_MODE && !unpack_to_dest),
                "32BIT_DEST is not supported for broadcast when unpack_to_dest is false");
            // Unlike the unary path above, the broadcast LLK takes no TensorShape, so it does not scale
            // its L1 tile index by the face count. Full-tile only until it is converted (tt-metal #47597).
            LLK_ASSERT(
                tensor_shape.face_r_dim == MAX_FACE_R_DIM && tensor_shape.num_faces_r_dim == MAX_NUM_FACES_R_DIM &&
                    tensor_shape.num_faces_c_dim == MAX_NUM_FACES_C_DIM,
                "Unary broadcast currently only supports 32x32 tiles (face_r_dim=16, 2x2 faces)");
            constexpr std::uint32_t unp_sel = unpack_to_dest ? p_unpacr::UNP_A : p_unpacr::UNP_B;
            constexpr ckernel::trisc::BfdResource engine =
                unpack_to_dest ? ckernel::trisc::BfdResource::Unp0 : ckernel::trisc::BfdResource::Unp1;
            llk_unpack_program_bfd<engine>(operand_id);
            _llk_unpack_unary_broadcast_operands_init_<unp_sel, BType, unpack_to_dest>(
                ckernel::trisc::bfd_current<engine>(), 1);
        }
    }
}

/**
 *
 * @brief Unpacks a single operand for unary and unary-broadcast paths.
 *
 * For the non-broadcast path the UNP_DEST routing decision is made solely from the `unpack_to_dest` template
 * parameter (no format inspection): when true this calls `_llk_unpack_unary_operand_to_dest_`, which carries the
 * UNPACK_MATH / MATH_PACK semaphore handshake and the SyncHalf bank flip; otherwise `_llk_unpack_unary_operand_`
 * on UNP_A.
 *
 * @tparam BType: Broadcast type; BroadcastType::NONE selects the plain unary path
 * @tparam acc_to_dest: Unused on Quasar; kept for API parity with Blackhole / other arches
 * @tparam binary_reuse_dest: Dest reuse mode (unary path only)
 * @tparam unpack_to_dest: when true, the (non-broadcast) primitive routes the operand through UNP_DEST regardless of
 * format
 * @param operand: The logical dataflow buffer id
 * @param tile_index: The index in the input CB to read from
 */
template <
    BroadcastType BType = BroadcastType::NONE,
    [[maybe_unused]] bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index) {
    LLK_TDMA_GUARD_NOTE_TDMA(operand);  // TEN-4746: real unpack (UNPACR) disarms this dfb
    WAYPOINT("UPAW");
    const std::uint32_t operand_id = get_operand_id(operand);
    const LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(operand_id);
    const std::uint32_t l1_tile_idx =
        local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_entry_idx + tile_index;
    if constexpr (BType == BroadcastType::NONE) {
        if constexpr (unpack_to_dest) {
            // EN_32BIT_DEST sizes the SyncHalf bank flip and is pinned to true regardless of DST_ACCUM_MODE: the pack
            // side (llk_pack_dest_section_done) pins it the same way, and the two must agree or unpack and pack
            // address different DEST halves. A 16-bit unpack-to-dest just uses half of each bank. Change both together.
            _llk_unpack_unary_operand_to_dest_<DST_SYNC_MODE, true /*EN_32BIT_DEST*/>(l1_tile_idx);
        } else {
            const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
            _llk_unpack_unary_operand_<p_unpacr::UNP_A, binary_reuse_dest>(l1_tile_idx, tensor_shape);
        }
    } else {
        constexpr std::uint32_t unp_sel = unpack_to_dest ? p_unpacr::UNP_A : p_unpacr::UNP_B;
        _llk_unpack_unary_broadcast_operands_<unp_sel, unpack_to_dest>(l1_tile_idx);
    }
    WAYPOINT("UPAD");
}

/**
 * @brief Unpacks a contiguous block of tiles for unary and unary-broadcast paths.
 *
 * @tparam BType: Broadcast type; BroadcastType::NONE selects the plain unary path
 * @tparam acc_to_dest: Unused on Quasar; kept for API parity with Blackhole / other arches
 * @tparam binary_reuse_dest: Dest reuse mode (unary path only)
 * @tparam unpack_to_dest: when true, the (non-broadcast) primitive routes the operand
 * through UNP_DEST regardless of format
 * @param operand: The logical dataflow buffer id
 * @param start_tile_index: The starting tile index within the input buffer
 * @param ntiles: The number of consecutive tiles to unpack
 */
// TODO: AM; Optimize block calls by using ntiles per unpack, issue #40798
template <
    BroadcastType BType = BroadcastType::NONE,
    [[maybe_unused]] bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_block(
    const std::uint32_t operand, const std::uint32_t start_tile_index, const std::uint32_t ntiles) {
    LLK_TDMA_GUARD_NOTE_TDMA(operand);  // TEN-4746: real unpack (UNPACR) disarms this dfb
    const std::uint32_t operand_id = get_operand_id(operand);
    const LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(operand_id);
    const std::uint32_t rd_entry_idx = local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].rd_entry_idx;
    [[maybe_unused]] const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
    for (std::uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        WAYPOINT("UPAW");
        if constexpr (BType == BroadcastType::NONE) {
            if constexpr (unpack_to_dest) {
                // EN_32BIT_DEST pinned to true to match the pack side, see llk_unpack_A.
                _llk_unpack_unary_operand_to_dest_<DST_SYNC_MODE, true /*EN_32BIT_DEST*/>(rd_entry_idx + tile_index);
            } else {
                _llk_unpack_unary_operand_<p_unpacr::UNP_A, binary_reuse_dest>(rd_entry_idx + tile_index, tensor_shape);
            }
        } else {
            constexpr std::uint32_t unp_sel = unpack_to_dest ? p_unpacr::UNP_A : p_unpacr::UNP_B;
            _llk_unpack_unary_broadcast_operands_<unp_sel, unpack_to_dest>(rd_entry_idx + tile_index);
        }
        WAYPOINT("UPAD");
    }
}

/**
 * @brief Toggle the unpacker engine to order a POP after its WAIT: STALLWAIT on SrcA-clear then a
 *        clear-SrcA UNPACR_NOP (no CB read, no DEST write, no pop). Reads nothing from L1.
 *
 * Call between llk_wait_tiles and llk_pop_tiles on a buffer that is being drained (by llk_pop_tiles) but
 * whose data is not needed. Unlike the WH/BH version -- a debug-only SrcA flush with no ordering role --
 * this is a required Quasar primitive: the UNPACR_NOP is a real unpacker TDMA that orders the POP_TILES
 * after its WAIT_TILES on dfb_id (TEN-4746 / #48552). Because it reads nothing, PACKER_L1_ACC is
 * undisturbed. The STALLWAIT ensures SrcA is free before the clear, so it cannot clobber a SrcA bank
 * still owned by an in-flight op; SrcA is cleared only (the next op re-unpacks it).
 *
 * @param dfb_id  The dataflow buffer whose WAIT/POP this orders. Disarms the TEN-4746 tile-counter guard
 *                that llk_wait_tiles armed for it (llk_pop_tiles asserts the buffer was disarmed).
 */
inline void llk_unpack_dummy(const std::uint32_t dfb_id) {
    TTI_STALLWAIT(p_stall::STALL_UNPACK, 0, 0, p_stall::SRCA_CLR);
    TTI_UNPACR_NOP(p_unpacr::UNP_A, 0, 0, 0, p_unpacr::UNP_CLRSRC_ZERO, p_unpacr::UNP_CLRSRC);
    LLK_TDMA_GUARD_NOTE_TDMA(dfb_id);  // TEN-4746: UNPACR_NOP orders POP after WAIT -> disarm this dfb
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_A_uninit() {}
