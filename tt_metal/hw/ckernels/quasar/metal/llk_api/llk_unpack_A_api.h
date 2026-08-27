// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_bfd_alloc.h"
#include "llk_unpack_common_api.h"
#include "llk_unpack_unary_broadcast_operands.h"
#include "llk_unpack_unary_operand.h"
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
 * all operands (including 16-bit) are routed to DEST.
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
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
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
            // Route to UNP_DEST purely on the op-writer flag (no format inspection). A 16-bit
            // operand is unpacked to DEST here too when the op writer requested it.
            if constexpr (unpack_to_dest) {
                llk_unpack_program_bfd<ckernel::trisc::BfdResource::Unp0>(operand_id);
                _llk_unpack_unary_operand_init_<
                    p_unpacr::UNP_DEST,
                    false /*transpose*/,
                    DST_ACCUM_MODE,
                    binary_reuse_dest,
                    true>(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), tensor_shape, 1);
                return;
            }
            llk_unpack_program_bfd<ckernel::trisc::BfdResource::Unp0>(operand_id);
            if (transpose_of_faces && within_face_16x16_transpose) {
                _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, true, DST_ACCUM_MODE, binary_reuse_dest, false>(
                    ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), tensor_shape, 1);
            } else {
                _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false, DST_ACCUM_MODE, binary_reuse_dest, false>(
                    ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), tensor_shape, 1);
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
 * For the non-broadcast path, destination synchronization is scoped by tile_regs_acquire/tile_regs_commit; the
 * UNP_DEST routing decision is made solely from the `unpack_to_dest` template parameter (no format inspection).
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
        const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
        if constexpr (unpack_to_dest) {
            _llk_unpack_unary_operand_<p_unpacr::UNP_DEST, binary_reuse_dest, true, DST_SYNC_MODE>(
                l1_tile_idx, tensor_shape);
        } else {
            _llk_unpack_unary_operand_<p_unpacr::UNP_A, binary_reuse_dest, false, DST_SYNC_MODE>(
                l1_tile_idx, tensor_shape);
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
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
    for (std::uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
        WAYPOINT("UPAW");
        if constexpr (BType == BroadcastType::NONE) {
            if constexpr (unpack_to_dest) {
                _llk_unpack_unary_operand_<p_unpacr::UNP_DEST, binary_reuse_dest, true, DST_SYNC_MODE>(
                    rd_entry_idx + tile_index, tensor_shape);
            } else {
                _llk_unpack_unary_operand_<p_unpacr::UNP_A, binary_reuse_dest, false, DST_SYNC_MODE>(
                    rd_entry_idx + tile_index, tensor_shape);
            }
        } else {
            constexpr std::uint32_t unp_sel = unpack_to_dest ? p_unpacr::UNP_A : p_unpacr::UNP_B;
            _llk_unpack_unary_broadcast_operands_<unp_sel, unpack_to_dest>(rd_entry_idx + tile_index);
        }
        WAYPOINT("UPAD");
    }
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_A_uninit() {}
