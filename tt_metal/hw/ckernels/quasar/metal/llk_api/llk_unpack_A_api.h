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
 * @brief TDMA engine the unpack_A paths program, the unpacker the operand is routed to.
 *
 * UNP_A / UNP_DEST map to Unp0, UNP_B maps to Unp1. The plain unary path always drives UNP_A.
 * Broadcast drives UNP_B unless the operand is routed to DEST. Dest-reuse takes
 * its L1 operand into whichever source register DEST does not occupy, so DEST_TO_SRCA drives UNP_B and
 * DEST_TO_SRCB drives UNP_A.
 *
 * @tparam BType: Broadcast type
 * @tparam binary_reuse_dest: Dest reuse mode
 * @tparam unpack_to_dest: When true, the operand is routed through UNP_DEST
 * @return The TDMA engine the unpack_A paths program
 */
template <BroadcastType BType, EltwiseBinaryReuseDestType binary_reuse_dest, bool unpack_to_dest>
inline constexpr ckernel::trisc::BfdResource unpack_a_bfd_resource() {
    if constexpr (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) {
        return binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA ? ckernel::trisc::BfdResource::Unp1
                                                                             : ckernel::trisc::BfdResource::Unp0;
    } else if constexpr (BType == BroadcastType::NONE) {
        return ckernel::trisc::BfdResource::Unp0;
    } else {
        return unpack_to_dest ? ckernel::trisc::BfdResource::Unp0 : ckernel::trisc::BfdResource::Unp1;
    }
}

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
 * unpack-to-dest primitive (its section handshake runs once per tile_regs_acquire / tile_regs_commit in
 * llk_unpack_wait_for_dest_available / llk_unpack_dest_section_done, not per tile); otherwise
 * `_llk_unpack_unary_operand_init_` targets UNP_A.
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
    bool acc_to_dest /*maybe_unused*/ = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_init(
    const std::uint32_t transpose_of_faces,
    const std::uint32_t within_face_16x16_transpose,
    const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    // The unpack_to_dest primitive takes no shape: it supports full 32x32 tiles only, asserted below.
    [[maybe_unused]] const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);

    constexpr ckernel::trisc::BfdResource engine = unpack_a_bfd_resource<BType, binary_reuse_dest, unpack_to_dest>();
    const std::uint8_t bfd_id = llk_unpack_program_bfd<engine>(operand_id);

    if constexpr (binary_reuse_dest != EltwiseBinaryReuseDestType::NONE) {
        static_assert(unpack_to_dest == false, "unpack_to_dest is not yet supported on Quasar");
        static_assert(acc_to_dest == false, "acc_to_dest is not yet supported on Quasar");
        static_assert(BType == BroadcastType::NONE, "On Quasar, only BroadcastType::NONE is supported for dest reuse");

        // For Quasar, the unp_sel field is ignored if binary_reuse_dest != EltwiseBinaryReuseDestType::NONE
        // CB_UNP in the reuse-dest MOP is UNP_B for DEST_TO_SRCA, UNP_A otherwise — unpack_a_bfd_resource
        // picks Unp1/Unp0 accordingly.
        _llk_unpack_unary_operand_init_<
            p_unpacr::UNP_A,
            false /* TRANSPOSE_EN */,
            false /* IS_32b_DEST_EN */,
            binary_reuse_dest>(bfd_id, tensor_shape, 1);
    } else {
        if constexpr (BType == BroadcastType::NONE) {
            LLK_ASSERT(
                transpose_of_faces == within_face_16x16_transpose,
                "Quasar unpack unary supports only full transpose (transpose_of_faces and within_face_16x16_transpose "
                "must match)");
            // Route to UNP_DEST purely on the op-writer flag (no format inspection). A 16-bit
            // operand is unpacked to DEST here too when the op writer requested it.
            if constexpr (unpack_to_dest) {
                // The to-dest placers add the SyncHalf bank offset to the UNP_DEST tile index with a fixed 32x32-tile
                // stride (_llk_unpack_dest_bank_tile_offset_), while the descriptor of a tiny-tile operand steps in
                // one-face tiles, so bank 1 would land at the wrong rows. Full tiles only.
                LLK_ASSERT(
                    tensor_shape.face_r_dim == MAX_FACE_R_DIM && tensor_shape.num_faces_r_dim == MAX_NUM_FACES_R_DIM &&
                        tensor_shape.num_faces_c_dim == MAX_NUM_FACES_C_DIM,
                    "unpack-to-dest supports only 32x32 tiles (face_r_dim=16, 2x2 faces)");
                // Transpose is not supported on UNP_DEST and is forced off inside the primitive.
                _llk_unpack_unary_operand_to_dest_init_(bfd_id);
            } else {
                if (transpose_of_faces && within_face_16x16_transpose) {
                    _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, true, DST_ACCUM_MODE, binary_reuse_dest>(
                        bfd_id, tensor_shape, 1);
                } else {
                    _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false, DST_ACCUM_MODE, binary_reuse_dest>(
                        bfd_id, tensor_shape, 1);
                }
            }
        } else {
            static_assert(!unpack_to_dest, "unpack_to_dest is not supported for unary broadcast");
            // Unlike the unary path above, the broadcast LLK takes no TensorShape, so it does not scale
            // its L1 tile index by the face count. Full-tile only until it is converted (tt-metal #47597).
            LLK_ASSERT(
                tensor_shape.face_r_dim == MAX_FACE_R_DIM && tensor_shape.num_faces_r_dim == MAX_NUM_FACES_R_DIM &&
                    tensor_shape.num_faces_c_dim == MAX_NUM_FACES_C_DIM,
                "Unary broadcast currently only supports 32x32 tiles (face_r_dim=16, 2x2 faces)");
            constexpr std::uint32_t unp_sel = unpack_to_dest ? p_unpacr::UNP_A : p_unpacr::UNP_B;
            _llk_unpack_unary_broadcast_operands_init_<unp_sel, BType, DST_ACCUM_MODE, unpack_to_dest>(bfd_id, 1);
        }
    }
}

/**
 *
 * @brief Unpacks a single operand for unary and unary-broadcast paths.
 *
 * For the non-broadcast path the UNP_DEST routing decision is made solely from the `unpack_to_dest` template
 * parameter (no format inspection): when true this calls `_llk_unpack_unary_operand_to_dest_tile_`, a sync-free
 * placer; the UNPACK_MATH / MATH_PACK section handshake and the SyncHalf bank flip live in
 * `llk_unpack_wait_for_dest_available` / `llk_unpack_dest_section_done` (tile_regs_acquire / tile_regs_commit), so
 * every call between the two lands in the same DEST section. Otherwise `_llk_unpack_unary_operand_` on UNP_A.
 *
 * @tparam BType: Broadcast type; BroadcastType::NONE selects the plain unary path
 * @tparam acc_to_dest: Unused on Quasar; kept for API parity with Blackhole / other arches
 * @tparam binary_reuse_dest: Dest reuse mode (unary path only)
 * @tparam unpack_to_dest: when true, the (non-broadcast) primitive routes the operand through UNP_DEST regardless of
 * format
 * @param operand: The logical dataflow buffer id
 * @param tile_index: The index in the input CB to read from
 * @param dst_tile_index: DEST tile index the tile lands at, relative to the current section base; must be below the
 * section capacity for the sync mode and DEST width (get_dest_max_tiles, asserted). Used on the non-broadcast
 * unpack-to-dest path only: there the unpacker writes DEST directly, so math cannot place the tile. Ignored otherwise,
 * including the broadcast paths, which unpack tile by tile.
 */
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest /*maybe_unused*/ = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A(
    const std::uint32_t operand, const std::uint32_t tile_index, const std::uint32_t dst_tile_index = 0) {
    LLK_TDMA_GUARD_NOTE_TDMA(operand);  // TEN-4746: real unpack (UNPACR) disarms this dfb
    LLK_REINIT_GUARD_ASSERT_MATCHES(
        (unpack_a_bfd_resource<BType, binary_reuse_dest, unpack_to_dest>()),
        operand,
        "unpack_A operand DFB differs from the one llk_unpack_A_init programmed");

    WAYPOINT("UPAW");
    const std::uint32_t operand_id = get_operand_id(operand);
    const LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(operand_id);
    // Consecutive tiles of a batch are stride_size_tiles entries apart, as on the pack side (#56194).
    const std::uint32_t l1_tile_idx = local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].cursor_entry_idx +
                                      tile_index * local_dfb_interface.stride_size_tiles;

    if constexpr (BType == BroadcastType::NONE) {
        if constexpr (unpack_to_dest) {
            // The unpacker places the tile itself, so the DEST index is checked here against the section capacity for
            // this sync mode and DEST width (4 tiles for SyncHalf with a 32-bit DEST): an out-of-range index would land
            // in the other bank while pack may still be draining it.
            constexpr std::uint32_t dest_section_tiles = ckernel::trisc::
                get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::trisc::DstTileShape::Tile32x32>();
            LLK_ASSERT(
                dst_tile_index < dest_section_tiles,
                "unpack-to-dest: dst_tile_index exceeds the DEST section capacity for this sync mode and DEST width");
            // Sync-free: section handshake is in llk_unpack_wait_for_dest_available / llk_unpack_dest_section_done.
            _llk_unpack_unary_operand_to_dest_tile_(l1_tile_idx, dst_tile_index);
        } else {
            const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
            _llk_unpack_unary_operand_<p_unpacr::UNP_A, binary_reuse_dest>(l1_tile_idx, tensor_shape);
        }
    } else {
        static_assert(!unpack_to_dest, "unpack_to_dest is not supported for unary broadcast");
        constexpr std::uint32_t unp_sel = unpack_to_dest ? p_unpacr::UNP_A : p_unpacr::UNP_B;
        _llk_unpack_unary_broadcast_operands_<unp_sel, unpack_to_dest>(l1_tile_idx);
    }
    WAYPOINT("UPAD");
}

/**
 * @brief Unpacks a block of tiles for unary and unary-broadcast paths.
 *
 * On the non-broadcast unpack-to-dest path the tiles land at DEST tiles [start_dst_tile_index, start_dst_tile_index +
 * ntiles) of the current section with no handshake of their own, see llk_unpack_A. The other paths, broadcast included,
 * unpack tile by tile.
 *
 * @tparam BType: Broadcast type; BroadcastType::NONE selects the plain unary path
 * @tparam acc_to_dest: Unused on Quasar; kept for API parity with Blackhole / other arches
 * @tparam binary_reuse_dest: Dest reuse mode (unary path only)
 * @tparam unpack_to_dest: when true, the (non-broadcast) primitive routes the operand
 * through UNP_DEST regardless of format
 * @param operand: The logical dataflow buffer id
 * @param start_tile_index: The starting tile index within the input buffer
 * @param ntiles: The number of consecutive tiles to unpack. On the non-broadcast unpack-to-dest path the block stays
 * inside the current DEST section, so start_dst_tile_index + ntiles must not exceed the section capacity for the sync
 * mode and DEST width (get_dest_max_tiles, asserted).
 * @param start_dst_tile_index: DEST tile index of the first tile, relative to the current section base; consecutive
 * tiles land at consecutive indices. Non-broadcast unpack-to-dest path only, ignored otherwise (the broadcast paths
 * unpack tile by tile).
 */
// TODO: AM; Optimize block calls by using ntiles per unpack, issue #40798
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest /*maybe_unused*/ = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_block(
    const std::uint32_t operand,
    const std::uint32_t start_tile_index,
    const std::uint32_t ntiles,
    const std::uint32_t start_dst_tile_index = 0) {
    LLK_TDMA_GUARD_NOTE_TDMA(operand);  // TEN-4746: real unpack (UNPACR) disarms this dfb
    LLK_REINIT_GUARD_ASSERT_MATCHES(
        (unpack_a_bfd_resource<BType, binary_reuse_dest, unpack_to_dest>()),
        operand,
        "unpack_A_block operand DFB differs from the one llk_unpack_A_init programmed");

    const std::uint32_t operand_id = get_operand_id(operand);
    const LocalDFBInterface& local_dfb_interface = get_local_dfb_interface(operand_id);
    const std::uint32_t rd_entry_idx = local_dfb_interface.tc_slots[local_dfb_interface.tc_idx].cursor_entry_idx;
    const std::uint32_t stride_size_tiles = local_dfb_interface.stride_size_tiles;
    if constexpr (BType == BroadcastType::NONE && unpack_to_dest) {
        WAYPOINT("UPAW");
        // The block stays inside the current DEST section, so its last tile is checked against the section capacity,
        // see llk_unpack_A.
        constexpr std::uint32_t dest_section_tiles = ckernel::trisc::
            get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::trisc::DstTileShape::Tile32x32>();
        LLK_ASSERT(
            start_dst_tile_index < dest_section_tiles && ntiles <= dest_section_tiles - start_dst_tile_index,
            "unpack-to-dest: start_dst_tile_index + ntiles exceeds the DEST section capacity for this sync mode and "
            "DEST width");
        // Consecutive tiles of a batch are stride_size_tiles entries apart, as on the pack side (#56194), so the block
        // primitive is given the stride and places each tile at its own L1 index.
        _llk_unpack_unary_operand_to_dest_block_(
            rd_entry_idx + start_tile_index * stride_size_tiles, start_dst_tile_index, ntiles, stride_size_tiles);
        WAYPOINT("UPAD");
    } else {
        [[maybe_unused]] const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
        for (std::uint32_t tile_index = start_tile_index; tile_index < start_tile_index + ntiles; tile_index++) {
            WAYPOINT("UPAW");
            // Consecutive tiles of a batch are stride_size_tiles entries apart, as on the pack side (#56194).
            const std::uint32_t l1_tile_idx = rd_entry_idx + tile_index * stride_size_tiles;
            if constexpr (BType == BroadcastType::NONE) {
                _llk_unpack_unary_operand_<p_unpacr::UNP_A, binary_reuse_dest>(l1_tile_idx, tensor_shape);
            } else {
                constexpr std::uint32_t unp_sel = unpack_to_dest ? p_unpacr::UNP_A : p_unpacr::UNP_B;
                _llk_unpack_unary_broadcast_operands_<unp_sel, unpack_to_dest>(l1_tile_idx);
            }
            WAYPOINT("UPAD");
        }
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
 * undisturbed. The NOP is a pure delay and leaves SrcA alone; nothing here clears it.
 *
 * @param dfb_id  The dataflow buffer whose WAIT/POP this orders. Disarms the TEN-4746 tile-counter guard
 *                that llk_wait_tiles armed for it (llk_pop_tiles asserts the buffer was disarmed).
 */
inline void llk_unpack_dummy(const std::uint32_t dfb_id) {
    TTI_STALLWAIT(p_stall::STALL_UNPACK, 0, 0, p_stall::SRCA_CLR);
    TTI_UNPACR_NOP(p_unpacr::UNP_A, 0, 0, 0, p_unpacr::UNP_CLRSRC_ZERO, p_unpacr::UNP_NOP);
    LLK_TDMA_GUARD_NOTE_TDMA(dfb_id);  // TEN-4746: UNPACR_NOP orders POP after WAIT -> disarm this dfb
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_A_uninit() {}
