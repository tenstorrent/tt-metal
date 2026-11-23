// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_unpack_matmul.h"
#include "llk_unpack_common_api.h"
#include "api/dataflow/dataflow_buffer.h"

/*************************************************************************
 * LLK UNPACK AB MATMUL
 *************************************************************************/

// What the tile-by-tile unpack of a bank-major operand needs at execute time.
struct MatmulUnpackState {
    std::uint8_t bfd_a = 0;        // operand A (input0 -> SrcB, UNPACR1)
    std::uint8_t bfd_b = 0;        // operand B (input1 -> SrcA, UNPACR0)
    bool transpose = false;        // SrcA is loaded by the four-face transpose replay
    bool full_tiles = true;        // both operands are four-face tiles: one UNPACR per tile, else the replay images
    std::uint32_t reuse_len = 0;   // tiny tiles: the reused operand's replay image is [0, reuse_len)
    std::uint32_t stream_len = 0;  // and the streamed operand's is [reuse_len, reuse_len + stream_len)
};
inline thread_local MatmulUnpackState matmul_unpack_state{};  // one per Neo, like the other TRISC state

// Column-strided in1 streaming is not implemented on Quasar; stride 1 is the default contiguous step.
inline void llk_unpack_AB_matmul_set_in1_column_stride(
    [[maybe_unused]] const std::uint32_t in1_cb_id, [[maybe_unused]] const std::uint32_t stride_tiles) {
    LLK_ASSERT(stride_tiles == 1, "Matmul in1 column stride is not supported on Quasar");
}

/**
* @brief Initialize unpacker for matrix multiply

* @tparam TRANSPOSE_EN: Enables transpose of a tile. Required; no default.
* @param operandA: The input0 operand circular buffer
* @param operandB: The input1 operand circular buffer
* @param ct_dim: number of tiles in the column dimension for input1 of matrix multiply
* @param rt_dim: number of tiles in the row dimension for input0 of matrix multiply
* @param kt_dim: number of tiles in the common dimension between input0 & input1 of matrix multiply
*
* This function initializes the unpacker to unpack operand 0 from the input0 operand circular buffer into SrcB
* and operand 1 from the input1 operand circular buffer into SrcA. Matrix multiply FPU operation does SrcB * SrcA.
*
* Each operand's table entry (fixed per-engine BFD id) is programmed here;
* the DFB ids are used only to fetch buffer info, never as BFD ids. Mind the role flip: operandA feeds
* UNPACR1 -> SrcB (Unp1) and operandB feeds UNPACR0 -> SrcA (Unp0).
* Operand tile shapes are read from circular-buffer metadata.
*/
template <bool TRANSPOSE_EN>
__attribute__((always_inline)) inline void llk_unpack_AB_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim) {
    // In0 -> srcB (UNPACR1)
    // In1 -> srcA (UNPACR0)
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    const ckernel::TensorShape src_b_shape = get_operand_tensor_shape(operandA_id);
    const ckernel::TensorShape src_a_shape = get_operand_tensor_shape(operandB_id);
    LLK_ASSERT(
        ckernel::validate_matmul_tensor_shapes_(src_b_shape, src_a_shape),
        "unsupported SrcB/input0 and SrcA/input1 TensorShape pair for matmul");

    // MxFp4 operands feeding matmul are ALWAYS unpacked as the 2x-packed src-register format
    // (MxFp4_2x_B) on Quasar. 2x dataformats are currently unsupported with transpose.
    LLK_ASSERT(
        !TRANSPOSE_EN || static_cast<DataFormat>(get_operand_src_format(operandB_id)) != DataFormat::MxFp4,
        "matmul SrcA transpose is not supported for 2x-format inputs");

    llk_unpack_program_bfd<ckernel::trisc::BFD_ID_UNP1>(operandA_id);
    constexpr std::uint8_t bfd_a = ckernel::trisc::BFD_ID_UNP1;
    llk_unpack_program_bfd<ckernel::trisc::BFD_ID_UNP0>(operandB_id);
    constexpr std::uint8_t bfd_b = ckernel::trisc::BFD_ID_UNP0;

    _llk_unpack_matmul_init_<TRANSPOSE_EN>(bfd_a, bfd_b, ct_dim, rt_dim, kt_dim, src_b_shape, src_a_shape);

    // A bank-major operand is unpacked tile by tile at execute time.
    matmul_unpack_state = {
        .bfd_a = bfd_a,
        .bfd_b = bfd_b,
        .transpose = TRANSPOSE_EN,
        .full_tiles = src_b_shape.total_num_faces() == NUM_FACES && src_a_shape.total_num_faces() == NUM_FACES};
    if (!matmul_unpack_state.full_tiles && (dfb_block_is_permuted(get_local_dfb_interface(operandA_id)) ||
                                            dfb_block_is_permuted(get_local_dfb_interface(operandB_id)))) {
        // Record the tiny-tile images again, exactly as _llk_unpack_matmul_variable_tile_mop_config_ did, to learn
        // their lengths, the reused operand's image first, the streamed operand's after it.
        const bool reuse_a = ct_dim >= rt_dim;
        matmul_unpack_state.reuse_len =
            reuse_a ? _llk_unpack_matmul_load_tile_replay_<p_unpacr::UNP_B, false>(0, bfd_a, src_b_shape, false)
                    : _llk_unpack_matmul_load_tile_replay_<p_unpacr::UNP_A, TRANSPOSE_EN>(0, bfd_b, src_a_shape, false);
        matmul_unpack_state.stream_len = reuse_a ? _llk_unpack_matmul_load_tile_replay_<p_unpacr::UNP_A, TRANSPOSE_EN>(
                                                       matmul_unpack_state.reuse_len, bfd_b, src_a_shape, true)
                                                 : _llk_unpack_matmul_load_tile_replay_<p_unpacr::UNP_B, false>(
                                                       matmul_unpack_state.reuse_len, bfd_a, src_b_shape, true);
    }

    // MxFp4 operands feeding matmul are ALWAYS unpacked as the 2x-packed src-register format
    // (MxFp4_2x_B) on Quasar. The generated unpack_dst_format[] table keeps the op-agnostic MX
    // default (Float16_b, needed by non-matmul MxFp4 ops), so override only the unpacker gasket
    // OUT_DATA_FORMAT here. Safe: this runs at init with the unpacker idle (before the first
    // UNPACR), OUT_DATA_FORMAT is a shadow register, and the buffer descriptor (keyed on the MxFp4
    // L1 format + tile geometry) is unchanged. EN_32BIT_DEST does not affect the MxFp4->MxFp4_2x_B reconfig validity,
    // so pass false.
    if (static_cast<DataFormat>(get_operand_src_format(operandB_id)) == DataFormat::MxFp4) {
        _llk_unpack_reconfig_data_format_src_<p_unpacr::UNP_A, false>(
            get_operand_src_format(operandB_id), static_cast<std::uint32_t>(DataFormat::MxFp4_2x_B));
    }
    if (static_cast<DataFormat>(get_operand_src_format(operandA_id)) == DataFormat::MxFp4) {
        _llk_unpack_reconfig_data_format_src_<p_unpacr::UNP_B, false>(
            get_operand_src_format(operandA_id), static_cast<std::uint32_t>(DataFormat::MxFp4_2x_B));
    }
}

/**
 * @brief Undo the MxFp4 -> MxFp4_2x_B unpacker OUT_DATA_FORMAT override from @ref llk_unpack_AB_matmul_init.
 *
 * @param operandA: The input0 operand circular buffer (matches the matmul init call)
 * @param operandB: The input1 operand circular buffer
 *
 * Restores each MxFp4 operand's unpacker OUT_DATA_FORMAT to the op-agnostic unpack_dst_format[]
 * value (Float16_b), so a following NON-matmul op on the same MxFp4 buffer unpacks correctly. This
 * explicit restore is required because the non-matmul unpack inits never reprogram OUT_DATA_FORMAT,
 * and llk_unpack_reconfig_data_format is silently skipped for a same-format operand
 * (should_reconfig_src_reg_df keys on the generated table, which never saw the 2x override).
 *
 * @note Call after all matmuls, before initializing the next op, whenever an MxFp4 matmul operand is
 * reused by a non-matmul op in the same kernel.
 * Same UNP mapping as init: In0(operandA)->SrcB->UNP_B, In1(operandB)->SrcA->UNP_A.
 */
inline void llk_unpack_AB_matmul_uninit(const std::uint32_t operandA, const std::uint32_t operandB) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    if (static_cast<DataFormat>(get_operand_src_format(operandB_id)) == DataFormat::MxFp4) {
        _llk_unpack_reconfig_data_format_src_<p_unpacr::UNP_A, false>(
            get_operand_src_format(operandB_id), unpack_dst_format[operandB_id]);
    }
    if (static_cast<DataFormat>(get_operand_src_format(operandA_id)) == DataFormat::MxFp4) {
        _llk_unpack_reconfig_data_format_src_<p_unpacr::UNP_B, false>(
            get_operand_src_format(operandA_id), unpack_dst_format[operandA_id]);
    }
}

/**
 * @brief Runtime-transpose overload, signature-compatible with the Wormhole/Blackhole llk_api.
 *
 * Quasar takes transpose as a template argument, but the shared Compute API passes it as a runtime
 * value. This overload absorbs that difference here rather than forcing an arch branch into the
 * Compute API.
 *
 * @param operandA: The input0 operand circular buffer
 * @param operandB: The input1 operand circular buffer
 * @param transpose: Transpose flag; enables srcA transpose
 * @param ct_dim: number of tiles in the column dimension for input1 of matrix multiply
 * @param rt_dim: number of tiles in the row dimension for input0 of matrix multiply
 * @param kt_dim: number of tiles in the common dimension between input0 & input1 of matrix multiply
 */
__attribute__((always_inline)) inline void llk_unpack_AB_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    if (transpose == 0) {
        llk_unpack_AB_matmul_init<false /*TRANSPOSE_EN*/>(operandA, operandB, ct_dim, rt_dim, kt_dim);
    } else {
        llk_unpack_AB_matmul_init<true /*TRANSPOSE_EN*/>(operandA, operandB, ct_dim, rt_dim, kt_dim);
    }
}

/**
 * @brief Unpacks a matmul block tile by tile, every L1 index set through the ring lookup.
 *
 * Loads the same SrcB / SrcA tile sequence as the matmul MOP: the reused operand first, then the streamed
 * tiles, one data-valid per tile, so the math thread's block MOP runs unchanged. Needed when an operand's
 * ring is bank-major: consecutive tiles of the block are then not at consecutive L1 indices, which the MOP's
 * index increments assume. Operand A row r is tile start_a + r * kt_dim and operand B column c is tile
 * start_b + c, both in page order.
 *
 * @param dfb_a: Operand A's (input0, SrcB) local DFB interface
 * @param dfb_b: Operand B's (input1, SrcA) local DFB interface
 * @param start_a: Page-order index of operand A's first tile (cursor + tile index)
 * @param start_b: Page-order index of operand B's first tile
 * @param ct_dim: number of tiles in the column dimension for input1 of matrix multiply
 * @param rt_dim: number of tiles in the row dimension for input0 of matrix multiply
 * @param kt_dim: number of tiles in the common dimension between input0 & input1 of matrix multiply
 * @param src_b_shape: Operand A's tile shape
 * @param src_a_shape: Operand B's tile shape
 * @note Uses the state @ref llk_unpack_AB_matmul_init recorded; called by @ref llk_unpack_AB_matmul.
 */
inline void llk_unpack_AB_matmul_tilewise(
    const LocalDFBInterface& dfb_a,
    const LocalDFBInterface& dfb_b,
    const std::uint32_t start_a,
    const std::uint32_t start_b,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim,
    const ckernel::TensorShape src_b_shape,
    const ckernel::TensorShape src_a_shape) {
    const MatmulUnpackState& st = matmul_unpack_state;
    const bool reuse_a = ct_dim >= rt_dim;
    const std::uint32_t scale_b = _llk_unpack_matmul_src_tile_scale_(src_b_shape);  // operand A -> SrcB
    const std::uint32_t scale_a = _llk_unpack_matmul_src_tile_scale_(src_a_shape);  // operand B -> SrcA
    // Tiny tiles: each operand's replay image, by which operand the MOP reuses.
    const std::uint32_t a_start = reuse_a ? 0 : st.reuse_len;
    const std::uint32_t a_len = reuse_a ? st.reuse_len : st.stream_len;
    const std::uint32_t b_start = reuse_a ? st.reuse_len : 0;
    const std::uint32_t b_len = reuse_a ? st.stream_len : st.reuse_len;

    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);
    auto load_a = [&](const std::uint32_t tile) {
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, dfb_ring_entry(dfb_a, tile) * scale_b);
        if (st.full_tiles) {
            TT_UNPACR1_TILE_INC(0 /*Dst_Tile_Idx_Inc*/, 0 /*Src_Tile_Idx_Inc*/, st.bfd_a, 1 /*SetDvalid*/);
        } else {
            TT_REPLAY(a_start, a_len, 0 /*last*/, 0 /*set_mutex*/, 0 /*execute_while_loading*/, 0 /*load_mode*/);
        }
    };
    auto load_b = [&](const std::uint32_t tile) {
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, dfb_ring_entry(dfb_b, tile) * scale_a);
        if (!st.full_tiles) {
            TT_REPLAY(b_start, b_len, 0 /*last*/, 0 /*set_mutex*/, 0 /*execute_while_loading*/, 0 /*load_mode*/);
        } else if (st.transpose) {
            // The four transposed faces recorded by init.
            TTI_REPLAY(0, NUM_FACES, 0 /*last*/, 0 /*set_mutex*/, 0 /*execute_while_loading*/, 0 /*load_mode*/);
        } else {
            TT_UNPACR0_TILE_INC(0 /*Dst_Tile_Idx_Inc*/, 0 /*Src_Tile_Idx_Inc*/, st.bfd_b, 1 /*SetDvalid*/);
        }
    };
    if (reuse_a) {
        for (std::uint32_t r = 0; r < rt_dim; r++) {
            load_a(start_a + r * kt_dim);
            for (std::uint32_t c = 0; c < ct_dim; c++) {
                load_b(start_b + c);
            }
        }
    } else {
        for (std::uint32_t c = 0; c < ct_dim; c++) {
            load_b(start_b + c);
            for (std::uint32_t r = 0; r < rt_dim; r++) {
                load_a(start_a + r * kt_dim);
            }
        }
    }
}

/**
 *
 * @brief Performs unpack operation for matrix multiply such that:
 *
 * @param operandA: The input0 operand circular buffer
 * @param operandB: The input1 operand circular buffer
 * @param tile_index_a: The index into the input0 CB (UNPACKER1 -> SRCB)
 * @param tile_index_b: The index into the input1 CB (UNPACKER0 -> SRCA)
 * @param ct_dim: number of tiles in the column dimension for input1 of matrix multiply
 * @param rt_dim: number of tiles in the row dimension for input0 of matrix multiply
 * @param kt_dim: number of tiles in the common dimension between input0 & input1 of matrix multiply
 *
 * This function unpacks input0 and input1 operands from the input circular buffers to the src registers such that:
 * Input 0 -> unpack to SrcB
 * Input 1 -> unpack to SrcA
 * The matrix multiply has the following dimensions:
 * Output [rt_dim, ct_dim] = Input0 [rt_dim, kt_dim] x Input1 [kt_dim, ct_dim]
 * This unpacker only sets up Input0 [rt_dim, 1] x Input1 [1, ct_dim]
 * kt_dim is assumed to be iterated over outside this api call
 * Operand tile shapes are read from circular-buffer metadata. An operand on a bank-major ring is unpacked tile
 * by tile through the ring lookup instead of the MOP.
 */
inline void llk_unpack_AB_matmul(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    // In0/InA -> srcB
    // In1/InB -> srcA
    LLK_TDMA_GUARD_NOTE_TDMA(operandA);  // TEN-4746: real unpack (UNPACR) disarms these dfbs
    LLK_TDMA_GUARD_NOTE_TDMA(operandB);
    LLK_REINIT_GUARD_ASSERT_MATCHES(
        ckernel::trisc::BFD_ID_UNP1,
        operandA,
        "unpack_AB_matmul operandA DFB differs from the one llk_unpack_AB_matmul_init programmed");
    LLK_REINIT_GUARD_ASSERT_MATCHES(
        ckernel::trisc::BFD_ID_UNP0,
        operandB,
        "unpack_AB_matmul operandB DFB differs from the one llk_unpack_AB_matmul_init programmed");

    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);
    const ckernel::TensorShape src_b_shape = get_operand_tensor_shape(operandA_id);
    const ckernel::TensorShape src_a_shape = get_operand_tensor_shape(operandB_id);

    const LocalDFBInterface& local_dfb_interface_a = get_local_dfb_interface(operandA_id);
    const LocalDFBInterface& local_dfb_interface_b = get_local_dfb_interface(operandB_id);
    const std::uint32_t start_a =
        local_dfb_interface_a.tc_slots[local_dfb_interface_a.tc_idx].cursor_entry_idx + tile_index_a;
    const std::uint32_t start_b =
        local_dfb_interface_b.tc_slots[local_dfb_interface_b.tc_idx].cursor_entry_idx + tile_index_b;

    WAYPOINT("UPMW");
    if (dfb_block_is_permuted(local_dfb_interface_a) || dfb_block_is_permuted(local_dfb_interface_b)) {
        llk_unpack_AB_matmul_tilewise(
            local_dfb_interface_a,
            local_dfb_interface_b,
            start_a,
            start_b,
            ct_dim,
            rt_dim,
            kt_dim,
            src_b_shape,
            src_a_shape);
    } else {
        _llk_unpack_matmul_(
            ct_dim,
            rt_dim,
            kt_dim,
            dfb_ring_entry(local_dfb_interface_a, start_a),
            dfb_ring_entry(local_dfb_interface_b, start_b),
            src_b_shape,
            src_a_shape);
    }
    WAYPOINT("UPMD");
}
