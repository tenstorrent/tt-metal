// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Four-score variant: subtract the maximum in L1 before entering fixed-DST
// ownership. Keep QK in lower DST and exp in upper DST. PACK owns both packs;
// no shared-packer mutex or MATH-thread packing is used here.
template <uint32_t scale_fp32, uint32_t stride>
static void sdpa_l1_pipeline_qk_row(
    uint32_t q_cb, uint32_t k_cb, uint32_t score_cb, uint32_t inout_cb,
    uint32_t max_cb, uint32_t q_index, uint32_t row, uint32_t cols, bool overlap_first_half) {
    CircularBuffer(max_cb).wait_front(row);
    sdpa_subtract_max_l1(inout_cb, max_cb, row - 1, stride);
    MATH((_llk_math_pack_sync_init_<DstSync::SyncFull, true>()));
    PACK((_llk_pack_dest_init_<DstSync::SyncFull, true>()));
    UNPACK(get_local_cb_interface(score_cb).fifo_rd_ptr = get_local_cb_interface(inout_cb).fifo_rd_ptr;);
    MATH((llk_math_matmul_init_no_mop<MATH_FIDELITY, MM_THROTTLE>(q_cb, k_cb, true, 4, 1)));
    configure_pack_width(inout_cb, 4);
    for (uint32_t col = 0; col < cols; col += 4) {
        const uint32_t previous = (row - 1) * stride + col;
        sdpa_pipeline_reload(score_cb, previous, 4);
        PACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC | p_stall::STALL_SFPU | p_stall::STALL_MATH>(semaphore::UNPACK_TO_DEST)));
        PACK(TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 512));
        PACK(for (uint32_t tile = 0; tile < 4; ++tile) {
            for (uint32_t face = 0; face < 4; ++face) {
                TT_ZEROACC(p_zeroacc::CLR_16, 1, 1, ADDR_MOD_3, get_dest_index_in_faces(tile, face));
            }
        });
        PACK((ckernel::sfpu::init_sdpa_exp_grid<scale_fp32>()));
        PACK((SFPU_UNARY_CALL(DstSync::SyncFull, true, calculate_sdpa_exp_grid_batch, (128), 4, VectorMode::None)));
        PACK((SFPU_UNARY_CALL(DstSync::SyncFull, true, calculate_sdpa_exp_stream_effective, (128), 4, VectorMode::None)));
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        PACK((llk_pack_relu_config(ReluConfig::zero())));
        PACK({
            const uint32_t address = get_output_tile_address<true, PackMode::Default>(get_output_id(inout_cb), previous);
            _llk_pack_<DstSync::SyncFull, true>(4, address);
        });
        PACK((t6_semaphore_get<p_stall::PACK>(semaphore::UNPACK_TO_DEST)));
        sdpa_stream_reconfig(k_cb, q_cb);
        UNPACK((llk_unpack_AB_matmul_init(q_cb, k_cb, true, 4, 1, 4)));
        tile_regs_acquire();
        MATH(TTI_ZEROACC(p_zeroacc::CLR_HALF, 1, 0, ADDR_MOD_1, 0));
        for (uint32_t k = 0; k < 4; ++k) {
            matmul_block_no_mop(q_cb, k_cb, q_index + k, col + k * stride, 0, true, 4, 1, 4);
        }
        MATH((_llk_math_dest_section_done_<DstSync::SyncFull, true>()));
        tile_regs_wait();
        PACK((llk_pack_relu_config(ReluConfig::none())));
        pack_tile<true>(0, inout_cb, row * stride + col);
        PACK(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK));
        PACK(TTI_ZEROACC(p_zeroacc::CLR_HALF, 1, 0, ADDR_MOD_1, 0));
        PACK((_llk_packer_set_math_semaphore_<p_stall::NONE>()));
        if (overlap_first_half && col + 4 == cols / 2) {
            PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::UNPACK_MATH_DONE)));
        }
    }
    MATH((_llk_math_pack_sync_init_<DST_SYNC_MODE, true>()));
    PACK((_llk_pack_dest_init_<DST_SYNC_MODE, true>()));
}
