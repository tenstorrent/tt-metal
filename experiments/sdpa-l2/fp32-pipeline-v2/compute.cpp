// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Scheduling probe only. Lower DST half: HiFi4 QK. Upper: two FP32 scores
// and one resident zero maximum. No SDPA operator or normalization claim.
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/exp.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa.h"
#endif

// UNPACK owns the score half only while UNPACK_TO_DEST is zero. PACK releases
// it after SFPU and score pack complete. No MATH mailbox or participation.
void reload_scores() {
    UNPACK((t6_semaphore_wait_on_max < p_stall::STALL_UNPACK | p_stall::STALL_SYNC > (semaphore::UNPACK_TO_DEST)));
    UNPACK((tensix_sync()));
    UNPACK((llk_unpack_reconfig_data_format<true, p_dim_stride_target::IGNORE, true>(2, 2)));
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(false, false, 2)));
    UNPACK((ckernel_template(8, 1, TT_OP_UNPACR(0, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1))
                .program()));
    UNPACK({
        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
        wait_for_next_context(2);
        volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
        const uint32_t base =
            unp_cfg_context == 0 ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;
        cfg[base] = get_local_cb_interface(2).fifo_rd_ptr - 1;
        semaphore_post(semaphore::UNPACK_SYNC);
        TTI_SETC16(SRCA_SET_Base_ADDR32, 0);
        const uint32_t dst_byte_addr = 16 * (4 + 4 * 64);
        if (unp_cfg_context == 0) {
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(1);
            cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(dst_byte_addr);
        } else {
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(1);
            cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(dst_byte_addr);
        }
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
        ckernel_template::run();
        t6_semaphore_get(semaphore::UNPACK_SYNC);
        unpack_to_dest_tile_done(unp_cfg_context, static_cast<uint32_t>(DataFormat::Float32));
        switch_config_context(unp_cfg_context);
    });
}

void process_scores() {
    PACK(
        (t6_semaphore_wait_on_zero < p_stall::STALL_SYNC | p_stall::STALL_SFPU |
         p_stall::STALL_MATH > (semaphore::UNPACK_TO_DEST)));
    // Blackhole ZEROACC face indices are local to the half; its half-select
    // comes from the math-offset register, unlike SFPU's full-DST addressing.
    PACK(TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 512));
    PACK(for (uint32_t tile = 0; tile < 2; ++tile) {
        for (uint32_t face = 0; face < 4; ++face) {
            TT_ZEROACC(p_zeroacc::CLR_16, 1, 1, ADDR_MOD_3, get_dest_index_in_faces(tile, face));
        }
    });
    PACK((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_fused_sub_exp, (0x3db504f3, false, true), 4, VectorMode::None)));
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    pack_tile<true>(4, 16, 4);
    pack_tile<true>(5, 16, 5);
    PACK((t6_semaphore_get<p_stall::PACK>(semaphore::UNPACK_TO_DEST)));
}

void kernel_main() {
    constexpr uint32_t repetitions = get_compile_time_arg_val(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    cb_wait_front(0, 4);
    cb_wait_front(1, 16);
    cb_wait_front(2, 2);
    cb_reserve_back(16, 6);
    PACK((llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, DST_ACCUM_MODE>()));
    PACK((
        SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_zero_sum, 6, VectorMode::None)));
    PACK((ckernel::sfpu::calculate_sdpa_fused_sub_exp<0x3db504f3, true>()));
    for (uint32_t rep = 0; rep < repetitions; ++rep) {
        reload_scores();
        process_scores();
        UNPACK((llk_unpack_reconfig_data_format<true, p_dim_stride_target::IGNORE, true>(1, 0)));
        mm_no_mop_init_short(0, 1, true, 4, 1, 4);
        tile_regs_acquire();
        MATH(TTI_ZEROACC(p_zeroacc::CLR_HALF, 1, 0, ADDR_MOD_1, 0));
        for (uint32_t k = 0; k < 4; ++k) {
            matmul_block_no_mop(0, 1, k, k * 4, 0, true, 4, 1, 4);
        }
        tile_regs_commit();
#ifdef PIPE_SERIAL
        tile_regs_wait();
#endif
        reload_scores();
        process_scores();
#ifndef PIPE_SERIAL
        tile_regs_wait();
#endif
        for (uint32_t j = 0; j < 4; ++j) {
            pack_tile<true>(j, 16, j);
        }
        // The upper half remains owned by the score pipeline. Never clear it
        // through generic SyncFull release, and never flip the lower-half base.
        PACK(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK));
        PACK(TTI_ZEROACC(p_zeroacc::CLR_HALF, 1, 0, ADDR_MOD_1, 0));
        PACK((_llk_packer_set_math_semaphore_<p_stall::NONE>()));
    }
    cb_push_back(16, 6);
    cb_pop_front(0, 4);
    cb_pop_front(1, 16);
    cb_pop_front(2, 2);
}
