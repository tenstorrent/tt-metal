// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Scheduling probe only. Lower DST half: HiFi4 QK. Upper: four FP32 scores.
// MATH may issue QK packs under shared-packer ownership. No SDPA operator or normalization claim.
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/exp.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa.h"
#endif
#ifdef TRISC_MATH
#include "llk_pack.h"
#endif

// UNPACK owns the score half only while UNPACK_TO_DEST is zero. PACK releases
// it after SFPU and score pack complete. No MATH mailbox or participation.
void reload_scores() {
    UNPACK((t6_semaphore_wait_on_max < p_stall::STALL_UNPACK | p_stall::STALL_SYNC > (semaphore::UNPACK_TO_DEST)));
    UNPACK((tensix_sync()));
    UNPACK((llk_unpack_reconfig_data_format<true, p_dim_stride_target::IGNORE, true>(2, 2)));
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(false, false, 2)));
    UNPACK((ckernel_template(16, 1, TT_OP_UNPACR(0, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1))
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
    PACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC | p_stall::STALL_SFPU | p_stall::STALL_MATH>(semaphore::UNPACK_TO_DEST)));
    PACK(TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 512));
    PACK(for (uint32_t tile = 0; tile < 4; ++tile) {
        for (uint32_t face = 0; face < 4; ++face) {
            TT_ZEROACC(p_zeroacc::CLR_16, 1, 1, ADDR_MOD_3, get_dest_index_in_faces(tile, face));
        }
    });
    PACK((ckernel::sfpu::init_sdpa_exp_grid<0x3db504f3>()));
    PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_exp_grid_batch, (128), 4, VectorMode::None)));
    PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_exp_stream_effective, (128), 4, VectorMode::None)));
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK | p_stall::STALL_SYNC, p_stall::WAIT_SFPU));
#ifdef PIPE_MATH_PACK
    PACK((t6_mutex_acquire(mutex::THREAD2_ADC)));
#endif
    pack_tile<true>(4, 16, 4);
#ifdef PIPE_MATH_PACK
    PACK(TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::PACK));
    PACK((t6_mutex_release(mutex::THREAD2_ADC)));
#endif
    PACK((t6_semaphore_get<p_stall::PACK>(semaphore::UNPACK_TO_DEST)));
}

void kernel_main() {
    constexpr uint32_t repetitions = get_compile_time_arg_val(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    cb_wait_front(0, 4);
    cb_wait_front(1, 16);
    cb_wait_front(2, 4);
    cb_reserve_back(16, 8);
    PACK((llk_pack_init<PackMode::Default>(16, 4)));
    exp_packthread_tile_init<true, 0x3db504f3, InputClamping::None>();
#ifdef PIPE_MATH_PACK
    // MATH firmware does not initialize CB interfaces. Transfer the fixed
    // output address once; the packer configuration is shared after startup.
    PACK((tensix_sync()));
    PACK((mailbox_write(ThreadId::MathThreadId, get_local_cb_interface(16).fifo_wr_ptr - 1)));
    // Mailbox writes name the recipient; reads name the sender.
    MATH(const uint32_t output_address = mailbox_read(ThreadId::PackThreadId););
    MATH((_llk_pack_configure_addrmod_<PackMode::Default>()));
    MATH((_llk_pack_mop_config_<PackMode::Default, false>(16, 32, 4, 4)));
    MATH((packer_addr_counter_init()));
    MATH(TTI_SETADCXX(p_setadc::PAC, 15, 0));
#endif
    for (uint32_t rep = 0; rep < repetitions; ++rep) {
        reload_scores();
        process_scores();
        UNPACK((llk_unpack_reconfig_data_format<true, p_dim_stride_target::IGNORE, true>(1, 0)));
        mm_no_mop_init_short(0, 1, true, 4, 1, 4);
#ifndef PIPE_MATH_PACK
        tile_regs_acquire();
#endif
        MATH(TTI_ZEROACC(p_zeroacc::CLR_HALF, 1, 0, ADDR_MOD_1, 0));
        for (uint32_t k = 0; k < 4; ++k) {
            matmul_block_no_mop(0, 1, k, k * 4, 0, true, 4, 1, 4);
        }
#ifdef PIPE_MATH_PACK
        MATH(TTI_STALLWAIT(p_stall::STALL_PACK | p_stall::STALL_SYNC, p_stall::MATH));
        MATH((t6_mutex_acquire(mutex::THREAD2_ADC)));
        MATH((_llk_pack_<DstSync::SyncFull, true>(0, output_address)));
        MATH(TTI_STALLWAIT(p_stall::STALL_SYNC | p_stall::STALL_MATH, p_stall::PACK));
        MATH((t6_mutex_release(mutex::THREAD2_ADC)));
#else
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, 16, 0);
        PACK(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK));
        PACK(TTI_ZEROACC(p_zeroacc::CLR_HALF, 1, 0, ADDR_MOD_1, 0));
        PACK((_llk_packer_set_math_semaphore_<p_stall::NONE>()));
#endif
    }
#ifdef PIPE_MATH_PACK
    MATH((t6_semaphore_post<p_stall::PACK>(semaphore::PACK_DONE)));
    PACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC | p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    PACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
#endif
    cb_push_back(16, 8);
    cb_pop_front(0, 4);
    cb_pop_front(1, 16);
    cb_pop_front(2, 4);
}
