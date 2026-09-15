// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
template <int pairs>
inline void hybrid_multiply() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        TTI_SFPLOAD(6, InstrModLoadStore::FP32, ADDR_MOD_6, 64 * pairs);
        TTI_SFPLOAD(0, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
        if constexpr (pairs == 2) {
            TTI_SFPLOAD(1, InstrModLoadStore::FP32, ADDR_MOD_6, 64);
        }
        TTI_SFPMUL(0, 6, p_sfpu::LCONST_0, 0, 0);
        if constexpr (pairs == 2) {
            TTI_SFPMUL(1, 6, p_sfpu::LCONST_0, 1, 0);
        }
        TTI_SFPNOP;
        if constexpr (pairs == 2) {
            TTI_SFPSTORE(0, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
            TTI_SFPSTORE(1, InstrModLoadStore::FP32, ADDR_MOD_7, 64);
        } else {
            TTI_SFPSTORE(0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
        }
    }
}
inline void hybrid_recip() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
    sfpi::vConstFloatPrgm0 = 2.0f;
    for (int i = 0; i < 32; ++i) {
        TTI_SFPLOAD(0, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
        TTI_SFPNOP;
        sfpi::vFloat x = sfpi::l_reg[sfpi::LRegs::LReg0];
        sfpi::l_reg[sfpi::LRegs::LReg0] = sfpu_reciprocal_iter<2>(x);
        TTI_SFPSTORE(0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
    }
}
}  // namespace ckernel::sfpu
#endif

// Runtime FP32 state section inside a kernel whose bulk math is BF16 DST.
static void hybrid_reconfig(uint32_t a, uint32_t b) {
    UNPACK((llk_unpack_reconfig_data_format<true, p_dim_stride_target::IGNORE, false>(a, b)));
    MATH((llk_math_reconfig_data_format<true>(a, b)));
    ComputeKernelSentinel::instance().set_srca(a).set_srcb(b);
}
static void hybrid_pack_config(uint32_t cb) {
    PACK((_llk_pack_reconfig_data_format_<true>(
        uint32_t(DataFormat::Float32), pack_dst_format[cb], get_local_cb_interface(cb).fifo_page_size, 32, 4)));
    PACK((llk_pack_init_impl<PackMode::Default>(uint32_t(DataFormat::Float32), 16, 32, 4, 1, false)));
    ComputeKernelSentinel::instance().set_pack(cb);
}
template <bool ooo = false>
static void hybrid_pack(uint32_t tile, uint32_t cb, uint32_t index = 0) {
    PACK(LLK_ASSERT(tile < 4, "Hybrid FP32 DST tile exceeds half capacity");)
    PACK(LLK_ASSERT_BLOCK(are_packers_configured_correctly(uint32_t(DataFormat::Float32), pack_dst_format[cb]));)
    PACK((llk_pack_impl<true, PackMode::Default>(tile, get_output_tile_address<ooo, PackMode::Default>(cb, index))));
}
template <BroadcastType B = BroadcastType::NONE>
static void hybrid_load(uint32_t cb, uint32_t index, uint32_t dst) {
    hybrid_reconfig(cb, cb);
    unary_bcast_init<B, true>(cb);
    unary_bcast<B, true>(cb, index, dst);
    unary_bcast_uninit<B>(cb);
}
template <int pairs>
static void hybrid_update(
    uint32_t old_cb, uint32_t new_cb, uint32_t corr_cb, uint32_t old_index, uint32_t new_index, uint32_t corr_index) {
    tile_regs_acquire();
    for (int p = 0; p < pairs; ++p) {
        hybrid_load(old_cb, old_index + p, p);
    }
    hybrid_load<BroadcastType::COL>(corr_cb, corr_index, pairs);
    tile_regs_commit<true>();
    tile_regs_wait();
    PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, true, hybrid_multiply, (pairs), 0, VectorMode::None)));
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    hybrid_pack_config(new_cb);
    PACK((llk_pack_reconfig_l1_acc(1)));
    for (int p = 0; p < pairs; ++p) {
        hybrid_pack<true>(p, new_cb, new_index + p);
    }
    tile_regs_release<true>();
}
template <uint32_t rows, uint32_t cols>
static void hybrid_correct(
    uint32_t old_out,
    uint32_t old_sum,
    uint32_t corr,
    uint32_t new_out,
    uint32_t new_sum,
    uint32_t old_row,
    uint32_t sum_row,
    uint32_t write_row) {
    enable_fp32_dest_acc();
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; c += 2) {
            hybrid_update<2>(old_out, new_out, corr, (old_row + r) * cols + c, (write_row + r) * cols + c, old_row + r);
        }
        // BF16 bulk softmax leaves column-wise partial sums, not only column 0.
        hybrid_update<1>(old_sum, new_sum, corr, sum_row + r, write_row + r, old_row + r);
    }
    disable_fp32_dest_acc();
    PACK((llk_math_sfpu_init_once()));
}
template <uint32_t cols>
static void hybrid_normalize(uint32_t sum, uint32_t out, uint32_t ones, uint32_t scratch, uint32_t dst, uint32_t rows) {
    enable_fp32_dest_acc();
    PACK((llk_pack_reconfig_l1_acc(0)));
    const uint32_t sum_alias = sum == 12 ? 17 : 18;
    for (uint32_t row = 0; row < rows; ++row) {
        cb_wait_front(sum, 1);
        cb_wait_front(ones, 1);
        cb_reserve_back(scratch, 1);
        // Final-only FPU reduction uses a BF16 view; recurrence itself uses full FP32.
        UNPACK(get_local_cb_interface(sum_alias).fifo_rd_ptr = get_local_cb_interface(sum).fifo_rd_ptr;)
        hybrid_reconfig(ones, sum_alias);
        UNPACK((llk_unpack_AB_matmul_init(sum_alias, ones, 0, 1, 1, 1)));
        MATH((llk_math_matmul_init<MathFidelity::HiFi4, MM_THROTTLE>(sum_alias, ones, 0, 1, 1)));
        tile_regs_acquire();
        UNPACK((llk_unpack_AB_matmul(sum_alias, ones, 0, 0, 1, 1, 1)));
        MATH((llk_math_matmul<MathFidelity::HiFi4, MM_THROTTLE>(0, 1, 1)));
        tile_regs_commit<true>();
        tile_regs_wait();
        PACK((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, true, hybrid_recip, 0, VectorMode::None)));
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        hybrid_pack_config(scratch);
        hybrid_pack(0, scratch);
        tile_regs_release<true>();
        cb_push_back(scratch, 1);
        cb_pop_front(sum, 1);
        cb_wait_front(scratch, 1);
        cb_wait_front(out, cols);
        cb_reserve_back(dst, cols);
        for (uint32_t c = 0; c < cols; ++c) {
            tile_regs_acquire();
            hybrid_load(out, c, 0);
            hybrid_load<BroadcastType::COL>(scratch, 0, 1);
            tile_regs_commit<true>();
            tile_regs_wait();
            PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, true, hybrid_multiply, (1), 0, VectorMode::None)));
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            hybrid_pack_config(dst);
            hybrid_pack(0, dst);
            tile_regs_release<true>();
        }
        cb_push_back(dst, cols);
        cb_pop_front(out, cols);
        cb_pop_front(scratch, 1);
    }
    disable_fp32_dest_acc();
    PACK((llk_math_sfpu_init_once()));
    pack_reconfig_data_format(scratch);
}
