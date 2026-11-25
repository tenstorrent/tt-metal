// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Combine consume microbench, compute: the accumulator compute (routed_experts_combine accumulator_compute.cpp) with
// combine_snap, for a bf16 row (today) or an fp8 row with per-16 power-of-two scales:
//
//   bf16:  DEST = row x gate (FPU, srcB the gate broadcast), then the snap: round(DEST / q)
//   fp8:   DEST = codes x factor (FPU ELWMUL, srcB column-broadcast row by row: an FP16 factor a face row = 16 codes,
//          the scale x 2^10: exact, LoFi), then the snap with the gate folded in: round(DEST x gate x 2^-10 / q)
//          (as sparse_sdpa_fused_dequant.hpp's ELWMUL with SRCB_BCAST_COL, srcA / srcB / DEST stepping together)
//
// then the int32 count packed with L1 accumulation into the token's sum (cb_acc). After the rows, each token's sum
// back to fp32 x q, bf16 out (cb_out). The rows are flat: a 1024-element "tile" is 1024 consecutive elements.
//
// Compile-time args: fp8, n_rows, row_tiles, tokens
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#ifdef TRISC_UNPACK
// fp8: one flat tile of each operand, all 4 faces into source rows 0..63 (face f to slot f), data valid once (the math
// takes the whole tile in one pass); as sparse_sdpa_fused_dequant.hpp's unpack, without its tilize
inline void fp8_dequant_unpack_tile(uint32_t address_a, uint32_t address_b) {
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    // SETADCZW mask 0b0101 = {ch0_z, ch1_z}: ch0_z the L1 face, ch1_z the source register face slot
    TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0101);
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_B, 0, 1, 0, 1, 0b0101);
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_B, 0, 2, 0, 2, 0b0101);
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_B, 0, 3, 0, 3, 0b0101);
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 1 /*dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b0101);
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_A, 0, 1, 0, 1, 0b0101);
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_A, 0, 2, 0, 2, 0b0101);
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_A, 0, 3, 0, 3, 0b0101);
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 1 /*dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

#ifdef TRISC_MATH
// fp8: ELWMUL with srcB's column 0 broadcast over each row, srcA / srcB / DEST rows stepping together (8 rows an
// instruction, 8 a tile): DEST row r = srcA row r x srcB row r, column 0 (sparse_sdpa_fused_dequant.hpp's math)
inline void fp8_dequant_math_init() {
    addr_mod_t{.srca = {.incr = 8}, .srcb = {.incr = 8}, .dest = {.incr = 8}}.set(ADDR_MOD_2);
    ckernel_template tmp(1, 8, TT_OP_ELWMUL(0, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_2, 0));
    tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
    tmp.program();
    math::reset_counters(p_setrwc::SET_ABD_F);
    math::_configure_src_zero_flag_(false);  // (fp8 zeros widen into srcA with a residual unless flushed)
}
inline void fp8_dequant_math_tile(uint32_t dst_index) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    ckernel_template::run();
    math::clear_dst_reg_addr();
}
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel::sfpu {
// A DEST face: fp32 v -> round(v x factor) (ties to even) as a sign-magnitude int32 (as accumulator_compute.cpp's
// snap_to_sign_magnitude_count, the factor at run time: 1 / q, or gate / q)
inline void snap_count_rt(uint32_t factor_bits) {
    const sfpi::vFloat factor = __builtin_bit_cast(float, factor_bits);
    const sfpi::vFloat largest_count = 2147483520.0f;
    const sfpi::vInt two_to_23_bits = 0x4B000000;
#pragma GCC unroll 8
    for (int face_row = 0; face_row < 8; face_row++) {
        const sfpi::vFloat scaled = sfpi::vFloat(sfpi::dst_reg[0]) * factor;
        const sfpi::vFloat magnitude = sfpi::min(sfpi::abs(scaled), largest_count);
        sfpi::vInt count_magnitude = sfpi::as<sfpi::vInt>(magnitude + 8388608.0f) - two_to_23_bits;
        const sfpi::vInt mantissa_shift = sfpi::exexp(magnitude) - 23;
        v_if(mantissa_shift >= 0) {
            count_magnitude = sfpi::as<sfpi::vInt>(sfpi::shft(
                sfpi::as<sfpi::vUInt>(sfpi::exman(magnitude, sfpi::MantissaMode::WithUnitBit)), mantissa_shift));
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::copysgn(sfpi::as<sfpi::vSMag>(count_magnitude), scaled);
        sfpi::dst_reg++;
    }
}
template <uint32_t step_bits>
inline void count_to_float() {
#pragma GCC unroll 8
    for (int face_row = 0; face_row < 8; face_row++) {
        const sfpi::vSMag count = sfpi::dst_reg[0];
        sfpi::dst_reg[0] =
            sfpi::convert<sfpi::vFloat>(count, sfpi::RoundMode::Nearest) * __builtin_bit_cast(float, step_bits);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

constexpr uint32_t kStepBits = 0x38800000u;         // q = 2^-14
constexpr uint32_t kInverseStepBits = 0x46800000u;  // 1 / q = 2^14
constexpr uint32_t kDestTiles = compute_kernel_lib::DEST_AUTO_LIMIT;

void kernel_main() {
    constexpr bool fp8 = get_compile_time_arg_val(0) != 0;
    constexpr uint32_t n_rows = get_compile_time_arg_val(1);
    constexpr uint32_t row_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tokens = get_compile_time_arg_val(3);
    constexpr uint32_t cb_row = 0, cb_w = 1, cb_fac = 2, cb_meta = 3, cb_out = 16, cb_acc = 24;
    constexpr uint32_t cb_b = fp8 ? cb_fac : cb_w;

    compute_kernel_hw_startup(cb_row, cb_b, cb_acc);
    if constexpr (fp8) {
        // the factor CB is UInt16 on the host (the JIT takes no FP16 CB next to the BF16 ones): srcB FP16
        UNPACK((_llk_unpack_reconfig_data_format_srcb_impl_<DST_ACCUM_MODE, p_dim_stride_target::FACE_ROW_MAJOR>(
            static_cast<uint32_t>(DataFormat::Float16),
            static_cast<uint32_t>(DataFormat::Float16),
            get_local_cb_interface(cb_fac).fifo_page_size,
            FACE_R_DIM,
            4)));
        MATH((_llk_math_reconfig_data_format_srcb_<DST_ACCUM_MODE>(static_cast<uint32_t>(DataFormat::Float16))));
        UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(cb_row, cb_fac, Transpose::None)));
    }
    uint8_t has_rows[tokens];
    for (uint32_t i = 0; i < tokens; ++i) {
        has_rows[i] = 0;
    }
    cb_reserve_back(cb_acc, tokens * row_tiles);
    {
        DeviceZoneScopedN("CMB-CMP");
        for (uint32_t n = 0; n < n_rows; ++n) {
            cb_wait_front(cb_meta, 1);
            const uint32_t token = read_tile_value(cb_meta, 0, 0);
            const uint32_t gate = read_tile_value(cb_meta, 0, 1);
            cb_pop_front(cb_meta, 1);
            // the snap's factor: 1 / q (bf16: the gate is in DEST already), gate / q (fp8): the gate's fp32 bits with
            // 14 added to the exponent (exact; a zero gate stays 0)
            // (fp8: the factor tile carries 2^10 too - kFactorShift: gate x 2^(14 - 10))
            const uint32_t factor_bits = fp8 ? (gate == 0 ? 0u : (gate << 16) + (4u << 23)) : kInverseStepBits;
            cb_wait_front(cb_row, row_tiles);
            if constexpr (fp8) {
                cb_wait_front(cb_fac, row_tiles);
            }
            cb_wait_front(cb_w, 1);
            pack_reconfig_l1_acc(has_rows[token] ? 1 : 0);
            for (uint32_t first = 0; first < row_tiles; first += kDestTiles) {
                const uint32_t count = row_tiles - first < kDestTiles ? row_tiles - first : kDestTiles;
                tile_regs_acquire();
                if constexpr (fp8) {
                    MATH((fp8_dequant_math_init()));  // (again a batch: the snap's SFPU init in between)
                    for (uint32_t j = 0; j < count; ++j) {
                        UNPACK((fp8_dequant_unpack_tile(
                            get_local_cb_interface(cb_row).fifo_rd_ptr - 1 +
                                (first + j) * get_local_cb_interface(cb_row).fifo_page_size,
                            get_local_cb_interface(cb_fac).fifo_rd_ptr - 1 +
                                (first + j) * get_local_cb_interface(cb_fac).fifo_page_size)));
                        MATH((fp8_dequant_math_tile(j)));
                    }
                } else {
                    mul_bcast_scalar_init(cb_row, cb_w);
                    for (uint32_t j = 0; j < count; ++j) {
                        mul_tiles_bcast_scalar(cb_row, cb_w, first + j, 0, j);
                    }
                }
                MATH((SFPU_UNARY_INIT(unused)));
                for (uint32_t j = 0; j < count; ++j) {
                    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                        DST_SYNC_MODE, DST_ACCUM_MODE, snap_count_rt, j, VectorMode::RC, factor_bits)));
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < count; ++j) {
                    pack_tile<true>(j, cb_acc, token * row_tiles + first + j);
                }
                tile_regs_release();
            }
            has_rows[token] = 1;
            cb_pop_front(cb_row, row_tiles);
            if constexpr (fp8) {
                cb_pop_front(cb_fac, row_tiles);
            }
            cb_pop_front(cb_w, 1);
        }
    }
    pack_reconfig_l1_acc(0);
    cb_push_back(cb_acc, tokens * row_tiles);
    cb_wait_front(cb_acc, tokens * row_tiles);
    reconfig_data_format_srca(cb_acc);
    pack_reconfig_data_format(cb_out);
    copy_init(cb_acc);
    MATH((SFPU_UNARY_INIT(unused)));
    for (uint32_t token = 0; token < tokens; ++token) {  // (every token got rows: n_rows >= tokens)
        cb_reserve_back(cb_out, row_tiles);
        for (uint32_t first = 0; first < row_tiles; first += kDestTiles) {
            const uint32_t count = row_tiles - first < kDestTiles ? row_tiles - first : kDestTiles;
            tile_regs_acquire();
            for (uint32_t j = 0; j < count; ++j) {
                copy_tile(cb_acc, token * row_tiles + first + j, j);
            }
            for (uint32_t j = 0; j < count; ++j) {
                MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, count_to_float, (kStepBits), j, VectorMode::RC)));
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < count; ++j) {
                pack_tile(j, cb_out);
            }
            tile_regs_release();
        }
        cb_push_back(cb_out, row_tiles);
    }
}
