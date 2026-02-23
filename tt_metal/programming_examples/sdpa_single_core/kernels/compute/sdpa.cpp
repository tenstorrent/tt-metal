// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <type_traits>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/debug/assert.h"
#include "api/debug/dprint.h"
#include "api/compute/compute_kernel_api.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "ckernel_sfpu_exp.h"
#include "sfpi.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/reduce_custom.h"

#include <tools/profiler/kernel_profiler.hpp>

using std::uint32_t;

#ifdef TRISC_PACK

constexpr auto bits = [](float x) constexpr { return __builtin_bit_cast(std::uint32_t, x); };
constexpr auto lo16 = [](float x) constexpr { return static_cast<std::uint16_t>(bits(x) & 0xFFFFu); };
constexpr auto hi16 = [](float x) constexpr { return static_cast<std::uint16_t>(bits(x) >> 16); };

#ifdef ARCH_WORMHOLE
#define ADDR_MOD_X ADDR_MOD_3
#else
#define ADDR_MOD_X ADDR_MOD_7
#endif

ALWI void INSERT_SFPNOP() {
#ifdef ARCH_WORMHOLE
    TTI_SFPNOP;
#endif
}

template <bool USE_SFPARECIP_INSTR, int POLY_DEGREE>
constexpr bool can_preload_ln2_constants() {
#ifdef ARCH_WORMHOLE
    return false;
#else
    return (USE_SFPARECIP_INSTR || POLY_DEGREE == 1 || POLY_DEGREE == 2);
#endif
}

/**
 * Computes exp(x) using polynomial approximation after range reduction.
 *
 * Scales by configured factor, then reduces to exp(r) * 2^k
 * where r = x - k*ln(2). Uses either SFPARECIP instruction or multi-term polynomial (degree 1-4)
 * to compute exp(r), then reconstructs full result via exponent manipulation,
 * clamping the exponent to handle large positive or negative inputs.
 *
 * @tparam USE_SFPARECIP_INSTR Use hardware SFPARECIP instruction (true) or polynomial evaluation (false). Only
 * supported on Blackhole.
 * @tparam SCALE_EN Apply scaling factor from LREG to input values
 * @tparam ITERATIONS Number of 32-element vectors to process per tile
 * @tparam POLY_DEGREE Polynomial degree (1-4) when USE_SFPARECIP_INSTR=false; higher improves accuracy
 * @tparam IS_FP32_DEST_ACC_EN Float32 accumulation to dest register enabled.
 * @tparam SCALE_BF16 Bfloat16 scale factor represented as uint16_t.
 */
template <
    bool SCALE_EN,
    int ITERATIONS,
    bool USE_SFPARECIP_INSTR,
    int POLY_DEGREE,
    bool IS_FP32_DEST_ACC_EN,
    uint16_t SCALE_BF16>
void calculate_exponential_polynomial() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);

    constexpr float LN2_RECIP = 1.44269504088896340736f;  // 1/ln(2)
    constexpr float M_LN2 = -0.69314718055994530942f;     // -ln(2)

    if (!USE_SFPARECIP_INSTR) {
        ASSERT(POLY_DEGREE >= 1 && POLY_DEGREE <= 4);

        // Evaluate polynomial f(x) = c0 + c1 * x + c2 * x^2 + ... using Horner's method.
        constexpr float c0 = (POLY_DEGREE == 1)   ? 1.03022936050163882354355235184958220293399209290987f
                             : (POLY_DEGREE == 2) ? 0.999848792924395313327307061545061386175496934006f
                             : (POLY_DEGREE == 3) ? 0.99992449655091231753798502608929170703152709521188f
                                                  : 1.0000001510806179002040134468008959160576106495165f;
        constexpr float c1 = (POLY_DEGREE == 1)   ? 1.0201394465967894800285756834161653337107187804001f
                             : (POLY_DEGREE == 2) ? 1.01508760098521056684783640695492761469306929535975f
                             : (POLY_DEGREE == 3) ? 0.99993960415029750534472970577402987498389428593233f
                                                  : 0.99996228117047652035114096488703457970402030983204f;
        constexpr float c2 = (POLY_DEGREE == 2)   ? 0.50628367056745568861842335616023694454759126020461f
                             : (POLY_DEGREE == 3) ? 0.50502329058055065591138054839814880512001604099324f
                                                  : 0.49998365704615426417337683145647067790385638465486f;
        constexpr float c3 = (POLY_DEGREE == 3) ? 0.16817330195731531429790827442800245470170482723302f
                                                : 0.16792157982882225102649214918047336097544632172075f;
        constexpr float c4 = 4.1959439860014343843000081999668024587178974865521e-2;

        switch (POLY_DEGREE) {
            case 4:
                TTI_SFPLOADI(p_sfpu::LREG3, 0xA, lo16(c4));
                TTI_SFPLOADI(p_sfpu::LREG3, 0x8, hi16(c4));
                [[fallthrough]];
            case 3:
                TTI_SFPLOADI(p_sfpu::LREG4, 0xA, lo16(c3));
                TTI_SFPLOADI(p_sfpu::LREG4, 0x8, hi16(c3));
                [[fallthrough]];
            case 2:
                TTI_SFPLOADI(p_sfpu::LREG5, 0xA, lo16(c2));
                TTI_SFPLOADI(p_sfpu::LREG5, 0x8, hi16(c2));
                [[fallthrough]];
            case 1:
                TTI_SFPLOADI(p_sfpu::LREG6, 0xA, lo16(c1));
                TTI_SFPLOADI(p_sfpu::LREG6, 0x8, hi16(c1));
                TTI_SFPLOADI(p_sfpu::LREG7, 0xA, lo16(c0));
                TTI_SFPLOADI(p_sfpu::LREG7, 0x8, hi16(c0));
            default: break;
        }
    }

    if constexpr (can_preload_ln2_constants<USE_SFPARECIP_INSTR, POLY_DEGREE>()) {
        TTI_SFPLOADI(p_sfpu::LREG3, 0xA, lo16(LN2_RECIP));
        TTI_SFPLOADI(p_sfpu::LREG3, 0x8, hi16(LN2_RECIP));
        TTI_SFPLOADI(p_sfpu::LREG4, 0xA, lo16(M_LN2));
        TTI_SFPLOADI(p_sfpu::LREG4, 0x8, hi16(M_LN2));
    }

    for (int d = 0; d < ITERATIONS; d++) {
        // Load the input.
        constexpr uint8_t input_type = IS_FP32_DEST_ACC_EN ? InstrModLoadStore::FP32 : InstrModLoadStore::FP16B;
        TTI_SFPLOAD(p_sfpu::LREG2, input_type, ADDR_MOD_X, 0);

        if constexpr (SCALE_EN) {
            TTI_SFPLOADI(p_sfpu::LREG0, 0, SCALE_BF16);
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
            INSERT_SFPNOP();
        }

        // Multiply by 1/ln(2) and round.
        if constexpr (can_preload_ln2_constants<USE_SFPARECIP_INSTR, POLY_DEGREE>()) {
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        } else {
            TTI_SFPLOADI(p_sfpu::LREG1, 0xA, lo16(LN2_RECIP));
            TTI_SFPLOADI(p_sfpu::LREG1, 0x8, hi16(LN2_RECIP));
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        }
        INSERT_SFPNOP();
        TTI_SFP_STOCH_RND(
            0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8);  // Clamp to [-127,+127].
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);

        if constexpr (USE_SFPARECIP_INSTR) {
#ifdef ARCH_BLACKHOLE
            // Calculate floor(x) by setting v=v-1 if v>u.
            TTI_SFPGT(0, p_sfpu::LREG0, p_sfpu::LREG1, 1);                                    // SFPGT_MOD1_SET_CC
            TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 2);  // SFPMAD_MOD1_NEGATE_VC
            TTI_SFPENCC(0, 0, 0, 0);

            // Calculate exp(x - k*ln2).
            TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG0, 2);
#else
            ASSERT(false);  // TTI_SFPARECIP instruction only supported on Blackhole".
#endif
        } else {
            if constexpr (can_preload_ln2_constants<USE_SFPARECIP_INSTR, POLY_DEGREE>()) {
                TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            } else {
                TTI_SFPLOADI(p_sfpu::LREG0, 0xA, lo16(M_LN2));
                TTI_SFPLOADI(p_sfpu::LREG0, 0x8, hi16(M_LN2));
                TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            }
            INSERT_SFPNOP();

            // Calculate polynomial.
            if constexpr (POLY_DEGREE == 1) {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            } else if constexpr (POLY_DEGREE == 2) {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            } else if constexpr (POLY_DEGREE == 3) {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            } else {  // degree 4.
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                INSERT_SFPNOP();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            }
            INSERT_SFPNOP();
        }

        // Multiply by 2^k.
        TT_SFPADDI(0x42fe, p_sfpu::LREG1, 0);  // Add 127.
        INSERT_SFPNOP();
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT8);
        TTI_SFPSETEXP(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        INSERT_SFPNOP();

        // Handle underflow: if k == 0, exp(x) = 0 (fixes -inf case).
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 6);  // Set LaneFlags = (LREG1 == 0) and enable CC.
        TTI_SFPLOADI(p_sfpu::LREG2, 0, 0);     // LREG2 = 0 ONLY for lanes where LREG1 == 0.
        TTI_SFPENCC(0, 0, 0, 0);               // Disable CC and clear LaneFlags - ALL lanes active again.

        // Store the result.
        if constexpr (!IS_FP32_DEST_ACC_EN) {
            // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it
            // so convert to bfloat16 using round-to-nearest-even.
            TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        }
        TTI_SFPSTORE(p_sfpu::LREG2, input_type, ADDR_MOD_X, 0);
        TTI_INCRWC(0, 4, 0, 0);  // Skip odd columns.
    }
}

/**
 * exp_tile on only the columns 0:8 of a face
 */
template <bool SDPA_EXP_APPROX_MODE, uint16_t scale_bf16>
void calculate_exponential_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
    if constexpr (SDPA_EXP_APPROX_MODE) {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::vFloat result = ckernel::sfpu::
                _calculate_exponential_piecewise_<EXP_APPROX_MODE, true /*SCALE_EN*/, true /*SKIP_POSITIVE_CHECK*/>(
                    val, scale_bf16);
            sfpi::dst_reg[0] = result;

            // Stride by 2 to skip columns 8:16 of the face
            sfpi::dst_reg += 2;
        }
    } else {
        constexpr int polynomial_degree = DST_ACCUM_MODE ? 4 : 2;
        calculate_exponential_polynomial<
            true,
            ITERATIONS_HALF_FACE,
            false,
            polynomial_degree,
            DST_ACCUM_MODE,
            scale_bf16>();
    }
}

template <bool SDPA_EXP_APPROX_MODE, uint16_t scale_bf16>
void exp_tile_first_column(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_<false /*APPROXIMATE*/>(
        calculate_exponential_first_column<SDPA_EXP_APPROX_MODE, scale_bf16>, idst, (int)VectorMode::C);
}

#endif

// Toggle: interleave drain sub_exp with split matmul for first QKT@V row.
// Requires sbh=1, kt_num_subblocks=2. Overlaps EXP (SFPU) with matmul (FPU).
#define OVERLAP_DRAIN_WITH_MATMUL

// Template-driven profiling: MaybeDeviceZoneScopedN(ENABLED, name)
// When ENABLED=true: RAII profileScope writes timestamps (same as DeviceZoneScopedN)
// When ENABLED=false: empty struct, zero overhead (compiler eliminates entirely)
// When PROFILE_KERNEL not defined: macro is no-op regardless of ENABLED
//
// Usage: sdpa_inner_loop_step<true, ...>() to profile a K-chunk iteration,
//        sdpa_inner_loop_step<false, ...>() for zero-overhead iterations.
#if defined(PROFILE_KERNEL)
template <bool Enabled, uint32_t timer_id>
struct MaybeProfileScope {
    inline __attribute__((always_inline)) MaybeProfileScope() {}
    inline __attribute__((always_inline)) ~MaybeProfileScope() {}
};
template <uint32_t timer_id>
struct MaybeProfileScope<true, timer_id> : kernel_profiler::profileScope<timer_id> {};

#define MaybeDeviceZoneScopedN(ENABLED, name)                                  \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    MaybeProfileScope<ENABLED, hash> zone;
#else
#define MaybeDeviceZoneScopedN(ENABLED, name)
#endif
/**
 * out_cb = exp((in0_cb - in1_cb) * scale_fp32)
 * only at 2*q_subblock and 2*q_subblock+1 elements
 * Caller manages cb_reserve_back/cb_push_back for out_cb.
 */
template <bool PROFILING_ENABLED, uint32_t scale_fp32, uint32_t SBH>
void sub_exp_first_col_blocks(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t q_subblock) {
    constexpr uint32_t tiles_per_row = SBH;
    const uint32_t global_row_base = q_subblock * tiles_per_row;
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    sub_tiles_init(in0_cb, in1_cb);
    exp_packthread_tile_init<EXP_APPROX_MODE, false>();

    cb_wait_front(in0_cb, (q_subblock + 1) * tiles_per_row);
    cb_wait_front(in1_cb, (q_subblock + 1) * tiles_per_row);

    {
        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "S_SUB");
        tile_regs_acquire();
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            uint32_t tile_index = global_row_base + i;
            sub_tiles(in0_cb, in1_cb, tile_index, tile_index, i /*dst_index*/);
        }
        tile_regs_commit();
    }

    {
        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "S_EXP_AND_PACK");
        tile_regs_wait();
        for (uint32_t dst_index = 0; dst_index < tiles_per_row; dst_index++) {
            PACK((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(dst_index)));
        }
        PACK(TTI_STALLWAIT(p_stall ::STALL_PACK, p_stall ::WAIT_SFPU));

        for (uint32_t i = 0; i < tiles_per_row; i++) {
            pack_tile<false>(i /*dst_index*/, out_cb);
        }

        tile_regs_release();
    }
}

/**
 * out_cb[row] += in0_cb[row] * bcast_cols(in1_cb[row])
 *
 * Computes the product and L1-accumulates it onto out_cb at the absolute position.
 * out_cb must be in RESERVED (not pushed) state — this writes into the reserved region
 * using pack_tile<true> with L1 accumulate, which is the correct CB protocol.
 *
 * This replaces the old mul_tiles_bcast_cols_inplace + add_block_inplace pair, which
 * incorrectly used pack_tile<true> to overwrite already-pushed CB data.
 */
template <uint32_t SBH>
void mul_bcast_cols_l1_acc(
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t q_subblock, uint32_t write_q_subblock = 0xFFFFFFFF) {
    if (write_q_subblock == 0xFFFFFFFF) {
        write_q_subblock = q_subblock;
    }
    constexpr uint32_t tiles_per_row = SBH;
    const uint32_t read_row_base = q_subblock * tiles_per_row;
    const uint32_t write_row_base = write_q_subblock * tiles_per_row;

    mul_bcast_cols_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, (q_subblock + 1) * tiles_per_row);
    cb_wait_front(in1_cb, (q_subblock + 1) * tiles_per_row);

    PACK((llk_pack_reconfig_l1_acc(1)));
    tile_regs_acquire();
    for (uint32_t i = 0; i < tiles_per_row; i++) {
        uint32_t src_tile_index = read_row_base + i;
        mul_tiles_bcast_cols(in0_cb, in1_cb, src_tile_index, src_tile_index, i /*dst_index*/);
    }
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t i = 0; i < tiles_per_row; i++) {
        pack_tile<true>(i, out_cb, write_row_base + i);  // L1 accumulate into reserved out_cb
    }
    tile_regs_release();
    PACK((llk_pack_reconfig_l1_acc(0)));
}

template <uint32_t SBH, uint32_t SBW>
void mul_block_bcast_cols_acc(
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t q_subblock, uint32_t write_q_subblock = 0xFFFFFFFF) {
    if (write_q_subblock == 0xFFFFFFFF) {
        write_q_subblock = q_subblock;
    }
    constexpr uint32_t tiles_per_row = SBH;
    constexpr uint32_t tiles_per_column = SBW;
    static_assert(tiles_per_row * tiles_per_column <= 8, "SBH * SBW must fit in DST (max 8 tiles)");
    const uint32_t read_row_base = q_subblock * tiles_per_row;
    const uint32_t write_row_base = write_q_subblock * tiles_per_row;
    mul_bcast_cols_init_short(in0_cb, in1_cb);

    cb_wait_front(in0_cb, (q_subblock + 1) * tiles_per_row * tiles_per_column);
    cb_wait_front(in1_cb, (q_subblock + 1) * tiles_per_row);

    PACK((llk_pack_reconfig_l1_acc(1 /*pack accumulate*/)));
    tile_regs_acquire();
    uint32_t dst_index = 0;
    for (uint32_t i = 0; i < tiles_per_row; i++) {
        for (uint32_t j = 0; j < tiles_per_column; j++) {
            uint32_t in0_tile_index = (read_row_base + i) * tiles_per_column + j;
            mul_tiles_bcast_cols(in0_cb, in1_cb, in0_tile_index, read_row_base + i, dst_index++);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    dst_index = 0;
    for (uint32_t i = 0; i < tiles_per_row; i++) {
        for (uint32_t j = 0; j < tiles_per_column; j++) {
            uint32_t out_tile_index = (write_row_base + i) * tiles_per_column + j;
            pack_tile<true>(dst_index++, out_cb, out_tile_index);
        }
    }
    tile_regs_release();
    PACK((llk_pack_reconfig_l1_acc(false)));
}

/**
 * sub_exp_block_bcast_cols: reads from row buffer, subtracts max, applies exp.
 * Reads from read_cb (row buffer), subtracts max_cb, applies exp, writes to write_cb (sequential),
 * and accumulates row sums into reduce_cb (absolute position with L1 accumulate).
 */
template <
    bool PROFILING_ENABLED,
    uint32_t scale_fp32,
    uint32_t SBH,
    uint32_t SBW,
    bool do_reduce = true,
    int vector_mode = (int)VectorMode::RC>
void sub_exp_block_bcast_cols(
    uint32_t read_cb,
    uint32_t max_cb,
    uint32_t write_cb,
    uint32_t reduce_cb,
    uint32_t cols_in_row,
    uint32_t q_subblock,
    uint32_t kt_subblock) {
    constexpr uint32_t tiles_per_row = SBH;
    constexpr uint32_t tiles_per_column = SBW;
    static_assert(tiles_per_row * tiles_per_column <= 8, "SBH * SBW must fit in DST (max 8 tiles)");
    const uint32_t max_row_base = q_subblock * tiles_per_row;
    const uint32_t global_col_base = kt_subblock * tiles_per_column;

    // Initialize operation
    sub_bcast_cols_init_short(read_cb, max_cb);
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));

    // Wait for tiles:
    // - read_cb: one full row (row buffer)
    // - max_cb: cumulative max values
    cb_wait_front(read_cb, tiles_per_row * cols_in_row);
    cb_wait_front(max_cb, (q_subblock + 1) * tiles_per_row);

    {
        tile_regs_acquire();
        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "SUB");
        uint32_t dst_index = 0;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            for (uint32_t j = 0; j < tiles_per_column; j++) {
                // Row buffer positions: row i starts at i * cols_in_row
                uint32_t read_tile = i * cols_in_row + global_col_base + j;
                sub_tiles_bcast_cols(read_cb, max_cb, read_tile, max_row_base + i, dst_index++);
            }
        }
        tile_regs_commit();
    }

    {
        tile_regs_wait();
        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "EXP");
        uint32_t dst_index = 0;
        constexpr int iterations = (vector_mode == (int)VectorMode::RC) ? 32 : 8;
        constexpr int vector_mode_exp = (vector_mode == (int)VectorMode::RC) ? (int)VectorMode::None : vector_mode;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            for (uint32_t j = 0; j < tiles_per_column; j++) {
                exp_packthread_tile<
                    true,   // approx
                    true,   // fast_and_approx
                    false,  // scale_en
                    false,  // skip_positive_check
                    InputClamping::None,
                    iterations>(dst_index++, vector_mode_exp);
            }
        }
        PACK(TTI_STALLWAIT(p_stall ::STALL_PACK, p_stall ::WAIT_SFPU));
    }

    {
        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "PACK SUB_EXP");

        // Pack to write_cb sequentially via pack_tile<false>
        uint32_t dst_index = 0;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            for (uint32_t j = 0; j < tiles_per_column; ++j) {
                pack_tile<false>(dst_index++, write_cb);
            }
        }

        // Reduce to reduce_cb at absolute positions with L1 accumulate
        if constexpr (do_reduce) {
            dst_index = 0;
            for (uint32_t i = 0; i < tiles_per_row; i++) {
                if (global_col_base > 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
                for (uint32_t j = 0; j < tiles_per_column; ++j) {
                    pack_tile<true>(dst_index++, reduce_cb, max_row_base + i);
                    if (global_col_base == 0 && j == 0) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                }
            }
        }
    }

    // Restore packer ReLU config after all exp operations complete
    PACK((llk_pack_relu_config(ReluType::NO_RELU)));

    tile_regs_release();
    if constexpr (do_reduce) {
        PACK((llk_pack_reconfig_l1_acc(0)));
    }
}

ALWI void sdpa_reduce_copy_tile_to_dst_init_short(uint32_t cbid, uint32_t transpose = 0) {
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        transpose, true /*transpose within 16x16 face*/, cbid)));

    MATH((llk_math_eltwise_unary_datacopy_init<
          A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          false,  // is_int_fpu_en
          false   // tilize
          >(cbid)));
}

/**
 * Reduces rows of in0_cb to single tiles via max-reduce, writing results sequentially to out_cb.
 * Caller manages cb_reserve_back/cb_push_back for out_cb.
 * in0_row_group_index overrides in0_cb read position (e.g., 0 when reading from a row buffer).
 */
template <PoolType pool_type, ReduceDim reduce_dim, uint32_t scale_cb, uint32_t cols, uint32_t SBH>
void reduce_c_row_group(
    uint32_t in0_cb,
    uint32_t out_cb,
    uint32_t prev_cb,
    uint32_t row_group_index,
    bool do_eltwise_max = false,
    uint32_t in0_row_group_index = 0xFFFFFFFF) {
    constexpr uint32_t GROUP_SIZE = SBH;
    const uint32_t row_start = row_group_index * GROUP_SIZE;

    const uint32_t in0_rgi = (in0_row_group_index != 0xFFFFFFFF) ? in0_row_group_index : row_group_index;
    const uint32_t in0_row_start = in0_rgi * GROUP_SIZE;

    const uint32_t cumulative_input_tiles = (in0_rgi + 1) * GROUP_SIZE * cols;
    const uint32_t cumulative_prev_tiles = (row_group_index + 1) * GROUP_SIZE;

    cb_wait_front(scale_cb, 1);
    cb_wait_front(in0_cb, cumulative_input_tiles);

    tile_regs_acquire();

    if (do_eltwise_max) {
        cb_wait_front(prev_cb, cumulative_prev_tiles);
        sdpa_reduce_copy_tile_to_dst_init_short(prev_cb);
        for (uint32_t i = 0; i < GROUP_SIZE; i++) {
            copy_tile(prev_cb, row_start + i, i);
        }
    }

    reduce_block_max_row_init<cols>();
    for (uint32_t i = 0; i < GROUP_SIZE; i++) {
        const uint32_t input_tile_start = (in0_row_start + i) * cols;
        reduce_block_max_row<cols>(in0_cb, scale_cb, input_tile_start, i);
    }
    reduce_block_max_row_uninit(in0_cb);

    tile_regs_commit();
    tile_regs_wait();

    for (uint32_t i = 0; i < GROUP_SIZE; i++) {
        pack_tile<false>(i, out_cb);
    }

    tile_regs_release();
}

/**
 * Initializes matmul HW, performs a blocked matmul of a subblock, and packs the result tiles to an output CB.
 *
 * Init: configures unpacker/math for the matmul dimensions (always called to ensure correct HW state).
 * Matmul phase: accumulates in0[row] @ in1[col] over the inner dimension into DST.
 * Pack phase: writes DST tiles to out_cb at row-major positions.
 *
 * @tparam TRANSPOSE   Whether to transpose in1 tiles during matmul
 * @tparam SUBBLOCK_W  Output subblock width in tiles (columns per subblock)
 * @tparam SUBBLOCK_H  Output subblock height in tiles (rows per subblock)
 * @tparam INNER_DIM   Inner (accumulation) dimension in tiles
 * @tparam IN1_STRIDE  Stride for in1 index per inner-dim step
 * @tparam OUT_NUM_COLS Total columns in the output matrix (for row offset calculation)
 */
template <
    bool TRANSPOSE,
    uint32_t SUBBLOCK_W,
    uint32_t SUBBLOCK_H,
    uint32_t INNER_DIM,
    uint32_t IN1_STRIDE,
    uint32_t OUT_NUM_COLS,
    bool SEQUENTIAL_OUTPUT = false>
void blocked_matmul_and_pack(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t in0_index_start,
    uint32_t in1_index_start,
    uint32_t q_subblock,
    uint32_t out_col_offset) {
    // --- Init: ensure HW is configured for this matmul ---
    mm_block_init_short(in0_cb, in1_cb, TRANSPOSE, SUBBLOCK_W, SUBBLOCK_H, INNER_DIM);

    // --- Matmul phase ---
    tile_regs_acquire();
    uint32_t dst_index = 0;
    uint32_t in0_index = in0_index_start;
    uint32_t in1_index = in1_index_start;
    for (uint32_t inner = 0; inner < INNER_DIM; ++inner) {
        matmul_block(in0_cb, in1_cb, in0_index, in1_index, dst_index, TRANSPOSE, SUBBLOCK_W, SUBBLOCK_H, INNER_DIM);
        in0_index++;
        in1_index += IN1_STRIDE;
    }
    tile_regs_commit();

    // --- Pack phase ---
    tile_regs_wait();
    if constexpr (SEQUENTIAL_OUTPUT) {
        uint32_t dst_idx = 0;
        for (uint32_t r = 0; r < SUBBLOCK_H; r++) {
            for (uint32_t c = 0; c < SUBBLOCK_W; c++) {
                pack_tile<false>(dst_idx++, out_cb);
            }
        }
    } else {
        uint32_t dst_idx = 0;
        for (uint32_t r = 0; r < SUBBLOCK_H; r++) {
            uint32_t out_row_offset = (r + q_subblock * SUBBLOCK_H) * OUT_NUM_COLS;
            for (uint32_t c = 0; c < SUBBLOCK_W; c++) {
                pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset + c);
                dst_idx++;
            }
        }
    }
    tile_regs_release();
}

// Temporary: pre-populate prev buffers so the first iteration is numerically correct without
// special-casing it inside the inner loop (which would make the first pass faster and pollute
// worst-case timing measurements). In production the inner loop would handle initialization
// in its first pass instead.
template <uint32_t Sq_chunk_t, uint32_t head_dim_t>
void InitPrevBuffers(uint32_t cb_sum, uint32_t cb_out, uint32_t cb_max, uint32_t cb_neginf) {
    // prev_sum: zeros (Sq_chunk_t tiles)
    cb_reserve_back(cb_sum, Sq_chunk_t);
    tile_regs_acquire();
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < Sq_chunk_t; i++) {
        pack_tile<false>(0, cb_sum);
    }
    tile_regs_release();
    cb_push_back(cb_sum, Sq_chunk_t);

    // prev_out: zeros (Sq_chunk_t * head_dim_t tiles)
    cb_reserve_back(cb_out, Sq_chunk_t * head_dim_t);
    tile_regs_acquire();
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < Sq_chunk_t * head_dim_t; i++) {
        pack_tile<false>(0, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, Sq_chunk_t * head_dim_t);

    // prev_max: -inf (Sq_chunk_t tiles)
    // Writer produces a -inf tile in cb_neginf; we copy it to DST then pack to cb_max.
    // Note: cb_neginf is NOT popped — it stays available across Q-chunk iterations.
    cb_wait_front(cb_neginf, 1);
    copy_tile_to_dst_init_short(cb_neginf);
    cb_reserve_back(cb_max, Sq_chunk_t);
    tile_regs_acquire();
    copy_tile(cb_neginf, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < Sq_chunk_t; i++) {
        pack_tile<false>(0, cb_max);
    }
    tile_regs_release();
    cb_push_back(cb_max, Sq_chunk_t);
}

// ===================== Normalization Functions =====================

#ifdef TRISC_MATH
template <bool legacy_compat = true>
void calculate_recip_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat out = ckernel::sfpu::_reciprocal_compat_<APPROX ? 2 : 3>(in);
        if constexpr (DST_ACCUM_MODE || APPROX) {
            sfpi::dst_reg[0] = out;
        } else {
            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(out, 0));
        }
        sfpi::dst_reg += 2;
    }
}

void recip_tile_first_column(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX>(calculate_recip_first_column<true>, idst, (int)VectorMode::C);
}
#endif

/**
 * Per-row streaming normalization for one subblock row (SBH tiles of sum, SBH*head_dim_t tiles of output).
 * For each sub-row tile:
 *   1. matmul_reduce: sum_tile × col_identity → scratch (collapses partial row sums to column 0)
 *   2. recip in-place: scratch = 1/sum
 *   3. normalize: out_tiles × bcast_cols(1/sum) → normalized_out (streaming, 1 tile at a time)
 * Consumes (pops) sum and output tiles from cur_sum_cb / cur_out_cb.
 * scratch_cb must be a 1-tile CB.
 */
template <bool PROFILING_ENABLED, uint32_t SBH, uint32_t head_dim_t_>
void normalize_row_streaming(
    uint32_t cur_sum_cb,
    uint32_t cur_out_cb,
    uint32_t col_identity_cb,
    uint32_t scratch_cb,
    uint32_t normalized_out_cb) {
    for (uint32_t s = 0; s < SBH; s++) {
        // 1+2. Fused matmul_reduce + recip: sum × col_identity → recip → 1/sum in scratch
        // Keeps the matmul result in DST[0] and applies recip directly, avoiding a
        // pack→scratch→copy-back-to-DST round-trip.
        {
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "NORM_MATMUL_RECIP");
            constexpr uint32_t N = 1;
            mm_block_init_short(cur_sum_cb, col_identity_cb, 0, N, 1, N);
            reconfig_data_format(col_identity_cb, cur_sum_cb);

            cb_wait_front(col_identity_cb, N);
            cb_wait_front(cur_sum_cb, 1);

            cb_reserve_back(scratch_cb, 1);
            tile_regs_acquire();
            matmul_block(cur_sum_cb, col_identity_cb, 0, 0, 0, 0, N, 1, N);
            // Recip directly in DST[0] — serial dependency, no overlap, but avoids DST round-trip
            recip_tile_init();
            MATH((recip_tile_first_column(0)));
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, scratch_cb);
            tile_regs_release();
            cb_push_back(scratch_cb, 1);

            cb_pop_front(cur_sum_cb, 1);
        }

        // 3. normalize: multiply all head_dim_t_ output tiles by bcast_cols(1/sum) in one DST batch
        static_assert(head_dim_t_ <= 8, "head_dim_t must fit in DST (max 8 tiles with fp16b double-buffer)");
        {
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "NORM_MUL_BCAST");
            mul_bcast_cols_init_short(cur_out_cb, scratch_cb);
            cb_wait_front(cur_out_cb, head_dim_t_);
            cb_wait_front(scratch_cb, 1);

            cb_reserve_back(normalized_out_cb, head_dim_t_);
            tile_regs_acquire();
            for (uint32_t j = 0; j < head_dim_t_; ++j) {
                mul_tiles_bcast_cols(cur_out_cb, scratch_cb, j, 0, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < head_dim_t_; ++j) {
                pack_tile(j, normalized_out_cb);
            }
            tile_regs_release();
            cb_push_back(normalized_out_cb, head_dim_t_);

            cb_pop_front(scratch_cb, 1);
            cb_pop_front(cur_out_cb, head_dim_t_);
        }
    }
}

// ==================================================================

template <
    bool PROFILING_ENABLED,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t Sv_chunk_t,
    uint32_t head_dim_t,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_v_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_exp_max_diff,
    uint32_t scale_fp32,
    uint32_t subblock_h,
    uint32_t cb_qkt_row_A,
    uint32_t cb_qkt_row_B,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
    uint32_t cb_normalized_out>
void sdpa_inner_loop_step(
    const uint32_t prev_max,
    const uint32_t cur_max,
    const uint32_t prev_sum,
    const uint32_t cur_sum,
    const uint32_t prev_out,
    const uint32_t cur_out,
    const bool is_last_iter) {
    constexpr uint32_t sbh = subblock_h;
    constexpr uint32_t in0_block_w = head_dim_t;
    constexpr uint32_t qkt_subblock_w = 8 / sbh;  // 8 when sbh=1, 4 when sbh=2
    constexpr uint32_t q_num_subblocks = Sq_chunk_t / sbh;
    constexpr uint32_t kt_num_subblocks = Sk_chunk_t / qkt_subblock_w;
    constexpr uint32_t q_subblock_num_tiles = sbh * in0_block_w;
    constexpr uint32_t row_tiles = sbh * Sk_chunk_t;

    static_assert(sbh * qkt_subblock_w <= 8, "sbh * qkt_subblock_w must fit in DST (max 8 tiles)");
    static_assert(Sk_chunk_t % qkt_subblock_w == 0, "Sk_chunk_t must be divisible by qkt_subblock_w");
    static_assert(Sq_chunk_t % sbh == 0, "Sq_chunk_t must be divisible by subblock_h");

    uint32_t pushed_rows = 0;
    uint32_t q_wait_tiles = q_subblock_num_tiles;
    uint32_t q_index_offset = 0;
    uint32_t kt_index_offset = 0;

    // Alternating row buffers for raw Q@KT matmul output
    uint32_t alias_cur_qkt_row = cb_qkt_row_A;
    uint32_t alias_prev_qkt_row = cb_qkt_row_B;

    exp_packthread_tile_init<true, true, scale_fp32, InputClamping::None>();

    pack_reconfig_data_format(cb_qkt_im);
    reconfig_data_format(cb_kt_in, cb_q_in);
    cb_reserve_back(cb_qkt_im, Sq_chunk_t * Sk_chunk_t);

    cb_wait_front(cb_kt_in, head_dim_t * Sk_chunk_t);
    cb_reserve_back(cur_sum, Sq_chunk_t);

    // ========== PHASE 1: Q@KT with alternating row buffers ==========
    for (uint32_t q_subblock = 0; q_subblock < q_num_subblocks; q_subblock++) {
        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "Softmax(Q@KT)");
        cb_wait_front(cb_q_in, q_wait_tiles);
        kt_index_offset = 0;

        // Reserve current row buffer for matmul output
        cb_reserve_back(alias_cur_qkt_row, row_tiles);

        for (uint32_t kt_subblock = 0; kt_subblock < kt_num_subblocks; ++kt_subblock) {
            if (q_subblock > 0) {
                uint32_t prev_q_subblock = q_subblock - 1;
                MATH(DPRINT << "SUB EXP for Q[" << prev_q_subblock << "] Kt[" << kt_subblock << "]" << ENDL());
                sub_exp_block_bcast_cols<PROFILING_ENABLED, scale_fp32, sbh, qkt_subblock_w, true>(
                    alias_prev_qkt_row, cur_max, cb_qkt_im, cur_sum, Sk_chunk_t, prev_q_subblock, kt_subblock);
            }

            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "Q@KT MM+Pack");
            blocked_matmul_and_pack<true, qkt_subblock_w, sbh, in0_block_w, Sk_chunk_t, Sk_chunk_t, true>(
                cb_q_in,
                cb_kt_in,
                alias_cur_qkt_row,
                q_index_offset,
                kt_index_offset,
                q_subblock,
                kt_subblock * qkt_subblock_w);
            kt_index_offset += qkt_subblock_w;
        }

        if (q_subblock > 0) {
            // Push softmax'd prev row to cb_qkt_im, free prev row buffer
            cb_push_back(cb_qkt_im, row_tiles);
            cb_pop_front(alias_prev_qkt_row, row_tiles);
        }

        // Push raw matmul row (makes it available for max reduce read)
        cb_push_back(alias_cur_qkt_row, row_tiles);

        // Max reduce: reads from current row buffer, writes to cur_max (sequential)
        MATH(DPRINT << "Max reduce for Q[" << q_subblock << ", :]" << ENDL());
        {
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "Reduce max");
            cb_reserve_back(cur_max, sbh);
            reduce_c_row_group<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_identity_scale_in, Sk_chunk_t, sbh>(
                alias_cur_qkt_row, cur_max, prev_max, q_subblock, true /*do_eltwise_max*/, 0 /*in0_row_group_index*/);
            cb_push_back(cur_max, sbh);
        }

        // Current row still has data (not popped) — becomes prev row in next iteration's sub_exp
        std::swap(alias_cur_qkt_row, alias_prev_qkt_row);
        q_index_offset += sbh * in0_block_w;
        q_wait_tiles += q_subblock_num_tiles;
    }

    cb_pop_front(cb_kt_in, head_dim_t * Sk_chunk_t);

    // ========== PHASE 2: Drain last row + QKT@V + SALAD ==========
    // After Phase 1: alias_prev_qkt_row holds the last raw matmul row (not yet softmax'd).
    // cb_qkt_im has (q_num_subblocks - 1) rows of softmax'd data pushed.
    MATH(DPRINT << "Starting QKT @ V computation" << ENDL());
    {
        constexpr uint32_t qktv_subblock_h = sbh;
        constexpr uint32_t qktv_subblock_w = 4;
        constexpr uint32_t qktv_in0_block_w = Sv_chunk_t;
        constexpr uint32_t qktv_q_num_subblocks = Sq_chunk_t / qktv_subblock_h;
        constexpr uint32_t qktv_v_num_subblocks = head_dim_t / qktv_subblock_w;
        constexpr uint32_t qktv_output_num_tiles = Sq_chunk_t * head_dim_t;
        constexpr uint32_t qktv_in0_subblock_num_tiles = qktv_subblock_h * qktv_in0_block_w;

        uint32_t qktv_in0_index_offset = 0;
        uint32_t qktv_in0_wait_tiles = qktv_in0_subblock_num_tiles;

        cb_wait_front(cb_v_in, Sv_chunk_t * head_dim_t);
        cb_reserve_back(cur_out, qktv_output_num_tiles);

        // ===== q_subblock 0: drain last row's sub_exp from alias_prev_qkt_row + first QKT@V matmul =====
        {
            MATH(DPRINT << "QKT@V: Processing Q_subblock 0 (drain)" << ENDL());
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "Softmax(Q@KT)@V");

#ifdef OVERLAP_DRAIN_WITH_MATMUL
            {
                constexpr uint32_t half_inner = qktv_in0_block_w / 2;
                static_assert(kt_num_subblocks == 2, "Overlap drain requires kt_num_subblocks==2");

                // 1. sub_exp drain kt=0: reads alias_prev_qkt_row → writes cb_qkt_im (sequential)
                MATH(DPRINT << "DRAIN OVERLAP: SUB_EXP kt=0" << ENDL());
                sub_exp_block_bcast_cols<PROFILING_ENABLED, scale_fp32, sbh, qkt_subblock_w, true>(
                    alias_prev_qkt_row, cur_max, cb_qkt_im, cur_sum, Sk_chunk_t, q_num_subblocks - 1, 0);

                // 2. matmul first half — FPU overlaps with EXP(kt=0) on SFPU
                // Wait for softmax'd tiles in cb_qkt_im (first row at position 0)
                cb_wait_front(cb_qkt_im, qktv_in0_wait_tiles);
                {
                    uint32_t v_index_offset = 0;
                    for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "QKT@V MM+Pack H1");
                        blocked_matmul_and_pack<
                            false,
                            qktv_subblock_w,
                            qktv_subblock_h,
                            half_inner,
                            head_dim_t,
                            head_dim_t>(
                            cb_qkt_im,
                            cb_v_in,
                            cur_out,
                            qktv_in0_index_offset,
                            v_index_offset,
                            0,
                            v_subblock * qktv_subblock_w);
                        v_index_offset += qktv_subblock_w;
                    }
                }

                // 3. sub_exp drain kt=1
                MATH(DPRINT << "DRAIN OVERLAP: SUB_EXP kt=1" << ENDL());
                sub_exp_block_bcast_cols<PROFILING_ENABLED, scale_fp32, sbh, qkt_subblock_w, true>(
                    alias_prev_qkt_row, cur_max, cb_qkt_im, cur_sum, Sk_chunk_t, q_num_subblocks - 1, 1);

                // Push last softmax'd row, free prev row buffer
                cb_push_back(cb_qkt_im, row_tiles);
                cb_pop_front(alias_prev_qkt_row, row_tiles);
                // NOTE: cur_sum stays in RESERVED state (not pushed).
                // SALAD will L1-accumulate corrected prev_sum onto it, then push at end.

                // 4. matmul second half with L1 accumulate
                PACK((llk_pack_reconfig_l1_acc(1)));
                {
                    uint32_t v_index_offset = 0;
                    for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                        MaybeDeviceZoneScopedN(PROFILING_ENABLED, "QKT@V MM+Pack H2");
                        blocked_matmul_and_pack<
                            false,
                            qktv_subblock_w,
                            qktv_subblock_h,
                            half_inner,
                            head_dim_t,
                            head_dim_t>(
                            cb_qkt_im,
                            cb_v_in,
                            cur_out,
                            qktv_in0_index_offset + half_inner,
                            half_inner * head_dim_t + v_index_offset,
                            0,
                            v_subblock * qktv_subblock_w);
                        v_index_offset += qktv_subblock_w;
                    }
                }
                PACK((llk_pack_reconfig_l1_acc(0)));
            }
#else
            {
                // Non-overlap path: drain all sub_exp, then full matmul
                MATH(DPRINT << "DRAIN: SUB_EXP for Q[" << q_num_subblocks - 1 << "]" << ENDL());
                for (uint32_t kt_subblock = 0; kt_subblock < kt_num_subblocks; ++kt_subblock) {
                    sub_exp_block_bcast_cols<PROFILING_ENABLED, scale_fp32, sbh, qkt_subblock_w, true>(
                        alias_prev_qkt_row, cur_max, cb_qkt_im, cur_sum, Sk_chunk_t, q_num_subblocks - 1, kt_subblock);
                }
                cb_push_back(cb_qkt_im, row_tiles);
                cb_pop_front(alias_prev_qkt_row, row_tiles);
                // cur_sum stays RESERVED — pushed after SALAD corrections

                cb_wait_front(cb_qkt_im, qktv_in0_wait_tiles);
                uint32_t v_index_offset = 0;
                for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                    MaybeDeviceZoneScopedN(PROFILING_ENABLED, "QKT@V MM+Pack");
                    blocked_matmul_and_pack<
                        false,
                        qktv_subblock_w,
                        qktv_subblock_h,
                        qktv_in0_block_w,
                        head_dim_t,
                        head_dim_t>(
                        cb_qkt_im,
                        cb_v_in,
                        cur_out,
                        qktv_in0_index_offset,
                        v_index_offset,
                        0,
                        v_subblock * qktv_subblock_w);
                    v_index_offset += qktv_subblock_w;
                }
            }
#endif
            qktv_in0_index_offset += qktv_subblock_h * qktv_in0_block_w;
            qktv_in0_wait_tiles += qktv_in0_subblock_num_tiles;
        }

        // SALAD correction for one completed row: sum/out L1-accumulate, then
        // optionally push + normalize on the last K iteration.
        auto salad_correct_row = [&](uint32_t salad_row, uint32_t w_salad, bool last_iter, uint32_t& pushed) {
            {
                MATH(DPRINT << "SALAD sum correction for Q[" << salad_row << "]" << ENDL());
                MaybeDeviceZoneScopedN(PROFILING_ENABLED, "S_SUM_CORR");
                mul_bcast_cols_l1_acc<sbh>(prev_sum, cb_exp_max_diff, cur_sum, salad_row, w_salad);
            }
            {
                MATH(DPRINT << "SALAD out correction for Q[" << salad_row << "]" << ENDL());
                MaybeDeviceZoneScopedN(PROFILING_ENABLED, "S_OUT_CORR");
                mul_block_bcast_cols_acc<sbh, head_dim_t>(prev_out, cb_exp_max_diff, cur_out, salad_row, w_salad);
            }
            if (last_iter) {
                MATH(DPRINT << "Row normalization for Q[" << salad_row << "]" << ENDL());
                MaybeDeviceZoneScopedN(PROFILING_ENABLED, "ROW_NORM");
                cb_push_back(cur_sum, sbh);
                cb_push_back(cur_out, sbh * head_dim_t);
                normalize_row_streaming<PROFILING_ENABLED, sbh, head_dim_t>(
                    cur_sum, cur_out, cb_col_identity, cb_recip_scratch, cb_normalized_out);
                pushed++;
            }
        };

        // ===== q_subblock 1..N-1: SALAD(prev) overlapped with matmul(cur) =====
        for (uint32_t q_subblock = 1; q_subblock < qktv_q_num_subblocks; ++q_subblock) {
            uint32_t salad_row = q_subblock - 1;
            // Adjusted write offsets: after pushing rows, wr_ptr advances,
            // so pack_tile<true> offsets must be relative to the new wr_ptr.
            // On non-last iterations pushed_rows=0, so these equal the originals.
            uint32_t w_salad = salad_row - pushed_rows;
            uint32_t w_q = q_subblock - pushed_rows;

            MATH(DPRINT << "QKT@V: Processing Q_subblock " << q_subblock << ", SALAD for row " << salad_row << ENDL());
            MaybeDeviceZoneScopedN(PROFILING_ENABLED, "Softmax(Q@KT)@V");
            cb_wait_front(cb_qkt_im, qktv_in0_wait_tiles);

            // 1. Compute exp_max_diff for PREVIOUS row (sequential output)
            {
                MATH(DPRINT << "SUB_EXP_m for Q[" << salad_row << "]" << ENDL());
                cb_reserve_back(cb_exp_max_diff, sbh);
                sub_exp_first_col_blocks<PROFILING_ENABLED, scale_fp32, sbh>(
                    prev_max, cur_max, cb_exp_max_diff, salad_row);
                cb_push_back(cb_exp_max_diff, sbh);
            }

            // 2. Full matmul for CURRENT row — FPU overlaps with SFPU EXP above
            // Uses w_q for the output row offset (adjusted for pushed rows)
            {
                uint32_t v_index_offset = 0;
                for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                    MaybeDeviceZoneScopedN(PROFILING_ENABLED, "QKT@V MM+Pack");
                    blocked_matmul_and_pack<
                        false,
                        qktv_subblock_w,
                        qktv_subblock_h,
                        qktv_in0_block_w,
                        head_dim_t,
                        head_dim_t>(
                        cb_qkt_im,
                        cb_v_in,
                        cur_out,
                        qktv_in0_index_offset,
                        v_index_offset,
                        w_q,
                        v_subblock * qktv_subblock_w);
                    v_index_offset += qktv_subblock_w;
                }
            }

            // 3. SALAD corrections + optional per-row normalization for PREVIOUS row
            salad_correct_row(salad_row, w_salad, is_last_iter, pushed_rows);

            qktv_in0_index_offset += qktv_subblock_h * qktv_in0_block_w;
            qktv_in0_wait_tiles += qktv_in0_subblock_num_tiles;
        }

        // ===== Pipeline drain: SALAD for last row =====
        {
            constexpr uint32_t salad_row = qktv_q_num_subblocks - 1;
            uint32_t w_salad = salad_row - pushed_rows;
            MATH(DPRINT << "Pipeline drain: SALAD for row " << salad_row << ENDL());

            cb_reserve_back(cb_exp_max_diff, sbh);
            sub_exp_first_col_blocks<PROFILING_ENABLED, scale_fp32, sbh>(prev_max, cur_max, cb_exp_max_diff, salad_row);
            cb_push_back(cb_exp_max_diff, sbh);

            salad_correct_row(salad_row, w_salad, is_last_iter, pushed_rows);
        }

        // Bulk push — skip on last iteration (all rows already consumed by per-row normalization)
        if (!is_last_iter) {
            cb_push_back(cur_sum, Sq_chunk_t);
            cb_push_back(cur_out, qktv_output_num_tiles);
        }

        cb_pop_front(cb_v_in, Sv_chunk_t * head_dim_t);
        cb_pop_front(cb_qkt_im, Sq_chunk_t * Sk_chunk_t);
    }
}

void kernel_main() {
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t Sv_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t head_dim_t = get_compile_time_arg_val(3);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t num_k_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(6);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(7);

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_kt_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_qkt_im = tt::CBIndex::c_2;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_qkt_row_A = tt::CBIndex::c_4;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_qkt_row_B = tt::CBIndex::c_6;

    constexpr uint32_t cb_out_A = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_B = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_A = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_B = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_A = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_B = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;

    constexpr uint32_t cb_neginf = tt::CBIndex::c_7;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_8;
    constexpr uint32_t cb_normalized_out = tt::CBIndex::c_9;
    constexpr uint32_t cb_recip_scratch = tt::CBIndex::c_10;

    mm_init(cb_q_in, cb_kt_in, cb_qkt_im);

    // One-time debug print of derived constexpr values
    constexpr uint32_t sbh = subblock_h;
    constexpr uint32_t in0_block_w = head_dim_t;
    constexpr uint32_t qkt_subblock_w = 8 / sbh;
    constexpr uint32_t q_num_subblocks = Sq_chunk_t / sbh;
    constexpr uint32_t kt_num_subblocks = Sk_chunk_t / qkt_subblock_w;
    MATH(
        DPRINT << "sbh=" << sbh << " qkt_subblock_w=" << qkt_subblock_w << " in0_block_w=" << in0_block_w
               << " q_num_subblocks=" << q_num_subblocks << " kt_num_subblocks=" << kt_num_subblocks << ENDL());

    for (uint32_t q = 0; q < num_q_chunks; q++) {
        // Initialize prev buffers (sum=0, out=0, max=-inf) for this Q chunk.
        // Always writes to A-variant CBs since the previous Q chunk's per-row normalization
        // consumed the A-variants (or they were never written for q=0).
        {
            DeviceZoneScopedN("DUMMY:InitPrevBuffers");
            InitPrevBuffers<Sq_chunk_t, head_dim_t>(cb_sum_A, cb_out_A, cb_max_A, cb_neginf);
        }

        // K-chunk loop with ping-pong buffers.
        // sdpa_inner_loop_step handles per-row normalization on the last K iteration,
        // streaming normalized tiles to cb_normalized_out.
        // To profile specific iterations, change the template bool:
        //   if (k_chunk == N) sdpa_inner_loop_step<true, ...>(...);
        //   else              sdpa_inner_loop_step<false, ...>(...);
        uint32_t alias_prev_sum = cb_sum_A, alias_cur_sum = cb_sum_B;
        uint32_t alias_prev_max = cb_max_A, alias_cur_max = cb_max_B;
        uint32_t alias_prev_out = cb_out_A, alias_cur_out = cb_out_B;

        for (uint32_t k_chunk = 0; k_chunk < num_k_chunks; k_chunk++) {
            DeviceZoneScopedN("sdpa_inner_loop_step");
            bool is_last = (k_chunk == num_k_chunks - 1);

            auto call_step = [&](auto profiling_tag) {
                sdpa_inner_loop_step<
                    decltype(profiling_tag)::value,
                    Sq_chunk_t,
                    Sk_chunk_t,
                    Sv_chunk_t,
                    head_dim_t,
                    cb_q_in,
                    cb_kt_in,
                    cb_v_in,
                    cb_qkt_im,
                    cb_identity_scale_in,
                    cb_exp_max_diff,
                    scale_fp32,
                    subblock_h,
                    cb_qkt_row_A,
                    cb_qkt_row_B,
                    cb_col_identity,
                    cb_recip_scratch,
                    cb_normalized_out>(
                    alias_prev_max,
                    alias_cur_max,
                    alias_prev_sum,
                    alias_cur_sum,
                    alias_prev_out,
                    alias_cur_out,
                    is_last);
            };

            // Uncomment this (or follow the pattern above) to profile specific iterations of the inner loop.
            // if (is_last)
            //     call_step(std::true_type{});
            // else
            call_step(std::false_type{});

            // Post-iteration cleanup: pop prev buffers
            cb_pop_front(cb_exp_max_diff, Sq_chunk_t);
            cb_pop_front(alias_prev_max, Sq_chunk_t);
            cb_pop_front(alias_prev_sum, Sq_chunk_t);
            cb_pop_front(alias_prev_out, Sq_chunk_t * head_dim_t);

            if (is_last) {
                // cur_sum and cur_out already consumed by per-row normalization.
                // Pop cur_max — no longer needed.
                cb_pop_front(alias_cur_max, Sq_chunk_t);
            } else {
                std::swap(alias_prev_max, alias_cur_max);
                std::swap(alias_prev_sum, alias_cur_sum);
                std::swap(alias_prev_out, alias_cur_out);
            }
        }

        // Pop Q after all K chunks are processed for this Q chunk
        cb_pop_front(cb_q_in, Sq_chunk_t * head_dim_t);
    }

    // Pop the permanent -inf tile now that all Q chunks are done
    cb_pop_front(cb_neginf, 1);
}
