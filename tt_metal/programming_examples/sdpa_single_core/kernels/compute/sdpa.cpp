// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

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

// High-granularity profiling marker sets for sdpa_inner_loop.
// Set 1: Q@KT phase (matmul, sub_exp, pack, max reduce)
// Set 2: QKT@V + SALAD phase (matmul, pack, rescale steps)
// Enable sets independently; Tracy has a ~250 marker limit per run.

// #define SDPA_PROFILING_SET_1
// #define SDPA_PROFILING_SET_2

#ifdef SDPA_PROFILING_SET_1
#define SDPA_DeviceZoneScopedN_1(name) DeviceZoneScopedN(name)
#else
#define SDPA_DeviceZoneScopedN_1(name)
#endif

#ifdef SDPA_PROFILING_SET_2
#define SDPA_DeviceZoneScopedN_2(name) DeviceZoneScopedN(name)
#else
#define SDPA_DeviceZoneScopedN_2(name)
#endif

/**
 * out_cb = exp((in0_cb - in1_cb) * scale_fp32)
 * only at 2*q_subblock and 2*q_subblock+1 elements
 */
template <uint32_t scale_fp32, uint32_t SBH>
void sub_exp_first_col_blocks(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t q_subblock) {
    constexpr uint32_t tiles_per_row = SBH;
    const uint32_t global_row_base = q_subblock * tiles_per_row;
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    sub_tiles_init(in0_cb, in1_cb);
    exp_packthread_tile_init<EXP_APPROX_MODE, false>();

    cb_wait_front(in0_cb, (q_subblock + 1) * tiles_per_row);
    cb_wait_front(in1_cb, (q_subblock + 1) * tiles_per_row);

    {
        tile_regs_acquire();
        uint32_t dst_index = 0;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            uint32_t tile_index = global_row_base + i;
            sub_tiles(in0_cb, in1_cb, tile_index, tile_index, i /*dst_index*/);
        }
        tile_regs_commit();
    }

    {
        tile_regs_wait();
        for (uint32_t dst_index = 0; dst_index < tiles_per_row; dst_index++) {
            PACK((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(dst_index)));
        }
        PACK(TTI_STALLWAIT(p_stall ::STALL_PACK, p_stall ::WAIT_SFPU));
    }

    cb_reserve_back(out_cb, tiles_per_row);
    {
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            uint32_t tile_index = global_row_base + i;
            pack_tile<true>(i /*dst_index*/, out_cb, tile_index);
        }
    }
    cb_push_back(out_cb, tiles_per_row);

    tile_regs_release();
}

/**
 * in0_cb += in1_cb
 */
template <uint32_t SBH>
void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t q_subblock) {
    constexpr uint32_t tiles_per_row = SBH;
    const uint32_t global_row_base = q_subblock * tiles_per_row;

    add_tiles_init(in0_cb, in1_cb);
    cb_wait_front(in0_cb, (q_subblock + 1) * tiles_per_row);
    cb_wait_front(in1_cb, (q_subblock + 1) * tiles_per_row);

    tile_regs_acquire();
    for (uint32_t i = 0; i < tiles_per_row; i++) {
        uint32_t src_tile_index = global_row_base + i;
        add_tiles(in0_cb, in1_cb, src_tile_index, src_tile_index, i /*dst_index*/);
    }
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t i = 0; i < tiles_per_row; i++) {
        pack_tile<true>(i, in0_cb, global_row_base + i);  // Pack back to original position in in0_cb
    }
    tile_regs_release();
}

template <uint32_t SBH>
void mul_tiles_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t q_subblock) {
    /**
     * Given in0_cb and in1_cb, multiply each tile of in0_cb by the corresponding tile of in1_cb
     * and bcast cols of in1_cb.
     */
    constexpr uint32_t tiles_per_row = SBH;
    const uint32_t global_row_base = q_subblock * tiles_per_row;
    mul_bcast_cols_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, (q_subblock + 1) * tiles_per_row);
    cb_wait_front(in1_cb, (q_subblock + 1) * tiles_per_row);

    tile_regs_acquire();
    for (uint32_t i = 0; i < tiles_per_row; i++) {
        uint32_t src_tile_index = global_row_base + i;
        mul_tiles_bcast_cols(in0_cb, in1_cb, src_tile_index, src_tile_index, i /*dst_index*/);
    }
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t i = 0; i < tiles_per_row; i++) {
        pack_tile<true>(i, in0_cb, global_row_base + i);  // Pack back to original position in in0_cb
    }
    tile_regs_release();
}

template <uint32_t SBH, uint32_t SBW>
void mul_block_bcast_cols_acc(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t q_subblock) {
    constexpr uint32_t tiles_per_row = SBH;
    constexpr uint32_t tiles_per_column = SBW;
    static_assert(tiles_per_row * tiles_per_column <= 8, "SBH * SBW must fit in DST (max 8 tiles)");
    const uint32_t global_row_base = q_subblock * tiles_per_row;
    mul_bcast_cols_init_short(in0_cb, in1_cb);

    cb_wait_front(in0_cb, (q_subblock + 1) * tiles_per_row * tiles_per_column);
    cb_wait_front(in1_cb, (q_subblock + 1) * tiles_per_row);

    PACK((llk_pack_reconfig_l1_acc(1 /*pack accumulate*/)));
    tile_regs_acquire();
    uint32_t dst_index = 0;
    for (uint32_t i = 0; i < tiles_per_row; i++) {
        for (uint32_t j = 0; j < tiles_per_column; j++) {
            uint32_t in0_tile_index = (global_row_base + i) * tiles_per_column + j;  // Absolute tile index for in0_cb
            mul_tiles_bcast_cols(in0_cb, in1_cb, in0_tile_index, global_row_base + i, dst_index++);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    dst_index = 0;
    for (uint32_t i = 0; i < tiles_per_row; i++) {
        for (uint32_t j = 0; j < tiles_per_column; j++) {
            uint32_t out_tile_index = (global_row_base + i) * tiles_per_column + j;  // Absolute tile index for in0_cb
            pack_tile<true>(dst_index++, out_cb, out_tile_index);  // Pack to original position in out_cb
        }
    }
    tile_regs_release();
    PACK((llk_pack_reconfig_l1_acc(false)));
}

template <
    uint32_t in0_cb,
    uint32_t scale_fp32,
    uint32_t SBH,
    uint32_t SBW,
    bool do_reduce = true,
    int vector_mode = (int)VectorMode::RC>
void sub_exp_block_bcast_cols_inplace(
    uint32_t in1_cb, uint32_t reduce_cb, uint32_t cols, uint32_t q_subblock, uint32_t kt_subblock) {
    constexpr uint32_t tiles_per_row = SBH;
    constexpr uint32_t tiles_per_column = SBW;
    static_assert(tiles_per_row * tiles_per_column <= 8, "SBH * SBW must fit in DST (max 8 tiles)");
    const uint32_t global_row_base = q_subblock * tiles_per_row;
    const uint32_t global_col_base = kt_subblock * tiles_per_column;

    // Initialize operation
    sub_bcast_cols_init_short(in0_cb, in1_cb);
    // exp_packthread_tile_init<true, true, scale_fp32>();  // todo: move outside.

    // Wait for tiles:
    // - in0_cb: cumulative wait since we never pop it
    // - in1_cb: cumulative wait since we never pop it
    cb_wait_front(in0_cb, (q_subblock + 1) * tiles_per_row * cols);
    cb_wait_front(in1_cb, (q_subblock + 1) * tiles_per_row);

    // if constexpr (do_reduce) {
    //     cb_reserve_back(reduce_cb, tiles_per_row);
    // }

    {
        SDPA_DeviceZoneScopedN_1("SUB");
        tile_regs_acquire();
        uint32_t dst_index = 0;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            for (uint32_t j = 0; j < tiles_per_column; j++) {
                uint32_t in0_tile_index =
                    (global_row_base + i) * cols + (global_col_base + j);  // Absolute tile index for in0_cb
                sub_tiles_bcast_cols(in0_cb, in1_cb, in0_tile_index, global_row_base + i, dst_index++);
            }
        }
        tile_regs_commit();
    }

    {
        SDPA_DeviceZoneScopedN_1("EXP");
        tile_regs_wait();
        uint32_t dst_index = 0;
        // Use fast exp with InputClamping::None and 32 iterations for 1.3x speedup
        // When vector_mode is RC, use 32 iterations with VectorMode::None
        // Otherwise, use 8 iterations with the original vector_mode
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
        SDPA_DeviceZoneScopedN_1("PACK SUB_EXP");

        tile_regs_wait();
        uint32_t dst_index = 0;
        for (uint32_t i = 0; i < tiles_per_row; i++) {
            for (uint32_t j = 0; j < tiles_per_column; ++j) {
                uint32_t in0_tile_index =
                    (global_row_base + i) * cols + (global_col_base + j);  // Absolute tile index for in0_cb
                pack_tile<true>(dst_index++, in0_cb, in0_tile_index);      // Pack back to original position in in0_cb
            }
        }

        if constexpr (do_reduce) {
            dst_index = 0;
            for (uint32_t i = 0; i < tiles_per_row; i++) {
                // While we have results in DST, take advantage of L1 accumulation
                // to reduce row x cols tiles to rows x 1 tiles.
                if (global_col_base > 0) {
                    // If on the same row, keep accumulating
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
                for (uint32_t j = 0; j < tiles_per_column; ++j) {
                    // Pack to local_row's position in reduce_cb (0 or 1 within this pair)
                    pack_tile<true>(dst_index++, reduce_cb, global_row_base + i);
                    if (global_col_base == 0 && j == 0) {
                        // If this was the first tile of a row, start accumulating
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                }
            }
        }
        tile_regs_release();
        if constexpr (do_reduce) {  // todo: move up?
            PACK((llk_pack_reconfig_l1_acc(0)));
        }
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

template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_cb,
    uint32_t scale_cb,
    uint32_t cols,
    uint32_t SBH,
    int vector_mode = static_cast<int>(VectorMode::C)>
void reduce_c_row_group(uint32_t out_cb, uint32_t prev_cb, uint32_t row_group_index, bool do_eltwise_max = false) {
    // Precondition: in0_cb has at least (row_group_index + 1) * SBH * cols tiles produced (row-major order)
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free (reserved by caller)
    // Precondition: prev_cb has at least (row_group_index + 1) * SBH tiles produced (if do_eltwise_max)
    // Postcondition: in0_cb unchanged (no pop)
    // Postcondition: out_cb has SBH more tiles written at positions [row_group_index*SBH, ...]

    constexpr uint32_t GROUP_SIZE = SBH;
    const uint32_t row_start = row_group_index * GROUP_SIZE;

    // Cumulative tile counts for cb_wait_front
    const uint32_t cumulative_input_tiles = (row_group_index + 1) * GROUP_SIZE * cols;
    const uint32_t cumulative_prev_tiles = (row_group_index + 1) * GROUP_SIZE;

    // Wait for scale (always needed, returns immediately if already available)
    cb_wait_front(scale_cb, 1);

    // Wait for input tiles up to and including this row group
    cb_wait_front(in0_cb, cumulative_input_tiles);

    tile_regs_acquire();

    if (do_eltwise_max) {
        cb_wait_front(prev_cb, cumulative_prev_tiles);
        /**
         * Copy previous max values into DST register.
         * Note that this special invocation of copy_tile is necessary to produce
         * tiles in DST with transposed faces, as `reduce_block_max_row` expects.
         */
        sdpa_reduce_copy_tile_to_dst_init_short(prev_cb);
        for (uint32_t i = 0; i < GROUP_SIZE; i++) {
            copy_tile(prev_cb, row_start + i, i);
        }
    }

    /**
     * For the GROUP_SIZE rows in this group, compute the max into DST registers.
     */
    reduce_block_max_row_init<cols>();
    for (uint32_t i = 0; i < GROUP_SIZE; i++) {
        const uint32_t input_tile_start = (row_start + i) * cols;
        const uint32_t reduce_dst_idx = i;
        reduce_block_max_row<cols>(in0_cb, scale_cb, input_tile_start, reduce_dst_idx);
    }
    reduce_block_max_row_uninit(in0_cb);

    tile_regs_commit();
    tile_regs_wait();

    // Pack results to output at the correct positions
    cb_reserve_back(out_cb, GROUP_SIZE);
    for (uint32_t i = 0; i < GROUP_SIZE; i++) {
        const uint32_t dst_idx = i;
        pack_tile<true>(dst_idx, out_cb, row_start + i);
    }
    cb_push_back(out_cb, GROUP_SIZE);

    tile_regs_release();
}

template <
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
    uint32_t subblock_h>
void sdpa_inner_loop(
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B,
    const uint32_t cb_out_A,
    const uint32_t cb_out_B,
    const uint32_t num_iter) {
    // Set up ping pong buffers
    // To be used (and swapped) later on, when we loop over Q chunks.
    uint32_t alias_prev_sum = cb_sum_A;
    uint32_t alias_cur_sum = cb_sum_B;
    uint32_t alias_prev_max = cb_max_A;
    uint32_t alias_cur_max = cb_max_B;
    uint32_t alias_prev_out = cb_out_A;
    uint32_t alias_cur_out = cb_out_B;

    constexpr uint32_t sbh = subblock_h;
    constexpr uint32_t in0_block_w = head_dim_t;
    constexpr uint32_t qkt_subblock_w = 8 / sbh;  // 8 when sbh=1, 4 when sbh=2
    constexpr uint32_t q_num_subblocks = Sq_chunk_t / sbh;
    constexpr uint32_t kt_num_subblocks = Sk_chunk_t / qkt_subblock_w;
    constexpr uint32_t q_subblock_num_tiles = sbh * in0_block_w;

    static_assert(sbh * qkt_subblock_w <= 8, "sbh * qkt_subblock_w must fit in DST (max 8 tiles)");
    static_assert(Sk_chunk_t % qkt_subblock_w == 0, "Sk_chunk_t must be divisible by qkt_subblock_w");
    static_assert(Sq_chunk_t % sbh == 0, "Sq_chunk_t must be divisible by subblock_h");

    MATH(
        DPRINT << "sbh=" << sbh << " qkt_subblock_w=" << qkt_subblock_w << " in0_block_w=" << in0_block_w
               << " q_num_subblocks=" << q_num_subblocks << " kt_num_subblocks=" << kt_num_subblocks << ENDL());

    for (uint32_t iter = 0; iter < num_iter; iter++) {
        // Reset per-iteration state
        MATH(DPRINT << "******************ITERATION " << iter << " ******************" << ENDL());
        DeviceZoneScopedN("sdpa_inner_loop");
        uint32_t q_wait_tiles = q_subblock_num_tiles;
        uint32_t q_index_offset = 0;
        uint32_t kt_index_offset = 0;

        // Initialize fast approximate exp with no input clamping for 1.3x speedup
        exp_packthread_tile_init<true, true, scale_fp32, InputClamping::None>();
        // Configure packer ReLU to clamp negative artifacts from approximate exp
        PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
        pack_reconfig_data_format(cb_qkt_im);
        reconfig_data_format(cb_kt_in, cb_q_in);
        cb_reserve_back(cb_qkt_im, Sq_chunk_t * Sk_chunk_t);

        cb_wait_front(cb_kt_in, head_dim_t * Sk_chunk_t);
        cb_reserve_back(alias_cur_sum, Sq_chunk_t);
        for (uint32_t q_subblock = 0; q_subblock < q_num_subblocks; q_subblock++) {
            DeviceZoneScopedN("Softmax(Q@KT)");
            cb_wait_front(cb_q_in, q_wait_tiles);
            kt_index_offset = 0;

            for (uint32_t kt_subblock = 0; kt_subblock < kt_num_subblocks; ++kt_subblock) {
                if (q_subblock > 0) {
                    uint32_t prev_q_subblock = q_subblock - 1;
                    MATH(DPRINT << "SUB EXP for Q[" << prev_q_subblock << "] Kt[" << kt_subblock << "]" << ENDL());
                    sub_exp_block_bcast_cols_inplace<cb_qkt_im, scale_fp32, sbh, qkt_subblock_w, true>(
                        alias_cur_max, alias_cur_sum, Sk_chunk_t, prev_q_subblock, kt_subblock);
                }

                {
                    {
                        mm_block_init_short(
                            cb_q_in,
                            cb_kt_in,
                            true /*transpose*/,
                            qkt_subblock_w /*ct_dim*/,
                            sbh /*rt_dim*/,
                            in0_block_w /*kt_dim*/);
                    }
                    MATH(DPRINT << "Matmul for Q[" << q_subblock << "] Kt[" << kt_subblock << "]" << ENDL());

                    {
                        SDPA_DeviceZoneScopedN_1("matmul_blocks");

                        tile_regs_acquire();
                        uint32_t dst_index = 0;
                        uint32_t q_index = q_index_offset;
                        uint32_t kt_index = kt_index_offset;
                        for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                            matmul_block(
                                cb_q_in,
                                cb_kt_in,
                                q_index,
                                kt_index,
                                dst_index,
                                true /*transpose*/,
                                qkt_subblock_w,
                                sbh,
                                in0_block_w);
                            q_index++;
                            kt_index += Sk_chunk_t;
                        }
                        // tensix_sync();
                        tile_regs_commit();
                    }
                }
                {
                    SDPA_DeviceZoneScopedN_1("Pack MM");
                    // Pack the subblock
                    tile_regs_wait();
                    PACK(DPRINT << "Pack for Q[" << q_subblock << "] Kt[" << kt_subblock << "]" << ENDL());
                    uint32_t dst_idx = 0;
                    uint32_t out_col_offset = kt_subblock * qkt_subblock_w;
                    for (uint32_t r = 0; r < sbh; r++) {
                        uint32_t out_row_offset = (r + q_subblock * sbh) * Sk_chunk_t;
                        for (uint32_t c = 0; c < qkt_subblock_w; c++) {
                            pack_tile<true>(dst_idx, cb_qkt_im, out_row_offset + out_col_offset + c);
                            dst_idx++;
                        }
                    }
                    tile_regs_release();
                    MATH(
                        DPRINT << "Packing " << sbh * qkt_subblock_w << " tiles to cb_qkt_im for Q[" << q_subblock
                               << "] Kt[" << kt_subblock << "]" << ENDL());
                }
                kt_index_offset += qkt_subblock_w;
            }
            cb_push_back(cb_qkt_im, sbh * Sk_chunk_t);

            // Max reduce
            MATH(DPRINT << "Max reduce for Q[" << q_subblock << ", :]" << ENDL());
            {
                SDPA_DeviceZoneScopedN_1("Reduce max");
                reduce_c_row_group<
                    PoolType::MAX,
                    ReduceDim::REDUCE_ROW,
                    cb_qkt_im,
                    cb_identity_scale_in,
                    Sk_chunk_t,
                    sbh>(alias_cur_max, alias_prev_max, q_subblock, true /*do_eltwise_max*/);
            }

            q_index_offset += sbh * in0_block_w;
            q_wait_tiles += q_subblock_num_tiles;
        }

        cb_pop_front(cb_q_in, head_dim_t * Sq_chunk_t);
        cb_pop_front(cb_kt_in, head_dim_t * Sk_chunk_t);

        // QKT @ V: compute attention output
        // in0 = cb_qkt_im: Sq_chunk_t × Sk_chunk_t (M × K) — already produced
        // in1 = cb_v_in:   Sv_chunk_t × head_dim_t  (K × N)
        // out = cb_out:     Sq_chunk_t × head_dim_t  (M × N)
        // Drain sub_exp for the last Q@KT row is interleaved with the first QKT@V q_subblock.
        // sub_exp uses SFPU for exp, matmul uses FPU — they overlap on different hardware units.
        MATH(DPRINT << "Starting QKT @ V computation" << ENDL());
        {
            constexpr uint32_t qktv_subblock_h = sbh;
            constexpr uint32_t qktv_subblock_w = 4;
            constexpr uint32_t qktv_in0_block_w = Sv_chunk_t;
            constexpr uint32_t qktv_q_num_subblocks = Sq_chunk_t / qktv_subblock_h;
            constexpr uint32_t qktv_v_num_subblocks = head_dim_t / qktv_subblock_w;
            constexpr uint32_t qktv_output_num_tiles = Sq_chunk_t * head_dim_t;
            constexpr uint32_t qktv_in0_subblock_num_tiles = qktv_subblock_h * qktv_in0_block_w;

            MATH(
                DPRINT << "qktv_in0_block_w=" << qktv_in0_block_w << " qktv_q_num_subblocks=" << qktv_q_num_subblocks
                       << " qktv_v_num_subblocks=" << qktv_v_num_subblocks
                       << " qktv_output_num_tiles=" << qktv_output_num_tiles
                       << " qktv_in0_subblock_num_tiles=" << qktv_in0_subblock_num_tiles << ENDL());

            uint32_t qktv_in0_index_offset = 0;
            uint32_t qktv_in0_wait_tiles = qktv_in0_subblock_num_tiles;

            MATH(DPRINT << "Waiting for cb_v_in: " << Sv_chunk_t * head_dim_t << " tiles" << ENDL());
            cb_wait_front(cb_v_in, Sv_chunk_t * head_dim_t);
            MATH(DPRINT << "Reserving alias_cur_out: " << qktv_output_num_tiles << " tiles" << ENDL());
            cb_reserve_back(alias_cur_out, qktv_output_num_tiles);

            for (uint32_t q_subblock = 0; q_subblock < qktv_q_num_subblocks; ++q_subblock) {
                MATH(DPRINT << "QKT@V: Processing Q_subblock " << q_subblock << ENDL());
                DeviceZoneScopedN("Softmax(Q@KT)@V");
                cb_wait_front(cb_qkt_im, qktv_in0_wait_tiles);

                // Drain: interleave sub_exp for last Q@KT row with first QKT@V matmul
                if (q_subblock == 0) {
                    MATH(DPRINT << "DRAIN: SUB_EXP for Q[" << q_num_subblocks - 1 << ENDL());
                    for (uint32_t kt_subblock = 0; kt_subblock < kt_num_subblocks; ++kt_subblock) {
                        sub_exp_block_bcast_cols_inplace<cb_qkt_im, scale_fp32, sbh, qkt_subblock_w, true>(
                            alias_cur_max, alias_cur_sum, Sk_chunk_t, q_num_subblocks - 1, kt_subblock);
                    }
                    cb_push_back(alias_cur_sum, Sq_chunk_t);
                }

                {
                    mm_block_init_short(
                        cb_qkt_im,
                        cb_v_in,
                        false /*transpose*/,
                        qktv_subblock_w /*ct_dim*/,
                        qktv_subblock_h /*rt_dim*/,
                        qktv_in0_block_w /*kt_dim*/);
                }

                uint32_t v_index_offset = 0;
                for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                    MATH(DPRINT << "QKT@V Matmul for Q[" << q_subblock << "] V[" << v_subblock << "]" << ENDL());
                    {
                        SDPA_DeviceZoneScopedN_2("QKT@V matmul");
                        tile_regs_acquire();

                        uint32_t dst_index = 0;
                        uint32_t in0_index = qktv_in0_index_offset;
                        uint32_t in1_index = v_index_offset;

                        for (uint32_t inner = 0; inner < qktv_in0_block_w; ++inner) {
                            matmul_block(
                                cb_qkt_im,
                                cb_v_in,
                                in0_index,
                                in1_index,
                                dst_index,
                                false /*transpose*/,
                                qktv_subblock_w,
                                qktv_subblock_h,
                                qktv_in0_block_w);
                            in0_index++;
                            in1_index += head_dim_t;
                        }
                        tile_regs_commit();
                    }

                    {
                        SDPA_DeviceZoneScopedN_2("QKT@V pack");
                        tile_regs_wait();
                        PACK(DPRINT << "QKT@V Pack for Q[" << q_subblock << "] V[" << v_subblock << "]" << ENDL());
                        uint32_t dst_idx = 0;
                        uint32_t out_col_offset = v_subblock * qktv_subblock_w;
                        for (uint32_t r = 0; r < qktv_subblock_h; r++) {
                            uint32_t out_row_offset = (r + q_subblock * qktv_subblock_h) * head_dim_t;
                            for (uint32_t c = 0; c < qktv_subblock_w; c++) {
                                pack_tile<true>(dst_idx, alias_cur_out, out_row_offset + out_col_offset + c);
                                dst_idx++;
                            }
                        }
                        tile_regs_release();
                        MATH(
                            DPRINT << "Packed " << qktv_subblock_h * qktv_subblock_w << " tiles to alias_cur_out for Q["
                                   << q_subblock << "] V[" << v_subblock << "]" << ENDL());
                    }

                    v_index_offset += qktv_subblock_w;
                }

                {
                    // SALAD: cb_exp_max_diff = slowexp((cb_prev_max - cb_cur_max) * scale)
                    MATH(DPRINT << "SUB_EXP_m for Q[" << q_subblock << "]" << ENDL());
                    SDPA_DeviceZoneScopedN_2("S_SUB_EXP");
                    sub_exp_first_col_blocks<scale_fp32, sbh>(
                        alias_prev_max, alias_cur_max, cb_exp_max_diff, q_subblock);
                    // todo: don't need these rows of prev_max anymore, so pop to free up buffer space now.
                }

                {
                    // SALAD: cb_prev_sum *= cb_exp_max_diff
                    MATH(DPRINT << "Mul tiles bcast cols for Q[" << q_subblock << "]" << ENDL());
                    SDPA_DeviceZoneScopedN_2("S_MUL_TILES");
                    mul_tiles_bcast_cols_inplace<sbh>(alias_prev_sum, cb_exp_max_diff, q_subblock);
                }

                {
                    // SALAD: cb_prev_sum += cb_cur_sum
                    MATH(DPRINT << "Add tiles inplace for Q[" << q_subblock << "]" << ENDL());
                    SDPA_DeviceZoneScopedN_2("S_ADD_TILES");
                    add_block_inplace<sbh>(alias_cur_sum, alias_prev_sum, q_subblock);
                }
                {
                    // SALAD: alias_cur_out += alias_prev_out * cb_exp_max_diff
                    MATH(DPRINT << "Element-wise mul of Q[" << q_subblock << "]" << ENDL());
                    SDPA_DeviceZoneScopedN_2("S_MUL_BLOCK");
                    mul_block_bcast_cols_acc<sbh, head_dim_t>(
                        alias_prev_out, cb_exp_max_diff, alias_cur_out, q_subblock);
                }

                MATH(
                    DPRINT << "Pushing " << qktv_subblock_h * head_dim_t << " tiles to alias_cur_out for Q_subblock "
                           << q_subblock << ENDL());
                cb_push_back(alias_cur_out, qktv_subblock_h * head_dim_t);

                qktv_in0_index_offset += qktv_subblock_h * qktv_in0_block_w;
                qktv_in0_wait_tiles += qktv_in0_subblock_num_tiles;
            }

            MATH(DPRINT << "Popping cb_v_in: " << Sv_chunk_t * head_dim_t << " tiles" << ENDL());
            cb_pop_front(cb_v_in, Sv_chunk_t * head_dim_t);
            MATH(DPRINT << "Popping cb_qkt_im: " << Sq_chunk_t * Sk_chunk_t << " tiles" << ENDL());
            cb_pop_front(cb_qkt_im, Sq_chunk_t * Sk_chunk_t);
        }

        // Restore packer ReLU config after all exp operations complete
        PACK((llk_pack_relu_config(ReluType::NO_RELU)));

        cb_pop_front(cb_exp_max_diff, Sq_chunk_t);

        // Pop prev buffers — frees them for reuse as "cur" in the next iteration.
        // [FUTURE: rescale/accumulate step would go here, BEFORE these pops,
        //  reading alias_prev_max/sum/out and accumulating into alias_cur_max/sum/out]
        cb_pop_front(alias_prev_max, Sq_chunk_t);
        cb_pop_front(alias_prev_sum, Sq_chunk_t);
        cb_pop_front(alias_prev_out, Sq_chunk_t * head_dim_t);

        if (iter < num_iter - 1) {
            // Swap: cur becomes prev for next iteration
            std::swap(alias_prev_max, alias_cur_max);
            std::swap(alias_prev_sum, alias_cur_sum);
            std::swap(alias_prev_out, alias_cur_out);
        } else {
            // Last iteration: consume final cur buffers
            cb_pop_front(alias_cur_max, Sq_chunk_t);
            cb_pop_front(alias_cur_sum, Sq_chunk_t);
            cb_wait_front(alias_cur_out, Sq_chunk_t * head_dim_t);
            cb_pop_front(alias_cur_out, Sq_chunk_t * head_dim_t);
        }
        MATH(DPRINT << "Finished iteration " << iter << ENDL());
    }  // end for (iter)
}

void kernel_main() {
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t Sv_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t head_dim_t = get_compile_time_arg_val(3);
    constexpr uint32_t num_iter = get_compile_time_arg_val(4);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(6);

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_kt_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_qkt_im = tt::CBIndex::c_2;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;

    constexpr uint32_t cb_out_A = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_B = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_A = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_B = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_A = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_B = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;

    mm_init(cb_q_in, cb_kt_in, cb_qkt_im);

    // Dummy pre-populate "prev" CBs.
    cb_reserve_back(cb_max_A, Sq_chunk_t);
    cb_push_back(cb_max_A, Sq_chunk_t);

    cb_reserve_back(cb_sum_A, Sq_chunk_t);
    cb_push_back(cb_sum_A, Sq_chunk_t);

    cb_reserve_back(cb_out_A, Sq_chunk_t * head_dim_t);
    cb_push_back(cb_out_A, Sq_chunk_t * head_dim_t);

    sdpa_inner_loop<
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
        subblock_h>(cb_max_A, cb_max_B, cb_sum_A, cb_sum_B, cb_out_A, cb_out_B, num_iter);
}
