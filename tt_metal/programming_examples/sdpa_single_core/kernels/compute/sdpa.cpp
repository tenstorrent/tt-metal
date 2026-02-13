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

// Toggle: interleave drain sub_exp with split matmul for first QKT@V row.
// Requires sbh=1, kt_num_subblocks=2. Overlaps EXP (SFPU) with matmul (FPU).
#define OVERLAP_DRAIN_WITH_MATMUL

// #define SDPA_PROFILING_SET_1
// #define SDPA_PROFILING_SET_2
// #define SDPA_PROFILING_SET_3
// #define SDPA_PROFILING_SET_4
// #define SDPA_PROFILING_SET_5

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

#ifdef SDPA_PROFILING_SET_3
#define SDPA_DeviceZoneScopedN_3(name) DeviceZoneScopedN(name)
#else
#define SDPA_DeviceZoneScopedN_3(name)
#endif

#ifdef SDPA_PROFILING_SET_4
#define SDPA_DeviceZoneScopedN_4(name) DeviceZoneScopedN(name)
#else
#define SDPA_DeviceZoneScopedN_4(name)
#endif

#ifdef SDPA_PROFILING_SET_5
#define SDPA_DeviceZoneScopedN_5(name) DeviceZoneScopedN(name)
#else
#define SDPA_DeviceZoneScopedN_5(name)
#endif

/**
 * Push tiles to CB (signaling UNPACK thread) but keep fifo_wr_ptr unchanged.
 * This prevents the address drift in pack_tile<true> that occurs when
 * fifo_wr_ptr advances past fifo_rd_ptr after incremental pushes.
 */
ALWI void cb_push_back_hold_wr_ptr(uint32_t cb_id, uint32_t num_tiles) {
    cb_push_back(cb_id, num_tiles);
    PACK(({
        auto& intf = get_local_cb_interface(cb_id);
        intf.fifo_wr_ptr -= num_tiles * intf.fifo_page_size;
        uint32_t fifo_start = intf.fifo_limit - intf.fifo_size;
        if (intf.fifo_wr_ptr < fifo_start) {
            intf.fifo_wr_ptr += intf.fifo_size;
        }
    }));
}

/**
 * out_cb = exp((in0_cb - in1_cb) * scale_fp32)
 * only at 2*q_subblock and 2*q_subblock+1 elements
 *
 * Writes: pack_tile<true> to out_cb at positions [global_row_base .. global_row_base + SBH - 1].
 * Push:   None — caller is responsible for cb_reserve_back before and cb_push_back after.
 */
template <uint32_t scale_fp32, uint32_t SBH>
void sub_exp_first_col_blocks_no_push(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t q_subblock) {
    const uint32_t global_row_base = q_subblock * SBH;
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    {
        SDPA_DeviceZoneScopedN_2("SUB_TILES_INIT");
        sub_tiles_init(in0_cb, in1_cb);
    }

    exp_packthread_tile_init<EXP_APPROX_MODE, false>();

    cb_wait_front(in0_cb, (q_subblock + 1) * SBH);
    cb_wait_front(in1_cb, (q_subblock + 1) * SBH);

    {
        SDPA_DeviceZoneScopedN_2("S_SUB");
        tile_regs_acquire();
        for (uint32_t i = 0; i < SBH; i++) {
            uint32_t tile_index = global_row_base + i;
            sub_tiles(in0_cb, in1_cb, tile_index, tile_index, i /*dst_index*/);
        }
        tile_regs_commit();
    }

    {
        SDPA_DeviceZoneScopedN_2("S_EXP_AND_PACK");
        tile_regs_wait();
        for (uint32_t dst_index = 0; dst_index < SBH; dst_index++) {
            PACK((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(dst_index)));
        }
        PACK(TTI_STALLWAIT(p_stall ::STALL_PACK, p_stall ::WAIT_SFPU));

        for (uint32_t i = 0; i < SBH; i++) {
            uint32_t tile_index = global_row_base + i;
            pack_tile<true>(i /*dst_index*/, out_cb, tile_index);
        }

        tile_regs_release();
    }
}

/**
 * in0_cb += in1_cb
 *
 * Writes: pack_tile<true> to in0_cb at positions [global_row_base .. global_row_base + SBH - 1].
 * Push:   None — in-place overwrite of already-produced tiles; no new tiles are signaled.
 */
template <uint32_t SBH>
void add_block_inplace_no_push(uint32_t in0_cb, uint32_t in1_cb, uint32_t q_subblock) {
    const uint32_t global_row_base = q_subblock * SBH;

    add_tiles_init(in0_cb, in1_cb);
    cb_wait_front(in0_cb, (q_subblock + 1) * SBH);
    cb_wait_front(in1_cb, (q_subblock + 1) * SBH);

    tile_regs_acquire();
    for (uint32_t i = 0; i < SBH; i++) {
        uint32_t src_tile_index = global_row_base + i;
        add_tiles(in0_cb, in1_cb, src_tile_index, src_tile_index, i /*dst_index*/);
    }
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t i = 0; i < SBH; i++) {
        pack_tile<true>(i, in0_cb, global_row_base + i);  // Pack back to original position in in0_cb
    }
    tile_regs_release();
}

/**
 * in0_cb *= in1_cb (with column broadcast of in1_cb)
 *
 * Writes: pack_tile<true> to in0_cb at positions [global_row_base .. global_row_base + SBH - 1].
 * Push:   None — in-place overwrite of already-produced tiles; no new tiles are signaled.
 */
template <uint32_t SBH>
void mul_tiles_bcast_cols_inplace_no_push(uint32_t in0_cb, uint32_t in1_cb, uint32_t q_subblock) {
    const uint32_t global_row_base = q_subblock * SBH;
    mul_bcast_cols_init_short(in0_cb, in1_cb);
    cb_wait_front(in0_cb, (q_subblock + 1) * SBH);
    cb_wait_front(in1_cb, (q_subblock + 1) * SBH);

    tile_regs_acquire();
    for (uint32_t i = 0; i < SBH; i++) {
        uint32_t src_tile_index = global_row_base + i;
        mul_tiles_bcast_cols(in0_cb, in1_cb, src_tile_index, src_tile_index, i /*dst_index*/);
    }
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t i = 0; i < SBH; i++) {
        pack_tile<true>(i, in0_cb, global_row_base + i);  // Pack back to original position in in0_cb
    }
    tile_regs_release();
}

/**
 * out_cb += in0_cb * in1_cb (with column broadcast of in1_cb, L1 accumulation into out_cb)
 *
 * Writes: pack_tile<true> to out_cb at positions [(global_row_base+i)*SBW + j] with L1 accumulation.
 * Push:   None — L1-accumulates into already-reserved tiles in out_cb; no new tiles are signaled.
 */
template <uint32_t SBH, uint32_t SBW>
void mul_block_bcast_cols_acc_no_push(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t q_subblock) {
    static_assert(SBH * SBW <= 8, "SBH * SBW must fit in DST (max 8 tiles)");
    const uint32_t global_row_base = q_subblock * SBH;
    mul_bcast_cols_init_short(in0_cb, in1_cb);

    cb_wait_front(in0_cb, (q_subblock + 1) * SBH * SBW);
    cb_wait_front(in1_cb, (q_subblock + 1) * SBH);

    PACK((llk_pack_reconfig_l1_acc(1 /*pack accumulate*/)));
    tile_regs_acquire();
    uint32_t dst_index = 0;
    for (uint32_t i = 0; i < SBH; i++) {
        for (uint32_t j = 0; j < SBW; j++) {
            uint32_t in0_tile_index = (global_row_base + i) * SBW + j;  // Absolute tile index for in0_cb
            mul_tiles_bcast_cols(in0_cb, in1_cb, in0_tile_index, global_row_base + i, dst_index++);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    dst_index = 0;
    for (uint32_t i = 0; i < SBH; i++) {
        for (uint32_t j = 0; j < SBW; j++) {
            uint32_t out_tile_index = (global_row_base + i) * SBW + j;  // Absolute tile index for in0_cb
            pack_tile<true>(dst_index++, out_cb, out_tile_index);  // Pack to original position in out_cb
        }
    }
    tile_regs_release();
    PACK((llk_pack_reconfig_l1_acc(false)));
}

/**
 * Fused SALAD rescale: single DST pass for both sum and output rescaling.
 *
 * Computes:
 *   cur_sum  += prev_sum * exp_max_diff  (col-broadcast)
 *   cur_out  += prev_out * exp_max_diff  (col-broadcast)
 *
 * Both multiplications share the same exp_max_diff bcast operand and data format,
 * so a single mul_bcast_cols_init_short suffices. By doing all muls in one DST
 * session we eliminate 1 acquire/commit/wait/release cycle per SALAD row.
 *
 * Also removes the dead pack-back to prev_sum (its updated value is never read).
 *
 * Writes: pack_tile<true> to cur_sum_cb and cur_out_cb with L1 accumulation.
 * Push:   None — caller manages push.
 *
 * @tparam SBH  Subblock height (tiles per row)
 * @tparam SBW  Output width in tiles (head_dim_t)
 */
template <uint32_t SBH, uint32_t SBW>
void salad_rescale_fused(
    uint32_t prev_sum_cb,
    uint32_t prev_out_cb,
    uint32_t bcast_cb,
    uint32_t cur_sum_cb,
    uint32_t cur_out_cb,
    uint32_t q_subblock) {
    // DST needs: SBH tiles for sum + SBH*SBW tiles for out
    constexpr uint32_t total_dst_tiles = SBH + SBH * SBW;
    static_assert(total_dst_tiles <= 8, "Fused SALAD must fit in DST (max 8 tiles)");

    const uint32_t global_row_base = q_subblock * SBH;

    // Single init covers both muls (same bcast type, same data format)
    mul_bcast_cols_init_short(prev_sum_cb, bcast_cb);

    // Wait for all inputs
    cb_wait_front(prev_sum_cb, (q_subblock + 1) * SBH);
    cb_wait_front(prev_out_cb, (q_subblock + 1) * SBH * SBW);
    cb_wait_front(bcast_cb, (q_subblock + 1) * SBH);

    // === Single DST pass: all muls ===
    tile_regs_acquire();

    // Phase 1: prev_sum * exp_max_diff -> DST[0 .. SBH-1]
    uint32_t dst_index = 0;
    for (uint32_t i = 0; i < SBH; i++) {
        uint32_t src_tile_index = global_row_base + i;
        mul_tiles_bcast_cols(prev_sum_cb, bcast_cb, src_tile_index, src_tile_index, dst_index++);
    }

    // Phase 2: prev_out * exp_max_diff -> DST[SBH .. SBH + SBH*SBW - 1]
    // Uses different src0 CB but same bcast CB and same data format — no reinit needed.
    for (uint32_t i = 0; i < SBH; i++) {
        for (uint32_t j = 0; j < SBW; j++) {
            uint32_t in0_tile_index = (global_row_base + i) * SBW + j;
            mul_tiles_bcast_cols(prev_out_cb, bcast_cb, in0_tile_index, global_row_base + i, dst_index++);
        }
    }

    tile_regs_commit();

    // === Single pack pass: all L1-acc packs ===
    tile_regs_wait();
    PACK((llk_pack_reconfig_l1_acc(1)));

    // Pack sum results: DST[0..SBH-1] -> cur_sum (L1 accumulate)
    // No pack-back to prev_sum — that value is never read again.
    dst_index = 0;
    for (uint32_t i = 0; i < SBH; i++) {
        pack_tile<true>(dst_index++, cur_sum_cb, global_row_base + i);
    }

    // Pack out results: DST[SBH..] -> cur_out (L1 accumulate)
    for (uint32_t i = 0; i < SBH; i++) {
        for (uint32_t j = 0; j < SBW; j++) {
            uint32_t out_tile_index = (global_row_base + i) * SBW + j;
            pack_tile<true>(dst_index++, cur_out_cb, out_tile_index);
        }
    }

    tile_regs_release();
    PACK((llk_pack_reconfig_l1_acc(false)));
}

/**
 * inout0_cb = exp((inout0_cb - in1_cb) * scale) with column broadcast, applied to one subblock.
 * Optionally reduces rows into reduce_cb via L1 accumulation.
 *
 * Writes: pack_tile<true> to inout0_cb at positions [(global_row_base+i)*cols + (global_col_base+j)].
 *         In-place overwrite of already-produced tiles on inout0_cb.
 *         If do_reduce: pack_tile<true> to reduce_cb at positions [global_row_base+i] with L1 accumulation.
 * Push:   None on either CB — caller manages push for the accumulated reduce_cb result.
 */
template <
    uint32_t inout0_cb,
    uint32_t scale_fp32,
    uint32_t SBH,
    uint32_t SBW,
    bool do_reduce = true,
    int vector_mode = (int)VectorMode::RC>
void sub_exp_block_bcast_cols_no_push(
    uint32_t in1_cb, uint32_t reduce_cb, uint32_t cols, uint32_t q_subblock, uint32_t kt_subblock) {
    static_assert(SBH * SBW <= 8, "SBH * SBW must fit in DST (max 8 tiles)");
    const uint32_t global_row_base = q_subblock * SBH;
    const uint32_t global_col_base = kt_subblock * SBW;

    // Initialize operation
    {
        SDPA_DeviceZoneScopedN_1("SUB_BCAST_INIT");
        sub_bcast_cols_init_short(inout0_cb, in1_cb);
    }
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));

    // Wait for tiles:
    // - in0_cb: cumulative wait since we never pop it
    // - in1_cb: cumulative wait since we never pop it
    cb_wait_front(inout0_cb, (q_subblock + 1) * SBH * cols);
    cb_wait_front(in1_cb, (q_subblock + 1) * SBH);

    {
        tile_regs_acquire();
        SDPA_DeviceZoneScopedN_1("SUB");
        uint32_t dst_index = 0;
        for (uint32_t i = 0; i < SBH; i++) {
            for (uint32_t j = 0; j < SBW; j++) {
                uint32_t in0_tile_index =
                    (global_row_base + i) * cols + (global_col_base + j);  // Absolute tile index for in0_cb
                sub_tiles_bcast_cols(inout0_cb, in1_cb, in0_tile_index, global_row_base + i, dst_index++);
            }
        }
        tile_regs_commit();
    }

    {
        tile_regs_wait();
        SDPA_DeviceZoneScopedN_1("EXP");
        uint32_t dst_index = 0;
        // Use fast exp with InputClamping::None and 32 iterations for 1.3x speedup
        // When vector_mode is RC, use 32 iterations with VectorMode::None
        // Otherwise, use 8 iterations with the original vector_mode
        constexpr int iterations = (vector_mode == (int)VectorMode::RC) ? 32 : 8;
        constexpr int vector_mode_exp = (vector_mode == (int)VectorMode::RC) ? (int)VectorMode::None : vector_mode;
        for (uint32_t i = 0; i < SBH; i++) {
            for (uint32_t j = 0; j < SBW; j++) {
                exp_packthread_tile<
                    true,   // approx
                    true,   // fast_and_approx
                    false,  // scale_en
                    false,  // skip_positive_check
                    InputClamping::None,
                    iterations>(dst_index++, vector_mode_exp);
            }
        }

        // Pack inout0_cb: one pack_tile per row, MOP=SBW handles all columns in the row.
        dst_index = 0;
        for (uint32_t i = 0; i < SBH; i++) {
            uint32_t in0_tile_index = (global_row_base + i) * cols + global_col_base;
            pack_tile<true>(dst_index, inout0_cb, in0_tile_index);
            dst_index += SBW;
        }

        // Reduce: L1-accumulate all SBW column tiles into one tile per row.
        // One pack_tile per row; MOP=SBW packs SBW tiles to the same L1 address, accumulating.
        if constexpr (do_reduce) {
            dst_index = 0;
            PACK((llk_pack_reconfig_l1_acc(1)));
            for (uint32_t i = 0; i < SBH; i++) {
                pack_tile<true>(dst_index, reduce_cb, global_row_base + i);
                dst_index += SBW;
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
 * Reduces SBH rows of in0_cb along columns, optionally eltwise-max'd with prev_cb.
 *
 * Writes: pack_tile<true> to out_cb at positions [row_start .. row_start + SBH - 1].
 * Push:   None — caller is responsible for cb_reserve_back before and cb_push_back after.
 */
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_cb,
    uint32_t scale_cb,
    uint32_t cols,
    uint32_t SBH,
    int vector_mode = static_cast<int>(VectorMode::C)>
void reduce_c_row_group_no_push(
    uint32_t out_cb, uint32_t prev_cb, uint32_t row_group_index, bool do_eltwise_max = false) {
    // Precondition: in0_cb has at least (row_group_index + 1) * SBH * cols tiles produced (row-major order)
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has space reserved by caller
    // Precondition: prev_cb has at least (row_group_index + 1) * SBH tiles produced (if do_eltwise_max)
    // Postcondition: in0_cb unchanged (no pop)
    // Postcondition: out_cb has SBH more tiles packed at positions [row_group_index*SBH, ...]

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
    PACK((llk_pack_mop_config<false, false, false>(out_cb, 1)));
    tile_regs_wait();

    {
        SDPA_DeviceZoneScopedN_3("Pack tiles");
        for (uint32_t i = 0; i < GROUP_SIZE; i++) {
            const uint32_t dst_idx = i;
            pack_tile<true>(dst_idx, out_cb, row_start + i);
        }
    }

    {
        SDPA_DeviceZoneScopedN_3("Reconfig back to 8");
        PACK((llk_pack_mop_config<false, false, false>(out_cb, 8)));
    }
    tile_regs_release();
}

/**
 * 1x8 blocked matmul: init hoisted to caller, packs 1 tile per row (MOP-based 8-wide packing).
 * Used for Q@KT with qkt_subblock_w=8. Assumes MOP=SUBBLOCK_W is set by caller.
 * Push: None.
 */
template <
    bool TRANSPOSE,
    uint32_t SUBBLOCK_W,
    uint32_t SUBBLOCK_H,
    uint32_t INNER_DIM,
    uint32_t IN1_STRIDE,
    uint32_t OUT_NUM_COLS>
void blocked_1x8_matmul_and_pack(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t in0_index_start,
    uint32_t in1_index_start,
    uint32_t q_subblock,
    uint32_t out_col_offset) {
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

    // --- Pack phase: relies on ambient MOP=SUBBLOCK_W ---
    tile_regs_wait();
    uint32_t dst_idx = 0;
    for (uint32_t r = 0; r < SUBBLOCK_H; r++) {
        uint32_t out_row_offset = (r + q_subblock * SUBBLOCK_H) * OUT_NUM_COLS;
        pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset);
        dst_idx += SUBBLOCK_W;
    }
    tile_regs_release();
}

/**
 * 1x4 blocked matmul: init hoisted to caller, packs all tiles individually.
 * Used for QKT@V with qktv_subblock_w=4.
 * Push: None.
 */
template <
    bool TRANSPOSE,
    uint32_t SUBBLOCK_W,
    uint32_t SUBBLOCK_H,
    uint32_t INNER_DIM,
    uint32_t IN1_STRIDE,
    uint32_t OUT_NUM_COLS>
void blocked_1x4_matmul_and_pack(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t in0_index_start,
    uint32_t in1_index_start,
    uint32_t q_subblock,
    uint32_t out_col_offset) {
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
    uint32_t dst_idx = 0;
    for (uint32_t r = 0; r < SUBBLOCK_H; r++) {
        uint32_t out_row_offset = (r + q_subblock * SUBBLOCK_H) * OUT_NUM_COLS;
        for (uint32_t c = 0; c < SUBBLOCK_W; c++) {
            pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset + c);
            dst_idx++;
        }
    }
    tile_regs_release();
}

/**
 * 1x4 blocked matmul (faster): init hoisted to caller, packs 1 tile per row (MOP-based 4-wide packing).
 * Used for QKT@V non-overlap path.
 * Push: None.
 */
template <
    bool TRANSPOSE,
    uint32_t SUBBLOCK_W,
    uint32_t SUBBLOCK_H,
    uint32_t INNER_DIM,
    uint32_t IN1_STRIDE,
    uint32_t OUT_NUM_COLS>
void blocked_1x4_matmul_and_pack_faster(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t in0_index_start,
    uint32_t in1_index_start,
    uint32_t q_subblock,
    uint32_t out_col_offset) {
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
    uint32_t dst_idx = 0;
    for (uint32_t r = 0; r < SUBBLOCK_H; r++) {
        uint32_t out_row_offset = (r + q_subblock * SUBBLOCK_H) * OUT_NUM_COLS;
        pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset);
        dst_idx += SUBBLOCK_W;
    }
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

        // pack_reconfig_data_format(cb_qkt_im);
        // reconfig_data_format(cb_kt_in, cb_q_in);
        cb_reserve_back(cb_qkt_im, Sq_chunk_t * Sk_chunk_t);

        cb_wait_front(cb_kt_in, head_dim_t * Sk_chunk_t);
        cb_reserve_back(alias_cur_sum, Sq_chunk_t);
        cb_reserve_back(alias_cur_max, Sq_chunk_t);
        PACK((llk_pack_mop_config<false, false, false>(cb_qkt_im, qkt_subblock_w)));
        for (uint32_t q_subblock = 0; q_subblock < q_num_subblocks; q_subblock++) {
            // DeviceZoneScopedN("Softmax(Q@KT)");
            cb_wait_front(cb_q_in, q_wait_tiles);
            kt_index_offset = 0;

            for (uint32_t kt_subblock = 0; kt_subblock < kt_num_subblocks; ++kt_subblock) {
                if (q_subblock > 0) {
                    uint32_t prev_q_subblock = q_subblock - 1;
                    MATH(DPRINT << "SUB EXP for Q[" << prev_q_subblock << "] Kt[" << kt_subblock << "]" << ENDL());
                    // SDPA_DeviceZoneScopedN_5("SUB EXP");
                    sub_exp_block_bcast_cols_no_push<cb_qkt_im, scale_fp32, sbh, qkt_subblock_w, true>(
                        alias_cur_max, alias_cur_sum, Sk_chunk_t, prev_q_subblock, kt_subblock);
                }

                {
                    SDPA_DeviceZoneScopedN_1("Q@KT MM+Pack");
                    // SDPA_DeviceZoneScopedN_5("Q@KT MM+Pack");
                    if (q_subblock > 0 || q_subblock == 0 && kt_subblock == 0) {
                        mm_block_init_short(cb_q_in, cb_kt_in, true, qkt_subblock_w, sbh, in0_block_w);
                    }
                    blocked_1x8_matmul_and_pack<true, qkt_subblock_w, sbh, in0_block_w, Sk_chunk_t, Sk_chunk_t>(
                        cb_q_in,
                        cb_kt_in,
                        cb_qkt_im,
                        q_index_offset,
                        kt_index_offset,
                        q_subblock,
                        kt_subblock * qkt_subblock_w);
                }
                kt_index_offset += qkt_subblock_w;
            }
            cb_push_back_hold_wr_ptr(cb_qkt_im, sbh * Sk_chunk_t);

            // Max reduce
            MATH(DPRINT << "Max reduce for Q[" << q_subblock << ", :]" << ENDL());
            {
                SDPA_DeviceZoneScopedN_1("Reduce max");
                reduce_c_row_group_no_push<
                    PoolType::MAX,
                    ReduceDim::REDUCE_ROW,
                    cb_qkt_im,
                    cb_identity_scale_in,
                    Sk_chunk_t,
                    sbh>(alias_cur_max, alias_prev_max, q_subblock, true /*do_eltwise_max*/);
                // alias_cur_max: reduce packed sbh tiles for this q_subblock.
                // Held push — keeps wr_ptr at CB base for next q_subblock's pack_tile<true>.
                cb_push_back_hold_wr_ptr(alias_cur_max, sbh);
            }

            q_index_offset += sbh * in0_block_w;
            q_wait_tiles += q_subblock_num_tiles;
        }
        PACK((llk_pack_mop_config<false, false, false>(cb_qkt_im, 1)));

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
            cb_reserve_back(cb_exp_max_diff, Sq_chunk_t);

            // ===== q_subblock 0: drain overlap + matmul, NO SALAD, NO push =====
            {
                MATH(DPRINT << "QKT@V: Processing Q_subblock 0" << ENDL());
                // DeviceZoneScopedN("Softmax(Q@KT)@V");
                cb_wait_front(cb_qkt_im, qktv_in0_wait_tiles);

#ifdef OVERLAP_DRAIN_WITH_MATMUL
                {
                    SDPA_DeviceZoneScopedN_5("DRAIN OVERLAP");
                    constexpr uint32_t half_inner = qktv_in0_block_w >> 2;
                    static_assert(kt_num_subblocks == 2, "Overlap drain requires kt_num_subblocks==2");

                    // 1. sub_exp drain pass 1 (kt=0): SUB on FPU, then EXP on SFPU
                    MATH(DPRINT << "DRAIN OVERLAP: SUB_EXP kt=0" << ENDL());
                    {
                        SDPA_DeviceZoneScopedN_5("SUB EXP");
                        sub_exp_block_bcast_cols_no_push<cb_qkt_im, scale_fp32, sbh, qkt_subblock_w, true>(
                            alias_cur_max, alias_cur_sum, Sk_chunk_t, q_num_subblocks - 1, 0);
                    }

                    // 2. matmul first half — FPU overlaps with EXP(kt=0) on SFPU
                    {
                        uint32_t v_index_offset = 0;
                        mm_block_init_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_subblock_h, half_inner);
                        for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                            SDPA_DeviceZoneScopedN_2("QKT@V MM+Pack H1");
                            SDPA_DeviceZoneScopedN_5("QKT@V MM+Pack H1");
                            blocked_1x4_matmul_and_pack<
                                false,
                                qktv_subblock_w,
                                qktv_subblock_h,
                                half_inner,
                                head_dim_t,
                                head_dim_t>(
                                cb_qkt_im,
                                cb_v_in,
                                alias_cur_out,
                                qktv_in0_index_offset,
                                v_index_offset,
                                0,
                                v_subblock * qktv_subblock_w);
                            v_index_offset += qktv_subblock_w;
                        }
                    }

                    // 3. sub_exp drain pass 2 (kt=1): SUB on FPU, then EXP on SFPU
                    MATH(DPRINT << "DRAIN OVERLAP: SUB_EXP kt=1" << ENDL());
                    {
                        SDPA_DeviceZoneScopedN_5("SUB EXP");
                        sub_exp_block_bcast_cols_no_push<cb_qkt_im, scale_fp32, sbh, qkt_subblock_w, true>(
                            alias_cur_max, alias_cur_sum, Sk_chunk_t, q_num_subblocks - 1, 1);
                    }
                    // alias_cur_sum: full-CB push (Sq_chunk_t tiles on Sq_chunk_t-tile CB).
                    // Regular push — wr_ptr wraps naturally to CB base. No hold needed.
                    cb_push_back(alias_cur_sum, Sq_chunk_t);

                    // 4. matmul second half with L1 accumulate — FPU overlaps with EXP(kt=1) on SFPU
                    PACK((llk_pack_reconfig_l1_acc(1)));
                    {
                        uint32_t v_index_offset = 0;
                        mm_block_init_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_subblock_h, half_inner);
                        for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                            SDPA_DeviceZoneScopedN_2("QKT@V MM+Pack H2");
                            SDPA_DeviceZoneScopedN_5("QKT@V MM+Pack H2");
                            blocked_1x4_matmul_and_pack<
                                false,
                                qktv_subblock_w,
                                qktv_subblock_h,
                                half_inner,
                                head_dim_t,
                                head_dim_t>(
                                cb_qkt_im,
                                cb_v_in,
                                alias_cur_out,
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
                    // Non-overlap path: drain + full matmul
                    MATH(DPRINT << "DRAIN: SUB_EXP for Q[" << q_num_subblocks - 1 << ENDL());
                    for (uint32_t kt_subblock = 0; kt_subblock < kt_num_subblocks; ++kt_subblock) {
                        sub_exp_block_bcast_cols_no_push<cb_qkt_im, scale_fp32, sbh, qkt_subblock_w, true>(
                            alias_cur_max, alias_cur_sum, Sk_chunk_t, q_num_subblocks - 1, kt_subblock);
                    }
                    // alias_cur_sum: full-CB push (Sq_chunk_t tiles on Sq_chunk_t-tile CB).
                    // Regular push — wr_ptr wraps naturally to CB base. No hold needed.
                    cb_push_back(alias_cur_sum, Sq_chunk_t);

                    uint32_t v_index_offset = 0;
                    mm_block_init_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_subblock_h, qktv_in0_block_w);
                    for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                        SDPA_DeviceZoneScopedN_2("QKT@V MM+Pack");
                        blocked_1x4_matmul_and_pack_faster<
                            false,
                            qktv_subblock_w,
                            qktv_subblock_h,
                            qktv_in0_block_w,
                            head_dim_t,
                            head_dim_t>(
                            cb_qkt_im,
                            cb_v_in,
                            alias_cur_out,
                            qktv_in0_index_offset,
                            v_index_offset,
                            0,
                            v_subblock * qktv_subblock_w);
                        v_index_offset += qktv_subblock_w;
                    }
                }
#endif
                // No SALAD, no push for q_subblock 0 — deferred to next iteration
                qktv_in0_index_offset += qktv_subblock_h * qktv_in0_block_w;
                qktv_in0_wait_tiles += qktv_in0_subblock_num_tiles;
            }

            // ===== q_subblock 1..3: SALAD(prev) overlapped with matmul(cur) =====
            for (uint32_t q_subblock = 1; q_subblock < qktv_q_num_subblocks; ++q_subblock) {
                uint32_t salad_row = q_subblock - 1;
                MATH(
                    DPRINT << "QKT@V: Processing Q_subblock " << q_subblock << ", SALAD for row " << salad_row
                           << ENDL());
                // DeviceZoneScopedN("Softmax(Q@KT)@V");
                cb_wait_front(cb_qkt_im, qktv_in0_wait_tiles);

                // 1. Compute exp_max_diff for PREVIOUS row — SFPU EXP will overlap with matmul
                {
                    MATH(DPRINT << "SUB_EXP_m for Q[" << salad_row << "]" << ENDL());
                    SDPA_DeviceZoneScopedN_5("SLOW EXP");
                    sub_exp_first_col_blocks_no_push<scale_fp32, sbh>(
                        alias_prev_max, alias_cur_max, cb_exp_max_diff, salad_row);
                    // cb_exp_max_diff: sub_exp packed sbh tiles for salad_row.
                    // Held push — keeps wr_ptr at CB base for next salad_row's pack_tile<true>.
                    cb_push_back_hold_wr_ptr(cb_exp_max_diff, sbh);
                }

                // 2. Full matmul for CURRENT row — FPU overlaps with SFPU EXP above
                {
                    SDPA_DeviceZoneScopedN_5("QKT@V MM+Pack");
                    uint32_t v_index_offset = 0;
                    mm_block_init_short(cb_qkt_im, cb_v_in, false, qktv_subblock_w, qktv_subblock_h, qktv_in0_block_w);
                    for (uint32_t v_subblock = 0; v_subblock < qktv_v_num_subblocks; ++v_subblock) {
                        SDPA_DeviceZoneScopedN_2("QKT@V MM+Pack");
                        blocked_1x4_matmul_and_pack<
                            false,
                            qktv_subblock_w,
                            qktv_subblock_h,
                            qktv_in0_block_w,
                            head_dim_t,
                            head_dim_t>(
                            cb_qkt_im,
                            cb_v_in,
                            alias_cur_out,
                            qktv_in0_index_offset,
                            v_index_offset,
                            q_subblock,
                            v_subblock * qktv_subblock_w);
                        v_index_offset += qktv_subblock_w;
                    }
                }

                // 3. Rest of SALAD for PREVIOUS row
                {
                    MATH(DPRINT << "Mul tiles bcast cols for Q[" << salad_row << "]" << ENDL());
                    SDPA_DeviceZoneScopedN_2("S_MUL_TILES");
                    mul_tiles_bcast_cols_inplace_no_push<sbh>(alias_prev_sum, cb_exp_max_diff, salad_row);
                }
                {
                    MATH(DPRINT << "Add tiles inplace for Q[" << salad_row << "]" << ENDL());
                    SDPA_DeviceZoneScopedN_2("S_ADD_TILES");
                    add_block_inplace_no_push<sbh>(alias_cur_sum, alias_prev_sum, salad_row);
                }
                {
                    MATH(DPRINT << "Element-wise mul of Q[" << salad_row << "]" << ENDL());
                    SDPA_DeviceZoneScopedN_2("S_MUL_BLOCK");
                    mul_block_bcast_cols_acc_no_push<sbh, head_dim_t>(
                        alias_prev_out, cb_exp_max_diff, alias_cur_out, salad_row);
                }

                // 4. Push completed row for PREVIOUS subblock.
                // alias_cur_out: matmul + salad_rescale packed tiles for salad_row.
                // Held push — signals UNPACK but keeps wr_ptr at CB base for next q_subblock's pack_tile<true>.
                MATH(
                    DPRINT << "Pushing " << qktv_subblock_h * head_dim_t << " tiles to alias_cur_out for Q_subblock "
                           << salad_row << ENDL());
                cb_push_back_hold_wr_ptr(alias_cur_out, qktv_subblock_h * head_dim_t);

                qktv_in0_index_offset += qktv_subblock_h * qktv_in0_block_w;
                qktv_in0_wait_tiles += qktv_in0_subblock_num_tiles;
            }

            // ===== Pipeline drain: SALAD for last row =====
            {
                constexpr uint32_t salad_row = qktv_q_num_subblocks - 1;
                MATH(DPRINT << "Pipeline drain: SALAD for row " << salad_row << ENDL());
                // DeviceZoneScopedN("SALAD drain");

                sub_exp_first_col_blocks_no_push<scale_fp32, sbh>(
                    alias_prev_max, alias_cur_max, cb_exp_max_diff, salad_row);
                cb_push_back_hold_wr_ptr(cb_exp_max_diff, sbh);
                mul_tiles_bcast_cols_inplace_no_push<sbh>(alias_prev_sum, cb_exp_max_diff, salad_row);
                add_block_inplace_no_push<sbh>(alias_cur_sum, alias_prev_sum, salad_row);
                mul_block_bcast_cols_acc_no_push<sbh, head_dim_t>(
                    alias_prev_out, cb_exp_max_diff, alias_cur_out, salad_row);

                // alias_cur_out: SALAD drain packed tiles for last row.
                // Held push — signals UNPACK but keeps wr_ptr at CB base (same as SALAD loop pushes).
                MATH(
                    DPRINT << "Pushing " << qktv_subblock_h * head_dim_t << " tiles to alias_cur_out for Q_subblock "
                           << salad_row << ENDL());
                cb_push_back_hold_wr_ptr(alias_cur_out, qktv_subblock_h * head_dim_t);
            }

            MATH(DPRINT << "Popping cb_v_in: " << Sv_chunk_t * head_dim_t << " tiles" << ENDL());
            cb_pop_front(cb_v_in, Sv_chunk_t * head_dim_t);
            MATH(DPRINT << "Popping cb_qkt_im: " << Sq_chunk_t * Sk_chunk_t << " tiles" << ENDL());
            cb_pop_front(cb_qkt_im, Sq_chunk_t * Sk_chunk_t);
        }

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
    // Regular push — each is a full-CB push so wr_ptr wraps naturally to CB base. No hold needed.
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
