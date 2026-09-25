// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The cyclic forward pass on one core (tt-flash-attn, Algorithm 8): the
// schedule's block pairs, each folding one key/value column into the
// online-softmax state of one query row block.
//
// Everything is computed in the transposed orientation of the backward
// kernel: S^T = K Q^T, P^T = exp(a (S^T - m)), and the accumulator is
// O^T = V^T P^T, every tile transposed within itself, so that the matmuls
// need no transposed score tile and the per-query statistics m and l are
// row broadcasts (one value per column of S^T, in row 0 of a tile).
//
// Per timestep, for the pair (i, j):
//
//   1. S^T for every live key tile, packed to L1 twice: rounded to the 19
//      bits the FPU reads for the column maximum, and exact for the
//      subtraction;
//   2. per query tile: m_new = max(m_old, colmax S^T), r = exp(a (m_old - m_new));
//   3. P^T = exp(a (S^T - m_new)), the subtraction on the SFPU from the
//      exact copy, packed rounded for the matmul;
//   4. l_new = r l_old + colsum P^T;
//   5. O^T <- r O^T + V^T P^T, the rescale on the SFPU from the exact seed,
//      the products accumulated by the FPU on top;
//   6. at the row's last visit: O = (O^T / l)^T in bfloat16 and
//      lse = a m + ln l in column layout, for the write kernel.
//
// The statistics' representation. The column reductions produce a row-layout
// tile: one value per query in row 0, and whatever partial results the
// reduction leaves in the other rows -- which nothing reads: every use is a
// row broadcast of row 0 through SrcB (19 bits), or a row-0 mask before the
// lse. m travels as a *full* tile, every row the same, made once per visit
// from the broadcast block maximum (exact: a maximum of 19-bit-rounded
// scores), so r = exp(a (m_old - m_new)) and the rescales of O^T and l are
// plain SFPU operations on exact copies and the running sum l stays exact.
// S^T - m is the FPU's row-broadcast subtraction, which reads S^T at 19 bits
// (rounded to nearest by the packer): P carries that 5e-4 relative rounding
// of the score, the same one sdpa_fw's and the backward's own probabilities
// carry; the lse stays flat in N because the rescale factors are exact.
//
// The threads. The probabilities' exponential -- the bulk of the SFPU work
// -- runs on the pack thread after the math thread's commit, overlapping the
// next column's FPU subtraction and the packs; every other SFPU operation
// (on exact unpack-to-dest copies) stays on the math thread, and the CB
// protocol keeps the two apart in time (see pack_sfpu). The SFPU's
// programmable constants are shared between the threads, so the exponential
// reloads its own before every run and restores register 11's -1.0 after.
//
// The state travels in the packet: read through the seed views (c_13 m,
// c_14 l, c_15 O^T), packed back through the out views (c_18, c_19, c_17)
// onto the same memory. At a row's first streak there is no state and the
// first update writes.

#include <api/compute/cb_api.h>
#include <api/compute/pack.h>
#include <api/compute/reconfig_data_format.h>
#include <api/compute/reg_api.h>
#include <hostdevcommon/kernel_structs.h>
#include <tensix.h>

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/mask.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/transpose_dest.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt-train/sources/ttml/metal/common/sdpa_compute_utils_common.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"
#include "api/dataflow/circular_buffer.h"  // invalidate_l1_cache

#ifndef DENSE_MODE
#define DENSE_MODE 0
#endif

#if DENSE_MODE
constexpr auto kMaskMode = ttml::metal::ops::cyclic_sdpa_bw::MaskMode::Dense;
#else
constexpr auto kMaskMode = ttml::metal::ops::cyclic_sdpa_bw::MaskMode::Causal;
#endif

namespace {

constexpr uint32_t kCores = get_compile_time_arg_val(0);
constexpr uint32_t qWt = get_compile_time_arg_val(1);
constexpr uint32_t vWt = get_compile_time_arg_val(2);
// The softmax scale a = 1/sqrt(d): folded into the exponential's constant
// (it computes exp(a x)) and applied to m in the lse.
constexpr uint32_t scaler_bits = get_compile_time_arg_val(3);
constexpr uint32_t block_size = get_compile_time_arg_val(4);
constexpr uint32_t Bt = get_compile_time_arg_val(5);
constexpr uint32_t score_tiles = Bt * Bt;
// The destination file is half-synchronised: four Float32 tiles for the math
// thread while the pack thread drains the other four. Every stage works in
// groups that fit: score tiles of a column (the scores and the probabilities
// need no other register, so all Bt <= 4 of them), two output tiles of a
// query tile with r above them, two query tiles' statistics.
constexpr uint32_t kGroup = Bt;
constexpr uint32_t kOutGroup = (qWt > 2u) ? 2u : qWt;
// The exponential's bias constant: 127, nothing folded in (exp(a x) exactly).
constexpr uint32_t exp_bias_bits = 0x42FE0000u;

constexpr uint32_t cb_query = tt::CBIndex::c_0;
constexpr uint32_t cb_key = tt::CBIndex::c_1;
constexpr uint32_t cb_value = tt::CBIndex::c_2;
constexpr uint32_t cb_value_t = tt::CBIndex::c_16;
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_6;
constexpr uint32_t kMaskTriangle = 0;
constexpr uint32_t cb_zero_tile = tt::CBIndex::c_8;
constexpr uint32_t cb_transpose_fence = tt::CBIndex::c_8;
constexpr uint32_t cb_ones_column = tt::CBIndex::c_28;
constexpr uint32_t cb_ones_row = tt::CBIndex::c_29;
constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_27;
constexpr uint32_t cb_scores = tt::CBIndex::c_10;        // S^T, 19-bit rounded
constexpr uint32_t cb_probs = tt::CBIndex::c_12;         // P^T, 19-bit rounded
constexpr uint32_t cb_rescale = tt::CBIndex::c_20;       // r, full tile, exact
constexpr uint32_t cb_block_max = tt::CBIndex::c_23;     // colmax S^T (fresh rows) or colmax S^T - m, row 0 (scratch)
constexpr uint32_t cb_max_seed = tt::CBIndex::c_13;      // m, full tile, exact (unpack to dest)
constexpr uint32_t cb_max_plain = tt::CBIndex::c_25;     // m, the same memory, for the FPU broadcasts
constexpr uint32_t cb_sum_seed = tt::CBIndex::c_14;      // l, exact (unpack to dest)
constexpr uint32_t cb_sum_plain = tt::CBIndex::c_26;     // l, the same memory, for the FPU broadcast
constexpr uint32_t cb_out_seed = tt::CBIndex::c_15;
constexpr uint32_t cb_max_out = tt::CBIndex::c_18;
constexpr uint32_t cb_sum_out = tt::CBIndex::c_19;
constexpr uint32_t cb_out_out = tt::CBIndex::c_17;
constexpr uint32_t cb_output = tt::CBIndex::c_21;
constexpr uint32_t cb_lse = tt::CBIndex::c_22;
constexpr uint32_t cb_slot_release = tt::CBIndex::c_7;
// A forwarded row's state goes to the reader one query tile at a time when the
// head dimension is two tiles or fewer, whole otherwise (see hand_over_tile);
// the relay reader keys on the same rule.
constexpr bool kStatePerTile = (qWt <= 2u);

// Math fidelity per matmul, measured on the forward: the scores lose (lse
// 4.7e-4 -> 1.15e-3) below HiFi3; the output matmul, P^T's 19 bits in SrcA,
// is as accurate at HiFi3 as at HiFi4 (O RMS 1.71e-3 both), so HiFi3. Time
// barely moves either way (HiFi2 everywhere saves 0.2 of 5.7 ms): the kernel
// is not FPU-bound.
#ifndef FID_S
#define FID_S 3
#endif
#ifndef FID_O
#define FID_O 3
#endif
constexpr MathFidelity fid(int phases) {
    return phases == 2 ? MathFidelity::HiFi2 : phases == 3 ? MathFidelity::HiFi3 : MathFidelity::HiFi4;
}
constexpr MathFidelity kFidS = fid(FID_S), kFidO = fid(FID_O);
template <MathFidelity MF>
void mm_init(uint32_t in0, uint32_t in1, uint32_t transpose) {
    MATH((llk_math_matmul_init<MF, MM_THROTTLE>(in0, in1, transpose)));
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, transpose)));
}
template <MathFidelity MF>
void mm_tiles(uint32_t in0, uint32_t in1, uint32_t t0, uint32_t t1, uint32_t idst) {
    UNPACK((llk_unpack_AB_matmul(in0, in1, t0, t1)));
    MATH((llk_math_matmul<MF, MM_THROTTLE>(idst)));
}
// One k-slice of an rt x ct block of output tiles: in0's tiles t0 + r * kt
// (row-major, kt tiles a row) against in1's t1 + c, into DST idst + r * ct
// + c. With ct = 1 the in1 tile is unpacked once and reused down the rt
// rows: the operand traffic of rt + 1 tiles instead of 2 rt.
template <MathFidelity MF>
void mm_block_init(uint32_t in0, uint32_t in1, uint32_t transpose, uint32_t ct, uint32_t rt, uint32_t kt) {
    MATH((llk_math_matmul_init<MF, MM_THROTTLE>(in0, in1, transpose, ct, rt)));
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, transpose, ct, rt, kt)));
}
template <MathFidelity MF>
void mm_block(uint32_t in0, uint32_t in1, uint32_t t0, uint32_t t1, uint32_t idst, uint32_t ct, uint32_t rt, uint32_t kt) {
    UNPACK((llk_unpack_AB_matmul(in0, in1, t0, t1, ct, rt, kt)));
    MATH((llk_math_matmul<MF, MM_THROTTLE>(idst, ct, rt)));
}

// The packer's "round to a 10-bit mantissa" control: on, a Float32 pack
// lands the 19 bits the Src registers keep, rounded to nearest rather than
// truncated on the way back in. Off for everything the SFPU reads exactly.
void pack_rounding(const bool on) {
    PACK((cfg_reg_rmw_tensix<
          PCK_DEST_RD_CTRL_Round_10b_mant_ADDR32,
          PCK_DEST_RD_CTRL_Round_10b_mant_SHAMT,
          PCK_DEST_RD_CTRL_Round_10b_mant_MASK>(on ? 1u : 0u)));
}

// The unpacker waits for everything the pack thread has packed so far (see
// the backward kernel): before a dest-register transpose's neighbours, and
// before reading back what was just packed.
void unpacker_fence() {
    cb_reserve_back(cb_transpose_fence, 1);
    cb_push_back(cb_transpose_fence, 1);
    cb_wait_front(cb_transpose_fence, 1);
    cb_pop_front(cb_transpose_fence, 1);
}

// Lazy rescaling (FlashAttention-4's trick). The running maximum m is only a
// reference point: the finished row O = O^T / l and lse = a m + ln l come out
// the same for any m, as long as exp(a (S - m)) does not overflow. So a query
// tile whose block maximum stays within FW_LAZY_THRESHOLD (in units of the
// scaled scores, e^8 ~ 3000 in Float32 is nowhere near overflow) of the m it
// carries keeps that m: no new maximum, no r = exp(a (m_old - m_new)), no
// rescale of l, and the output products can be added onto O^T in L1 by the
// packer instead of being multiplied through DST. The check itself is the
// FPU's difference colmax - m (row 0 of the tiles in cb_block_max, formed in
// the maximum's own acquire) compared on the unpack thread, which owns the
// circular buffers' read pointers; the verdict reaches the other two threads
// through the mailboxes, so all three take the same branches. Positive
// Float32 numbers order like their bit patterns and negative ones are
// negative as integers, so the comparison is on the raw words.
#ifndef FW_EXP_GUARD
#define FW_EXP_GUARD 2  // see the exponential's steps
#endif
#ifndef FW_LAZY_THRESHOLD
#define FW_LAZY_THRESHOLD 8.0F
#endif
uint32_t lazy_need_mask() {
    uint32_t mask = 0u;
    UNPACK({
        constexpr float scale = __builtin_bit_cast(float, scaler_bits);
        constexpr float threshold = FW_LAZY_THRESHOLD / scale;
        constexpr int32_t threshold_bits = __builtin_bit_cast(int32_t, threshold);
        // The packer rewrote these words this timestep; the CB credit orders
        // them, the fence keeps a RISC data cache (off by default) from
        // serving last timestep's verdict.
        invalidate_l1_cache();
        for (uint32_t a = 0; a < Bt; ++a) {
            const uint32_t address = get_tile_l1_byte_address(get_operand_id(cb_block_max), a);
            const volatile int32_t* words = reinterpret_cast<const volatile int32_t*>(address);
            bool need = false;
            // Row 0 of a 32 x 32 tile: the first row of face 0 and of face 1.
#ifdef FW_EXPERIMENT_CHECK_ONE
            constexpr uint32_t kCheckWords = 1u;  // timing experiment: results wrong
#else
            constexpr uint32_t kCheckWords = 16u;
#endif
            for (uint32_t c = 0; c < kCheckWords; ++c) {
                need |= words[c] > threshold_bits;
                need |= words[256u + c] > threshold_bits;
            }
            mask |= (need ? 1u : 0u) << a;
        }
        mailbox_write(ckernel::ThreadId::MathThreadId, mask);
        mailbox_write(ckernel::ThreadId::PackThreadId, mask);
    })
    MATH(mask = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    PACK(mask = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
    return mask;
}

// V_j^T from the resident V_j: Bt x vWt bf16 tiles in, vWt x Bt out, each
// tile transposed by the unpacker on the way into DST. Once per residency
// interval; it is what lets O^T be a plain matmul with P^T as its second operand.
void transpose_value_block() {
    cb_wait_front(cb_value, Bt * vWt);
    cb_reserve_back(cb_value_t, Bt * vWt);
    pack_reconfig_data_format(cb_value_t);
    reconfig_data_format_srca(cb_value);
    transpose_init(cb_value);
    for (uint32_t e = 0; e < vWt; ++e) {
        for (uint32_t b = 0; b < Bt; ++b) {
            tile_regs_acquire();
            transpose_tile(cb_value, b * vWt + e, /* register idx */ 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(/* register idx */ 0, cb_value_t);  // tile e * Bt + b
            tile_regs_release();
        }
    }
    cb_push_back(cb_value_t, Bt * vWt);
    cb_wait_front(cb_value_t, Bt * vWt);
}

// Broadcast row 0 of a statistic tile down the rows of a DST tile, through
// SrcB at 19 bits (the backward's broadcast_statistic_rows_to_dst): the
// statistic's other rows, whatever the reduction left there, are not read.
// The source must be a plain (not unpack-to-dest) buffer: the 32-bit unary
// broadcast path does not broadcast (measured), the 19-bit one does.
void broadcast_row0_to_dst(const uint32_t idst, const uint32_t cb_statistics, const uint32_t stat_tile) {
    reconfig_data_format_srcb(cb_statistics);
    UNPACK((llk_unpack_A_init<BroadcastType::ROW, false, EltwiseBinaryReuseDestType::NONE, false>(
        false, false, cb_statistics)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          ckernel::DataCopyType::B2D,
          DST_ACCUM_MODE,
          BroadcastType::ROW>(cb_statistics)));
    unary_bcast<BroadcastType::ROW>(cb_statistics, stat_tile, idst);
}



#if defined(TRISC_MATH)
// The backward kernel's exponential, on the math thread: 2^z with z = a x / ln 2
// + bias as a two-chain polynomial in raw SFPU instructions, relative error
// within 2.9e-6. (The generic exp_tile measured 4.3 of a launch's 10 ms; the
// 21-bit exponential sdpa_fw uses is biased by ~1e-3 and, compounded through
// the rescale factors of 32 hops, put the lse off by 3e-2.) The backward runs
// it on the pack thread; here every SFPU operation is on the math thread, so
// the LLK's own per-tile framing (dst address, the four faces, clear) wraps
// the face body. The programmable constants are shared with the sfpi-compiled
// operations (max, sub, mul, reciprocal, log), which program what they need
// in their inits; ours are (re)programmed in exp_prepare.
// See the backward kernel for the derivation of every step.
namespace math_sfpu {

// 2^f for f in [0, 1) as 1 + c1 f + c2 f^2 + c3 f^3 (minimax, 9.5e-5
// relative: under the 19 bits the probabilities are stored at).
constexpr uint32_t kExpC1 = 0x3F31F01Eu;
constexpr uint32_t kExpC2 = 0x3E691DD8u;
constexpr uint32_t kExpC3 = 0x3D9DFC59u;

constexpr uint32_t kBiasReg = p_sfpu::LREG6;
constexpr uint32_t kC1Reg = p_sfpu::LREG7;
constexpr uint32_t kScaleReg = p_sfpu::LREG12;
constexpr uint32_t kC2Reg = p_sfpu::LREG13;
constexpr uint32_t kC3Reg = p_sfpu::LREG14;

constexpr uint32_t kMadNegateVa = 1u;
constexpr uint32_t kSetExpFromInt = 0u;
constexpr uint32_t kCastIntToFloat = 0u;

inline void load_constant(const uint32_t reg, const uint32_t bits) {
    TTI_SFPLOADI(reg, sfpi::SFPLOADI_MOD0_UPPER, static_cast<uint16_t>(bits >> 16));
    TTI_SFPLOADI(reg, sfpi::SFPLOADI_MOD0_LOWER, static_cast<uint16_t>(bits & 0xFFFFu));
}

inline void program_constant(const uint32_t reg, const uint32_t bits) {
    load_constant(p_sfpu::LREG0, bits);
    TTI_SFPCONFIG(0, reg, 0);
}

inline void init() {
    ckernel::sfpu::_init_sfpu_config_reg();
}

// The address modes and every constant the exponential relies on, reloaded
// before every run of tiles: the other SFPU operations program these too.
inline void exp_prepare() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
    constexpr float exp_scale = __builtin_bit_cast(float, scaler_bits);
    constexpr uint32_t inv_ln2_bits = __builtin_bit_cast(uint32_t, exp_scale * 1.4426950408889634F);
    program_constant(kScaleReg, inv_ln2_bits);
    program_constant(kC2Reg, kExpC2);
    program_constant(kC3Reg, kExpC3);
    load_constant(kBiasReg, exp_bias_bits);
    load_constant(kC1Reg, kExpC1);
}

#if FW_EXP_GUARD == 2
// The argument clamped at the bias (exp(a x) for a x < -88 is 0 as 2^-127
// flushes) by one max against the zero constant, whose min half the
// hardware drops.
#define FW_EXP_STEP(step, x, i, f, column)                                                                 \
    if constexpr (step == 0) TTI_SFPLOAD(x, InstrModLoadStore::DEFAULT, ADDR_MOD_7, column);               \
    if constexpr (step == 1) TTI_SFPMAD(x, kScaleReg, kBiasReg, x, 0);                                      \
    if constexpr (step == 2) TTI_SFPSWAP(0, p_sfpu::LCONST_0, x, sfpi::SFPSWAP_MOD1_VEC_MAX_MIN);           \
    if constexpr (step == 3)                                                                                 \
        TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_ZERO, 0, x, x, i, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT16);     \
    if constexpr (step == 4) TTI_SFPCAST(i, f, kCastIntToFloat);                                            \
    if constexpr (step == 5) TTI_SFPMAD(f, p_sfpu::LCONST_1, x, f, kMadNegateVa);                           \
    if constexpr (step == 6) TTI_SFPMAD(f, kC3Reg, kC2Reg, x, 0);                                           \
    if constexpr (step == 7) TTI_SFPMAD(x, f, kC1Reg, x, 0);                                                \
    if constexpr (step == 8) TTI_SFPMAD(x, f, p_sfpu::LCONST_1, x, 0);                                      \
    if constexpr (step == 9) TTI_SFPSETEXP(0, x, i, kSetExpFromInt);                                        \
    if constexpr (step == 10 && column == 0) TTI_SFPSTORE(i, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);    \
    if constexpr (step == 10 && column != 0) TTI_SFPSTORE(i, InstrModLoadStore::DEFAULT, ADDR_MOD_6, column);
constexpr uint32_t kExpSteps = 11;
#elif FW_EXP_GUARD == 1
// The backward's guard: a negative argument's mask, and the integer part
// masked to zero.
#define FW_EXP_STEP(step, x, i, f, column)                                                                 \
    if constexpr (step == 0) TTI_SFPLOAD(x, InstrModLoadStore::DEFAULT, ADDR_MOD_7, column);               \
    if constexpr (step == 1) TTI_SFPMAD(x, kScaleReg, kBiasReg, x, 0);                                      \
    if constexpr (step == 2)                                                                                 \
        TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_ZERO, 0, x, x, i, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT16);     \
    if constexpr (step == 3) TTI_SFPCAST(i, f, kCastIntToFloat);                                            \
    if constexpr (step == 4) TTI_SFPMAD(f, p_sfpu::LCONST_1, x, f, kMadNegateVa);                           \
    if constexpr (step == 5) TTI_SFPGT(0, p_sfpu::LCONST_0, x, 8);                                          \
    if constexpr (step == 6) TTI_SFPAND(0, x, i, 0);                                                         \
    if constexpr (step == 7) TTI_SFPMAD(f, kC3Reg, kC2Reg, x, 0);                                           \
    if constexpr (step == 8) TTI_SFPMAD(x, f, kC1Reg, x, 0);                                                \
    if constexpr (step == 9) TTI_SFPMAD(x, f, p_sfpu::LCONST_1, x, 0);                                      \
    if constexpr (step == 10) TTI_SFPSETEXP(0, x, i, kSetExpFromInt);                                       \
    if constexpr (step == 11 && column == 0) TTI_SFPSTORE(i, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);    \
    if constexpr (step == 11 && column != 0) TTI_SFPSTORE(i, InstrModLoadStore::DEFAULT, ADDR_MOD_6, column);
constexpr uint32_t kExpSteps = 12;
#else
// No guard: a timing experiment, wrong wherever a x < -88 (every masked score).
#define FW_EXP_STEP(step, x, i, f, column)                                                                 \
    if constexpr (step == 0) TTI_SFPLOAD(x, InstrModLoadStore::DEFAULT, ADDR_MOD_7, column);               \
    if constexpr (step == 1) TTI_SFPMAD(x, kScaleReg, kBiasReg, x, 0);                                      \
    if constexpr (step == 2)                                                                                 \
        TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_ZERO, 0, x, x, i, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT16);     \
    if constexpr (step == 3) TTI_SFPCAST(i, f, kCastIntToFloat);                                            \
    if constexpr (step == 4) TTI_SFPMAD(f, p_sfpu::LCONST_1, x, f, kMadNegateVa);                           \
    if constexpr (step == 5) TTI_SFPMAD(f, kC3Reg, kC2Reg, x, 0);                                           \
    if constexpr (step == 6) TTI_SFPMAD(x, f, kC1Reg, x, 0);                                                \
    if constexpr (step == 7) TTI_SFPMAD(x, f, p_sfpu::LCONST_1, x, 0);                                      \
    if constexpr (step == 8) TTI_SFPSETEXP(0, x, i, kSetExpFromInt);                                        \
    if constexpr (step == 9 && column == 0) TTI_SFPSTORE(i, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);     \
    if constexpr (step == 9 && column != 0) TTI_SFPSTORE(i, InstrModLoadStore::DEFAULT, ADDR_MOD_6, column);
constexpr uint32_t kExpSteps = 10;
#endif

template <uint32_t step>
inline void exp_pair_step() {
    FW_EXP_STEP(step, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, 0);
    FW_EXP_STEP(step, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG5, 2);
}

template <uint32_t step = 0>
inline void exp_pair_body() {
    exp_pair_step<step>();
    if constexpr (step + 1 < kExpSteps) {
        exp_pair_body<step + 1>();
    }
}

// A face is 16 x 16: four groups of four rows, each two vectors; the two
// chains interleaved instruction by instruction.
inline void exp_face() {
    constexpr int kBodyLen = 2 * kExpSteps;
    TTI_REPLAY(0, kBodyLen, 1, 1);
    exp_pair_body();
#pragma GCC unroll 4
    for (uint32_t i = 1; i < 4u; ++i) {
        TTI_REPLAY(0, kBodyLen, 0, 0);
    }
}

// One tile: the LLK's framing sets the dst address, runs the body on each of
// the four faces, and clears the address.
inline void exp_tile(const uint32_t tile) {
    _llk_math_eltwise_unary_sfpu_params_(exp_face, tile, VectorMode::RC);
}

}  // namespace math_sfpu
#endif

#if defined(TRISC_PACK)
// The same exponential on the pack thread, for the probabilities: the bulk
// of the SFPU work, run on the committed half of the destination file while
// the math thread forms the next column's S^T - m, as the backward and
// ttnn's streaming kernel do. It never runs at the same time as a
// math-thread SFPU operation: the math thread's next SFPU work (the sum's
// rescale) starts after the probabilities are pushed, which is after every
// exponential here. (Every other SFPU operation stays on the math thread:
// they consume unpack-to-dest copies, which the pack thread's SFPU read
// before the unpacker had finished writing them -- measured as a moving
// 3e-3 to 2e-2 error on O.)
namespace pack_sfpu {

// tile_regs_wait gates only the packer on the math thread's commit; the
// vector unit's loads need their own gate on the same semaphore.
inline void wait_for_math_done() {
    TTI_SEMWAIT(p_stall::STALL_SFPU, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
}

// The packer waits for the vector unit's last store before reading a tile.
inline void wait_before_pack() {
    TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
}


// 2^f for f in [0, 1) as 1 + c1 f + c2 f^2 + c3 f^3 (minimax, 9.5e-5
// relative: under the 19 bits the probabilities are stored at).
constexpr uint32_t kExpC1 = 0x3F31F01Eu;
constexpr uint32_t kExpC2 = 0x3E691DD8u;
constexpr uint32_t kExpC3 = 0x3D9DFC59u;

constexpr uint32_t kBiasReg = p_sfpu::LREG6;
constexpr uint32_t kC1Reg = p_sfpu::LREG7;
constexpr uint32_t kScaleReg = p_sfpu::LREG12;
constexpr uint32_t kC2Reg = p_sfpu::LREG13;
constexpr uint32_t kC3Reg = p_sfpu::LREG14;

constexpr uint32_t kMadNegateVa = 1u;
constexpr uint32_t kSetExpFromInt = 0u;
constexpr uint32_t kCastIntToFloat = 0u;

inline void load_constant(const uint32_t reg, const uint32_t bits) {
    TTI_SFPLOADI(reg, sfpi::SFPLOADI_MOD0_UPPER, static_cast<uint16_t>(bits >> 16));
    TTI_SFPLOADI(reg, sfpi::SFPLOADI_MOD0_LOWER, static_cast<uint16_t>(bits & 0xFFFFu));
}

inline void program_constant(const uint32_t reg, const uint32_t bits) {
    load_constant(p_sfpu::LREG0, bits);
    TTI_SFPCONFIG(0, reg, 0);
}

inline void init() {
    ckernel::sfpu::_init_sfpu_config_reg();
}

// The address modes and every constant the exponential relies on, reloaded
// before every run of tiles: the other SFPU operations program these too.
inline void exp_prepare() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
    constexpr float exp_scale = __builtin_bit_cast(float, scaler_bits);
    constexpr uint32_t inv_ln2_bits = __builtin_bit_cast(uint32_t, exp_scale * 1.4426950408889634F);
    program_constant(kScaleReg, inv_ln2_bits);
    program_constant(kC2Reg, kExpC2);
    program_constant(kC3Reg, kExpC3);
    load_constant(kBiasReg, exp_bias_bits);
    load_constant(kC1Reg, kExpC1);
}

#if FW_EXP_GUARD == 2
// The argument clamped at the bias (exp(a x) for a x < -88 is 0 as 2^-127
// flushes) by one max against the zero constant, whose min half the
// hardware drops.
#define FW_EXP_STEP_P(step, x, i, f, column)                                                                 \
    if constexpr (step == 0) TTI_SFPLOAD(x, InstrModLoadStore::DEFAULT, ADDR_MOD_7, column);               \
    if constexpr (step == 1) TTI_SFPMAD(x, kScaleReg, kBiasReg, x, 0);                                      \
    if constexpr (step == 2) TTI_SFPSWAP(0, p_sfpu::LCONST_0, x, sfpi::SFPSWAP_MOD1_VEC_MAX_MIN);           \
    if constexpr (step == 3)                                                                                 \
        TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_ZERO, 0, x, x, i, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT16);     \
    if constexpr (step == 4) TTI_SFPCAST(i, f, kCastIntToFloat);                                            \
    if constexpr (step == 5) TTI_SFPMAD(f, p_sfpu::LCONST_1, x, f, kMadNegateVa);                           \
    if constexpr (step == 6) TTI_SFPMAD(f, kC3Reg, kC2Reg, x, 0);                                           \
    if constexpr (step == 7) TTI_SFPMAD(x, f, kC1Reg, x, 0);                                                \
    if constexpr (step == 8) TTI_SFPMAD(x, f, p_sfpu::LCONST_1, x, 0);                                      \
    if constexpr (step == 9) TTI_SFPSETEXP(0, x, i, kSetExpFromInt);                                        \
    if constexpr (step == 10 && column == 0) TTI_SFPSTORE(i, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);    \
    if constexpr (step == 10 && column != 0) TTI_SFPSTORE(i, InstrModLoadStore::DEFAULT, ADDR_MOD_6, column);
constexpr uint32_t kExpSteps = 11;
#elif FW_EXP_GUARD == 1
// The backward's guard: a negative argument's mask, and the integer part
// masked to zero.
#define FW_EXP_STEP_P(step, x, i, f, column)                                                                 \
    if constexpr (step == 0) TTI_SFPLOAD(x, InstrModLoadStore::DEFAULT, ADDR_MOD_7, column);               \
    if constexpr (step == 1) TTI_SFPMAD(x, kScaleReg, kBiasReg, x, 0);                                      \
    if constexpr (step == 2)                                                                                 \
        TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_ZERO, 0, x, x, i, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT16);     \
    if constexpr (step == 3) TTI_SFPCAST(i, f, kCastIntToFloat);                                            \
    if constexpr (step == 4) TTI_SFPMAD(f, p_sfpu::LCONST_1, x, f, kMadNegateVa);                           \
    if constexpr (step == 5) TTI_SFPGT(0, p_sfpu::LCONST_0, x, 8);                                          \
    if constexpr (step == 6) TTI_SFPAND(0, x, i, 0);                                                         \
    if constexpr (step == 7) TTI_SFPMAD(f, kC3Reg, kC2Reg, x, 0);                                           \
    if constexpr (step == 8) TTI_SFPMAD(x, f, kC1Reg, x, 0);                                                \
    if constexpr (step == 9) TTI_SFPMAD(x, f, p_sfpu::LCONST_1, x, 0);                                      \
    if constexpr (step == 10) TTI_SFPSETEXP(0, x, i, kSetExpFromInt);                                       \
    if constexpr (step == 11 && column == 0) TTI_SFPSTORE(i, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);    \
    if constexpr (step == 11 && column != 0) TTI_SFPSTORE(i, InstrModLoadStore::DEFAULT, ADDR_MOD_6, column);
constexpr uint32_t kExpSteps = 12;
#else
// No guard: a timing experiment, wrong wherever a x < -88 (every masked score).
#define FW_EXP_STEP_P(step, x, i, f, column)                                                                 \
    if constexpr (step == 0) TTI_SFPLOAD(x, InstrModLoadStore::DEFAULT, ADDR_MOD_7, column);               \
    if constexpr (step == 1) TTI_SFPMAD(x, kScaleReg, kBiasReg, x, 0);                                      \
    if constexpr (step == 2)                                                                                 \
        TTI_SFP_STOCH_RND(sfpi::SFPSTOCHRND_RND_ZERO, 0, x, x, i, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT16);     \
    if constexpr (step == 3) TTI_SFPCAST(i, f, kCastIntToFloat);                                            \
    if constexpr (step == 4) TTI_SFPMAD(f, p_sfpu::LCONST_1, x, f, kMadNegateVa);                           \
    if constexpr (step == 5) TTI_SFPMAD(f, kC3Reg, kC2Reg, x, 0);                                           \
    if constexpr (step == 6) TTI_SFPMAD(x, f, kC1Reg, x, 0);                                                \
    if constexpr (step == 7) TTI_SFPMAD(x, f, p_sfpu::LCONST_1, x, 0);                                      \
    if constexpr (step == 8) TTI_SFPSETEXP(0, x, i, kSetExpFromInt);                                        \
    if constexpr (step == 9 && column == 0) TTI_SFPSTORE(i, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);     \
    if constexpr (step == 9 && column != 0) TTI_SFPSTORE(i, InstrModLoadStore::DEFAULT, ADDR_MOD_6, column);
constexpr uint32_t kExpSteps = 10;
#endif

template <uint32_t step>
inline void exp_pair_step() {
    FW_EXP_STEP_P(step, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, 0);
    FW_EXP_STEP_P(step, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG5, 2);
}

template <uint32_t step = 0>
inline void exp_pair_body() {
    exp_pair_step<step>();
    if constexpr (step + 1 < kExpSteps) {
        exp_pair_body<step + 1>();
    }
}

// A face is 16 x 16: four groups of four rows, each two vectors; the two
// chains interleaved instruction by instruction. The body sits in the
// thread's replay buffer, recorded once per run of tiles (exp_record,
// below) rather than re-pushed on every face: 4 replay words a face
// instead of 4 + the body.
constexpr int kExpBodyLen = 2 * kExpSteps;

inline void exp_record() {
#ifdef FW_EXPERIMENT_REPLAY_PER_FACE
#else
    TTI_REPLAY(0, kExpBodyLen, /* execute_while_loading */ 0, /* load_mode */ 1);
    exp_pair_body();
#endif
}

inline void exp_face() {
#ifdef FW_EXPERIMENT_REPLAY_PER_FACE
    TTI_REPLAY(0, kExpBodyLen, 1, 1);
    exp_pair_body();
#pragma GCC unroll 4
    for (uint32_t i = 1; i < 4u; ++i) {
        TTI_REPLAY(0, kExpBodyLen, 0, 0);
    }
#else
#pragma GCC unroll 4
    for (uint32_t i = 0; i < 4u; ++i) {
        TTI_REPLAY(0, kExpBodyLen, 0, 0);
    }
#endif
}

// One tile, framed as the backward does it: the dst address set to the tile
// (per thread, through SETC16), the body on each of the four faces, the
// address cleared. Not the LLK's framing, whose start issues a
// STALLWAIT(STALL_SFPU, MATH) per tile: the matrix unit is the math
// thread's and busy under this exponential by design, so that wait is at
// best a wasted slot and at worst serialises the two threads.
inline void set_dst(const uint32_t tile) {
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, (tile << 6) + get_dest_buffer_base());
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

inline void next_face() {
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
}

inline void clear_dst() {
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

inline void exp_tile(const uint32_t tile) {
#ifdef FW_EXPERIMENT_LLK_EXP_FRAMING
    _llk_math_eltwise_unary_sfpu_params_(exp_face, tile, VectorMode::RC);
#else
    set_dst(tile);
    for (uint32_t face = 0; face < 4u; ++face) {
        exp_face();
        next_face();
    }
    clear_dst();
#endif
}

}  // namespace pack_sfpu
#endif

// exp(a x) in place on a DST tile, on the math thread's SFPU (see math_sfpu).
// FW_EXPERIMENT_* are timing experiments only (the results are wrong).
constexpr uint32_t scaler_bf16_bits = scaler_bits >> 16;  // exact: the scale is a power of two
void exp_scaled(const uint32_t idst) {
#ifdef FW_EXPERIMENT_NO_EXP
    (void)idst;
#elif defined(FW_EXPERIMENT_GENERIC_EXP)
    binop_with_scalar_tile_init();
    mul_unary_tile(idst, scaler_bits);
    exp_tile_init</* approx */ false>();
    exp_tile</* approx */ false>(idst);
#elif defined(FW_EXPERIMENT_SDPA_EXP)
    sdpa_exp_tile_scaled<scaler_bits, scaler_bf16_bits>(idst);
#else
    MATH((math_sfpu::exp_prepare()));
    MATH((math_sfpu::exp_tile(idst)));
#endif
}



}  // namespace

// Under NoC event tracing the zones below are noise, and at a whole launch
// they overflow the per-core marker buffer, which breaks the trace's zone
// pairing; compiled out there, unchanged in an ordinary profiling build.
#if defined(PROFILE_NOC_EVENTS)
#undef DeviceZoneScopedN
#define DeviceZoneScopedN(name)
#endif

void kernel_main() {
    const uint32_t my_core = get_arg_val<uint32_t>(0);
    const uint32_t slice_count = get_arg_val<uint32_t>(1);

    using ttml::metal::ops::cyclic_sdpa_bw::CyclicSchedule;
    using ttml::metal::ops::cyclic_sdpa_bw::kNoCore;
    constexpr CyclicSchedule sched(kCores, kMaskMode);
    constexpr uint32_t kTimesteps = sched.num_timesteps();

    compute_kernel_hw_startup(cb_query, cb_key, cb_scores);
    MATH((math_sfpu::init()));
    copy_init(cb_query);
    matmul_init(cb_key, cb_query);
    cb_wait_front(cb_attn_mask, 2);
    cb_wait_front(cb_ones_column, 1);
    cb_wait_front(cb_ones_row, 1);
    cb_wait_front(cb_reduce_scaler, 1);

    for (uint32_t s = 0; s < slice_count; ++s) {
    for (uint32_t t = 0; t < kTimesteps; ++t) {
        DeviceZoneScopedN("T-STEP");
        const auto pair = sched.pair(my_core, t);
        const uint32_t g = s * kTimesteps + t;
        const bool diagonal = (DENSE_MODE == 0) && (pair.i == pair.j);
        const bool skip_masked = (Bt > 1u) && diagonal;
        // Where the row stands: a first streak start has no state to seed
        // from; the row's last visit finishes it.
        const bool fresh = !sched.producer(my_core, t).internal && !sched.is_later_streak_start(pair.i, t);
        const bool final = sched.next_consumer(pair.i, t) == kNoCore && !sched.has_later_active(pair.i, t);
        // Live key tiles of query tile a on a diagonal pair: b <= a.
        const auto n_live = [&](uint32_t a) { return skip_masked ? a + 1u : Bt; };

        // ---- column state
        const bool column_changed = (t == 0u) || (sched.pair(my_core, t - 1u).j != pair.j);
        if (column_changed && g > 0u) {
            cb_pop_front(cb_key, Bt * qWt);
            cb_pop_front(cb_value, Bt * vWt);
            cb_pop_front(cb_value_t, Bt * vWt);
        }
        if (column_changed) {
            DeviceZoneScopedN("COL-CHANGE");
            transpose_value_block();
        }
        {
            DeviceZoneScopedN("WAIT-PACKET");
            // Q and the column only: the scores and the block maximum need
            // nothing of the state, which the previous consumer is still
            // finishing while this core computes them.
            cb_wait_front(cb_query, Bt * qWt);
            cb_wait_front(cb_key, Bt * qWt);
        }
        cb_reserve_back(cb_max_out, Bt);
        cb_reserve_back(cb_sum_out, Bt);
        cb_reserve_back(cb_out_out, Bt * qWt);

        // ---- 1. S^T = K Q^T, a column of the score grid per acquire (half
        // the destination file, so the math thread's next column overlaps
        // the pack thread's packs of this one): the block matmul, the query
        // tile unpacked once per k and reused down the live key tiles;
        // packed rounded to the 19 bits the FPU reads back.
        {
            DeviceZoneScopedN("SCORES");
            cb_reserve_back(cb_scores, score_tiles);
            pack_reconfig_data_format(cb_scores);
            pack_rounding(true);
            for (uint32_t a = 0; a < Bt; ++a) {
                const uint32_t live = n_live(a);
                tile_regs_acquire();
                reconfig_data_format(cb_query, cb_key);
#ifdef FW_EXPERIMENT_NO_TRANSPOSE
                constexpr uint32_t kTransposeQ = 0;  // timing experiment: K Q instead of K Q^T, results wrong
#else
                constexpr uint32_t kTransposeQ = 1;
#endif
                mm_block_init<kFidS>(cb_key, cb_query, kTransposeQ, /* ct */ 1, /* rt */ live, /* kt */ qWt);
                for (uint32_t k = 0; k < qWt; ++k) {
                    mm_block<kFidS>(cb_key, cb_query, k, a * qWt + k, 0, /* ct */ 1, /* rt */ live, /* kt */ qWt);
                }
                if (diagonal) {
                    // The triangle, -inf where the key index exceeds the query index.
                    reconfig_data_format(cb_attn_mask, cb_zero_tile);
                    add_init(cb_attn_mask, cb_zero_tile, /* acc_to_dest */ true);
                    add_tiles(cb_attn_mask, cb_zero_tile, kMaskTriangle, 0, a);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t b = 0; b < Bt; ++b) {
                    if (b >= live) {
                        break;
                    }
                    pack_tile</* out_of_order */ true>(b, cb_scores, b * Bt + a);
                }
                tile_regs_release();
            }
            pack_rounding(false);
            cb_push_back(cb_scores, score_tiles);
            cb_wait_front(cb_scores, score_tiles);
        }

        // Which query tiles rescale this timestep (bit a): all of a fresh
        // row's, else those the check below says (see lazy_need_mask).
        uint32_t need_mask = (1u << Bt) - 1u;

        // ---- 2. Per query tile, the block maximum less the m the row
        // carries (the block maximum itself for a fresh row): the reductions
        // into DST -- all Bt tiles in one acquire, before the state is needed
        // -- then the FPU's difference with the DST tile through SrcA and m
        // from L1 through SrcB; row layout, to scratch. The unpack thread
        // then reads the verdict off row 0.
#ifndef FW_EXPERIMENT_NO_STATS
        {
            DeviceZoneScopedN("MAX");
            static_assert(Bt <= 4u, "the block maxima of a timestep share one half of the destination file");
            cb_reserve_back(cb_block_max, Bt);
            tile_regs_acquire();
            // One init for the Bt reductions: nothing between them re-inits.
            reconfig_data_format(cb_scores, cb_reduce_scaler);
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_scores, cb_reduce_scaler, cb_block_max);
            for (uint32_t a = 0; a < Bt; ++a) {
                const uint32_t live = n_live(a);
                for (uint32_t b = 0; b < Bt; ++b) {
                    if (b >= live) {
                        break;
                    }
                    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_scores, cb_reduce_scaler, b * Bt + a, 0, a);
                }
            }
            reduce_uninit();
            {
                DeviceZoneScopedN("WAIT-STATE");
                cb_wait_front(cb_max_seed, Bt);
                cb_wait_front(cb_max_plain, Bt);
                cb_wait_front(cb_sum_seed, Bt);
                cb_wait_front(cb_sum_plain, Bt);
                cb_wait_front(cb_out_seed, Bt * qWt);
            }
            if (!fresh) {
                reconfig_data_format(cb_scores, cb_max_plain);
                sub_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_max_plain);
                for (uint32_t a = 0; a < Bt; ++a) {
                    sub_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_max_plain, a, a);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_block_max);
            for (uint32_t a = 0; a < Bt; ++a) {
                pack_tile</* out_of_order */ true>(a, cb_block_max, a);
            }
            tile_regs_release();
            cb_push_back(cb_block_max, Bt);
            cb_wait_front(cb_block_max, Bt);
#ifndef FW_EXPERIMENT_NO_LAZY
            if (!fresh) {
                need_mask = lazy_need_mask();
            }
#endif
        }

        // ---- 2c. For the query tiles that rescale: with d = max(colmax - m_old, 0)
        // down every row, m_new = m_old + d (the maximum, to the difference's
        // 19 bits -- any reference works as long as P^T and r use the same
        // one, and they read this tile) and r = exp(-a d). A fresh row takes
        // the block maximum as its m.
        {
            DeviceZoneScopedN("RESCALE");
            constexpr uint32_t kDReg = 0, kNewReg = 1;
            if (!fresh) {
                cb_reserve_back(cb_rescale, Bt);
            }
            for (uint32_t a = 0; a < Bt; ++a) {
                if (((need_mask >> a) & 1u) == 0u) {
                    continue;  // m stays the reference it is; no r
                }
                tile_regs_acquire();
                broadcast_row0_to_dst(kDReg, cb_block_max, a);
                if (!fresh) {
                    relu_tile_init();
                    relu_tile(kDReg);
                    reconfig_data_format_srca(cb_scores, cb_max_seed);
                    copy_init(cb_max_seed);
                    copy_tile(cb_max_seed, a, kNewReg);  // m_old, exact
                    add_binary_tile_init();
                    add_binary_tile(kNewReg, kDReg, kNewReg);
                    negative_tile_init();
                    negative_tile(kDReg);
                    exp_scaled(kDReg);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_max_out);
                pack_tile</* out_of_order */ true>(fresh ? kDReg : kNewReg, cb_max_out, a);
                if (!fresh) {
                    pack_reconfig_data_format(cb_rescale);
                    pack_tile</* out_of_order */ true>(kDReg, cb_rescale, a);
                }
                tile_regs_release();
            }
            if (!fresh) {
                cb_push_back(cb_rescale, Bt);
                cb_wait_front(cb_rescale, Bt);
            }
            // m_new is read back below through the plain view of the same memory.
            unpacker_fence();
        }
#else
        cb_wait_front(cb_max_seed, Bt);
        cb_wait_front(cb_max_plain, Bt);
        cb_wait_front(cb_sum_seed, Bt);
        cb_wait_front(cb_sum_plain, Bt);
        cb_wait_front(cb_out_seed, Bt * qWt);
        cb_reserve_back(cb_block_max, Bt);
        cb_push_back(cb_block_max, Bt);
        cb_wait_front(cb_block_max, Bt);
        if (!fresh) {
            cb_reserve_back(cb_rescale, Bt);
            cb_push_back(cb_rescale, Bt);
            cb_wait_front(cb_rescale, Bt);
        }
#endif

        // ---- 3-5. Per query tile a: P^T(:, a) = exp(a (S^T - m)) -- the
        // FPU's row-broadcast subtraction from L1 (S^T at 19 bits, m's row 0
        // through SrcB) and the exponential on the pack thread -- then, one
        // query tile behind, l += colsum P^T and O^T += V^T P^T. The lag is
        // the overlap: while the pack thread exponentiates column a in one
        // half of the destination file, the math thread reduces and
        // multiplies column a - 1 in the other. P^T is stored a column at a
        // time (tile a * Bt + b) so a column's sums can start before the
        // next column's exponential ends.
        {
            DeviceZoneScopedN("PROBS-SUM-UPDATE");
            cb_reserve_back(cb_probs, score_tiles);

            const auto probs_column = [&](uint32_t a) {
                const uint32_t live = n_live(a);
                tile_regs_acquire();
                reconfig_data_format(cb_scores, cb_max_plain);
                bcast_init<EltwiseBinaryType::ELWSUB, BroadcastType::ROW>(cb_scores, cb_max_plain);
                for (uint32_t b = 0; b < Bt; ++b) {
                    if (b >= live) {
                        break;
                    }
                    sub_tiles_bcast_rows(cb_scores, cb_max_plain, b * Bt + a, a, b);
                }
                tile_regs_commit();
                tile_regs_wait();
#ifndef FW_EXPERIMENT_NO_EXP
                PACK((pack_sfpu::wait_for_math_done()));
                PACK((pack_sfpu::exp_prepare()));
                PACK((pack_sfpu::exp_record()));
                for (uint32_t b = 0; b < Bt; ++b) {
                    if (b >= live) {
                        break;
                    }
                    PACK((pack_sfpu::exp_tile(b)));
                }
                PACK((pack_sfpu::wait_before_pack()));
#endif
                // The column's tiles, relative to this push's write pointer.
                pack_reconfig_data_format(cb_probs);
                pack_rounding(true);
                for (uint32_t b = 0; b < Bt; ++b) {
                    if (b >= live) {
                        break;
                    }
                    pack_tile</* out_of_order */ true>(b, cb_probs, b);
                }
                // After the release: it drains the packer, so the rounding
                // bit cannot change under the last tiles' pack.
                tile_regs_release();
                pack_rounding(false);
                cb_push_back(cb_probs, Bt);
            };

            // colsum P^T(:, a) into register kSumReg (row 0; the other rows
            // hold the reduction's leftovers, never read).
            constexpr uint32_t kSumReg = 0;
            const auto block_sum = [&](uint32_t a, uint32_t live) {
                reconfig_data_format(cb_probs, cb_reduce_scaler);
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_probs, cb_reduce_scaler, cb_sum_out);
                for (uint32_t b = 0; b < Bt; ++b) {
                    if (b >= live) {
                        break;
                    }
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_probs, cb_reduce_scaler, a * Bt + b, 0, kSumReg);
                }
                reduce_uninit();
            };
            // V^T P^T(:, a) for output tiles k0 .. k0 + nk - 1 into registers
            // reg0 ..; V^T is the first operand (bf16, SrcB), P^T the second
            // (Float32, SrcA).
            const auto products = [&](uint32_t a, uint32_t k0, uint32_t nk, uint32_t reg0, uint32_t live) {
                reconfig_data_format(cb_probs, cb_value_t);
                mm_block_init<kFidO>(cb_value_t, cb_probs, /* transpose */ 0, /* ct */ 1, /* rt */ nk, /* kt */ Bt);
                for (uint32_t b = 0; b < Bt; ++b) {
                    if (b >= live) {
                        break;
                    }
                    // The P^T tile once, V^T's nk tiles of key tile b streamed.
                    mm_block<kFidO>(cb_value_t, cb_probs, k0 * Bt + b, a * Bt + b, reg0, /* ct */ 1, /* rt */ nk, /* kt */ Bt);
                }
            };

            // A rescaled query tile (exact path): l = r l_old + colsum, then
            // O^T two tiles per acquire with r in the register above them,
            // the products accumulated in DST. A tile that keeps its m (and a
            // fresh row): the sum and the first output tiles in one acquire,
            // the rest four per acquire, added onto l and O^T in L1 by the
            // packer (fresh rows written outright).
            constexpr uint32_t kRReg = kOutGroup;
            constexpr uint32_t kFirstGroup = (qWt > 3u) ? 3u : qWt;
            constexpr uint32_t kAccGroup = (qWt > 4u) ? 4u : qWt;
            // A forwarded row's state goes to the reader one query tile at a
            // time, as soon as the tile's l and O^T are final, so the reader
            // forwards it under the remaining columns instead of after them;
            // the packs above are then relative to the advanced write
            // pointers. A finished row (final) is repacked below in place, so
            // its state is handed over whole afterwards, at the slot's base.
            // Per tile only at d <= 64 (measured on the ring of eight: -8% on
            // the 4-head forward step at d = 64, +7.5% at d = 128). At d = 64
            // the whole state's forward outlasts the next timestep's scores
            // and maxima, so the early writes remove a wait; at d = 128 those
            // stages already cover the forward, and the early writes only
            // compete with the unpackers for L1 under the last columns.
            const bool whole = final || !kStatePerTile;
            const auto hand_over_tile = [&]() {
                if (!whole) {
                    cb_push_back(cb_max_out, 1);
                    cb_push_back(cb_sum_out, 1);
                    cb_push_back(cb_out_out, qWt);
                }
            };
            const auto sum_and_update = [&](uint32_t a) {
                const uint32_t live = n_live(a);
                const bool rescaled = ((need_mask >> a) & 1u) != 0u;
                cb_wait_front(cb_probs, (a + 1u) * Bt);
                if (rescaled && !fresh) {
                    tile_regs_acquire();
                    block_sum(a, live);
                    reconfig_data_format_srca(cb_probs, cb_rescale);
                    copy_init(cb_rescale);
                    copy_tile(cb_rescale, a, 2);
                    reconfig_data_format_srca(cb_rescale, cb_sum_seed);
                    copy_init(cb_sum_seed);
                    copy_tile(cb_sum_seed, a, 1);
                    mul_binary_tile_init();
                    mul_binary_tile(1, 2, 1);
                    add_binary_tile_init();
                    add_binary_tile(kSumReg, 1, kSumReg);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_reconfig_data_format(cb_sum_out);
                    pack_tile</* out_of_order */ true>(kSumReg, cb_sum_out, whole ? a : 0u);
                    tile_regs_release();
                    for (uint32_t k0 = 0; k0 < qWt; k0 += kOutGroup) {
                        const uint32_t nk = (qWt - k0 < kOutGroup) ? qWt - k0 : kOutGroup;
                        tile_regs_acquire();
                        reconfig_data_format_srca(cb_probs, cb_rescale);
                        copy_init(cb_rescale);
                        copy_tile(cb_rescale, a, kRReg);
                        reconfig_data_format_srca(cb_rescale, cb_out_seed);
                        copy_init(cb_out_seed);
                        for (uint32_t i = 0; i < kOutGroup; ++i) {
                            if (i >= nk) {
                                break;
                            }
                            copy_tile(cb_out_seed, a * qWt + k0 + i, i);
                        }
                        mul_binary_tile_init();
                        for (uint32_t i = 0; i < kOutGroup; ++i) {
                            if (i >= nk) {
                                break;
                            }
                            mul_binary_tile(i, kRReg, i);
                        }
                        products(a, k0, nk, 0, live);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_reconfig_data_format(cb_out_out);
                        for (uint32_t i = 0; i < kOutGroup; ++i) {
                            if (i >= nk) {
                                break;
                            }
                            pack_tile</* out_of_order */ true>(i, cb_out_out, (whole ? a * qWt : 0u) + k0 + i);
                        }
                        tile_regs_release();
                    }
                    hand_over_tile();
                    return;
                }
                const bool accumulate = !fresh;
                constexpr uint32_t nk0 = kFirstGroup;
                tile_regs_acquire();
                block_sum(a, live);
                products(a, 0, nk0, kSumReg + 1, live);
                tile_regs_commit();
                tile_regs_wait();
                // (l and O^T share the Float32 format: one pack configuration.)
                pack_reconfig_data_format(cb_sum_out);
                if (accumulate) {
                    pack_reconfig_l1_acc(true);
                }
                pack_tile</* out_of_order */ true>(kSumReg, cb_sum_out, whole ? a : 0u);
                for (uint32_t i = 0; i < kFirstGroup; ++i) {
                    pack_tile</* out_of_order */ true>(kSumReg + 1 + i, cb_out_out, (whole ? a * qWt : 0u) + i);
                }
                if (accumulate) {
                    pack_reconfig_l1_acc(false);
                }
                tile_regs_release();
                for (uint32_t k0 = nk0; k0 < qWt; k0 += kAccGroup) {
                    const uint32_t nk = (qWt - k0 < kAccGroup) ? qWt - k0 : kAccGroup;
                    tile_regs_acquire();
                    products(a, k0, nk, 0, live);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_reconfig_data_format(cb_out_out);
                    if (accumulate) {
                        pack_reconfig_l1_acc(true);
                    }
                    for (uint32_t i = 0; i < kAccGroup; ++i) {
                        if (i >= nk) {
                            break;
                        }
                        pack_tile</* out_of_order */ true>(i, cb_out_out, (whole ? a * qWt : 0u) + k0 + i);
                    }
                    if (accumulate) {
                        pack_reconfig_l1_acc(false);
                    }
                    tile_regs_release();
                }
                hand_over_tile();
            };

            // The exact path copies the state into DST through the unpacker,
            // whose handshake with the math thread waits for an idle vector
            // unit; the pack thread's exponential, stalled on the next
            // column's commit, is not idle, and the two wait for each other
            // (a hang, met in training, where the maximum does grow). So a
            // timestep with any rescaled tile runs its columns in order:
            // each column's sums after its own exponential, none of them
            // under the next column's.
            // (One call site per stage: the kernel is near the config
            // buffer's size limit at d = 128.)
            const bool exact_any = !fresh && need_mask != 0u;
            const uint32_t lag = exact_any ? 0u : 1u;
            for (uint32_t a = 0; a < Bt + lag; ++a) {
                if (a < Bt) {
                    probs_column(a);
                }
                if (a >= lag) {
                    sum_and_update(a - lag);
                }
            }
        }

        // ---- 6. The finished row: O = (O^T / l)^T in bfloat16, lse = a m + ln l.
        //
        // A dest-register transpose clears the source registers when it is
        // done, so an operand the unpacker has already delivered for the next
        // copy is lost (see the backward kernel; measured here: the lse of
        // every query tile but the first came out without its m). The
        // transposes are therefore kept in loops of their own -- copy,
        // transpose, pack -- with the unpacker fenced after every tile, which
        // holds the next copy's unpack behind this tile's pack.
        if (final) {
            DeviceZoneScopedN("FINISH");
            // The sums and the accumulator packed above are read back through
            // the views of the same memory.
            unpacker_fence();
            // O^T / l, back onto the packet (dead after this); two output
            // tiles per acquire, 1/l in the register above them.
            {
                constexpr uint32_t kInvReg = kOutGroup;
                for (uint32_t a = 0; a < Bt; ++a) {
                    for (uint32_t k0 = 0; k0 < qWt; k0 += kOutGroup) {
                        const uint32_t nk = (qWt - k0 < kOutGroup) ? qWt - k0 : kOutGroup;
                        tile_regs_acquire();
                        broadcast_row0_to_dst(kInvReg, cb_sum_plain, a);
                        recip_tile_init</* legacy_compat */ false>();
                        recip_tile</* legacy_compat */ false>(kInvReg);
                        reconfig_data_format_srca(cb_scores, cb_out_seed);
                        copy_init(cb_out_seed);
                        for (uint32_t i = 0; i < kOutGroup; ++i) {
                            if (i >= nk) {
                                break;
                            }
                            copy_tile(cb_out_seed, a * qWt + k0 + i, i);
                        }
                        mul_binary_tile_init();
                        for (uint32_t i = 0; i < kOutGroup; ++i) {
                            if (i >= nk) {
                                break;
                            }
                            mul_binary_tile(i, kInvReg, i);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_reconfig_data_format(cb_out_out);
                        for (uint32_t i = 0; i < kOutGroup; ++i) {
                            if (i >= nk) {
                                break;
                            }
                            pack_tile</* out_of_order */ true>(i, cb_out_out, a * qWt + k0 + i);
                        }
                        tile_regs_release();
                    }
                }
            }
            unpacker_fence();
            // Every tile transposed within itself, into the bfloat16 output.
            cb_reserve_back(cb_output, Bt * qWt);
            pack_reconfig_data_format(cb_output);
            reconfig_data_format_srca(cb_out_seed, cb_out_seed);
            for (uint32_t t_idx = 0; t_idx < Bt * qWt; ++t_idx) {
                tile_regs_acquire();
                copy_init(cb_out_seed);
                copy_tile(cb_out_seed, t_idx, 0);
                transpose_dest_init</* is_32bit */ true>(cb_out_seed);
                transpose_dest</* is_32bit */ true>(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile</* out_of_order */ true>(0, cb_output, t_idx);
                tile_regs_release();
                unpacker_fence();
            }
            // lse = a m + ln l, row layout, kept to row 0, then transposed into
            // column layout: the value per row in column 0, zeros elsewhere.
            cb_reserve_back(cb_lse, Bt);
            constexpr uint32_t kLseReg = 0, kMaskReg = 1, kLogReg = 2;
            for (uint32_t a = 0; a < Bt; ++a) {
                tile_regs_acquire();
                // The conditional reconfig must name what SrcA actually holds:
                // the O^T tiles before the first query tile, the bfloat16 row
                // mask after every one (measured: naming the wrong one left
                // SrcA in bfloat16 and unpacked m as such).
                reconfig_data_format_srca(a == 0u ? cb_out_seed : cb_ones_row, cb_max_seed);
                copy_init(cb_max_seed);
                copy_tile(cb_max_seed, a, kLseReg);
                binop_with_scalar_tile_init();
                mul_unary_tile(kLseReg, scaler_bits);
                reconfig_data_format_srca(cb_max_seed, cb_sum_seed);
                copy_init(cb_sum_seed);
                copy_tile(cb_sum_seed, a, kLogReg);
                log_tile_init</* fast_and_approx */ false>();
                log_tile</* fast_and_approx */ false>(kLogReg);
                add_binary_tile_init();
                add_binary_tile(kLseReg, kLogReg, kLseReg);
                reconfig_data_format_srca(cb_sum_seed, cb_ones_row);
                copy_init(cb_ones_row);
                copy_tile(cb_ones_row, 0, kMaskReg);
                mask_tile_init();
                mask_tile(kLseReg, kMaskReg);
                transpose_dest_init</* is_32bit */ true>(cb_ones_row);
                transpose_dest</* is_32bit */ true>(kLseReg);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_lse);
                pack_tile</* out_of_order */ true>(kLseReg, cb_lse, a);
                tile_regs_release();
                unpacker_fence();
            }
            cb_push_back(cb_output, Bt * qWt);
            cb_push_back(cb_lse, Bt);
        }

        // ---- hand the updated state to the reader (a forwarded row's went
        // tile by tile above), pop the slot, release it.
        if (final || !kStatePerTile) {
            cb_push_back(cb_max_out, Bt);
            cb_push_back(cb_sum_out, Bt);
            cb_push_back(cb_out_out, Bt * qWt);
        }
        {
            DeviceZoneScopedN("T-POPS");
            cb_pop_front(cb_query, Bt * qWt);
            cb_pop_front(cb_max_seed, Bt);
            cb_pop_front(cb_max_plain, Bt);
            cb_pop_front(cb_sum_seed, Bt);
            cb_pop_front(cb_sum_plain, Bt);
            cb_pop_front(cb_out_seed, Bt * qWt);
            cb_pop_front(cb_scores, score_tiles);
            cb_pop_front(cb_probs, score_tiles);
            cb_pop_front(cb_block_max, Bt);
            if (!fresh) {
                cb_pop_front(cb_rescale, Bt);
            }
        }
        // The token is pushed by the pack thread while the slot's pops above
        // run on the unpack thread, and nothing orders the two; it is safe
        // because every unpack read of the slot precedes the last pack this
        // push follows. Do not add an unpack read of the slot after it.
        cb_reserve_back(cb_slot_release, 1);
        cb_push_back(cb_slot_release, 1);
    }
    }  // slices
}
