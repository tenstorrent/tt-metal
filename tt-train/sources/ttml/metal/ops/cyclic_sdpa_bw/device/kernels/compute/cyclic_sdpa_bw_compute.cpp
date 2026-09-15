// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The cyclic backward pass on one core: the schedule's T + 1 block pairs,
// each contributing to one row gradient and two column gradients.
//
// Every operand of a timestep arrives from DRAM and every result goes back to
// DRAM, including the column gradients. The paper keeps (K_j, V_j, dK_j,
// dV_j) resident on the owning core and changes columns twice per core, which
// is where its DRAM-traffic argument comes from; this step deliberately does
// not, and reloads them each timestep instead. The arithmetic and the
// schedule are unaffected, and the residency rules are validated separately
// by the transport probe -- which poisons the column pages in DRAM so that a
// first visit reading them is caught. Restoring residency belongs with the
// relay, which is where it pays.
//
// What that buys is a compute kernel with no state carried between
// timesteps: each timestep is the single block pair of
// cyclic_pair_compute.cpp, which is separately tested, wrapped in a loop.
// Every gradient uses the same three buffers -- a seed from the reader, an
// accumulator, and an output the writer takes -- because
//
//   * sdpa_bw's accumulate path packs without reserving, and only a
//     reserve/push cycle by *this* kernel leaves the packer's write pointer
//     where that expects; a push from the reader does not, since the write
//     pointers are per RISC. The copy from seed to accumulator is what
//     establishes it;
//   * a buffer the reader fills and this kernel accumulates onto would let
//     the writer's cb_wait_front be satisfied by the reader's push, and it
//     would write the value back unaccumulated. The copy out gives the writer
//     a buffer pushed exactly once per result.
//
// Only dQ needs the chip-wide barrier. Each column belongs to exactly one
// core, so dK and dV are never contended.

#include <api/compute/cb_api.h>
#include <api/compute/pack.h>
#include <api/compute/reconfig_data_format.h>
#include <api/compute/reg_api.h>
#include <hostdevcommon/kernel_structs.h>
#include <tensix.h>

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/mask.h"
#include "api/compute/matmul.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/transpose_dest.h"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"
#include "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/compute/sdpa_bw_compute_utils.hpp"
#ifdef TRISC_PACK
#include "ckernel_sfpu_binary.h"
#include "ckernel_sfpu_exp.h"
#endif

// COLUMN_RESIDENT: keep the whole column state across a residency interval
// instead of taking it fresh every timestep, which is the paper's
// column-state management. Each core changes its resident column exactly
// twice over T + 1 timesteps, always at a diagonal block, giving three
// intervals: its first column, its other column, then the first again.
//
// K_j and V_j are read once per interval, and this kernel releases the
// storage by popping at the change. dK_j and dV_j accumulate in L1 for the
// whole interval and are handed over once, at its end, for the write kernel
// to store. At an interval start they either begin from nothing -- a first
// visit, where the first update overwrites rather than accumulates, so the
// zeros never come from DRAM -- or from the value in DRAM, on the one
// revisit each core makes.
//
// Algorithm 2's reader still supplies the column per timestep, so the two
// have to agree on who pops and when, hence a switch rather than a change.
#ifndef COLUMN_RESIDENT
#define COLUMN_RESIDENT 0
#endif

#ifndef RELEASE_TOKEN
#define RELEASE_TOKEN 0
#endif

// FOLD_SCALE_INTO_KEY: where the softmax's 1/sqrt(d) lives.
//
// It appears twice in the arithmetic, on the scores and again on dS. Where
// a is a power of two (d = 64, 256) it goes into K, exactly, since K is
// bfloat16:
//
//     S  = Q (aK)^T = a Q K^T     the score path then needs no scale at all,
//     dQ = G (aK)   = a G K       with dS left unscaled,
//     dK = G^T Q                  short by a -- but dK is an accumulator, so
//                                 one scale at its handover fixes it,
//     dV = P^T dO                 untouched.
//
// K is resident for a whole residency interval, so scaling it costs one pass
// per interval, twice per core over the run.
//
// Where a is not a power of two (d = 128) it goes into the exponential and
// the statistic instead. The exponential computes exp(a x) with a folded
// into its 1/ln 2 constant, and the writer seeds the scores not with -L but
// with -(L + ln sqrt(d)) sqrt(d), so what comes out is
//
//     P' = exp(a S - L - ln sqrt(d)) = a P     dS' = P' (dP - D) = a dS  exact,
//     dQ = G' K, dK = G'^T Q                   both right as they stand,
//     dV = P'^T dO                             short by a -- an accumulator,
//                                              one scale at its handover.
//
// Either way the score pass itself is the same code: -L seeded by the FPU,
// no SFPU work but the exponential and the dS^T multiply, and one column
// gradient scaled once per residency interval. The host picks by checking
// whether a's mantissa is zero.
#ifndef FOLD_SCALE_INTO_KEY
#define FOLD_SCALE_INTO_KEY 0
#endif

// SEED_COLUMN_GRADIENTS: start every column's gradients from what is in DRAM,
// not only on a revisit. A first visit then reads dK_j, dV_j and adds to
// them rather than overwriting, so the op accumulates into its outputs.
//
// This is what a ring step wants: the caller passes its running accumulators
// as the outputs and the kernels add this step's contribution in place, which
// removes a zeroing copy and an add per gradient per step from the host --
// six dispatches a step, each costing more than the kernel does. dQ already
// works this way (it is seeded from DRAM at every streak start); this makes
// the column gradients match.
#ifndef SEED_COLUMN_GRADIENTS
#define SEED_COLUMN_GRADIENTS 0
#endif

// DENSE_MODE selects the unmasked schedule: every block pair is live, which
// is what a ring-attention step needs when the visiting key/value chunk is
// earlier in the sequence than the local query chunk. It changes the schedule
// (2T timesteps in two passes rather than T + 1) and, in the compute kernel,
// removes the intra-block mask; nothing else about the relay changes.
#ifndef DENSE_MODE
#define DENSE_MODE 0
#endif

#if DENSE_MODE
constexpr auto kMaskMode = ttml::metal::ops::cyclic_sdpa_bw::MaskMode::Dense;
#else
constexpr auto kMaskMode = ttml::metal::ops::cyclic_sdpa_bw::MaskMode::Causal;
#endif

// RELEASE_TOKEN: publish a token once this timestep's packet slot has been
// popped, so the relay reader knows the slot is free and can hand its
// producer the credit at the release -- the paper's timing -- instead of two
// timesteps later when it reserves the slot itself. The reader cannot see the
// pop any other way: the release happens on this RISC, and a circular
// buffer's acked count is not something the other side can read.

namespace {

constexpr uint32_t kCores = get_compile_time_arg_val(0);
constexpr uint32_t qWt = get_compile_time_arg_val(1);
constexpr uint32_t vWt = get_compile_time_arg_val(2);
constexpr uint32_t scaler_bits = get_compile_time_arg_val(3);
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(4);
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(5);
constexpr uint32_t block_size = get_compile_time_arg_val(6);
// Row-tiles per block: B = Bt * 32. What it buys is in the block-pair
// kernel's commit message and in docs/overlaps.md -- at Bt = 1 the score
// stages are one tile each and the matmul pipeline latency has nothing to
// hide behind.
constexpr uint32_t Bt = get_compile_time_arg_val(7);
// sqrt(d): the exponential applies the softmax scale to its whole argument,
// so the statistic being subtracted is divided by it first -- once per row of
// score tiles instead of a scale pass on each of the Bt tiles.
constexpr uint32_t inv_scaler_bits = get_compile_time_arg_val(8);
constexpr uint32_t score_tiles = Bt * Bt;

// The transposed orientation.
//
// Everything here is computed transposed: S^T = K Q^T, P^T = exp(S^T - L^T),
// dP^T = V dO^T, dS^T = P^T (dP^T - D^T), and the gradients as
//
//     dV_j  += P^T  dO_i        dK_j += dS^T Q_i        dQ_i^T += K_j^T dS^T
//
// so that the two operands the column gradients need, P^T and dS^T, are what
// the score pass produces, and no score tile is ever transposed. The matmul
// can transpose its second operand for free, which gives S^T and dP^T
// directly from the untransposed Q and dO. What has to be transposed instead
// is small and rare: K^T once per residency interval (a bf16 block, through
// the unpacker), and dQ at the ends of a streak, where it enters or leaves
// the relay in DRAM's untransposed layout -- the packet carries dQ^T between
// consumers. The statistics L and D are needed broadcast along rows of S^T,
// which is a column broadcast in the old orientation and a row broadcast
// here; the reader gathers them into row 0 of a tile for that.
//
// The whole of S^T, P^T, dP^T and dS^T for a column of the score grid is
// formed inside the DST registers and packed once: P^T for the dV matmul and
// dS^T for the dK and dQ matmuls. Nothing Float32 is ever copied from L1
// back into DST, so no buffer needs the unpack-to-dest mode -- which matters,
// because a buffer in that mode cannot also be read by a matmul.
//
// Score tiles are indexed (b, a) = (key tile, query tile), row-major, tile
// b * Bt + a. A query tile a is a *column* of this grid, so the pass loops
// over a on the outside and packs the column out of order into its row-major
// place.

// DST for one group of the score grid: a query tile a against kGroup key
// tiles b, with P^T in registers 0..kGroup-1 and dS^T in kGroup..2kGroup-1.
// A group is at most two key tiles, so a group is at most four Float32
// registers -- half the file -- and the math and pack threads can work on
// alternate halves: the math thread's matmuls for one group run while the
// pack thread's exponentials, multiplies and packs finish the previous one.
constexpr uint32_t kGroup = (Bt > 2u) ? 2u : Bt;
constexpr uint32_t kGroups = Bt / kGroup;  // per query tile
constexpr uint32_t score_reg(uint32_t i) {
    return i;
}
constexpr uint32_t grad_score_reg(uint32_t i) {
    return kGroup + i;
}

// Operands, all per timestep.
constexpr uint32_t cb_query = tt::CBIndex::c_0;
constexpr uint32_t cb_key = tt::CBIndex::c_1;
constexpr uint32_t cb_key_scaled = tt::CBIndex::c_27;  // a * K, when exact
#if FOLD_SCALE_INTO_KEY
constexpr uint32_t cb_key_operand = cb_key_scaled;
#else
constexpr uint32_t cb_key_operand = cb_key;
#endif
constexpr uint32_t cb_key_operand_t = tt::CBIndex::c_16;  // the same block, transposed, for dQ^T
constexpr uint32_t cb_value = tt::CBIndex::c_2;
constexpr uint32_t cb_grad_output = tt::CBIndex::c_3;
// L and D as loaded, one value per row in column 0, are dataflow-side
// scratch (c_4, c_5); this kernel takes only the prepared tiles below.
// L and -D with the block's 32 values in row 0, one tile per row tile. D
// arrives negated because it seeds the dP^T accumulation: the matmul adds
// V dO^T onto it, which leaves dP^T - D^T without a subtraction.
constexpr uint32_t cb_neg_lse_row = tt::CBIndex::c_13;
constexpr uint32_t cb_neg_u_row = tt::CBIndex::c_14;
// The Src registers keep 19 of a Float32's 32 bits, so a statistic arrives in
// two parts: the value (top 19 bits kept) and the remainder those 19 bits
// drop. Both statistics come negated: they seed the registers their matmul
// accumulates into (-L under K Q^T, -D under V dO^T), so no subtraction is
// ever done. The remainders are bfloat16 in column 0 of a tile and enter as
// one more product of the same matmul, against a column of ones -- a
// rank-one term that adds them along every row. Where the exponential
// carries the softmax scale the writer has already multiplied L's seed by
// sqrt(d) and shifted it by ln sqrt(d) (see FOLD_SCALE_INTO_KEY); the tiles
// here are the same either way.
constexpr uint32_t cb_neg_lse_rem = tt::CBIndex::c_9;  // bfloat16, column 0
constexpr uint32_t cb_neg_u_rem = tt::CBIndex::c_29;
constexpr uint32_t cb_ones_column = tt::CBIndex::c_28;
// The causal mask in additive form, two bfloat16 tiles: the transposed
// triangle (0 kept, -inf masked) for the diagonal score tile and an all -inf
// tile for the wholly masked ones. Added to S^T by an accumulating FPU add
// against a zero tile -- the fence buffer's page, which the writer zeroes --
// so the exponential makes the zeros itself and no SFPU pass is needed.
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_6;
constexpr uint32_t kMaskTriangle = 0;
constexpr uint32_t kMaskAll = 1;
constexpr uint32_t cb_zero_tile = tt::CBIndex::c_8;
constexpr uint32_t cb_slot_release = tt::CBIndex::c_7;

// Intermediates, transposed. dP^T never leaves the registers: the score pass
// forms dS^T = P^T (dP^T - D^T) in DST and packs P^T and dS^T once each.
constexpr uint32_t cb_attention_weights = tt::CBIndex::c_10;   // P^T
constexpr uint32_t cb_grad_scores = tt::CBIndex::c_12;         // dS^T

// The row gradient: the packet's seed in, the packet's next hop out, and a
// The two are views of the same memory (the host creates them as one
// buffer with two indices, the same slots): the outgoing packet is the
// incoming one, updated where it lies. This kernel reads the seed through
// the first view and packs through the second, and their pointers advance
// together, one slot a timestep, on this side and on the reader's.
constexpr uint32_t cb_grad_query_seed = tt::CBIndex::c_15;
constexpr uint32_t cb_grad_query_out = tt::CBIndex::c_17;
// A one-tile buffer for unpacker_fence (below); its page is also the zero
// tile the mask add takes.
constexpr uint32_t cb_transpose_fence = tt::CBIndex::c_8;
// Column gradients: what the reader loaded, the accumulator, the writer's copy.
constexpr uint32_t cb_grad_key_seed = tt::CBIndex::c_18;
constexpr uint32_t cb_grad_key_accum = tt::CBIndex::c_19;
constexpr uint32_t cb_grad_key_out = tt::CBIndex::c_20;
constexpr uint32_t cb_grad_value_seed = tt::CBIndex::c_21;
constexpr uint32_t cb_grad_value_accum = tt::CBIndex::c_22;
constexpr uint32_t cb_grad_value_out = tt::CBIndex::c_23;

// pack_tiles_to_output with a multiply on the way through.
//
// One column gradient needs it at both ends of its life: dK with the scale
// folded into K, dV with the scale in the exponential (see
// FOLD_SCALE_INTO_KEY). The accumulator runs short by a factor of a, so the
// handover multiplies by 1/a -- but a revisit then reads that value back
// from DRAM as its seed, so the seed is multiplied by a again first. Both
// are one block of tiles once per residency interval, which is twice per
// core over the whole run.
void pack_tiles_scaled(
    const uint32_t cb_source, const uint32_t cb_output, const uint32_t num_tiles, const uint32_t scale_bits) {
    cb_wait_front(cb_source, num_tiles);
    cb_reserve_back(cb_output, num_tiles);

    pack_reconfig_data_format(cb_output);
    reconfig_data_format(cb_source, cb_source);

    copy_init(cb_source);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        tile_regs_acquire();
        copy_tile(cb_source, tile_idx, /* register idx */ 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(/* register idx */ 0, scale_bits);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(/* register idx */ 0, cb_output);
        tile_regs_release();
    }
    cb_push_back(cb_output, num_tiles);
    cb_pop_front(cb_source, num_tiles);
}

// The column gradient that carries the scale: seeded by a, handed over by
// 1/a. Where the scale is folded into K that is dK and the factor is exact;
// otherwise dV, and the two multiplies round once each per residency
// interval. The other column gradient is copied as it is.
#if FOLD_SCALE_INTO_KEY
constexpr uint32_t seed_scale_bits = inv_scaler_bits;   // dK's seed, divided by a
constexpr uint32_t handover_scale_bits = scaler_bits;   // dK's handover, times a
#else
constexpr uint32_t seed_scale_bits = scaler_bits;       // dV's seed, times a
constexpr uint32_t handover_scale_bits = inv_scaler_bits;  // dV's handover, divided by a
#endif
void reload_column_gradients() {
    // The scaled copy first: a matmul issued straight after an SFPU pass
    // (with only a pack between) came out as zeros; the plain copies that
    // follow are what keeps the two apart.
#if FOLD_SCALE_INTO_KEY
    pack_tiles_scaled(cb_grad_key_seed, cb_grad_key_accum, Bt * qWt, seed_scale_bits);
    pack_tiles_to_output(cb_grad_value_seed, cb_grad_value_accum, Bt * vWt);
#else
    pack_tiles_scaled(cb_grad_value_seed, cb_grad_value_accum, Bt * vWt, seed_scale_bits);
    pack_tiles_to_output(cb_grad_key_seed, cb_grad_key_accum, Bt * qWt);
#endif
}
void handover_column_gradients() {
#if FOLD_SCALE_INTO_KEY
    pack_tiles_to_output(cb_grad_value_accum, cb_grad_value_out, Bt * vWt);
    pack_tiles_scaled(cb_grad_key_accum, cb_grad_key_out, Bt * qWt, handover_scale_bits);
#else
    pack_tiles_scaled(cb_grad_value_accum, cb_grad_value_out, Bt * vWt, handover_scale_bits);
    pack_tiles_to_output(cb_grad_key_accum, cb_grad_key_out, Bt * qWt);
#endif
}

// The packer for P^T and dS^T: Float32 in and out, but with the packer's
// "round to a 10-bit mantissa" control set, so what lands in L1 is the 19
// bits the Src registers keep, rounded to nearest by the packer instead of
// truncated by the unpacker. Truncation shrinks every product by 2^-11 on
// average: a systematic -3e-4 to -5e-4 on dQ, dK and dV that this removes
// for free. The next unconditional pack reconfig (dQ's) clears the bit.
void pack_score_outputs_rounded() {
    pack_reconfig_data_format(cb_attention_weights);
    PACK((cfg_reg_rmw_tensix<
          PCK_DEST_RD_CTRL_Round_10b_mant_ADDR32,
          PCK_DEST_RD_CTRL_Round_10b_mant_SHAMT,
          PCK_DEST_RD_CTRL_Round_10b_mant_MASK>(1)));
}

// Math fidelity per matmul: the number of multiply phases, 2..4. The
// library's matmul_init/matmul_tiles take the kernel-wide MATH_FIDELITY
// (HiFi4); these take it as a template, because HiFi4 is not the most
// accurate choice everywhere. Measured per product with the accuracy
// report (Src operands here are bf16, or Float32 rounded to 19 bits):
// S = K Q^T (bf16 x bf16) is neutral at HiFi3; dK = dS^T Q, with the
// signed 19-bit operand in SrcB, is 45% *more* accurate at HiFi3 -- the
// fourth phase carries a +2e-4 inflation there; dV = P^T dO, the positive
// 19-bit operand in SrcB, is 30% worse at HiFi3; dP = V dO^T and dQ =
// K^T dS^T cost dQ 25% at HiFi3. HiFi2 anywhere is 30x worse. So S and dK
// run at HiFi3 (a quarter of their FPU time saved, -1% to -6% overall) and
// the rest at HiFi4.
#ifndef FID_S
#define FID_S 3
#endif
#ifndef FID_DP
#define FID_DP 4
#endif
#ifndef FID_DV
#define FID_DV 4
#endif
#ifndef FID_DK
#define FID_DK 3
#endif
#ifndef FID_DQ
#define FID_DQ 4
#endif
constexpr MathFidelity fid(int phases) {
    return phases == 2 ? MathFidelity::HiFi2 : phases == 3 ? MathFidelity::HiFi3 : MathFidelity::HiFi4;
}
constexpr MathFidelity kFidS = fid(FID_S), kFidDP = fid(FID_DP), kFidDV = fid(FID_DV), kFidDK = fid(FID_DK),
                        kFidDQ = fid(FID_DQ);
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

// The unpacker waits here for everything the pack thread has packed so far:
// a push the pack thread makes after its packs, waited on before the next
// unpack. Two uses. A dest-register transpose must not overlap the unpack of
// the next matmul's SrcB operand (the transpose has the unpacker mark SrcB
// valid for it and clears both sources when done; an operand already
// delivered gets cleared and that matmul sums zeros), and packed sums must
// be in L1 before they are read back. The buffer is private to this kernel:
// waiting on one another RISC pops would deadlock.
void unpacker_fence() {
    cb_reserve_back(cb_transpose_fence, 1);
    cb_push_back(cb_transpose_fence, 1);
    cb_wait_front(cb_transpose_fence, 1);
    cb_pop_front(cb_transpose_fence, 1);
}

// Every tile of the dQ packet transposed within itself, in place: read
// through the seed view straight into DST (exact), transposed there, packed
// back through the out view to the same address. The unpacker reads a tile
// whole before the packer writes it, and the fence after keeps the
// transposes clear of the next matmul.
void transpose_packet_in_place(const uint32_t cb_prev_srca) {
    reconfig_data_format_srca(cb_prev_srca, cb_grad_query_seed);
    for (uint32_t p = 0; p < Bt * qWt; ++p) {
        tile_regs_acquire();
        // Both inits every tile: the transpose reprograms the math MOP the
        // copy needs.
        copy_init(cb_grad_query_seed);
        copy_tile(cb_grad_query_seed, p, /* register idx */ 0);
        transpose_dest_init</* is_32bit */ true>(cb_grad_query_seed);
        transpose_dest</* is_32bit */ true>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile</* out_of_order */ true>(0, cb_grad_query_out, p);
        tile_regs_release();
    }
    unpacker_fence();
}

// Broadcast a query tile's statistic (row 0 of its row-layout tile) down the
// rows of DST. Once per column of the score grid, since every key tile of
// the column shares it. The statistic goes through SrcB as it did through
// the column broadcast before; same precision.
void broadcast_statistic_rows_to_dst(
    const uint32_t tmp_reg, const uint32_t cb_statistics, const uint32_t stat_tile) {
    reconfig_data_format_srcb(cb_statistics);
    UNPACK((llk_unpack_A_init<BroadcastType::ROW, false, EltwiseBinaryReuseDestType::NONE, false>(
        false, false, cb_statistics)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          ckernel::DataCopyType::B2D,
          DST_ACCUM_MODE,
          BroadcastType::ROW>(cb_statistics)));
    unary_bcast<BroadcastType::ROW>(cb_statistics, stat_tile, tmp_reg);
}

#if defined(TRISC_PACK)
// The pack thread's SFPU: the same exponential and multiply the math thread
// would run, addressed the same way (DEST_TARGET is per thread), minus the
// wait for the math unit that the math-thread versions carry -- ordering
// against the FPU is the semaphore's job here.
namespace pack_sfpu {

// The exponential's polynomial for 2^f over the fraction f, taken as the
// float value of its 23 bits (so c_k carries a factor 2^-23k): a minimax
// fit of degree four with its ends pinned just inside [1, 2). Relative
// error within 3e-6 everywhere.
constexpr uint32_t kExpC0 = 0x3F80000Du;
constexpr uint32_t kExpC1 = 0x33B16976u;
constexpr uint32_t kExpC2 = 0x2777360Au;
constexpr uint32_t kExpC3 = 0x1AD512EEu;
constexpr uint32_t kExpC4 = 0x0E5DE35Du;

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

// Hold the packer until this thread's SFPU writes have landed.
inline void wait_before_pack() {
    TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
}

// Once per kernel: the SFPU config register, the address mods the
// exponential and the multiply walk the tile with, and the exponential's
// constants (1/ln 2 in LREG12, its c2 in LREG13) -- what sdpa_exp_tile_init
// does on the math thread, for this thread's copies of those registers.
// Where K does not carry the softmax scale the exponential does: a / ln 2
// in LREG12 makes it compute exp(a x), the one-multiply fold sdpa_fw uses.
inline void init() {
    ckernel::sfpu::_init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
#if FOLD_SCALE_INTO_KEY
    constexpr float exp_scale = 1.0F;
#else
    constexpr float exp_scale = __builtin_bit_cast(float, scaler_bits);
#endif
    constexpr uint32_t inv_ln2_bits = __builtin_bit_cast(uint32_t, exp_scale * 1.4426950408889634F);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, static_cast<uint16_t>(inv_ln2_bits >> 16));
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, static_cast<uint16_t>(inv_ln2_bits & 0xFFFFu));
    TTI_SFPCONFIG(0, p_sfpu::LREG12, 0);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, kExpC2 >> 16);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, kExpC2 & 0xFFFFu);
    TTI_SFPCONFIG(0, p_sfpu::LREG13, 0);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, kExpC4 >> 16);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, kExpC4 & 0xFFFFu);
    TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);
}

// The library clamps z = a x / ln 2 + 127 at 255 against overflow. Here x
// is a score less its row's logsumexp, so exp(a x) <= 1 up to rounding and
// the clamp is dead: two instructions per vector saved. The lower side,
// -inf from the mask included, is the z > 0 mask in the body.

// The exponential's constants into their registers: c0 in LREG7, c1 in
// LREG6, c3 in LREG4, 127 in LREG5 (c2 and c4 sit in the programmable
// constant registers 13 and 14 from init). Once per group of tiles, since
// nothing else touches these registers between the group's exponentials.
inline void exp_prepare() {
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fe);  // 127.0
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, kExpC1 >> 16);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, kExpC1 & 0xFFFFu);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, kExpC0 >> 16);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, kExpC0 & 0xFFFFu);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, kExpC3 >> 16);
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, kExpC3 & 0xFFFFu);
}

// One face (eight vectors) of exp(x) in place. The same instruction stream
// as _sfpu_exp_21f_bf16_tti_ -- x / ln 2 split into integer and fraction
// bits, the fraction refined by a polynomial, the integer part set as the
// exponent -- less the overflow clamp, and with the polynomial raised from
// degree two to four. The library's quadratic (and a 16-bit immediate for its constant term) leaves
// P with a relative error of 1.2e-3 RMS and a mean of +2.2e-4: bfloat16
// accuracy, and by far the largest error in every gradient once the
// statistics are exact. Two more multiply-adds per vector, in registers
// the exponential was not using, bring it to 3e-6, below the 19 bits the
// Src registers keep. The polynomial is constrained to stay inside [1, 2)
// over the whole fraction range, since SETEXP only replaces the exponent.
inline void exp_face() {
    // The body below is recorded into the replay buffer the first time and
    // replayed for the other seven vectors; the count must match exactly.
    constexpr int kBodyLen = 15;
    TTI_REPLAY(0, kBodyLen, 1, 1);
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);           // x
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG12, p_sfpu::LREG5, p_sfpu::LREG3, 0);      // z = a x / ln 2 + 127
    TTI_SFPEXEXP(0, p_sfpu::LREG3, p_sfpu::LREG1, 0);                                // exponent of z
    TTI_SFPEXMAN(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);                                // mantissa of z, hidden bit in
    TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);                                 // fixed point: integer.fraction
    TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPEXMAN_MOD1_PAD9);         // the 23 fraction bits
    TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);                                    // as a float f in [0, 2^23)
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG4, p_sfpu::LREG2, 0);      // c4 f + c3
    TTI_SFPGT(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 8);                                // mask: z > 0
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG2, 0);      // .. f + c2
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG2, 0);       // .. f + c1
    TTI_SFPAND(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG0, 1);                      // integer part, 0 where z <= 0
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG1, 0);       // 2^fraction = .. f + c0
    TTI_SFPSETEXP(0, p_sfpu::LREG1, p_sfpu::LREG0, 2);                               // times 2^integer
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
#pragma GCC unroll 8
    for (uint32_t i = 1; i < 8u; ++i) {
        TTI_REPLAY(0, kBodyLen, 0, 0);
    }
}

// P = exp(a x) in place, four faces.
inline void exp_tile(const uint32_t tile) {
    set_dst(tile);
    for (uint32_t face = 0; face < 4u; ++face) {
        exp_face();
        next_face();
    }
    clear_dst();
}

// out = a * b, Float32, four faces.
inline void mul_tiles(const uint32_t a, const uint32_t b, const uint32_t out) {
    set_dst(0);
    for (uint32_t face = 0; face < 4u; ++face) {
        ckernel::sfpu::calculate_sfpu_binary_mul</*APPROX*/ false, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE>(
            a, b, out);
        next_face();
    }
    clear_dst();
}

}  // namespace pack_sfpu
#endif

#if FOLD_SCALE_INTO_KEY
// a * K from the K block at the front of cb_key: Bt x qWt bf16 tiles, one
// SFPU multiply each. Exact, since the host folds only power-of-two scales.
void scale_key_block() {
    cb_wait_front(cb_key, Bt * qWt);
    cb_reserve_back(cb_key_scaled, Bt * qWt);
    reconfig_data_format_srca(cb_key);
    copy_init(cb_key);
    pack_reconfig_data_format(cb_key_scaled);
    for (uint32_t t0 = 0; t0 < Bt * qWt; ++t0) {
        tile_regs_acquire();
        copy_tile(cb_key, t0, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scaler_bits);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_key_scaled);
        tile_regs_release();
    }
    cb_push_back(cb_key_scaled, Bt * qWt);
    cb_wait_front(cb_key_scaled, Bt * qWt);
}
#endif

// K_j^T from the resident K_j (scaled where the scale is folded): Bt x qWt
// bf16 tiles in, qWt x Bt out, each tile transposed by the unpacker on the
// way into DST. Once per residency interval; it is what lets dQ^T be a plain
// matmul with dS^T as its second operand, so no score tile is transposed.
void transpose_key_block() {
    cb_wait_front(cb_key_operand, Bt * qWt);
    cb_reserve_back(cb_key_operand_t, Bt * qWt);
    pack_reconfig_data_format(cb_key_operand_t);
    reconfig_data_format_srca(cb_key_operand);
    transpose_init(cb_key_operand);
    for (uint32_t e = 0; e < qWt; ++e) {
        for (uint32_t b = 0; b < Bt; ++b) {
            tile_regs_acquire();
            transpose_tile(cb_key_operand, b * qWt + e, /* register idx */ 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(/* register idx */ 0, cb_key_operand_t);  // tile e * Bt + b
            tile_regs_release();
        }
    }
    cb_push_back(cb_key_operand_t, Bt * qWt);
    cb_wait_front(cb_key_operand_t, Bt * qWt);
}

}  // namespace

void kernel_main() {
    const uint32_t my_core = get_arg_val<uint32_t>(0);
    // Slices this group runs in sequence; see the relay reader. This kernel
    // touches no DRAM, so it needs only the count, not which slices.
    const uint32_t slice_count = get_arg_val<uint32_t>(1);

    using ttml::metal::ops::cyclic_sdpa_bw::CyclicSchedule;
    using ttml::metal::ops::cyclic_sdpa_bw::kNoCore;
    constexpr CyclicSchedule sched(kCores, kMaskMode);
    constexpr uint32_t kTimesteps = sched.num_timesteps();

#if COLUMN_RESIDENT
    // Which columns this core owns, whether each has been resident before --
    // a first visit starts the gradients from nothing, a revisit from DRAM --
    // and whether the current interval's updates accumulate or overwrite.
    const auto owned = sched.owned_columns(my_core);
    bool visited[2] = {false, false};
    bool column_accumulating = false;
#endif

    // Folding the scale into K works without residency too: dK is then divided
    // by the scale at every reload and multiplied at every handover, which is
    // exact for the power-of-two scales the host folds, and what the resident
    // path does once per residency interval this path does once per timestep.
    // It is what keeps the two variants bitwise comparable.

    compute_kernel_hw_startup(cb_query, cb_key, cb_attention_weights);
    // The score pass's FPU -> SFPU handshake, one count per column posted by
    // the math thread and taken by the pack thread.
    MATH((t6_semaphore_init(semaphore::FPU_SFPU, 0, semaphore::SEMAPHORE_MAX_VALUE)));
    PACK((pack_sfpu::init()));
    copy_init(cb_query);
    matmul_init(cb_key_operand, cb_query);
    cb_wait_front(cb_attn_mask, 2);
    cb_wait_front(cb_ones_column, onetile);

    for (uint32_t s = 0; s < slice_count; ++s) {
#if COLUMN_RESIDENT
    // A new slice is a new problem: its columns have not been visited, and
    // its first update to each column gradient writes rather than adds.
    visited[0] = false;
    visited[1] = false;
    column_accumulating = false;
#endif
    for (uint32_t t = 0; t < kTimesteps; ++t) {
        const auto pair = sched.pair(my_core, t);
        const uint32_t g = s * kTimesteps + t;  // global timestep across slices
        // Dense mode masks nothing, so a pair with i == j is an ordinary
        // full block there and must not take the triangular mask.
        const bool diagonal = (DENSE_MODE == 0) && (pair.i == pair.j);
        // Whether this timestep has wholly masked score tiles to skip: a
        // diagonal pair of more than one row tile. Compile-time false at
        // Bt = 1, so that block height keeps its straight-line loops.
        const bool skip_masked = (Bt > 1u) && diagonal;
        // Where dQ_i stands in the relay. Inside a streak the packet carries
        // dQ^T from the previous consumer and the next consumer wants dQ^T
        // back; at a row's first streak start the seed came from DRAM as dQ,
        // and at its last streak end dQ goes back to DRAM as dQ. Without the
        // relay every timestep is both.
#if COLUMN_RESIDENT
        // A spill that a later streak of the same row will reload -- inside
        // this launch -- stays in the packet's transposed form, so only a
        // row's very first seed (the op's dQ input) and its very last spill
        // (the op's dQ output) are transposed. The transposed form keeps
        // every tile in its place and transposes it within itself, so the
        // DRAM pages and the tile order are the same either way.
        const bool seed_transposed =
            sched.producer(my_core, t).internal || sched.is_later_streak_start(pair.i, t);
        const bool emit_transposed =
            sched.next_consumer(pair.i, t) != kNoCore || sched.has_later_active(pair.i, t);
#else
        constexpr bool seed_transposed = false;
        constexpr bool emit_transposed = false;
#endif

#if COLUMN_RESIDENT
        // Popped only when the column changes, which releases the storage for
        // the next column. Waiting every timestep is free once it is there.
        const bool column_changed = (t == 0u) || (sched.pair(my_core, t - 1u).j != pair.j);
        const bool column_ends =
            (t + 1u == kTimesteps) || (sched.pair(my_core, t + 1u).j != pair.j);
        const uint32_t owned_slot = (pair.j == owned.first) ? 0u : 1u;
        if (column_changed && g > 0u) {
            cb_pop_front(cb_key, Bt * qWt);
            cb_pop_front(cb_value, Bt * vWt);
            cb_pop_front(cb_key_operand_t, Bt * qWt);
#if FOLD_SCALE_INTO_KEY
            cb_pop_front(cb_key_scaled, Bt * qWt);
#endif
        }
#if FOLD_SCALE_INTO_KEY
        // a * K, once for the residency interval. Everything that reads K --
        // the scores and dQ -- reads this copy instead.
        if (column_changed) {
            scale_key_block();
        }
#endif
        if (column_changed) {
            transpose_key_block();
            if (visited[owned_slot] || SEED_COLUMN_GRADIENTS) {
                // A revisit, or every visit when accumulating into the
                // outputs: the interval starts from what is in DRAM.
                visited[owned_slot] = true;
                reload_column_gradients();
                column_accumulating = true;
            } else {
                // A first visit: the first update writes rather than adds, so
                // the gradients start at zero without reading zeros.
                column_accumulating = false;
                visited[owned_slot] = true;
            }
        }
#else
        // Without residency the column arrives every timestep, scaled and
        // transposed copies included.
        cb_wait_front(cb_key, Bt * qWt);
#if FOLD_SCALE_INTO_KEY
        scale_key_block();
#endif
        transpose_key_block();
#endif
        {
            DeviceZoneScopedN("WAIT-PACKET");
            {
                DeviceZoneScopedN("W-Q");
                cb_wait_front(cb_query, Bt * qWt);
            }
            {
                DeviceZoneScopedN("W-KV");
                cb_wait_front(cb_key, Bt * qWt);
                cb_wait_front(cb_value, Bt * vWt);
            }
            {
                DeviceZoneScopedN("W-DO");
                cb_wait_front(cb_grad_output, Bt * vWt);
            }
            {
                DeviceZoneScopedN("W-ROWS");
                cb_wait_front(cb_neg_lse_row, Bt);
                cb_wait_front(cb_neg_u_row, Bt);
                cb_wait_front(cb_neg_lse_rem, Bt);
                cb_wait_front(cb_neg_u_rem, Bt);
            }
        }

        // ---- The score pass, one column of the score grid at a time (one
        // query tile against every key tile), entirely in the registers:
        //
        //   S^T  = K Q^T           (scale folded into K where exact), masked
        //                          on the diagonal block,
        //   P^T  = exp(S^T - L^T)  with the column's L broadcast once,
        //   dP^T - D^T             by seeding the registers with -D and letting
        //                          the matmul V dO^T accumulate on top,
        //   dS^T = P^T (dP^T - D^T)  one SFPU multiply per tile,
        //
        // then P^T and dS^T are packed once each. dP^T is never written to L1
        // and P^T is never read back into DST, which is what makes the pass
        // cheaper than three: at Bt = 4 it is 32 packs a timestep instead of
        // 64, and no copies. D is exact through this (it enters DST as a
        // Float32 broadcast) where the FPU subtract it replaces rounded dP^T
        // to 19 bits on the way in.
        {
        DeviceZoneScopedN("SCORES");
        cb_reserve_back(cb_attention_weights, score_tiles);
        cb_reserve_back(cb_grad_scores, score_tiles);
        // Both outputs are Float32 buffers packed as Tf32 -- see
        // pack_score_outputs_rounded. Once per timestep is enough; every
        // later pack reconfigures unconditionally.
        pack_score_outputs_rounded();
        for (uint32_t a = 0; a < Bt; ++a) {
        for (uint32_t h = 0; h < kGroups; ++h) {
            const uint32_t b0 = h * kGroup;  // first key tile of the group
            // On a diagonal pair the tiles whose key index exceeds the query
            // index are wholly masked: P^T and dS^T are exact zeros there, so
            // nothing is computed or packed for them, and the gradient
            // updates below leave their products out. The live tiles of the
            // group are the first n_live (key indices rise with i).
            const uint32_t n_live =
                !skip_masked ? kGroup : (b0 > a ? 0u : (a + 1u - b0 < kGroup ? a + 1u - b0 : kGroup));
            if (n_live == 0u) {
                continue;
            }
            // reconfig_data_format takes (SrcA, SrcB); the matmul's first operand
            // goes to SrcB and its second to SrcA.
            tile_regs_acquire();
            // -L first, by the FPU: broadcast into the score registers, its
            // remainder as a rank-one product against the column of ones,
            // then K Q^T accumulated on top. No SFPU subtract at all.
            for (uint32_t i = 0; i < kGroup; ++i) {
                if (i >= n_live) {
                    break;  // constant trip count keeps the loop unrolled
                }
                broadcast_statistic_rows_to_dst(score_reg(i), cb_neg_lse_row, a);
            }
            reconfig_data_format(cb_query, cb_key_operand);
            mm_init<kFidS>(cb_key_operand, cb_query, /* transpose */ 1);
            for (uint32_t i = 0; i < kGroup; ++i) {
                if (i >= n_live) {
                    break;  // constant trip count keeps the loop unrolled
                }
                const uint32_t b = b0 + i;
                mm_tiles<kFidS>(cb_ones_column, cb_neg_lse_rem, 0, a, score_reg(i));
                for (uint32_t k = 0; k < qWt; ++k) {
                    mm_tiles<kFidS>(cb_key_operand, cb_query, b * qWt + k, a * qWt + k, score_reg(i));
                }
            }
            if (diagonal && b0 + n_live > a) {
                // The mask, added to the diagonal tile (b = a): -inf where the
                // key index exceeds the query index. The exponential then
                // makes exact zeros there.
                reconfig_data_format(cb_attn_mask, cb_zero_tile);
                add_tiles_init(cb_attn_mask, cb_zero_tile, /* acc_to_dest */ true);
                add_tiles(cb_attn_mask, cb_zero_tile, kMaskTriangle, 0, score_reg(a - b0));
            }

            // S^T is complete in the registers: hand it to the pack thread's
            // SFPU, which exponentiates while the FPU goes on to dP^T.
            MATH((t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU)));

            // dP^T - D^T for the group: -D broadcast into every register of
            // the second half, then V dO^T accumulated onto it. The FPU adds
            // into DST.
            for (uint32_t i = 0; i < kGroup; ++i) {
                if (i >= n_live) {
                    break;  // constant trip count keeps the loop unrolled
                }
                broadcast_statistic_rows_to_dst(grad_score_reg(i), cb_neg_u_row, a);
            }
            reconfig_data_format(cb_grad_output, cb_value);
            mm_init<kFidDP>(cb_value, cb_grad_output, /* transpose */ 1);
            for (uint32_t i = 0; i < kGroup; ++i) {
                if (i >= n_live) {
                    break;  // constant trip count keeps the loop unrolled
                }
                const uint32_t b = b0 + i;
                for (uint32_t k = 0; k < vWt; ++k) {
                    mm_tiles<kFidDP>(cb_value, cb_grad_output, b * vWt + k, a * vWt + k, grad_score_reg(i));
                }
                // The remainder of -D, along every row: ones-column x (its
                // column-0 tile)^T. Same formats as V and dO, so the same init.
                mm_tiles<kFidDP>(cb_ones_column, cb_neg_u_rem, 0, a, grad_score_reg(i));
            }

            // Pack thread: the exponentials as soon as S^T is posted, then --
            // after the math thread's commit, which says dP^T - D^T is in --
            // the multiplies, and the packs once the SFPU has written them.
            PACK((t6_semaphore_wait_on_zero<p_stall::STALL_SFPU>(semaphore::FPU_SFPU)));
            PACK((pack_sfpu::exp_prepare()));
            for (uint32_t i = 0; i < kGroup; ++i) {
                if (i >= n_live) {
                    break;  // constant trip count keeps the loop unrolled
                }
                PACK((pack_sfpu::exp_tile(score_reg(i))));
            }
            PACK((t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU)));
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < kGroup; ++i) {
                if (i >= n_live) {
                    break;  // constant trip count keeps the loop unrolled
                }
                PACK((pack_sfpu::mul_tiles(grad_score_reg(i), score_reg(i), grad_score_reg(i))));
            }
            PACK((pack_sfpu::wait_before_pack()));
            for (uint32_t i = 0; i < kGroup; ++i) {
                if (i >= n_live) {
                    break;  // constant trip count keeps the loop unrolled
                }
                const uint32_t b = b0 + i;
                pack_tile</* out_of_order */ true>(score_reg(i), cb_attention_weights, b * Bt + a);
                pack_tile</* out_of_order */ true>(grad_score_reg(i), cb_grad_scores, b * Bt + a);
            }
            tile_regs_release();
        }
        }
        cb_push_back(cb_attention_weights, score_tiles);
        cb_push_back(cb_grad_scores, score_tiles);
        cb_wait_front(cb_attention_weights, score_tiles);
        cb_wait_front(cb_grad_scores, score_tiles);
        }

        // ---- dQ_i^T += K_j^T dS^T, on the packet where it lies. The products
        // are formed in the registers from zero and the packer adds them onto
        // the seed in L1 (the same L1 accumulation the column gradients use),
        // so no seed is ever copied into the registers. Where the seed came
        // from DRAM as dQ it is first transposed in place, tile by tile, and
        // where the result goes back to DRAM it is transposed back the same
        // way -- both exact (unpack straight to DST, transpose, pack), and
        // both once per streak. (Transposing the products in the registers
        // instead, to save the second pass, needs a fence after every block
        // and measured slower.)
        {
            DeviceZoneScopedN("UPDATE-DQ");
            cb_wait_front(cb_grad_query_seed, Bt * qWt);
            // The out view's next slot is the seed's slot (see the buffers).
            cb_reserve_back(cb_grad_query_out, Bt * qWt);
            pack_reconfig_data_format(cb_grad_query_out);
            if (!seed_transposed) {
                DeviceZoneScopedN("SEED-T");
                transpose_packet_in_place(cb_grad_output);
            }
            // K^T is the first operand (bf16, SrcB), dS^T the second (Float32,
            // SrcA). The previous SrcA operand was dO (bf16) or, after the
            // transposes, the seed (Float32, unpack-to-dest).
            reconfig_data_format(/* SrcA */ cb_grad_scores, /* SrcB */ cb_key_operand_t);
            mm_init<kFidDQ>(cb_key_operand_t, cb_grad_scores, /* transpose */ 0);
            pack_reconfig_l1_acc(true);
            for (uint32_t a = 0; a < Bt; ++a) {
                for (uint32_t k0 = 0; k0 < qWt; k0 += block_size) {
                    tile_regs_acquire();
                    for (uint32_t bi = 0; bi < block_size; ++bi) {
                        // Key tiles above the query tile hold no dS^T on a
                        // diagonal pair (see the score pass).
                        for (uint32_t b = 0; b < Bt; ++b) {
                            if (skip_masked && b > a) {
                                break;
                            }
                            mm_tiles<kFidDQ>(cb_key_operand_t, cb_grad_scores, (k0 + bi) * Bt + b, b * Bt + a, bi);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t bi = 0; bi < block_size; ++bi) {
                        pack_tile</* out_of_order */ true>(bi, cb_grad_query_out, a * qWt + k0 + bi);
                    }
                    tile_regs_release();
                }
            }
            pack_reconfig_l1_acc(false);
            if (!emit_transposed) {
                // Streak end: dQ goes back to DRAM untransposed. The sums must
                // have landed in L1 before the unpacker reads them back.
                unpacker_fence();
                transpose_packet_in_place(cb_grad_scores);
            }
            cb_push_back(cb_grad_query_out, Bt * qWt);
            cb_pop_front(cb_grad_query_seed, Bt * qWt);
        }

        // ---- dV_j += P^T dO_i, summed over the block's query tiles
        {
            DeviceZoneScopedN("UPDATE-DV");
#if COLUMN_RESIDENT
            const bool dv_accumulate = column_accumulating;
#else
            // Both seeds come from DRAM every timestep here.
            reload_column_gradients();
            const bool dv_accumulate = true;
#endif
            // The previous pack was dQ into the relay's buffer.
            pack_reconfig_data_format(cb_grad_query_out, cb_grad_value_accum);
            if (!dv_accumulate) {
                cb_reserve_back(cb_grad_value_accum, Bt * vWt);
            } else {
                pack_reconfig_l1_acc(true);
            }
            for (uint32_t b = 0; b < Bt; ++b) {
                for (uint32_t k0 = 0; k0 < vWt; k0 += block_size) {
                    tile_regs_acquire();
                    // SrcA takes the second operand (dO, bf16), SrcB the first (P^T, Float32).
                    reconfig_data_format(cb_grad_output, cb_attention_weights);
                    mm_init<kFidDV>(cb_attention_weights, cb_grad_output, /* transpose */ 0);
                    for (uint32_t bi = 0; bi < block_size; ++bi) {
                        // Query tiles below the key tile hold no P^T on a
                        // diagonal pair (see the score pass).
                        for (uint32_t a = 0; a < Bt; ++a) {
                            if (skip_masked && a < b) {
                                continue;
                            }
                            mm_tiles<kFidDV>(cb_attention_weights, cb_grad_output, b * Bt + a, a * vWt + k0 + bi, bi);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t bi = 0; bi < block_size; ++bi) {
                        pack_tile(bi, cb_grad_value_accum);
                    }
                    tile_regs_release();
                }
            }
            if (dv_accumulate) {
                pack_reconfig_l1_acc(false);
                cb_pop_front(cb_grad_value_accum, Bt * vWt);
                cb_reserve_back(cb_grad_value_accum, Bt * vWt);
            }
            cb_push_back(cb_grad_value_accum, Bt * vWt);
            cb_wait_front(cb_grad_value_accum, Bt * vWt);
        }

        // ---- dK_j += dS^T Q_i. The reconfig arguments must name what the
        // previous operation actually left in the packer and in SrcA, because
        // those reconfigs are conditional and skip when the formats already
        // match. Here the previous operation is the dV update just above.
        {
            DeviceZoneScopedN("UPDATE-DK");
#if COLUMN_RESIDENT
            const bool dk_accumulate = column_accumulating;
#else
            const bool dk_accumulate = true;
#endif
            // The previous operation is the dV update, so its accumulator is
            // what the packer was last set from and dO what SrcA was.
            constexpr uint32_t cb_prev_pack = cb_grad_value_accum;
            constexpr uint32_t cb_prev_srca = cb_grad_output;
            pack_reconfig_data_format(cb_prev_pack, cb_grad_key_accum);
            if (!dk_accumulate) {
                cb_reserve_back(cb_grad_key_accum, Bt * qWt);
            } else {
                pack_reconfig_l1_acc(true);
            }
            for (uint32_t b = 0; b < Bt; ++b) {
                for (uint32_t k0 = 0; k0 < qWt; k0 += block_size) {
                    tile_regs_acquire();
                    reconfig_data_format_srca(cb_prev_srca, cb_query);
                    mm_init<kFidDK>(cb_grad_scores, cb_query, /* transpose */ 0);
                    for (uint32_t bi = 0; bi < block_size; ++bi) {
                        for (uint32_t a = 0; a < Bt; ++a) {
                            if (skip_masked && a < b) {
                                continue;
                            }
                            mm_tiles<kFidDK>(cb_grad_scores, cb_query, b * Bt + a, a * qWt + k0 + bi, bi);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t bi = 0; bi < block_size; ++bi) {
                        pack_tile(bi, cb_grad_key_accum);
                    }
                    tile_regs_release();
                }
            }
            if (dk_accumulate) {
                pack_reconfig_l1_acc(false);
                cb_pop_front(cb_grad_key_accum, Bt * qWt);
                cb_reserve_back(cb_grad_key_accum, Bt * qWt);
            }
            cb_push_back(cb_grad_key_accum, Bt * qWt);
            cb_wait_front(cb_grad_key_accum, Bt * qWt);
#if COLUMN_RESIDENT
            column_accumulating = true;
            // Hand both column gradients over once, at the end of the interval.
            if (column_ends) {
                handover_column_gradients();
            }
#else
            handover_column_gradients();
#endif
        }

        cb_pop_front(cb_query, Bt * qWt);
#if !COLUMN_RESIDENT
        cb_pop_front(cb_key, Bt * qWt);
        cb_pop_front(cb_value, Bt * vWt);
        cb_pop_front(cb_key_operand_t, Bt * qWt);
#if FOLD_SCALE_INTO_KEY
        cb_pop_front(cb_key_scaled, Bt * qWt);
#endif
#endif
        cb_pop_front(cb_grad_output, Bt * vWt);
        cb_pop_front(cb_neg_lse_row, Bt);
        cb_pop_front(cb_neg_u_row, Bt);
        cb_pop_front(cb_neg_lse_rem, Bt);
        cb_pop_front(cb_neg_u_rem, Bt);
        cb_pop_front(cb_attention_weights, score_tiles);
        cb_pop_front(cb_grad_scores, score_tiles);

#if RELEASE_TOKEN
        // Slot t mod 2 is free now: every read of it is done.
        cb_reserve_back(cb_slot_release, 1);
        cb_push_back(cb_slot_release, 1);
#endif
    }
    }  // slices
}
