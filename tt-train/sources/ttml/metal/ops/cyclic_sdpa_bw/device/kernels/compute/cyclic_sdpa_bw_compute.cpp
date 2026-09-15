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
// It appears twice in the arithmetic, on the scores and again on dS, and both
// can go by scaling K instead:
//
//     S  = Q (aK)^T = a Q K^T     the score path then needs no scale at all,
//     dQ = G (aK)   = a G K       with dS left unscaled,
//     dK = G^T Q                  short by a -- but dK is an accumulator, so
//                                 one scale at its handover fixes it,
//     dV = P^T dO                 untouched.
//
// K is resident for a whole residency interval, so scaling it costs one pass
// per interval, twice per core over the run, against Bt * Bt passes on dS and
// Bt on the statistic every timestep. At Bt = 4 that removes 20 SFPU passes
// per timestep.
//
// Only where it is exact: K is bfloat16, so a * K rounds unless a is a power
// of two -- d = 64 and 256 yes, d = 128 no -- and the host enables this only
// then. The other path keeps the scale folded into the exponential instead.
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

// DST for one column of the score grid: P^T in registers 0..Bt-1 and dS^T in
// Bt..2Bt-1, where the column's broadcast statistic and the causal mask lived
// while the scores were being formed -- both are dead by the time dP^T is
// started, so the two halves of the pass share the file. 2 Bt registers,
// which is the whole Float32 file at Bt = 4.
constexpr uint32_t score_reg(uint32_t b) {
    return b;
}
constexpr uint32_t grad_score_reg(uint32_t b) {
    return Bt + b;
}
constexpr uint32_t stat_reg = Bt;
constexpr uint32_t mask_reg = Bt + 1u;

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
// L and D as they travel: one value per row in column 0. Not read here --
// the row-layout copies below are -- but popped so the slots turn over.
constexpr uint32_t cb_lse = tt::CBIndex::c_4;
constexpr uint32_t cb_u_scalar = tt::CBIndex::c_5;
// L and -D with the block's 32 values in row 0, one tile per row tile. D
// arrives negated because it seeds the dP^T accumulation: the matmul adds
// V dO^T onto it, which leaves dP^T - D^T without a subtraction.
constexpr uint32_t cb_lse_row = tt::CBIndex::c_13;
constexpr uint32_t cb_neg_u_row = tt::CBIndex::c_14;
// The Src registers keep 19 of a Float32's 32 bits, so a statistic arrives in
// two parts: the value (top 19 bits kept) and the remainder those 19 bits
// drop, which the register keeps 10 more bits of. L's remainder is a second
// row-layout tile, subtracted after the first; -D's remainder is bfloat16 in
// column 0 of a tile, and enters dP^T as one more product of the matmul,
// against a column of ones -- a rank-one term that adds it along every row.
constexpr uint32_t cb_lse_rem = tt::CBIndex::c_30;
constexpr uint32_t cb_neg_u_rem = tt::CBIndex::c_29;
constexpr uint32_t cb_ones_column = tt::CBIndex::c_28;
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_6;  // transposed causal
constexpr uint32_t cb_slot_release = tt::CBIndex::c_7;

// Intermediates, transposed. dP^T never leaves the registers: the score pass
// forms dS^T = P^T (dP^T - D^T) in DST and packs P^T and dS^T once each.
constexpr uint32_t cb_attention_weights = tt::CBIndex::c_10;   // P^T
constexpr uint32_t cb_grad_scores = tt::CBIndex::c_12;         // dS^T

// The row gradient: the packet's seed in, the packet's next hop out, and a
// scratch copy of the seed transposed for the timesteps where it came from
// DRAM as dQ rather than from the packet as dQ^T.
constexpr uint32_t cb_grad_query_seed = tt::CBIndex::c_15;
constexpr uint32_t cb_grad_query_seed_t = tt::CBIndex::c_11;
constexpr uint32_t cb_grad_query_out = tt::CBIndex::c_17;
// A one-tile buffer whose push, made by the pack thread after the packs that
// follow a dest-register transpose, is what the unpacker waits on before it
// may feed the next matmul. See UPDATE-DQ.
constexpr uint32_t cb_transpose_fence = tt::CBIndex::c_8;
// Column gradients: what the reader loaded, the accumulator, the writer's copy.
constexpr uint32_t cb_grad_key_seed = tt::CBIndex::c_18;
constexpr uint32_t cb_grad_key_accum = tt::CBIndex::c_19;
constexpr uint32_t cb_grad_key_out = tt::CBIndex::c_20;
constexpr uint32_t cb_grad_value_seed = tt::CBIndex::c_21;
constexpr uint32_t cb_grad_value_accum = tt::CBIndex::c_22;
constexpr uint32_t cb_grad_value_out = tt::CBIndex::c_23;

#if FOLD_SCALE_INTO_KEY
// pack_tiles_to_output with a multiply on the way through.
//
// dK needs it at both ends of its life. Folding the scale into K leaves the
// accumulator short by a factor of a, so the handover multiplies by a -- but
// a revisit then reads that already-scaled value back from DRAM as its seed,
// so the seed is divided by a again first. Both are Bt * qWt tiles once per
// residency interval, which is twice per core over the whole run.
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
#endif

// sdpa_bw's apply_mask_on_reg with the scratch register named rather than
// assumed to be the next one along: score tiles are contiguous here, so the
// register after one is another score tile.
void apply_mask_at(
    const uint32_t scores_reg,
    const uint32_t mask_register,
    const uint32_t cb_mask,
    const uint32_t minus_one,
    const uint32_t custom_inf) {
    copy_init(cb_mask);
    copy_tile(cb_mask, /* tile_idx */ 0, mask_register);

    mask_tile_init();
    mask_tile(scores_reg, mask_register);

    // No scale here: the exponential applies it, and the mask's minus
    // infinity survives being scaled either way.
    binop_with_scalar_tile_init();
    add_unary_tile(mask_register, minus_one);
    mul_unary_tile(mask_register, custom_inf);

    add_binary_tile_init();
    add_binary_tile(scores_reg, mask_register, scores_reg);
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

// P = exp(a(S - L/a)) for every score tile of one column, with the scale
// folded into the exponential rather than applied to S beforehand, and L
// subtracted in two parts: the broadcast register holds the value (19 bits),
// then its remainder, each divided by the scale where the exponential
// carries it.
//
// sdpa_exp_tile_scaled folds the whole FP32 scale into LREG12 at init time on
// Blackhole -- one SFPU pass per score tile that no longer happens -- and
// pre-multiplies by a bfloat16 scale on Wormhole. It computes exp(a * x), so
// the caller supplies L/a and the identity exp(a(S - L/a)) = exp(aS - L) does
// the rest. sdpa_fw already works this way, keeping its scores and its
// running maximum unscaled.
void subtract_statistic_column(const uint32_t first_reg, const uint32_t count, const uint32_t broadcast_reg) {
#if !FOLD_SCALE_INTO_KEY
    // The exponential carries the scale, so what it subtracts must be
    // divided by it first.
    binop_with_scalar_tile_init();
    mul_unary_tile(broadcast_reg, inv_scaler_bits);
#endif
    sub_binary_tile_init();
    for (uint32_t b = 0; b < count; ++b) {
        sub_binary_tile(first_reg + b, broadcast_reg, first_reg + b);
    }
}

void exp_column(const uint32_t first_reg, const uint32_t count) {
#if FOLD_SCALE_INTO_KEY
    // S already carries the scale, having come out of (aK) Q^T.
    sdpa_exp_tile_init</*approx*/ false, /*SCALE_EN*/ false>();
#else
    sdpa_exp_tile_init</*approx*/ false, /*SCALE_EN*/ true, scaler_bits>();
#endif
    for (uint32_t b = 0; b < count; ++b) {
        sdpa_exp_tile(first_reg + b);
    }
}

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

    static_assert(
        FOLD_SCALE_INTO_KEY == 0 || COLUMN_RESIDENT == 1,
        "folding the scale into K assumes the resident column path: without residency dK is handed "
        "over every timestep and its seed re-read every timestep, so the scale would compound");

    compute_kernel_hw_startup(cb_query, cb_key, cb_attention_weights);
    copy_init(cb_query);
    matmul_init(cb_key_operand, cb_query);
    cb_wait_front(cb_attn_mask, onetile);
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
        // Where dQ_i stands in the relay. Inside a streak the packet carries
        // dQ^T from the previous consumer and the next consumer wants dQ^T
        // back; at a streak start the seed came from DRAM as dQ, and at a
        // streak end dQ goes back to DRAM as dQ. Without the relay every
        // timestep is both.
#if COLUMN_RESIDENT
        const bool seed_transposed = sched.producer(my_core, t).internal;
        const bool emit_transposed = sched.next_consumer(pair.i, t) != kNoCore;
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
        if (column_changed) {
            transpose_key_block();
            if (visited[owned_slot] || SEED_COLUMN_GRADIENTS) {
                // A revisit, or every visit when accumulating into the
                // outputs: the interval starts from what is in DRAM.
                visited[owned_slot] = true;
#if FOLD_SCALE_INTO_KEY
                pack_tiles_scaled(cb_grad_key_seed, cb_grad_key_accum, Bt * qWt, inv_scaler_bits);
#else
                pack_tiles_to_output(cb_grad_key_seed, cb_grad_key_accum, Bt * qWt);
#endif
                pack_tiles_to_output(cb_grad_value_seed, cb_grad_value_accum, Bt * vWt);
                column_accumulating = true;
            } else {
                // A first visit: the first update writes rather than adds, so
                // the gradients start at zero without reading zeros.
                column_accumulating = false;
                visited[owned_slot] = true;
            }
        }
#else
        // Without residency the column arrives every timestep, transposed
        // copy included.
        cb_wait_front(cb_key, Bt * qWt);
        transpose_key_block();
#endif
        {
            DeviceZoneScopedN("WAIT-PACKET");
            cb_wait_front(cb_query, Bt * qWt);
            cb_wait_front(cb_key, Bt * qWt);
            cb_wait_front(cb_value, Bt * vWt);
            cb_wait_front(cb_grad_output, Bt * vWt);
            cb_wait_front(cb_lse_row, Bt);
            cb_wait_front(cb_neg_u_row, Bt);
            cb_wait_front(cb_lse_rem, Bt);
            cb_wait_front(cb_neg_u_rem, Bt);
            cb_wait_front(cb_lse, Bt);
            cb_wait_front(cb_u_scalar, Bt);
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
        // Both outputs are Float32; the previous pack may have been the bf16
        // K^T. Once per timestep is enough.
        pack_reconfig_data_format(cb_attention_weights);
        for (uint32_t a = 0; a < Bt; ++a) {
            // reconfig_data_format takes (SrcA, SrcB); the matmul's first operand
            // goes to SrcB and its second to SrcA.
            reconfig_data_format(cb_query, cb_key_operand);
            matmul_init(cb_key_operand, cb_query, /* transpose */ 1);
            tile_regs_acquire();
            for (uint32_t b = 0; b < Bt; ++b) {
                for (uint32_t k = 0; k < qWt; ++k) {
                    matmul_tiles(cb_key_operand, cb_query, b * qWt + k, a * qWt + k, score_reg(b));
                }
            }
            if (diagonal) {
                // The diagonal tile of column a is key tile b = a.
                apply_mask_at(score_reg(a), mask_reg, cb_attn_mask, minus_one_bits, custom_inf_bits);
            }

            broadcast_statistic_rows_to_dst(stat_reg, cb_lse_row, a);
            subtract_statistic_column(score_reg(0), Bt, stat_reg);
            broadcast_statistic_rows_to_dst(stat_reg, cb_lse_rem, a);
            subtract_statistic_column(score_reg(0), Bt, stat_reg);
            exp_column(score_reg(0), Bt);
            for (uint32_t b = 0; b < Bt; ++b) {
                if (diagonal && b > a) {
                    // Key tile after the query tile: wholly masked. Zeroing
                    // P^T there lets every later sum run over all of the
                    // block's tiles without knowing about the triangle,
                    // since dS^T inherits the zero through its P^T factor.
                    binop_with_scalar_tile_init();
                    mul_unary_tile(score_reg(b), /* 0.0f */ 0u);
                }
            }

            // dP^T - D^T for the column: -D broadcast into every register of
            // the second half (the statistic and mask registers are free by
            // now), then V dO^T accumulated onto it. The FPU adds into DST.
            for (uint32_t b = 0; b < Bt; ++b) {
                broadcast_statistic_rows_to_dst(grad_score_reg(b), cb_neg_u_row, a);
            }
            reconfig_data_format(cb_grad_output, cb_value);
            matmul_init(cb_value, cb_grad_output, /* transpose */ 1);
            for (uint32_t b = 0; b < Bt; ++b) {
                for (uint32_t k = 0; k < vWt; ++k) {
                    matmul_tiles(cb_value, cb_grad_output, b * vWt + k, a * vWt + k, grad_score_reg(b));
                }
                // The remainder of -D, along every row: ones-column x (its
                // column-0 tile)^T. Same formats as V and dO, so the same init.
                matmul_tiles(cb_ones_column, cb_neg_u_rem, 0, a, grad_score_reg(b));
            }
            // dS^T = P^T (dP^T - D^T), and the softmax scale where K does not
            // carry it.
            mul_binary_tile_init();
            for (uint32_t b = 0; b < Bt; ++b) {
                mul_binary_tile(grad_score_reg(b), score_reg(b), grad_score_reg(b));
            }
#if !FOLD_SCALE_INTO_KEY
            binop_with_scalar_tile_init();
            for (uint32_t b = 0; b < Bt; ++b) {
                mul_unary_tile(grad_score_reg(b), scaler_bits);
            }
#endif
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t b = 0; b < Bt; ++b) {
                pack_tile</* out_of_order */ true>(score_reg(b), cb_attention_weights, b * Bt + a);
                pack_tile</* out_of_order */ true>(grad_score_reg(b), cb_grad_scores, b * Bt + a);
            }
            tile_regs_release();
        }
        cb_push_back(cb_attention_weights, score_tiles);
        cb_push_back(cb_grad_scores, score_tiles);
        cb_wait_front(cb_attention_weights, score_tiles);
        cb_wait_front(cb_grad_scores, score_tiles);
        }

        // ---- dQ_i^T = (dQ_i^T from the packet) + K_j^T dS^T, straight from the
        // seed to the outgoing packet: the seed tiles are copied into the DST
        // registers, transposed there if they came from DRAM as dQ, the
        // matmul accumulates on top (the FPU adds into DST), and the result
        // is transposed back before the pack if it goes to DRAM.
        {
            DeviceZoneScopedN("UPDATE-DQ");
            cb_wait_front(cb_grad_query_seed, Bt * qWt);
            // A dest transpose must not overlap the unpack of a matmul operand
            // into SrcB: the transpose has the unpacker mark SrcB valid for it
            // and clears both sources when it is done, and an operand the
            // unpacker had already run ahead and delivered for the next matmul
            // is what gets cleared -- the dV update then sums zeros. So both
            // transposes below sit behind a buffer handshake that the unpacker
            // has to wait on before it may feed the next matmul: the seed is
            // transposed into a scratch buffer of its own, and the emitted dQ
            // is waited for before the dV update starts.
            uint32_t seed_cb = cb_grad_query_seed;
            if (!seed_transposed) {
                DeviceZoneScopedN("SEED-T");
                cb_reserve_back(cb_grad_query_seed_t, Bt * qWt);
                pack_reconfig_data_format(cb_grad_query_seed_t);
                reconfig_data_format_srca(cb_grad_output, cb_grad_query_seed);
                for (uint32_t a = 0; a < Bt; ++a) {
                    for (uint32_t e = 0; e < qWt; ++e) {
                        tile_regs_acquire();
                        // Both inits every tile: the transpose reprograms the
                        // math MOP the copy needs.
                        copy_init(cb_grad_query_seed);
                        copy_tile(cb_grad_query_seed, a * qWt + e, /* register idx */ 0);
                        transpose_dest_init</* is_32bit */ true>(cb_grad_query_seed);
                        transpose_dest</* is_32bit */ true>(0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile</* out_of_order */ true>(0, cb_grad_query_seed_t, e * Bt + a);
                        tile_regs_release();
                    }
                }
                cb_push_back(cb_grad_query_seed_t, Bt * qWt);
                cb_wait_front(cb_grad_query_seed_t, Bt * qWt);
                seed_cb = cb_grad_query_seed_t;
            }
            cb_reserve_back(cb_grad_query_out, Bt * qWt);
            pack_reconfig_data_format(cb_grad_query_out);
            for (uint32_t a = 0; a < Bt; ++a) {
                for (uint32_t k0 = 0; k0 < qWt; k0 += block_size) {
                    tile_regs_acquire();
                    // The seed tiles, dQ^T in either buffer, go through SrcA as
                    // they always did (Float32 in, Float32 out). The previous
                    // SrcA operand was dO (bf16) for the first block and dS^T
                    // (Float32) after that.
                    reconfig_data_format_srca((a == 0u && k0 == 0u) ? cb_grad_output : cb_grad_scores, seed_cb);
                    copy_init(seed_cb);
                    for (uint32_t bi = 0; bi < block_size; ++bi) {
                        copy_tile(seed_cb, (k0 + bi) * Bt + a, bi);
                    }
                    // dQ^T[e][a] += sum_b K^T[e][b] dS^T[b][a]: K^T is the first
                    // operand (bf16, SrcB), dS^T the second (Float32, SrcA).
                    reconfig_data_format(/* SrcA */ cb_grad_scores, /* SrcB */ cb_key_operand_t);
                    matmul_init(cb_key_operand_t, cb_grad_scores, /* transpose */ 0);
                    for (uint32_t bi = 0; bi < block_size; ++bi) {
                        for (uint32_t b = 0; b < Bt; ++b) {
                            matmul_tiles(cb_key_operand_t, cb_grad_scores, (k0 + bi) * Bt + b, b * Bt + a, bi);
                        }
                    }
                    if (!emit_transposed) {
                        // Streak end: dQ goes back to DRAM untransposed. In the
                        // registers, exact; nothing else of this acquire reads
                        // the unpacker after this point, and the fence below
                        // keeps the dV matmul's operands out of the way.
                        transpose_dest_init</* is_32bit */ true>(cb_grad_query_out);
                        for (uint32_t bi = 0; bi < block_size; ++bi) {
                            transpose_dest</* is_32bit */ true>(bi);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t bi = 0; bi < block_size; ++bi) {
                        const uint32_t e = k0 + bi;
                        const uint32_t out_tile = emit_transposed ? e * Bt + a : a * qWt + e;
                        pack_tile</* out_of_order */ true>(bi, cb_grad_query_out, out_tile);
                    }
                    tile_regs_release();
                }
            }
            cb_push_back(cb_grad_query_out, Bt * qWt);
            if (!emit_transposed) {
                // The fence: pushed by the pack thread once the packs above,
                // and so the transposes before them, are done; waited on by the
                // unpacker before it unpacks anything for the dV matmul. The
                // packet buffer itself cannot serve, since the relay reader pops
                // it.
                cb_reserve_back(cb_transpose_fence, 1);
                cb_push_back(cb_transpose_fence, 1);
                cb_wait_front(cb_transpose_fence, 1);
                cb_pop_front(cb_transpose_fence, 1);
            }
            cb_pop_front(cb_grad_query_seed, Bt * qWt);
            if (!seed_transposed) {
                cb_pop_front(cb_grad_query_seed_t, Bt * qWt);
            }
        }

        // ---- dV_j += P^T dO_i, summed over the block's query tiles
        {
            DeviceZoneScopedN("UPDATE-DV");
#if COLUMN_RESIDENT
            const bool dv_accumulate = column_accumulating;
#else
            pack_tiles_to_output(cb_grad_value_seed, cb_grad_value_accum, Bt * vWt);
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
                    matmul_init(cb_attention_weights, cb_grad_output, /* transpose */ 0);
                    for (uint32_t bi = 0; bi < block_size; ++bi) {
                        for (uint32_t a = 0; a < Bt; ++a) {
                            matmul_tiles(cb_attention_weights, cb_grad_output, b * Bt + a, a * vWt + k0 + bi, bi);
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
#if !COLUMN_RESIDENT
            pack_tiles_to_output(cb_grad_value_accum, cb_grad_value_out, Bt * vWt);
#endif
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
            pack_tiles_to_output(cb_grad_key_seed, cb_grad_key_accum, Bt * qWt);
            const bool dk_accumulate = true;
#endif
#if COLUMN_RESIDENT
            // The previous operation is the dV update, so its accumulator is
            // what the packer was last set from and dO what SrcA was.
            constexpr uint32_t cb_prev_pack = cb_grad_value_accum;
            constexpr uint32_t cb_prev_srca = cb_grad_output;
#else
            // Here the previous operation is the dK seed copy just above,
            // which packed to the accumulator and read the seed.
            constexpr uint32_t cb_prev_pack = cb_grad_key_accum;
            constexpr uint32_t cb_prev_srca = cb_grad_key_seed;
#endif
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
                    matmul_init(cb_grad_scores, cb_query, /* transpose */ 0);
                    for (uint32_t bi = 0; bi < block_size; ++bi) {
                        for (uint32_t a = 0; a < Bt; ++a) {
                            matmul_tiles(cb_grad_scores, cb_query, b * Bt + a, a * qWt + k0 + bi, bi);
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
                pack_tiles_to_output(cb_grad_value_accum, cb_grad_value_out, Bt * vWt);
#if FOLD_SCALE_INTO_KEY
                pack_tiles_scaled(cb_grad_key_accum, cb_grad_key_out, Bt * qWt, scaler_bits);
#else
                pack_tiles_to_output(cb_grad_key_accum, cb_grad_key_out, Bt * qWt);
#endif
            }
#else
            pack_tiles_to_output(cb_grad_key_accum, cb_grad_key_out, Bt * qWt);
#endif
        }

        cb_pop_front(cb_query, Bt * qWt);
#if !COLUMN_RESIDENT
        cb_pop_front(cb_key, Bt * qWt);
        cb_pop_front(cb_value, Bt * vWt);
        cb_pop_front(cb_key_operand_t, Bt * qWt);
#endif
        cb_pop_front(cb_grad_output, Bt * vWt);
        cb_pop_front(cb_lse, Bt);
        cb_pop_front(cb_u_scalar, Bt);
        cb_pop_front(cb_lse_row, Bt);
        cb_pop_front(cb_neg_u_row, Bt);
        cb_pop_front(cb_lse_rem, Bt);
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
