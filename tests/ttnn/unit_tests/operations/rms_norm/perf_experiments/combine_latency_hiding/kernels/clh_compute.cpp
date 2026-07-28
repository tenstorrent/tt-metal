// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm compute — op_design.md §8.
//
// Per core, per row-block of HT_BLOCK tile-rows:
//
//   pass A (all NW chunks):  [tilize] -> square -> reduce<SUM,REDUCE_ROW,
//                            AccumulateViaAdd> into cb_partials, finalizing the
//                            last chunk with reduce_mean(n_reduced = W)
//   phase 4 (once):          AddUnary(eps) -> Rsqrt  => cb_rms_recip (1/rms)
//   pass B (all NW chunks):  [tilize] -> mul<Col>(x, 1/rms) -> mul<Row>(., gamma)
//                            -> [untilize]
//
// FUSE_SQ (Refinement 5) collapses pass A's two FPU passes into one whenever the
// chunking allows it (NW == 1, no partial-W mask): the FPU's accumulate-into-DEST
// mode turns `square` into `mul_tiles(x, x, acc_to_dest)` over a sticky D0, so
// Sum_w x_w^2 falls out of the SAME pass that computed the squares and no x^2
// block is ever staged through L1. What it publishes is the identical raw
// elementwise accumulator the pairwise-add datapath publishes, so phase 4, the
// finalize and the whole cross-core combine are untouched by it. Measured
// motive: the sharded geometries are MATH-bound (BLOCK_SHARDED (1,1,8192,1024)
// spends 63.0 of 85.2 us on TRISCs alone), so op count is the only lever left.
//
// Under the cross-core W-split (W_SPLIT, §4.2) this core owns only a SLICE of W,
// so pass A stops one step earlier: no chunk finalizes, cb_partials keeps the
// RAW elementwise x^2 accumulator, and a copy publishes it for the writer's
// gather leg. The combine then folds the gathered accumulators with the SAME
// reduce — the local chunk-accumulate, done across cores instead of across
// chunks — and n_reduced stays the grand total W. Its shape is the CW1 x CW2
// topology knob (Refinement 3): CW2 == 1 is one flat fold on the root over all
// CW tiles; CW2 > 1 stages it, so each row LEADER folds CW1 tiles raw (again
// never finalizing) and the root finalizes over CW2 row-sums. Phase 4 onwards is
// byte identical on every core; only the producer of cb_rms_sum changes (the
// reader's multicast receive instead of phase 3).
//
// Every loop trip count and every helper block shape is a function of the block
// knobs (HT_BLOCK / WT_CHUNK / NW / CW) — never of a whole-op dimension.
//
// PER-STAGE INSTRUMENTATION (permanent, Perf 1). Every stage boundary carries a
// MaybeDeviceZoneScope: cmp_gamma_tilize / cmp_tilize_a / cmp_wait_x /
// cmp_square / cmp_rowsum / cmp_publish / cmp_combine / cmp_rsqrt / cmp_scale /
// cmp_gamma_mul / cmp_tilize_b / cmp_untilize. The macro is free when the
// profiler is off (perf_instrumentation.hpp's durability contract) — NEVER
// remove one, and extend the set to any new predicate-guarded path so per-stage
// observability never regresses. The compute kernel is compiled three times, so
// each zone reports separately on UNPACK / MATH / PACK: that split is what tells
// an FPU-op-count problem apart from an unpack/pack-throughput problem.
//
// All compute goes through ttnn/cpp/ttnn/kernel_lib helpers. Phases 2, 5 and 6
// drop from the `square`/`mul` convenience wrappers to `eltwise_chain` directly
// in the resident regimes, because only the chain surface exposes the
// per-operand TileOffset the resident-block fast path needs (the convenience
// wrappers do not forward it). That is a helper *overload* choice, not a raw-LLK
// substitution: the same BinaryFpu + PackTile elements the wrappers emit.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace {
constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_input_rm = 3;
constexpr uint32_t cb_gamma_rm = 4;
constexpr uint32_t cb_ones = 5;
constexpr uint32_t cb_group_partials = 6;
constexpr uint32_t cb_rms_mean = 7;
constexpr uint32_t cb_partial_out = 8;
constexpr uint32_t cb_group_partials2 = 9;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_output_rm = 17;
constexpr uint32_t cb_x_squared = 24;
constexpr uint32_t cb_partials = 25;
constexpr uint32_t cb_rms_sum = 26;
constexpr uint32_t cb_rms_recip = 27;
constexpr uint32_t cb_scaled = 28;
}  // namespace

namespace ckl = compute_kernel_lib;

// ===========================================================================
// Perf 1 — phase 4's SFPU scoped to the lanes phase 5 actually reads.
// ===========================================================================
//
// MEASURED MOTIVE. Phase 4 (`AddUnary(eps)` then `Rsqrt` over the REDUCE_ROW
// statistic) was the op's single largest real compute item: 906 ns per tile,
// 29.0 us per core on the BLOCK_SHARDED (1,1,8192,1024) profile = 38 % of that
// kernel — measured with the cross-core combine ABLATED AWAY, so it is work, not
// waiting. The tournament bench (perf_experiments/rsqrt_lane_and_window)
// established by ablation that the 906 ns is ~100 % per-tile SFPU LANE work and
// ~0 % per-window overhead: blocking the chain is a measured NULL (1.00x), and
// the fitted cost is 23 ns per 32-lane accurate-rsqrt vector against an 81 ns
// copy+pack floor.
//
// THE WASTE. `cb_rms_sum` holds a REDUCE_ROW result, so only COLUMN 0 is
// meaningful (op_design.md §4.1: "1 tile per tile-row, col-0 valid"). Its sole
// consumer is phase 5's `BinaryFpu<..., cb_rms_recip, Mul, BroadcastDim::Col>`,
// i.e. `mul_tiles_bcast<BroadcastType::COL>`, which reproduces column 0 across
// the row and never reads columns 1..31. The stock chain nevertheless computed
// all 32 vector ops per tile per pass, twice (two separate elements = two
// separate SFPU passes). 8 vector ops in ONE pass produce the identical
// column 0: 912.7 -> 258.3 ns/tile, 3.53x, measured at every ht in {1,2,4,8,16}
// (2.95x-3.60x), at every CB format pair and at BOTH DEST modes.
//
// SAFETY PRECONDITION, verified on device rather than argued: the bench fed
// `mul_tiles_bcast<COL>` a tile whose columns 1..31 held poison (7.5e3 / 3e-4)
// and asserted the product over the WHOLE output tile — max rel-err 0.0078,
// pure bf16 rounding. The broadcast reads column 0 only, so leaving columns
// 1..31 of `cb_rms_recip` unwritten provably cannot change the op's output. If a
// future refinement ever gives `cb_rms_recip` a second consumer that reads other
// lanes, this element must revert to a full-tile scope.
//
// RAW-LLK JUSTIFICATION (required so a later helper-usage pass does not "fix"
// this back and undo the win). Two mechanisms are not reachable through the
// stock helpers today:
//   (a) SFPU WORK SCOPE — `rsqrt_tile` hardcodes `VectorMode::RC` and
//       `ITERATIONS = 8`, and `add_unary_tile` likewise; neither the compute-API
//       wrapper nor the `Rsqrt`/`AddUnary` chain elements expose a vector-mode,
//       iteration-count or DEST-address-stride knob.
//   (b) PASS FUSION — `AddUnary` and `Rsqrt` are separate chain elements, hence
//       separate SFPU passes with separate DEST-address setup + STALLWAIT and
//       separate full walks. There is no "unary op with a pre-added scalar"
//       element to compose.
// The BODY is the stock accurate rsqrt kernel verbatim:
// `_calculate_sqrt_body_<APPROX, RECIPROCAL=true, FAST_APPROX=false>` plus the
// `!fp32_dest_acc_en` round-to-nearest store — exactly what
// `calculate_rsqrt<APPROX, 8, DST_ACCUM_MODE, false, false>` runs. Same
// function, same precision, fewer lanes. The precision contract
// (`fp32_dest_acc_en` / `math_fidelity` / `math_approx_mode` / dtypes) is
// untouched; folding `+eps` into the body's argument in fact removes one bf16
// DEST round trip, so the fused result is marginally MORE accurate (measured
// col-0 PCC 0.9999968 vs the stock chain's 0.9999967).
// Everything else stays on the helper surface: this is a `ckl::UnaryOp` CRTP
// element, so `eltwise_chain` still owns the CB lifecycle, the dtype reconfig
// and the dst-sync window — only the SFPU body is ours.
//
// PREDICATE. Compile-time and architectural only: `_calculate_sqrt_body_` and
// the SFPU `Converter` exist on Wormhole B0 and Blackhole but not on Quasar, so
// Quasar keeps the stock two-element chain, byte-identical. There is no
// shape/dtype/layout guard because the enabling condition is a structural
// invariant of the op, not a property of the input.
#if defined(ARCH_QUASAR)
#define RMS_NORM_COL0_RSQRT 0
#else
#define RMS_NORM_COL0_RSQRT 1
#endif

#if RMS_NORM_COL0_RSQRT
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"            // _calculate_sqrt_body_
#include "sfpu/ckernel_sfpu_converter.h"  // Converter::as_float
#endif

namespace {
#ifdef TRISC_MATH
// The SFPU walks a face as [rg0-even, rg0-odd, rg1-even, rg1-odd, ...]; column 0
// lives only in the EVEN-parity vectors, so visit offsets 0,2,4,6 and skip the
// odd ones. Net dst_reg advance is +8 == the stock ITERATIONS=8, so
// `VectorMode::C`'s face-0 -> face-2 stepping composes unchanged and column 0 is
// covered for all 32 rows in 4 vector ops per face instead of 8.
template <int NVEC, int STRIDE>
sfpi_inline void rms_norm_rsqrt_add_col0_body(uint32_t eps_bits) {
    const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(eps_bits);
    for (int d = 0; d < NVEC; d++) {
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(
            sfpi::dst_reg[0] + eps);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += STRIDE;
    }
}
#endif

/// `rsqrt(x + eps)` in ONE SFPU pass, scoped to the vector ops holding COLUMN 0.
/// For a REDUCE_ROW statistic consumed through `BroadcastDim::Col` only.
template <ckl::Dst Slot = ckl::Dst::D0>
struct RsqrtAddUnaryColZero : ckl::UnaryOp<RsqrtAddUnaryColZero<Slot>, Slot> {
    uint32_t eps_bits;
    constexpr explicit RsqrtAddUnaryColZero(uint32_t e) noexcept : eps_bits(e) {}
    static ALWI void init() { rsqrt_tile_init(); }  // programs the shared sqrt vConst*Prgm constants
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        const uint32_t idst = ckl::to_u32(Slot) + slot_offset;
        const uint32_t eps = eps_bits;
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            [eps]() { rms_norm_rsqrt_add_col0_body<4, 2>(eps); }, idst, ckernel::VectorMode::C)));
    }
};
}  // namespace
#endif  // RMS_NORM_COL0_RSQRT

void kernel_main() {
    // ---- regime flags (§5.2) ----
    constexpr bool IS_RM = get_compile_time_arg_val(0) != 0;
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(1) != 0;
    constexpr bool IS_RM_GAMMA = get_compile_time_arg_val(2) != 0;
    constexpr bool X_RESIDENT = get_compile_time_arg_val(3) != 0;
    constexpr bool GAMMA_RESIDENT = get_compile_time_arg_val(4) != 0;
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(5) != 0;
    // ---- block knobs (§1.2) ----
    constexpr uint32_t WT = get_compile_time_arg_val(6);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(7);
    constexpr uint32_t WT_LAST = get_compile_time_arg_val(8);
    constexpr uint32_t NW = get_compile_time_arg_val(9);
    constexpr uint32_t HT_BLOCK = get_compile_time_arg_val(10);
    // W-chunks the reader coalesces per push on the resident TILE path (see
    // rms_norm_program_descriptor._x_read_chunks). The cumulative wait below has
    // to be quantized to this, since that is the granularity data becomes
    // visible at. Always 1 on the RM path (compute's own tilize is the producer).
    constexpr uint32_t X_READ_CHUNKS = IS_RM ? 1u : get_compile_time_arg_val(11);
    // ---- geometry ----
    constexpr uint32_t W_VALID_LAST = get_compile_time_arg_val(12);
    constexpr uint32_t N_REDUCED = get_compile_time_arg_val(13);  // true element count == W
    // ---- cross-core W-split (§4.2) ----
    constexpr bool W_SPLIT = get_compile_time_arg_val(14) != 0;
    constexpr uint32_t CW = get_compile_time_arg_val(15);   // cores per combine group
    constexpr uint32_t CW1 = get_compile_time_arg_val(16);  // stage-1 fan-in (row -> leader)
    constexpr uint32_t CW2 = get_compile_time_arg_val(17);  // stage-2 fan-in (leaders -> root)
    constexpr bool TWO_STAGE = CW2 > 1;
    static_assert(CW1 * CW2 == CW, "combine stages must tile CW");
    // ---- fused square-accumulate (Refinement 5) ----
    // Collapses phases 2 and 3 into one FPU pass: mul_tiles(x, x, acc_to_dest)
    // over a tile-row leaves Sum_w x_w^2 in DEST, which IS the raw elementwise
    // accumulator the finalize (and the cross-core combine) already consume.
    // The host owns the predicate — see FUSE_SQUARE_ACCUM in the program
    // descriptor for the two structural preconditions it enforces (NW == 1, so
    // the accumulator never has to survive a tile_regs_acquire; and no
    // partial-W mask, which lives on the reduce helper's scaler hook).
    constexpr bool FUSE_SQ = get_compile_time_arg_val(18) != 0;
    static_assert(!FUSE_SQ || NW == 1, "fused square-accumulate requires NW == 1");
    static_assert(!FUSE_SQ || !HAS_PARTIAL_W, "fused square-accumulate cannot apply the partial-W mask");
    // ---- combine_latency_hiding (Perf 2 tournament) ----
    // CLH_VARIANT: 0 baseline (byte-identical), 1 prefetch_a, 2 defer_passb.
    // CLH_ELIGIBLE: host-derived predicate (W_SPLIT && FUSE_SQ && !TWO_STAGE &&
    // NW==1 && !IS_RM && SHARDED_IN && SHARDED_OUT) -- precomputed on the host
    // because sharded_in/sharded_out never made it into this kernel's own CT
    // prefix. Any variant != 0 outside this predicate falls back to the
    // byte-identical loop below (predicate-guarded fast path, same style as
    // RMS_NORM_COL0_RSQRT / the gamma broadcast).
    constexpr uint32_t CLH_VARIANT = get_compile_time_arg_val(19);
    constexpr bool CLH_ELIGIBLE = get_compile_time_arg_val(20) != 0;
    constexpr bool CLH_PIPELINE = CLH_ELIGIBLE && (CLH_VARIANT != 0);

    static_assert(WT_LAST == WT_CHUNK, "compute assumes uniform chunk widths");
    static_assert(NW * WT_CHUNK == WT, "chunking must tile Wt exactly");
    static_assert(!(NW > 1 && HT_BLOCK > 1), "R7: NW > 1 requires HT_BLOCK == 1");
    static_assert(X_READ_CHUNKS >= 1 && NW % X_READ_CHUNKS == 0, "read batch must tile NW");

    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);
    const uint32_t eps_bits = get_arg_val<uint32_t>(1);
    const uint32_t is_root = get_arg_val<uint32_t>(2);
    const uint32_t is_last_w_core = get_arg_val<uint32_t>(3);
    const uint32_t is_leader = get_arg_val<uint32_t>(4);

    // Filler core (inside a group's multicast rectangle, owns no data).
    if (num_tile_rows == 0) {
        return;
    }

    compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles);

    // ---- phase 0a: RM gamma tilized once and held resident -----------------
    if constexpr (HAS_GAMMA && GAMMA_RESIDENT && IS_RM_GAMMA) {
        MaybeDeviceZoneScope("cmp_gamma_tilize");
        for (uint32_t wc = 0; wc < NW; ++wc) {
            ckl::tilize<WT_CHUNK, cb_gamma_rm, cb_gamma>(/*num_blocks=*/1, /*total_input_pages=*/1);
        }
    }
    if constexpr (HAS_GAMMA && GAMMA_RESIDENT) {
        // R8: waited once, held for the whole kernel, never popped.
        cb_wait_front(cb_gamma, WT);
    }

    // Non-tile-aligned W: the 0/1 mask tile the reader filled zeroes the padded
    // lanes of the LAST reduce-dim tile. n_reduced stays the true count (W).
    // Under a W-split only the core whose slice ENDS on the tensor's last
    // W-tile owns that tile, so only it applies the mask.
    const auto partial = (HAS_PARTIAL_W && (!W_SPLIT || is_last_w_core != 0))
                             ? ckl::ReducePartialScaler::partial_mask(W_VALID_LAST, 0)
                             : ckl::ReducePartialScaler::none();

    constexpr uint32_t cb_scale_out = HAS_GAMMA ? cb_scaled : cb_output_tiles;
    // Where the fused square-accumulate publishes its raw Sum_w x^2. Under a
    // W-split that tile is the writer's gather payload and goes straight to
    // cb_partial_out (one producer, one consumer, exactly as the NW == 1
    // pairwise path already did); otherwise it stays compute-internal and the
    // fold below reads it back out of cb_partials.
    constexpr uint32_t cb_accum = W_SPLIT ? cb_partial_out : cb_partials;

    // Tiles per DEST-sync window (Refinement 5). `EltwiseShape`'s block_size
    // defaults to 1, and at 1 the chain runs a WHOLE
    // tile_regs_acquire/commit/wait/release round — plus a pack phase — around
    // every single tile. examples/compute_block_size measures that fixed
    // per-window cost at ~1.6 us per extra pass, 1.65x end to end.
    //
    // This is not a guessed constant: eltwise_chain clamps block_size to the
    // chain's OWN compile-time DEST capacity (chain_max_block_v =
    // DEST_AUTO_LIMIT / lane_width) and to 1 for any chain whose operand
    // policies are per-tile, so asking for the register file's size is exactly
    // "the coarsest block that fits DEST", re-derived per chain and per
    // fp32_dest_acc_en / dst_full_sync_en setting.
    constexpr uint32_t DEST_BLOCK = ckl::DEST_AUTO_LIMIT;
    // A REDUCE_ROW result is column-shaped, so it broadcasts back across
    // columns via BroadcastDim::Col (eltwise_chain.hpp:526-528).
    constexpr auto rms_kind = (HT_BLOCK > 1) ? ckl::OperandKind::Col : ckl::OperandKind::Scalar;
    constexpr auto gamma_kind = (HT_BLOCK > 1) ? ckl::OperandKind::Row : ckl::OperandKind::Block;
    constexpr auto x_life = X_RESIDENT ? ckl::InputLifecycle::CallerManaged : ckl::InputLifecycle::Bulk;

    const uint32_t num_row_blocks = (num_tile_rows + HT_BLOCK - 1) / HT_BLOCK;

    // =======================================================================
    // combine_latency_hiding PIPELINED PATH (prefetch_a / defer_passb).
    // =======================================================================
    //
    // Only reachable when CLH_ELIGIBLE (host-checked: W_SPLIT && FUSE_SQ &&
    // !TWO_STAGE && NW==1 && !IS_RM && SHARDED_IN && SHARDED_OUT — the exact
    // BLOCK_SHARDED focus regime) AND a pipelined variant was requested. Every
    // other regime/variant combination falls through to the ORIGINAL loop
    // below, byte-identical.
    //
    // MECHANISM. do_pass_a/do_stall_rsqrt/do_pass_b are the SAME three
    // sub-phases the original loop runs per row-block, extracted so they can
    // be called OUT OF hb-adjacent order. Two things make that safe:
    //
    //   1. cb_input_tiles is read via an ABSOLUTE TileOffset
    //      (hb*HT_BLOCK*WT), not the "front + implicit per-hb pop" addressing
    //      the original loop relies on -- so any hb's window is reachable
    //      regardless of which hb was processed most recently. This requires
    //      deferring EVERY pop on cb_input_tiles to the ONE cleanup pop after
    //      the whole loop (the shard is pushed once, upfront, by the reader
    //      before this kernel's loop even starts, so a single upfront
    //      cb_wait_front covers every row-block this loop will ever touch).
    //   2. cb_partial_out is declared at clh_pipeline_depth*HT_BLOCK pages
    //      (program descriptor), so hb+1's pass-A push never blocks on the
    //      writer having drained hb's -- it may not have yet.
    //
    // Every OTHER CB (cb_group_partials, cb_rms_mean, cb_rms_sum,
    // cb_rms_recip, cb_scaled, cb_output_tiles) is produced/consumed by the
    // reader/writer/compute in EXACTLY the same relative FIFO order as the
    // original loop -- do_pass_a/do_stall_rsqrt/do_pass_b are each still
    // called once per hb, in ascending hb order, for each CB's own producer or
    // consumer role. Only the WALL-CLOCK INTERLEAVING across the three
    // sub-phases moves. Neither the reader nor the writer kernel changes.
    if constexpr (CLH_PIPELINE) {
        // NOTE: deliberately NOT a static_assert here. `kernel_main` is not a
        // template, so `if constexpr`'s "discarded statement" exemption for
        // static_assert does not apply -- a static_assert on
        // (W_SPLIT && X_RESIDENT && ...) would fire unconditionally for EVERY
        // regime this kernel is ever compiled for, including the ones where
        // CLH_PIPELINE is correctly false but those individual flags do not
        // happen to line up (e.g. a WIDTH_SHARDED decode cell where
        // GAMMA_RESIDENT is false). Measured: this broke every non-focus case
        // the first time it was added. CLH_ELIGIBLE (host) is the sole gate;
        // trust it.

        // The whole per-core shard is already resident (pushed once, upfront,
        // by rdr_shard_publish) -- one wait covers every hb this loop touches.
        cb_wait_front(cb_input_tiles, num_tile_rows * WT);

        auto block_ht = [&](uint32_t hb) -> uint32_t {
            uint32_t ht = num_tile_rows - hb * HT_BLOCK;
            return ht > HT_BLOCK ? HT_BLOCK : ht;
        };

        // ---- pass A: square + DEST-accumulate row-block `hb`'s raw Sum_w x^2,
        // packed straight into cb_partial_out (this regime is always FUSE_SQ
        // && W_SPLIT). Absolute TileOffset `hb*HT_BLOCK*WT` into the resident
        // cb_input_tiles strip -- see mechanism note (1) above.
        auto do_pass_a = [&](uint32_t hb) {
            const uint32_t ht = block_ht(hb);
            const uint32_t row_base = hb * HT_BLOCK * WT;
            const auto ablk = ckl::EltwiseShape::grid(ht, WT_CHUNK, DEST_BLOCK);
            MaybeDeviceZoneScope("cmp_square");
            ckl::eltwise_chain(
                ablk,
                ckl::BinaryFpu<
                    cb_input_tiles,
                    cb_input_tiles,
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::None,
                    ckl::InputLifecycle::CallerManaged,
                    ckl::InputLifecycle::CallerManaged,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::Dst::D0,
                    ckl::OperandKind::Block,
                    ckl::OperandKind::Block,
                    ckl::TileOffset::Set,
                    ckl::TileOffset::Set,
                    ckl::DestAccumulation::Enabled>{row_base, row_base},
                ckl::PackTile<cb_accum, ckl::OutputLifecycle::DestAccumulation>{});
        };

        // ---- the combine round trip's local half (root's fold, flat only --
        // TWO_STAGE is excluded by CLH_ELIGIBLE) + phase 4's rsqrt. This is
        // exactly the STALL the idea targets: cb_rms_sum's Streaming CopyTile
        // blocks on the reader's rdr_mcast, which itself blocks on the writer
        // having shipped this row-block's partial and every sibling in the
        // group having done the same.
        auto do_stall_rsqrt = [&](uint32_t hb) {
            const uint32_t ht = block_ht(hb);
            {
                MaybeDeviceZoneScope("cmp_combine");
                if (is_root) {
                    ckl::reduce_mean<
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_group_partials,
                        cb_ones,
                        cb_rms_mean,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        ckl::ReduceInputBlockShape::of(ht, CW1, 1),
                        N_REDUCED,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::NoAccumulation{},
                        ckl::ReducePartialScaler::none());
                    if (ht < HT_BLOCK) {
                        cb_pop_front(cb_group_partials, (HT_BLOCK - ht) * CW1);
                    }
                }
            }
            {
                MaybeDeviceZoneScope("cmp_rsqrt");
#if RMS_NORM_COL0_RSQRT
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(ht),
                    ckl::CopyTile<cb_rms_sum, ckl::Dst::D0, ckl::InputLifecycle::Streaming>{},
                    RsqrtAddUnaryColZero<ckl::Dst::D0>{eps_bits},
                    ckl::PackTile<cb_rms_recip, ckl::OutputLifecycle::Streaming>{});
#else
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(ht),
                    ckl::CopyTile<cb_rms_sum, ckl::Dst::D0, ckl::InputLifecycle::Streaming>{},
                    ckl::AddUnary<ckl::Dst::D0>{eps_bits},
                    ckl::Rsqrt<>{},
                    ckl::PackTile<cb_rms_recip, ckl::OutputLifecycle::Streaming>{});
#endif
            }
            if (ht < HT_BLOCK) {
                cb_pop_front(cb_rms_sum, HT_BLOCK - ht);
            }
        };

        // ---- pass B: scale by 1/rms, then gamma. Same absolute TileOffset
        // trick on cb_input_tiles' read; cb_rms_recip/cb_scaled/cb_output_tiles
        // keep their normal FIFO discipline since a given hb's B is still
        // pushed/popped as ONE atomic unit, just moved in time.
        auto do_pass_b = [&](uint32_t hb) {
            const uint32_t ht = block_ht(hb);
            const uint32_t row_base = hb * HT_BLOCK * WT;
            const auto bblk = ckl::EltwiseShape::grid(ht, WT_CHUNK, DEST_BLOCK);
            {
                MaybeDeviceZoneScope("cmp_scale");
                ckl::eltwise_chain(
                    bblk,
                    ckl::BinaryFpu<
                        cb_input_tiles,
                        cb_rms_recip,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Col,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::InputLifecycle::HeldBulk,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        rms_kind,
                        ckl::TileOffset::Set,
                        ckl::TileOffset::Unset>{row_base, 0},
                    ckl::PackTile<cb_scale_out, ckl::OutputLifecycle::Chunked>{});
            }
            {
                MaybeDeviceZoneScope("cmp_gamma_mul");
                ckl::eltwise_chain(
                    bblk,
                    ckl::BinaryFpu<
                        cb_scaled,
                        cb_gamma,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Row,
                        ckl::InputLifecycle::Bulk,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        gamma_kind,
                        ckl::TileOffset::Unset,
                        ckl::TileOffset::Set>{0, 0},
                    ckl::PackTile<cb_output_tiles, ckl::OutputLifecycle::Chunked>{});
            }
            cb_pop_front(cb_rms_recip, ht);
        };

        if constexpr (CLH_VARIANT == 1) {
            // prefetch_a: A(0); for hb { if hb+1 exists: A(hb+1); stall+rsqrt(hb);
            // passB(hb) }. Pulls the NEXT block's pass A as early as possible --
            // right after the CURRENT block's own pass A -- so hb+1's combine
            // round trip starts ticking while this core still has (the rest of
            // hb's own stall, if any + rsqrt(hb) + passB(hb) + the FOLLOWING
            // prefetch) queued ahead of the next wait.
            do_pass_a(0);
            for (uint32_t hb = 0; hb < num_row_blocks; ++hb) {
                if (hb + 1 < num_row_blocks) {
                    do_pass_a(hb + 1);
                }
                do_stall_rsqrt(hb);
                do_pass_b(hb);
            }
        } else {
            // defer_passb (CLH_VARIANT == 2): A(0); stall+rsqrt(0); for hb {
            // if hb+1 exists: A(hb+1); passB(hb); if hb+1 exists:
            // stall+rsqrt(hb+1) }. The literal reading of "overlap passB(hb)
            // with hb+1's combine": A(hb+1) is issued right after rsqrt(hb),
            // not before it, so the overlap window is passB(hb) alone rather
            // than the wider window prefetch_a gets.
            do_pass_a(0);
            do_stall_rsqrt(0);
            for (uint32_t hb = 0; hb < num_row_blocks; ++hb) {
                if (hb + 1 < num_row_blocks) {
                    do_pass_a(hb + 1);
                }
                do_pass_b(hb);
                if (hb + 1 < num_row_blocks) {
                    do_stall_rsqrt(hb + 1);
                }
            }
        }

        // ---- cleanup: the ONE deferred pop covering every hb's window -------
        cb_pop_front(cb_input_tiles, num_tile_rows * WT);
        return;
    }

    // =======================================================================
    // ORIGINAL loop (baseline; also the fallback for any regime/variant
    // combination CLH_PIPELINE does not cover). Byte-identical to the real
    // op's rms_norm_compute.cpp.
    // =======================================================================
    for (uint32_t hb = 0; hb < num_row_blocks; ++hb) {
        uint32_t ht = num_tile_rows - hb * HT_BLOCK;
        if (ht > HT_BLOCK) {
            ht = HT_BLOCK;
        }
        const auto blk = ckl::EltwiseShape::grid(ht, WT_CHUNK, DEST_BLOCK);

        // ================= pass A: mean(x^2) over the whole W ==============
        for (uint32_t wc = 0; wc < NW; ++wc) {
            if constexpr (IS_RM) {
                // Resident regime: this fills the strip in place, chunk by chunk,
                // so pass B needs no re-tilize. Streaming: one block at a time.
                MaybeDeviceZoneScope("cmp_tilize_a");
                ckl::tilize<WT_CHUNK, cb_input_rm, cb_input_tiles>(ht, ht * 32u);
            }
            const uint32_t x_base = wc * WT_CHUNK;
            if constexpr (X_RESIDENT) {
                MaybeDeviceZoneScope("cmp_wait_x");
                // R8: CallerManaged — the chain neither waits nor pops; we do
                // both. Waiting CUMULATIVELY (rather than for the whole strip
                // upfront) is what lets the producer stay a batch ahead of
                // compute. Rounded UP to the producer's push granularity:
                // X_READ_CHUNKS == NW collapses to one wait for the full strip.
                // NW > 1 => HT_BLOCK == 1 (R7), so the strip is one flat Wt
                // strip and chunk wc occupies [wc*WT_CHUNK, +WT_CHUNK).
                const uint32_t batches_ready = (wc / X_READ_CHUNKS) + 1u;
                cb_wait_front(cb_input_tiles, batches_ready * X_READ_CHUNKS * ht * WT_CHUNK);
            }

            // ---- phases 2+3 FUSED: DEST-accumulated x^2 --------------------
            //
            // One pass instead of two. `DestAccumulation::Enabled` pins the
            // BinaryFpu to a sticky D0 that eltwise_chain holds across an outer
            // row's whole Wt, so the chunk's W-tiles accumulate in DEST and are
            // packed ONCE per tile-row. What lands is the raw elementwise
            // Sum_w x_w^2 — byte-for-byte the object the pairwise-add datapath
            // publishes with `Accumulate::at` (never `at_last`), so neither the
            // local finalize below nor the cross-core combine can tell the
            // difference. Under a W-split that raw tile is the gather payload
            // and NOTHING else is owed here; otherwise the within-tile fold and
            // the 1/N still have to run, now over `ht` tiles instead of
            // `ht * WT_CHUNK`.
            if constexpr (FUSE_SQ) {
                {
                    MaybeDeviceZoneScope("cmp_square");
                    if constexpr (X_RESIDENT) {
                        ckl::eltwise_chain(
                            blk,
                            ckl::BinaryFpu<
                                cb_input_tiles,
                                cb_input_tiles,
                                ckl::BinaryFpuOp::Mul,
                                ckl::BroadcastDim::None,
                                ckl::InputLifecycle::CallerManaged,
                                ckl::InputLifecycle::CallerManaged,
                                ckl::BinaryDataFormatReconfig::Input,
                                ckl::Dst::D0,
                                ckl::OperandKind::Block,
                                ckl::OperandKind::Block,
                                ckl::TileOffset::Set,
                                ckl::TileOffset::Set,
                                ckl::DestAccumulation::Enabled>{x_base, x_base},
                            ckl::PackTile<cb_accum, ckl::OutputLifecycle::DestAccumulation>{});
                    } else {
                        ckl::eltwise_chain(
                            blk,
                            ckl::BinaryFpu<
                                cb_input_tiles,
                                cb_input_tiles,
                                ckl::BinaryFpuOp::Mul,
                                ckl::BroadcastDim::None,
                                ckl::InputLifecycle::Bulk,
                                ckl::InputLifecycle::Bulk,
                                ckl::BinaryDataFormatReconfig::Input,
                                ckl::Dst::D0,
                                ckl::OperandKind::Block,
                                ckl::OperandKind::Block,
                                ckl::TileOffset::Unset,
                                ckl::TileOffset::Unset,
                                ckl::DestAccumulation::Enabled>{},
                            ckl::PackTile<cb_accum, ckl::OutputLifecycle::DestAccumulation>{});
                    }
                }
                if constexpr (!W_SPLIT) {
                    MaybeDeviceZoneScope("cmp_rowsum");
                    // The fold the accumulate deliberately skipped: one
                    // within-tile REDUCE_ROW per tile-row, then 1/n_reduced.
                    // of(ht, 1, 1) — the W extent is already summed away.
                    ckl::reduce_mean<
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_partials,
                        cb_scaler,
                        cb_rms_sum,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        ckl::ReduceInputBlockShape::of(ht, 1, 1),
                        N_REDUCED,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::NoAccumulation{},
                        ckl::ReducePartialScaler::none());
                }
            } else {
                // ---- phase 2: x^2 ----
                {
                    MaybeDeviceZoneScope("cmp_square");
                    if constexpr (X_RESIDENT) {
                        ckl::eltwise_chain(
                            blk,
                            ckl::BinaryFpu<
                                cb_input_tiles,
                                cb_input_tiles,
                                ckl::BinaryFpuOp::Mul,
                                ckl::BroadcastDim::None,
                                ckl::InputLifecycle::CallerManaged,
                                ckl::InputLifecycle::CallerManaged,
                                ckl::BinaryDataFormatReconfig::Input,
                                ckl::Dst::D0,
                                ckl::OperandKind::Block,
                                ckl::OperandKind::Block,
                                ckl::TileOffset::Set,
                                ckl::TileOffset::Set>{x_base, x_base},
                            ckl::PackTile<cb_x_squared, ckl::OutputLifecycle::Chunked>{});
                    } else {
                        ckl::square<
                            cb_input_tiles,
                            cb_x_squared,
                            ckl::InputLifecycle::Bulk,
                            ckl::OutputLifecycle::Chunked,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::PackTileReconfig::Output,
                            ckl::OperandKind::Block>(blk);
                    }
                }

                MaybeDeviceZoneScope("cmp_rowsum");
                // ---- phase 3: chunked SUM -> mean on the finalizing chunk ----
                //
                // Under a W-split NO chunk finalizes: this core owns only a SLICE of
                // W, so both the within-tile fold and the 1/N are premature. Every
                // chunk therefore uses Accumulate::at (never at_last), which leaves
                // cb_partials holding the RAW elementwise-accumulated x^2 tile — the
                // exact object the cross-core combine needs. Shipping the *reduced*
                // tile instead would be wrong: AccumulateViaAdd's finalize writes the
                // row sum into column 0 and leaves the surviving x^2 lanes in columns
                // 1..31, so a second REDUCE_ROW over such tiles double-counts them
                // (measured: mean(x^2) of an all-ones W=64 came out 8.75, not 1.0).
                const auto rshape = ckl::ReduceInputBlockShape::of(ht, WT_CHUNK, 1);
                const bool finalize_here = !W_SPLIT && (wc + 1 == NW);
                if constexpr (NW == 1) {
                    if constexpr (W_SPLIT) {
                        // Single chunk, so the accumulator is written ONCE and never
                        // reloaded: pack it straight into the writer's gather CB and
                        // skip the republishing copy below. cb_partial_out still has
                        // exactly one producer (this) and one consumer (the writer).
                        ckl::reduce<
                            ckernel::PoolType::SUM,
                            ckernel::ReduceDim::REDUCE_ROW,
                            cb_x_squared,
                            cb_scaler,
                            cb_partial_out,
                            ckl::ReduceInputPolicy::BulkWaitBulkPop,
                            ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                            ckl::ReduceAlgorithm::AccumulateViaAdd>(
                            rshape,
                            ckl::ReduceInputMemoryLayout::contiguous(),
                            ckl::Accumulate::at(cb_partial_out, wc),
                            ckl::NoOp{},
                            partial);
                    } else {
                        ckl::reduce_mean<
                            ckernel::ReduceDim::REDUCE_ROW,
                            cb_x_squared,
                            cb_scaler,
                            cb_rms_sum,
                            ckl::ReduceInputPolicy::BulkWaitBulkPop,
                            ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                            ckl::ReduceAlgorithm::AccumulateViaAdd>(
                            rshape,
                            N_REDUCED,
                            ckl::ReduceInputMemoryLayout::contiguous(),
                            ckl::NoAccumulation{},
                            partial);
                    }
                } else if (finalize_here) {
                    ckl::reduce_mean<
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_x_squared,
                        cb_scaler,
                        cb_rms_sum,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        rshape,
                        N_REDUCED,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::Accumulate::at_last(cb_partials, wc),
                        partial);
                } else {
                    // Non-finalizing chunk. The partial-W mask rides the chunk that
                    // owns the tensor's last W-tile, which under a W-split is this
                    // core's last chunk (and only on the last-W core).
                    const bool last_chunk = (wc + 1 == NW);
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_x_squared,
                        cb_scaler,
                        cb_partials,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        rshape,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::Accumulate::at(cb_partials, wc),
                        ckl::NoOp{},
                        last_chunk ? partial : ckl::ReducePartialScaler::none());
                }
            }
        }

        // Hand the raw accumulator to the writer's gather leg. With NW > 1
        // cb_partials is a compute->compute read-modify-write across the chunk
        // loop, so it cannot ALSO be the writer's CB (single producer / single
        // consumer) and one copy publishes the settled tile per tile-row. With
        // NW == 1 there is no read-modify-write, so the reduce above already
        // packed into cb_partial_out and this whole pass is gone — worth having
        // as its own case because a compute pass costs ~320 ns of fixed
        // overhead (examples/compute_block_size) and this one sits on the
        // combine's serial path, ahead of the gather.
        if constexpr (W_SPLIT && NW > 1) {
            MaybeDeviceZoneScope("cmp_publish");
            ckl::copy<cb_partials, cb_partial_out>(ckl::EltwiseShape::tiles(ht));
        }

        // ========== phase 3b: cross-core combine (W-split only) ============
        // The combine folds the raw slice-accumulators the writers gathered into
        // ONE mean(x^2) per tile-row. That fold is EXACTLY the local chunk
        // accumulate, done across cores instead of across chunks: AccumulateViaAdd
        // elementwise-adds the gathered tiles into DEST, folds the result within
        // the tile ONCE, and applies 1/n_reduced with n_reduced = W, the GRAND
        // total (§4.2 "Finalize"). Gathered tiles are laid out h-major
        // (tile h*fan_in + slot), so of(ht, fan_in) reads them contiguously.
        //
        // With CW2 == 1 there is one fold, on the root, over all CW tiles.
        // With CW2 > 1 it is staged: every LEADER first folds its row's CW1
        // tiles WITHOUT finalizing — Accumulate::at (never at_last) keeps the raw
        // elementwise accumulator, the same object a worker's chunk loop
        // produces, so the second fold cannot double-count the surviving x^2
        // lanes — and republishes it through cb_partial_out for its own writer.
        // The root then finalizes over just the CW2 row-sums.
        if constexpr (W_SPLIT) {
            MaybeDeviceZoneScope("cmp_combine");
            if constexpr (TWO_STAGE) {
                if (is_leader) {
                    // One accumulate call, never reloaded -> pack the row sum
                    // straight back into cb_partial_out for this core's own
                    // writer to ship on to the root (same CB the slice partial
                    // rode: one producer, one consumer, two sequential pushes).
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_group_partials,
                        cb_ones,
                        cb_partial_out,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        ckl::ReduceInputBlockShape::of(ht, CW1, 1),
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::Accumulate::at(cb_partial_out, 0),
                        ckl::NoOp{},
                        ckl::ReducePartialScaler::none());
                    if (ht < HT_BLOCK) {
                        cb_pop_front(cb_group_partials, (HT_BLOCK - ht) * CW1);
                    }
                }
                if (is_root) {
                    ckl::reduce_mean<
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_group_partials2,
                        cb_ones,
                        cb_rms_mean,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        ckl::ReduceInputBlockShape::of(ht, CW2, 1),
                        N_REDUCED,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::NoAccumulation{},
                        ckl::ReducePartialScaler::none());
                    if (ht < HT_BLOCK) {
                        cb_pop_front(cb_group_partials2, (HT_BLOCK - ht) * CW2);
                    }
                }
            } else if (is_root) {
                ckl::reduce_mean<
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_group_partials,
                    cb_ones,
                    cb_rms_mean,
                    ckl::ReduceInputPolicy::BulkWaitBulkPop,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ckl::ReduceAlgorithm::AccumulateViaAdd>(
                    ckl::ReduceInputBlockShape::of(ht, CW1, 1),
                    N_REDUCED,
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::NoAccumulation{},
                    ckl::ReducePartialScaler::none());
                // The reader publishes a fixed HT_BLOCK*CW1 block so the gather
                // slots stay at a constant L1 offset; drop the unused tail.
                if (ht < HT_BLOCK) {
                    cb_pop_front(cb_group_partials, (HT_BLOCK - ht) * CW1);
                }
            }
        }

        // ================= phase 4: 1/sqrt(mean + eps) =====================
        // One dst-sync window for both SFPU ops; the FPU consumer in phase 5
        // reads it back from L1 (DEST reuse measures slower for an FPU consumer).
        // Under a W-split cb_rms_sum is produced by the READER (the root's
        // broadcast), not by phase 3 — every core then finalizes identically.
        {
            MaybeDeviceZoneScope("cmp_rsqrt");
#if RMS_NORM_COL0_RSQRT
            // Perf 1 fast path: one SFPU pass over the 8 vectors that hold
            // column 0 (see the justification above the element). 3.53x measured.
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(ht),
                ckl::CopyTile<cb_rms_sum, ckl::Dst::D0, ckl::InputLifecycle::Streaming>{},
                RsqrtAddUnaryColZero<ckl::Dst::D0>{eps_bits},
                ckl::PackTile<cb_rms_recip, ckl::OutputLifecycle::Streaming>{});
#else
            // Fallback (Quasar): byte-identical to the pre-Perf-1 spelling.
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(ht),
                ckl::CopyTile<cb_rms_sum, ckl::Dst::D0, ckl::InputLifecycle::Streaming>{},
                ckl::AddUnary<ckl::Dst::D0>{eps_bits},
                ckl::Rsqrt<>{},
                ckl::PackTile<cb_rms_recip, ckl::OutputLifecycle::Streaming>{});
#endif
        }
        if constexpr (W_SPLIT) {
            // Same fixed-block contract as the gather: the reader publishes
            // HT_BLOCK pages so the multicast lands at a constant L1 offset.
            if (ht < HT_BLOCK) {
                cb_pop_front(cb_rms_sum, HT_BLOCK - ht);
            }
        }

        // ================= pass B: scale (and gamma), then write ===========
        for (uint32_t wc = 0; wc < NW; ++wc) {
            if constexpr (IS_RM && !X_RESIDENT) {
                MaybeDeviceZoneScope("cmp_tilize_b");
                ckl::tilize<WT_CHUNK, cb_input_rm, cb_input_tiles>(ht, ht * 32u);
            }
            const uint32_t x_base = wc * WT_CHUNK;

            // ---- phase 5: x * (1/rms), broadcast across columns ----
            {
                MaybeDeviceZoneScope("cmp_scale");
                if constexpr (X_RESIDENT) {
                    ckl::eltwise_chain(
                        blk,
                        ckl::BinaryFpu<
                            cb_input_tiles,
                            cb_rms_recip,
                            ckl::BinaryFpuOp::Mul,
                            ckl::BroadcastDim::Col,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::InputLifecycle::HeldBulk,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::Dst::D0,
                            ckl::OperandKind::Block,
                            rms_kind,
                            ckl::TileOffset::Set,
                            ckl::TileOffset::Unset>{x_base, 0},
                        ckl::PackTile<cb_scale_out, ckl::OutputLifecycle::Chunked>{});
                } else {
                    ckl::mul<
                        cb_input_tiles,
                        cb_rms_recip,
                        cb_scale_out,
                        ckl::BroadcastDim::Col,
                        x_life,
                        ckl::InputLifecycle::HeldBulk,
                        ckl::OutputLifecycle::Chunked,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::PackTileReconfig::Output,
                        ckl::OperandKind::Block,
                        rms_kind>(blk);
                }
            }

            // ---- phase 6: * gamma, broadcast down the rows ----
            if constexpr (HAS_GAMMA) {
                MaybeDeviceZoneScope("cmp_gamma_mul");
                if constexpr (IS_RM_GAMMA && !GAMMA_RESIDENT) {
                    ckl::tilize<WT_CHUNK, cb_gamma_rm, cb_gamma>(/*num_blocks=*/1, /*total_input_pages=*/1);
                }
                if constexpr (GAMMA_RESIDENT) {
                    ckl::eltwise_chain(
                        blk,
                        ckl::BinaryFpu<
                            cb_scaled,
                            cb_gamma,
                            ckl::BinaryFpuOp::Mul,
                            ckl::BroadcastDim::Row,
                            ckl::InputLifecycle::Bulk,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::Dst::D0,
                            ckl::OperandKind::Block,
                            gamma_kind,
                            ckl::TileOffset::Unset,
                            ckl::TileOffset::Set>{0, x_base},
                        ckl::PackTile<cb_output_tiles, ckl::OutputLifecycle::Chunked>{});
                } else {
                    ckl::mul<
                        cb_scaled,
                        cb_gamma,
                        cb_output_tiles,
                        ckl::BroadcastDim::Row,
                        ckl::InputLifecycle::Bulk,
                        ckl::InputLifecycle::Bulk,
                        ckl::OutputLifecycle::Chunked,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::PackTileReconfig::Output,
                        ckl::OperandKind::Block,
                        gamma_kind>(blk);
                }
            }

            // ---- phase 7: back to row-major sticks ----
            if constexpr (IS_RM) {
                MaybeDeviceZoneScope("cmp_untilize");
                ckl::untilize<WT_CHUNK, cb_output_tiles, cb_output_rm>(ht);
            }
        }

        // ================= phase 8: release the held CBs ===================
        // R2: cb_rms_recip is HeldBulk across all NW chunks of pass B.
        cb_pop_front(cb_rms_recip, ht);
        if constexpr (X_RESIDENT) {
            cb_pop_front(cb_input_tiles, ht * WT);
        }
    }
}
