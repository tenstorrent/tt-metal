// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BAKE-OFF KERNEL for rms_norm design lamp P4 (idea I5): can the apply
// pass's TWO block-wide FPU passes be fused into ONE, deleting the cb_normed L1
// round trip?
//
// Concept isolation (/perf-lab): nothing here but the apply. Every operand is a
// RESIDENT L1 shard pinned zero-copy under a CB (no DRAM, no NoC, no reader, no
// writer, no reduce, no mcast), so the measured delta is the apply's compute
// alone. `ITERS` repeats the apply so the per-block cost can be recovered as the
// SLOPE of duration vs. iteration count — the focus geometry's block is 3 tiles,
// which is far below the launch floor of a single dispatch.
//
// out[r][c] = x[r][c] * rstd[r] * gamma[c]
//   x     : ROWS_T x COLS resident tiles, row stride COLS   (cb_in)
//   rstd  : ROWS_T COLUMN-0-VALID tiles                     (cb_rstd)
//   gamma : COLS ROW-0-VALID tiles                          (cb_gamma)
//
// VARIANTS (compile-time arg 2):
//   0 BASELINE     the op's current approach, verbatim: x*rstd (BroadcastDim::Col)
//                  -> cb_normed, then normed*gamma (BroadcastDim::Row) -> out.
//                  Two chains, two DEST windows, two packs, one L1 round trip.
//   1 FUSED_RSTD   ONE chain. Prep expands the ROWS_T column-0-valid rstd tiles to
//                  full tiles (UnaryBcast<Col>, ROWS_T tile-ops — cheap, it scales
//                  with the tile-ROW count, not the hidden width), then per tile:
//                  BinaryFpu<x, gamma, Mul, BroadcastDim::Row> -> DEST, then
//                  DestReuseBinary<rstd_full, Mul> in-place on the SAME DEST slot,
//                  then ONE pack. cb_normed is gone.
//   2 FUSED_GAMMA  the same fusion with the operands swapped: prep expands GAMMA
//                  (COLS tile-ops — the expansion Refinement 4 priced and declined),
//                  BinaryFpu<x, rstd, Mul, BroadcastDim::Col> then
//                  DestReuseBinary<gamma_full>.
//   3 FUSED_SFPU   ONE chain, no expansion at all: BinaryFpu<x, rstd, Mul, Col> -> D0,
//                  UnaryBcast<Row, gamma> -> D1, SFPU MulBinary D0*D1 -> D0, one pack.
//                  Uses 2 DEST lanes per tile.
//   4 FOLD_GAMMA   (ROWS_T == 1 only) fold rstd INTO gamma: rstd_full = bcast(rstd),
//                  g'[c] = rstd_full * gamma[c], then out = x * g' with NO broadcast.
//                  Three chains but the apply itself is one plain multiply.
//
// KNOBS: BLK = eltwise_chain block_size (DEST lanes per outer iter — this is what
// amortizes the per-block-iteration init switch a fused chain pays, see below).
// OUT_BULK selects the output CB lifecycle (0 = the op's PerTile/PerTile streaming
// quantum, 1 = Upfront/AtEnd bulk). RECONFIG_MODE gates the per-operand data-format
// reconfig the op leaves Enabled on both passes.
//
// ---------------------------------------------------------------------------
// MEASURED (blackhole_p150b, 1 core unless stated; bf16 / TILE / HiFi2 /
// fp32_dest_acc_en=False; per-block ns = slope of DEVICE KERNEL DURATION over ITERS;
// every row below PASSED the correctness gate, pcc >= 0.99998)
//
//   (rows_t,cols)  baseline  +bulk_out  +blk2  +blk4  +blk6 | FUSED_RSTD(best blk)
//   (1,3)   focus     566       479      510     -     518  |  642   (1.13x SLOWER)
//   (1,16)           1777      1450     1327   1322   1336  | 1813   (1.02x slower)
//   (3,32)           9247      7321     6946   6027   6136  | 9147   (flat)
//   (16,4)           6514      4972     4801   4071   4059  | 6849   (1.05x slower)
//   (1,112)         10669      8546     7428   6916   6753  | 10465  (flat)
//   (1,3) on 64 cores: 567 / 477 / 510 — identical to 1 core (no contention effect).
//
// FUSED_GAMMA is worse everywhere (715 / 2408 / 10352 / 6515 / 14710) — the COLS-sized
// expansion is exactly the cost Refinement 4 predicted. FOLD_GAMMA (rows_t==1 only)
// lands on top of the blocked baseline (621 / 1397 / — / — / 6559), i.e. it buys nothing
// that blocking does not. FUSED_SFPU is CORRECT and the most PRECISE option but 3.4x
// (focus) to 8x (3,32: 73657 ns) slower — the SFPU multiply is not competitive with the
// FPU one here.
//
// So the FUSION IS NOT THE LEVER: the apply's cost is not the pass count or the
// cb_normed round trip, it is the PER-TILE fixed cost of the two chains (output
// reserve/push quantum + DEST window). Fusing removes one pack per tile but pays two MOP
// re-inits per block iteration, which is strictly worse.
// ---------------------------------------------------------------------------
//
// TWO HARD CONSTRAINTS FOUND WHILE MEASURING (both are helper-level, not idea-level):
//  * DestReuseBinary at HiFi2 is INCORRECT when the DEST window uses all 8 lanes
//    (BLK == 8 at fp32_dest_acc_en=False): pcc 0.982, amax 5.6 (junk leaks in), while
//    BLK <= 6 is exact. Suspected cause: the high-fidelity `TTI_INCRWC(CR_D, ...)` at
//    llk_math_eltwise_binary.h:497-503 advances the DEST row counter past the last tile
//    of the bank. Cap BLK at 6 (bf16) if this element is ever adopted.
//  * BLK > 1 must not be paired with ReservePolicy::PerTile on the output: the chain
//    reserves ONE page per block iteration (eltwise_chain.inl:2785) but packs
//    `inner_count` tiles, so the CB overruns and the kernel HANGS. Not static_asserted.
//
// WHY A FUSED CHAIN CANNOT HOIST ITS INIT (the load-bearing constraint):
// `DestReuseBinary` is the ONLY chain element that can consume a DEST-resident
// value as an FPU operand, and it has NO BroadcastDim parameter
// (eltwise_chain.hpp:518-520 vs. BinaryFpu's `BroadcastDim Bcast` at :508) — hence
// the expansion prep. And because the fused chain then holds TWO DIFFERENT
// math-MOP element types, `chain_math_mop_uniform` is false
// (eltwise_chain.inl:2277-2279), so `hoist_math` is false and BOTH elements
// re-emit `init()` + their srcA/srcB reconfig once per BLOCK ITERATION
// (eltwise_chain.inl:2709-2712, called from :3105 inside the `wt_base += block_size`
// loop). At BLK == 1 that is two MOP reprogrammings PER TILE; BLK == 8 amortizes
// them over 8 tiles. This is not a helper bypass — it is why the fused variants are
// measured at several BLK values.
//
// NOT A PRECISION KNOB: fp32_dest_acc_en / math_fidelity / dtypes come from the
// host and are identical for every variant. MEASURED precision, median(got/true) —
// dest-reuse costs NOTHING against the L1 round trip, at either precision corner:
//   bf16 / HiFi2 / fp32_dest_acc_en=False, (16,4): baseline 0.996829, FUSED_RSTD 0.996853
//   fp32 / HiFi2 / fp32_dest_acc_en=True,  (1,16): baseline 0.988258, FUSED_RSTD 0.988279
//   fp32 / HiFi4 / fp32_dest_acc_en=True,  (1,16): baseline 0.997919, FUSED_RSTD 0.997919
//                                                  (pcc 1.000000 both; fused's amax is
//                                                   LOWER, 0.0245 vs 0.0280)
// i.e. Refinement 4's second objection — "dest reuse routes DEST through a Src register
// at bf16, a real precision loss for float32" — is measurably FALSE: both paths land in
// the same 19-bit Src register, so the L1 round trip preserves nothing extra.
// FUSED_SFPU is the one option that is measurably MORE precise (0.999924 / 0.994313 /
// 0.998624 on those three rows) because its second multiply never leaves DEST.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_rstd = 1;
constexpr uint32_t cb_gamma = 2;
constexpr uint32_t cb_exp = 3;   // expanded operand (rstd_full / gamma_full / g')
constexpr uint32_t cb_exp2 = 4;  // FOLD_GAMMA's rstd_full staging
constexpr uint32_t cb_normed = 14;
constexpr uint32_t cb_out = 16;

constexpr uint32_t V_BASELINE = 0;
constexpr uint32_t V_FUSED_RSTD = 1;
constexpr uint32_t V_FUSED_GAMMA = 2;
constexpr uint32_t V_FUSED_SFPU = 3;
constexpr uint32_t V_FOLD_GAMMA = 4;
}  // namespace

void kernel_main() {
    constexpr uint32_t ROWS_T = get_compile_time_arg_val(0);
    constexpr uint32_t COLS = get_compile_time_arg_val(1);
    constexpr uint32_t VARIANT = get_compile_time_arg_val(2);
    constexpr uint32_t BLK = get_compile_time_arg_val(3);
    constexpr uint32_t ITERS = get_compile_time_arg_val(4);
    constexpr bool OUT_BULK = get_compile_time_arg_val(5) != 0;
    // 0 = reconfig off everywhere, 1 = on everywhere (what the op does today),
    // 2 = on for the FIRST stage only (the second stage's operand formats are
    //     unchanged from the first's, so its reconfig is the provably-inert one).
    constexpr uint32_t RECONFIG_MODE = get_compile_time_arg_val(6);

    static_assert(VARIANT != V_FOLD_GAMMA || ROWS_T == 1, "FOLD_GAMMA needs a one-tile-row block");

    constexpr uint32_t N_BLOCK = ROWS_T * COLS;
    constexpr auto RC = (RECONFIG_MODE != 0) ? ckl::DataFormatReconfig::Enabled : ckl::DataFormatReconfig::Disabled;
    // Second-stage operands: dropped under both mode 0 and mode 2.
    constexpr auto RC2 = (RECONFIG_MODE == 1) ? ckl::DataFormatReconfig::Enabled : ckl::DataFormatReconfig::Disabled;

    // The op's apply reads the resident input block with caller-managed policies at
    // row stride CB_W_TILES; here CB_W_TILES == COLS.
    constexpr auto in_spec = ckl::input(
        cb_in, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block, RC, ckl::TileOffset::Strided);
    constexpr auto rstd_spec =
        ckl::input(cb_rstd, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Col, RC);
    constexpr auto gamma_spec =
        ckl::input(cb_gamma, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Row, RC);
    constexpr auto gamma_spec2 =
        ckl::input(cb_gamma, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Row, RC2);

    // Output lifecycle: the op streams per tile (its writer drains a tile-row while
    // compute produces the next); the bulk form reserves the block upfront.
    constexpr auto out_spec = OUT_BULK
                                  ? ckl::output(cb_out, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, RC2)
                                  : ckl::output(cb_out, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, RC2);
    constexpr auto normed_spec =
        OUT_BULK ? ckl::output(cb_normed, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, RC)
                 : ckl::output(cb_normed, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, RC);

    compute_kernel_hw_startup(cb_in, cb_rstd, cb_out);

    // Every input CB is PINNED zero-copy over a resident L1 shard: the data is
    // already there, so one push publishes it for the whole launch. Never popped.
    cb_reserve_back(cb_in, N_BLOCK);
    cb_push_back(cb_in, N_BLOCK);
    cb_reserve_back(cb_rstd, ROWS_T);
    cb_push_back(cb_rstd, ROWS_T);
    cb_reserve_back(cb_gamma, COLS);
    cb_push_back(cb_gamma, COLS);

    const ckl::StridedTileRange src{0, COLS};

    for (uint32_t it = 0; it < ITERS; ++it) {
        // ONE zone per variant (same marker cost on every leg, so the comparison
        // is not contaminated by a differing number of zone executions).
        {
            MaybeDeviceZoneScope("bench_apply");
            if constexpr (VARIANT == V_BASELINE) {
                // ---- the op's current approach, verbatim ----
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(ROWS_T, COLS, BLK),
                    ckl::BinaryFpu<in_spec, rstd_spec, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col>{src},
                    ckl::PackTile<normed_spec>{});
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(ROWS_T, COLS, BLK),
                    ckl::BinaryFpu<
                        ckl::input(
                            cb_normed, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block, RC2),
                        gamma_spec2,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Row>{},
                    ckl::PackTile<out_spec>{});
            } else if constexpr (VARIANT == V_FUSED_RSTD) {
                // prep: ROWS_T column-0-valid rstd tiles -> ROWS_T full tiles.
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(ROWS_T, 1),
                    ckl::UnaryBcast<ckl::BroadcastDim::Col, rstd_spec>{},
                    ckl::PackTile<ckl::output(cb_exp, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, RC)>{});
                // one DEST window, one pack: (x * gamma) * rstd_full
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(ROWS_T, COLS, BLK),
                    ckl::BinaryFpu<in_spec, gamma_spec, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Row>{src},
                    ckl::DestReuseBinary<
                        ckl::input(cb_exp, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Col, RC2),
                        ckl::BinaryFpuOp::Mul,
                        ckl::DestReuseType::DEST_TO_SRCA>{},
                    ckl::PackTile<out_spec>{});
            } else if constexpr (VARIANT == V_FUSED_GAMMA) {
                // prep: COLS row-0-valid gamma tiles -> COLS full tiles.
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(1, COLS),
                    ckl::UnaryBcast<ckl::BroadcastDim::Row, gamma_spec>{},
                    ckl::PackTile<ckl::output(cb_exp, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, RC)>{});
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(ROWS_T, COLS, BLK),
                    ckl::BinaryFpu<in_spec, rstd_spec, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col>{src},
                    ckl::DestReuseBinary<
                        ckl::input(cb_exp, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Row, RC2),
                        ckl::BinaryFpuOp::Mul,
                        ckl::DestReuseType::DEST_TO_SRCA>{},
                    ckl::PackTile<out_spec>{});
            } else if constexpr (VARIANT == V_FUSED_SFPU) {
                // No expansion: gamma rides a second DEST lane and the combine is SFPU.
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(ROWS_T, COLS, BLK),
                    ckl::BinaryFpu<in_spec, rstd_spec, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col, ckl::Dst::D0>{
                        src},
                    // NOT CopyTile: gamma is ROW-0-VALID, so an elementwise SFPU
                    // multiply needs it broadcast down the tile, not copied verbatim
                    // (measured: CopyTile here gives pcc -0.089 — it multiplies by the
                    // structurally-invalid rows).
                    ckl::UnaryBcast<ckl::BroadcastDim::Row, gamma_spec2, ckl::Dst::D1>{},
                    ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
                    ckl::PackTile<out_spec, ckl::Dst::D0>{});
            } else {
                // FOLD_GAMMA — g' = rstd (x) gamma is rank-1, so it is only affordable
                // when the block is ONE tile-row (the decode geometry). The guard is
                // written against VARIANT, not bare `ROWS_T == 1`: `kernel_main` is not a
                // template, so a static_assert inside a DISCARDED `if constexpr` branch is
                // still evaluated and would break every other variant at ROWS_T > 1.
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(1, 1),
                    ckl::UnaryBcast<ckl::BroadcastDim::Col, rstd_spec>{},
                    ckl::PackTile<ckl::output(cb_exp2, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, RC)>{});
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(1, COLS, BLK),
                    ckl::BinaryFpu<
                        ckl::input(
                            cb_exp2, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Scalar, RC),
                        gamma_spec,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Row>{},
                    ckl::PackTile<ckl::output(cb_exp, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, RC)>{});
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(ROWS_T, COLS, BLK),
                    ckl::BinaryFpu<
                        in_spec,
                        ckl::input(cb_exp, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Row, RC2),
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::None>{src},
                    ckl::PackTile<out_spec>{});
            }
        }
        // Recycle the output CB so the next iteration re-runs the same block. Same
        // cost on every variant; excluded from the zone.
        if (it + 1 < ITERS) {
            cb_wait_front(cb_out, N_BLOCK);
            cb_pop_front(cb_out, N_BLOCK);
        }
    }
}
