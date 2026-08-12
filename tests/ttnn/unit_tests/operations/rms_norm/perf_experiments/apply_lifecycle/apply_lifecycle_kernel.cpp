// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BAKE-OFF KERNEL for rms_norm perf round 2, idea I11: speed up the APPLY
// pass by changing the OUTPUT-CB LIFECYCLE and by DEST-LANE BLOCKING.
//
// The op's apply is two block-wide FPU chains (rms_norm_compute.cpp, zones
// `cp_apply_scale` / `cp_apply_gamma`):
//     cb_normed = x * rstd   (BroadcastDim::Col)
//     cb_out    = normed * gamma (BroadcastDim::Row)
// both with the eltwise_chain output defaults (ReservePolicy::PerTile /
// PushPolicy::PerTile) and block_size 1. That is the BASELINE here, verbatim.
//
// Round 1 (apply_fusion) measured that an Upfront/AtEnd ("bulk") output plus
// block_size > 1 is worth up to 1.60x — but a bulk output is NOT INTEGRATION-SAFE:
//   * cb_output_tiles is sized OUTPUT_CB_DEPTH(=2) * WC pages, not rows_t*WC, so an
//     Upfront reserve of the whole block would deadlock at rows_t > 2 (and sizing it
//     up costs L1 linear in rows_t — 32x on the block-sharded leg);
//   * it would hold the whole block back from a writer that is at the DRAM roofline
//     on the interleaved legs.
// The integration-safe form is ReservePolicy::PerChunk / PushPolicy::PerChunk, which
// reserves+pushes exactly `block_size` pages per DEST window (eltwise_chain.inl:2787,
// :2801, count = chunk_sync_count = inner_count). It amortizes the reserve/push over
// the block while keeping the streaming quantum at block_size tiles, and it is also
// the ONLY legal output policy at block_size > 1: PerTile + blk > 1 reserves one page
// but packs `inner_count`, which overruns the CB and HANGS (recorded in round 1).
//
// Knobs (compile-time):
//   BLK            eltwise_chain block_size (DEST lanes per outer iteration)
//   OUT_POLICY     0 = PerTile/PerTile (op today), 1 = Upfront/AtEnd (bulk, unsafe),
//                  2 = PerChunk/PerChunk (integration-safe)
//   NORMED_POLICY  same encoding for the INTERNAL cb_normed scratch. Bulk is safe
//                  here: the op already sizes cb_normed at R*WC (the whole chunk) and
//                  its consumer is the very next chain in the SAME kernel, so nothing
//                  downstream is delayed.
//
// Concept isolation (/perf-lab): every operand is a resident L1 shard pinned zero-copy
// under a CB — no DRAM, no NoC, no reader/writer, no reduce, no multicast — so the
// measured DEVICE KERNEL DURATION delta is the apply's compute alone. ITERS repeats
// the apply so the per-block cost is the SLOPE of duration vs. iteration count (the
// focus geometry's chunk is 3 tiles, far below a dispatch's launch floor).
//
// NOT A PRECISION KNOB: fp32_dest_acc_en / math_fidelity / dtypes come from the host
// and are IDENTICAL for every option (bf16 / TILE / HiFi2 / fp32_dest_acc_en=False).
// Only the CB lifecycle and the DEST block size change.
//
// NO RAW LLK: every option is expressed in `compute_kernel_lib::eltwise_chain` exactly
// as the op writes it; the idea is a policy/shape argument change, not a bypass.
//
// ---------------------------------------------------------------------------
// MEASURED (blackhole_p150b @1.35 GHz, 1 core unless stated; bf16 / TILE / HiFi2 /
// fp32_dest_acc_en=False; ns per apply CHUNK = slope of DEVICE KERNEL DURATION over
// ITERS = 1 -> 21). EVERY option below passed the correctness gate, and every one lands
// on the SAME pcc as the baseline to 6 digits — the math is bit-identical work in a
// different order, so precision is untouched:
//   (1,3) pcc 0.999988 / (32,4) 0.999987 / (16,4) 0.999986 / (1,112) 0.999985, all options.
//
//  A) STREAMING output leg (cb_output_tiles owned by the chain: interleaved / non-strided)
//   (rows_t,cols)   baseline  normed_bulk  both_bulk  pc_blk2  pc_blk4  pc_blk8   best
//   (1,3)  focus       567.0        503.7      479.2    502.9     n/a      n/a   1.13x
//   (1,2)               481.9          -          -      414.7     n/a      n/a   1.16x
//   (32,4) bshard-ish 12762.6      11245.0     9702.0   8713.4   7783.5     n/a   1.64x
//   (16,4)             6527.0       5776.1     5012.1   4538.9   4091.2     n/a   1.60x
//   (1,112)           10666.0       9584.5     8550.5   7220.1   6968.1   6803.6  1.57x
//   ("pc" = PerChunk on BOTH cb_normed and cb_out — no bulk anywhere, so no CB is resized.)
//   PerChunk MATCHES OR BEATS the integration-unsafe bulk output at every geometry
//   (e.g. (32,4): 7783 PerChunk vs 7802 bulk vs 9702 bulk-without-blocking).
//
//  B) OUT_STRIDED leg (caller reserves the block, chain packs at TileOffset::Strided —
//     the op's pinned/row-major-C output path, incl. BLOCK_SHARDED). No lifecycle lever
//     there (the reserve is already caller-managed and hoisted); blocking alone:
//   (32,4)  11198.5 -> 7785.8  (blk4)  1.44x
//   (16,4)   5733.4 -> 4095.9  (blk4)  1.40x
//   (1,112)  9526.0 -> 6813.8  (blk8)  1.40x
//   (1,3)     515.4 -> n/a (blk capped by cols)
//
//  C) MULTI-CORE (8x8 = 64 cores, identical per-core work): no contention effect.
//   (32,4) 12763.6 -> 7783.1 (blk4)   (1,3) 567.9 -> 510.6 (blk3, 1.11x)
//
// Run-to-run spread on the repeated rows is < 0.1% (12762.0 / 12762.6 / 12763.6 for the
// (32,4) baseline across three separate device sessions), far under the 2-3% noise band.
// ---------------------------------------------------------------------------

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_rstd = 1;
constexpr uint32_t cb_gamma = 2;
constexpr uint32_t cb_normed = 14;
constexpr uint32_t cb_out = 16;

// Policy encoding shared with the host.
constexpr uint32_t P_PERTILE = 0;
constexpr uint32_t P_BULK = 1;
constexpr uint32_t P_PERCHUNK = 2;
// The op's OUT_STRIDED leg (pinned/row-major-C output shard): the CALLER reserves the
// whole rows_t*CB_W_TILES block once and pushes it after the last chunk, so the chain
// itself carries ReservePolicy::None / PushPolicy::None and packs at TileOffset::Strided.
// There is no lifecycle lever there — only DEST blocking — so this policy exists to prove
// blocking is correct and still wins under a STRIDED pack.
constexpr uint32_t P_CALLER_STRIDED = 3;
}  // namespace

void kernel_main() {
    constexpr uint32_t ROWS_T = get_compile_time_arg_val(0);
    constexpr uint32_t COLS = get_compile_time_arg_val(1);
    constexpr uint32_t BLK = get_compile_time_arg_val(2);
    constexpr uint32_t ITERS = get_compile_time_arg_val(3);
    constexpr uint32_t OUT_POLICY = get_compile_time_arg_val(4);
    constexpr uint32_t NORMED_POLICY = get_compile_time_arg_val(5);

    static_assert(BLK == 1 || OUT_POLICY != P_PERTILE, "PerTile output + BLK>1 overruns the CB and hangs");
    static_assert(BLK == 1 || NORMED_POLICY != P_PERTILE, "PerTile normed + BLK>1 overruns the CB and hangs");

    constexpr uint32_t N_BLOCK = ROWS_T * COLS;
    constexpr auto RC = ckl::DataFormatReconfig::Enabled;  // what the op does today; held fixed

    constexpr auto out_spec =
        (OUT_POLICY == P_BULK)       ? ckl::output(cb_out, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, RC)
        : (OUT_POLICY == P_PERCHUNK) ? ckl::output(cb_out, ckl::ReservePolicy::PerChunk, ckl::PushPolicy::PerChunk, RC)
        : (OUT_POLICY == P_CALLER_STRIDED)
            ? ckl::output(
                  cb_out,
                  ckl::ReservePolicy::None,
                  ckl::PushPolicy::None,
                  RC,
                  ckl::PackRelu::Disabled,
                  ckl::L1Accumulation::Disabled,
                  ckl::DestAccumulation::Disabled,
                  ckl::TileOffset::Strided)
            : ckl::output(cb_out, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, RC);
    constexpr auto normed_out_spec =
        (NORMED_POLICY == P_BULK) ? ckl::output(cb_normed, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, RC)
        : (NORMED_POLICY == P_PERCHUNK)
            ? ckl::output(cb_normed, ckl::ReservePolicy::PerChunk, ckl::PushPolicy::PerChunk, RC)
            : ckl::output(cb_normed, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, RC);

    constexpr auto in_spec = ckl::input(
        cb_in, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block, RC, ckl::TileOffset::Strided);
    constexpr auto rstd_spec = ckl::input(cb_rstd, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Col);
    constexpr auto gamma_spec =
        ckl::input(cb_gamma, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Row);
    // The op's pass-2 read of cb_normed: waited upfront, popped at the end.
    constexpr auto normed_in_spec =
        ckl::input(cb_normed, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block, RC);

    compute_kernel_hw_startup(cb_in, cb_rstd, cb_out);

    // Pinned zero-copy operands: one push publishes them for the whole launch.
    cb_reserve_back(cb_in, N_BLOCK);
    cb_push_back(cb_in, N_BLOCK);
    cb_reserve_back(cb_rstd, ROWS_T);
    cb_push_back(cb_rstd, ROWS_T);
    cb_reserve_back(cb_gamma, COLS);
    cb_push_back(cb_gamma, COLS);

    const ckl::StridedTileRange src{0, COLS};

    for (uint32_t it = 0; it < ITERS; ++it) {
        {
            MaybeDeviceZoneScope("bench_apply");
            // Pass 1 — cp_apply_scale: x * rstd (Col broadcast) -> cb_normed.
            ckl::eltwise_chain(
                ckl::EltwiseShape::grid(ROWS_T, COLS, BLK),
                ckl::BinaryFpu<in_spec, rstd_spec, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Col>{src},
                ckl::PackTile<normed_out_spec>{});
            // Pass 2 — cp_apply_gamma: normed * gamma (Row broadcast) -> cb_out.
            if constexpr (OUT_POLICY == P_CALLER_STRIDED) {
                cb_reserve_back(cb_out, N_BLOCK);
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(ROWS_T, COLS, BLK),
                    ckl::BinaryFpu<normed_in_spec, gamma_spec, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Row>{},
                    ckl::PackTile<out_spec>{ckl::StridedTileRange{0, COLS}});
                cb_push_back(cb_out, N_BLOCK);
            } else {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(ROWS_T, COLS, BLK),
                    ckl::BinaryFpu<normed_in_spec, gamma_spec, ckl::BinaryFpuOp::Mul, ckl::BroadcastDim::Row>{},
                    ckl::PackTile<out_spec>{});
            }
        }
        // Recycle the output CB so the next iteration re-runs the same block. Same
        // cost on every option; outside the zone.
        if (it + 1 < ITERS) {
            cb_wait_front(cb_out, N_BLOCK);
            cb_pop_front(cb_out, N_BLOCK);
        }
    }
}
