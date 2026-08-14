// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BENCH — rms_norm's cross-core combine, compute half.
//
// Two compute phases of the real op and nothing else:
//   * `square_accumulate_block` -> ckl::sum_of_squares over (B, S)      [EVERY core]
//   * `combine_block`           -> sum the NSLICE gathered partials for each of this
//                                  core's OWN_ROWS rows, collapse within the tile,
//                                  finalize (x * 1/W, + eps, rsqrt)     [OWNERS]
//
// The combine has two IMPLEMENTATIONS, selected by COMBINE_IMPL, and the whole
// point of the bench is that the second one can read a landing buffer the first
// one cannot address:
//
//   COMBINE_IMPL == 0  (helper, shipped)
//       ckl::reduce<SUM, REDUCE_ROW, ...>(ReduceInputBlockShape::of(OWN_ROWS, NSLICE),
//                                         ReduceInputMemoryLayout::contiguous())
//       Requires the row-major landing map: output row o's NSLICE partials must be
//       CONTIGUOUS at o * NSLICE.
//
//   COMBINE_IMPL == 1  (raw, this bench)
//       Output o's partials are at (o * ROW_PITCH + k * REDUCE_STRIDE).  With
//       (ROW_PITCH, REDUCE_STRIDE) = (NSLICE, 1) this is bit-for-bit the helper's
//       row-major walk; with (1, OWN_ROWS) it is the COALESCED landing map.
//
// ---------------------------------------------------------------------------
// RAW-LLK JUSTIFICATION (helper bypassed: compute_kernel_lib::reduce)
//
// The coalesced landing map puts the reduce axis (slice) on the OUTER index and
// the output axis (row) on the INNER one.  compute_kernel_lib::reduce cannot
// express that block, for two independent reasons:
//
//  (1) ReduceInputMemoryLayout carries only `row_stride` — a row PITCH, asserted
//      >= cols.  It expresses "rows are padded", i.e. addr(r, k) = r*pitch + k.
//      There is no column stride, so addr(r, k) = r + k*OWN_ROWS is inexpressible
//      for REDUCE_ROW.
//  (2) The transposed spelling that WOULD index it correctly is
//      reduce<SUM, REDUCE_COL, ..., AccumulateViaAdd, ..., ReduceWithinTile::Skip>
//      with of(NSLICE, OWN_ROWS): the AccumulateViaAdd COL walk is exactly
//      `start = o, stride = row_pitch` (reduce_helpers_compute.inl:380,245).  But
//      REDUCE_COL's within-tile Collapse folds the wrong axis for these
//      column-carrying partials, so it needs ReduceWithinTile::Skip plus a caller
//      REDUCE_ROW collapse in post_reduce_op — and Skip is UNREACHABLE: the
//      `static_assert(within_tile == Collapse)` that guards the ReduceTile
//      datapath (reduce_helpers_compute.inl:886) sits AFTER the
//      `if constexpr (AccumulateViaAdd) { ...; return; }` block, so it is not in a
//      discarded statement and fires for the AccumulateViaAdd instantiation too.
//
// So the body below is a literal transcription of
// detail::reduce_accumulate_via_add<SUM, REDUCE_COL, ..., BulkWaitBulkPop,
// INPUT_AND_OUTPUT, NoAccumulation, Fin, ReduceWithinTile::Skip> with the
// REDUCE_ROW collapse moved into the finalize — the same reconfig, the same
// sfpu_reduce_init, the same odd-count copy_tile seed, the same acc_to_dest
// pairwise add order, the same per-output pack.  Only the two index constants
// differ, which is why COMBINE_IMPL == 1 at (NSLICE, 1) is a controlled stand-in
// for the helper and isolates the NoC change.
// ---------------------------------------------------------------------------

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_sq_partials = 2;
constexpr uint32_t cb_gathered = 4;
constexpr uint32_t cb_stat_out = 5;
constexpr uint32_t cb_rms_recip = 6;
constexpr uint32_t cb_scaler = 7;

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);         // hidden tiles per core
    constexpr uint32_t B = get_compile_time_arg_val(1);         // rows per block
    constexpr uint32_t NSLICE = get_compile_time_arg_val(2);    // s — cores per row-group
    constexpr uint32_t OWN_ROWS = get_compile_time_arg_val(3);  // rows THIS core reduces
    constexpr uint32_t IN_WAIT_TILES = get_compile_time_arg_val(4);
    constexpr uint32_t COMBINE_IMPL = get_compile_time_arg_val(5);
    constexpr uint32_t ROW_PITCH = get_compile_time_arg_val(6);       // output o -> first tile
    constexpr uint32_t REDUCE_STRIDE = get_compile_time_arg_val(7);   // step between an output's partials
    constexpr uint32_t BLOCK_TILES = B * S;
    constexpr uint32_t GATHERED_TILES = NSLICE * OWN_ROWS;

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t is_owner = get_arg_val<uint32_t>(1);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(2);
    const uint32_t eps_bits = get_arg_val<uint32_t>(3);

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_stat_out);

    // Same fan-in dispatch as the real op (depends on `s` only, so it is identical
    // for every variant at a given geometry).
    constexpr uint32_t COMBINE_ACCUMULATE_MIN_TILES = 4;
    constexpr auto COMBINE_ALGORITHM =
        NSLICE >= COMBINE_ACCUMULATE_MIN_TILES ? ckl::ReduceAlgorithm::AccumulateViaAdd : ckl::ReduceAlgorithm::Auto;

    constexpr auto x_held = ckl::input(cb_in, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block);
    constexpr auto block_shape = ckl::IterationShape::grid(B, S);

    auto finalize = [inv_w_bits, eps_bits](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, inv_w_bits);
        add_unary_tile(dst_idx, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    };

    // ---- the raw strided combine (see the RAW-LLK JUSTIFICATION above) ----
    constexpr DataFormat dst_fmt = DST_ACCUM_MODE ? DataFormat::Float32 : DataFormat::Float16_b;
    auto combine_raw = [&finalize]() {
        reconfig_data_format(cb_gathered, cb_gathered);  // both add operands = the gathered CB
        pack_reconfig_data_format(cb_stat_out);
        sfpu_reduce_init<PoolType::SUM, dst_fmt>();
        for (uint32_t o = 0; o < OWN_ROWS; ++o) {
            const uint32_t start = o * ROW_PITCH;
            tile_regs_acquire();
            uint32_t k = 0;
            if constexpr (NSLICE & 1u) {
                // Odd count: seed DST with a unary copy, exactly as the helper does.
                copy_tile_init(cb_gathered);
                copy_tile(cb_gathered, start, 0);
                k = 1;
            }
            add_init(cb_gathered, cb_gathered, true /* acc_to_dest */);
            for (; k < NSLICE; k += 2) {
                add_tiles(cb_gathered, cb_gathered, start + k * REDUCE_STRIDE, start + (k + 1) * REDUCE_STRIDE, 0);
            }
            // The within-tile collapse the helper's ReduceWithinTile::Collapse would
            // do for REDUCE_ROW.  Hoisted here because the COL walk that indexes the
            // coalesced block would otherwise collapse the wrong axis.
            sfpu_reduce<PoolType::SUM, dst_fmt, ReduceDim::REDUCE_ROW>(0, 1, 1);
            finalize(0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_stat_out, 1);
            pack_tile(0, cb_stat_out);
            cb_push_back(cb_stat_out, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_gathered, GATHERED_TILES);
    };

    for (uint32_t block = 0; block < num_blocks; ++block) {
        {
            MaybeDeviceZoneScope("cp_wait_in");
            cb_wait_front(cb_in, IN_WAIT_TILES);
        }
        {
            MaybeDeviceZoneScope("cp_sumsq");
            ckl::sum_of_squares<x_held, ckl::row_output(cb_sq_partials)>(block_shape);
        }

        if (is_owner) {
            {
                // Hoisted out of the reduce's internal BulkWait so `cp_combine`
                // measures the reduce PAYLOAD and this zone measures the gather incast.
                MaybeDeviceZoneScope("cp_combine_wait");
                cb_wait_front(cb_gathered, GATHERED_TILES);
            }
            MaybeDeviceZoneScope("cp_combine");
            if constexpr (COMBINE_IMPL == 0) {
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_gathered,
                    cb_scaler,
                    cb_stat_out,
                    ckl::ReduceInputPolicy::BulkWaitBulkPop,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ReduceFp32Mode::Fast,
                    COMBINE_ALGORITHM,
                    ckl::NoAccumulation,
                    decltype(finalize)>(
                    ckl::ReduceInputBlockShape::of(OWN_ROWS, NSLICE),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::NoAccumulation{},
                    finalize);
            } else {
                combine_raw();
            }
        }

        // The real op's `cp_rms_wait` — the broadcast gate.
        {
            MaybeDeviceZoneScope("cp_rms_wait");
            cb_wait_front(cb_rms_recip, B);
        }
        cb_pop_front(cb_rms_recip, B);
        cb_pop_front(cb_in, BLOCK_TILES);
    }
}
