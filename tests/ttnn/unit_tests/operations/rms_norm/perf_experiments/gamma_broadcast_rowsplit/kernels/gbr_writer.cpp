// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm writer (BRISC / NoC1) — op_design.md §4.1 and §4.2.
//
// Mirrors the reader's row-block/chunk nest exactly: per core, per row-block of
// HT_BLOCK tile-rows, drain NW chunks of WT_CHUNK W-tiles of this core's W
// slice. The TILE path writes whole tile pages with one barrier per chunk (the
// batched twin of the reader's batched chunk read); the ROW_MAJOR path goes
// through dataflow_kernel_lib::write_sticks_after_untilize, which skips the L1
// W-padding and writes only the valid sticks of a short last tile-row — that is
// what makes non-aligned H/W native with no host-side slice.
//
// SHARDED output (SHARDED_OUT): there is NO write at all. cb_output_tiles is
// placed directly on the core's own L1 output shard, so compute's pack lands in
// the output tensor in place — zero-copy, no NoC traffic.
//
// CROSS-CORE W-SPLIT (W_SPLIT): this kernel owns the *gather* half of the
// combine. Per row-block it ships this core's raw sum(x^2) tiles into slot
// `s1_slot` of its LEADER's cb_group_partials and bumps that leader's stage-1
// counter; the leader's reader waits for all CW1 of them (see
// rms_norm_reader.cpp). With CW2 == 1 the leader is the group root and that is
// the whole combine. With CW2 > 1 the combine is TWO-STAGE, and a leader ships a
// SECOND tile — the row sum compute folded out of its stage-1 gather — into slot
// `s2_slot` of the root's cb_group_partials2, bumping the stage-2 counter. Both
// pushes ride the SAME cb_partial_out (one producer, one consumer, two
// sequential pushes), so the staged topology adds no CB.
// Many-to-one has no multicast form and no helper covers it, so each leg is a
// plain noc_async_write + barrier + noc_semaphore_inc
// (references/cross_core_reduction_design.md §1/§7).
//
// PER-STAGE INSTRUMENTATION (permanent, Perf 1). Stage boundaries carry a
// MaybeDeviceZoneScope: wtr_gather_hop / wtr_write. Free when the profiler is
// off (perf_instrumentation.hpp's durability contract) — never remove one, and
// extend the set to any new predicate-guarded path.
//
// Helper substitution note: the TILE path uses TensorAccessor +
// noc_async_write_page directly because write_sticks_after_untilize writes
// *sticks* produced by the untilize helper, and the TILE path never untilizes —
// its input contract cannot be met. op_design.md §7 mandates exactly this.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_group_partials = 6;
constexpr uint32_t cb_partial_out = 8;
constexpr uint32_t cb_group_partials2 = 9;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_output_rm = 17;
}  // namespace

namespace dkl = dataflow_kernel_lib;

void kernel_main() {
    // ---- regime flags (§5.2) ----
    constexpr bool IS_RM = get_compile_time_arg_val(0) != 0;
    // args 1..5 are the reader/compute regime flags; kept in one shared layout.
    // ---- block knobs (§1.2) ----
    constexpr uint32_t WT = get_compile_time_arg_val(6);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(7);
    constexpr uint32_t WT_LAST = get_compile_time_arg_val(8);
    constexpr uint32_t NW = get_compile_time_arg_val(9);
    constexpr uint32_t HT_BLOCK = get_compile_time_arg_val(10);
    // arg 11 (X_READ_CHUNKS) is a read-side knob; the writer's granularity is
    // set by the compute output CB, which always publishes one chunk at a time.
    // ---- geometry ----
    constexpr uint32_t CHUNK_ROW_BYTES = get_compile_time_arg_val(13);
    constexpr uint32_t LAST_ROW_BYTES = get_compile_time_arg_val(14);
    constexpr uint32_t TOTAL_STICKS = get_compile_time_arg_val(17);
    // ---- cross-core W-split (§4.2) ----
    constexpr bool W_SPLIT = get_compile_time_arg_val(18) != 0;
    constexpr uint32_t CW = get_compile_time_arg_val(19);
    constexpr uint32_t WT_STRIDE = get_compile_time_arg_val(20);
    constexpr bool SHARDED_OUT = get_compile_time_arg_val(22) != 0;
    constexpr uint32_t SEM_GATHER = get_compile_time_arg_val(23);
    constexpr uint32_t CW1 = get_compile_time_arg_val(24);
    constexpr uint32_t CW2 = get_compile_time_arg_val(25);
    constexpr uint32_t SEM_GATHER2 = get_compile_time_arg_val(26);
    constexpr bool TWO_STAGE = CW2 > 1;
    static_assert(CW1 * CW2 == CW, "combine stages must tile CW");

    // 27..36 are the two multicast-family CT blocks (reader-side; the writer
    // shares the layout so a knob cannot drift between them).
    // FORK (gamma_broadcast_rowsplit): the shared dataflow CT block grew by the
    // experiment's gamma-broadcast tail (2 flag words + 2 x 5-word mcast blocks),
    // so the accessor base moves 37 -> 49. Nothing else in this file changes.
    constexpr auto out_args = TensorAccessorArgs<49>();

    static_assert(WT_LAST == WT_CHUNK, "writer assumes uniform chunk widths");

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(1);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(2);
    const uint32_t wt_start = get_arg_val<uint32_t>(3);
    const uint32_t s1_slot = get_arg_val<uint32_t>(4);
    const uint32_t leader_x = get_arg_val<uint32_t>(5);
    const uint32_t leader_y = get_arg_val<uint32_t>(6);
    const uint32_t is_last_w_core = get_arg_val<uint32_t>(7);
    const uint32_t root_x = get_arg_val<uint32_t>(8);
    const uint32_t root_y = get_arg_val<uint32_t>(9);
    const uint32_t s2_slot = get_arg_val<uint32_t>(10);
    const uint32_t is_leader = get_arg_val<uint32_t>(11);

    // Filler core (inside the multicast rectangle, owns no data).
    if (num_tile_rows == 0) {
        return;
    }

    const auto out_acc = TensorAccessor(out_args, dst_addr);
    const uint32_t out_tile_bytes = get_tile_size(cb_output_tiles);
    const uint32_t chunk0 = wt_start / WT_CHUNK;

    // Gather targets: the leader's cb_group_partials (stage 1) and the root's
    // cb_group_partials2 (stage 2). Every core declares both CBs at the same
    // size, so each sits at the same L1 offset everywhere and this core's own
    // write pointer IS the remote base address.
    [[maybe_unused]] const uint32_t partial_tile_bytes = get_tile_size(cb_partial_out);
    [[maybe_unused]] const uint32_t gather_base = get_write_ptr(cb_group_partials);
    [[maybe_unused]] const uint32_t gather2_base = get_write_ptr(cb_group_partials2);
    [[maybe_unused]] const uint32_t gather_sem_addr = get_semaphore<ProgrammableCoreType::TENSIX>(SEM_GATHER);
    [[maybe_unused]] const uint32_t gather2_sem_addr = get_semaphore<ProgrammableCoreType::TENSIX>(SEM_GATHER2);

    // One gather hop: ship `ht` settled cb_partial_out tiles into `slot` of the
    // destination core's gather CB (laid out h-major, tile h*fan_in + slot, so
    // the folding core reads them as a contiguous (ht x fan_in) block), then let
    // it know they LANDED.
    [[maybe_unused]] auto gather_hop = [&](uint32_t ht,
                                           uint32_t dst_x,
                                           uint32_t dst_y,
                                           uint32_t base,
                                           uint32_t fan_in,
                                           uint32_t slot,
                                           uint32_t sem_addr) {
        cb_wait_front(cb_partial_out, ht);
        uint32_t src = get_read_ptr(cb_partial_out);
        for (uint32_t h = 0; h < ht; ++h) {
            const uint64_t dst = get_noc_addr(dst_x, dst_y, base + (h * fan_in + slot) * partial_tile_bytes);
            noc_async_write(src, dst, partial_tile_bytes);
            src += partial_tile_bytes;
        }
        // The data must have LANDED before the destination sees the counter move.
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(dst_x, dst_y, sem_addr), 1);
        cb_pop_front(cb_partial_out, ht);
    };

    const uint32_t num_row_blocks = (num_tile_rows + HT_BLOCK - 1) / HT_BLOCK;
    for (uint32_t hb = 0; hb < num_row_blocks; ++hb) {
        const uint32_t row0 = start_tile_row + hb * HT_BLOCK;
        uint32_t ht = num_tile_rows - hb * HT_BLOCK;
        if (ht > HT_BLOCK) {
            ht = HT_BLOCK;
        }

        uint32_t valid_rows = ht * 32u;
        if constexpr (IS_RM) {
            const uint32_t remaining = TOTAL_STICKS - row0 * 32u;
            if (remaining < valid_rows) {
                valid_rows = remaining;
            }
        }

        // ---- gather legs: this core's raw partial sum, then (leaders only)
        // the row sum compute folded out of the stage-1 gather.
        if constexpr (W_SPLIT) {
            MaybeDeviceZoneScope("wtr_gather_hop");
            gather_hop(ht, leader_x, leader_y, gather_base, CW1, s1_slot, gather_sem_addr);
            if constexpr (TWO_STAGE) {
                if (is_leader) {
                    gather_hop(ht, root_x, root_y, gather2_base, CW2, s2_slot, gather2_sem_addr);
                }
            }
        }

        for (uint32_t wc = 0; wc < NW; ++wc) {
            MaybeDeviceZoneScope("wtr_write");
            if constexpr (SHARDED_OUT) {
                // Zero-copy: compute packed straight into the output shard.
                continue;
            } else if constexpr (IS_RM) {
                const uint32_t rb = (is_last_w_core && wc + 1 == NW) ? LAST_ROW_BYTES : CHUNK_ROW_BYTES;
                dkl::write_sticks_after_untilize<cb_output_rm>(
                    out_acc, valid_rows, rb, row0 * 32u, (chunk0 + wc) * CHUNK_ROW_BYTES);
            } else {
                const uint32_t n = ht * WT_CHUNK;
                cb_wait_front(cb_output_tiles, n);
                uint32_t addr = get_read_ptr(cb_output_tiles);
                for (uint32_t h = 0; h < ht; ++h) {
                    const uint32_t base_tile = (row0 + h) * WT_STRIDE + wt_start + wc * WT_CHUNK;
                    for (uint32_t t = 0; t < WT_CHUNK; ++t) {
                        noc_async_write_page(base_tile + t, out_acc, addr);
                        addr += out_tile_bytes;
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(cb_output_tiles, n);
            }
        }
    }

    // noc_semaphore_inc issues a NON-POSTED atomic; leaving one outstanding at
    // kernel exit leaves the core's NoC transaction counters unbalanced, which
    // the dispatcher never sees drain (measured as a device hang on the sharded
    // cells, whose writer has no trailing write barrier to absorb it).
    if constexpr (W_SPLIT) {
        noc_async_atomic_barrier();
    }
}
