// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// compact_stat_gather micro-benchmark — WRITER (NoC1).
//
// Two jobs: the CONTRIBUTOR half of the combine (this core's stat -> the root's
// landing buffer, plus one progress increment), and — so the bench is
// correctness-checkable — the root drains the broadcast result to DRAM.  That
// drain is identical in every mode, so it is constant scaffolding.
//
// The gather is the only thing that differs between modes, and the difference IS
// the idea:
//
//   MODE_RAW_TILE   (0)  1 x 4096 B  -> landing page (r*s + c)          <- baseline
//   MODE_COLLAPSE_4K(1)  1 x 4096 B  -> landing page (r*s + c)
//   MODE_COLLAPSE_2K(2)  2 x 1024 B  -> faces 0 and 2 of page (r*s + c)
//   MODE_ROW_128B   (3)  2 x   64 B  -> ROW (c%32) of landing page ((c/32)*B + r)
//
// MODE 3's two destination offsets are the two face-rows that make up ROW c of a
// tile: cols 0-15 live in face (c<16 ? 0 : 2) at byte 64*(c%16) inside that face,
// cols 16-31 in face (c<16 ? 1 : 3) at the same in-face offset.  Face bases are
// 0 / 1024 / 2048 / 3072 for a 32x32 fp32 tile, so every offset is a multiple of
// 64 — a legal single NoC write on both ends, no scatter.
//
// Raw-API note: the gather is s different sources -> s different destinations on
// ONE core, the mirror image of mcast_pipe's one-source-to-a-rectangle, and
// kernel_lib has no gather helper.  Same reasoning as the op's own writer.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

constexpr uint32_t cb_sq_partials = 2;
constexpr uint32_t cb_slice_stat = 3;
constexpr uint32_t cb_gathered_partials = 4;
constexpr uint32_t cb_rms_recip = 6;

constexpr uint32_t MODE_RAW_TILE = 0;
constexpr uint32_t MODE_COLLAPSE_4K = 1;
constexpr uint32_t MODE_COLLAPSE_2K = 2;
constexpr uint32_t MODE_ROW_128B = 3;

constexpr uint32_t FACE_ROW_BYTES = 64;  // 16 fp32 lanes
constexpr uint32_t FACE_BYTES = 1024;    // 16x16 fp32
constexpr uint32_t TILE_ROWS_PER_TILE = 32;

void kernel_main() {
    constexpr uint32_t SLICE_HIDDEN_TILES = get_compile_time_arg_val(0);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(1);
    constexpr uint32_t NUM_HIDDEN_SLICES = get_compile_time_arg_val(2);
    constexpr uint32_t STAT_TILE_BYTES = get_compile_time_arg_val(3);
    constexpr uint32_t GATHER_SEM_ID = get_compile_time_arg_val(4);
    constexpr uint32_t MODE = get_compile_time_arg_val(5);
    constexpr uint32_t LANDING_SEM_ID = get_compile_time_arg_val(6);
    // Ablation knob (/perf-measure discipline): 0 keeps the whole synchronization
    // scaffolding (CB wait/pop, barriers, trip counts) but stubs the DRAM DRAIN
    // payload, so the measured number is the combine and not the bench's
    // correctness plumbing.  Correctness runs use DRAIN=1.
    constexpr uint32_t DRAIN = get_compile_time_arg_val(7);
    constexpr auto out_args = TensorAccessorArgs<8>();

    constexpr uint32_t src_cb = (MODE == MODE_RAW_TILE) ? cb_sq_partials : cb_slice_stat;
    constexpr bool NEEDS_LANDING_EDGE = (MODE == MODE_COLLAPSE_2K) || (MODE == MODE_ROW_128B);

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t slice_index = get_arg_val<uint32_t>(2);
    const uint32_t root_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t root_noc_y = get_arg_val<uint32_t>(4);
    const uint32_t is_root = get_arg_val<uint32_t>(5);
    const uint32_t stat_page_base = get_arg_val<uint32_t>(6);

    Noc noc;
    const auto output_accessor = TensorAccessor(out_args, output_addr);
    Semaphore<> gather_progress(GATHER_SEM_ID);

    const uint32_t gather_base = get_write_ptr(cb_gathered_partials);

    // MODE 3 landing geometry, resolved once (all runtime-invariant per core).
    const uint32_t c32 = slice_index % TILE_ROWS_PER_TILE;
    const uint32_t landing_tile_row = slice_index / TILE_ROWS_PER_TILE;
    const uint32_t in_face_off = FACE_ROW_BYTES * (c32 % 16);
    const uint32_t face_a = (c32 < 16) ? 0u : 2u;
    const uint32_t row_off_a = face_a * FACE_BYTES + in_face_off;
    const uint32_t row_off_b = (face_a + 1) * FACE_BYTES + in_face_off;

    // The landing buffer's un-owned lanes are zeroed by the ROOT's reader; nothing
    // else orders that zero against THIS core's first gather write.  Wait once.
    if constexpr (NEEDS_LANDING_EDGE) {
        Semaphore<> landing_ready(LANDING_SEM_ID);
        landing_ready.wait_min(1);
    }

    for (uint32_t block = 0; block < num_blocks; ++block) {
        cb_wait_front(src_cb, BLOCK_ROWS);
        const uint32_t src = get_read_ptr(src_cb);

        for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
            const uint32_t s0 = src + r * STAT_TILE_BYTES;
            if constexpr (MODE == MODE_ROW_128B) {
                const uint32_t page = landing_tile_row * BLOCK_ROWS + r;
                const uint32_t dst = gather_base + page * STAT_TILE_BYTES;
                noc_async_write(s0, get_noc_addr(root_noc_x, root_noc_y, dst + row_off_a), FACE_ROW_BYTES);
                noc_async_write(s0 + FACE_BYTES, get_noc_addr(root_noc_x, root_noc_y, dst + row_off_b), FACE_ROW_BYTES);
            } else if constexpr (MODE == MODE_COLLAPSE_2K) {
                const uint32_t page = r * NUM_HIDDEN_SLICES + slice_index;
                const uint32_t dst = gather_base + page * STAT_TILE_BYTES;
                noc_async_write(s0, get_noc_addr(root_noc_x, root_noc_y, dst), FACE_BYTES);
                noc_async_write(
                    s0 + 2 * FACE_BYTES, get_noc_addr(root_noc_x, root_noc_y, dst + 2 * FACE_BYTES), FACE_BYTES);
            } else {
                const uint32_t page = r * NUM_HIDDEN_SLICES + slice_index;
                noc_async_write(
                    s0, get_noc_addr(root_noc_x, root_noc_y, gather_base + page * STAT_TILE_BYTES), STAT_TILE_BYTES);
            }
        }
        noc_async_write_barrier();
        gather_progress.up(noc, root_noc_x, root_noc_y, 1);
        cb_pop_front(src_cb, BLOCK_ROWS);

        // ---- drain: the root writes the finalized 1/rms tiles to DRAM so the
        //      bench is checkable.  Every core pops (the mcast landed on all of
        //      them), only the root moves bytes.  Constant across modes.
        cb_wait_front(cb_rms_recip, BLOCK_ROWS);
        if (DRAIN && is_root) {
            const uint32_t l1 = get_read_ptr(cb_rms_recip);
            for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                noc_async_write(
                    l1 + r * STAT_TILE_BYTES,
                    output_accessor.get_noc_addr(stat_page_base + block * BLOCK_ROWS + r),
                    STAT_TILE_BYTES);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb_rms_recip, BLOCK_ROWS);
    }
}
