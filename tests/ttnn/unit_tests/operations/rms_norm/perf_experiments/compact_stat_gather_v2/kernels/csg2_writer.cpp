// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// compact_stat_gather_v2 micro-benchmark — WRITER (NoC1).
//
// The CONTRIBUTOR half of the combine (this core's per-row stat -> the OWNER of
// that row), plus — so the bench is checkable — the root's DRAM drain of the
// broadcast result.  The drain is identical in every mode and is stubbed for the
// perf runs (/perf-measure: keep the scaffolding, drop the payload).
//
// The gather is the only thing that differs between modes, and the difference IS
// the idea.  Per block row r (owner o = r / OWN_ROWS, j = r % OWN_ROWS,
// c = this core's slice_index):
//
//   MODE_RAW_4K       (0)  1 x 4096 B -> landing page (j*s + c)        <- baseline
//   MODE_ROW_128B     (1)  2 x   64 B -> ROW (c%32) of landing page ((c/32)*OWN_ROWS + j)
//   MODE_COLLAPSE_2K  (2)  2 x 1024 B -> faces 0 and 2 of landing page (j*s + c)
//   MODE_ROW_64B_PROBE(3)  1 x   64 B -> as MODE 1 but only the FIRST face-row.
//                          WRONG BY CONSTRUCTION; it exists only to price the
//                          second NoC transaction of MODE 1.
//
// MODE 1's two destination offsets are the two face-rows that make up ROW c of a
// tile: cols 0-15 live in face (c<16 ? 0 : 2) at byte 64*(c%16) inside that face,
// cols 16-31 in face (c<16 ? 1 : 3) at the same in-face offset.  Face bases are
// 0 / 1024 / 2048 / 3072 for a 32x32 fp32 tile, so every offset is a multiple of
// 64 — a legal single NoC write on both ends, no scatter.
//
// A single write cannot cover both face-rows: the SOURCE halves sit 1024 B apart
// with 960 B of (zero) padding between them, and shipping that padding would
// overwrite the landing rows belonging to the OTHER contributors.  Two
// transactions per (owner, owned row) is therefore the floor for this spelling.
//
// Raw-API note: the gather is s different sources -> s different destination
// pages on one core, the mirror image of mcast_pipe's one-source-to-a-rectangle,
// and kernel_lib has no gather helper.  Same reasoning as the op's own writer.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

constexpr uint32_t cb_sq_partials = 2;
constexpr uint32_t cb_gathered_partials = 4;
constexpr uint32_t cb_rms_recip = 6;
constexpr uint32_t cb_stat_compact = 10;
constexpr uint32_t cb_zeros = 11;  // one fp32 tile of zeros: the boot pad-zero source

constexpr uint32_t MODE_RAW_4K = 0;
constexpr uint32_t MODE_ROW_128B = 1;
constexpr uint32_t MODE_COLLAPSE_2K = 2;
constexpr uint32_t MODE_ROW_64B_PROBE = 3;

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
    constexpr uint32_t DRAIN = get_compile_time_arg_val(6);
    constexpr uint32_t NUM_OWNERS = get_compile_time_arg_val(7);
    constexpr uint32_t OWN_ROWS = get_compile_time_arg_val(8);
    constexpr auto out_args = TensorAccessorArgs<9>();

    constexpr uint32_t src_cb = (MODE == MODE_RAW_4K) ? cb_sq_partials : cb_stat_compact;
    constexpr bool NEEDS_LANDING_EDGE = (MODE != MODE_RAW_4K);

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t slice_index = get_arg_val<uint32_t>(2);
    const uint32_t root_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t root_noc_y = get_arg_val<uint32_t>(4);
    const uint32_t is_root = get_arg_val<uint32_t>(5);
    const uint32_t stat_page_base = get_arg_val<uint32_t>(6);
    constexpr uint32_t owner_xy_base = 7;

    Noc noc;
    const auto output_accessor = TensorAccessor(out_args, output_addr);
    Semaphore<> gather_progress(GATHER_SEM_ID);

    const uint32_t gather_base = get_write_ptr(cb_gathered_partials);

    // MODE 1/3 landing geometry, resolved once (runtime-invariant per core).
    const uint32_t c32 = slice_index % TILE_ROWS_PER_TILE;
    const uint32_t landing_tile_row = slice_index / TILE_ROWS_PER_TILE;
    const uint32_t in_face_off = FACE_ROW_BYTES * (c32 % 16);
    const uint32_t face_a = (c32 < 16) ? 0u : 2u;
    const uint32_t row_off_a = face_a * FACE_BYTES + in_face_off;
    const uint32_t row_off_b = (face_a + 1) * FACE_BYTES + in_face_off;

    // ---- boot: neutralize the landing lanes NOBODY ever writes ----
    //
    // MODE 1/3 fill only rows 0..s-1 of each landing tile; MODE 2 fills only faces
    // 0 and 2 of each landing page.  Every lane the owner's reduce reads but no
    // contributor writes MUST be zero or it enters the sum as a phantom
    // contributor.
    //
    // The zeroing is assigned so that EVERY BYTE OF THE LANDING BUFFER HAS EXACTLY
    // ONE WRITER, which is what removes the cross-core race outright:
    //   MODE 1/3 — the pad rows [s, 32) of owner `o`'s landing tiles are zeroed by
    //              contributor `o` (num_owners <= s, so the map is total and
    //              injective).  No other core ever touches those bytes.
    //   MODE 2   — page (j*s + c) belongs to contributor c alone, so each
    //              contributor zeroes faces 1 and 3 of its OWN pages.
    // Ordering against the owner's reduce is free: these writes precede this
    // core's per-block `noc_async_write_barrier()` + `gather_progress` increment,
    // and the owner's reduce waits on all s of those increments.
    //
    // This replaces round 1's "owner zeroes, everyone waits on a boot semaphore"
    // edge, which the reduce-scatter topology turns into a multi-sender handshake
    // that HANGS (measured — see csg2_reader.cpp).
    if constexpr (NEEDS_LANDING_EDGE) {
        Noc zero_noc;
        CircularBuffer cb_zeros_obj(cb_zeros);
        zero_noc.async_write_zeros(cb_zeros_obj, STAT_TILE_BYTES);
        zero_noc.write_zeros_l1_barrier();
        const uint32_t zsrc = get_write_ptr(cb_zeros);

        if constexpr (MODE == MODE_ROW_128B || MODE == MODE_ROW_64B_PROBE) {
            if (slice_index < NUM_OWNERS) {
                const uint32_t ox = get_arg_val<uint32_t>(owner_xy_base + 2 * slice_index);
                const uint32_t oy = get_arg_val<uint32_t>(owner_xy_base + 2 * slice_index + 1);
                constexpr uint32_t LANDING_TILE_ROWS = (NUM_HIDDEN_SLICES + 31) / 32;
                for (uint32_t ht = 0; ht < LANDING_TILE_ROWS; ++ht) {
                    const uint32_t left = NUM_HIDDEN_SLICES - 32 * ht;
                    const uint32_t valid = left < TILE_ROWS_PER_TILE ? left : TILE_ROWS_PER_TILE;
                    for (uint32_t j = 0; j < OWN_ROWS; ++j) {
                        const uint32_t dst = gather_base + (ht * OWN_ROWS + j) * STAT_TILE_BYTES;
                        if (valid < 16) {
                            const uint32_t len = FACE_BYTES - FACE_ROW_BYTES * valid;
                            const uint32_t off = FACE_ROW_BYTES * valid;
                            noc_async_write(zsrc, get_noc_addr(ox, oy, dst + off), len);
                            noc_async_write(zsrc, get_noc_addr(ox, oy, dst + FACE_BYTES + off), len);
                            noc_async_write(zsrc, get_noc_addr(ox, oy, dst + 2 * FACE_BYTES), 2 * FACE_BYTES);
                        } else if (valid < 32) {
                            const uint32_t len = FACE_BYTES - FACE_ROW_BYTES * (valid - 16);
                            const uint32_t off = FACE_ROW_BYTES * (valid - 16);
                            noc_async_write(zsrc, get_noc_addr(ox, oy, dst + 2 * FACE_BYTES + off), len);
                            noc_async_write(zsrc, get_noc_addr(ox, oy, dst + 3 * FACE_BYTES + off), len);
                        }
                    }
                }
            }
        } else if constexpr (MODE == MODE_COLLAPSE_2K) {
            for (uint32_t o = 0; o < NUM_OWNERS; ++o) {
                const uint32_t ox = get_arg_val<uint32_t>(owner_xy_base + 2 * o);
                const uint32_t oy = get_arg_val<uint32_t>(owner_xy_base + 2 * o + 1);
                for (uint32_t j = 0; j < OWN_ROWS; ++j) {
                    const uint32_t dst =
                        gather_base + (j * NUM_HIDDEN_SLICES + slice_index) * STAT_TILE_BYTES;
                    noc_async_write(zsrc, get_noc_addr(ox, oy, dst + FACE_BYTES), FACE_BYTES);
                    noc_async_write(zsrc, get_noc_addr(ox, oy, dst + 3 * FACE_BYTES), FACE_BYTES);
                }
            }
        }
        noc_async_write_barrier();
    }

    for (uint32_t block = 0; block < num_blocks; ++block) {
        cb_wait_front(src_cb, BLOCK_ROWS);
        const uint32_t src = get_read_ptr(src_cb);

        for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
            const uint32_t s0 = src + r * STAT_TILE_BYTES;
            const uint32_t owner = r / OWN_ROWS;
            const uint32_t j = r % OWN_ROWS;
            const uint32_t ox = get_arg_val<uint32_t>(owner_xy_base + 2 * owner);
            const uint32_t oy = get_arg_val<uint32_t>(owner_xy_base + 2 * owner + 1);
            if constexpr (MODE == MODE_ROW_128B || MODE == MODE_ROW_64B_PROBE) {
                const uint32_t page = landing_tile_row * OWN_ROWS + j;
                const uint32_t dst = gather_base + page * STAT_TILE_BYTES;
                noc_async_write(s0, get_noc_addr(ox, oy, dst + row_off_a), FACE_ROW_BYTES);
                if constexpr (MODE == MODE_ROW_128B) {
                    noc_async_write(s0 + FACE_BYTES, get_noc_addr(ox, oy, dst + row_off_b), FACE_ROW_BYTES);
                }
            } else if constexpr (MODE == MODE_COLLAPSE_2K) {
                const uint32_t page = j * NUM_HIDDEN_SLICES + slice_index;
                const uint32_t dst = gather_base + page * STAT_TILE_BYTES;
                noc_async_write(s0, get_noc_addr(ox, oy, dst), FACE_BYTES);
                noc_async_write(s0 + 2 * FACE_BYTES, get_noc_addr(ox, oy, dst + 2 * FACE_BYTES), FACE_BYTES);
            } else {
                const uint32_t page = j * NUM_HIDDEN_SLICES + slice_index;
                noc_async_write(s0, get_noc_addr(ox, oy, gather_base + page * STAT_TILE_BYTES), STAT_TILE_BYTES);
            }
        }
        noc_async_write_barrier();
        for (uint32_t o = 0; o < NUM_OWNERS; ++o) {
            gather_progress.up(
                noc, get_arg_val<uint32_t>(owner_xy_base + 2 * o), get_arg_val<uint32_t>(owner_xy_base + 2 * o + 1), 1);
        }
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
