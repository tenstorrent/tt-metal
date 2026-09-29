// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Per core: the maximum of each valid row of a range of bf16 TILE pages of the local logits and the lowest id that
// holds it (ttnn.argmax's tie rule).  ROWS = 1 (the decode step): row 0 of a tile is the first row of faces 0 and 1 (32
// bytes each, at tile offsets 0 and 512); both are read with the 64-byte DRAM grain.  ROWS > 1 (the lanes, row r =
// lane r): rows 0..15 are the first ROWS x 32 bytes of faces 0 and 1, rows 16.. the same of faces 2 and 3; each tile
// is staged at its own 2 KB slot and every row is scanned in id order.  The pair (value as fp32 bits, id) of row r goes
// to lanes 4c..4c+1 of row r of the fp32 pairs rows (16 bytes per core per row).  bf16 values compare through a
// sign-magnitude key (zeros canonicalized to +0.0): the first strict maximum in id order is the lowest id.
// With candidates > 0 (the decode step of a sampling server) each core also keeps its top `candidates` pairs in key
// order, lowest id first among equal keys, and writes them as one fp32 ROW_MAJOR page [values | ids] of the lists
// tensor (page = core); the argmax pair is the list's first entry, so the greedy outputs are unchanged.
// LANE_SPLIT (rows > 1): one lane row per core, (row, tile range) work items over the grid: the core reads only row r
// of its tiles (the two 64-byte grains that hold it, as the one-row path reads row 0) and writes its pair to pairs row
// r, lane 4 g of its tile group g; the merge then compares the groups of each row in group order (increasing id
// ranges), so every lane meets the one-row path's rule: bitwise the per-core all-rows scan below it. Named compile-time
// args: cb_stage, lanes_per_tile, rows, candidates, lane_split.  Compile-time args: TensorAccessorArgs(logits),
// (pairs), (lists when candidates > 0).  Runtime args: 0 logits addr, 1 pairs addr, 2 first tile, 3 tile count,
// 4 core index (the tile group), 5 lists addr (candidates > 0), 6 row (lane_split).

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

constexpr uint32_t CB_STAGE = get_named_compile_time_arg_val("cb_stage");
constexpr uint32_t LANES = get_named_compile_time_arg_val("lanes_per_tile");
constexpr uint32_t ROWS = get_named_compile_time_arg_val("rows");
constexpr uint32_t CANDIDATES = get_named_compile_time_arg_val("candidates");
constexpr uint32_t LANE_SPLIT = get_named_compile_time_arg_val("lane_split");
static_assert(
    LANE_SPLIT == 0 || (ROWS > 1 && CANDIDATES == 0), "lane_split is the lanes' form (rows > 1, no candidate row)");
static_assert(CANDIDATES == 0 || ROWS == 1, "the candidate row folds into the one-row scan");
// GT_CANDIDATE_ROW (the host defines it with candidates > 0) compiles the list branch: a discarded `if constexpr`
// branch is still checked in this non-template function, and its TensorAccessorArgs index is out of range without
// the lists tensor.
#ifdef GT_CANDIDATE_ROW
static_assert(CANDIDATES > 0, "GT_CANDIDATE_ROW needs candidates > 0");
#else
static_assert(CANDIDATES == 0, "candidates > 0 needs GT_CANDIDATE_ROW");
#endif
constexpr uint32_t GRAIN = 64;
constexpr uint32_t TILE_BYTES = 2048, FACE_BYTES = 512, ROW_BYTES = 32;

// -0.0 is canonicalized to +0.0 first: the chain compares as floats (the zeros tie, the first lane wins) and its max
// reduce returns +0.0 for a zero maximum.
FORCE_INLINE uint16_t canonical(uint16_t bits) { return (bits & 0x7FFFu) ? bits : uint16_t(0); }
FORCE_INLINE uint32_t key_of(uint16_t bits) { return (bits & 0x8000u) ? (~bits & 0xFFFFu) : (bits | 0x8000u); }

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t pairs_addr = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t core = get_arg_val<uint32_t>(4);
    constexpr auto a_logits = TensorAccessorArgs<0>();
    constexpr auto a_pairs = TensorAccessorArgs<a_logits.next_compile_time_args_offset()>();
    const auto logits = TensorAccessor(a_logits, logits_addr);
    const auto pairs = TensorAccessor(a_pairs, pairs_addr);

    Noc noc;
    DataflowBuffer stage(CB_STAGE);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    volatile tt_l1_ptr uint16_t* halves = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
    if constexpr (ROWS == 1 || LANE_SPLIT) {
        FUSED_ZONE("fz_gt_scan_row");
        // tile t of this core: the two 64-byte grains holding row `row` of faces (row >> 4) * 2 and + 1 land at 128 t
        // and 128 t + 64 (row 0 of faces 0 and 1 for the one-row step); an odd row sits 32 bytes into its grain
        const uint32_t row = LANE_SPLIT ? get_arg_val<uint32_t>(6) : 0;
        const uint32_t face_lo = ((row >> 4) << 1) * FACE_BYTES;
        const uint32_t row_bytes = (row & 15) * ROW_BYTES;
        const uint32_t grain_off = row_bytes & ~(GRAIN - 1);
        const uint32_t in_grain = (row_bytes & (GRAIN - 1)) / 2;  // half-words into the grain: 0 or 16
        {
            FUSED_ZONE("fz_gt_scan_read");
            for (uint32_t t = 0; t < count; ++t) {
                noc.async_read(
                    logits,
                    stage,
                    GRAIN,
                    {.page_id = first + t, .offset_bytes = face_lo + grain_off},
                    {.offset_bytes = 128 * t});
                noc.async_read(
                    logits,
                    stage,
                    GRAIN,
                    {.page_id = first + t, .offset_bytes = face_lo + FACE_BYTES + grain_off},
                    {.offset_bytes = 128 * t + 64});
            }
            noc.async_read_barrier();
        }

        const uint32_t out_offset = ((128 * count) + 15) & ~15u;
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + out_offset);
#ifndef GT_CANDIDATE_ROW
        {
            uint32_t best_key = 0, best_id = 0, best_bits = 0;
            bool any = false;
            for (uint32_t t = 0; t < count; ++t) {
                for (uint32_t lane = 0; lane < LANES; ++lane) {
                    const uint32_t word =
                        64 * t + in_grain + (lane < 16 ? lane : 32 + (lane - 16));  // face 0 row 0, then face 1 row 0
                    const uint16_t bits = canonical(halves[word]);
                    const uint32_t key = key_of(bits);
                    if (!any || key > best_key) {
                        any = true;
                        best_key = key;
                        best_bits = bits;
                        best_id = (first + t) * LANES + lane;
                    }
                }
            }
            out[0] = best_bits << 16;  // the bf16 maximum widened to fp32 (exact)
            out[1] = best_id;
            out[2] = 0;
            out[3] = 0;
            noc.async_write(
                stage, pairs, 16, {.offset_bytes = out_offset}, {.page_id = row, .offset_bytes = 16 * core});
        }
#else
        {
            constexpr auto a_lists = TensorAccessorArgs<a_pairs.next_compile_time_args_offset()>();
            const auto lists = TensorAccessor(a_lists, get_arg_val<uint32_t>(5));
            // The core's top CANDIDATES in key order, lowest id first among equal keys: an element enters only when its
            // key is strictly above the last kept one (an earlier id keeps a boundary slot) and moves left only past
            // strictly smaller keys (the stable insertion), so entry 0 is the first strict maximum in id order: the
            // argmax pair above.
            uint32_t ckey[CANDIDATES], cid[CANDIDATES];
            uint16_t cbits[CANDIDATES];
            uint32_t n = 0;
            for (uint32_t t = 0; t < count; ++t) {
                for (uint32_t lane = 0; lane < LANES; ++lane) {
                    const uint32_t word = 64 * t + in_grain + (lane < 16 ? lane : 32 + (lane - 16));
                    const uint16_t bits = canonical(halves[word]);
                    const uint32_t key = key_of(bits);
                    if (n == CANDIDATES && key <= ckey[CANDIDATES - 1]) {
                        continue;
                    }
                    uint32_t j = n < CANDIDATES ? n : CANDIDATES - 1;
                    while (j > 0 && ckey[j - 1] < key) {
                        ckey[j] = ckey[j - 1];
                        cbits[j] = cbits[j - 1];
                        cid[j] = cid[j - 1];
                        --j;
                    }
                    ckey[j] = key;
                    cbits[j] = bits;
                    cid[j] = (first + t) * LANES + lane;
                    if (n < CANDIDATES) {
                        ++n;
                    }
                }
            }
            out[0] = static_cast<uint32_t>(cbits[0]) << 16;
            out[1] = cid[0];
            out[2] = 0;
            out[3] = 0;
            constexpr uint32_t LIST_BYTES = 8 * CANDIDATES;
            const uint32_t list_offset = out_offset + 16;
            volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + list_offset);
            for (uint32_t i = 0; i < CANDIDATES; ++i) {
                list[i] = i < n ? static_cast<uint32_t>(cbits[i]) << 16 : 0u;  // the bf16 value widened to fp32 (exact)
                list[CANDIDATES + i] =
                    i < n ? cid[i] : 0xFFFFFFFFu;  // no id: an unfilled slot (a core with under CANDIDATES lanes)
            }
            noc.async_write(
                stage, pairs, 16, {.offset_bytes = out_offset}, {.page_id = row, .offset_bytes = 16 * core});
            noc.async_write(
                stage, lists, LIST_BYTES, {.offset_bytes = list_offset}, {.page_id = core, .offset_bytes = 0});
        }
#endif
        noc.async_write_barrier();
    } else {
        FUSED_ZONE("fz_gt_scan_rows");
        // tile t at slot 2048 t: rows 0..15 of faces 0 and 1 (their first LO_BYTES), rows 16.. of faces 2 and 3
        constexpr uint32_t LO_ROWS = ROWS < 16 ? ROWS : 16;
        constexpr uint32_t HI_ROWS = ROWS > 16 ? ROWS - 16 : 0;
        constexpr uint32_t LO_BYTES = (LO_ROWS * ROW_BYTES + GRAIN - 1) & ~(GRAIN - 1);
        constexpr uint32_t HI_BYTES = (HI_ROWS * ROW_BYTES + GRAIN - 1) & ~(GRAIN - 1);
        {
            FUSED_ZONE("fz_gt_scan_read_rows");
            for (uint32_t t = 0; t < count; ++t) {
                const uint32_t slot = TILE_BYTES * t;
                noc.async_read(
                    logits, stage, LO_BYTES, {.page_id = first + t, .offset_bytes = 0}, {.offset_bytes = slot});
                noc.async_read(
                    logits,
                    stage,
                    LO_BYTES,
                    {.page_id = first + t, .offset_bytes = FACE_BYTES},
                    {.offset_bytes = slot + FACE_BYTES});
                if constexpr (HI_ROWS > 0) {
                    noc.async_read(
                        logits,
                        stage,
                        HI_BYTES,
                        {.page_id = first + t, .offset_bytes = 2 * FACE_BYTES},
                        {.offset_bytes = slot + 2 * FACE_BYTES});
                    noc.async_read(
                        logits,
                        stage,
                        HI_BYTES,
                        {.page_id = first + t, .offset_bytes = 3 * FACE_BYTES},
                        {.offset_bytes = slot + 3 * FACE_BYTES});
                }
            }
            noc.async_read_barrier();
        }

        const uint32_t out_offset = TILE_BYTES * count;  // 16-byte aligned
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + out_offset);
        for (uint32_t r = 0; r < ROWS; ++r) {
            uint32_t best_key = 0, best_id = 0, best_bits = 0;
            bool any = false;
            const uint32_t row_half =
                (((r >> 4) << 1) * FACE_BYTES + (r & 15) * ROW_BYTES) / 2;  // face (r >> 4) * 2, row r & 15
            for (uint32_t t = 0; t < count; ++t) {
                const uint32_t slot_half = (TILE_BYTES * t) / 2;
                for (uint32_t lane = 0; lane < LANES; ++lane) {
                    // lanes 0..15 in the even face, 16..31 in the odd face (FACE_BYTES / 2 halves further on)
                    const uint32_t word = slot_half + row_half + (lane < 16 ? lane : (FACE_BYTES / 2) + (lane - 16));
                    const uint16_t bits = canonical(halves[word]);
                    const uint32_t key = key_of(bits);
                    if (!any || key > best_key) {
                        any = true;
                        best_key = key;
                        best_bits = bits;
                        best_id = (first + t) * LANES + lane;
                    }
                }
            }
            out[4 * r] = best_bits << 16;  // the bf16 maximum widened to fp32 (exact)
            out[4 * r + 1] = best_id;
            out[4 * r + 2] = 0;
            out[4 * r + 3] = 0;
            noc.async_write(
                stage, pairs, 16, {.offset_bytes = out_offset + 16 * r}, {.page_id = r, .offset_bytes = 16 * core});
        }
        noc.async_write_barrier();
    }
    stage.push_back(1);
}
