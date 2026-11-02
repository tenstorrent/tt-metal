// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Per core: the maximum of each valid row of a range of bf16 TILE pages of the local logits and the lowest id that
// holds it (ttnn.argmax's tie rule).  Rows 0..15 sit in faces 0 and 1, rows 16.. in faces 2 and 3; rows = 1 reads
// row 0 with the 64-byte DRAM grain.  The pair (value as fp32 bits, id) of row r goes to lanes 4c..4c+1 of row r of
// the fp32 pairs rows.  bf16 values compare through a sign-magnitude key (zeros canonicalized to +0.0), so the first
// strict maximum in id order is the lowest id.
// With candidates > 0 each core also writes its top `candidates` pairs (key order, lowest id first among equal keys)
// as one fp32 ROW_MAJOR page [values | ids] of the lists tensor (page = core); entry 0 is the argmax pair.
// lane_split (rows > 1): each core reads only row r of its tiles and writes its pair to pairs row r, lane 4 g of its
// tile group g; the merge compares the groups in increasing id order, so the result is bitwise the all-rows scan's.
// Named compile-time args: cb_stage, lanes_per_tile, rows, candidates, lane_split.  Compile-time args:
// TensorAccessorArgs(logits), (pairs), (lists when candidates > 0).  Runtime args: 0 logits addr, 1 pairs addr,
// 2 first tile, 3 tile count, 4 core index (the tile group), 5 lists addr (candidates > 0), 6 row (lane_split).

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_stage = get_named_compile_time_arg_val("cb_stage");
constexpr uint32_t lanes_per_tile = get_named_compile_time_arg_val("lanes_per_tile");
constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
constexpr uint32_t candidates = get_named_compile_time_arg_val("candidates");
constexpr uint32_t lane_split = get_named_compile_time_arg_val("lane_split");
static_assert(
    lane_split == 0 || (rows > 1 && candidates == 0), "lane_split is the lanes' form (rows > 1, no candidate row)");
static_assert(candidates == 0 || rows == 1, "the candidate row folds into the one-row scan");
// GT_CANDIDATE_ROW (the host defines it with candidates > 0) compiles the list branch: a discarded `if constexpr`
// branch is still checked in this non-template function, and its TensorAccessorArgs index is out of range without
// the lists tensor.
#ifdef GT_CANDIDATE_ROW
static_assert(candidates > 0, "GT_CANDIDATE_ROW needs candidates > 0");
#else
static_assert(candidates == 0, "candidates > 0 needs GT_CANDIDATE_ROW");
#endif
constexpr uint32_t grain_bytes = 64;
constexpr uint32_t bf16_tile_bytes = 2048, face_bytes = 512, face_row_bytes = 32;

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
    constexpr auto logits_args = TensorAccessorArgs<0>();
    constexpr auto pairs_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    const auto logits = TensorAccessor(logits_args, logits_addr);
    const auto pairs = TensorAccessor(pairs_args, pairs_addr);

    const Noc noc;
    DataflowBuffer stage(cb_stage);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    CoreLocalMem<volatile uint16_t> halves(base);
    if constexpr (rows == 1 || lane_split) {
        // tile t of this core: the two 64-byte grains holding row `row` of faces (row >> 4) * 2 and + 1 land at 128 t
        // and 128 t + 64 (row 0 of faces 0 and 1 for the one-row step); an odd row sits 32 bytes into its grain
        const uint32_t row = lane_split ? get_arg_val<uint32_t>(6) : 0;
        const uint32_t left_face_offset = ((row >> 4) << 1) * face_bytes;
        const uint32_t row_bytes = (row & 15) * face_row_bytes;
        const uint32_t grain_offset = row_bytes & ~(grain_bytes - 1);
        const uint32_t in_grain = (row_bytes & (grain_bytes - 1)) / 2;  // half-words into the grain: 0 or 16
        {
            for (uint32_t t = 0; t < count; ++t) {
                noc.async_read(
                    logits,
                    stage,
                    grain_bytes,
                    {.page_id = first + t, .offset_bytes = left_face_offset + grain_offset},
                    {.offset_bytes = 128 * t});
                noc.async_read(
                    logits,
                    stage,
                    grain_bytes,
                    {.page_id = first + t, .offset_bytes = left_face_offset + face_bytes + grain_offset},
                    {.offset_bytes = 128 * t + 64});
            }
            noc.async_read_barrier();
        }

        const uint32_t out_offset = ((128 * count) + 15) & ~15u;
        CoreLocalMem<volatile uint32_t> out(base + out_offset);
#ifndef GT_CANDIDATE_ROW
        {
            uint32_t best_key = 0, best_id = 0, best_bits = 0;
            bool any = false;
            for (uint32_t t = 0; t < count; ++t) {
                for (uint32_t lane = 0; lane < lanes_per_tile; ++lane) {
                    const uint32_t word =
                        64 * t + in_grain + (lane < 16 ? lane : 32 + (lane - 16));  // face 0 row 0, then face 1 row 0
                    const uint16_t bits = canonical(halves[word]);
                    const uint32_t key = key_of(bits);
                    if (!any || key > best_key) {
                        any = true;
                        best_key = key;
                        best_bits = bits;
                        best_id = (first + t) * lanes_per_tile + lane;
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
            constexpr auto lists_args = TensorAccessorArgs<pairs_args.next_compile_time_args_offset()>();
            const auto lists = TensorAccessor(lists_args, get_arg_val<uint32_t>(5));
            // The core's top `candidates` in key order, lowest id first among equal keys: an element enters only when
            // its key is strictly above the last kept one (an earlier id keeps a boundary slot) and moves left only
            // past strictly smaller keys (the stable insertion), so entry 0 is the first strict maximum in id order:
            // the argmax pair above.
            uint32_t candidate_keys[candidates], candidate_ids[candidates];
            uint16_t candidate_bits[candidates];
            uint32_t kept = 0;
            for (uint32_t t = 0; t < count; ++t) {
                for (uint32_t lane = 0; lane < lanes_per_tile; ++lane) {
                    const uint32_t word = 64 * t + in_grain + (lane < 16 ? lane : 32 + (lane - 16));
                    const uint16_t bits = canonical(halves[word]);
                    const uint32_t key = key_of(bits);
                    if (kept == candidates && key <= candidate_keys[candidates - 1]) {
                        continue;
                    }
                    uint32_t insert_at = kept < candidates ? kept : candidates - 1;
                    while (insert_at > 0 && candidate_keys[insert_at - 1] < key) {
                        candidate_keys[insert_at] = candidate_keys[insert_at - 1];
                        candidate_bits[insert_at] = candidate_bits[insert_at - 1];
                        candidate_ids[insert_at] = candidate_ids[insert_at - 1];
                        --insert_at;
                    }
                    candidate_keys[insert_at] = key;
                    candidate_bits[insert_at] = bits;
                    candidate_ids[insert_at] = (first + t) * lanes_per_tile + lane;
                    if (kept < candidates) {
                        ++kept;
                    }
                }
            }
            out[0] = static_cast<uint32_t>(candidate_bits[0]) << 16;
            out[1] = candidate_ids[0];
            out[2] = 0;
            out[3] = 0;
            constexpr uint32_t list_bytes = 8 * candidates;
            const uint32_t list_offset = out_offset + 16;
            CoreLocalMem<volatile uint32_t> list(base + list_offset);
            for (uint32_t i = 0; i < candidates; ++i) {
                // the bf16 value widened to fp32 (exact); no id marks an unfilled slot (a core with under
                // `candidates` lanes)
                list[i] = i < kept ? static_cast<uint32_t>(candidate_bits[i]) << 16 : 0u;
                list[candidates + i] = i < kept ? candidate_ids[i] : 0xFFFFFFFFu;
            }
            noc.async_write(
                stage, pairs, 16, {.offset_bytes = out_offset}, {.page_id = row, .offset_bytes = 16 * core});
            noc.async_write(
                stage, lists, list_bytes, {.offset_bytes = list_offset}, {.page_id = core, .offset_bytes = 0});
        }
#endif
        noc.async_write_barrier();
    } else {
        // tile t at slot 2048 t: rows 0..15 of faces 0 and 1 (their first low_bytes), rows 16.. of faces 2 and 3
        constexpr uint32_t low_rows = rows < 16 ? rows : 16;
        constexpr uint32_t high_rows = rows > 16 ? rows - 16 : 0;
        constexpr uint32_t low_bytes = (low_rows * face_row_bytes + grain_bytes - 1) & ~(grain_bytes - 1);
        constexpr uint32_t high_bytes = (high_rows * face_row_bytes + grain_bytes - 1) & ~(grain_bytes - 1);
        {
            for (uint32_t t = 0; t < count; ++t) {
                const uint32_t slot = bf16_tile_bytes * t;
                noc.async_read(
                    logits, stage, low_bytes, {.page_id = first + t, .offset_bytes = 0}, {.offset_bytes = slot});
                noc.async_read(
                    logits,
                    stage,
                    low_bytes,
                    {.page_id = first + t, .offset_bytes = face_bytes},
                    {.offset_bytes = slot + face_bytes});
                if constexpr (high_rows > 0) {
                    noc.async_read(
                        logits,
                        stage,
                        high_bytes,
                        {.page_id = first + t, .offset_bytes = 2 * face_bytes},
                        {.offset_bytes = slot + 2 * face_bytes});
                    noc.async_read(
                        logits,
                        stage,
                        high_bytes,
                        {.page_id = first + t, .offset_bytes = 3 * face_bytes},
                        {.offset_bytes = slot + 3 * face_bytes});
                }
            }
            noc.async_read_barrier();
        }

        const uint32_t out_offset = bf16_tile_bytes * count;  // 16-byte aligned
        CoreLocalMem<volatile uint32_t> out(base + out_offset);
        for (uint32_t r = 0; r < rows; ++r) {
            uint32_t best_key = 0, best_id = 0, best_bits = 0;
            bool any = false;
            const uint32_t row_half =
                (((r >> 4) << 1) * face_bytes + (r & 15) * face_row_bytes) / 2;  // face (r >> 4) * 2, row r & 15
            for (uint32_t t = 0; t < count; ++t) {
                const uint32_t slot_half = (bf16_tile_bytes * t) / 2;
                for (uint32_t lane = 0; lane < lanes_per_tile; ++lane) {
                    // lanes 0..15 in the even face, 16..31 in the odd face (face_bytes / 2 halves further on)
                    const uint32_t word = slot_half + row_half + (lane < 16 ? lane : (face_bytes / 2) + (lane - 16));
                    const uint16_t bits = canonical(halves[word]);
                    const uint32_t key = key_of(bits);
                    if (!any || key > best_key) {
                        any = true;
                        best_key = key;
                        best_bits = bits;
                        best_id = (first + t) * lanes_per_tile + lane;
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
