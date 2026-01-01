// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One core: the per-core (value, id) pairs of scan.cpp -> per row the local maximum (bf16 TILE [1,1,rows,1]: lane
// (r, 0) of a zero tile), the local argmax (uint32 [1,1,rows]) and the packed fp32 row [value | float(id)] the
// resolve's all_gather takes.  Cores are visited in increasing id-range order, so the first strict maximum is the
// lowest id.  packed_lanes = 2: one [1,1,1,2] row (8 bytes at page 0); 16: [1,1,rows,16] rows (64-byte pages, row r =
// page r: value, float(id), zeros).
// With candidates > 0 it also merges the cores' sorted lists (scan.cpp) into the shard's top `candidates` (ties keep
// the lowest id), the ids rebased by the shard's first global id (exact below 2^24), written as the fp32 ROW_MAJOR row
// [values | global ids] the candidate row's all_gather takes.
// Named compile-time args: cb_stage, cores, rows, packed_lanes, candidates.  Compile-time args:
// TensorAccessorArgs(pairs), (zero tile), (values), (indices), (packed), (lists), (vocab start tile), (row) (the last
// three with candidates > 0).  Runtime args: 0 pairs addr, 1 zero-tile addr, 2 values addr, 3 indices addr,
// 4 packed addr, 5 lists addr, 6 vocab-start addr, 7 row addr.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_stage = get_named_compile_time_arg_val("cb_stage");
constexpr uint32_t cores = get_named_compile_time_arg_val("cores");
constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
constexpr uint32_t packed_lanes =
    get_named_compile_time_arg_val("packed_lanes");  // 2: [value | id]; 16: the lanes' 64-byte rows
constexpr uint32_t candidates =
    get_named_compile_time_arg_val("candidates");  // 0, or the sampling server's per-shard row
static_assert(candidates == 0 || rows == 1, "the candidate row folds into the one-row merge");
#ifdef GT_CANDIDATE_ROW  // the host defines it with candidates > 0 (scan.cpp explains the guard)
static_assert(candidates > 0, "GT_CANDIDATE_ROW needs candidates > 0");
#else
static_assert(candidates == 0, "candidates > 0 needs GT_CANDIDATE_ROW");
#endif
constexpr uint32_t bf16_tile_bytes = 2048;
constexpr uint32_t pairs_bytes = ((16 * cores) + 63) & ~63u;
constexpr uint32_t packed_row_bytes = packed_lanes * 4;
static_assert(packed_lanes == 2 || packed_lanes == 16, "packed rows are [value | id] or 64-byte lane rows");

FORCE_INLINE uint32_t key_of(uint32_t fp32_bits) {
    fp32_bits = (fp32_bits & 0x7FFFFFFFu) ? fp32_bits : 0u;  // scan.cpp already canonicalized -0.0; keep the rule here
    return (fp32_bits & 0x80000000u) ? ~fp32_bits : (fp32_bits | 0x80000000u);
}

union IdFp32 {
    float value;
    uint32_t word;
};

void kernel_main() {
    constexpr auto pairs_args = TensorAccessorArgs<0>();
    constexpr auto zero_args = TensorAccessorArgs<pairs_args.next_compile_time_args_offset()>();
    constexpr auto values_args = TensorAccessorArgs<zero_args.next_compile_time_args_offset()>();
    constexpr auto indices_args = TensorAccessorArgs<values_args.next_compile_time_args_offset()>();
    constexpr auto packed_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    const auto pairs = TensorAccessor(pairs_args, get_arg_val<uint32_t>(0));
    const auto zero = TensorAccessor(zero_args, get_arg_val<uint32_t>(1));
    const auto values = TensorAccessor(values_args, get_arg_val<uint32_t>(2));
    const auto indices = TensorAccessor(indices_args, get_arg_val<uint32_t>(3));
    const auto packed = TensorAccessor(packed_args, get_arg_val<uint32_t>(4));

    const Noc noc;
    DataflowBuffer stage(cb_stage);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    CoreLocalMem<volatile uint32_t> words(base);
    CoreLocalMem<volatile uint16_t> tile(base);
    constexpr uint32_t stage_tile = 0, stage_pairs = bf16_tile_bytes;
    noc.async_read(zero, stage, bf16_tile_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = stage_tile});
    if constexpr (rows == 1 && packed_lanes == 2) {
        constexpr uint32_t stage_out = bf16_tile_bytes + pairs_bytes;
        noc.async_read(pairs, stage, pairs_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = stage_pairs});
        noc.async_read_barrier();

        uint32_t best_key = 0, best_bits = 0, best_id = 0;
        for (uint32_t c = 0; c < cores; ++c) {
            const uint32_t bits = words[stage_pairs / 4 + 4 * c];
            const uint32_t key = key_of(bits);
            if (c == 0 || key > best_key) {
                best_key = key;
                best_bits = bits;
                best_id = words[stage_pairs / 4 + 4 * c + 1];
            }
        }
        tile[0] = best_bits >> 16;  // lane (0, 0) of the value tile
        IdFp32 id_fp32;
        id_fp32.value = static_cast<float>(best_id);  // the chain's typecast: exact below 2^24 (soft-float int -> fp32)
        words[stage_out / 4] = best_bits;
        words[stage_out / 4 + 1] = id_fp32.word;
        words[stage_out / 4 + 2] = 0;
        words[stage_out / 4 + 3] = 0;
        words[stage_out / 4 + 4] = best_id;
        noc.async_write(
            stage, values, bf16_tile_bytes, {.offset_bytes = stage_tile}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(stage, packed, 8, {.offset_bytes = stage_out}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(stage, indices, 4, {.offset_bytes = stage_out + 16}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write_barrier();
    } else {
        // pairs row r at stage_pairs + r * pairs_bytes; the packed rows then the indices row after them
        constexpr uint32_t stage_packed = stage_pairs + rows * pairs_bytes;
        constexpr uint32_t stage_indices = stage_packed + rows * packed_row_bytes;
        for (uint32_t r = 0; r < rows; ++r) {
            noc.async_read(
                pairs,
                stage,
                pairs_bytes,
                {.page_id = r, .offset_bytes = 0},
                {.offset_bytes = stage_pairs + r * pairs_bytes});
        }
        noc.async_read_barrier();
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t row_words = (stage_pairs + r * pairs_bytes) / 4;
            uint32_t best_key = 0, best_bits = 0, best_id = 0;
            for (uint32_t c = 0; c < cores; ++c) {
                const uint32_t bits = words[row_words + 4 * c];
                const uint32_t key = key_of(bits);
                if (c == 0 || key > best_key) {
                    best_key = key;
                    best_bits = bits;
                    best_id = words[row_words + 4 * c + 1];
                }
            }
            // lane (r, 0) of the value tile: face (r >> 4) * 2, row r & 15 (16-bit word (r >> 4) * 512 + (r & 15) * 16)
            tile[(r >> 4) * 512 + (r & 15) * 16] = best_bits >> 16;
            IdFp32 id_fp32;
            id_fp32.value = static_cast<float>(best_id);
            const uint32_t packed_words = (stage_packed + r * packed_row_bytes) / 4;
            for (uint32_t k = 0; k < packed_row_bytes / 4; ++k) {
                words[packed_words + k] = 0;
            }
            words[packed_words] = best_bits;
            words[packed_words + 1] = id_fp32.word;
            words[stage_indices / 4 + r] = best_id;
            noc.async_write(
                stage,
                packed,
                packed_row_bytes,
                {.offset_bytes = stage_packed + r * packed_row_bytes},
                {.page_id = r, .offset_bytes = 0});
        }
        noc.async_write(
            stage, values, bf16_tile_bytes, {.offset_bytes = stage_tile}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(stage, indices, rows * 4, {.offset_bytes = stage_indices}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write_barrier();
    }
#ifdef GT_CANDIDATE_ROW
    {
        constexpr auto lists_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
        constexpr auto start_args = TensorAccessorArgs<lists_args.next_compile_time_args_offset()>();
        constexpr auto row_args = TensorAccessorArgs<start_args.next_compile_time_args_offset()>();
        const auto lists = TensorAccessor(lists_args, get_arg_val<uint32_t>(5));
        const auto start_tile = TensorAccessor(start_args, get_arg_val<uint32_t>(6));
        const auto row = TensorAccessor(row_args, get_arg_val<uint32_t>(7));
        constexpr uint32_t list_bytes = 8 * candidates;
        // past the branches' staging (zero tile, pairs rows, packed rows, the indices row), 64-byte aligned
        constexpr uint32_t stage_lists =
            ((bf16_tile_bytes + rows * pairs_bytes + rows * packed_row_bytes + 128) + 63) & ~63u;
        constexpr uint32_t stage_start = stage_lists + cores * list_bytes;
        constexpr uint32_t stage_row = stage_start + 64;
        for (uint32_t c = 0; c < cores; ++c) {
            noc.async_read(
                lists,
                stage,
                list_bytes,
                {.page_id = c, .offset_bytes = 0},
                {.offset_bytes = stage_lists + c * list_bytes});
        }
        noc.async_read(
            start_tile, stage, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = stage_start});  // element (0, 0)
        noc.async_read_barrier();
        CoreLocalMem<volatile uint32_t> list_words(base + stage_lists);
        CoreLocalMem<volatile uint32_t> out_row(base + stage_row);
        IdFp32 start;
        start.word = words[stage_start / 4];
        const uint32_t start_id =
            static_cast<uint32_t>(start.value);  // the shard's first global id (a multiple of the shard width)
        uint32_t next_entry[cores];
        for (uint32_t c = 0; c < cores; ++c) {
            next_entry[c] = 0;
        }
        for (uint32_t i = 0; i < candidates; ++i) {
            uint32_t best_core = cores, best_key = 0;
            for (uint32_t c = 0; c < cores; ++c) {
                const uint32_t entry = next_entry[c];
                if (entry == candidates || list_words[c * 2 * candidates + candidates + entry] == 0xFFFFFFFFu) {
                    continue;
                }
                const uint32_t key = key_of(list_words[c * 2 * candidates + entry]);
                if (best_core == cores || key > best_key) {
                    best_core = c;
                    best_key = key;
                }
            }
            IdFp32 global_id;
            // no core has an entry left: fewer than `candidates` lanes in the shard, which cannot happen at the
            // vocabulary's width
            if (best_core == cores) {
                out_row[i] = 0;
                global_id.value = 0.0f;
                out_row[candidates + i] = global_id.word;
                continue;
            }
            const uint32_t entry = next_entry[best_core]++;
            out_row[i] = list_words[best_core * 2 * candidates + entry];
            global_id.value =
                static_cast<float>(list_words[best_core * 2 * candidates + candidates + entry] + start_id);
            out_row[candidates + i] = global_id.word;
        }
        noc.async_write(stage, row, 8 * candidates, {.offset_bytes = stage_row}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write_barrier();
    }
#endif
    stage.push_back(1);
}
