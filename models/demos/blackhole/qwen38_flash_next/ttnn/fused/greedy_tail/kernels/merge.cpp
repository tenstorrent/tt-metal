// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One core: the per-core (value, id) pairs of scan.cpp -> per row the local maximum (bf16 TILE [1,1,rows,1]: lane
// (r, 0) of a zero tile), the local argmax (uint32 [1,1,rows]) and the packed fp32 row [value | float(id)] the
// resolve's all_gather takes.  Cores are visited in increasing order (increasing id ranges), so the first strict
// maximum is the lowest id.  ROWS = 1 (the decode step): the packed row is [1,1,1,2] (8 bytes at page 0).  ROWS > 1
// (the lanes): the pairs rows are pages 0..rows-1, the packed rows [1,1,rows,16] (row r = page r: value, float(id),
// zeros; 64-byte pages at the DRAM grain); the lanes' single row (rows = 1, packed_lanes = 16) takes the same path.
// With candidates > 0 (the decode step of a sampling server) it also merges the cores' sorted lists (scan.cpp) into the
// shard's top `candidates`: the first strict maximum over the list heads in core order (cores hold increasing id
// ranges, a list holds lower ids first among equal keys, so ties keep the lowest id), the values widened to fp32 and
// the ids rebased by the shard's first global id (the chain's typecast and add, exact below 2^24), written as the
// fp32 ROW_MAJOR row [values | global ids] the candidate row's all_gather takes.
// Named compile-time args: cb_stage, cores, rows, packed_lanes, candidates.  Compile-time args:
// TensorAccessorArgs(pairs), (zero tile), (values), (indices), (packed), (lists), (vocab start tile), (row) (the last
// three with candidates > 0).  Runtime args: 0 pairs addr, 1 zero-tile addr, 2 values addr, 3 indices addr,
// 4 packed addr, 5 lists addr, 6 vocab-start addr, 7 row addr.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

constexpr uint32_t CB_STAGE = get_named_compile_time_arg_val("cb_stage");
constexpr uint32_t CORES = get_named_compile_time_arg_val("cores");
constexpr uint32_t ROWS = get_named_compile_time_arg_val("rows");
constexpr uint32_t PACKED_LANES =
    get_named_compile_time_arg_val("packed_lanes");  // 2: [value | id]; 16: the lanes' 64-byte rows
constexpr uint32_t CANDIDATES =
    get_named_compile_time_arg_val("candidates");  // 0, or the sampling server's per-shard row
static_assert(CANDIDATES == 0 || ROWS == 1, "the candidate row folds into the one-row merge");
#ifdef GT_CANDIDATE_ROW  // the host defines it with candidates > 0 (scan.cpp explains the guard)
static_assert(CANDIDATES > 0, "GT_CANDIDATE_ROW needs candidates > 0");
#else
static_assert(CANDIDATES == 0, "candidates > 0 needs GT_CANDIDATE_ROW");
#endif
constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t PAIRS_BYTES = ((16 * CORES) + 63) & ~63u;
constexpr uint32_t PACKED_ROW_BYTES = PACKED_LANES * 4;
static_assert(PACKED_LANES == 2 || PACKED_LANES == 16, "packed rows are [value | id] or 64-byte lane rows");

FORCE_INLINE uint32_t key_of(uint32_t fp32_bits) {
    fp32_bits = (fp32_bits & 0x7FFFFFFFu) ? fp32_bits : 0u;  // scan.cpp already canonicalized -0.0; keep the rule here
    return (fp32_bits & 0x80000000u) ? ~fp32_bits : (fp32_bits | 0x80000000u);
}

union IdFp32 {
    float f;
    uint32_t u;
};

void kernel_main() {
    constexpr auto a_pairs = TensorAccessorArgs<0>();
    constexpr auto a_zero = TensorAccessorArgs<a_pairs.next_compile_time_args_offset()>();
    constexpr auto a_values = TensorAccessorArgs<a_zero.next_compile_time_args_offset()>();
    constexpr auto a_indices = TensorAccessorArgs<a_values.next_compile_time_args_offset()>();
    constexpr auto a_packed = TensorAccessorArgs<a_indices.next_compile_time_args_offset()>();
    const auto pairs = TensorAccessor(a_pairs, get_arg_val<uint32_t>(0));
    const auto zero = TensorAccessor(a_zero, get_arg_val<uint32_t>(1));
    const auto values = TensorAccessor(a_values, get_arg_val<uint32_t>(2));
    const auto indices = TensorAccessor(a_indices, get_arg_val<uint32_t>(3));
    const auto packed = TensorAccessor(a_packed, get_arg_val<uint32_t>(4));

    Noc noc;
    DataflowBuffer stage(CB_STAGE);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
    constexpr uint32_t STAGE_TILE = 0, STAGE_PAIRS = TILE_BYTES;
    noc.async_read(zero, stage, TILE_BYTES, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_TILE});
    if constexpr (ROWS == 1 && PACKED_LANES == 2) {
        FUSED_ZONE("fz_gt_m_one_row");
        constexpr uint32_t STAGE_OUT = TILE_BYTES + PAIRS_BYTES;
        noc.async_read(pairs, stage, PAIRS_BYTES, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_PAIRS});
        noc.async_read_barrier();

        uint32_t best_key = 0, best_bits = 0, best_id = 0;
        for (uint32_t c = 0; c < CORES; ++c) {
            const uint32_t bits = words[STAGE_PAIRS / 4 + 4 * c];
            const uint32_t key = key_of(bits);
            if (c == 0 || key > best_key) {
                best_key = key;
                best_bits = bits;
                best_id = words[STAGE_PAIRS / 4 + 4 * c + 1];
            }
        }
        tile[0] = best_bits >> 16;  // lane (0, 0) of the value tile
        IdFp32 id_fp32;
        id_fp32.f = static_cast<float>(best_id);  // the chain's typecast: exact below 2^24 (soft-float int -> fp32)
        words[STAGE_OUT / 4] = best_bits;
        words[STAGE_OUT / 4 + 1] = id_fp32.u;
        words[STAGE_OUT / 4 + 2] = 0;
        words[STAGE_OUT / 4 + 3] = 0;
        words[STAGE_OUT / 4 + 4] = best_id;
        noc.async_write(stage, values, TILE_BYTES, {.offset_bytes = STAGE_TILE}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(stage, packed, 8, {.offset_bytes = STAGE_OUT}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(stage, indices, 4, {.offset_bytes = STAGE_OUT + 16}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write_barrier();
    } else {
        FUSED_ZONE("fz_gt_m_rows");
        // pairs row r at STAGE_PAIRS + r * PAIRS_BYTES; the packed rows then the indices row after them
        constexpr uint32_t STAGE_PACKED = STAGE_PAIRS + ROWS * PAIRS_BYTES;
        constexpr uint32_t STAGE_IDX = STAGE_PACKED + ROWS * PACKED_ROW_BYTES;
        for (uint32_t r = 0; r < ROWS; ++r) {
            noc.async_read(
                pairs,
                stage,
                PAIRS_BYTES,
                {.page_id = r, .offset_bytes = 0},
                {.offset_bytes = STAGE_PAIRS + r * PAIRS_BYTES});
        }
        noc.async_read_barrier();
        for (uint32_t r = 0; r < ROWS; ++r) {
            const uint32_t row_words = (STAGE_PAIRS + r * PAIRS_BYTES) / 4;
            uint32_t best_key = 0, best_bits = 0, best_id = 0;
            for (uint32_t c = 0; c < CORES; ++c) {
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
            id_fp32.f = static_cast<float>(best_id);
            const uint32_t packed_words = (STAGE_PACKED + r * PACKED_ROW_BYTES) / 4;
            for (uint32_t k = 0; k < PACKED_ROW_BYTES / 4; ++k) {
                words[packed_words + k] = 0;
            }
            words[packed_words] = best_bits;
            words[packed_words + 1] = id_fp32.u;
            words[STAGE_IDX / 4 + r] = best_id;
            noc.async_write(
                stage,
                packed,
                PACKED_ROW_BYTES,
                {.offset_bytes = STAGE_PACKED + r * PACKED_ROW_BYTES},
                {.page_id = r, .offset_bytes = 0});
        }
        noc.async_write(stage, values, TILE_BYTES, {.offset_bytes = STAGE_TILE}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(stage, indices, ROWS * 4, {.offset_bytes = STAGE_IDX}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write_barrier();
    }
#ifdef GT_CANDIDATE_ROW
    {
        FUSED_ZONE("fz_gt_m_candidates");
        constexpr auto a_lists = TensorAccessorArgs<a_packed.next_compile_time_args_offset()>();
        constexpr auto a_start = TensorAccessorArgs<a_lists.next_compile_time_args_offset()>();
        constexpr auto a_row = TensorAccessorArgs<a_start.next_compile_time_args_offset()>();
        const auto lists = TensorAccessor(a_lists, get_arg_val<uint32_t>(5));
        const auto start_tile = TensorAccessor(a_start, get_arg_val<uint32_t>(6));
        const auto row = TensorAccessor(a_row, get_arg_val<uint32_t>(7));
        constexpr uint32_t LIST_BYTES = 8 * CANDIDATES;
        // past the branches' staging (zero tile, pairs rows, packed rows, the indices row), 64-byte aligned
        constexpr uint32_t STAGE_LISTS =
            ((TILE_BYTES + ROWS * PAIRS_BYTES + ROWS * PACKED_ROW_BYTES + 128) + 63) & ~63u;
        constexpr uint32_t STAGE_START = STAGE_LISTS + CORES * LIST_BYTES;
        constexpr uint32_t STAGE_ROW = STAGE_START + 64;
        for (uint32_t c = 0; c < CORES; ++c) {
            noc.async_read(
                lists,
                stage,
                LIST_BYTES,
                {.page_id = c, .offset_bytes = 0},
                {.offset_bytes = STAGE_LISTS + c * LIST_BYTES});
        }
        noc.async_read(
            start_tile, stage, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_START});  // element (0, 0)
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* list_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + STAGE_LISTS);
        volatile tt_l1_ptr uint32_t* out_row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + STAGE_ROW);
        IdFp32 start;
        start.u = words[STAGE_START / 4];
        const uint32_t start_id =
            static_cast<uint32_t>(start.f);  // the shard's first global id (a multiple of the shard width)
        uint32_t head[CORES];
        for (uint32_t c = 0; c < CORES; ++c) {
            head[c] = 0;
        }
        for (uint32_t i = 0; i < CANDIDATES; ++i) {
            uint32_t best_c = CORES, best_key = 0;
            for (uint32_t c = 0; c < CORES; ++c) {
                const uint32_t h = head[c];
                if (h == CANDIDATES || list_words[c * 2 * CANDIDATES + CANDIDATES + h] == 0xFFFFFFFFu) {
                    continue;
                }
                const uint32_t key = key_of(list_words[c * 2 * CANDIDATES + h]);
                if (best_c == CORES || key > best_key) {
                    best_c = c;
                    best_key = key;
                }
            }
            IdFp32 gid;
            if (best_c == CORES) {  // fewer than CANDIDATES lanes in the shard: cannot happen at the vocabulary's width
                out_row[i] = 0;
                gid.f = 0.0f;
                out_row[CANDIDATES + i] = gid.u;
                continue;
            }
            const uint32_t h = head[best_c]++;
            out_row[i] = list_words[best_c * 2 * CANDIDATES + h];
            gid.f = static_cast<float>(list_words[best_c * 2 * CANDIDATES + CANDIDATES + h] + start_id);
            out_row[CANDIDATES + i] = gid.u;
        }
        noc.async_write(stage, row, 8 * CANDIDATES, {.offset_bytes = STAGE_ROW}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write_barrier();
    }
#endif
    stage.push_back(1);
}
