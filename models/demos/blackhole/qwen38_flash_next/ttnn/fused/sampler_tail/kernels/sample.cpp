// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One core: the on-device sampler of ttnn/device_sampler.py, bit for bit on the RISC.  Per row r of the candidate rows
// (fp32 ROW_MAJOR [1,1,rows,256]: shard d's 32 values then its 32 global ids at lanes 64 d ..): the presence penalty on
// the ids the row's history marks (fp32 subtract), the top_k lanes by value descending and global id ascending (-0.0 as
// +0.0), the temperature table at floor(1024 (s_max - s_j)), exact integer prefix sums, tau = top_p * S_k and
// theta = u * S (soft-float RNE, as the SFPU's), and the kept prefix and inverse CDF as the composite counts them; the
// chosen id goes to lane r of the fp32 token tile and its bit into the row's history.  The greedy flag copies the
// greedy tile.  History: rows x history_words uint32 words in this core's L1 shard, bit id of row r at word
// r * history_words + (id >> 5).  Bit r of the guard page's lane 0 (fp32) is set when a token row r did not read could
// change its draw (the kept or top-k minimum below the shard floor, the largest shard minimum); 0 under greedy.
// Named compile-time args: cb_stage, rows, lanes (128), table_size, hist_words (history_words).  Compile-time args:
// TensorAccessorArgs of row, greedy, policy, uniforms, table, token, guard.  Runtime args: 0 row, 1 greedy, 2 policy,
// 3 uniforms, 4 table, 5 history (this core's L1 address), 6 token, 7 guard addresses.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_stage = get_named_compile_time_arg_val("cb_stage");
constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
constexpr uint32_t lanes = get_named_compile_time_arg_val("lanes");
constexpr uint32_t table_size = get_named_compile_time_arg_val("table_size");
constexpr uint32_t history_words = get_named_compile_time_arg_val("hist_words");
constexpr uint32_t shard_lanes = 32;           // candidates per shard
constexpr uint32_t row_bytes = 2 * lanes * 4;  // 256 fp32 lanes: values and ids per shard
constexpr uint32_t fp32_tile_bytes = 4096;     // one fp32 tile
constexpr uint32_t grain_bytes = 64;           // the DRAM read grain
constexpr uint32_t max_top_k = shard_lanes;
constexpr uint32_t stage_tile = 0;
constexpr uint32_t stage_rows = fp32_tile_bytes;
constexpr uint32_t stage_policy = stage_rows + rows * row_bytes;
constexpr uint32_t stage_uniforms = stage_policy + grain_bytes;
constexpr uint32_t stage_table = stage_uniforms + grain_bytes;  // max_top_k 64-byte chunks
constexpr uint32_t stage_guard = stage_table + max_top_k * grain_bytes;
constexpr uint32_t stage_work = stage_guard + grain_bytes;  // the working arrays (L1, not the RISC stack)
constexpr uint32_t work_values = stage_work / 4, work_ids = work_values + lanes, work_keys = work_ids + lanes;
constexpr uint32_t work_sorted_ids = work_keys + lanes, work_sorted_values = work_sorted_ids + max_top_k;
constexpr uint32_t work_weights = work_sorted_values + max_top_k, work_inclusive = work_weights + max_top_k;
constexpr uint32_t work_chunk = work_inclusive + max_top_k, work_end = work_chunk + max_top_k;
constexpr uint32_t stage_bytes = work_end * 4;
static_assert(rows >= 1 && rows <= 16, "rows fit the token tile's first face row");

union Bits {
    float value;
    uint32_t word;
};

FORCE_INLINE uint32_t key_of(uint32_t bits) {
    bits = (bits & 0x7FFFFFFFu) ? bits : 0u;  // -0.0 orders as +0.0 (the SFPU compares them equal)
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

FORCE_INLINE float as_float(uint32_t bits) {
    Bits pun;
    pun.word = bits;
    return pun.value;
}

FORCE_INLINE uint32_t as_bits(float value) {
    Bits pun;
    pun.value = value;
    return pun.word;
}

void kernel_main() {
    constexpr auto row_args = TensorAccessorArgs<0>();
    constexpr auto greedy_args = TensorAccessorArgs<row_args.next_compile_time_args_offset()>();
    constexpr auto policy_args = TensorAccessorArgs<greedy_args.next_compile_time_args_offset()>();
    constexpr auto uniforms_args = TensorAccessorArgs<policy_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<uniforms_args.next_compile_time_args_offset()>();
    constexpr auto token_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto guard_args = TensorAccessorArgs<token_args.next_compile_time_args_offset()>();
    const auto row = TensorAccessor(row_args, get_arg_val<uint32_t>(0));
    const auto greedy = TensorAccessor(greedy_args, get_arg_val<uint32_t>(1));
    const auto policy = TensorAccessor(policy_args, get_arg_val<uint32_t>(2));
    const auto uniforms = TensorAccessor(uniforms_args, get_arg_val<uint32_t>(3));
    const auto table = TensorAccessor(table_args, get_arg_val<uint32_t>(4));
    CoreLocalMem<volatile uint32_t> history(get_arg_val<uint32_t>(5));
    const auto token = TensorAccessor(token_args, get_arg_val<uint32_t>(6));
    const auto guard = TensorAccessor(guard_args, get_arg_val<uint32_t>(7));

    const Noc noc;
    DataflowBuffer stage(cb_stage);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    CoreLocalMem<volatile uint32_t> words(base);
    {
        for (uint32_t r = 0; r < rows; ++r) {
            noc.async_read(
                row, stage, row_bytes, {.page_id = r, .offset_bytes = 0}, {.offset_bytes = stage_rows + r * row_bytes});
        }
        noc.async_read(policy, stage, grain_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = stage_policy});
        noc.async_read(
            uniforms, stage, grain_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = stage_uniforms});
        noc.async_read_barrier();
    }

    CoreLocalMem<volatile uint32_t> policy_words = words + stage_policy / 4;
    const uint32_t top_k = static_cast<uint32_t>(as_float(policy_words[0]));  // an exact small integer
    const float top_p = as_float(policy_words[1]);
    const float min_weight = as_float(policy_words[2]);
    const bool greedy_flag = as_float(policy_words[3]) != 0.0f;
    const float presence = as_float(policy_words[4]);
    CoreLocalMem<volatile uint32_t> guard_words = words + stage_guard / 4;
    for (uint32_t i = 0; i < grain_bytes / 4; ++i) {
        guard_words[i] = 0;
    }

    if (greedy_flag) {
        // the composite's greedy_row * 1 + sampled * 0: the greedy tile, bitwise
        noc.async_read(greedy, stage, fp32_tile_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = stage_tile});
        noc.async_read_barrier();
        noc.async_write(stage, token, fp32_tile_bytes, {.offset_bytes = stage_tile}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(stage, guard, grain_bytes, {.offset_bytes = stage_guard}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write_barrier();
        stage.push_back(1);
        return;
    }

    CoreLocalMem<volatile uint32_t> tile = words + stage_tile / 4;
    for (uint32_t i = 0; i < fp32_tile_bytes / 4; ++i) {
        tile[i] = 0;
    }
    CoreLocalMem<volatile uint32_t> values = words + work_values;
    CoreLocalMem<volatile uint32_t> ids = words + work_ids;
    CoreLocalMem<volatile uint32_t> keys = words + work_keys;
    CoreLocalMem<volatile uint32_t> sorted_ids = words + work_sorted_ids;
    CoreLocalMem<volatile uint32_t> sorted_values = words + work_sorted_values;  // fp32 bits
    CoreLocalMem<volatile uint32_t> weights = words + work_weights;
    CoreLocalMem<volatile uint32_t> inclusive = words + work_inclusive;
    CoreLocalMem<volatile uint32_t> chunk_offset = words + work_chunk;
    uint32_t guard_mask = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        CoreLocalMem<volatile uint32_t> row_lanes = words + (stage_rows + r * row_bytes) / 4;
        CoreLocalMem<volatile uint32_t> row_history = history + r * history_words;
        uint32_t floor_key = 0;  // the shard floor: an unread token is at most its shard's smallest raw value
        for (uint32_t d = 0; d < lanes / shard_lanes; ++d) {
            uint32_t shard_min_key = 0xFFFFFFFFu;
            for (uint32_t c = 0; c < shard_lanes; ++c) {
                const uint32_t lane = d * shard_lanes + c;
                values[lane] = row_lanes[d * 2 * shard_lanes + c];
                ids[lane] = static_cast<uint32_t>(as_float(row_lanes[d * 2 * shard_lanes + shard_lanes + c]));
                keys[lane] = key_of(values[lane]);
                if (keys[lane] < shard_min_key) {
                    shard_min_key = keys[lane];
                }
            }
            if (shard_min_key > floor_key) {
                floor_key = shard_min_key;
            }
        }
        // the presence penalty on the ids this row has emitted (fp32 subtract, round to nearest even)
        if (presence != 0.0f) {
            for (uint32_t j = 0; j < lanes; ++j) {
                if ((row_history[ids[j] >> 5] >> (ids[j] & 31)) & 1u) {
                    values[j] = as_bits(as_float(values[j]) - presence);
                    keys[j] = key_of(values[j]);
                }
            }
        }
        // the first top_k lanes of the order value descending, global id ascending
        uint32_t taken[lanes / 32] = {0, 0, 0, 0};
        for (uint32_t k = 0; k < top_k; ++k) {
            uint32_t best = lanes, best_key = 0, best_id = 0;
            for (uint32_t j = 0; j < lanes; ++j) {
                if ((taken[j >> 5] >> (j & 31)) & 1u) {
                    continue;
                }
                if (best == lanes || keys[j] > best_key || (keys[j] == best_key && ids[j] < best_id)) {
                    best = j;
                    best_key = keys[j];
                    best_id = ids[j];
                }
            }
            taken[best >> 5] |= 1u << (best & 31);
            sorted_values[k] = values[best];
            sorted_ids[k] = ids[best];
        }
        // the table weights: index floor(1024 (s_max - s_j)), clamped; gathered as 64-byte chunks
        const float s_max = as_float(sorted_values[0]);
        for (uint32_t k = 0; k < top_k; ++k) {
            const float below = as_float(sorted_values[k]) - s_max;  // <= 0, fp32 RNE (the composite's subtract)
            const float scaled = below * -1024.0f;                   // exact: a power of two
            uint32_t index;
            if (scaled >= static_cast<float>(table_size - 1)) {
                index = table_size - 1;
            } else if (scaled > 0.0f) {
                index = static_cast<uint32_t>(scaled);  // floor of a positive value
            } else {
                index = 0;  // 0.0 or -0.0 (equal to the maximum)
            }
            const uint32_t byte = index * 4;
            chunk_offset[k] = byte & (grain_bytes - 1);
            noc.async_read(
                table,
                stage,
                grain_bytes,
                {.page_id = 0, .offset_bytes = byte - chunk_offset[k]},
                {.offset_bytes = stage_table + k * grain_bytes});
        }
        noc.async_read_barrier();
        for (uint32_t k = 0; k < top_k; ++k) {
            weights[k] = static_cast<uint32_t>(as_float(words[(stage_table + k * grain_bytes + chunk_offset[k]) / 4]));
        }
        // exact prefix sums; the lanes at and beyond top_k weigh 0 (inclusive = the top-k total)
        uint32_t running = 0;
        for (uint32_t k = 0; k < top_k; ++k) {
            running += weights[k];
            inclusive[k] = running;
        }
        const uint32_t total_top_k = running;
        const float tau = top_p * static_cast<float>(total_top_k);  // RNE site 1
        uint32_t kept_top_p = 0, kept_min_weight = 0;
        for (uint32_t j = 0; j < lanes; ++j) {
            const uint32_t exclusive = j < top_k ? inclusive[j] - weights[j] : total_top_k;
            if (static_cast<float>(exclusive) <= tau) {
                ++kept_top_p;
            }
            if (j < top_k && static_cast<float>(weights[j]) >= min_weight) {
                ++kept_min_weight;
            }
        }
        uint32_t kept = kept_top_p < kept_min_weight ? kept_top_p : kept_min_weight;
        if (kept < 1) {
            kept = 1;  // lane 0 weighs 2**18 and its exclusive prefix is 0: never taken, the composite's invariant
        }
        if (key_of(sorted_values[kept - 1]) <= floor_key || key_of(sorted_values[top_k - 1]) < floor_key) {
            guard_mask |= 1u << r;
        }
        const uint32_t total_kept = inclusive[kept - 1];
        CoreLocalMem<volatile uint32_t> uniform_words = words + stage_uniforms / 4;
        const float theta = as_float(uniform_words[r]) * static_cast<float>(total_kept);  // RNE site 2
        uint32_t below = 0;
        for (uint32_t j = 0; j < lanes; ++j) {
            const uint32_t prefix = j < top_k ? inclusive[j] : total_top_k;
            if (static_cast<float>(prefix) <= theta) {
                ++below;
            }
        }
        const uint32_t lane = below < kept - 1 ? below : kept - 1;
        const uint32_t chosen = sorted_ids[lane];
        tile[(r >> 4) * 256 + (r & 15)] = as_bits(static_cast<float>(chosen));  // lane (0, r) of the token tile
        row_history[chosen >> 5] |= 1u << (chosen & 31);
    }
    guard_words[0] = as_bits(static_cast<float>(guard_mask));
    noc.async_write(stage, token, fp32_tile_bytes, {.offset_bytes = stage_tile}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write(stage, guard, grain_bytes, {.offset_bytes = stage_guard}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write_barrier();
    stage.push_back(1);
}
