// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One core: the on-device sampler of ttnn/device_sampler.py as one program.  Per row r of the replicated candidate
// rows (fp32 ROW_MAJOR [1,1,rows,256]: shard d's 32 values then its 32 global ids at lanes 64 d ..), the composite's
// arithmetic on the RISC, bit for bit: the presence penalty on the ids the row's history bit array marks (fp32
// subtract), the lanes ordered by value descending and global id ascending (sign-magnitude integer keys, -0.0 as
// +0.0; the top top_k lanes selected), the temperature table indexed by floor(1024 (s_max - s_j)) (fp32 subtract and
// an exact power-of-two multiply; the table entries gathered from DRAM), the exact integer prefix sums, the two fp32
// products tau = top_p * S_k and theta = u * S (the RISC's soft-float multiply is IEEE round-to-nearest-even, as the
// SFPU's), the kept prefix and the inverse CDF over all 128 lanes as the composite counts them, the chosen id into
// lane r of the fp32 token tile, and the id's bit set in the row's history.  The greedy flag copies the greedy tile
// (the composite's greedy_row * 1 + splat * 0).  The history is the row-major uint32 bit array of the L1 shard on
// this core (rows x hist_words words, bit id of row r at word r * hist_words + (id >> 5)).
// Named compile-time args: cb_stage, rows, lanes (128), table_size, hist_words.  Compile-time args: TensorAccessorArgs
// of row, greedy, policy, uniforms, table, token.  Runtime args: 0 row, 1 greedy, 2 policy, 3 uniforms, 4 table,
// 5 history (this core's L1 address), 6 token addresses.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

constexpr uint32_t CB_STAGE = get_named_compile_time_arg_val("cb_stage");
constexpr uint32_t ROWS = get_named_compile_time_arg_val("rows");
constexpr uint32_t LANES = get_named_compile_time_arg_val("lanes");
constexpr uint32_t TABLE_SIZE = get_named_compile_time_arg_val("table_size");
constexpr uint32_t HIST_WORDS = get_named_compile_time_arg_val("hist_words");
constexpr uint32_t SHARD_LANES = 32;           // candidates per shard
constexpr uint32_t ROW_BYTES = 2 * LANES * 4;  // 256 fp32 lanes: values and ids per shard
constexpr uint32_t TILE_BYTES = 4096;          // one fp32 tile
constexpr uint32_t GRAIN = 64;                 // the DRAM read grain
constexpr uint32_t MAX_TOP_K = SHARD_LANES;
constexpr uint32_t STAGE_TILE = 0;
constexpr uint32_t STAGE_ROWS = TILE_BYTES;
constexpr uint32_t STAGE_POLICY = STAGE_ROWS + ROWS * ROW_BYTES;
constexpr uint32_t STAGE_UNIFORMS = STAGE_POLICY + GRAIN;
constexpr uint32_t STAGE_TABLE = STAGE_UNIFORMS + GRAIN;          // MAX_TOP_K 64-byte chunks
constexpr uint32_t STAGE_WORK = STAGE_TABLE + MAX_TOP_K * GRAIN;  // the working arrays (L1, not the RISC stack)
constexpr uint32_t WORK_VALUES = STAGE_WORK / 4, WORK_IDS = WORK_VALUES + LANES, WORK_KEYS = WORK_IDS + LANES;
constexpr uint32_t WORK_SORTED_IDS = WORK_KEYS + LANES, WORK_SORTED_VALUES = WORK_SORTED_IDS + MAX_TOP_K;
constexpr uint32_t WORK_WEIGHTS = WORK_SORTED_VALUES + MAX_TOP_K, WORK_INCLUSIVE = WORK_WEIGHTS + MAX_TOP_K;
constexpr uint32_t WORK_CHUNK = WORK_INCLUSIVE + MAX_TOP_K, WORK_END = WORK_CHUNK + MAX_TOP_K;
constexpr uint32_t STAGE_BYTES = WORK_END * 4;
static_assert(ROWS >= 1 && ROWS <= 16, "rows fit the token tile's first face row");

union Bits {
    float f;
    uint32_t u;
};

FORCE_INLINE uint32_t key_of(uint32_t bits) {
    bits = (bits & 0x7FFFFFFFu) ? bits : 0u;  // -0.0 orders as +0.0 (the SFPU compares them equal)
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

FORCE_INLINE float as_float(uint32_t bits) {
    Bits b;
    b.u = bits;
    return b.f;
}

FORCE_INLINE uint32_t as_bits(float f) {
    Bits b;
    b.f = f;
    return b.u;
}

void kernel_main() {
    constexpr auto a_row = TensorAccessorArgs<0>();
    constexpr auto a_greedy = TensorAccessorArgs<a_row.next_compile_time_args_offset()>();
    constexpr auto a_policy = TensorAccessorArgs<a_greedy.next_compile_time_args_offset()>();
    constexpr auto a_uniforms = TensorAccessorArgs<a_policy.next_compile_time_args_offset()>();
    constexpr auto a_table = TensorAccessorArgs<a_uniforms.next_compile_time_args_offset()>();
    constexpr auto a_token = TensorAccessorArgs<a_table.next_compile_time_args_offset()>();
    const auto row = TensorAccessor(a_row, get_arg_val<uint32_t>(0));
    const auto greedy = TensorAccessor(a_greedy, get_arg_val<uint32_t>(1));
    const auto policy = TensorAccessor(a_policy, get_arg_val<uint32_t>(2));
    const auto uniforms = TensorAccessor(a_uniforms, get_arg_val<uint32_t>(3));
    const auto table = TensorAccessor(a_table, get_arg_val<uint32_t>(4));
    volatile tt_l1_ptr uint32_t* history = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(5));
    const auto token = TensorAccessor(a_token, get_arg_val<uint32_t>(6));

    Noc noc;
    DataflowBuffer stage(CB_STAGE);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    {
        FUSED_ZONE("fz_st_setup");
        for (uint32_t r = 0; r < ROWS; ++r) {
            noc.async_read(
                row, stage, ROW_BYTES, {.page_id = r, .offset_bytes = 0}, {.offset_bytes = STAGE_ROWS + r * ROW_BYTES});
        }
        noc.async_read(policy, stage, GRAIN, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_POLICY});
        noc.async_read(uniforms, stage, GRAIN, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_UNIFORMS});
        noc.async_read_barrier();
    }

    volatile tt_l1_ptr uint32_t* policy_words = words + STAGE_POLICY / 4;
    const uint32_t top_k = static_cast<uint32_t>(as_float(policy_words[0]));  // an exact small integer
    const float top_p = as_float(policy_words[1]);
    const float min_weight = as_float(policy_words[2]);
    const bool greedy_flag = as_float(policy_words[3]) != 0.0f;
    const float presence = as_float(policy_words[4]);

    if (greedy_flag) {
        FUSED_ZONE("fz_st_greedy");
        // the composite's greedy_row * 1 + sampled * 0: the greedy tile, bitwise
        noc.async_read(greedy, stage, TILE_BYTES, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_TILE});
        noc.async_read_barrier();
        noc.async_write(stage, token, TILE_BYTES, {.offset_bytes = STAGE_TILE}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write_barrier();
        stage.push_back(1);
        return;
    }

    volatile tt_l1_ptr uint32_t* tile = words + STAGE_TILE / 4;
    for (uint32_t i = 0; i < TILE_BYTES / 4; ++i) {
        tile[i] = 0;
    }
    volatile tt_l1_ptr uint32_t* values = words + WORK_VALUES;
    volatile tt_l1_ptr uint32_t* ids = words + WORK_IDS;
    volatile tt_l1_ptr uint32_t* keys = words + WORK_KEYS;
    volatile tt_l1_ptr uint32_t* sorted_ids = words + WORK_SORTED_IDS;
    volatile tt_l1_ptr uint32_t* sorted_values = words + WORK_SORTED_VALUES;  // fp32 bits
    volatile tt_l1_ptr uint32_t* weights = words + WORK_WEIGHTS;
    volatile tt_l1_ptr uint32_t* inclusive = words + WORK_INCLUSIVE;
    volatile tt_l1_ptr uint32_t* chunk_offset = words + WORK_CHUNK;
    for (uint32_t r = 0; r < ROWS; ++r) {
        FUSED_ZONE("fz_st_row");
        volatile tt_l1_ptr uint32_t* lanes = words + (STAGE_ROWS + r * ROW_BYTES) / 4;
        volatile tt_l1_ptr uint32_t* hist = history + r * HIST_WORDS;
        for (uint32_t d = 0; d < LANES / SHARD_LANES; ++d) {
            for (uint32_t c = 0; c < SHARD_LANES; ++c) {
                values[d * SHARD_LANES + c] = lanes[d * 2 * SHARD_LANES + c];
                ids[d * SHARD_LANES + c] =
                    static_cast<uint32_t>(as_float(lanes[d * 2 * SHARD_LANES + SHARD_LANES + c]));
            }
        }
        // the presence penalty on the ids this row has emitted (fp32 subtract, round to nearest even)
        if (presence != 0.0f) {
            for (uint32_t j = 0; j < LANES; ++j) {
                if ((hist[ids[j] >> 5] >> (ids[j] & 31)) & 1u) {
                    values[j] = as_bits(as_float(values[j]) - presence);
                }
            }
        }
        for (uint32_t j = 0; j < LANES; ++j) {
            keys[j] = key_of(values[j]);
        }
        // the first top_k lanes of the order value descending, global id ascending
        uint32_t taken[LANES / 32] = {0, 0, 0, 0};
        for (uint32_t k = 0; k < top_k; ++k) {
            uint32_t best = LANES, best_key = 0, best_id = 0;
            for (uint32_t j = 0; j < LANES; ++j) {
                if ((taken[j >> 5] >> (j & 31)) & 1u) {
                    continue;
                }
                if (best == LANES || keys[j] > best_key || (keys[j] == best_key && ids[j] < best_id)) {
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
            if (scaled >= static_cast<float>(TABLE_SIZE - 1)) {
                index = TABLE_SIZE - 1;
            } else if (scaled > 0.0f) {
                index = static_cast<uint32_t>(scaled);  // floor of a positive value
            } else {
                index = 0;  // 0.0 or -0.0 (equal to the maximum)
            }
            const uint32_t byte = index * 4;
            chunk_offset[k] = byte & (GRAIN - 1);
            noc.async_read(
                table,
                stage,
                GRAIN,
                {.page_id = 0, .offset_bytes = byte - chunk_offset[k]},
                {.offset_bytes = STAGE_TABLE + k * GRAIN});
        }
        noc.async_read_barrier();
        for (uint32_t k = 0; k < top_k; ++k) {
            weights[k] = static_cast<uint32_t>(as_float(words[(STAGE_TABLE + k * GRAIN + chunk_offset[k]) / 4]));
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
        for (uint32_t j = 0; j < LANES; ++j) {
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
        const uint32_t total_kept = inclusive[kept - 1];
        volatile tt_l1_ptr uint32_t* uniform_words = words + STAGE_UNIFORMS / 4;
        const float theta = as_float(uniform_words[r]) * static_cast<float>(total_kept);  // RNE site 2
        uint32_t below = 0;
        for (uint32_t j = 0; j < LANES; ++j) {
            const uint32_t prefix = j < top_k ? inclusive[j] : total_top_k;
            if (static_cast<float>(prefix) <= theta) {
                ++below;
            }
        }
        const uint32_t lane = below < kept - 1 ? below : kept - 1;
        const uint32_t chosen = sorted_ids[lane];
        tile[(r >> 4) * 256 + (r & 15)] = as_bits(static_cast<float>(chosen));  // lane (0, r) of the token tile
        hist[chosen >> 5] |= 1u << (chosen & 31);
    }
    noc.async_write(stage, token, TILE_BYTES, {.offset_bytes = STAGE_TILE}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write_barrier();
    stage.push_back(1);
}
