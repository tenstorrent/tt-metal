// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The MTP pass's point-mass acceptance on one core, the device sampler's law (must match sampler_tail's sample.cpp:
// the same table weights, exact integer prefix sums and one fp32 multiply per decision, no division anywhere).
// Rows 0 .. k-1 are the verify rows facing drafts d_1 .. d_k, row k the bonus row.  Row j accepts d_{j+1} iff
// fl32(u_j * S_j) < w_j(d_{j+1}) with S_j the kept total of row j and w_j(d) the kept weight of d (0 when d is not
// kept; a tie rejects).  At the first rejection x* is drawn from row j without d (its weight skipped in the prefix
// sums) with the second uniform v; with every draft accepted x* is drawn from the bonus row with v.  Outputs are the
// split verify's decision buffers as write_verify_decision writes them (accept tile, accept index, next token, the
// alignment lanes [d_1 .. d_a*, x*, sentinel ...]) and a statistics row.
// Named compile-time args: cb_stage, rows (k + 1), lanes (128), table_size.  Compile-time args: TensorAccessorArgs of
// candidate_rows, draft_row, policy, uniforms, table, accept_tile, accept_index, next_token, alignment, stats.  Runtime
// args: the ten addresses in that order.

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
constexpr uint32_t drafts = rows - 1;
constexpr uint32_t shard_lanes = 32;
constexpr uint32_t row_bytes = 2 * lanes * 4;
constexpr uint32_t fp32_tile_bytes = 4096;
constexpr uint32_t grain_bytes = 64;
constexpr uint32_t token_row_bytes = 32 * 4;  // the fp32 [1,1,1,32] rows: drafts, uniforms, alignment
constexpr uint32_t stats_lanes = 16;
constexpr uint32_t max_top_k = shard_lanes;
constexpr uint32_t stage_tile = 0;
constexpr uint32_t stage_rows = fp32_tile_bytes;
constexpr uint32_t stage_drafts = stage_rows + rows * row_bytes;
constexpr uint32_t stage_policy = stage_drafts + token_row_bytes;
constexpr uint32_t stage_uniforms = stage_policy + grain_bytes;
constexpr uint32_t stage_alignment = stage_uniforms + token_row_bytes;
constexpr uint32_t stage_stats = stage_alignment + token_row_bytes;
constexpr uint32_t stage_index = stage_stats + stats_lanes * 4;   // the accept index word (its own aligned grain)
constexpr uint32_t stage_next_token = stage_index + grain_bytes;  // the next token word (its own aligned grain: a NoC
                                                                  // write's L1 source is read at grain alignment)
constexpr uint32_t stage_table = stage_next_token + grain_bytes;
constexpr uint32_t stage_work = stage_table + max_top_k * grain_bytes;
constexpr uint32_t work_values = stage_work / 4, work_ids = work_values + lanes, work_keys = work_ids + lanes;
constexpr uint32_t work_sorted_ids = work_keys + lanes, work_sorted_values = work_sorted_ids + max_top_k;
constexpr uint32_t work_weights = work_sorted_values + max_top_k, work_inclusive = work_weights + max_top_k;
constexpr uint32_t work_chunk = work_inclusive + max_top_k, work_end = work_chunk + max_top_k;
constexpr uint32_t stage_bytes = work_end * 4;
static_assert(rows >= 2 && rows <= 6, "a pass has 1..5 drafts and its bonus row");
static_assert(drafts + 2 <= 32, "the uniforms row holds u_0 .. u_{k-1} and v");

// statistics lanes
constexpr uint32_t stat_weight = 0;      // w_j(d_{j+1}) for j < 5, -1 where the row was not evaluated
constexpr uint32_t stat_total = 5;       // S_j (the kept total) for j < 5, -1 where not evaluated
constexpr uint32_t stat_guard = 10;      // bit j: row j's kept minimum did not clear the shard floor
constexpr uint32_t stat_resampled = 11;  // 1 when x* came from a rejected row's residual
constexpr uint32_t stat_accepted = 12;   // a*
constexpr uint32_t stat_token = 13;      // x*
constexpr uint32_t stat_theta = 14;      // fl32(v * S) of the draw that produced x*
constexpr uint32_t stat_kept = 15;       // the kept count of the row that produced x*

union Bits {
    float value;
    uint32_t word;
};

FORCE_INLINE uint32_t key_of(uint32_t bits) {
    bits = (bits & 0x7FFFFFFFu) ? bits : 0u;
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

namespace {

// The policy row's sampling parameters.
struct SamplingPolicy {
    uint32_t top_k;
    float top_p;
    float min_weight;
};

// Row `row`'s kept set in the work arrays of the stage at `words`: sorted_ids / sorted_values / weights / inclusive
// over its top_k lanes, `kept` and its total `total_kept`; bit `row` of `guard_mask` when the kept minimum does not
// clear the shard floor (the largest shard minimum). The table weights are read through `stage`.
template <typename Table>
void build_row(
    uint32_t row,
    CoreLocalMem<volatile uint32_t> words,
    const SamplingPolicy& sampling_policy,
    const Table& table,
    DataflowBuffer& stage,
    uint32_t& kept,
    uint32_t& total_kept,
    uint32_t& guard_mask) {
    const Noc noc;
    const uint32_t top_k = sampling_policy.top_k;
    const float top_p = sampling_policy.top_p;
    const float min_weight = sampling_policy.min_weight;
    CoreLocalMem<volatile uint32_t> values = words + work_values;
    CoreLocalMem<volatile uint32_t> ids = words + work_ids;
    CoreLocalMem<volatile uint32_t> keys = words + work_keys;
    CoreLocalMem<volatile uint32_t> sorted_ids = words + work_sorted_ids;
    CoreLocalMem<volatile uint32_t> sorted_values = words + work_sorted_values;
    CoreLocalMem<volatile uint32_t> weights = words + work_weights;
    CoreLocalMem<volatile uint32_t> inclusive = words + work_inclusive;
    CoreLocalMem<volatile uint32_t> chunk_offset = words + work_chunk;
    uint32_t floor_key = 0;
    {
        CoreLocalMem<volatile uint32_t> row_lanes = words + (stage_rows + row * row_bytes) / 4;
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
    }
    {
        const float s_max = as_float(sorted_values[0]);
        for (uint32_t k = 0; k < top_k; ++k) {
            const float below = as_float(sorted_values[k]) - s_max;
            const float scaled = below * -1024.0f;
            uint32_t index;
            if (scaled >= static_cast<float>(table_size - 1)) {
                index = table_size - 1;
            } else if (scaled > 0.0f) {
                index = static_cast<uint32_t>(scaled);
            } else {
                index = 0;
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
        uint32_t running = 0;
        for (uint32_t k = 0; k < top_k; ++k) {
            weights[k] = static_cast<uint32_t>(as_float(words[(stage_table + k * grain_bytes + chunk_offset[k]) / 4]));
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
        kept = kept_top_p < kept_min_weight ? kept_top_p : kept_min_weight;
        if (kept < 1) {
            kept = 1;
        }
        total_kept = inclusive[kept - 1];
        if (key_of(sorted_values[kept - 1]) <= floor_key) {
            guard_mask |= 1u << row;
        }
    }
}

// The theta rule over the prefix sums of the first `kept` lanes of the stage's work arrays at `words`, skipping lane
// `skip` (`lanes` = no skip): the lane whose prefix first exceeds fl32(u * total), capped at the last kept lane; the
// bits of theta go to `theta_bits`.
FORCE_INLINE uint32_t draw(
    uint32_t uniform_bits,
    uint32_t skip,
    uint32_t total,
    uint32_t kept,
    CoreLocalMem<volatile uint32_t> words,
    uint32_t& theta_bits) {
    CoreLocalMem<volatile uint32_t> sorted_ids = words + work_sorted_ids;
    CoreLocalMem<volatile uint32_t> weights = words + work_weights;
    const float theta = as_float(uniform_bits) * static_cast<float>(total);  // RNE site 2
    theta_bits = as_bits(theta);
    uint32_t below = 0, count = 0, running = 0;
    for (uint32_t i = 0; i < kept; ++i) {
        if (i == skip) {
            continue;
        }
        running += weights[i];
        ++count;
        if (static_cast<float>(running) <= theta) {
            ++below;
        }
    }
    if (count == 0) {
        return sorted_ids[skip == 0 ? 1 : 0];  // unreachable: an accepted-with-certainty lane is never rejected
    }
    uint32_t lane = below < count - 1 ? below : count - 1;
    for (uint32_t i = 0; i < kept; ++i) {
        if (i == skip) {
            continue;
        }
        if (lane == 0) {
            return sorted_ids[i];
        }
        --lane;
    }
    return sorted_ids[kept - 1];
}

}  // namespace

void kernel_main() {
    constexpr auto candidate_rows_args = TensorAccessorArgs<0>();
    constexpr auto draft_row_args = TensorAccessorArgs<candidate_rows_args.next_compile_time_args_offset()>();
    constexpr auto policy_args = TensorAccessorArgs<draft_row_args.next_compile_time_args_offset()>();
    constexpr auto uniforms_args = TensorAccessorArgs<policy_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<uniforms_args.next_compile_time_args_offset()>();
    constexpr auto accept_tile_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto accept_index_args = TensorAccessorArgs<accept_tile_args.next_compile_time_args_offset()>();
    constexpr auto next_token_args = TensorAccessorArgs<accept_index_args.next_compile_time_args_offset()>();
    constexpr auto alignment_args = TensorAccessorArgs<next_token_args.next_compile_time_args_offset()>();
    constexpr auto stats_args = TensorAccessorArgs<alignment_args.next_compile_time_args_offset()>();
    const auto candidate_rows = TensorAccessor(candidate_rows_args, get_arg_val<uint32_t>(0));
    const auto draft_row = TensorAccessor(draft_row_args, get_arg_val<uint32_t>(1));
    const auto policy = TensorAccessor(policy_args, get_arg_val<uint32_t>(2));
    const auto uniforms = TensorAccessor(uniforms_args, get_arg_val<uint32_t>(3));
    const auto table = TensorAccessor(table_args, get_arg_val<uint32_t>(4));
    const auto accept_tile = TensorAccessor(accept_tile_args, get_arg_val<uint32_t>(5));
    const auto accept_index = TensorAccessor(accept_index_args, get_arg_val<uint32_t>(6));
    const auto next_token = TensorAccessor(next_token_args, get_arg_val<uint32_t>(7));
    const auto alignment = TensorAccessor(alignment_args, get_arg_val<uint32_t>(8));
    const auto stats = TensorAccessor(stats_args, get_arg_val<uint32_t>(9));

    const Noc noc;
    DataflowBuffer stage(cb_stage);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    CoreLocalMem<volatile uint32_t> words(base);
    {
        for (uint32_t row = 0; row < rows; ++row) {
            noc.async_read(
                candidate_rows,
                stage,
                row_bytes,
                {.page_id = row, .offset_bytes = 0},
                {.offset_bytes = stage_rows + row * row_bytes});
        }
        noc.async_read(
            draft_row, stage, token_row_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = stage_drafts});
        noc.async_read(policy, stage, grain_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = stage_policy});
        noc.async_read(
            uniforms, stage, token_row_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = stage_uniforms});
        noc.async_read_barrier();
    }

    CoreLocalMem<volatile uint32_t> policy_words = words + stage_policy / 4;
    const SamplingPolicy sampling_policy = {
        .top_k = static_cast<uint32_t>(as_float(policy_words[0])),
        .top_p = as_float(policy_words[1]),
        .min_weight = as_float(policy_words[2])};
    CoreLocalMem<volatile uint32_t> draft_words = words + stage_drafts / 4;
    CoreLocalMem<volatile uint32_t> uniform_words = words + stage_uniforms / 4;
    CoreLocalMem<volatile uint32_t> alignment_words = words + stage_alignment / 4;
    CoreLocalMem<volatile uint32_t> stat_words = words + stage_stats / 4;
    CoreLocalMem<volatile uint32_t> index_words = words + stage_index / 4;
    CoreLocalMem<volatile uint32_t> next_token_words = words + stage_next_token / 4;
    CoreLocalMem<volatile uint32_t> tile = words + stage_tile / 4;
    CoreLocalMem<volatile uint32_t> sorted_ids = words + work_sorted_ids;
    CoreLocalMem<volatile uint32_t> weights = words + work_weights;
    for (uint32_t i = 0; i < fp32_tile_bytes / 4; ++i) {
        tile[i] = 0;
    }
    for (uint32_t i = 0; i < stats_lanes; ++i) {
        stat_words[i] = as_bits(i < stat_guard ? -1.0f : 0.0f);
    }
    for (uint32_t i = 0; i < grain_bytes / 4; ++i) {
        index_words[i] = 0;
        next_token_words[i] = 0;
    }
    const uint32_t sentinel = draft_words[drafts];  // the draft lanes past k hold the zero-embedding sentinel
    for (uint32_t i = 0; i < 32; ++i) {
        alignment_words[i] = sentinel;
    }

    // the last built row's kept count and total; the guard bits of every row built
    uint32_t kept = 0, total_kept = 0, guard_mask = 0;
    uint32_t accepted = drafts, token = 0, theta_bits = 0, resampled = 0, decided_kept = 0;
    {
        for (uint32_t j = 0; j < drafts; ++j) {
            build_row(j, words, sampling_policy, table, stage, kept, total_kept, guard_mask);
            const uint32_t draft = static_cast<uint32_t>(as_float(draft_words[j]));
            uint32_t draft_lane = lanes, draft_weight = 0;
            for (uint32_t i = 0; i < kept; ++i) {
                if (sorted_ids[i] == draft) {
                    draft_lane = i;
                    draft_weight = weights[i];
                    break;
                }
            }
            stat_words[stat_weight + j] = as_bits(static_cast<float>(draft_weight));
            stat_words[stat_total + j] = as_bits(static_cast<float>(total_kept));
            const float draft_theta = as_float(uniform_words[j]) * static_cast<float>(total_kept);  // RNE
            if (draft_theta < static_cast<float>(draft_weight)) {
                alignment_words[j] = draft_words[j];
                continue;
            }
            accepted = j;
            token = draw(uniform_words[drafts], draft_lane, total_kept - draft_weight, kept, words, theta_bits);
            resampled = 1;
            decided_kept = kept;
            break;
        }
    }
    if (accepted == drafts) {
        build_row(drafts, words, sampling_policy, table, stage, kept, total_kept, guard_mask);
        token = draw(uniform_words[drafts], lanes, total_kept, kept, words, theta_bits);
        decided_kept = kept;
    }
    {
        alignment_words[accepted] = as_bits(static_cast<float>(token));
        tile[0] = as_bits(static_cast<float>(accepted));
        index_words[0] = accepted;
        next_token_words[0] = as_bits(static_cast<float>(token));
        stat_words[stat_guard] = as_bits(static_cast<float>(guard_mask));
        stat_words[stat_resampled] = as_bits(static_cast<float>(resampled));
        stat_words[stat_accepted] = as_bits(static_cast<float>(accepted));
        stat_words[stat_token] = as_bits(static_cast<float>(token));
        stat_words[stat_theta] = theta_bits;
        stat_words[stat_kept] = as_bits(static_cast<float>(decided_kept));

        noc.async_write(
            stage, accept_tile, fp32_tile_bytes, {.offset_bytes = stage_tile}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(stage, accept_index, 4, {.offset_bytes = stage_index}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(stage, next_token, 4, {.offset_bytes = stage_next_token}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(
            stage, alignment, token_row_bytes, {.offset_bytes = stage_alignment}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(
            stage, stats, stats_lanes * 4, {.offset_bytes = stage_stats}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write_barrier();
    }
    stage.push_back(1);
}
