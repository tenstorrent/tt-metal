// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The MTP pass's point-mass acceptance on one core, the device sampler's law (fused/sampler_tail/kernels/sample.cpp:
// the same table weights, exact integer prefix sums and the single fp32 multiply per decision, no division anywhere).
// Rows 0 .. k-1 are the verify rows facing drafts d_1 .. d_k, row k the bonus row.  Row j accepts d_{j+1} iff
// fl32(u_j * S_j) < w_j(d_{j+1}) with S_j the kept total of row j and w_j(d) the kept weight of d (0 when d is not
// kept; a tie rejects).  At the first rejection x* is drawn from row j without d (its weight skipped in the prefix
// sums) with the second uniform v; with every draft accepted x* is drawn from the bonus row with v.  Outputs are the
// split verify's decision buffers as write_verify_decision writes them (accept tile, accept index, next token, the
// alignment lanes [d_1 .. d_a*, x*, sentinel ...]) and a statistics row for the ledger.
// Named compile-time args: cb_stage, rows (k + 1), lanes (128), table_size.  Compile-time args: TensorAccessorArgs of
// rows, drafts, policy, uniforms, table, accept_tile, accept_index, next_token, alignment, stats.  Runtime args: the
// ten addresses in that order.

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
constexpr uint32_t DRAFTS = ROWS - 1;
constexpr uint32_t SHARD_LANES = 32;
constexpr uint32_t ROW_BYTES = 2 * LANES * 4;
constexpr uint32_t TILE_BYTES = 4096;
constexpr uint32_t GRAIN = 64;
constexpr uint32_t TOKEN_ROW_BYTES = 32 * 4;  // the fp32 [1,1,1,32] rows: drafts, uniforms, alignment
constexpr uint32_t STATS_LANES = 16;
constexpr uint32_t MAX_TOP_K = SHARD_LANES;
constexpr uint32_t STAGE_TILE = 0;
constexpr uint32_t STAGE_ROWS = TILE_BYTES;
constexpr uint32_t STAGE_DRAFTS = STAGE_ROWS + ROWS * ROW_BYTES;
constexpr uint32_t STAGE_POLICY = STAGE_DRAFTS + TOKEN_ROW_BYTES;
constexpr uint32_t STAGE_UNIFORMS = STAGE_POLICY + GRAIN;
constexpr uint32_t STAGE_ALIGN = STAGE_UNIFORMS + TOKEN_ROW_BYTES;
constexpr uint32_t STAGE_STATS = STAGE_ALIGN + TOKEN_ROW_BYTES;
constexpr uint32_t STAGE_INDEX = STAGE_STATS + STATS_LANES * 4;  // the accept index word (its own aligned grain)
constexpr uint32_t STAGE_NEXT = STAGE_INDEX + GRAIN;             // the next token word (its own aligned grain: a NoC
                                                                 // write's L1 source is read at grain alignment)
constexpr uint32_t STAGE_TABLE = STAGE_NEXT + GRAIN;
constexpr uint32_t STAGE_WORK = STAGE_TABLE + MAX_TOP_K * GRAIN;
constexpr uint32_t WORK_VALUES = STAGE_WORK / 4, WORK_IDS = WORK_VALUES + LANES, WORK_KEYS = WORK_IDS + LANES;
constexpr uint32_t WORK_SORTED_IDS = WORK_KEYS + LANES, WORK_SORTED_VALUES = WORK_SORTED_IDS + MAX_TOP_K;
constexpr uint32_t WORK_WEIGHTS = WORK_SORTED_VALUES + MAX_TOP_K, WORK_INCLUSIVE = WORK_WEIGHTS + MAX_TOP_K;
constexpr uint32_t WORK_CHUNK = WORK_INCLUSIVE + MAX_TOP_K, WORK_END = WORK_CHUNK + MAX_TOP_K;
constexpr uint32_t STAGE_BYTES = WORK_END * 4;
static_assert(ROWS >= 2 && ROWS <= 6, "a pass has 1..5 drafts and its bonus row");
static_assert(DRAFTS + 2 <= 32, "the uniforms row holds u_0 .. u_{k-1} and v");

// statistics lanes
constexpr uint32_t STAT_WEIGHT = 0;      // w_j(d_{j+1}) for j < 5, -1 where the row was not evaluated
constexpr uint32_t STAT_TOTAL = 5;       // S_j (the kept total) for j < 5, -1 where not evaluated
constexpr uint32_t STAT_GUARD = 10;      // bit j: row j's kept minimum did not clear the shard floor
constexpr uint32_t STAT_RESAMPLED = 11;  // 1 when x* came from a rejected row's residual
constexpr uint32_t STAT_ACCEPTED = 12;   // a*
constexpr uint32_t STAT_TOKEN = 13;      // x*
constexpr uint32_t STAT_THETA = 14;      // fl32(v * S) of the draw that produced x*
constexpr uint32_t STAT_KEPT = 15;       // the kept count of the row that produced x*

union Bits {
    float f;
    uint32_t u;
};

FORCE_INLINE uint32_t key_of(uint32_t bits) {
    bits = (bits & 0x7FFFFFFFu) ? bits : 0u;
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
    constexpr auto a_rows = TensorAccessorArgs<0>();
    constexpr auto a_drafts = TensorAccessorArgs<a_rows.next_compile_time_args_offset()>();
    constexpr auto a_policy = TensorAccessorArgs<a_drafts.next_compile_time_args_offset()>();
    constexpr auto a_uniforms = TensorAccessorArgs<a_policy.next_compile_time_args_offset()>();
    constexpr auto a_table = TensorAccessorArgs<a_uniforms.next_compile_time_args_offset()>();
    constexpr auto a_accept_tile = TensorAccessorArgs<a_table.next_compile_time_args_offset()>();
    constexpr auto a_accept_index = TensorAccessorArgs<a_accept_tile.next_compile_time_args_offset()>();
    constexpr auto a_next_token = TensorAccessorArgs<a_accept_index.next_compile_time_args_offset()>();
    constexpr auto a_alignment = TensorAccessorArgs<a_next_token.next_compile_time_args_offset()>();
    constexpr auto a_stats = TensorAccessorArgs<a_alignment.next_compile_time_args_offset()>();
    const auto rows = TensorAccessor(a_rows, get_arg_val<uint32_t>(0));
    const auto drafts = TensorAccessor(a_drafts, get_arg_val<uint32_t>(1));
    const auto policy = TensorAccessor(a_policy, get_arg_val<uint32_t>(2));
    const auto uniforms = TensorAccessor(a_uniforms, get_arg_val<uint32_t>(3));
    const auto table = TensorAccessor(a_table, get_arg_val<uint32_t>(4));
    const auto accept_tile = TensorAccessor(a_accept_tile, get_arg_val<uint32_t>(5));
    const auto accept_index = TensorAccessor(a_accept_index, get_arg_val<uint32_t>(6));
    const auto next_token = TensorAccessor(a_next_token, get_arg_val<uint32_t>(7));
    const auto alignment = TensorAccessor(a_alignment, get_arg_val<uint32_t>(8));
    const auto stats = TensorAccessor(a_stats, get_arg_val<uint32_t>(9));

    Noc noc;
    DataflowBuffer stage(CB_STAGE);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    {
        FUSED_ZONE("fz_ma_stage");
        for (uint32_t r = 0; r < ROWS; ++r) {
            noc.async_read(
                rows,
                stage,
                ROW_BYTES,
                {.page_id = r, .offset_bytes = 0},
                {.offset_bytes = STAGE_ROWS + r * ROW_BYTES});
        }
        noc.async_read(
            drafts, stage, TOKEN_ROW_BYTES, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_DRAFTS});
        noc.async_read(policy, stage, GRAIN, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_POLICY});
        noc.async_read(
            uniforms, stage, TOKEN_ROW_BYTES, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_UNIFORMS});
        noc.async_read_barrier();
    }

    volatile tt_l1_ptr uint32_t* policy_words = words + STAGE_POLICY / 4;
    const uint32_t top_k = static_cast<uint32_t>(as_float(policy_words[0]));
    const float top_p = as_float(policy_words[1]);
    const float min_weight = as_float(policy_words[2]);
    volatile tt_l1_ptr uint32_t* draft_words = words + STAGE_DRAFTS / 4;
    volatile tt_l1_ptr uint32_t* uniform_words = words + STAGE_UNIFORMS / 4;
    volatile tt_l1_ptr uint32_t* align_words = words + STAGE_ALIGN / 4;
    volatile tt_l1_ptr uint32_t* stat_words = words + STAGE_STATS / 4;
    volatile tt_l1_ptr uint32_t* index_words = words + STAGE_INDEX / 4;
    volatile tt_l1_ptr uint32_t* next_words = words + STAGE_NEXT / 4;
    volatile tt_l1_ptr uint32_t* tile = words + STAGE_TILE / 4;
    volatile tt_l1_ptr uint32_t* values = words + WORK_VALUES;
    volatile tt_l1_ptr uint32_t* ids = words + WORK_IDS;
    volatile tt_l1_ptr uint32_t* keys = words + WORK_KEYS;
    volatile tt_l1_ptr uint32_t* sorted_ids = words + WORK_SORTED_IDS;
    volatile tt_l1_ptr uint32_t* sorted_values = words + WORK_SORTED_VALUES;
    volatile tt_l1_ptr uint32_t* weights = words + WORK_WEIGHTS;
    volatile tt_l1_ptr uint32_t* inclusive = words + WORK_INCLUSIVE;
    volatile tt_l1_ptr uint32_t* chunk_offset = words + WORK_CHUNK;
    for (uint32_t i = 0; i < TILE_BYTES / 4; ++i) {
        tile[i] = 0;
    }
    for (uint32_t i = 0; i < STATS_LANES; ++i) {
        stat_words[i] = as_bits(i < STAT_GUARD ? -1.0f : 0.0f);
    }
    for (uint32_t i = 0; i < GRAIN / 4; ++i) {
        index_words[i] = 0;
        next_words[i] = 0;
    }
    const uint32_t sentinel = draft_words[DRAFTS];  // the draft lanes past k hold the zero-embedding sentinel
    for (uint32_t i = 0; i < 32; ++i) {
        align_words[i] = sentinel;
    }

    // The row's kept set: sorted_ids / sorted_values / weights / inclusive over its top_k lanes, kept and its
    // total; the guard bit when the kept minimum does not clear the shard floor (the largest shard minimum).
    uint32_t kept = 0, total_kept = 0, guard_mask = 0;
    auto build_row = [&](uint32_t r) {
        uint32_t floor_key = 0;
        {
            FUSED_ZONE("fz_ma_select");
            volatile tt_l1_ptr uint32_t* lanes = words + (STAGE_ROWS + r * ROW_BYTES) / 4;
            for (uint32_t d = 0; d < LANES / SHARD_LANES; ++d) {
                uint32_t shard_min_key = 0xFFFFFFFFu;
                for (uint32_t c = 0; c < SHARD_LANES; ++c) {
                    const uint32_t j = d * SHARD_LANES + c;
                    values[j] = lanes[d * 2 * SHARD_LANES + c];
                    ids[j] = static_cast<uint32_t>(as_float(lanes[d * 2 * SHARD_LANES + SHARD_LANES + c]));
                    keys[j] = key_of(values[j]);
                    if (keys[j] < shard_min_key) {
                        shard_min_key = keys[j];
                    }
                }
                if (shard_min_key > floor_key) {
                    floor_key = shard_min_key;
                }
            }
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
        }
        {
            FUSED_ZONE("fz_ma_prefix");
            const float s_max = as_float(sorted_values[0]);
            for (uint32_t k = 0; k < top_k; ++k) {
                const float below = as_float(sorted_values[k]) - s_max;
                const float scaled = below * -1024.0f;
                uint32_t index;
                if (scaled >= static_cast<float>(TABLE_SIZE - 1)) {
                    index = TABLE_SIZE - 1;
                } else if (scaled > 0.0f) {
                    index = static_cast<uint32_t>(scaled);
                } else {
                    index = 0;
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
            uint32_t running = 0;
            for (uint32_t k = 0; k < top_k; ++k) {
                weights[k] = static_cast<uint32_t>(as_float(words[(STAGE_TABLE + k * GRAIN + chunk_offset[k]) / 4]));
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
            kept = kept_top_p < kept_min_weight ? kept_top_p : kept_min_weight;
            if (kept < 1) {
                kept = 1;
            }
            total_kept = inclusive[kept - 1];
            if (key_of(sorted_values[kept - 1]) <= floor_key) {
                guard_mask |= 1u << r;
            }
        }
    };

    // The theta rule over the kept prefix sums, skipping lane `skip` (LANES = no skip): the lane whose prefix first
    // exceeds fl32(u * total), capped at the last kept lane.
    auto draw = [&](uint32_t u_bits, uint32_t skip, uint32_t total, uint32_t* theta_bits) -> uint32_t {
        FUSED_ZONE("fz_ma_draw");
        const float theta = as_float(u_bits) * static_cast<float>(total);  // RNE site 2
        *theta_bits = as_bits(theta);
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
    };

    uint32_t accepted = DRAFTS, token = 0, theta_bits = 0, resampled = 0, decided_kept = 0;
    {
        FUSED_ZONE("fz_ma_decide");
        for (uint32_t j = 0; j < DRAFTS; ++j) {
            build_row(j);
            const uint32_t draft = static_cast<uint32_t>(as_float(draft_words[j]));
            uint32_t lane_d = LANES, w_d = 0;
            for (uint32_t i = 0; i < kept; ++i) {
                if (sorted_ids[i] == draft) {
                    lane_d = i;
                    w_d = weights[i];
                    break;
                }
            }
            stat_words[STAT_WEIGHT + j] = as_bits(static_cast<float>(w_d));
            stat_words[STAT_TOTAL + j] = as_bits(static_cast<float>(total_kept));
            const float theta_j = as_float(uniform_words[j]) * static_cast<float>(total_kept);  // RNE
            if (theta_j < static_cast<float>(w_d)) {
                align_words[j] = draft_words[j];
                continue;
            }
            accepted = j;
            token = draw(uniform_words[DRAFTS], lane_d, total_kept - w_d, &theta_bits);
            resampled = 1;
            decided_kept = kept;
            break;
        }
    }
    if (accepted == DRAFTS) {
        FUSED_ZONE("fz_ma_bonus");
        build_row(DRAFTS);
        token = draw(uniform_words[DRAFTS], LANES, total_kept, &theta_bits);
        decided_kept = kept;
    }
    {
        FUSED_ZONE("fz_ma_write");
        align_words[accepted] = as_bits(static_cast<float>(token));
        tile[0] = as_bits(static_cast<float>(accepted));
        index_words[0] = accepted;
        next_words[0] = as_bits(static_cast<float>(token));
        stat_words[STAT_GUARD] = as_bits(static_cast<float>(guard_mask));
        stat_words[STAT_RESAMPLED] = as_bits(static_cast<float>(resampled));
        stat_words[STAT_ACCEPTED] = as_bits(static_cast<float>(accepted));
        stat_words[STAT_TOKEN] = as_bits(static_cast<float>(token));
        stat_words[STAT_THETA] = theta_bits;
        stat_words[STAT_KEPT] = as_bits(static_cast<float>(decided_kept));

        noc.async_write(
            stage, accept_tile, TILE_BYTES, {.offset_bytes = STAGE_TILE}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(stage, accept_index, 4, {.offset_bytes = STAGE_INDEX}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(stage, next_token, 4, {.offset_bytes = STAGE_NEXT}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(
            stage, alignment, TOKEN_ROW_BYTES, {.offset_bytes = STAGE_ALIGN}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(
            stage, stats, STATS_LANES * 4, {.offset_bytes = STAGE_STATS}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write_barrier();
    }
    stage.push_back(1);
}
