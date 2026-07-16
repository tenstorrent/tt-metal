// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Trace-safe compact routing metadata for DiffusionGemma denoise MoE.
//
// The denoise router supplies exactly K assignments for each of S tokens.  This
// kernel groups those assignments into fixed 32-row expert-homogeneous segments
// without a host readback.  MAX_SEGMENTS is a compile-time upper bound; unused
// segments are mapped to expert 0 and are never selected by token_slot.
//
// Inputs (ROW_MAJOR DRAM):
//   indices [1,1,S,K] uint32
//   values  [1,1,S,K] bf16       (normalised router weights)
//   scale   [1,1,1,E] bf16       (per-expert scale)
//
// Outputs (ROW_MAJOR DRAM):
//   slot_token [1, MAX_SEGMENTS*32] uint32
//   token_slot [K, S] uint32
//   route_weight [K, S] bf16      (expert-id sorted, scale applied)
//   sparsity [1,1,MAX_SEGMENTS,E] bf16, exactly one non-zero per segment

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {

inline float bf16_to_f32(uint16_t bits) {
    union {
        uint32_t u;
        float f;
    } value;
    value.u = static_cast<uint32_t>(bits) << 16;
    return value.f;
}

inline uint16_t f32_to_bf16(float value) {
    union {
        uint32_t u;
        float f;
    } bits;
    bits.f = value;
    const uint32_t rounding_bias = 0x7FFFu + ((bits.u >> 16) & 1u);
    return static_cast<uint16_t>((bits.u + rounding_bias) >> 16);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t sequence_length = get_compile_time_arg_val(0);
    constexpr uint32_t num_experts = get_compile_time_arg_val(1);
    constexpr uint32_t top_k = get_compile_time_arg_val(2);
    constexpr uint32_t max_segments = get_compile_time_arg_val(3);
    constexpr uint32_t segment_rows = get_compile_time_arg_val(4);
    constexpr uint32_t max_rows = max_segments * segment_rows;

    constexpr uint32_t indices_read_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t values_read_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t scale_read_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t slot_token_write_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t token_slot_page_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t route_weight_page_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t sparsity_page_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t indices_cb_page_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t values_cb_page_bytes = get_compile_time_arg_val(13);
    constexpr uint32_t slot_token_cb_page_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t token_slot_cb_page_bytes = get_compile_time_arg_val(15);
    constexpr uint32_t route_weight_cb_page_bytes = get_compile_time_arg_val(16);
    constexpr uint32_t sparsity_cb_page_bytes = get_compile_time_arg_val(17);
    constexpr uint32_t dense_map_write_bytes = get_compile_time_arg_val(18);
    constexpr uint32_t dense_map_cb_page_bytes = get_compile_time_arg_val(19);

    constexpr auto indices_args = TensorAccessorArgs<20>();
    constexpr auto values_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto scale_args = TensorAccessorArgs<values_args.next_compile_time_args_offset()>();
    constexpr auto slot_token_args = TensorAccessorArgs<scale_args.next_compile_time_args_offset()>();
    constexpr auto token_slot_args = TensorAccessorArgs<slot_token_args.next_compile_time_args_offset()>();
    constexpr auto route_weight_args = TensorAccessorArgs<token_slot_args.next_compile_time_args_offset()>();
    constexpr auto sparsity_args = TensorAccessorArgs<route_weight_args.next_compile_time_args_offset()>();
    constexpr auto dense_map_args = TensorAccessorArgs<sparsity_args.next_compile_time_args_offset()>();
    constexpr auto dense_column_args = TensorAccessorArgs<dense_map_args.next_compile_time_args_offset()>();

    const uint32_t indices_addr = get_arg_val<uint32_t>(0);
    const uint32_t values_addr = get_arg_val<uint32_t>(1);
    const uint32_t scale_addr = get_arg_val<uint32_t>(2);
    const uint32_t slot_token_addr = get_arg_val<uint32_t>(3);
    const uint32_t token_slot_addr = get_arg_val<uint32_t>(4);
    const uint32_t route_weight_addr = get_arg_val<uint32_t>(5);
    const uint32_t sparsity_addr = get_arg_val<uint32_t>(6);
    const uint32_t dense_map_addr = get_arg_val<uint32_t>(7);
    const uint32_t dense_column_addr = get_arg_val<uint32_t>(8);

    Noc noc;
    CircularBuffer cb_indices(tt::CBIndex::c_0);
    CircularBuffer cb_values(tt::CBIndex::c_1);
    CircularBuffer cb_scale(tt::CBIndex::c_2);
    CircularBuffer cb_slot_token(tt::CBIndex::c_3);
    CircularBuffer cb_token_slot(tt::CBIndex::c_4);
    CircularBuffer cb_route_weight(tt::CBIndex::c_5);
    CircularBuffer cb_sparsity(tt::CBIndex::c_6);
    CircularBuffer cb_dense_map(tt::CBIndex::c_7);
    CircularBuffer cb_dense_column(tt::CBIndex::c_8);

    const auto s_indices = TensorAccessor(indices_args, indices_addr);
    const auto s_values = TensorAccessor(values_args, values_addr);
    const auto s_scale = TensorAccessor(scale_args, scale_addr);
    const auto s_slot_token = TensorAccessor(slot_token_args, slot_token_addr);
    const auto s_token_slot = TensorAccessor(token_slot_args, token_slot_addr);
    const auto s_route_weight = TensorAccessor(route_weight_args, route_weight_addr);
    const auto s_sparsity = TensorAccessor(sparsity_args, sparsity_addr);
    const auto s_dense_map = TensorAccessor(dense_map_args, dense_map_addr);
    const auto s_dense_column = TensorAccessor(dense_column_args, dense_column_addr);

    // Blackhole DRAM NoC transactions require the low six address bits of
    // source and destination to match.  Each CB page carries 63 bytes of
    // alignment slack; compute the matching L1 offset from the runtime buffer
    // address instead of relying on allocator placement.
    const uint32_t indices_l1_offset = (indices_addr - cb_indices.get_write_ptr()) & 0x3Fu;
    const uint32_t values_l1_offset = (values_addr - cb_values.get_write_ptr()) & 0x3Fu;
    const uint32_t scale_l1_offset = (scale_addr - cb_scale.get_write_ptr()) & 0x3Fu;

    // Keep all compact router metadata in L1.  Reading it once avoids a second
    // top-k and lets the two packing passes run without further NoC traffic.
    for (uint32_t token = 0; token < sequence_length; ++token) {
        cb_indices.reserve_back(1);
        cb_values.reserve_back(1);
        noc.async_read(
            s_indices, cb_indices, indices_read_bytes, {.page_id = token}, {.offset_bytes = indices_l1_offset});
        noc.async_read(s_values, cb_values, values_read_bytes, {.page_id = token}, {.offset_bytes = values_l1_offset});
        noc.async_read_barrier();
        cb_indices.push_back(1);
        cb_values.push_back(1);
    }
    cb_indices.wait_front(sequence_length);
    cb_values.wait_front(sequence_length);

    cb_scale.reserve_back(1);
    noc.async_read(s_scale, cb_scale, scale_read_bytes, {.page_id = 0}, {.offset_bytes = scale_l1_offset});
    noc.async_read_barrier();
    cb_scale.push_back(1);
    cb_scale.wait_front(1);

    const uint32_t indices_base = cb_indices.get_read_ptr() + indices_l1_offset;
    const uint32_t values_base = cb_values.get_read_ptr() + values_l1_offset;
    const uint16_t* scale = reinterpret_cast<const uint16_t*>(cb_scale.get_read_ptr() + scale_l1_offset);

    uint16_t expert_count[num_experts];
    uint16_t expert_seen[num_experts];
    uint16_t expert_overflow_base[num_experts];
    uint16_t segment_expert[max_segments];
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        expert_count[expert] = 0;
        expert_seen[expert] = 0;
        expert_overflow_base[expert] = 0;
    }
    for (uint32_t segment = 0; segment < max_segments; ++segment) {
        segment_expert[segment] = 0;
    }
    for (uint32_t expert = 0; expert < num_experts && expert < max_segments; ++expert) {
        segment_expert[expert] = expert;
    }

    for (uint32_t token = 0; token < sequence_length; ++token) {
        const uint32_t* indices = reinterpret_cast<const uint32_t*>(indices_base + token * indices_cb_page_bytes);
        for (uint32_t k = 0; k < top_k; ++k) {
            const uint32_t expert = indices[k];
            if (expert < num_experts) {
                ++expert_count[expert];
            }
        }
    }

    // The first E segments are a fixed one-segment-per-expert primary bank.
    // That bank uses the existing roofline-tuned C=32 batched matmul.  Only
    // assignments beyond the first 32/expert enter the sparse overflow bank.
    uint32_t total_segments = num_experts;
    uint32_t overflow_segments = 0;
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        expert_overflow_base[expert] = overflow_segments;
        const uint32_t count = static_cast<uint32_t>(expert_count[expert]);
        const uint32_t overflow = count > segment_rows ? count - segment_rows : 0;
        const uint32_t segments = (overflow + segment_rows - 1) / segment_rows;
        for (uint32_t segment = 0; segment < segments; ++segment) {
            if (total_segments >= max_segments) {
                return;
            }
            segment_expert[total_segments++] = expert;
            ++overflow_segments;
        }
    }
    // For unique top-k assignments, MAX_SEGMENTS is constructed from the
    // mathematical upper bound E + floor((S*K-E)/32), so this cannot overflow.
    if (total_segments > max_segments) {
        return;
    }

    cb_slot_token.reserve_back(max_segments);
    cb_token_slot.reserve_back(top_k);
    cb_route_weight.reserve_back(top_k);
    cb_dense_column.reserve_back(top_k);
    const uint32_t slot_token_l1_offset = (slot_token_addr - cb_slot_token.get_write_ptr()) & 0x3Fu;
    const uint32_t token_slot_l1_offset = (token_slot_addr - cb_token_slot.get_write_ptr()) & 0x3Fu;
    const uint32_t route_weight_l1_offset = (route_weight_addr - cb_route_weight.get_write_ptr()) & 0x3Fu;
    const uint32_t dense_column_l1_offset = (dense_column_addr - cb_dense_column.get_write_ptr()) & 0x3Fu;
    const uint32_t slot_token_base = cb_slot_token.get_write_ptr() + slot_token_l1_offset;
    const uint32_t token_slot_base = cb_token_slot.get_write_ptr() + token_slot_l1_offset;
    const uint32_t route_weight_base = cb_route_weight.get_write_ptr() + route_weight_l1_offset;
    const uint32_t dense_column_base = cb_dense_column.get_write_ptr() + dense_column_l1_offset;

    for (uint32_t segment = 0; segment < max_segments; ++segment) {
        volatile tt_l1_ptr uint32_t* slot_page =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_token_base + segment * slot_token_cb_page_bytes);
        for (uint32_t row = 0; row < segment_rows; ++row) {
            slot_page[row] = 0;
        }
    }

    for (uint32_t token = 0; token < sequence_length; ++token) {
        const uint32_t* indices = reinterpret_cast<const uint32_t*>(indices_base + token * indices_cb_page_bytes);
        const uint16_t* values = reinterpret_cast<const uint16_t*>(values_base + token * values_cb_page_bytes);
        uint32_t sorted_expert[top_k];
        uint16_t sorted_value[top_k];
        for (uint32_t k = 0; k < top_k; ++k) {
            sorted_expert[k] = indices[k];
            sorted_value[k] = values[k];
        }
        // The dense combine reduces in expert-id order.  Preserve that order so
        // the fixed ragged path has the same BF16 accumulation sequence.
        for (uint32_t i = 1; i < top_k; ++i) {
            const uint32_t expert = sorted_expert[i];
            const uint16_t value = sorted_value[i];
            uint32_t j = i;
            while (j > 0 && sorted_expert[j - 1] > expert) {
                sorted_expert[j] = sorted_expert[j - 1];
                sorted_value[j] = sorted_value[j - 1];
                --j;
            }
            sorted_expert[j] = expert;
            sorted_value[j] = value;
        }

        for (uint32_t k = 0; k < top_k; ++k) {
            const uint32_t expert = sorted_expert[k];
            const uint32_t rank = expert_seen[expert]++;
            const uint32_t packed_segment = rank < segment_rows
                                                ? expert
                                                : num_experts + static_cast<uint32_t>(expert_overflow_base[expert]) +
                                                      (rank - segment_rows) / segment_rows;
            const uint32_t packed_row = packed_segment * segment_rows + rank % segment_rows;
            const uint32_t row_in_segment = packed_row % segment_rows;
            volatile tt_l1_ptr uint32_t* slot_page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                slot_token_base + packed_segment * slot_token_cb_page_bytes);
            volatile tt_l1_ptr uint32_t* token_slot_page =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(token_slot_base + k * token_slot_cb_page_bytes);
            volatile tt_l1_ptr uint16_t* route_weight_page =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(route_weight_base + k * route_weight_cb_page_bytes);
            volatile tt_l1_ptr uint32_t* dense_column_page =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dense_column_base + k * token_slot_cb_page_bytes);
            slot_page[row_in_segment] = token;
            token_slot_page[token] = packed_row;
            dense_column_page[token] = expert * sequence_length + rank;
            route_weight_page[token] = f32_to_bf16(bf16_to_f32(sorted_value[k]) * bf16_to_f32(scale[expert]));
        }
    }

    cb_slot_token.push_back(max_segments);
    cb_token_slot.push_back(top_k);
    cb_route_weight.push_back(top_k);
    cb_dense_column.push_back(top_k);
    cb_slot_token.wait_front(max_segments);
    cb_token_slot.wait_front(top_k);
    cb_route_weight.wait_front(top_k);
    cb_dense_column.wait_front(top_k);

    for (uint32_t segment = 0; segment < max_segments; ++segment) {
        noc.async_write(
            cb_slot_token,
            s_slot_token,
            slot_token_write_bytes,
            {.offset_bytes = slot_token_l1_offset},
            {.page_id = segment});
        noc.async_write_barrier();
        cb_slot_token.pop_front(1);
    }
    for (uint32_t k = 0; k < top_k; ++k) {
        noc.async_write(
            cb_token_slot, s_token_slot, token_slot_page_bytes, {.offset_bytes = token_slot_l1_offset}, {.page_id = k});
        noc.async_write(
            cb_route_weight,
            s_route_weight,
            route_weight_page_bytes,
            {.offset_bytes = route_weight_l1_offset},
            {.page_id = k});
        noc.async_write(
            cb_dense_column,
            s_dense_column,
            token_slot_page_bytes,
            {.offset_bytes = dense_column_l1_offset},
            {.page_id = k});
        noc.async_write_barrier();
        cb_token_slot.pop_front(1);
        cb_route_weight.pop_front(1);
        cb_dense_column.pop_front(1);
    }
    noc.async_write_barrier();

    constexpr uint16_t bf16_one = 0x3F80;
    for (uint32_t segment = 0; segment < max_segments; ++segment) {
        cb_sparsity.reserve_back(1);
        const uint32_t sparsity_l1_offset = (sparsity_addr - cb_sparsity.get_write_ptr()) & 0x3Fu;
        volatile tt_l1_ptr uint16_t* row =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_sparsity.get_write_ptr() + sparsity_l1_offset);
        for (uint32_t expert = 0; expert < num_experts; ++expert) {
            row[expert] = 0;
        }
        row[segment_expert[segment]] = bf16_one;
        cb_sparsity.push_back(1);
        cb_sparsity.wait_front(1);
        noc.async_write(
            cb_sparsity, s_sparsity, sparsity_page_bytes, {.offset_bytes = sparsity_l1_offset}, {.page_id = segment});
        noc.async_write_barrier();
        cb_sparsity.pop_front(1);
    }

    constexpr uint32_t dense_tiles_per_expert = sequence_length / 32;
    constexpr uint32_t segment_tile_rows = segment_rows / 32;
    constexpr uint32_t dense_tile_rows = num_experts * dense_tiles_per_expert;
    cb_dense_map.reserve_back(1);
    const uint32_t dense_map_l1_offset = (dense_map_addr - cb_dense_map.get_write_ptr()) & 0x3Fu;
    volatile tt_l1_ptr uint32_t* dense_map =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_dense_map.get_write_ptr() + dense_map_l1_offset);
    for (uint32_t row = 0; row < dense_tile_rows; ++row) {
        dense_map[row] = 0xFFFFFFFFu;
    }
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        for (uint32_t local = 0; local < segment_tile_rows; ++local) {
            dense_map[expert * dense_tiles_per_expert + local] = expert * segment_tile_rows + local;
        }
        const uint32_t count = static_cast<uint32_t>(expert_count[expert]);
        const uint32_t overflow = count > segment_rows ? count - segment_rows : 0;
        const uint32_t overflow_count = (overflow + segment_rows - 1) / segment_rows;
        for (uint32_t overflow_segment = 0; overflow_segment < overflow_count; ++overflow_segment) {
            const uint32_t packed_segment =
                num_experts + static_cast<uint32_t>(expert_overflow_base[expert]) + overflow_segment;
            for (uint32_t local = 0; local < segment_tile_rows; ++local) {
                const uint32_t dense_row =
                    expert * dense_tiles_per_expert + (overflow_segment + 1) * segment_tile_rows + local;
                if (dense_row < (expert + 1) * dense_tiles_per_expert) {
                    dense_map[dense_row] = packed_segment * segment_tile_rows + local;
                }
            }
        }
    }
    cb_dense_map.push_back(1);
    cb_dense_map.wait_front(1);
    noc.async_write(
        cb_dense_map, s_dense_map, dense_map_write_bytes, {.offset_bytes = dense_map_l1_offset}, {.page_id = 0});
    noc.async_write_barrier();
    cb_dense_map.pop_front(1);

    cb_indices.pop_front(sequence_length);
    cb_values.pop_front(sequence_length);
    cb_scale.pop_front(1);
}
