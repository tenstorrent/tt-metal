// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Build baseline dense [S,E*C] combine rows from compact, already-scaled
// columns/weights.  Token rows are distributed across cores; no cumulative
// routing state is needed here.

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t sequence_length = get_compile_time_arg_val(0);
    constexpr uint32_t num_experts = get_compile_time_arg_val(1);
    constexpr uint32_t top_k = get_compile_time_arg_val(2);
    constexpr uint32_t column_read_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t weight_read_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t output_write_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t column_cb_page = get_compile_time_arg_val(6);
    constexpr uint32_t weight_cb_page = get_compile_time_arg_val(7);

    constexpr auto column_args = TensorAccessorArgs<8>();
    constexpr auto weight_args = TensorAccessorArgs<column_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();

    const uint32_t column_addr = get_arg_val<uint32_t>(0);
    const uint32_t weight_addr = get_arg_val<uint32_t>(1);
    const uint32_t output_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_token = get_arg_val<uint32_t>(3);
    const uint32_t end_token = get_arg_val<uint32_t>(4);

    Noc noc;
    CircularBuffer cb_column(tt::CBIndex::c_0);
    CircularBuffer cb_weight(tt::CBIndex::c_1);
    CircularBuffer cb_output(tt::CBIndex::c_2);
    const auto s_column = TensorAccessor(column_args, column_addr);
    const auto s_weight = TensorAccessor(weight_args, weight_addr);
    const auto s_output = TensorAccessor(output_args, output_addr);

    const uint32_t column_offset = (column_addr - cb_column.get_write_ptr()) & 0x3Fu;
    const uint32_t weight_offset = (weight_addr - cb_weight.get_write_ptr()) & 0x3Fu;
    for (uint32_t k = 0; k < top_k; ++k) {
        cb_column.reserve_back(1);
        cb_weight.reserve_back(1);
        noc.async_read(s_column, cb_column, column_read_bytes, {.page_id = k}, {.offset_bytes = column_offset});
        noc.async_read(s_weight, cb_weight, weight_read_bytes, {.page_id = k}, {.offset_bytes = weight_offset});
        noc.async_read_barrier();
        cb_column.push_back(1);
        cb_weight.push_back(1);
    }
    cb_column.wait_front(top_k);
    cb_weight.wait_front(top_k);
    const uint32_t column_base = cb_column.get_read_ptr() + column_offset;
    const uint32_t weight_base = cb_weight.get_read_ptr() + weight_offset;

    for (uint32_t token = start_token; token < end_token; ++token) {
        const uint32_t output_offset = (output_addr - cb_output.get_write_ptr()) & 0x3Fu;
        cb_output.reserve_back(1);
        volatile tt_l1_ptr uint16_t* row =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_output.get_write_ptr() + output_offset);
        for (uint32_t column = 0; column < num_experts * sequence_length; ++column) {
            row[column] = 0;
        }
        for (uint32_t k = 0; k < top_k; ++k) {
            const uint32_t* columns = reinterpret_cast<const uint32_t*>(column_base + k * column_cb_page);
            const uint16_t* weights = reinterpret_cast<const uint16_t*>(weight_base + k * weight_cb_page);
            row[columns[token]] = weights[token];
        }
        cb_output.push_back(1);
        cb_output.wait_front(1);
        noc.async_write(cb_output, s_output, output_write_bytes, {.offset_bytes = output_offset}, {.page_id = token});
        noc.async_write_barrier();
        cb_output.pop_front(1);
    }

    cb_column.pop_front(top_k);
    cb_weight.pop_front(top_k);
}
