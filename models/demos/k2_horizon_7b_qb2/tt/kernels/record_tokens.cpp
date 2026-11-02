// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// All buffers are replicated uint32 ROW_MAJOR, with a 32-token/128-byte row.
// The history cursor belongs to the device; wrap permits indefinite low-level
// replay, while generation reads at most the supported context before reuse.
void kernel_main() {
    constexpr uint32_t capacity = get_compile_time_arg_val(0);
    constexpr auto token_args = TensorAccessorArgs<1>();
    constexpr auto index_args = TensorAccessorArgs<token_args.next_compile_time_args_offset()>();
    constexpr auto history_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();
    const auto tokens = TensorAccessor(token_args, get_arg_val<uint32_t>(0), 128);
    const auto index = TensorAccessor(index_args, get_arg_val<uint32_t>(1), 128);
    const auto history = TensorAccessor(history_args, get_arg_val<uint32_t>(2), 128);
    const uint32_t scratch = get_write_ptr(0);
    noc_async_read(tokens.get_noc_addr(0), scratch, 128);
    noc_async_read(index.get_noc_addr(0), scratch + 128, 128);
    noc_async_read_barrier();
    auto cursor = reinterpret_cast<volatile uint32_t*>(scratch + 128);
    const uint32_t row = cursor[0] % capacity;
    noc_async_write(scratch, history.get_noc_addr(row), 128);
    cursor[0] = (row + 1) % capacity;
    noc_async_write(scratch + 128, index.get_noc_addr(0), 128);
    noc_async_write_barrier();
}
