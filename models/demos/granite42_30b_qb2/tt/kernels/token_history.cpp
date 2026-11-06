// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t capacity = get_compile_time_arg_val(1);
    constexpr auto token_args = TensorAccessorArgs<2>();
    constexpr auto history_args = TensorAccessorArgs<token_args.next_compile_time_args_offset()>();
    constexpr auto cursor_args = TensorAccessorArgs<history_args.next_compile_time_args_offset()>();
    const auto tokens = TensorAccessor(token_args, get_arg_val<uint32_t>(0), batch * 4);
    const auto history = TensorAccessor(history_args, get_arg_val<uint32_t>(1), batch * 4);
    const auto cursor = TensorAccessor(cursor_args, get_arg_val<uint32_t>(2), 4);
    const uint32_t scratch = get_write_ptr(0);
    noc_async_read(get_noc_addr(0, tokens), scratch, batch * 4);
    noc_async_read(get_noc_addr(0, cursor), scratch + 128, 4);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* index = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 128);
    if (*index < capacity) {
        noc_async_write(scratch, get_noc_addr(*index, history), batch * 4);
        noc_async_write_barrier();
        *index += 1;
        noc_async_write(scratch + 128, get_noc_addr(0, cursor), 4);
        noc_async_write_barrier();
    }
}
