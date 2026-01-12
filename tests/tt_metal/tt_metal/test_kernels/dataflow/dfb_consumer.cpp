// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"
#include "api/debug/dprint.h"

void kernel_main() {
    const uint32_t dst_addr_base = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    experimental::DataflowBuffer dfb(0);
    experimental::Noc noc;

    // uint32_t dst_addr_base = get_arg_val<uint32_t>(0);
    uint32_t entry_size = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(dst_args, dst_addr_base, entry_size);

    for (uint32_t tile_id = 0; tile_id < 16; tile_id++) {
        dfb.wait_front(1);
        noc.async_write(dfb, tensor_accessor, entry_size, {}, {.page_id = tile_id});
        dfb.pop_front(1);
    }
    noc.async_write_barrier();
}
