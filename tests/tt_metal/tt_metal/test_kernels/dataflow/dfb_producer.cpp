// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"
#include "api/debug/dprint.h"

void kernel_main() {
    const uint32_t src_addr_base = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    experimental::DataflowBuffer dfb(0);
    experimental::Noc noc;

    // uint32_t src_addr_base = get_arg_val<uint32_t>(0);
    uint32_t entry_size = dfb.get_entry_size();
    const auto tensor_accessor = TensorAccessor(src_args, src_addr_base, entry_size);

    for (uint32_t tile_id = 0; tile_id < 16; tile_id++) {
        dfb.reserve_back(1);
        noc.async_read(tensor_accessor, dfb, entry_size, {.page_id = tile_id}, {});
        noc.async_read_barrier();
        dfb.push_back(1);
    }
    dfb.finish();
}
