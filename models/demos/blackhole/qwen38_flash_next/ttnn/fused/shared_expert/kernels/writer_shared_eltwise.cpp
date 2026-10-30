// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared expert eltwise program, writer: this core's intermediate tile into the down linear's input shard (page c)
// and, on the scalar core, the column-broadcast sigmoid tile.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t intermediate_addr = get_arg_val<uint32_t>(0);
    const uint32_t sigmoid_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile = get_arg_val<uint32_t>(2);
    const uint32_t has_scalar = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_inter = get_named_compile_time_arg_val("cb_inter");
    constexpr uint32_t cb_sig_bcast = get_named_compile_time_arg_val("cb_sig_bcast");
    constexpr uint32_t tile_bytes = 2048;
    constexpr auto intermediate_args = TensorAccessorArgs<0, 0>();
    constexpr auto sigmoid_args = TensorAccessorArgs<
        intermediate_args.next_compile_time_args_offset(),
        intermediate_args.next_common_runtime_args_offset()>();
    const auto intermediate = TensorAccessor(intermediate_args, intermediate_addr);
    const auto sigmoid = TensorAccessor(sigmoid_args, sigmoid_addr);

    const Noc noc;
    DataflowBuffer intermediate_tile(cb_inter);
    DataflowBuffer sigmoid_tile(cb_sig_bcast);
    {
        intermediate_tile.wait_front(1);
        noc.async_write(intermediate_tile, intermediate, tile_bytes, {}, {.page_id = tile});
        if (has_scalar) {
            sigmoid_tile.wait_front(1);
            noc.async_write(sigmoid_tile, sigmoid, tile_bytes, {}, {.page_id = 0});
        }
        noc.async_write_barrier();
        intermediate_tile.pop_front(1);
        if (has_scalar) {
            sigmoid_tile.pop_front(1);
        }
    }
}
