// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared expert eltwise program, writer: this core's intermediate tile into the down linear's input shard (page c)
// and, on the scalar core, the column-broadcast sigmoid tile.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t inter_addr = get_arg_val<uint32_t>(0);
    const uint32_t sig_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile = get_arg_val<uint32_t>(2);
    const uint32_t has_scalar = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_inter = get_named_compile_time_arg_val("cb_inter");
    constexpr uint32_t cb_sig_bcast = get_named_compile_time_arg_val("cb_sig_bcast");
    constexpr uint32_t tile_bytes = 2048;
    constexpr auto inter_args = TensorAccessorArgs<0, 0>();
    constexpr auto sig_args =
        TensorAccessorArgs<inter_args.next_compile_time_args_offset(), inter_args.next_common_runtime_args_offset()>();
    const auto inter = TensorAccessor(inter_args, inter_addr);
    const auto sig = TensorAccessor(sig_args, sig_addr);

    DataflowBuffer inter_tile(cb_inter);
    DataflowBuffer sig_tile(cb_sig_bcast);
    {
        FUSED_ZONE("fz_se_w_main");
        inter_tile.wait_front(1);
        noc_async_write(inter_tile.get_read_ptr(), inter.get_noc_addr(tile), tile_bytes);
        if (has_scalar) {
            sig_tile.wait_front(1);
            noc_async_write(sig_tile.get_read_ptr(), sig.get_noc_addr(0), tile_bytes);
        }
        noc_async_write_barrier();
        inter_tile.pop_front(1);
        if (has_scalar) {
            sig_tile.pop_front(1);
        }
    }
}
