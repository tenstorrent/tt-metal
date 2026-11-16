// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B17 fused attention back: writer, core i writes its tile to page i of out [1,1,1,H*NT*32] TILE.  CT: 0 cb_out,
// then TensorAccessorArgs(out). Common: out addr. RT: i
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    const uint32_t i = get_arg_val<uint32_t>(0);
    const auto oa = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), 2048);
    CircularBuffer o(cb_out);
    o.wait_front(1);
    noc_async_write(o.get_read_ptr(), oa.get_noc_addr(i), 2048);
    noc_async_write_barrier();
    o.pop_front(1);
}
