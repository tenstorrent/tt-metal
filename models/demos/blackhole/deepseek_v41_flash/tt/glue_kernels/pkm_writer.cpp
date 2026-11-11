// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// T19 packed-mHC mixes, writer: the mixes tile of row r -> page r.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, 4096);
    experimental::CB out(cb_out);
    out.wait_front(1);
    noc.async_write(out, o_acc, 4096, {.offset_bytes = 0}, {.page_id = r, .offset_bytes = 0});
    noc.async_writes_flushed();
    out.pop_front(1);
    noc.async_write_barrier();
}
