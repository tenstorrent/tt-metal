// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// otherlayers: writer of the indexer-q RoPE program: one output tile per core to page t of the [R,1,32,128] bf16 TILE
// tensor (DRAM or L1 interleaved).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ol_zones.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    const uint32_t t = ol_core_idx(get_common_arg_val<uint32_t>(1));
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    OLZ("Z_WALL");
    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, 2048);
    experimental::CB out(cb_out);
    out.wait_front(1);
    noc.async_write(out, o_acc, 2048, {.offset_bytes = 0}, {.page_id = t, .offset_bytes = 0});
    noc.async_write_barrier();
    out.pop_front(1);
}
