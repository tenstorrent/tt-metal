// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 decode merge writer: out [S,32,512] TILE, tile id s*16 + j0 + j.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t TPC = get_compile_time_arg_val(0);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2, 1>();
    constexpr uint32_t TILE_B = 2048;
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t s = get_arg_val<uint32_t>(0);
    const uint32_t j0 = get_arg_val<uint32_t>(1);
    const auto oa = TensorAccessor(o_args, o_addr, TILE_B);
    CircularBuffer o(CB_OUT);
    DeviceZoneScopedN("XW_ALL");
    o.wait_front(TPC);
    uint32_t p = o.get_read_ptr();
    for (uint32_t j = 0; j < TPC; ++j) {
        noc_async_write(p + j * TILE_B, oa.get_noc_addr(s * 16 + j0 + j), TILE_B);
    }
    noc_async_write_barrier();
    o.pop_front(TPC);
}
