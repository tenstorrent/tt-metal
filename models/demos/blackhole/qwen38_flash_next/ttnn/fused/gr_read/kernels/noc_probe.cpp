// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Each core writes its own NoC coordinates (my_x[0], my_y[0]) as two uint32 words into page `slot` of a uint32
// TILE tensor; the host reads them back as the logical -> NoC map the multicast rectangle needs.
// Compile-time args: 0.. TensorAccessorArgs(output).  Runtime args: 0 output address, 1 slot.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t slot = get_arg_val<uint32_t>(1);
    constexpr auto args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_coordinates = 0;
    DataflowBuffer coordinates(cb_coordinates);
    coordinates.reserve_back(1);
    const CoreLocalMem<volatile uint32_t> words(coordinates.get_write_ptr());
    words[0] = my_x[0];
    words[1] = my_y[0];
    words[2] = my_x[1];
    words[3] = my_y[1];
    coordinates.push_back(1);
    const auto out = TensorAccessor(args, addr);
    const Noc noc;
    coordinates.wait_front(1);
    noc.async_write(coordinates, out, 16, {.offset_bytes = 0}, {.page_id = slot});
    noc.async_write_barrier();
    coordinates.pop_front(1);
}
