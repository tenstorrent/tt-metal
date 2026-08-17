// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writer for the per-tile triangle-solve LLK test. Reads tile 0 from cb_x and writes it to the
// output buffer. CT args: [<output TensorAccessorArgs...>]. Runtime args: [out_addr].

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_x = tt::CBIndex::c_2;
    const uint32_t tile_bytes = get_tile_size(cb_x);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out_gen = TensorAccessor(out_args, out_addr, tile_bytes);

    Noc noc;
    CircularBuffer cb_x_o(cb_x);

    cb_x_o.wait_front(1);
    noc.async_write(cb_x_o, out_gen, tile_bytes, {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write_barrier();
    cb_x_o.pop_front(1);
}
