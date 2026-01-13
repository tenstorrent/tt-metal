// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/endpoints.h"
#include "experimental/tensor.h"

void kernel_main() {
    // Constexpr
    constexpr uint32_t num_dram_channels = 8;
    constexpr uint32_t log_base_2_of_num_dram_channels = 3;
    constexpr uint32_t cb_id_out0 = 16;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t stick_size = get_arg_val<uint32_t>(2);

    // TODO(agrebenisan): This isn't good... here we are assuming
    // that the stick size dictates tiles c, but stick size
    // doesn't necessarily need to be divisible by tiles c...
    // this is only the case really for tilize
    const uint32_t num_tiles_c = stick_size / 64;  // Assuming 2 bytes per datum, there are 64 bytes per tile row
    uint32_t stick_id = 0;

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(dst_args, dst_addr, stick_size);

    experimental::CircularBuffer cb(cb_id_out0);
    experimental::Noc noc;

    for (uint32_t i = 0; i < num_sticks / 32; i++) {
        // We reserve back an entire tile row and issue a bunch of reads
        cb.wait_front(num_tiles_c);
        uint32_t cb_read_offset = 0;
        for (uint32_t j = 0; j < 32; j++) {
            noc.async_write(cb, s, stick_size, {.offset_bytes = cb_read_offset}, {.page_id = stick_id});
            cb_read_offset += stick_size;
            stick_id++;
        }

        noc.async_write_barrier();
        cb.pop_front(num_tiles_c);
    }
}
