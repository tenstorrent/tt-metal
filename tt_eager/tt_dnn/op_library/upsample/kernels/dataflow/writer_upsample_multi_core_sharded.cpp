// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

    uint32_t stick_nbytes = get_arg_val<uint32_t>(0);
    uint32_t in_nsticks_local = get_arg_val<uint32_t>(1);
    uint32_t scale_h = get_arg_val<uint32_t>(2);
    uint32_t scale_w = get_arg_val<uint32_t>(3);
    uint32_t in_w = get_arg_val<uint32_t>(4);
    uint32_t out_w = get_arg_val<uint32_t>(5);
    uint32_t start_in_stick_id = get_arg_val<uint32_t>(6);

    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(1);

    uint32_t l1_read_addr = get_read_ptr(in_cb_id);
    uint32_t l1_write_addr = get_write_ptr(out_cb_id);

    // cb_wait_front(in_cb_id, in_nsticks_local);

    uint32_t in_stick_row_id = start_in_stick_id / in_w;    // assuming shard begins with a new row. TODO: generalize?
    uint32_t l1_write_addr_stick = l1_write_addr;
    // for each input stick
    for (uint32_t i = start_in_stick_id; i < start_in_stick_id + in_nsticks_local; ++ i) {
        cb_reserve_back(out_cb_id, scale_h * scale_w);

        uint32_t l1_write_addr_local = l1_write_addr_stick;
        for (uint32_t j = 0; j < scale_h; ++j) {
            l1_write_addr_local = l1_write_addr_stick + j * out_w * stick_nbytes;
            // replicate stick scale_h times.
            for (size_t k = 0; k < scale_w; ++k) {
                // replicate stick scale_w times.
                uint64_t dst_noc_addr = get_noc_addr(l1_write_addr_local);
                noc_async_write(l1_read_addr, dst_noc_addr, stick_nbytes);
                l1_write_addr_local += stick_nbytes;
            }
        }
        // move to the next input stick
        l1_read_addr += stick_nbytes;
        // move to the next output stick
        l1_write_addr_stick += stick_nbytes * scale_w;

        // if this is the end of a row, skip the next (scale_h - 1) rows
        l1_write_addr_stick += (i == (in_w * (in_stick_row_id + 1) - 1)) * out_w * stick_nbytes * (scale_h - 1);
        in_stick_row_id += (i == (in_w * (in_stick_row_id + 1) - 1));

        noc_async_write_barrier();
        cb_push_back(out_cb_id, scale_h * scale_w);
    }

    // cb_pop_front(in_cb_id, in_nsticks_local);
}
