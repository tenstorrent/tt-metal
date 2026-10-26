// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "tests/tt_metal/tt_metal/test_kernels/misc/watcher_test_helpers.h"

void kernel_main() {
    const uint32_t local_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_addr = get_arg_val<uint32_t>(1);
    const uint32_t dst_addr = get_arg_val<uint32_t>(2);
    const uint32_t dram_x = get_arg_val<uint32_t>(3);
    const uint32_t dram_y = get_arg_val<uint32_t>(4);
    const uint32_t read_id = get_arg_val<uint32_t>(5);
    const uint32_t write_id = get_arg_val<uint32_t>(6);
    const bool reverse = get_arg_val<uint32_t>(7);
    const uint32_t mode = get_arg_val<uint32_t>(8);  // 0: both, 1: read only, 2: write only
    const bool should_trip = mode == 0 && read_id == write_id;
    if (should_trip) {
        // The negative test never returns, including on a runtime without the new check.
        signal_completion_before_hang();
    }
    Noc reader(read_id), writer(write_id);
    CoreLocalMem<uint32_t> local(local_addr);
    UnicastEndpoint dram;
    auto read = [&]() {
        reader.async_read(dram, local, 32, {.noc_x = dram_x, .noc_y = dram_y, .addr = src_addr}, {});
        reader.async_read_barrier();
    };
    auto write = [&]() {
        writer.async_write(local, dram, 32, {}, {.noc_x = dram_x, .noc_y = dram_y, .addr = dst_addr});
        writer.async_write_barrier();
    };
    if (mode == 1) {
        read();
    } else if (mode == 2) {
        write();
    } else if (reverse) {
        write();
        read();
    } else {
        read();
        write();
    }
    if (should_trip) {
        while (true) {}
    }
}
