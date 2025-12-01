// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"
#include "dataflow_api.h"

void kernel_main() {
    set_l1_data_cache<true>();
    uint32_t poll_addr = get_arg_val<uint32_t>(0);
    uint32_t value_to_write = get_arg_val<uint32_t>(1);

    experimental::Semaphore sem(get_arg_val<uint32_t>(2));
    experimental::CoreLocalMem<uint32_t> poll_value(poll_addr);

    sem.wait(1);
    poll_value[0] = value_to_write;
    set_l1_data_cache<false>();
}
