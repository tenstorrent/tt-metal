// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "hw/inc/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t table_address = get_compile_time_arg_val(0);
    constexpr uint32_t num_logical_cols = get_compile_time_arg_val(1);
    constexpr uint32_t num_logical_rows = get_compile_time_arg_val(2);

    experimental::CoreLocalMem<uint32_t> table(table_address);

    for (uint16_t i = 0; i < num_logical_cols; i++) {
        coord_t virtual_coord = get_virtual_coord_from_worker_logical_coord(i, 0);
        *table = virtual_coord.x;
        table++;
    }
    for (uint16_t i = 0; i < num_logical_rows; i++) {
        coord_t virtual_coord = get_virtual_coord_from_worker_logical_coord(0, i);
        *table = virtual_coord.y;
        table++;
    }
}
