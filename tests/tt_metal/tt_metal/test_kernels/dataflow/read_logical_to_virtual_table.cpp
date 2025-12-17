// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t table_address = get_compile_time_arg_val(0);
    constexpr uint32_t num_logical_cols = get_compile_time_arg_val(1);
    constexpr uint32_t num_logical_rows = get_compile_time_arg_val(2);

    volatile tt_l1_ptr uint32_t* table_address_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_address);

    for (uint16_t i = 0; i < num_logical_cols; i++) {
        coord_t virtual_coord = get_virtual_coord_from_worker_logical_coord(i, 0);
        *table_address_ptr = virtual_coord.x;
        table_address_ptr++;
    }
    for (uint16_t i = 0; i < num_logical_rows; i++) {
        coord_t virtual_coord = get_virtual_coord_from_worker_logical_coord(0, i);
        *table_address_ptr = virtual_coord.y;
        table_address_ptr++;
    }
}
