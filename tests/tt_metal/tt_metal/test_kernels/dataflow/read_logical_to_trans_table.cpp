// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t table_address = get_compile_time_arg_val(0);
    constexpr uint32_t num_logical_cols = get_compile_time_arg_val(1);
    constexpr uint32_t num_logical_rows = get_compile_time_arg_val(2);

    volatile tt_l1_ptr uint32_t* table_address_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_address);

    for (uint16_t i = 0; i < num_logical_cols; i++) {
        *table_address_ptr = logical_col_to_translated_col[i];
        table_address_ptr += 1;
    }
    for (uint16_t i = 0; i < num_logical_rows; i++) {
        *table_address_ptr = logical_row_to_translated_row[i];
        table_address_ptr += 1;
    }
}
