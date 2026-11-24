// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#if defined(ARCH_QUASAR)
#include "risc_common.h"
#endif

void kernel_main() {
    constexpr uint32_t table_address = get_arg(args::table_address);
    constexpr uint32_t num_logical_cols = get_arg(args::num_logical_cols);
    constexpr uint32_t num_logical_rows = get_arg(args::num_logical_rows);

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
#if defined(ARCH_QUASAR)
    // Quasar L1 is cacheable: flush the table so the host sees it.
    flush_l2_cache_range(table_address, (num_logical_cols + num_logical_rows) * sizeof(uint32_t));
#endif
}
