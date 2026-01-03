// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

extern uint32_t rta_count;
extern uint32_t crta_count;

void kernel_main() {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    uint32_t l1_write_addr = get_write_ptr(cb_in0);
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(l1_write_addr);

    // Write RTA and CRTA count to L1 for validation
    ptr[0] = rta_count;
    ptr[1] = crta_count;

    // Write RTA and CRTA payload to L1 for validation
    for (size_t i = 0; i < rta_count; i++) {
        // + 2 as first two are counts
        ptr[i + 2] = get_arg_val<uint32_t>(i);
    }

    for (size_t i = 0; i < crta_count; i++) {
        // rta_count + 2 as first two are counts + rta args
        ptr[i + rta_count + 2] = get_common_arg_val<uint32_t>(i);
    }

#ifdef MAX_RTA_IDX
    // Access RTA: this should have a watcher assert when MAX_RTA_IDX >= rta_count
    uint32_t rta = get_arg_val<uint32_t>(MAX_RTA_IDX);
#endif

#ifdef MAX_CRTA_IDX
    // Access CRTA: this should have a watcher assert when MAX_CRTA_IDX >= crta_count
    uint32_t crta = get_common_arg_val<uint32_t>(MAX_CRTA_IDX);
#endif
}
