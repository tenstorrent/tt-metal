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
    constexpr std::uint32_t base_l1_address = get_arg(args::base_l1_address);
    constexpr std::uint32_t intermediate_l1_addr = get_arg(args::intermediate_l1_addr);
    constexpr std::uint32_t size_bytes = get_arg(args::size_bytes);

    for (uint32_t id = 0; id < NUM_L1_BANKS; id += 1) {
        uint32_t bank_id;
        bank_id = umodsi3_const_divisor<NUM_L1_BANKS>(id);
        uint32_t l1_address = base_l1_address + bank_to_l1_offset[bank_id];
        uint32_t noc_xy = l1_bank_to_noc_xy[noc_index][bank_id];
        uint64_t noc_addr = get_noc_addr_helper(noc_xy, l1_address);

        noc_async_read(noc_addr, intermediate_l1_addr, size_bytes);
        noc_async_read_barrier();
#if defined(ARCH_QUASAR)
        // Quasar L1 is cacheable: drop any stale cached copy of the buffer the NOC just filled.
        invalidate_l1_dcache(intermediate_l1_addr);
        invalidate_l2_cache_range(intermediate_l1_addr, size_bytes);
#endif

        volatile tt_l1_ptr uint32_t* read_value_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(intermediate_l1_addr);
        uint32_t read_val = *read_value_ptr;
        *read_value_ptr = read_val + 1;
#if defined(ARCH_QUASAR)
        // Push the incremented value to L1 so the NOC write below sends it.
        flush_l2_cache_range(intermediate_l1_addr, size_bytes);
#endif

        noc_async_write(intermediate_l1_addr, noc_addr, size_bytes);
        noc_async_write_barrier();
    }
}
