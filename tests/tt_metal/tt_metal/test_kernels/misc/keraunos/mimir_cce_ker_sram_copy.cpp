// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A Mimir CCE (DRISC) copies 32-bit words between its own SRAM and a Keraunos package address by plain load and
// store through the 64-bit package SPA. Local SRAM is accessed through the uncached alias, since the host writes it
// and reads it without going through this hart's cache, the worker -> router direction of the fabric. Nothing but the pointer
// differs between the two directions.
//
// Runtime args:
//   0, 1: Keraunos package address, low and high 32 bits
//   2:    number of 32-bit words
//   3:    local SRAM offset (this CCE's SRAM)
//   4:    0 = store local words to Keraunos, 1 = load Keraunos words into local SRAM
//   5:    status word, local SRAM offset: 0x1D3A0001 started, 0x1D3A0003 done

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"

void kernel_main() {
    const uint64_t addr = (static_cast<uint64_t>(get_arg_val<uint32_t>(1)) << 32) | get_arg_val<uint32_t>(0);
    const uint32_t words = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3) + MEM_L1_UNCACHED_BASE);
    const uint32_t load = get_arg_val<uint32_t>(4);
    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(5) + MEM_L1_UNCACHED_BASE);
    volatile uint32_t* remote = reinterpret_cast<volatile uint32_t*>(addr);

    status[0] = 0x1D3A0001;
    for (uint32_t i = 0; i < words; i++) {
        if (load) {
            local[i] = remote[i];
        } else {
            remote[i] = local[i];
        }
    }
    asm volatile("fence rw, rw" ::: "memory");
    status[4] = 0x1D3A0003;
}
