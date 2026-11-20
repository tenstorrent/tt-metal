// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reports what experimental/keraunos_cce.h says about the core the kernel runs on.
//
// Runtime arg 0: status block, CCE-relative SRAM offset. Words written (uint32, uncached alias):
//   0 my_eth_cce()   1 my_hsio()   2 my_hart()   3 num_hsio   4 done   5 num_mimirs
//   6, 7  hsio_sram_addr(my_hsio(), my_eth_cce(), 0), low and high
//   8, 9  mimir_gddr_addr(1, 0)    10, 11 mimir_cce_sram_addr(1, 0)
//   12    own_sram_uncached_addr(0) low   13 peer_sram_uncached_addr(0) low

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/keraunos_cce.h"

void kernel_main() {
    using namespace experimental::keraunos;
    volatile uint32_t* status =
        reinterpret_cast<volatile uint32_t*>(own_sram_uncached_addr(get_arg_val<uint32_t>(0)));
    status[0] = my_eth_cce();
    status[1] = my_hsio();
    status[2] = my_hart();
    status[3] = spa_routes()->num_hsio;
    status[5] = spa_routes()->num_mimirs;
    const uint64_t own = hsio_sram_addr(my_hsio(), my_eth_cce(), 0);
    status[6] = static_cast<uint32_t>(own);
    status[7] = static_cast<uint32_t>(own >> 32);
    const uint64_t gddr = mimir_gddr_addr(1, 0);
    status[8] = static_cast<uint32_t>(gddr);
    status[9] = static_cast<uint32_t>(gddr >> 32);
    const uint64_t cce_sram = mimir_cce_sram_addr(1, 0);
    status[10] = static_cast<uint32_t>(cce_sram);
    status[11] = static_cast<uint32_t>(cce_sram >> 32);
    status[12] = static_cast<uint32_t>(own_sram_uncached_addr(0));
    status[13] = static_cast<uint32_t>(peer_sram_uncached_addr(0));
    asm volatile("fence rw, rw" ::: "memory");
    status[4] = 0x1D3A0003;
}
