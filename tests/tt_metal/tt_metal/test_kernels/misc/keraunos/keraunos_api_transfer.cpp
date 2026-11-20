// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One transfer between this CCE's SRAM and a remote target, with the remote address computed by
// experimental/keraunos_cce.h from the host-written route table. The engine is IDMA (async_write/async_read and a
// barrier) or the CPU (write32/read32 per word).
//
// Runtime args:
//   0: target kind: 0 = HSIO SRAM, 1 = Mimir CCE SRAM, 2 = Mimir GDDR
//   1: target index: HSIO tile, global Mimir CCE, or Mimir
//   2: HSIO CCE (kind 0 only)
//   3: remote offset, low 32 bits   4: remote offset, high 32 bits (the CCE-relative offset, or GDDR offset)
//   5: bytes
//   6: local offset in this CCE's SRAM (CCE-relative)
//   7: direction: 0 = local -> remote, 1 = remote -> local
//   8: status block, CCE-relative SRAM offset
//   9: engine: 0 = IDMA, 1 = CPU
// Status (uncached alias): 0 started, 4 done.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/keraunos_cce.h"

void kernel_main() {
    using namespace experimental::keraunos;
    const uint32_t kind = get_arg_val<uint32_t>(0);
    const uint32_t index = get_arg_val<uint32_t>(1);
    const uint32_t cce = get_arg_val<uint32_t>(2);
    const uint64_t offset = (static_cast<uint64_t>(get_arg_val<uint32_t>(4)) << 32) | get_arg_val<uint32_t>(3);
    const uint32_t bytes = get_arg_val<uint32_t>(5);
    const uint32_t local_offset = get_arg_val<uint32_t>(6);
    const bool read = get_arg_val<uint32_t>(7) != 0;
    volatile uint32_t* status =
        reinterpret_cast<volatile uint32_t*>(own_sram_uncached_addr(get_arg_val<uint32_t>(8)));
    const bool cpu = get_arg_val<uint32_t>(9) != 0;

    uint64_t remote = 0;
    if (kind == 0) {
        remote = hsio_sram_addr(index, cce, static_cast<uint32_t>(offset));
    } else if (kind == 1) {
        remote = mimir_cce_sram_addr(index, offset);
    } else {
        remote = mimir_gddr_addr(index, offset);
    }

    status[0] = 0x1D3A0001;
    if (cpu) {
        volatile uint32_t* local = reinterpret_cast<volatile uint32_t*>(own_sram_uncached_addr(local_offset));
        for (uint32_t i = 0; i < bytes / sizeof(uint32_t); i++) {
            if (read) {
                local[i] = read32(remote + i * sizeof(uint32_t));
            } else {
                write32(remote + i * sizeof(uint32_t), local[i]);
            }
        }
    } else {
        const uint64_t local = local_spa_addr(my_eth_cce(), local_offset);
        if (read) {
            async_read(remote, local, bytes, 1);
        } else {
            async_write(local, remote, bytes, 1);
        }
        barrier(1);
    }
    asm volatile("fence rw, rw" ::: "memory");
    status[4] = 0x1D3A0003;
}
