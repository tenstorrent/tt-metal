// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric worker on a Mimir CCE: builds a packet (a two-word header and a payload) in a Keraunos router's channel
// buffer by stores through experimental::keraunos_hsio_sram_address, publishes it by storing a monotonic sequence
// number to the router's counter, then waits for the router to return a credit into this CCE's SRAM.
//
// Packet: word 0 = 0xFAB10000 | payload bytes, word 1 = sequence, payload words follow (word i = tag << 16 | i).
//
// Runtime args:
//   0, 1: router's HSIO tile and CCE
//   2:    channel buffer, CCE-relative offset in the router's SRAM
//   3:    counter, CCE-relative offset in the router's SRAM
//   4:    payload words
//   5:    payload tag
//   6:    credit word, local SRAM offset (the router stores the sequence number there)
//   7:    status block, local SRAM offset: 1 = sent, 2 = credit seen, 4 = done

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"
#include "experimental/gddr_cce.h"

void kernel_main() {
    const uint32_t hsio = get_arg_val<uint32_t>(0);
    const uint32_t cce = get_arg_val<uint32_t>(1);
    const uint32_t buffer_offset = get_arg_val<uint32_t>(2);
    const uint32_t counter_offset = get_arg_val<uint32_t>(3);
    const uint32_t words = get_arg_val<uint32_t>(4);
    const uint32_t tag = get_arg_val<uint32_t>(5);
    volatile uint32_t* credit =
        reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + get_arg_val<uint32_t>(6));
    volatile uint32_t* status =
        reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + get_arg_val<uint32_t>(7));

    using namespace experimental;
    constexpr uint32_t kWordBytes = sizeof(uint32_t);
    keraunos_hsio_sram_write_uint32(hsio, cce, buffer_offset, 0xFAB10000u | (words * kWordBytes));
    keraunos_hsio_sram_write_uint32(hsio, cce, buffer_offset + kWordBytes, 1);
    for (uint32_t i = 0; i < words; i++) {
        keraunos_hsio_sram_write_uint32(hsio, cce, buffer_offset + (2 + i) * kWordBytes, (tag << 16) | i);
    }
    asm volatile("fence w, w" ::: "memory");
    // Reading the last word back orders the payload stores ahead of the counter store across D2D.
    (void)keraunos_hsio_sram_read_uint32(hsio, cce, buffer_offset + (2 + words - 1) * kWordBytes);
    keraunos_hsio_sram_write_uint32(hsio, cce, counter_offset, 1);
    asm volatile("fence w, w" ::: "memory");
    status[1] = 1;

    while (*credit != 1) {
    }
    status[2] = 1;
    status[4] = 0x1D3A0003;
}
