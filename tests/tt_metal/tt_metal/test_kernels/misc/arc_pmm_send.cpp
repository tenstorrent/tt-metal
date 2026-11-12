// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sends one 4-byte message to ARC's PMM (power management messaging) block with a NoC write, the way the BRISC L1
// ECC halt handler announces its record. Runs on Tensix, ERISC and DRISC.
// arg 0: local L1 address for the message, 16 B-aligned like the PMM message register, so the write needs no shift
// arg 1: the message
// arg 2, 3: ARC's x, y on this kernel's NOC
// arg 4: the PMM message register's address on ARC

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    const uint32_t local_addr = get_arg_val<uint32_t>(0);
    const uint32_t message = get_arg_val<uint32_t>(1);
    const uint32_t arc_x = get_arg_val<uint32_t>(2);
    const uint32_t arc_y = get_arg_val<uint32_t>(3);
    const uint32_t pmm_message_addr = get_arg_val<uint32_t>(4);

    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_addr) = message;
    // The message must be in L1 before the NoC reads it.
    asm volatile("fence" ::: "memory");

    Noc noc;
    UnicastEndpoint src;
    UnicastEndpoint dst;
    noc.async_write(
        src, dst, sizeof(uint32_t), {.addr = local_addr}, {.noc_x = arc_x, .noc_y = arc_y, .addr = pmm_message_addr});
    // Wait only until the write has left this core, as the BRISC handler does not wait for an ack from ARC either. The
    // host reads the mailbox after the program completes, long after the write lands.
    noc.async_writes_flushed();
}
