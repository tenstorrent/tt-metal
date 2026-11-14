// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// CCE0 <-> CCE1 flag ping-pong on one Keraunos HSIO tile. The flag words live in the initiator's SRAM; the
// responder reaches them through the sibling window. Both use the uncached alias so neither spins on a stale line.
// The initiator raises ping to round i and waits for pong i; the responder waits for ping i and answers.
//
// Status words (uint32, uncached alias): 0 started, 1 last completed round, 4 done.
//
// Runtime args:
//   0: flags, SRAM offset in the initiator's SRAM (ping at +0, pong at +4)
//   1: role, 0 initiator (flags are local), 1 responder (flags are in the sibling CCE)
//   2: rounds
//   3: status block, local SRAM offset

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"

void kernel_main() {
    const bool initiator = get_arg_val<uint32_t>(1) == 0;
    volatile uint32_t* flags = reinterpret_cast<volatile uint32_t*>(
        (initiator ? MEM_L1_UNCACHED_BASE : MEM_IERISC_UNCACHED_BASE + ETH_CCE_PEER_BASE) + get_arg_val<uint32_t>(0));
    const uint32_t rounds = get_arg_val<uint32_t>(2);
    volatile uint32_t* status =
        reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + get_arg_val<uint32_t>(3));
    volatile uint32_t* ping = &flags[0];
    volatile uint32_t* pong = &flags[1];

    status[0] = 0x1D3A0001;
    for (uint32_t i = 1; i <= rounds; i++) {
        if (initiator) {
            *ping = i;
            while (*pong != i) {
            }
        } else {
            while (*ping != i) {
            }
            *pong = i;
        }
        status[1] = i;
    }
    asm volatile("fence rw, rw" ::: "memory");
    status[4] = 0x1D3A0003;
}
