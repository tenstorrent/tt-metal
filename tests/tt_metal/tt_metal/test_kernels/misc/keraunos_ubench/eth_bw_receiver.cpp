// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ethernet bandwidth receiver, on CCE0 at the far end of the link (IDLE_ETH, DM0); see eth_bw_sender.cpp. Passive for
// the payload: the sends land in its data region without it. It answers the sender's two flags (1 = handshake,
// 2 = closing) by storing the same value into the sender's ack word through the sender tile's global SRAM address.
// The host checks the data region afterwards.
//
// Ethernet RX writes are not snooped (no snoop on the send), so the flag is read through the uncached alias.
//
// Runtime args: 0 base, 1 polls per wait, 2-3 global address of the sender's ack word.

#include <cstdint>

#include "tests/tt_metal/tt_metal/test_kernels/misc/keraunos_ubench/keraunos_ubench.h"

using namespace keraunos_ubench;

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0);
    const uint32_t polls = get_arg_val<uint32_t>(1);
    const uint64_t ack_global = arg_u64(2);

    volatile Result* r = result_block(base, 0);
    r->kernel = static_cast<uint32_t>(KernelId::EthBwReceiver);
    r->hart = 0;
    r->status = kStatusRunning;

    const volatile uint32_t* flag = uncached(base + kEthFlagOffset);
    volatile uint32_t* ack = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(ack_global));

    for (uint32_t want = 1; want <= 2; want++) {
        if (!wait_eq<uint32_t>(flag, want, polls)) {
            r->detail = want;
            finish(r, kStatusTimeout);
            return;
        }
        *ack = want;
        asm volatile("fence iorw, iorw" ::: "memory");
        r->iterations_done = want;
    }
    finish(r, kStatusDone);
}
