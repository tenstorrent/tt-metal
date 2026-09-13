// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// One end of a worker-to-worker ping-pong. Each round the initiator stamps PP_TX and writes the round number into
// the peer's flag; the peer spins on the flag, stamps PP_RX, stamps PP_TX and writes back; the initiator spins and
// stamps PP_RX. Initiator and NoC alternate by round, so each direction's one-way time averages over both rings.
// The flag is a NoC atomic increment, so no source data has to be in L1 when the NIU fetches it (inline and 64 B
// writes both went out with the previous round's value now and then); a spin gives up after kSpinLimit polls,
// leaving the round it was waiting for at flag_addr + 4 for the host.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t kSpinLimit = 1u << 24;

void kernel_main() {
    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t peer_x = get_arg_val<uint32_t>(1);
    const uint32_t peer_y = get_arg_val<uint32_t>(2);
    const uint32_t flag_addr = get_arg_val<uint32_t>(3);
    const uint32_t rounds = get_arg_val<uint32_t>(4);
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    noc_local_state_init(1);
    auto send = [&](uint8_t noc) { noc_semaphore_inc(get_noc_addr(peer_x, peer_y, flag_addr, noc), 1, noc); };
    auto wait = [&](uint32_t r) {
        for (uint32_t polls = 0; *flag < r; polls++) {
            invalidate_l1_cache();
            if (polls == kSpinLimit) {
                flag[1] = r;
                return false;
            }
        }
        return true;
    };
    for (uint32_t r = 1; r <= rounds; r++) {
        const uint8_t noc = static_cast<uint8_t>((r >> 1) & 1u);
        if ((r & 1u) == role) {
            {
                DeviceZoneScopedN("PP_TX");
                send(noc);
            }
            if (!wait(r)) {
                break;
            }
            {
                DeviceZoneScopedN("PP_RX");
            }
        } else {
            if (!wait(r)) {
                break;
            }
            {
                DeviceZoneScopedN("PP_RX");
            }
            {
                DeviceZoneScopedN("PP_TX");
                send(noc);
            }
        }
    }
    noc_async_atomic_barrier(0);
    noc_async_atomic_barrier(1);
}
