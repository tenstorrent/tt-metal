// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args_spec.hpp"

void kernel_main() {
    using namespace dataflow_kernel_lib;
    constexpr auto channel = MCAST_SPEC_ARGS(channel);
    static_assert(channel.ack_count == UINT32_MAX && channel.num_senders == 2);
    static_assert(channel.signal == DataReadySignal::Counter);
    constexpr uint32_t rounds = get_arg(args::rounds);
    const uint32_t seed = get_arg(args::seed);
    const bool acknowledges = get_arg(args::acknowledges) != 0;
    auto pad = Scratchpad<uint32_t>(scratch::pad);
    const uint32_t source = pad.get_base_address();
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source);
    Noc noc;
    auto sender = channel.optional_sender(noc);
    auto receiver = channel.optional_receiver(noc);
    Semaphore ready(sem::channel_mcast_data_ready);
    for (uint32_t round = 0; round < rounds; ++round) {
        // Passive landing cores may lag: distinct round slots keep their unread
        // data live. Do not zero the landing area or acknowledge on their behalf.
        const uint32_t destination = source + 64 + 64 * round;
        if (channel.should_send(round)) {
            for (uint32_t i = 0; i < 16; ++i) {
                words[i] = seed + round * 100 + i;
            }
            sender->send(source, destination, 64);
        } else if (acknowledges) {
            receiver->receive(round);
        } else {
            // Host handshake_cores chooses the expected population, not which
            // kernel branches send ACKs. Passive cores only wait for publication.
            ready.wait_min(round + 1);
        }
        uint32_t sum = 0;
        auto received = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination);
        for (uint32_t i = 0; i < 16; ++i) {
            sum += received[i];
        }
        words[256 + round] = sum;
    }
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::report_addr)) = source;
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
