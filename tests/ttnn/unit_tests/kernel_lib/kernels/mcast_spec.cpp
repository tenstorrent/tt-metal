// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args_spec.hpp"

void kernel_main() {
    using namespace dataflow_kernel_lib;
    constexpr auto channel = MCAST_ARGS(channel);
    constexpr auto second = MCAST_ARGS(second);
    constexpr auto absent = MCAST_ARGS(absent);
    static_assert(!absent.active);
    constexpr uint32_t rounds = get_arg(args::rounds);
    constexpr bool control = get_arg(args::control) != 0;
    const uint32_t seed = get_arg(args::seed);
    auto pad = Scratchpad<uint32_t>(scratch::pad);
    const uint32_t src = pad.get_base_address();
    const uint32_t dst = src + 64;
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    Noc noc;
    auto sender = channel.optional_sender(noc);
    auto receiver = channel.optional_receiver(noc);
    for (uint32_t round = 0; round < rounds; ++round) {
        uint32_t result = 0;
        if (channel.should_send(round)) {
            if constexpr (control) {
                sender->send_signal();
                result = channel.signal == DataReadySignal::Counter ? round + 1 : VALID;
            } else {
                for (uint32_t i = 0; i < 16; ++i) {
                    words[i] = seed + round * 100 + i;
                }
                sender->send(src, dst, 64);
                for (uint32_t i = 0; i < 16; ++i) {
                    result += (i + 1) * words[16 + i];
                }
            }
        } else if (receiver) {
            if constexpr (control) {
                result = receiver->receive_signal(round);
            } else {
                [&](auto& pipe) {
                    if constexpr (channel.transfer_mode == TransferMode::ChainUnicast) {
                        pipe.receive_and_forward(dst, 64, round);
                    } else {
                        pipe.receive(round);
                    }
                }(*receiver);
                for (uint32_t i = 0; i < 16; ++i) {
                    result += (i + 1) * words[16 + i];
                }
            }
        }
        words[32 + round] = result;
    }
    // A second attached family has its own resources and vararg slice.
    auto second_sender = second.optional_sender(noc);
    auto second_receiver = second.optional_receiver(noc);
    words[40] = 0;
    if (second.should_send(0)) {
        for (uint32_t i = 0; i < 16; ++i) {
            words[i] = seed + 1000 + i;
        }
        second_sender->send(src, src + 192, 64);
        for (uint32_t i = 0; i < 16; ++i) {
            words[40] += (i + 1) * words[48 + i];
        }
    } else if (second_receiver) {
        second_receiver->receive(0);
        for (uint32_t i = 0; i < 16; ++i) {
            words[40] += (i + 1) * words[48 + i];
        }
    }
    words[41] = get_vararg(0);
    words[42] = get_vararg(1);
    words[43] = absent.optional_sender(noc) || absent.optional_receiver(noc) ? 0 : 0xA55A;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::report_addr)) = src;
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
