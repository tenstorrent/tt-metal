// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

template <uint32_t ACTION, typename Channel>
void access_internal_semaphore(const Channel& channel) {
    if constexpr (ACTION == 5) {
        static_cast<void>(channel.data_ready);
    } else if constexpr (ACTION == 6) {
        static_cast<void>(channel.consumer_ready);
    } else if constexpr (ACTION == 7) {
        static_cast<void>(channel.signal_source);
    }
}

void kernel_main() {
    using namespace dataflow_kernel_lib;
    constexpr McastArgs<3, 1> channel;
    static_assert(channel.next_runtime_args_offset() == get_compile_time_arg_val(1));
    static_assert(channel.next_compile_time_args_offset() == get_compile_time_arg_val(2));
    constexpr uint32_t action = get_compile_time_arg_val(0);
    Noc noc;
    if constexpr (action == 0) {
        auto sender = channel.optional_sender(noc);
        auto receiver = channel.optional_receiver(noc);
        // Ordinary branches must compile with inert unavailable faces. The host
        // supplies false, so this compiler contract does not publish or wait.
        if (get_arg_val<uint32_t>(0)) {
            sender->send_signal();
            receiver->receive(0);
        }
    } else if constexpr (action == 1) {
        auto sender = channel.sender(noc);
    } else if constexpr (action == 2) {
        auto receiver = channel.receiver(noc);
    } else if constexpr (action == 3) {
        volatile uint32_t x = channel.sender_x();
    } else if constexpr (action == 4) {
        ASSERT(channel.sender_x() == get_compile_time_arg_val(channel.next_compile_time_args_offset()));
        ASSERT(channel.sender_y() == get_compile_time_arg_val(channel.next_compile_time_args_offset() + 1));
    } else if constexpr (action >= 5) {
        access_internal_semaphore<action>(channel);
    }
}
