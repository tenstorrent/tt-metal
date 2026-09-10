// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/cpp/ttnn/kernel_lib/mcast_args.hpp"
using namespace dataflow_kernel_lib;
template <bool FORWARD, typename Receiver>
void receive_for_contract(Receiver& receiver) {
    if constexpr (FORWARD) {
        receiver.receive_and_forward(0u, 2048u, 0u);
    } else {
        receiver.receive(0u);
    }
}
void kernel_main() {
    constexpr auto args = McastArgs<0, 0>();
    Noc noc;
    auto receiver = args.receiver(noc);
    receive_for_contract<get_compile_time_arg_val(args.next_compile_time_args_offset())>(receiver);
}
