// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

void kernel_main() {
    constexpr dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("in1_mcast_ct_offset"),
        get_named_compile_time_arg_val("in1_mcast_rt_offset")>
        handshake;
    const Noc noc;
    auto sender = handshake.sender(noc);
    sender.send_signal();
    noc.async_write_barrier();
}
