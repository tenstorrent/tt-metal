// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Transport micro-benchmark: one fixed sender pushes ROUNDS payloads of BYTES to a receiver set,
// using whichever transport (hardware multicast or chain unicast) the host family selected.
// Every participating core records its own elapsed wall-clock cycles for the loop into the output
// tensor page indexed by its runtime core index. Sender time is the end-to-end figure: it includes
// the per-round consumer-ready handshake from all receivers.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "risc_common.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"
using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("mcast_ct_offset"),
        get_named_compile_time_arg_val("mcast_rt_offset")>();
    constexpr uint32_t rounds = get_compile_time_arg_val(0);
    constexpr uint32_t bytes = get_compile_time_arg_val(1);
    constexpr auto output_args = TensorAccessorArgs<2>();
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(0));
    const uint32_t page_id = get_arg_val<uint32_t>(1);

    Noc noc;
    CircularBuffer payload(0), scratch(1);
    payload.reserve_back(1);
    scratch.reserve_back(1);
    const uint32_t buf = payload.get_write_ptr();
    auto* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch.get_write_ptr());

    auto sender = mc.optional_sender(noc);
    auto receiver = mc.optional_receiver(noc);

    const uint64_t t0 = get_timestamp();
    if (sender) {
        for (uint32_t r = 0; r < rounds; ++r) {
            sender->send(buf, buf, bytes);
        }
    } else if (receiver) {
        for (uint32_t r = 0; r < rounds; ++r) {
            receiver->receive_and_forward(buf, bytes, r);
        }
    }
    const uint64_t t1 = get_timestamp();

    result[0] = static_cast<uint32_t>(t1 - t0);
    result[1] = static_cast<uint32_t>((t1 - t0) >> 32);
    result[2] = sender ? 1u : (receiver ? 2u : 0u);
    result[3] = mc.transfer_mode == TransferMode::ChainUnicast ? 1u : 0u;
    noc.async_write(CoreLocalMem<uint32_t>(scratch.get_write_ptr()), output, 2048, {}, {.page_id = page_id});
    noc.async_write_barrier();
}
