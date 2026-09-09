// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
using namespace dataflow_kernel_lib;
void kernel_main() {
    constexpr auto mc = McastArgs<0, 4>();
    constexpr uint32_t base = mc.next_compile_time_args_offset();
    constexpr uint32_t rounds = get_compile_time_arg_val(base);
    constexpr bool alternating = get_compile_time_arg_val(base + 1);
    constexpr bool caller_managed = get_compile_time_arg_val(base + 2);
    constexpr bool control = get_compile_time_arg_val(base + 3);
    constexpr auto input_args = TensorAccessorArgs<base + 4>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);
    const uint32_t output_start = get_arg_val<uint32_t>(2);
    const bool inside = get_arg_val<uint32_t>(3);
    Noc noc;
    CircularBuffer source(0), destination(1);
    source.reserve_back(1);
    destination.reserve_back(1);
    const uint32_t src = source.get_write_ptr(), dst = destination.get_write_ptr();
    const auto input = TensorAccessor(input_args, input_addr);
    const auto output = TensorAccessor(output_args, output_addr);
    auto sender = mc.optional_sender(noc);
    auto receiver = mc.optional_receiver(noc);
    for (uint32_t r = 0; r < rounds; ++r) {
        if (mc.should_send(r)) {
            if constexpr (control) {
                if constexpr (caller_managed) {
                    // Fixed senders keep VALID unchanged; rotating senders retain the cleanup fence.
                    sender->send_signal<SourceL1Guard::CallerManaged>();
                } else {
                    sender->send_signal();
                }
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst) =
                    mc.signal == DataReadySignal::Counter ? r + 1 : VALID;
            } else {
                const bool in_place = !alternating || (r / mc.num_senders) % 2 == 0;
                // Stage directly into the selected source CB before sending.
                if (in_place) {
                    noc.async_read(input, destination, 2048, {.page_id = r}, {});
                } else {
                    noc.async_read(input, source, 2048, {.page_id = r}, {});
                }
                noc.async_read_barrier();
                if constexpr (caller_managed) {
                    sender->send<SourceL1Guard::CallerManaged>(in_place ? dst : src, dst, 2048);
                    // The caller owns protection; drain before staging the next source.
                    noc.async_write_barrier();
                } else {
                    sender->send(in_place ? dst : src, dst, 2048);
                }
            }
            if (!inside) {
                continue;
            }
        } else if (receiver) {
            if constexpr (control) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst) = receiver->receive_signal(r);
            } else {
                receiver->receive(r);
            }
        } else {
            continue;
        }
        noc.async_write(destination, output, 2048, {}, {.page_id = output_start + r});
        noc.async_write_barrier();
    }
}
