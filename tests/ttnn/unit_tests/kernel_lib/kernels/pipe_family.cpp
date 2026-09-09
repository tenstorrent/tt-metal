// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr auto mc = McastArgs<0, 6>();
    constexpr auto absent = McastArgs<mc.next_compile_time_args_offset(), mc.next_runtime_args_offset()>();
    constexpr auto barrier = McastArgs<absent.next_compile_time_args_offset(), absent.next_runtime_args_offset()>();
    static_assert(!absent.active);
    constexpr uint32_t base = barrier.next_compile_time_args_offset();
    constexpr uint32_t rounds = get_compile_time_arg_val(base);
    constexpr bool control = get_compile_time_arg_val(base + 1);
    constexpr bool caller_managed = get_compile_time_arg_val(base + 2);
    constexpr bool dynamic = get_compile_time_arg_val(base + 3);
    constexpr auto input_args = TensorAccessorArgs<base + 4>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(0));
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(1));
    const uint32_t input_base = get_arg_val<uint32_t>(2);
    const uint32_t output_base = get_arg_val<uint32_t>(3);
    const bool inside = get_arg_val<uint32_t>(4);
    const bool spectator = get_arg_val<uint32_t>(5);
    Noc noc;
    CircularBuffer source(0), destination(1);
    source.reserve_back(4);
    destination.reserve_back(4);
    const uint32_t source_addr = source.get_write_ptr(), destination_addr = destination.get_write_ptr();
    for (uint32_t i = 0; i < 8192 / 4; ++i) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_addr)[i] = 0x5a5a5a5a;
    }
    auto sender = mc.optional_sender(noc);
    auto receiver = mc.optional_receiver(noc);
    auto barrier_sender = barrier.optional_sender(noc);
    auto barrier_receiver = barrier.optional_receiver(noc);
    // Start and finish barriers include nonparticipants, so their sentinel is observed after all
    // payload writes, without racing initialization or a sender's final transfer.
    if (barrier_sender) {
        barrier_sender->send_signal();
    } else {
        barrier_receiver->receive_signal();
    }
    for (uint32_t r = 0; r < rounds; ++r) {
        const uint32_t pages = dynamic ? 1 + (r % 2) : 1;
        const uint32_t dst = destination_addr + (dynamic && (r % 2) ? 4096 : 0);
        const bool in_place = (r / mc.num_senders) % 2 == 0;
        if (mc.should_send(r)) {
            if constexpr (control) {
                sender->send_signal(mc.signal == DataReadySignal::Counter ? VALID : 7);
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst) =
                    mc.signal == DataReadySignal::Counter ? r + 1 : 7;
            } else {
                const uint32_t src = in_place ? dst : source_addr;
                for (uint32_t p = 0; p < pages; ++p) {
                    noc.async_read(
                        input, CoreLocalMem<uint32_t>(src + 2048 * p), 2048, {.page_id = input_base + r * 2 + p}, {});
                }
                noc.async_read_barrier();
                if constexpr (caller_managed) {
                    sender->send<SourceL1Guard::CallerManaged>(src, dst, 2048 * pages);
                    noc.async_write_barrier();
                } else {
                    sender->send(src, dst, 2048 * pages);
                }
            }
        } else if (receiver) {
            if constexpr (control) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst) = receiver->receive_signal(r);
            } else {
                receiver->receive(r);
            }
        }
        if (inside) {
            for (uint32_t p = 0; p < pages; ++p) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(dst + 2048 * p), output, 2048, {}, {.page_id = output_base + r * 2 + p});
            }
            noc.async_write_barrier();
        }
    }
    if (barrier_sender) {
        barrier_sender->send_signal();
    } else {
        barrier_receiver->receive_signal();
    }
    if (spectator) {
        noc.async_write(
            CoreLocalMem<uint32_t>(destination_addr), output, 2048, {}, {.page_id = output_base + rounds * 2});
        noc.async_write_barrier();
    }
}
