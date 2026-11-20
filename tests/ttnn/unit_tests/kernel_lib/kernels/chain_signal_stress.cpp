// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr auto chain = McastArgs<
        get_named_compile_time_arg_val("chain_mcast_ct_offset"),
        get_named_compile_time_arg_val("chain_mcast_rt_offset")>();
    constexpr auto reverse = McastArgs<
        get_named_compile_time_arg_val("reverse_mcast_ct_offset"),
        get_named_compile_time_arg_val("reverse_mcast_rt_offset")>();
    static_assert(chain.transfer_mode == TransferMode::ChainUnicast);
    constexpr uint32_t base = 0;
    constexpr uint32_t rounds = 128, bytes = 65536, words = bytes / 4;
    constexpr uint32_t events = get_compile_time_arg_val(base);  // 0: payload, 1: control, 2: mixed.
    constexpr auto sender_guard =
        get_compile_time_arg_val(base + 1) ? SourceL1Guard::CallerManaged : SourceL1Guard::Guard;
    constexpr auto receiver_guard =
        get_compile_time_arg_val(base + 2) ? SourceL1Guard::CallerManaged : SourceL1Guard::Guard;
    const auto output = TensorAccessor(TensorAccessorArgs<base + 4>(), get_arg_val<uint32_t>(0));
    const uint32_t rank = get_arg_val<uint32_t>(1);
    constexpr bool includes_sender = get_compile_time_arg_val(base + 3);
    Noc noc;
    CircularBuffer source(0), destination(1), observations(2);
    source.reserve_back(bytes / 2048);
    destination.reserve_back(bytes / 2048);
    observations.reserve_back(1);
    const uint32_t src_base = source.get_write_ptr(), dst = destination.get_write_ptr();
    const uint32_t result_address = observations.get_write_ptr();
    auto* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_address);
    auto* received = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t i = 0; i < 512; ++i) {
        result[i] = 0;
    }
    for (uint32_t r = 0; r < rounds; ++r) {
        // Reconstruct every event: Counter progression and an in-flight signal source cannot
        // live in the object, nor be reset by its constructor/destructor.
        auto sender = chain.optional_sender(noc);
        auto receiver = chain.optional_receiver(noc);
        auto reverse_sender = reverse.optional_sender(noc);
        auto reverse_receiver = reverse.optional_receiver(noc);
        const bool control = events == 1 || (events == 2 && r % 2);
        const uint32_t signal = chain.signal == DataReadySignal::Counter ? VALID : 7u + r % 5;
        if (reverse_sender) {
            // C sends Y before receiving X. A must advance past first-hop injection to ack Y.
            reverse_sender->send_signal();
        }
        if (rank == 2 && r % 3 == 0) {
            for (uint32_t i = 0; i < 1000; ++i) {
                asm volatile("nop");
            }
        }
        if (sender) {
            if (control) {
                sender->send_signal<sender_guard>(signal);
                // No CallerManaged flush here: next successor ack must protect the internal word.
            } else {
                const uint32_t src = includes_sender && r % 3 == 0 ? src_base : dst;
                auto* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
                for (uint32_t i = 0; i < words; ++i) {
                    data[i] = (r + 1) * 65537u ^ i;
                }
                sender->send<sender_guard>(src, dst, bytes);
                if (src != dst) {
                    // Local destination completion is guaranteed even under CallerManaged.
                    for (uint32_t i = 0; i < words; ++i) {
                        result[r] += received[i] != ((r + 1) * 65537u ^ i);
                    }
                }
                if constexpr (sender_guard == SourceL1Guard::CallerManaged) {
                    for (uint32_t i = 0; i < 64; ++i) {
                        asm volatile("nop");
                    }
                    noc.async_writes_flushed();
                }
                // Guard must have protected source departure before return; this deliberately
                // overwrites the entire source before waiting for another successor ack.
                for (uint32_t i = 0; i < words; ++i) {
                    data[i] = 0xdeadbeef;
                }
            }
        } else if (receiver) {
            if (control) {
                result[r] += receiver->receive_signal(r) != (chain.signal == DataReadySignal::Counter ? r + 1 : signal);
            } else {
                receiver->receive_and_forward<receiver_guard>(dst, bytes, r);
                // Reading our received buffer is safe while an outgoing write also reads it.
                for (uint32_t i = 0; i < words; ++i) {
                    result[r] += received[i] != ((r + 1) * 65537u ^ i);
                }
                if constexpr (receiver_guard == SourceL1Guard::CallerManaged) {
                    noc.async_writes_flushed();
                }
                for (uint32_t i = 0; i < words; ++i) {
                    received[i] = 0xdeadbeef;
                }
            }
        }
        if (reverse_receiver) {
            reverse_receiver->receive_signal(r);
        }
    }
    // No per-round DRAM traffic/barriers can accidentally protect the signal source.
    noc.async_writes_flushed();
    noc.async_atomic_barrier();
    noc.async_write(CoreLocalMem<uint32_t>(result_address), output, 2048, {}, {.page_id = rank});
    noc.async_write_barrier();
}
