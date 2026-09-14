// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"

// Independent probe of the write/relay ordering used by chain forwarding.
void kernel_main() {
    constexpr uint32_t bytes = 65536, rounds = 128;
    constexpr bool flush_between = get_compile_time_arg_val(0);
    const auto output = TensorAccessor(TensorAccessorArgs<1>(), get_arg_val<uint32_t>(0));
    const uint32_t rank = get_arg_val<uint32_t>(1);
    const uint32_t prev_x = get_arg_val<uint32_t>(2), prev_y = get_arg_val<uint32_t>(3);
    const uint32_t next_x = get_arg_val<uint32_t>(4), next_y = get_arg_val<uint32_t>(5);
    Noc noc;
    Semaphore<> ready(0), ack(1), signal(2);
    CircularBuffer payload(0), observations(1);
    payload.reserve_back(bytes / 2048);
    observations.reserve_back(1);
    const uint32_t address = payload.get_write_ptr(), result_address = observations.get_write_ptr();
    auto* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    auto* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_address);
    for (uint32_t i = 0; i < 512; ++i) {
        result[i] = 0;
    }
    for (uint32_t r = 0; r < rounds; ++r) {
        if (rank > 0) {
            if (rank == 2 && r % 3 == 0) {
                for (uint32_t i = 0; i < 1000; ++i) {
                    asm volatile("nop");
                }
            }
            ready.set(INVALID);
            ack.up(noc, prev_x, prev_y, 1);
            ready.wait(VALID);
            // No barrier between readiness and inspection of every payload word.
            for (uint32_t i = 0; i < bytes / 4; ++i) {
                result[r] += data[i] != ((r + 1) * 65537u ^ i);
            }
        }
        if (rank < 2) {
            ack.wait(1);
            ack.set(0);
            if (rank == 0) {
                for (uint32_t i = 0; i < bytes / 4; ++i) {
                    data[i] = (r + 1) * 65537u ^ i;
                }
            }
            noc.async_write(
                CoreLocalMem<uint32_t>(address),
                UnicastEndpoint{},
                bytes,
                {},
                {.noc_x = next_x, .noc_y = next_y, .addr = address});
            if constexpr (flush_between) {
                noc.async_writes_flushed();
            }
            signal.relay_unicast(noc, ready, next_x, next_y);
            noc.async_writes_flushed();
        }
    }
    noc.async_atomic_barrier();
    noc.async_write(CoreLocalMem<uint32_t>(result_address), output, 2048, {}, {.page_id = rank});
    noc.async_write_barrier();
}
