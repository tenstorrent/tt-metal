// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader half of the persistent D2H stream service (paired with persistent_d2h_writer.cpp,
// running on the other data-movement RISC of the same service core). The reader owns
// backing-tensor DRAM reads and the worker-sync handshake: it multicasts transfer_done
// to worker cores, waits for all workers to ack, then reads tensor pages from DRAM into
// data-CB slots for the writer to drain. The reader never touches the D2H socket.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "../../../unified_kernels/termination.hpp"

constexpr uint32_t num_socket_pages = get_compile_time_arg_val(0);
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(1);
constexpr uint32_t pages_per_chunk = get_compile_time_arg_val(2);
constexpr uint32_t data_cbuf_index = get_compile_time_arg_val(3);
constexpr uint32_t worker_sync_enabled = get_compile_time_arg_val(4);
constexpr uint32_t transfer_done_sem_addr = get_compile_time_arg_val(5);
constexpr uint32_t write_ack_counter_addr = get_compile_time_arg_val(6);
constexpr uint32_t worker_mcast_noc_x_start = get_compile_time_arg_val(7);
constexpr uint32_t worker_mcast_noc_y_start = get_compile_time_arg_val(8);
constexpr uint32_t worker_mcast_noc_x_end = get_compile_time_arg_val(9);
constexpr uint32_t worker_mcast_noc_y_end = get_compile_time_arg_val(10);
constexpr uint32_t num_workers = get_compile_time_arg_val(11);
constexpr auto input_tensor_accessor_args = TensorAccessorArgs<12>();

void kernel_main() {
    Noc noc;
    CircularBuffer data_cbuf(data_cbuf_index);

    const uint32_t termination_semaphore_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_tensor_addr = get_arg_val<uint32_t>(1);

    auto input_tensor_accessor = TensorAccessor(input_tensor_accessor_args, input_tensor_addr);

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);

    uint64_t worker_mcast_addr = 0;
    if constexpr (worker_sync_enabled) {
        worker_mcast_addr = get_noc_multicast_addr(
            worker_mcast_noc_x_start,
            worker_mcast_noc_y_start,
            worker_mcast_noc_x_end,
            worker_mcast_noc_y_end,
            transfer_done_sem_addr);
    }
    volatile tt_l1_ptr uint32_t* write_ack_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_ack_counter_addr);
    uint32_t last_write_ack = 0;

    bool terminated = false;
    while (!terminated) {
        if constexpr (worker_sync_enabled) {
            // Device 2.0 migration: legacy primitive retained — transfer_done_sem_addr is a GlobalSemaphore
            // address, and Semaphore<> binds to per-program ids via get_semaphore<>(id) (no GlobalSemaphore
            // wrapper exists), so Semaphore::inc_multicast cannot target it.
            noc_semaphore_inc_multicast(worker_mcast_addr, /*incr=*/1, /*num_dests=*/num_workers);
            noc.async_atomic_barrier();
        }

        // Always wait for write_ack before reading DRAM. When worker_sync is enabled, workers
        // ack via the write_ack_counter after producing backing-tensor data. When worker_sync
        // is disabled (host-only path), the host increments write_ack_counter via
        // notify_backing_ready(). In both cases this gates one transfer per ack, preventing the
        // reader from free-running and producing data the host hasn't requested yet.
        while (true) {
            invalidate_l1_cache();
            const uint32_t cur = *write_ack_ptr;
            if ((cur - last_write_ack) == num_workers) {
                last_write_ack = cur;
                break;
            }
            if (termination_semaphore[0] == 1) {
                terminated = true;
                break;
            }
        }
        if (terminated) {
            break;
        }

        for (uint32_t chunk = 0; chunk < num_socket_pages; ++chunk) {
            while (!data_cbuf.pages_reservable_at_back(1)) {
                invalidate_l1_cache();
                if (termination_semaphore[0] == 1) {
                    terminated = true;
                    break;
                }
            }
            if (terminated) {
                break;
            }

            const uint32_t base_page = chunk * pages_per_chunk;
            for (uint32_t i = 0; i < pages_per_chunk; ++i) {
                noc.async_read<NocOptions::DEFAULT, input_tensor_page_size>(
                    input_tensor_accessor,
                    data_cbuf,
                    input_tensor_page_size,
                    {.page_id = base_page + i},
                    {.offset_bytes = i * input_tensor_page_size});
            }
            noc.async_read_barrier();
            data_cbuf.push_back(1);
        }

        if constexpr (num_socket_pages == 0) {
            // Metadata-only: no DRAM payload. Emit a single empty token page per transfer so the
            // writer's metadata push stays gated behind this transfer's write_ack (the reader/writer
            // split otherwise decouples the ack-wait from the socket push).
            while (!data_cbuf.pages_reservable_at_back(1)) {
                invalidate_l1_cache();
                if (termination_semaphore[0] == 1) {
                    terminated = true;
                    break;
                }
            }
            if (!terminated) {
                data_cbuf.push_back(1);
            }
        }

        if (terminated) {
            break;
        }
    }

    noc.async_read_barrier();
}
