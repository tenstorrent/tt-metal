// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writer half of the persistent H2D stream service (paired with persistent_h2d_reader.cpp,
// running on the other data-movement RISC of the same service core). The writer consumes
// full socket pages the reader stages in the data CB, scatters their tensor pages into the
// backing DRAM tensor, and -- because only it knows when a transfer's DRAM writes are
// complete -- owns the metadata multicast and the per-transfer worker-sync handshake.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "../../../unified_kernels/termination.hpp"

constexpr uint32_t num_socket_pages = get_compile_time_arg_val(0);
constexpr uint32_t output_tensor_page_size = get_compile_time_arg_val(1);
constexpr uint32_t pages_per_chunk = get_compile_time_arg_val(2);
constexpr uint32_t data_cb_index = get_compile_time_arg_val(3);
constexpr uint32_t metadata_enabled = get_compile_time_arg_val(4);
constexpr uint32_t metadata_cb_index = get_compile_time_arg_val(5);
constexpr uint32_t worker_sync_enabled = get_compile_time_arg_val(6);
constexpr auto output_tensor_accessor_args = TensorAccessorArgs<7>();

// Runtime args:
//   [0]  termination_semaphore_addr
//   [1]  output_tensor_addr
//   [2]  data_ready_sem_addr
//   [3]  consumed_counter_addr
//   [4]  worker_mcast_noc_x_start
//   [5]  worker_mcast_noc_y_start
//   [6]  worker_mcast_noc_x_end
//   [7]  worker_mcast_noc_y_end
//   [8]  num_workers
//   [9]  metadata_size_bytes
//   [10] metadata_l1_addr
//   [11] completion_pcie_xy_enc
//   [12] completion_pcie_addr_lo
//   [13] completion_pcie_addr_hi
//   [14] completion_src_l1_addr
//
// DRAM-completion push block. After a transfer's tensor pages have committed
// to DRAM the writer bumps a monotonic counter and pushes it to its host-pinned slot over PCIe,
// so the host-side barrier() can confirm the backing tensor (not just the socket FIFO) has
// drained. The reader's early socket ack only recycles the host FIFO slot; it no longer
// implies the data has landed in DRAM, so this counter -- not the socket ack -- is what
// barrier() keys on. The PCIe encoding is built from TRANSLATED coords, so it is
// NOC-independent and valid on this kernel's NOC (RISCV_1_default) even though the socket
// reader issues PCIe transactions on the other NOC.

// Push the monotonic transfers-completed counter from an L1 scratch word to its host pinned slot
// over PCIe. Mirrors socket_notify_sender's bytes_acked push; the 64-bit PCIe offset requires
// the stateful write path. Fire-and-forget -- the host polls the pinned slot in barrier().
inline void push_completion_counter(
    uint32_t value,
    uint32_t completion_pcie_xy_enc,
    uint32_t completion_pcie_addr_lo,
    uint32_t completion_pcie_addr_hi,
    uint32_t completion_src_l1_addr) {
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(completion_src_l1_addr);
    *src = value;
    const uint64_t pcie_addr =
        (static_cast<uint64_t>(completion_pcie_addr_hi) << 32) | static_cast<uint64_t>(completion_pcie_addr_lo);
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
    noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        NOC_INDEX, completion_src_l1_addr, completion_pcie_xy_enc, pcie_addr, sizeof(uint32_t));
}

void kernel_main() {
    const uint32_t termination_semaphore_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_tensor_addr = get_arg_val<uint32_t>(1);
    const uint32_t data_ready_sem_addr = get_arg_val<uint32_t>(2);
    const uint32_t consumed_counter_addr = get_arg_val<uint32_t>(3);
    const uint32_t worker_mcast_noc_x_start = get_arg_val<uint32_t>(4);
    const uint32_t worker_mcast_noc_y_start = get_arg_val<uint32_t>(5);
    const uint32_t worker_mcast_noc_x_end = get_arg_val<uint32_t>(6);
    const uint32_t worker_mcast_noc_y_end = get_arg_val<uint32_t>(7);
    const uint32_t num_workers = get_arg_val<uint32_t>(8);
    const uint32_t metadata_size_bytes = get_arg_val<uint32_t>(9);
    const uint32_t metadata_l1_addr = get_arg_val<uint32_t>(10);
    const uint32_t completion_pcie_xy_enc = get_arg_val<uint32_t>(11);
    const uint32_t completion_pcie_addr_lo = get_arg_val<uint32_t>(12);
    const uint32_t completion_pcie_addr_hi = get_arg_val<uint32_t>(13);
    const uint32_t completion_src_l1_addr = get_arg_val<uint32_t>(14);

    auto output_tensor_accessor = TensorAccessor(output_tensor_accessor_args, output_tensor_addr);

    volatile tt_l1_ptr uint32_t* termination_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_semaphore_addr);

    uint64_t worker_mcast_addr = 0;
    volatile tt_l1_ptr uint32_t* consumed_ptr = nullptr;
    uint32_t last_consumed = 0;
    if constexpr (worker_sync_enabled) {
        // The host (set_worker_mcast_corners) already ordered these corners for this kernel's
        // NOC, so "start" is the corner the NOC reaches first (the high corner on NOC 1) --
        // pass them through verbatim rather than assuming start < end.
        worker_mcast_addr = get_noc_multicast_addr(
            worker_mcast_noc_x_start,
            worker_mcast_noc_y_start,
            worker_mcast_noc_x_end,
            worker_mcast_noc_y_end,
            data_ready_sem_addr);
        consumed_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(consumed_counter_addr);
    }

    uint64_t metadata_mcast_addr = 0;
    if constexpr (metadata_enabled) {
        metadata_mcast_addr = get_noc_multicast_addr(
            worker_mcast_noc_x_start,
            worker_mcast_noc_y_start,
            worker_mcast_noc_x_end,
            worker_mcast_noc_y_end,
            metadata_l1_addr);
    }

    uint32_t transfers_completed = 0;

    bool terminated = false;
    while (!terminated) {
        // Drain one full transfer's worth of socket pages from the data CB and scatter each
        // into the backing tensor. A terminable wait lets the writer exit once the reader has
        // stopped and the CB has drained, rather than blocking forever.
        for (uint32_t chunk = 0; chunk < num_socket_pages; ++chunk) {
            if (!deepseek_b1_ops::cb_wait_for_pages_with_termination(data_cb_index, 1, termination_semaphore)) {
                terminated = true;
                break;
            }
            const uint32_t cb_l1_addr = get_read_ptr(data_cb_index);
            const uint32_t base_page = chunk * pages_per_chunk;
            uint32_t src = cb_l1_addr;
            for (uint32_t i = 0; i < pages_per_chunk; ++i) {
                const uint64_t noc_dst = output_tensor_accessor.get_noc_addr(base_page + i);
                noc_async_write<output_tensor_page_size>(src, noc_dst, output_tensor_page_size);
                src += output_tensor_page_size;
            }
            noc_async_writes_flushed();
            cb_pop_front(data_cb_index, 1);
        }
        if (terminated) {
            break;
        }

        if constexpr (metadata_enabled) {
            if (!deepseek_b1_ops::cb_wait_for_pages_with_termination(metadata_cb_index, 1, termination_semaphore)) {
                break;
            }
            const uint32_t meta_l1_addr = get_read_ptr(metadata_cb_index);
            noc_async_write_multicast(
                meta_l1_addr, metadata_mcast_addr, metadata_size_bytes, /*num_dests=*/num_workers);
            noc_async_writes_flushed();
            cb_pop_front(metadata_cb_index, 1);
        }

        // Publish the completion counter only after the barrier so a host barrier()
        // that observes it can safely read the backing tensor
        noc_async_write_barrier();
        push_completion_counter(
            ++transfers_completed,
            completion_pcie_xy_enc,
            completion_pcie_addr_lo,
            completion_pcie_addr_hi,
            completion_src_l1_addr);
        noc_async_write_barrier();

        if constexpr (worker_sync_enabled) {
            noc_semaphore_inc_multicast(worker_mcast_addr, /*incr=*/1, /*num_dests=*/num_workers);

            while (true) {
                invalidate_l1_cache();
                const uint32_t cur = *consumed_ptr;
                if ((cur - last_consumed) == num_workers) {
                    last_consumed = cur;
                    break;
                }
            }
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
