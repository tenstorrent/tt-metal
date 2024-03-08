// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"

// #include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

FORCE_INLINE void fetch_chunk(
    const uint32_t max_pages_per_chunk,
    const uint32_t total_pages_to_read,
    uint32_t& num_pages_read,
    const uint32_t& cb_id,
    const uint32_t& page_size,
    uint64_t remote_l1_read_addr) {
    const uint32_t num_pages_this_chunk = std::min(total_pages_to_read - num_pages_read, max_pages_per_chunk);

    for (uint32_t i = 0; i < num_pages_this_chunk; ++i) {
        cb_reserve_back(cb_id, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id);
        noc_async_read(remote_l1_read_addr, l1_write_addr, page_size);
        remote_l1_read_addr += page_size;
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }

    num_pages_read += num_pages_this_chunk;
}

void kernel_main() {
    const uint32_t eth_receiver_l1_base_addr = get_compile_time_arg_val(0);
    const uint32_t eth_receiver_l1_sem_addr = get_compile_time_arg_val(1);
    const uint32_t num_pages_per_read_chunk = get_arg_val<uint32_t>(0);
    const uint32_t total_pages_to_read = get_arg_val<uint32_t>(1);
    const uint32_t page_size = get_arg_val<uint32_t>(2);
    const uint32_t receiver_erisc_datamover_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t receiver_erisc_datamover_noc_y = get_arg_val<uint32_t>(4);
    // Worker local L1 semaphore that erisc datamover signals to
    const uint32_t receiver_read_sem_addr = get_arg_val<uint32_t>(5);

    DPRINT << " rwr: args: eth_receiver_l1_base_addr="<<
        eth_receiver_l1_base_addr<<
        "\n\teth_receiver_l1_sem_addr="<<eth_receiver_l1_sem_addr<<
        "\n\tnum_pages_per_read_chunk="<<num_pages_per_read_chunk<<
        "\n\ttotal_pages_to_read="<<total_pages_to_read<<
        "\n\tpage_size="<<page_size<<
        "\n\treceiver_erisc_datamover_noc_x="<<receiver_erisc_datamover_noc_x<<
        "\n\treceiver_erisc_datamover_noc_y="<<receiver_erisc_datamover_noc_y<<
        "\n\treceiver_read_sem_addr="<<receiver_read_sem_addr<<
        "\n";

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    // Eth receiver will set this semaphore when data is available
    volatile tt_l1_ptr uint32_t* receiver_read_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_read_sem_addr);

    // Address of the buffer on the eth receiver, this is different per receiver worker core
    const uint64_t eth_receiver_l1_base_noc_addr =
        get_noc_addr(receiver_erisc_datamover_noc_x, receiver_erisc_datamover_noc_y, eth_receiver_l1_base_addr);
    // Address of the semaphore on the eth receiver, this is the same per receiver worker core
    const uint64_t eth_receiver_l1_semaphore_noc_addr =
        get_noc_addr(receiver_erisc_datamover_noc_x, receiver_erisc_datamover_noc_y, eth_receiver_l1_sem_addr);

    DPRINT << " rwr: noc_index " << (uint32_t)noc_index << "\n";
    DPRINT << " rwr: my_x[0],my_y[0] " << (uint32_t)my_x[0] << "," << (uint32_t)my_y[0] << "\n";
    DPRINT << " rwr: my_x[1],my_y[1] " << (uint32_t)my_x[1] << "," << (uint32_t)my_y[1] << "\n";
    uint32_t num_pages_read = 0;
    while (num_pages_read < total_pages_to_read) {
        DPRINT << " rwr: page " << num_pages_read << " waiting for semaphore at " << (uint32_t)receiver_read_sem_addr << "\n";
        noc_semaphore_wait(receiver_read_semaphore_addr_ptr, 1);
        DPRINT << " rwr: got semaphore signal from sender erisc\n";
        noc_semaphore_set(receiver_read_semaphore_addr_ptr, 0);
        // Read page by page so that writer can be kicked off instead of being blocked waiting for full chunk to be read
        // Look into perf/optimizations for this
        DPRINT << " rwr: fetch chunk\n";
        fetch_chunk(
            num_pages_per_read_chunk,
            total_pages_to_read,
            num_pages_read,
            cb_id_in0,
            page_size,
            eth_receiver_l1_base_noc_addr);
        DPRINT << " rwr: increment semaphore on eth core at address " << eth_receiver_l1_sem_addr << "\n";
        noc_semaphore_inc(eth_receiver_l1_semaphore_noc_addr, 1);
    }

}
