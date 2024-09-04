// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_edm_utils.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_edm_adapters.hpp"

void kernel_main() {
    constexpr uint32_t eth_receiver_l1_base_addr = get_compile_time_arg_val(0);
    constexpr uint32_t eth_receiver_l1_sem_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_buffers_per_channel = get_compile_time_arg_val(2);
    constexpr ttnn::ccl::EriscDataMoverTerminationMode termination_mode = static_cast<ttnn::ccl::EriscDataMoverTerminationMode>(get_compile_time_arg_val(3));
    const uint32_t num_pages_per_read_chunk = get_arg_val<uint32_t>(0);
    const uint32_t total_pages_to_read = get_arg_val<uint32_t>(1);
    const uint32_t page_size = get_arg_val<uint32_t>(2);
    const uint32_t receiver_erisc_datamover_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t receiver_erisc_datamover_noc_y = get_arg_val<uint32_t>(4);
    // Worker local L1 semaphore that erisc datamover signals to
    volatile uint32_t* const  receiver_read_sem_addr = reinterpret_cast<volatile uint32_t* const >(get_semaphore(get_arg_val<uint32_t>(5)));
    const uint32_t num_buffers_per_edm_channel = get_arg_val<uint32_t>(6);

    ccl::edm::WorkerToEdmReader<termination_mode> reader(
        ttnn::ccl::WorkerXY(receiver_erisc_datamover_noc_x, receiver_erisc_datamover_noc_y),
        eth_receiver_l1_base_addr,
        num_buffers_per_channel,
        eth_receiver_l1_sem_addr,
        num_pages_per_read_chunk * page_size,
        receiver_read_sem_addr);

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;

    for (uint32_t i = 0; i < total_pages_to_read; i += num_pages_per_read_chunk) {
        bool last_message = (i + num_pages_per_read_chunk) >= total_pages_to_read;
        uint32_t num_pages_to_read = std::min(total_pages_to_read - i, num_pages_per_read_chunk);
        reader.wait_for_payload_available();
        reader.fetch_payload_blocking(cb_id_in0, num_pages_to_read, page_size, last_message);
    }

    reader.close();
}
