// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "tests/tt_metal/tt_fabric/feature_bringup/kernels/fabric_elastic_channels.hpp"
#include "core_config.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

// What do I need to get the next chunk:
// address? (could be an ID instead, in which case it could be packed in a single word)

// What happens if a worker is granted a chunk but it doesn't use it and then it needs to be relinquished to the pool?


// protocol is as follows:

// 1 bit to indicate new value
// value to indicate rest of it


void kernel_main() {
    constexpr size_t N_CHUNKS = get_compile_time_arg_val(0);
    constexpr size_t CHUNK_N_PKTS = get_compile_time_arg_val(1);

    size_t arg_idx = 0;
    size_t n_pkts = get_arg_val<size_t>(arg_idx++);
    size_t src_addr = get_arg_val<size_t>(arg_idx++);
    uint32_t dest_eth_noc_x = get_arg_val<size_t>(arg_idx++);
    uint32_t dest_eth_noc_y = get_arg_val<size_t>(arg_idx++);
    size_t payload_size = get_arg_val<size_t>(arg_idx++);

    auto next_chunk_ptr = reinterpret_cast<volatile uint32_t *>(get_semaphore<ProgrammableCoreType::TENSIX>(get_arg_val<uint32_t>(arg_idx++)));
    auto from_eth_flow_control_ptr = reinterpret_cast<volatile uint32_t *>(get_semaphore<ProgrammableCoreType::TENSIX>(get_arg_val<uint32_t>(arg_idx++)));
    size_t to_eth_flow_control_stream_id = get_arg_val<size_t>(arg_idx++);
    
    tt::tt_fabric::FabricWriterAdapter<N_CHUNKS, CHUNK_N_PKTS> fabric_writer_adapter(next_chunk_ptr);

    const uint64_t dest_sem_noc_addr = get_noc_addr(dest_eth_noc_x, dest_eth_noc_y, get_stream_reg_write_addr(to_eth_flow_control_stream_id));
    size_t pkts_sent = 0;
    while (pkts_sent < n_pkts) {
        if (fabric_writer_adapter.has_valid_destination()) {
            auto dest_bank_addr = fabric_writer_adapter.get_next_write_address();
            auto dest_noc_addr = get_noc_addr(dest_eth_noc_x, dest_eth_noc_y, (uint32_t)dest_bank_addr);
            noc_async_write(src_addr, dest_noc_addr, payload_size);
            noc_semaphore_inc(dest_sem_noc_addr, pack_value_for_inc_on_write_stream_reg_write(1));

            fabric_writer_adapter.advance_to_next_buffer_slot();
            pkts_sent++;
        } else if (fabric_writer_adapter.new_chunk_is_available()) {
            fabric_writer_adapter.update_to_new_chunk();
        }
    }
}
