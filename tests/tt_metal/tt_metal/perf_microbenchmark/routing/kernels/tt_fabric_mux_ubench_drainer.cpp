// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_utils.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

#include <cstddef>
// clang-format on

constexpr uint8_t NUM_BUFFERS = get_compile_time_arg_val(0);
constexpr size_t BUFFER_SIZE_BYTES = get_compile_time_arg_val(1);
constexpr size_t status_address = get_compile_time_arg_val(2);
constexpr size_t termination_signal_address = get_compile_time_arg_val(3);
constexpr size_t connection_info_address = get_compile_time_arg_val(4);
constexpr size_t connection_handshake_address = get_compile_time_arg_val(5);
constexpr size_t sender_flow_control_address = get_compile_time_arg_val(6);
constexpr size_t channel_base_address = get_compile_time_arg_val(7);

constexpr size_t memory_map_start_address = get_compile_time_arg_val(8);
constexpr size_t memory_map_end_address = get_compile_time_arg_val(9);

namespace tt::tt_fabric {
using DrainerChannelBuffer = EthChannelBuffer<NUM_BUFFERS>;
using DrainerChannelClientLocationInfo = EDMChannelWorkerLocationInfo;
using DrainerChannelWorkerInterface = EdmChannelWorkerInterface<NUM_BUFFERS>;
using DrainerStatus = EDMStatus;
}  // namespace tt::tt_fabric

void kernel_main() {
    // clear out memory map
    zero_l1_buf(
        reinterpret_cast<tt_l1_ptr uint32_t*>(memory_map_start_address),
        memory_map_end_address - memory_map_start_address);

    auto status_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(status_address);
    status_ptr[0] = tt::tt_fabric::DrainerStatus::STARTED;

    // This mirrors an EDM interface. The Worker -> EDM interface has the worker communicate to the EDM interface via a
    // autoinc stream register where the register holds #slots free.
    constexpr uint32_t slots_free_stream_id =
        tt::tt_fabric::WorkerToFabricMuxSender<0>::sender_channel_0_free_slots_stream_id;
    init_ptr_val(slots_free_stream_id, NUM_BUFFERS);

    tt::tt_fabric::DrainerChannelBuffer drainer_channel(
        channel_base_address, BUFFER_SIZE_BYTES, sizeof(PACKET_HEADER_TYPE), 0 /* channel_id */);

    auto connection_worker_info_ptr =
        reinterpret_cast<volatile tt::tt_fabric::DrainerChannelClientLocationInfo*>(connection_info_address);
    connection_worker_info_ptr->edm_read_counter = 0;

    tt::tt_fabric::DrainerChannelWorkerInterface worker_interface(
        connection_worker_info_ptr,
        reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(sender_flow_control_address),
        reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(connection_handshake_address),
        0 /* unused, sender_sync_noc_cmd_buf */,
        tt::tt_fabric::MUX_TO_WORKER_INTERFACE_STARTING_READ_COUNTER_VALUE);

    bool connection_established = false;

    volatile auto termination_signal_ptr =
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_address);

    status_ptr[0] = tt::tt_fabric::DrainerStatus::READY_FOR_TRAFFIC;
    while (!got_immediate_termination_signal(termination_signal_ptr)) {
        invalidate_l1_cache();
        bool has_unsent_payload = get_ptr_val(slots_free_stream_id) != NUM_BUFFERS;
        if (has_unsent_payload) {
            worker_interface.local_write_counter.increment();
            worker_interface.local_read_counter.increment();
            worker_interface.notify_worker_of_read_counter_update();
            increment_local_update_ptr_val(slots_free_stream_id, 1);
        }
        check_worker_connections(worker_interface, connection_established, slots_free_stream_id);
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
    status_ptr[0] = tt::tt_fabric::DrainerStatus::TERMINATED;
}
