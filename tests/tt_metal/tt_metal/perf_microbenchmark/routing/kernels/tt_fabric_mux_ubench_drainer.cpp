// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_utils.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"

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

namespace tt::tt_fabric {
using DrainerChannelBuffer = EthChannelBuffer<NUM_BUFFERS>;
using DrainerChannelClientLocationInfo = EDMChannelWorkerLocationInfo;
using DrainerChannelWorkerInterface = EdmChannelWorkerInterface<NUM_BUFFERS>;
using DrainerStatus = EDMStatus;
}  // namespace tt::tt_fabric

void kernel_main() {
    auto status_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(status_address);
    status_ptr[0] = tt::tt_fabric::DrainerStatus::STARTED;

    tt::tt_fabric::DrainerChannelBuffer drainer_channel(
        channel_base_address,
        BUFFER_SIZE_BYTES,
        sizeof(PACKET_HEADER_TYPE),
        0, /* unused, eth_transaction_ack_word_addr */
        0 /* channel_id */);

    auto connection_worker_info_ptr =
        reinterpret_cast<volatile tt::tt_fabric::DrainerChannelClientLocationInfo*>(connection_info_address);
    connection_worker_info_ptr->edm_rdptr = 0;

    tt::tt_fabric::DrainerChannelWorkerInterface worker_interface(
        connection_worker_info_ptr,
        reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(sender_flow_control_address),
        reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(connection_handshake_address),
        0 /* unused, sender_sync_noc_cmd_buf */);

    bool connection_established = false;

    volatile auto termination_signal_ptr =
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_address);

    status_ptr[0] = tt::tt_fabric::DrainerStatus::READY_FOR_TRAFFIC;
    while (!got_immediate_termination_signal(termination_signal_ptr)) {
        if (worker_interface.has_unsent_payload()) {
            auto& local_wrptr = worker_interface.local_wrptr;
            local_wrptr.increment();

            auto& local_rdptr = worker_interface.local_rdptr;
            local_rdptr.increment();
            worker_interface.template update_worker_copy_of_read_ptr<false>(local_rdptr.get_ptr());
        }
        check_worker_connections(worker_interface, connection_established);
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
    status_ptr[0] = tt::tt_fabric::DrainerStatus::TERMINATED;
}
