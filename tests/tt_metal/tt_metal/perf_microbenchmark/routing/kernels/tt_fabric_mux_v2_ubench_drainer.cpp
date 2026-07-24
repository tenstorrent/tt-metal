// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_utils.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
// clang-format on

#include <cstddef>

constexpr uint8_t NUM_BUFFERS = get_compile_time_arg_val(0);
constexpr size_t status_address = get_compile_time_arg_val(1);
constexpr size_t connection_info_address = get_compile_time_arg_val(2);
constexpr size_t connection_handshake_address = get_compile_time_arg_val(3);
constexpr uint8_t upstream_free_slots_stream_id = get_compile_time_arg_val(4);

namespace tt::tt_fabric {
using DrainerChannelClientLocationInfo = EDMChannelWorkerLocationInfo;
using DrainerChannelWorkerInterface =
    StaticSizedSenderChannelWorkerInterface<tt::tt_fabric::worker_handshake_noc, NUM_BUFFERS>;
using DrainerStatus = EDMStatus;
}  // namespace tt::tt_fabric

void kernel_main() {
    size_t rt_args_idx = 0;
    const uint32_t expected_total_packets_low = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t expected_total_packets_high = get_arg_val<uint32_t>(rt_args_idx++);
    const uint64_t expected_total_packets =
        static_cast<uint64_t>(expected_total_packets_low) | (static_cast<uint64_t>(expected_total_packets_high) << 32);

    auto status_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(status_address);
    status_ptr[0] = tt::tt_fabric::DrainerStatus::STARTED;
    init_ptr_val(upstream_free_slots_stream_id, NUM_BUFFERS);
    auto connection_worker_info_ptr =
        reinterpret_cast<volatile tt::tt_fabric::DrainerChannelClientLocationInfo*>(connection_info_address);

    auto connection_handshake_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(connection_handshake_address);
    tt::tt_fabric::DrainerChannelWorkerInterface worker_interface(
        connection_worker_info_ptr,
        nullptr,
        connection_handshake_ptr,
        0,
        tt::tt_fabric::MUX_TO_WORKER_INTERFACE_STARTING_READ_COUNTER_VALUE);

    status_ptr[0] = tt::tt_fabric::DrainerStatus::READY_FOR_TRAFFIC;
    while (true) {
        invalidate_l1_cache();
        const auto connection_state = connection_handshake_ptr[0];
        if (connection_state == tt::tt_fabric::connection_interface::open_connection_value ||
            connection_state == tt::tt_fabric::connection_interface::close_connection_request_value) {
            break;
        }
    }

    worker_interface.template cache_producer_noc_addr<true>();
    worker_interface.notify_worker_of_read_counter_update();

    uint64_t remaining_packets = expected_total_packets;
    while (remaining_packets != 0) {
        while (get_ptr_val(upstream_free_slots_stream_id) == NUM_BUFFERS) {
            invalidate_l1_cache();
        }

        // Hoist the stream reg update early to avoid stale reads for the next packet since read and writes are to
        // different addresses and CPU can't guarantee ordering.
        increment_local_update_ptr_val(upstream_free_slots_stream_id, 1);

        worker_interface.local_read_counter.increment();
        worker_interface.notify_worker_of_read_counter_update();
        remaining_packets -= 1;
    }

    while (!worker_interface.has_worker_teardown_request()) {
        invalidate_l1_cache();
    }
    worker_interface.template teardown_worker_connection<true, true>();

    noc_async_write_barrier();
    noc_async_atomic_barrier();
    status_ptr[0] = tt::tt_fabric::DrainerStatus::TERMINATED;
}
