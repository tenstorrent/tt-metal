// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"

#include <cstddef>
#include <cstdint>

namespace tt::tt_fabric::connection {

template <bool SEND_CREDIT_ADDR = false, bool posted = false, uint8_t WORKER_HANDSHAKE_NOC = noc_index>
FORCE_INLINE void open_start(
    size_t router_worker_location_info_addr,
    size_t my_credit_address,
    size_t worker_teardown_addr,
    uint8_t edm_noc_x,
    uint8_t edm_noc_y) {
    const auto dest_noc_addr_coord_only = get_noc_addr(edm_noc_x, edm_noc_y, 0);

    tt::tt_fabric::EDMChannelWorkerLocationInfo* worker_location_info_ptr =
        reinterpret_cast<tt::tt_fabric::EDMChannelWorkerLocationInfo*>(router_worker_location_info_addr);

    const uint64_t dest_edm_location_info_addr =
        dest_noc_addr_coord_only | reinterpret_cast<size_t>(
                                       router_worker_location_info_addr +
                                       offsetof(tt::tt_fabric::EDMChannelWorkerLocationInfo, worker_semaphore_address));
    // write the address of our local copy of read counter (that EDM is supposed to update)

    noc_inline_dw_write<InlineWriteDst::L1, posted>(
        dest_edm_location_info_addr, reinterpret_cast<size_t>(my_credit_address), 0xf, WORKER_HANDSHAKE_NOC);
    const uint64_t edm_teardown_semaphore_address_address =
        dest_noc_addr_coord_only |
        reinterpret_cast<uint64_t>(&(worker_location_info_ptr->worker_teardown_semaphore_address));
    // Write our local teardown ack address to EDM
    noc_inline_dw_write<InlineWriteDst::L1, posted>(
        edm_teardown_semaphore_address_address,
        reinterpret_cast<size_t>(worker_teardown_addr),
        0xf,
        WORKER_HANDSHAKE_NOC);
    // Write out core noc-xy coord to EDM
    const uint64_t connection_worker_xy_address =
        dest_noc_addr_coord_only | reinterpret_cast<uint64_t>(&(worker_location_info_ptr->worker_xy));
    noc_inline_dw_write<InlineWriteDst::L1, posted>(
        connection_worker_xy_address, WorkerXY(my_x[0], my_y[0]).to_uint32(), 0xf, WORKER_HANDSHAKE_NOC);
}

template <bool posted = false, uint8_t WORKER_HANDSHAKE_NOC = noc_index>
void open_finish(
    size_t edm_connection_handshake_l1_addr,
    volatile uint32_t* worker_teardown_addr,
    uint8_t edm_noc_x,
    uint8_t edm_noc_y) {
    const auto edm_connection_handshake_noc_addr = get_noc_addr(edm_noc_x, edm_noc_y, edm_connection_handshake_l1_addr);
    // Order here is important
    // We need to write our read counter value to the register before we signal the EDM
    // As EDM will potentially increment the register as well if it sees a connection
    // active. As a result, we barrier here to avoid that scenario.
    noc_async_read_barrier(WORKER_HANDSHAKE_NOC);

    noc_inline_dw_write<InlineWriteDst::L1, posted>(
        edm_connection_handshake_noc_addr, connection_interface::open_connection_value, 0xf, WORKER_HANDSHAKE_NOC);
    *worker_teardown_addr = 0;
}

namespace worker {
void close_start(
    uint32_t last_slot_index,
    size_t edm_connection_handshake_l1_addr,
    size_t downstream_router_last_buffer_slot_index_addr,
    uint8_t edm_noc_x,
    uint8_t edm_noc_y) {
    const auto dest_noc_addr_coord_only = get_noc_addr(edm_noc_x, edm_noc_y, 0);

    // buffer index stored at location after handshake addr
    const uint64_t remote_buffer_index_addr =
        dest_noc_addr_coord_only | downstream_router_last_buffer_slot_index_addr /*edm_copy_of_wr_counter_addr*/;
    noc_inline_dw_write(remote_buffer_index_addr, last_slot_index /*this->get_buffer_slot_index()*/);

    const uint64_t dest_edm_connection_state_addr = dest_noc_addr_coord_only | edm_connection_handshake_l1_addr;
    noc_inline_dw_write(dest_edm_connection_state_addr, connection_interface::close_connection_request_value);
}
}  // namespace worker

}  // namespace tt::tt_fabric::connection
