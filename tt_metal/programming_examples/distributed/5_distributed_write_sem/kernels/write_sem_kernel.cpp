// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"  // required in all kernels using DPRINT
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
using namespace tt::tt_fabric::linear::experimental;
void kernel_main() {
    size_t arg_idx = 0;
    uint32_t device_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t global_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);

    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION>(
            arg_idx);

    volatile tt_l1_ptr uint32_t* global_semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr);

    // 18,18 corresponds to the virtual coord of Worker Core 0,0
    uint64_t semaphore_noc_addr = safe_get_noc_addr(18, 18, global_semaphore_addr, 0);
    DPRINT << "Device " << device_id << " semaphore NOC address: " << semaphore_noc_addr << "\n";

    auto pkt_semaphore_hdr = PacketHeaderPool::allocate_header();
    pkt_semaphore_hdr->to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{semaphore_noc_addr, static_cast<uint32_t>(1)});  // increment 1
    // Perform local semaphore increment
    // noc_semaphore_inc(semaphore_noc_addr, 1);

    // Perform remote semaphore increment
    tt::tt_fabric::WorkerToFabricEdmSender cur_connection;
    if (device_id == 0) {
        cur_connection = fabric_connection.get_forward_connection();
    } else {
        cur_connection = fabric_connection.get_backward_connection();
    }
    DPRINT << device_id << " Waiting for empty slot on connection \n";
    cur_connection.wait_for_empty_write_slot();
    DPRINT << device_id << " Got empty slot on connection. Sending " << sizeof(PACKET_HEADER_TYPE) << " bytes \n";
    fabric_set_unicast_route<false>(pkt_semaphore_hdr, 1);
    cur_connection.send_payload_flush_blocking_from_address((uint32_t)pkt_semaphore_hdr, sizeof(PACKET_HEADER_TYPE));
    noc_semaphore_wait_min(global_semaphore_ptr, 1);
    DPRINT << "Hello from Device " << device_id << ", connected: " << (int)fabric_connection.is_logically_connected()
           << "\n";
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }
}
