// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

// EDM / fabric helpers
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

using namespace tt;
using namespace tt::tt_fabric;

void kernel_main() {
    size_t idx = 0;

    const uint32_t src_l1_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t payload_bytes = get_arg_val<uint32_t>(idx++);
    const uint32_t dest_noc_lo = get_arg_val<uint32_t>(idx++);
    const uint32_t dest_noc_hi = get_arg_val<uint32_t>(idx++);
    const uint64_t dest_noc_addr = (uint64_t(dest_noc_hi) << 32) | uint64_t(dest_noc_lo);

    // Build the sender connection from the remaining rt args
    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);

    // One reusable header allocated in L1
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();
    zero_l1_buf((uint32_t*)header, sizeof(PACKET_HEADER_TYPE));

    // Program header (unicast write to a NOC address, payload size in bytes)
    header->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, payload_bytes);

    // Open, send, close
    sender.open();
    sender.wait_for_empty_write_slot();

    // push payload (without header)
    sender.send_payload_without_header_non_blocking_from_address(src_l1_addr, payload_bytes);

    // push header (this completes the packet)
    sender.send_payload_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));

    sender.close();
}
