#pragma once

#include <cstdint>
#include <algorithm>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"

namespace tt::tt_fabric {
// Send data via fabric with automatic chunking for large payloads
// Handles payloads larger than max_packet_size by splitting into chunks
template <typename FabricSenderType, typename PacketHeaderType, typename NocCommandType>
FORCE_INLINE void fabric_write_chunked(
    FabricSenderType& fabric_sender,
    volatile PacketHeaderType* packet_header,
    uint64_t dest_noc_addr,
    uint32_t src_l1_addr,
    uint32_t total_bytes,
    uint32_t max_packet_size) {
    uint32_t remaining = total_bytes;
    uint32_t offset = 0;
    while (remaining > 0) {
        uint32_t chunk_size = (remaining < max_packet_size) ? remaining : max_packet_size;
        packet_header->to_noc_unicast_write(NocCommandType{dest_noc_addr + offset}, chunk_size);
        fabric_sender.wait_for_empty_write_slot();
        fabric_sender.send_payload_without_header_non_blocking_from_address(src_l1_addr + offset, chunk_size);
        fabric_sender.send_payload_flush_blocking_from_address(
            reinterpret_cast<uint32_t>(packet_header), sizeof(PacketHeaderType));
        offset += chunk_size;
        remaining -= chunk_size;
    }
}

// Send data via fabric with automatic chunking for large payloads
// Handles payloads larger than max_packet_size by splitting into chunks
// Caller must ensure:
//   - noc_async_write_barrier() at some point after this API
//   - L1 space: packet headers and src_l1, not touched until noc_async_write_barrier() is called at caller side
template <typename FabricSenderType, typename PacketHeaderType, typename NocCommandType>
FORCE_INLINE void fabric_write_chunked_nonblocking(
    FabricSenderType& fabric_sender,
    volatile PacketHeaderType* packet_headers,
    uint64_t dest_noc_addr,
    uint32_t src_l1_addr,
    uint32_t total_bytes,
    uint32_t max_packet_size,
    uint32_t dst_chip_id,
    uint32_t dst_mesh_id) {
    uint32_t remaining = total_bytes;
    uint32_t offset = 0;
    uint32_t header_idx = 0;
    while (remaining > 0) {
        volatile PacketHeaderType* packet_header = &packet_headers[header_idx];
        fabric_set_unicast_route(packet_header, dst_chip_id, dst_mesh_id);
        uint32_t chunk_size = (remaining < max_packet_size) ? remaining : max_packet_size;
        packet_header->to_noc_unicast_write(NocCommandType{dest_noc_addr + offset}, chunk_size);
        fabric_sender.wait_for_empty_write_slot();
        fabric_sender.send_payload_without_header_non_blocking_from_address(src_l1_addr + offset, chunk_size);
        fabric_sender.send_payload_flush_non_blocking_from_address(
            reinterpret_cast<uint32_t>(packet_header), sizeof(PacketHeaderType));
        offset += chunk_size;
        remaining -= chunk_size;
        header_idx++;
    }
}
};  // namespace tt::tt_fabric
