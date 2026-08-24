// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Exercises the TX-side packet-header validation (tt::tt_fabric::is_valid / is_valid_payload_size)
// directly, one case per check, and writes each boolean verdict to L1 for the host to check. This
// calls the predicates directly rather than through the send-path ASSERT so a corrupted header does
// not hang the device; it just records true/false. Result word layout matches the expected[] array
// in test_packet_header_validation.cpp.

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_header_validate.hpp"

using namespace tt::tt_fabric;

void kernel_main() {
    uint32_t result_addr = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr uint32_t* results = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);
    uint32_t i = 0;

    // Header storage overlay (headers are used as overlays on raw memory, not stack-constructed).
    alignas(16) uint8_t hdr_buf[256];

    // --- LowLatencyPacketHeader (1D low-latency): universal checks ---
    {
        auto* h = reinterpret_cast<LowLatencyPacketHeader*>(hdr_buf);
        h->noc_send_type = NOC_UNICAST_WRITE;
        h->payload_size_bytes = 16;
        results[i++] = is_valid(*h);  // 0: valid -> true
    }
    {
        auto* h = reinterpret_cast<LowLatencyPacketHeader*>(hdr_buf);
        h->noc_send_type = static_cast<NocSendType>(NOC_SEND_TYPE_LAST + 5);
        h->payload_size_bytes = 16;
        results[i++] = is_valid(*h);  // 1: bad noc_send_type -> false
    }
    {
        auto* h = reinterpret_cast<LowLatencyPacketHeader*>(hdr_buf);
        h->noc_send_type = NOC_UNICAST_ATOMIC_INC;  // header-only op must carry no payload
        h->payload_size_bytes = 16;
        results[i++] = is_valid(*h);  // 2: nonzero payload on header-only op -> false
    }

    // --- PacketHeader (1D dynamic): chip_send_type check ---
    {
        auto* h = reinterpret_cast<PacketHeader*>(hdr_buf);
        h->chip_send_type = CHIP_UNICAST;
        h->noc_send_type = NOC_UNICAST_WRITE;
        results[i++] = is_valid(*h);  // 3: valid -> true
    }
    {
        auto* h = reinterpret_cast<PacketHeader*>(hdr_buf);
        h->chip_send_type = static_cast<ChipSendType>(CHIP_SEND_TYPE_LAST + 5);
        h->noc_send_type = NOC_UNICAST_WRITE;
        results[i++] = is_valid(*h);  // 4: bad chip_send_type -> false
    }

    // --- HybridMeshPacketHeader (2D): hop_index bounds (the observed hang) ---
    {
        auto* h = reinterpret_cast<HybridMeshPacketHeader*>(hdr_buf);
        h->routing_fields.hop_index = 0;
        h->noc_send_type = NOC_UNICAST_WRITE;
        results[i++] = is_valid(*h);  // 5: valid -> true
    }
    {
        auto* h = reinterpret_cast<HybridMeshPacketHeader*>(hdr_buf);
        h->routing_fields.hop_index = get_max_num_hops<HybridMeshPacketHeader>::value + 100;
        h->noc_send_type = NOC_UNICAST_WRITE;
        results[i++] = is_valid(*h);  // 6: hop_index out of bounds -> false
    }
    {
        auto* h = reinterpret_cast<HybridMeshPacketHeader*>(hdr_buf);
        h->routing_fields.hop_index = 0;
        h->noc_send_type = static_cast<NocSendType>(NOC_SEND_TYPE_LAST + 5);
        results[i++] = is_valid(*h);  // 7: bad noc_send_type -> false
    }

    // --- is_valid_payload_size: header field vs actual transfer size ---
    {
        auto* h = reinterpret_cast<LowLatencyPacketHeader*>(hdr_buf);
        h->noc_send_type = NOC_UNICAST_WRITE;
        h->payload_size_bytes = 16;
        results[i++] = is_valid_payload_size(*h, 16 + sizeof(LowLatencyPacketHeader));  // 8: consistent -> true
    }
    {
        auto* h = reinterpret_cast<LowLatencyPacketHeader*>(hdr_buf);
        h->noc_send_type = NOC_UNICAST_WRITE;
        h->payload_size_bytes = 16;
        results[i++] = is_valid_payload_size(*h, 999);  // 9: inconsistent -> false
    }
}
