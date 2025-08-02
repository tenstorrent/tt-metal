// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/hw/inc/dataflow_api_addrgen.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp"

namespace tt::tt_fabric {

namespace linear {

namespace addrgen_detail {

template <bool DRAM>
uint32_t get_page_size(const InterleavedAddrGen<DRAM>& s) {
    return s.aligned_page_size;
}

template <bool DRAM>
uint32_t get_page_size(const InterleavedAddrGenFast<DRAM>& s) {
    return s.page_size;
}
}  // namespace addrgen_detail

// Placeholder max page size for the addrgen until the page size is properly visible by the worker
// https://github.com/tenstorrent/tt-metal/issues/25966
static constexpr uint32_t max_fabric_addrgen_page_size = 2048;

FORCE_INLINE void validate_max_page_size(uint32_t page_size) {
    ASSERT((page_size > max_fabric_addrgen_page_size));
    if ((page_size > max_fabric_addrgen_page_size)) {
        WAYPOINT("HUNG");
        // hang to promopt investigation
        while (1) {
        }
    }
}

template <typename AddrGenType>
FORCE_INLINE void to_noc_unicast_write(
    volatile PACKET_HEADER_TYPE* pkt_hdr, const uint32_t id, const AddrGenType& d, uint32_t offset = 0) {
    pkt_hdr->noc_send_type = NOC_UNICAST_WRITE;

    pkt_hdr->command_fields.unicast_write.noc_address = d.get_noc_addr(id, offset, edm_to_local_chip_noc);
    auto page_size = addrgen_detail::get_page_size(d);
    pkt_hdr->payload_size_bytes = page_size;

    validate_max_page_size(page_size);
}

template <typename AddrGenType>
FORCE_INLINE void to_noc_fused_unicast_write_atomic_inc(
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    const NocUnicastAtomicIncCommandHeader& atomic_inc_spec,
    const uint32_t id,
    const AddrGenType& d,
    uint32_t offset = 0) {
    pkt_hdr->noc_send_type = NOC_FUSED_UNICAST_ATOMIC_INC;

    auto page_size = addrgen_detail::get_page_size(d, offset, edm_to_local_chip_noc);
    pkt_hdr->payload_size_bytes = page_size;

    pkt_hdr->command_fields.unicast_seminc_fused.noc_address = d.get_noc_addr(id, offset, edm_to_local_chip_noc);
    pkt_hdr->command_fields.unicast_seminc_fused.semaphore_noc_address = atomic_inc_spec.noc_address;
    pkt_hdr->command_fields.unicast_seminc_fused.val = atomic_inc_spec.val;
    pkt_hdr->command_fields.unicast_seminc_fused.wrap = atomic_inc_spec.wrap;
    pkt_hdr->command_fields.unicast_seminc_fused.flush = atomic_inc_spec.flush;

    validate_max_page_size(page_size);
}

template <typename AddrGenType>
FORCE_INLINE void to_noc_unicast_scatter_write(
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    const uint32_t id0,
    const uint32_t id1,
    const AddrGenType& d,
    uint32_t offset0 = 0,
    uint32_t offset1 = 0) {
    pkt_hdr->noc_send_type = NOC_UNICAST_SCATTER_WRITE;

    auto page_size = addrgen_detail::get_page_size(d);
    auto payload_size = page_size * 2;
    pkt_hdr->payload_size_bytes = payload_size;

    pkt_hdr->command_fields.unicast_scatter_write.noc_address[0] = d.get_noc_addr(id0, offset0, edm_to_local_chip_noc);
    pkt_hdr->command_fields.unicast_scatter_write.noc_address[1] = d.get_noc_addr(id1, offset1, edm_to_local_chip_noc);
    pkt_hdr->command_fields.unicast_scatter_write.chunk_size[0] = page_size;

    validate_max_page_size(payload_size);
}

}  // namespace linear

};  // namespace tt::tt_fabric
