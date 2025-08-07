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
static constexpr uint32_t max_fabric_addrgen_payload_size = 4532;

FORCE_INLINE void validate_max_payload_size(uint32_t payload_size) {
    ASSERT((page_size > max_fabric_addrgen_payload_size));
    if ((payload_size > max_fabric_addrgen_payload_size)) {
        WAYPOINT("HUNG");
        // hang to prompt investigation
        while (1) {
        }
    }
}

template <typename AddrGenType>
FORCE_INLINE void to_noc_unicast_write(
    volatile PACKET_HEADER_TYPE* pkt_hdr, const uint32_t id, const AddrGenType& d, uint32_t offset = 0) {
    auto noc_address = d.get_noc_addr(id, offset, edm_to_local_chip_noc);
    auto page_size = addrgen_detail::get_page_size(d);
    pkt_hdr->to_noc_unicast_write(NocUnicastCommandHeader{noc_address}, page_size);

    validate_max_payload_size(page_size);
}

template <typename AddrGenType>
FORCE_INLINE void to_noc_fused_unicast_write_atomic_inc(
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    const NocUnicastAtomicIncCommandHeader& atomic_inc_spec,
    const uint32_t id,
    const AddrGenType& d,
    uint32_t offset = 0) {
    auto page_size = addrgen_detail::get_page_size(d);
    auto noc_address = d.get_noc_addr(id, offset, edm_to_local_chip_noc);

    pkt_hdr->to_noc_fused_unicast_write_atomic_inc(
        NocUnicastAtomicIncFusedCommandHeader(
            noc_address, atomic_inc_spec.noc_address, atomic_inc_spec.val, atomic_inc_spec.wrap, atomic_inc_spec.flush),
        page_size);

    validate_max_payload_size(page_size);
}

template <typename AddrGenType>
FORCE_INLINE void to_noc_unicast_scatter_write(
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    const uint32_t id0,
    const uint32_t id1,
    const AddrGenType& d,
    uint32_t offset0 = 0,
    uint32_t offset1 = 0) {
    auto page_size = addrgen_detail::get_page_size(d);
    auto payload_size = page_size * 2;

    auto noc_address0 = d.get_noc_addr(id0, offset0, edm_to_local_chip_noc);
    auto noc_address1 = d.get_noc_addr(id1, offset1, edm_to_local_chip_noc);

    pkt_hdr->to_noc_unicast_scatter_write(
        NocUnicastScatterCommandHeader({{noc_address0, noc_address1}, static_cast<uint16_t>(page_size)}), payload_size);

    validate_max_payload_size(payload_size);
}

}  // namespace linear

};  // namespace tt::tt_fabric
