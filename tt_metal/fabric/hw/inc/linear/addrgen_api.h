// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/hw/inc/dataflow_api_addrgen.h"
#include "tt_metal/hw/inc/accessor/tensor_accessor.h"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

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

template <typename ShardingInfoType>
uint32_t get_page_size(const experimental::ShardedAddrGen<ShardingInfoType>& d) {
    return d.CONSTANT_ARGS.page_size_jump;
}

template <typename DSpec>
uint32_t get_page_size(const TensorAccessor<DSpec>& d) {
    return d.page_size;
}

template <typename AddrGenType>
FORCE_INLINE uint64_t get_noc_address(const AddrGenType& d, const uint32_t id, uint32_t offset = 0) {
    uint64_t noc_address = d.get_noc_addr(id, offset, edm_to_local_chip_noc);
#if defined(ARCH_WORMHOLE)
    // We do this for 2 reasons:
    // 1. Wormhole doesn't support virtual coordinates for DRAM cores (and we could be writing to one)
    // 2. We want to write to the best bank for the fabric writer noc, so we'd prefer to get the noc
    // coordinate for that.
    // However, the fabric APIs canonically expect coordinates in "noc0" system, so we need to flip
    // them to noc0
    //
    // A little less efficient, but:
    // a) cleaner
    // b) less blast-radius (more incremental change)
    // c) compiler may see the redundant transformation after inlinine
    auto noc_address_components = get_noc_address_components(noc_address);
    auto noc_addr = safe_get_noc_addr(
        noc_address_components.first.x,
        noc_address_components.first.y,
        noc_address_components.second,
        edm_to_local_chip_noc);
    noc_address = noc_addr;
#endif
    return noc_address;
}

}  // namespace addrgen_detail

// Placeholder max page size for the addrgen until the page size is properly visible by the worker
// https://github.com/tenstorrent/tt-metal/issues/25966
static constexpr uint32_t max_fabric_addrgen_payload_size = 4532;

FORCE_INLINE void validate_max_payload_size(uint32_t payload_size) {
    ASSERT((payload_size <= max_fabric_addrgen_payload_size));
    if ((payload_size > max_fabric_addrgen_payload_size)) {
        WAYPOINT("HUNG");
        // hang to prompt investigation
        while (1) {
        }
    }
}

template <typename AddrGenType>
FORCE_INLINE void to_noc_unicast_write(
    uint32_t packet_payload_size,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    const uint32_t id,
    const AddrGenType& d,
    uint32_t offset = 0) {
    auto noc_address = addrgen_detail::get_noc_address(d, id, offset);
    pkt_hdr->to_noc_unicast_write(NocUnicastCommandHeader{noc_address}, packet_payload_size);
    validate_max_payload_size(packet_payload_size);
}

template <typename AddrGenType>
FORCE_INLINE void to_noc_unicast_write(
    volatile PACKET_HEADER_TYPE* pkt_hdr, const uint32_t id, const AddrGenType& d, uint32_t offset = 0) {
    auto page_size = addrgen_detail::get_page_size(d);
    to_noc_unicast_write(page_size, pkt_hdr, id, d, offset);
}

template <typename AddrGenType>
FORCE_INLINE void to_noc_fused_unicast_write_atomic_inc(
    uint32_t page_size,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    const NocUnicastAtomicIncCommandHeader& atomic_inc_spec,
    const uint32_t id,
    const AddrGenType& d,
    uint32_t offset = 0) {
    auto noc_address = addrgen_detail::get_noc_address(d, id, offset);

    pkt_hdr->to_noc_fused_unicast_write_atomic_inc(
        NocUnicastAtomicIncFusedCommandHeader(
            noc_address, atomic_inc_spec.noc_address, atomic_inc_spec.val, atomic_inc_spec.wrap, atomic_inc_spec.flush),
        page_size);

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
    to_noc_fused_unicast_write_atomic_inc(page_size, pkt_hdr, atomic_inc_spec, id, d, offset);
}

template <typename AddrGenType>
FORCE_INLINE void to_noc_unicast_scatter_write(
    uint32_t page_size,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    const uint32_t id0,
    const uint32_t id1,
    const AddrGenType& d,
    uint32_t offset0 = 0,
    uint32_t offset1 = 0) {
    auto payload_size = page_size * 2;

    auto noc_address0 = addrgen_detail::get_noc_address(d, id0, offset0);
    auto noc_address1 = addrgen_detail::get_noc_address(d, id1, offset1);

    pkt_hdr->to_noc_unicast_scatter_write(
        NocUnicastScatterCommandHeader({{noc_address0, noc_address1}, static_cast<uint16_t>(page_size)}), payload_size);

    validate_max_payload_size(payload_size);
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
    to_noc_unicast_scatter_write(page_size, pkt_hdr, id0, id1, d, offset0, offset1);
}

}  // namespace linear

};  // namespace tt::tt_fabric
