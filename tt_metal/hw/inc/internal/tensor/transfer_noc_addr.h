// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "internal/tensor/binding_id.h"

template <typename DSpecT>
struct TensorAccessor;

namespace tensor_accessor::detail {
// The only way in to TensorAccessor's private, un-noted address members (it befriends this).
struct TransferAccess {
    template <typename Accessor>
    static uint64_t page(const Accessor& accessor, uint32_t page_id, uint32_t offset, uint8_t noc) {
        return accessor.transfer_noc_addr(page_id, offset, noc);
    }
    template <typename Accessor>
    static uint64_t shard(const Accessor& accessor, uint32_t shard_id, uint32_t offset, uint8_t noc) {
        return accessor.transfer_shard_noc_addr(shard_id, offset, noc);
    }
};
// The NoC address of a page (or shard), for a library transfer path that records its own exact access for op-to-op R/W
// inference (api/dataflow/buf_rw_note.h): the Noc API traits, the page iterators, the legacy page/shard free functions.
// Library-internal: a kernel that took an address from here would skip the record its access needs.
//
// TensorAccessor::get_noc_addr records its tensor as read and written, because the raw address it hands out can be used
// for anything. These return the same address without that record. Any other address generator records nothing, so for
// it this is just its get_noc_addr.
template <typename DSpec>
inline uint64_t transfer_noc_addr(
    const ::TensorAccessor<DSpec>& accessor, uint32_t page_id, uint32_t offset, uint8_t noc) {
    return TransferAccess::page(accessor, page_id, offset, noc);
}

template <typename AddrGen>
inline uint64_t transfer_noc_addr(const AddrGen& addrgen, uint32_t page_id, uint32_t offset, uint8_t noc) {
    return addrgen.get_noc_addr(page_id, offset, noc);
}

template <typename DSpec>
inline uint64_t transfer_shard_noc_addr(
    const ::TensorAccessor<DSpec>& accessor, uint32_t shard_id, uint32_t offset, uint8_t noc) {
    return TransferAccess::shard(accessor, shard_id, offset, noc);
}

// Whether the page iterators may leave a page's software address uncomputed (AccessorPage::kLazyNocAddr): when the
// Quasar address generator serves the accessor's transfers (a bound accessor, in a Quasar DM build with the ATT
// backend), the transfer doesn't need it, and AccessorPage::noc_addr() computes it if anything asks. Not for
// interleaved tensors whose banks the host found unwalkable (TT_TA_ADDRGEN_INTERLEAVED_{DRAM,L1}_SW): every transfer
// of those uses the software address, which the iterator computes incrementally.
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM) && defined(NOC_ATT_ENABLED) && !defined(TT_TA_ADDRGEN_DISABLE)
#if defined(TT_TA_ADDRGEN_INTERLEAVED_DRAM_SW)
inline constexpr bool kInterleavedDramSwOnly = true;
#else
inline constexpr bool kInterleavedDramSwOnly = false;
#endif
#if defined(TT_TA_ADDRGEN_INTERLEAVED_L1_SW)
inline constexpr bool kInterleavedL1SwOnly = true;
#else
inline constexpr bool kInterleavedL1SwOnly = false;
#endif
template <typename DSpec>
inline constexpr bool interleaved_sw_only_v =
    DSpec::is_interleaved && (DSpec::is_dram ? kInterleavedDramSwOnly : kInterleavedL1SwOnly);
template <typename Accessor, typename = void>
inline constexpr bool lazy_page_addr_v = false;
template <typename Accessor>
inline constexpr bool lazy_page_addr_v<Accessor, std::void_t<decltype(Accessor::DSpec::binding_id)>> =
    Accessor::DSpec::binding_id != NO_BINDING_ID && !interleaved_sw_only_v<typename Accessor::DSpec>;
#else
template <typename Accessor>
inline constexpr bool lazy_page_addr_v = false;
#endif

}  // namespace tensor_accessor::detail
