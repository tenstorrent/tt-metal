// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "internal/tensor/binding_id.h"

template <typename DSpecT>
struct TensorAccessor;
class AbstractTensorAccessorWrapper;

namespace tensor_accessor::detail {
// Whether a NoC transfer reads or writes the tensor. The HW address generator has a source and a destination side;
// reads use the source side, writes the destination side.
enum class TransferDir : uint8_t { Read = 0, Write = 1 };

// The transfer address of a page: from the hardware address generator where it serves the accessor, else the software
// address. Defined in internal/tensor/generated_noc_addr.h; declared here so AbstractTensorAccessorWrapper
// (api/tensor/tensor_accessor.h) can erase it alongside get_noc_addr, and the wrapper's overload can reach the erased
// function.
template <TransferDir Dir, bool MayPush = false, typename Accessor>
inline uint64_t generated_noc_addr(const Accessor& accessor, uint32_t page_id, uint32_t offset, uint8_t noc);
template <TransferDir Dir, bool MayPush = false>
inline uint64_t generated_noc_addr(
    const AbstractTensorAccessorWrapper& wrapper, uint32_t page_id, uint32_t offset, uint8_t noc);

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
    // For library code and for kernels that still address another core's or a DRAM bank's
    // region by raw address; a kernel accessing its own local shard should hold a LocalTensorAccessor, which is noted.
    template <typename Accessor>
    static constexpr uint32_t bank_base(const Accessor& accessor) {
        return accessor.bank_base_address;
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

// The address an iterator page holds while its software address is uncomputed (see lazy_page_addr_v).
inline constexpr uint64_t kLazyNocAddr = ~0ull;

// Whether an interleaved L1 tensor's transfers use software addresses because this device's L1 banks are not one
// ascending ATT selector run: the host defines TT_TA_ADDRGEN_INTERLEAVED_L1_SW then (build_env_manager.cpp), since the
// address generator's interleaved recipe assumes bank b's selector is bank 0's + b. DRAM banks always form a run.
template <typename DSpec>
constexpr bool interleaved_uses_software() {
#if defined(TT_TA_ADDRGEN_INTERLEAVED_L1_SW)
    return DSpec::is_interleaved && !DSpec::is_dram;
#else
    return false;
#endif
}

// Whether the page iterators may leave a page's software address uncomputed (kLazyNocAddr): when the
// Quasar address generator serves the accessor's transfers (a bound accessor, in a Quasar DM build with the ATT
// backend), the transfer doesn't need it, and AccessorPage::noc_addr() computes it if anything asks.
// Same condition as TT_TA_ADDRGEN_ACTIVE (internal/tensor/generated_noc_addr.h); TT_TA_ADDRGEN_DISABLE only forces the
// software path for the microbenchmarks' baseline.
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM) && defined(NOC_ATT_ENABLED) && !defined(TT_METAL_TTSIM) && \
    !defined(TT_TA_ADDRGEN_DISABLE)
template <typename Accessor, typename = void>
inline constexpr bool lazy_page_addr_v = false;
template <typename Accessor>
inline constexpr bool lazy_page_addr_v<Accessor, std::void_t<decltype(Accessor::DSpec::binding_id)>> =
    Accessor::DSpec::binding_id != NO_BINDING_ID && !interleaved_uses_software<typename Accessor::DSpec>();
#else
template <typename Accessor>
inline constexpr bool lazy_page_addr_v = false;
#endif

}  // namespace tensor_accessor::detail
