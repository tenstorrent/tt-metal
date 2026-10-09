// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This file includes helpers to generate the remote address for  TensorAccessor endpoints and iterators.
// Each helper tries to invoke the HW AddrGen with a fallback to SW computed address.
//
// Note: The TensorAccessor public address getters do not take this path because they may not necessarily
// be requesting addresses in the sequence that the HW AddrGen is programmed for. These getters are soon to be removed.

#include <cstdint>

#include "api/tensor/page.h"
#include "api/tensor/tensor_accessor.h"

// TT_TA_ADDRGEN_DISABLE (a per-test kernel define) compiles the hardware path out so every transfer address is software
// to build the software baseline of the AddrGen microbenchmarks
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM) && defined(NOC_ATT_ENABLED) && !defined(TT_TA_ADDRGEN_DISABLE)
#define TT_TA_ADDRGEN_ACTIVE 1
#endif

#if defined(TT_TA_ADDRGEN_ACTIVE)
#include "internal/tt-2xx/quasar/tensor/addrgen_sequencer.h"
#endif

namespace tensor_accessor::detail {

#if defined(TT_TA_ADDRGEN_STATS)
// Test-instrumentation only: Per-DM-core counts of the addresses the hardware produced
struct TransferStats {
    uint32_t hw = 0;      // addresses from the address generator
    uint32_t pushes = 0;  // of those, how many went straight into the command buffer
};
inline thread_local TransferStats transfer_stats{};
#define TT_TA_ADDRGEN_COUNT_HW(addr)                            \
    do {                                                        \
        ++::tensor_accessor::detail::transfer_stats.hw;         \
        if ((addr) == ::tt_addrgen::kAddrPushed) {              \
            ++::tensor_accessor::detail::transfer_stats.pushes; \
        }                                                       \
    } while (0)
#else
#define TT_TA_ADDRGEN_COUNT_HW(addr) ((void)0)
#endif

}  // namespace tensor_accessor::detail

namespace tensor_accessor {

// Transfer address of page at page_id (+ offset bytes) of accessor. Accessor may be any type with
// get_noc_addr(page_id, offset, noc) - e.g. something wrapped by AbstractTensorAccessorWrapper.
//
// MayPush (true in the Noc::async_read/write issue path only): the result may be kAddrPushed instead of an address,
// meaning the remote address is already in the direction's command buffer.
template <TransferDir Dir, bool MayPush, typename Accessor>
inline __attribute__((always_inline)) uint64_t
generated_noc_addr(const Accessor& accessor, uint32_t page_id, uint32_t offset, uint8_t noc) {
#if defined(TT_TA_ADDRGEN_ACTIVE)
    if constexpr (tt_addrgen::has_hw_recipe<Accessor>) {
        uint64_t addr;
        if (tt_addrgen::try_generate_noc_addr<Dir, MayPush>(accessor, page_id, offset, noc, addr)) {
            TT_TA_ADDRGEN_COUNT_HW(addr);
            return addr;
        }
    }
#endif
    return detail::transfer_noc_addr(accessor, page_id, offset, noc);
}

// Transfer address of an iterator page. Same decision as above; the software fallback is the page's own address.
template <TransferDir Dir, bool MayPush = false, typename Accessor>
inline __attribute__((always_inline)) uint64_t
generated_noc_addr(const AccessorPage<Accessor>& page, uint32_t offset, uint8_t noc) {
#if defined(TT_TA_ADDRGEN_ACTIVE)
    if constexpr (tt_addrgen::has_hw_recipe<Accessor>) {
        uint64_t addr;
        if (tt_addrgen::try_generate_noc_addr<Dir, MayPush>(page.accessor(), page.page_id(), offset, noc, addr)) {
            TT_TA_ADDRGEN_COUNT_HW(addr);
            return addr;
        }
    }
#endif
    return page.sw_noc_addr() + offset;
}

// Transfer address of a shard_pages() page: the sequence follows the shard's storage order (page_in_shard), which is
// what shard_pages() yields.
template <TransferDir Dir, bool MayPush = false, typename Accessor>
inline __attribute__((always_inline)) uint64_t
generated_noc_addr(const ShardPage<Accessor>& page, uint32_t offset, uint8_t noc) {
#if defined(TT_TA_ADDRGEN_ACTIVE)
    if constexpr (tt_addrgen::has_hw_recipe<Accessor>) {
        uint64_t addr;
        if (tt_addrgen::try_generate_shard_page_noc_addr<Dir, MayPush>(
                page.accessor(), page.shard_id(), page.page_in_shard(), offset, noc, addr)) {
            TT_TA_ADDRGEN_COUNT_HW(addr);
            return addr;
        }
    }
#endif
    return page.sw_noc_addr() + offset;
}

// Transfer address of a whole shard, or `offset` bytes into it (ShardView). Consecutive transfers into the same shard
// reuse the base the hardware produced for it.
template <TransferDir Dir, typename Accessor>
inline __attribute__((always_inline)) uint64_t
generated_shard_noc_addr(const Accessor& accessor, uint32_t shard_id, uint32_t offset, uint8_t noc) {
#if defined(TT_TA_ADDRGEN_ACTIVE)
    if constexpr (tt_addrgen::has_hw_recipe<Accessor>) {
        uint64_t addr;
        if (tt_addrgen::try_generate_shard_noc_addr<Dir>(accessor, shard_id, offset, noc, addr)) {
            TT_TA_ADDRGEN_COUNT_HW(addr);
            return addr;
        }
    }
#endif
    return detail::transfer_shard_noc_addr(accessor, shard_id, offset, noc);
}

}  // namespace tensor_accessor
