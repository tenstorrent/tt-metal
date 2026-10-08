// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Addresses for NoC *transfers* of TensorAccessor pages.
//
// get_noc_addr() is a pure function: anyone may call it, for any purpose (computing an address to store, to
// multicast, to compare), in any order. The NoC transfer path instead asks for addresses through
// transfer_noc_addr(), which is only called from the noc_traits_t specializations -- i.e. only when the address is
// about to be handed to a NoC transaction. That separation lets the transfer path be backed by stateful hardware
// (on Quasar, the address-generator sequencer, which serves pages from programmed sequences) without a plain
// get_noc_addr() call ever advancing or disturbing a sequence.
//
// Where the hardware path doesn't serve a transfer (other arches, ATT off, an accessor without a binding id, or a
// request the sequencer declines) transfer_noc_addr() returns exactly what get_noc_addr() would, through the un-noted
// detail::transfer_noc_addr() (internal/tensor/transfer_noc_addr.h): the NoC API that called it records the exact
// access for op-to-op R/W inference, so the software address must not also record a read and write.
// (detail::transfer_noc_addr() itself stays software-only: the legacy free functions use it without a direction.)
//
// Included by api/tensor/noc_traits.h, the only caller.

#include <cstdint>

#include "api/tensor/page.h"
#include "api/tensor/tensor_accessor.h"

// The hardware path: Quasar DM cores with the ATT address backend. TT_TA_ADDRGEN_DISABLE (a per-kernel define) compiles
// it out so every transfer address is software; it exists only to build the software baseline of the address-generator
// microbenchmarks (TensorAccessorAddrgenPerf, TensorAccessorAddrgenShardedPerf) and is not set in production.
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM) && defined(NOC_ATT_ENABLED) && !defined(TT_TA_ADDRGEN_DISABLE)
#define TT_TA_ADDRGEN_ACTIVE 1
#endif

// Push: on a sequence hit, the address generator writes the remote address straight into the command buffer the NoC API
// issues on, instead of returning it (Noc::async_read / async_write, api/tensor/noc_traits.h). On wherever the hardware
// path is; TT_TA_ADDRGEN_NO_PUSH turns it off (every hardware address is then popped back to the RISC-V).
#if defined(TT_TA_ADDRGEN_ACTIVE) && !defined(TT_TA_ADDRGEN_NO_PUSH)
#define TT_TA_ADDRGEN_PUSH 1
#endif

#if defined(TT_TA_ADDRGEN_ACTIVE)
#include "internal/tt-2xx/quasar/tensor/addrgen_sequencer.h"
#endif

namespace tensor_accessor::detail {

#if defined(TT_TA_ADDRGEN_STATS)
// Per-DM-core counts of the transfer addresses the hardware produced, so tests can check that the hardware (not the
// software fallback, whose addresses are just as correct) served them. Test instrumentation only.
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

// Transfer address of page `page_id` (+ `offset` bytes) of `accessor`. Accessor may be any type with
// get_noc_addr(page_id, offset, noc) -- e.g. something wrapped by AbstractTensorAccessorWrapper.
//
// MayPush (the Noc::async_read / async_write issue path only): the result may be kAddrPushed instead of an address,
// meaning the remote address is already in the direction's command buffer, which the caller then issues without
// writing it.
// (MayPush defaults to false in the declaration in tensor_accessor.h.)
template <TransferDir Dir, bool MayPush, typename Accessor>
inline uint64_t transfer_noc_addr(const Accessor& accessor, uint32_t page_id, uint32_t offset, uint8_t noc) {
#if defined(TT_TA_ADDRGEN_ACTIVE)
    if constexpr (tt_addrgen::has_hw_recipe<Accessor>) {
        uint64_t addr;
        if (tt_addrgen::try_transfer_noc_addr<Dir, MayPush>(accessor, page_id, offset, noc, addr)) {
            TT_TA_ADDRGEN_COUNT_HW(addr);
            return addr;
        }
    }
#endif
    return detail::transfer_noc_addr(accessor, page_id, offset, noc);
}

// Transfer address of an iterator page. Same decision as above; the software fallback is the page's own address.
template <TransferDir Dir, bool MayPush = false, typename Accessor>
inline uint64_t transfer_noc_addr(const AccessorPage<Accessor>& page, uint32_t offset, uint8_t noc) {
#if defined(TT_TA_ADDRGEN_ACTIVE)
    if constexpr (tt_addrgen::has_hw_recipe<Accessor>) {
        uint64_t addr;
        if (tt_addrgen::try_transfer_noc_addr<Dir, MayPush>(page.accessor(), page.page_id(), offset, noc, addr)) {
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
inline uint64_t transfer_noc_addr(const ShardPage<Accessor>& page, uint32_t offset, uint8_t noc) {
#if defined(TT_TA_ADDRGEN_ACTIVE)
    if constexpr (tt_addrgen::has_hw_recipe<Accessor>) {
        uint64_t addr;
        if (tt_addrgen::try_transfer_shard_page_noc_addr<Dir, MayPush>(
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
inline uint64_t transfer_shard_noc_addr(const Accessor& accessor, uint32_t shard_id, uint32_t offset, uint8_t noc) {
#if defined(TT_TA_ADDRGEN_ACTIVE)
    if constexpr (tt_addrgen::has_hw_recipe<Accessor>) {
        uint64_t addr;
        if (tt_addrgen::try_transfer_shard_noc_addr<Dir>(accessor, shard_id, offset, noc, addr)) {
            TT_TA_ADDRGEN_COUNT_HW(addr);
            return addr;
        }
    }
#endif
    return detail::transfer_shard_noc_addr(accessor, shard_id, offset, noc);
}

}  // namespace tensor_accessor
