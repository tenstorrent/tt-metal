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
// (the Quasar overlay address generator walks pages in sequence) without a plain get_noc_addr() call ever advancing
// or disturbing that walk.
//
// Where the hardware path doesn't apply (other arches, ATT off, a layout with no hardware recipe yet, or device
// tables the recipe can't walk) transfer_noc_addr() returns exactly what get_noc_addr() would, through the un-noted
// detail::transfer_noc_addr() (internal/tensor/transfer_noc_addr.h): the NoC API that called it records the exact
// access for op-to-op R/W inference, so the software address must not also record a read and write.
// (detail::transfer_noc_addr() itself stays software-only: the legacy free functions use it without a direction.)
//
// Included by api/tensor/noc_traits.h, the only caller.

#include <cstdint>

#include "api/tensor/page.h"
#include "api/tensor/tensor_accessor.h"

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM) && defined(NOC_ATT_ENABLED) && !defined(TT_TA_ADDRGEN_DISABLE)
#define TT_TA_ADDRGEN_ACTIVE 1
#endif

// Push: on a walk hit, the address generator writes the remote address straight into the command buffer the NoC API
// issues on, instead of returning it (Noc::async_read / async_write, api/tensor/noc_traits.h). Off where software must
// see every address: the address trace, and watcher NoC sanitizing. TT_TA_ADDRGEN_NO_PUSH turns it off.
#if defined(TT_TA_ADDRGEN_ACTIVE) && !defined(TT_TA_ADDRGEN_NO_PUSH) && !defined(TT_TA_ADDRGEN_TRACE) && \
    !(defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_NOC_SANITIZE)) && !defined(PROFILE_NOC_EVENTS)
#define TT_TA_ADDRGEN_PUSH 1
#endif

namespace tensor_accessor::detail {

#if defined(TT_TA_ADDRGEN_STATS)
// Per-DM-core counts of how each transfer address was produced. Test instrumentation only (TT_TA_ADDRGEN_STATS),
// so production kernels pay nothing.
struct TransferStats {
    uint32_t hw = 0;              // hardware address generator
    uint32_t sw_ineligible = 0;   // layout has a hardware recipe, but this device's bank tables don't fit it
    uint32_t sw_unsupported = 0;  // no hardware recipe for this layout/accessor (or the hardware path is off)
    uint32_t seeks = 0;           // of the hw addresses, how many needed the walk (re)programmed in software first
    uint32_t skips = 0;           // of the hw addresses, how many the hardware skipped forward to (no reprogramming)
    uint32_t restores = 0;     // of the hw addresses, how many first reloaded a parked walk (spilling the side's walk)
    uint32_t write_seeks = 0;  // seeks / restores / fallbacks of the write direction only (reads = totals minus these)
    uint32_t write_restores = 0;
    uint32_t fallbacks =
        0;  // software by walk policy: no side for the walk (NO_SPILL), or the request broke the stream
    uint32_t write_fallbacks = 0;
    uint32_t pushes = 0;  // of the hw addresses, how many went straight into the command buffer (push)
    // Cycles spent in the reload path, summed over reloads (rdcycle): read back the side's position, exchange the walk
    // states, write the parked walk's programming and position, serve the request.
    uint32_t reload_save_cycles = 0;
    uint32_t reload_swap_cycles = 0;
    uint32_t reload_restore_cycles = 0;
    uint32_t reload_serve_cycles = 0;
};
inline thread_local TransferStats transfer_stats{};
#define TT_TA_ADDRGEN_COUNT(field) (++::tensor_accessor::detail::transfer_stats.field)
#else
#define TT_TA_ADDRGEN_COUNT(field) ((void)0)
#endif

#if defined(TT_TA_ADDRGEN_STATS)
// Count what a hardware transfer cost (see PopInfo): seeks, skips, restores; writes also in their own counters.
#define TT_TA_ADDRGEN_COUNT_HW(Dir, info)                \
    do {                                                 \
        TT_TA_ADDRGEN_COUNT(hw);                         \
        if ((info).seeked) {                             \
            TT_TA_ADDRGEN_COUNT(seeks);                  \
            if constexpr ((Dir) == TransferDir::Write) { \
                TT_TA_ADDRGEN_COUNT(write_seeks);        \
            }                                            \
        }                                                \
        if ((info).skipped) {                            \
            TT_TA_ADDRGEN_COUNT(skips);                  \
        }                                                \
        if ((info).pushed) {                             \
            TT_TA_ADDRGEN_COUNT(pushes);                 \
        }                                                \
        if ((info).restored) {                           \
            TT_TA_ADDRGEN_COUNT(restores);               \
            if constexpr ((Dir) == TransferDir::Write) { \
                TT_TA_ADDRGEN_COUNT(write_restores);     \
            }                                            \
        }                                                \
    } while (0)

// Count a request the hardware path declined: policy (fallbacks) or a device whose banks it can't walk (ineligible).
#define TT_TA_ADDRGEN_COUNT_SW(Dir, info)                \
    do {                                                 \
        if ((info).fallback) {                           \
            TT_TA_ADDRGEN_COUNT(fallbacks);              \
            if constexpr ((Dir) == TransferDir::Write) { \
                TT_TA_ADDRGEN_COUNT(write_fallbacks);    \
            }                                            \
        } else {                                         \
            TT_TA_ADDRGEN_COUNT(sw_ineligible);          \
        }                                                \
    } while (0)
#else
// PopInfo is empty without the instrumentation (tensor_accessor_addrgen.h).
#define TT_TA_ADDRGEN_COUNT_HW(Dir, info) ((void)(info))
#define TT_TA_ADDRGEN_COUNT_SW(Dir, info) ((void)(info))
#endif

}  // namespace tensor_accessor::detail

#if defined(TT_TA_ADDRGEN_ACTIVE)
#include "internal/tt-2xx/quasar/tensor/tensor_accessor_addrgen.h"
#endif

// Debug: print every hardware transfer address next to the software one (TT_TA_ADDRGEN_TRACE; needs device print).
#if defined(TT_TA_ADDRGEN_ACTIVE) && defined(TT_TA_ADDRGEN_TRACE)
#include "api/debug/device_print.h"
#define TT_TA_ADDRGEN_TRACE_ADDR(dir, index, hw, sw, info)                           \
    DEVICE_PRINT(                                                                    \
        "addrgen dir {} page {} hw 0x{:x} sw 0x{:x} seek {} skip {} restore {}{}\n", \
        static_cast<uint32_t>(dir),                                                  \
        index,                                                                       \
        hw,                                                                          \
        sw,                                                                          \
        static_cast<uint32_t>((info).seeked),                                        \
        static_cast<uint32_t>((info).skipped),                                       \
        static_cast<uint32_t>((info).restored),                                      \
        (hw) == (sw) ? "" : "  MISMATCH")
#else
#define TT_TA_ADDRGEN_TRACE_ADDR(dir, index, hw, sw, info) ((void)0)
#endif

namespace tensor_accessor {

// Transfer address of page `page_id` (+ `offset` bytes) of `accessor`. Accessor may be any type with
// get_noc_addr(page_id, offset, noc) -- e.g. something wrapped by AbstractTensorAccessorWrapper.
//
// MayPush (the Noc::async_read / async_write issue path only): the result may be kAddrInCmdBuf instead of an address,
// meaning the remote address is already in the direction's command buffer, which the caller then issues without
// writing it.
// (MayPush defaults to false in the declaration in tensor_accessor.h.)
template <TransferDir Dir, bool MayPush, typename Accessor>
inline uint64_t transfer_noc_addr(const Accessor& accessor, uint32_t page_id, uint32_t offset, uint8_t noc) {
#if defined(TT_TA_ADDRGEN_ACTIVE)
    if constexpr (tt_addrgen::has_hw_recipe<Accessor>) {
        uint64_t hw_addr;
        tt_addrgen::PopInfo info;
        if (tt_addrgen::try_transfer_noc_addr<Dir, MayPush>(accessor, page_id, offset, noc, hw_addr, info)) {
            TT_TA_ADDRGEN_TRACE_ADDR(
                Dir, page_id, hw_addr, detail::transfer_noc_addr(accessor, page_id, offset, noc), info);
            TT_TA_ADDRGEN_COUNT_HW(Dir, info);
            return hw_addr;
        }
        TT_TA_ADDRGEN_COUNT_SW(Dir, info);
        return detail::transfer_noc_addr(accessor, page_id, offset, noc);
    }
#endif
    TT_TA_ADDRGEN_COUNT(sw_unsupported);
    return detail::transfer_noc_addr(accessor, page_id, offset, noc);
}

// Transfer address of an iterator page. Same decision as above; the software fallback reuses the address the
// iterator already computed instead of recomputing it.
template <TransferDir Dir, bool MayPush = false, typename Accessor>
inline uint64_t transfer_noc_addr(const AccessorPage<Accessor>& page, uint32_t offset, uint8_t noc) {
#if defined(TT_TA_ADDRGEN_ACTIVE)
    if constexpr (tt_addrgen::has_hw_recipe<Accessor>) {
        uint64_t hw_addr;
        tt_addrgen::PopInfo info;
        if (tt_addrgen::try_transfer_noc_addr<Dir, MayPush>(
                page.accessor(), page.page_id(), offset, noc, hw_addr, info)) {
            TT_TA_ADDRGEN_COUNT_HW(Dir, info);
            return hw_addr;
        }
        TT_TA_ADDRGEN_COUNT_SW(Dir, info);
        return static_cast<const Page&>(page).noc_addr() + offset;
    }
#endif
    TT_TA_ADDRGEN_COUNT(sw_unsupported);
    return static_cast<const Page&>(page).noc_addr() + offset;
}

// Transfer address of a shard_pages() page. The hardware walk follows the shard's storage order (page_in_shard),
// which is what shard_pages() yields; the software fallback reuses the iterator's address.
template <TransferDir Dir, bool MayPush = false, typename Accessor>
inline uint64_t transfer_noc_addr(const ShardPage<Accessor>& page, uint32_t offset, uint8_t noc) {
#if defined(TT_TA_ADDRGEN_ACTIVE)
    if constexpr (tt_addrgen::has_hw_recipe<Accessor>) {
        uint64_t hw_addr;
        tt_addrgen::PopInfo info;
        if (tt_addrgen::try_transfer_shard_page_noc_addr<Dir, MayPush>(
                page.accessor(), page.shard_id(), page.page_in_shard(), offset, noc, hw_addr, info)) {
            TT_TA_ADDRGEN_COUNT_HW(Dir, info);
            return hw_addr;
        }
        TT_TA_ADDRGEN_COUNT_SW(Dir, info);
        return static_cast<const Page&>(page).noc_addr() + offset;
    }
#endif
    TT_TA_ADDRGEN_COUNT(sw_unsupported);
    return static_cast<const Page&>(page).noc_addr() + offset;
}

// Transfer address of a whole shard, or `offset` bytes into it (ShardView). Consecutive transfers into the same shard
// reuse the base the hardware produced for it (counted as hw, not as a seek).
template <TransferDir Dir, typename Accessor>
inline uint64_t transfer_shard_noc_addr(const Accessor& accessor, uint32_t shard_id, uint32_t offset, uint8_t noc) {
#if defined(TT_TA_ADDRGEN_ACTIVE)
    if constexpr (tt_addrgen::has_hw_recipe<Accessor>) {
        uint64_t hw_addr;
        tt_addrgen::PopInfo info;
        if (tt_addrgen::try_transfer_shard_noc_addr<Dir>(accessor, shard_id, offset, noc, hw_addr, info)) {
            TT_TA_ADDRGEN_COUNT_HW(Dir, info);
            return hw_addr;
        }
        TT_TA_ADDRGEN_COUNT_SW(Dir, info);
        return detail::transfer_shard_noc_addr(accessor, shard_id, offset, noc);
    }
#endif
    TT_TA_ADDRGEN_COUNT(sw_unsupported);
    return detail::transfer_shard_noc_addr(accessor, shard_id, offset, noc);
}

}  // namespace tensor_accessor
