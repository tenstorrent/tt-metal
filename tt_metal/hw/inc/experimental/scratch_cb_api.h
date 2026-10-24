// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/scratch_cb.h"

#ifndef ARCH_BLACKHOLE
#error "experimental/scratch_cb_api.h is Blackhole-only"
#endif

#ifdef COMPILE_FOR_TRISC
#include "api/compute/common_globals.h"
#include "llk_assert.h"
#if defined(TRISC_PACK) || defined(TRISC_UNPACK)
#include "experimental/2_0/llk_scratch_cb.h"
#endif
#else  // !COMPILE_FOR_TRISC
#include "api/dataflow/dataflow_api.h"
#endif

// Scratch CB (experimental): producer/consumer sync between a DM kernel and a compute kernel, without
// allocating a circular buffer.
//
// It works like the reserve/push/wait/pop calls of a CB, but only the synchronization is provided.
// You own the L1 memory: pick the address, write and read it yourself, and step through your slots
// yourself. Nothing here allocates memory or moves a read/write pointer.
//
// Who can talk to whom
//   One side is a DM kernel and the other is compute:
//     DM  -> UNPACK   (DM reserves and pushes, UNPACK waits and pops)
//     PACK -> DM      (PACK reserves and pushes, DM waits and pops)
//   On compute, reserve/push run on PACK and wait/pop run on UNPACK; on the other TRISCs they compile
//   to nothing, as with CircularBuffer.
//
// Template parameters
//   Ch        Channel, 0 or 1. Each channel is an independent producer/consumer pair, so you can run
//             two at once (e.g. channel 0 for DM -> UNPACK and channel 1 for PACK -> DM). Channel Ch
//             uses the hardware counters of CB 62 + Ch. The host does not reserve CB 62 or 63, so a
//             program that uses channel Ch must not also create CB 62 + Ch. With DFBs, slots are
//             assigned lowest-free per node, so a node may have at most 62 DFBs (relay DFBs included)
//             if it uses channel 0, or 63 if it uses only channel 1. With watcher on, DM reserve/wait
//             assert that the slot is unused.
//   Capacity  How many pages the channel can hold at once (1..65535, default 1). With Capacity 1 the
//             producer waits until the consumer has popped before writing again. A larger Capacity
//             lets the producer run ahead, as a ring of Capacity slots. Use the same Capacity in every
//             call on both sides of a channel.
//
// Function arguments
//   num_pages  How many pages to reserve/push/wait/pop, 1..Capacity. A "page" is whatever unit you
//              choose (a tile, a buffer, a message); both sides just have to agree on it. Same as
//              num_pages in cb_reserve_back/cb_push_back/cb_wait_front/cb_pop_front. More than
//              Capacity can never become free or available, so reserve/wait would spin forever.
//              Checked on DM with watcher or TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1, and on compute
//              with TT_METAL_LLK_ASSERTS=1.
//   l1_addr    Optional, default 0. Only labels the event in the sync profiler, e.g. pass the slot
//              address to see which slot a wait was stuck on. It does not select memory or affect
//              synchronization, and is compiled out when the profiler is off.
//
// Rules
//   - Each channel has exactly one producer and one consumer, and they stay the same for the whole
//     kernel launch. These calls are not a barrier across all TRISCs.
//   - Both sides must walk the slots in the same order.
//   - On DM, the slot must be done before you hand it over: a producer that filled it with
//     noc_async_read calls noc_async_read_barrier() before push, and a consumer that drained it with
//     noc_async_write calls noc_async_write_barrier() (or noc_async_writes_flushed()) before pop.
//   - Reserve and wait only check that space or data is available; calling them twice does not
//     reserve twice. Push only what you reserved, pop only what you waited for.
//   - Counters start at 0 for every kernel launch and wrap at 16 bits, so a channel can be reused
//     any number of times within a launch as long as pushes and pops balance.
//
// Example: DM fills two slots in turn on channel 0 and UNPACK consumes them.
//   constexpr uint32_t kCapacity = 2;
//
//   DM kernel:
//     for (uint32_t i = 0; i < n; ++i) {
//         uint32_t slot = base_addr + (i % kCapacity) * page_size;
//         scratch_reserve_back<0, kCapacity>(1);   // wait for a free slot
//         ... write the slot (e.g. noc_async_read into it), then noc_async_read_barrier() ...
//         scratch_push_back<0, kCapacity>(1);      // hand it to compute
//     }
//
//   Compute kernel (write ::experimental, since compute kernels see ckernel::experimental too):
//     for (uint32_t i = 0; i < n; ++i) {
//         uint32_t slot = base_addr + (i % kCapacity) * page_size;
//         ::experimental::scratch_wait_front<0, kCapacity>(1);   // wait for DM to fill it
//         ... unpack from the slot using the compute 2.0 API ...
//         ::experimental::scratch_pop_front<0, kCapacity>(1);    // give the slot back to DM
//     }
//
//   To label profiler events with the slot, pass it last: scratch_wait_front<0, kCapacity>(1, slot);

namespace experimental {

// Producer: block until num_pages pages of channel Ch are free.
template <std::uint32_t Ch, std::uint16_t Capacity = 1>
inline void scratch_reserve_back(std::int32_t num_pages, std::uint32_t l1_addr = 0) {
    static_assert(Ch < kScratchCbChannels, "scratch CB channel must be 0 or 1");
    static_assert(Capacity > 0, "scratch CB capacity must be positive");
#ifdef COMPILE_FOR_TRISC
    LLK_ASSERT(num_pages > 0 && num_pages <= Capacity, "scratch CB num_pages must be in 1..Capacity");
    PACK((scratch_cb_detail::llk_scratch_pack_reserve_back<Ch, Capacity>(num_pages, l1_addr)));
#else
    ASSERT(num_pages > 0 && num_pages <= Capacity);
    constexpr std::uint32_t cb = scratch_cb_detail::cb_id<Ch>();
#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_CB_SANITIZE)
    // Firmware zeroes unused CB slots only under watcher CB sanitize.
    ASSERT(get_local_cb_interface(cb).fifo_size == 0);
#endif
    uintptr_t pages_acked_ptr = (uintptr_t)get_cb_tiles_acked_ptr(cb);
    std::uint32_t pages_received = get_cb_tiles_received_ptr(cb)[0];

    std::int32_t free_space_pages;
    WAYPOINT("SRBW");
    {
        SYNC_WAIT("SYNC-SCRATCH-CB-RESERVE", l1_addr);
        do {
            invalidate_l1_cache();
            std::uint16_t pages_acked = (std::uint16_t)reg_read(pages_acked_ptr);
            std::uint16_t free_space_pages_wrap = Capacity - (std::uint16_t)(pages_received - pages_acked);
            free_space_pages = (std::int32_t)free_space_pages_wrap;
        } while (free_space_pages < num_pages);
    }
    WAYPOINT("SRBD");
#endif
}

// Producer: publish num_pages pages of channel Ch.
template <std::uint32_t Ch, std::uint16_t Capacity = 1>
inline void scratch_push_back(std::int32_t num_pages, std::uint32_t l1_addr = 0) {
    static_assert(Ch < kScratchCbChannels, "scratch CB channel must be 0 or 1");
    static_assert(Capacity > 0, "scratch CB capacity must be positive");
#ifdef COMPILE_FOR_TRISC
    LLK_ASSERT(num_pages > 0 && num_pages <= Capacity, "scratch CB num_pages must be in 1..Capacity");
    PACK((scratch_cb_detail::llk_scratch_pack_push_back<Ch>(num_pages, l1_addr)));
#else
    ASSERT(num_pages > 0 && num_pages <= Capacity);
    volatile tt_reg_ptr std::uint32_t* pages_received_ptr = get_cb_tiles_received_ptr(scratch_cb_detail::cb_id<Ch>());
    SYNC_SIGNAL("SYNC-SCRATCH-CB-PUSH", l1_addr);
    pages_received_ptr[0] += num_pages;
#endif
}

// Consumer: block until num_pages pages of channel Ch have been pushed.
template <std::uint32_t Ch, std::uint16_t Capacity = 1>
inline void scratch_wait_front(std::int32_t num_pages, std::uint32_t l1_addr = 0) {
    static_assert(Ch < kScratchCbChannels, "scratch CB channel must be 0 or 1");
    static_assert(Capacity > 0, "scratch CB capacity must be positive");
#ifdef COMPILE_FOR_TRISC
    LLK_ASSERT(num_pages > 0 && num_pages <= Capacity, "scratch CB num_pages must be in 1..Capacity");
    UNPACK((scratch_cb_detail::llk_scratch_unpack_wait_front<Ch>(num_pages, l1_addr)));
#else
    ASSERT(num_pages > 0 && num_pages <= Capacity);
    constexpr std::uint32_t cb = scratch_cb_detail::cb_id<Ch>();
#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_CB_SANITIZE)
    ASSERT(get_local_cb_interface(cb).fifo_size == 0);
#endif
    std::uint32_t pages_acked = get_cb_tiles_acked_ptr(cb)[0];
    uintptr_t pages_received_ptr = (uintptr_t)get_cb_tiles_received_ptr(cb);

    std::uint16_t pages_received;
    WAYPOINT("SWFW");
    {
        SYNC_WAIT("SYNC-SCRATCH-CB-WAIT", l1_addr);
        do {
            pages_received = ((std::uint16_t)reg_read(pages_received_ptr)) - pages_acked;
        } while (pages_received < num_pages);
    }
    WAYPOINT("SWFD");
#endif
}

// Consumer: free num_pages pages of channel Ch.
template <std::uint32_t Ch, std::uint16_t Capacity = 1>
inline void scratch_pop_front(std::int32_t num_pages, std::uint32_t l1_addr = 0) {
    static_assert(Ch < kScratchCbChannels, "scratch CB channel must be 0 or 1");
    static_assert(Capacity > 0, "scratch CB capacity must be positive");
#ifdef COMPILE_FOR_TRISC
    LLK_ASSERT(num_pages > 0 && num_pages <= Capacity, "scratch CB num_pages must be in 1..Capacity");
    UNPACK((scratch_cb_detail::llk_scratch_unpack_pop_front<Ch>(num_pages, l1_addr)));
#else
    ASSERT(num_pages > 0 && num_pages <= Capacity);
    volatile tt_reg_ptr std::uint32_t* pages_acked_ptr = get_cb_tiles_acked_ptr(scratch_cb_detail::cb_id<Ch>());
    SYNC_SIGNAL("SYNC-SCRATCH-CB-POP", l1_addr);
    pages_acked_ptr[0] += num_pages;
#endif
}

}  // namespace experimental
