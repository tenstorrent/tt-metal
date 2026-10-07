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
#if defined(TRISC_PACK) || defined(TRISC_UNPACK)
#include "experimental/2_0/llk_scratch_cb.h"
#endif
#else  // !COMPILE_FOR_TRISC
#include "api/dataflow/dataflow_api.h"
#endif

// Experimental DM <-> TRISC sync on caller-owned L1 scratch, with no CB ID. Same counters and the same
// register path as a CB; performance must be measured on hardware. Channel Ch (0 or 1) owns the
// stream counters of CB 62 + Ch:
// tiles_received (written only by the producer) and tiles_acked (written only by the consumer).
//
// CB IDs 62 and 63 are reserved by the Blackhole host allocation limit. The caller owns the
// L1 slots and picks which one to use; l1_addr only tags the profiler sync events. Capacity defaults to
// one page per channel: alternate <0> and <1> for two independently synchronized scratch slots. An
// explicit Capacity supports a FIFO ring per channel; use the same Capacity in all calls on both sides.
// One side is DM and the other is compute: DM -> UNPACK or PACK -> DM. On compute, calls on
// the other TRISC threads compile to nothing, as in CircularBuffer.
//
// Firmware zeros both counters before every kernel (init_sync_registers in trisc.cc). Each channel has
// one producer and one consumer with fixed roles for the launch. Balanced push/pop operations allow
// reuse within the launch. Both sides must use the same FIFO order and backing slots.
// Complete DM NoC writes before push and NoC reads before pop using the appropriate NoC barrier.
// The address is a profiler tag, not an allocation or an independent synchronization resource.
//
// A page is a caller-defined unit of work, not a tile format or a CB descriptor. This API allocates
// no payload memory and does not advance pointers. Capacity is a compile-time number of such units
// (1..65535); the caller supplies enough storage for all outstanding units. Counters wrap at 16 bits.
// Each call requires 0 < num_pages <= Capacity. Reserve/wait only observe availability: repeated
// calls before push/pop are cumulative checks, not additional reservations. Publish only after
// reserve succeeds, and release only units covered by wait. Both sides must agree on the unit size.
//
// Example, channel 0 with two caller-managed slots (one credit per slot):
//   DM:     scratch_reserve_back<0, 2>(slot_addr, 1);
//           ... write slot; complete NoC writes with the appropriate barrier ...
//           scratch_push_back<0, 2>(slot_addr, 1);
//   UNPACK: scratch_wait_front<0, 2>(slot_addr, 1);
//           ... unpack from slot using the compute 2.0 API ...
//           scratch_pop_front<0, 2>(slot_addr, 1);
// Both sides select slots in the same FIFO order. These calls are not an all-TRISC barrier and do
// not support multiple producers or consumers on one channel, and endpoints cannot change within a launch.

namespace experimental {

// Producer: block until num_pages pages of channel Ch are free.
template <std::uint32_t Ch, std::uint16_t Capacity = 1>
inline void scratch_reserve_back(std::uint32_t l1_addr, std::int32_t num_pages) {
    static_assert(Ch < kScratchCbChannels, "scratch CB channel must be 0 or 1");
    static_assert(Capacity > 0, "scratch CB capacity must be positive");
#ifdef COMPILE_FOR_TRISC
    PACK((scratch_cb_detail::llk_scratch_pack_reserve_back<Ch, Capacity>(l1_addr, num_pages)));
#else
    constexpr std::uint32_t cb = scratch_cb_detail::cb_id<Ch>();
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
inline void scratch_push_back(std::uint32_t l1_addr, std::int32_t num_pages) {
    static_assert(Ch < kScratchCbChannels, "scratch CB channel must be 0 or 1");
    static_assert(Capacity > 0, "scratch CB capacity must be positive");
#ifdef COMPILE_FOR_TRISC
    PACK((scratch_cb_detail::llk_scratch_pack_push_back<Ch>(l1_addr, num_pages)));
#else
    volatile tt_reg_ptr std::uint32_t* pages_received_ptr = get_cb_tiles_received_ptr(scratch_cb_detail::cb_id<Ch>());
    SYNC_SIGNAL("SYNC-SCRATCH-CB-PUSH", l1_addr);
    pages_received_ptr[0] += num_pages;
#endif
}

// Consumer: block until num_pages pages of channel Ch have been pushed.
template <std::uint32_t Ch, std::uint16_t Capacity = 1>
inline void scratch_wait_front(std::uint32_t l1_addr, std::int32_t num_pages) {
    static_assert(Ch < kScratchCbChannels, "scratch CB channel must be 0 or 1");
    static_assert(Capacity > 0, "scratch CB capacity must be positive");
#ifdef COMPILE_FOR_TRISC
    UNPACK((scratch_cb_detail::llk_scratch_unpack_wait_front<Ch>(l1_addr, num_pages)));
#else
    constexpr std::uint32_t cb = scratch_cb_detail::cb_id<Ch>();
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
inline void scratch_pop_front(std::uint32_t l1_addr, std::int32_t num_pages) {
    static_assert(Ch < kScratchCbChannels, "scratch CB channel must be 0 or 1");
    static_assert(Capacity > 0, "scratch CB capacity must be positive");
#ifdef COMPILE_FOR_TRISC
    UNPACK((scratch_cb_detail::llk_scratch_unpack_pop_front<Ch>(l1_addr, num_pages)));
#else
    volatile tt_reg_ptr std::uint32_t* pages_acked_ptr = get_cb_tiles_acked_ptr(scratch_cb_detail::cb_id<Ch>());
    SYNC_SIGNAL("SYNC-SCRATCH-CB-POP", l1_addr);
    pages_acked_ptr[0] += num_pages;
#endif
}

}  // namespace experimental
