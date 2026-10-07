// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/scratch_cb.h"
#ifdef TRISC_PACK
#include "llk_io_pack.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_io_unpack.h"
#endif

namespace experimental::scratch_cb_detail {

// Tensix 4-byte word address of a stream register, masked to 18 bits (as in llk_io).
inline std::uint32_t tensix_reg_addr(volatile std::uint32_t* ptr) {
    return static_cast<std::uint32_t>((reinterpret_cast<std::uintptr_t>(ptr) >> 2) & 0x3ffff);
}

// Local copies of this thread's own counter: the STOREREG that publishes it lands only after the
// packer/unpacker finishes, so reading the register back could see a stale value. Zeroed by kernel crt.
#ifdef TRISC_PACK
inline std::uint16_t pages_received[kScratchCbChannels];
#endif
#ifdef TRISC_UNPACK
inline std::uint16_t pages_acked[kScratchCbChannels];
#endif

#ifdef TRISC_PACK
template <std::uint32_t Ch, std::uint16_t Capacity>
inline void llk_scratch_pack_reserve_back(std::int32_t num_pages, std::uint32_t l1_addr) {
    volatile tt_reg_ptr std::uint32_t* pages_acked_ptr = get_cb_tiles_acked_ptr(cb_id<Ch>());
    std::uint16_t received = pages_received[Ch];

    std::int32_t free_pages;
    {
        SYNC_WAIT("SYNC-SCRATCH-CB-RESERVE", l1_addr);
        do {
            std::uint16_t acked = (std::uint16_t)reg_read((std::uint32_t)pages_acked_ptr);
            // 16-bit subtraction: the counters may wrap.
            std::uint16_t free_pages_wrap = Capacity - (std::uint16_t)(received - acked);
            free_pages = (std::int32_t)free_pages_wrap;
        } while (free_pages < num_pages);
    }
}

template <std::uint32_t Ch>
inline void llk_scratch_pack_push_back(std::int32_t num_pages, std::uint32_t l1_addr) {
    SYNC_SIGNAL("SYNC-SCRATCH-CB-PUSH", l1_addr);
    pages_received[Ch] += num_pages;
    // Publish only after the packer has finished writing the page.
    TT_SETDMAREG(0, pages_received[Ch], 0, LO_16(p_gpr_pack::NUM_MSGS_RECEIVED));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
    TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED, tensix_reg_addr(get_cb_tiles_received_ptr(cb_id<Ch>())));
}
#endif  // TRISC_PACK

#ifdef TRISC_UNPACK
template <std::uint32_t Ch>
inline void llk_scratch_unpack_wait_front(std::int32_t num_pages, std::uint32_t l1_addr) {
    volatile tt_l1_ptr std::uint32_t* pages_received_ptr = get_cb_tiles_received_ptr(cb_id<Ch>());
    std::uint16_t num_pages_u = (std::uint16_t)num_pages;

    std::uint16_t available;
    {
        SYNC_WAIT("SYNC-SCRATCH-CB-WAIT", l1_addr);
        do {
            std::uint16_t received = (std::uint16_t)reg_read((std::uint32_t)pages_received_ptr);
            available = received - pages_acked[Ch];
        } while (available < num_pages_u);
    }
}

template <std::uint32_t Ch>
inline void llk_scratch_unpack_pop_front(std::int32_t num_pages, std::uint32_t l1_addr) {
    SYNC_SIGNAL("SYNC-SCRATCH-CB-POP", l1_addr);
    pages_acked[Ch] += num_pages;
    // Publish only after the unpacker has finished reading the page.
    TT_SETDMAREG(0, pages_acked[Ch], 0, LO_16(4));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);
    TT_STOREREG(4, tensix_reg_addr(get_cb_tiles_acked_ptr(cb_id<Ch>())));
}
#endif  // TRISC_UNPACK

}  // namespace experimental::scratch_cb_detail
