// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "dest_order.h"
#include "llk_dest_dvalid.h"

namespace ckernel {

#if defined(TRISC_UNPACK)
constexpr dest_order::client dest_order_this_thread = dest_order::client::UNPACK;
#elif defined(TRISC_MATH)
constexpr dest_order::client dest_order_this_thread = dest_order::client::FPU;
#elif defined(TRISC_PACK)
constexpr dest_order::client dest_order_this_thread = dest_order::client::PACK;
#elif defined(TRISC_ISOLATE_SFPU)
constexpr dest_order::client dest_order_this_thread = dest_order::client::SFPU;
#endif

template <dest_order::client C>
constexpr dest_dvalid_client dvalid_client_of = static_cast<dest_dvalid_client>(static_cast<std::uint32_t>(C));

inline bool dest_dvalid_participates = false;

template <typename CHAIN>
inline void llk_dest_dvalid_configure() {
    constexpr bool participates = CHAIN::contains(dest_order_this_thread);
    dest_dvalid_participates = participates;

    if constexpr (participates) {
        constexpr dest_order::client succ = CHAIN::collapsed::successor(dest_order_this_thread);
        constexpr bool is_ring_start = CHAIN::collapsed::first() == dest_order_this_thread;

        _llk_dest_dvalid_configure_<dvalid_client_of<dest_order_this_thread>>(
            CHAIN::collapsed::mask(), dest_order::bit_of(succ), is_ring_start);
    }
}

template <typename CHAIN>
inline void llk_dest_dvalid_teardown() {
    if constexpr (CHAIN::contains(dest_order_this_thread)) {
        _llk_dest_dvalid_disable_<dvalid_client_of<dest_order_this_thread>>();
    }
    dest_dvalid_participates = false;
}

template <typename CHAIN, dest_order::client FROM, dest_order::client TO>
inline void llk_dest_dvalid_reconfig() {
    if constexpr (dest_order_this_thread == FROM) {
        constexpr dest_order::client new_succ = TO;
        constexpr bool is_ring_start = CHAIN::collapsed::first() == dest_order_this_thread;

        _llk_dest_dvalid_configure_<dvalid_client_of<dest_order_this_thread>>(
            CHAIN::collapsed::mask(), dest_order::bit_of(new_succ), is_ring_start);
    }
}

inline void llk_dest_dvalid_passthrough_if_skipped() {
    if constexpr (
        dest_order_this_thread == dest_order::client::UNPACK || dest_order_this_thread == dest_order::client::SFPU) {
        if (dest_dvalid_participates && !dest_order::was_touched(dest_order_this_thread)) {
            _llk_dest_dvalid_passthrough_<dvalid_client_of<dest_order_this_thread>, DST_SYNC_MODE>();
        }
    }
    dest_order::reset_touched();
}

}  // namespace ckernel
