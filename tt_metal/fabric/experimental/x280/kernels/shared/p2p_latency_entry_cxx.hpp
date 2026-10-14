// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/shared/p2p_latency_entry_cxx.hpp -- suite-owned X280 entry-point
 * template for p2p_latency participant images.
 *
 * Member supplies Backend, Section, SHAPE, POLL_MODE, TX_MODE, validate(),
 * result_addr(), backend_init(), and backend_shutdown().
 */
#ifndef X280_P2P_LATENCY_ENTRY_CXX_HPP
#define X280_P2P_LATENCY_ENTRY_CXX_HPP

#include <stdint.h>

#include "p2p_latency_frontend.h"
#include "p2p_latency_frontend_cxx.hpp"
#include "p2p_latency_layout.h"
#include "rt/x280_hw.h"

namespace p2p_latency {

/* Runs a copied participant section through validation, backend lifetime, and
 * the suite frontend. Validation failure leaves the result address untouched;
 * backend initialization failure publishes ARG_LAYOUT_MISMATCH. Returns zero
 * after every path. */
template <typename Member>
inline int run_participant_from_section(uint64_t hartid, uint64_t run_value, const typename Member::Section& section) {
    if (!Member::validate(section, hartid)) {
        return 0;
    }

    volatile p2p_latency_result_t* result_v =
        reinterpret_cast<volatile p2p_latency_result_t*>(Member::result_addr(section));
    if (result_v == nullptr) {
        return 0;
    }
    /* This hart is the only writer; the host reads after cleanup. */
    p2p_latency_result_t& r = *const_cast<p2p_latency_result_t*>(result_v);

    typename Member::Backend::state_t state;
    if (!Member::backend_init(state, section, hartid, run_value, r)) {
        r.terminal_state = static_cast<uint32_t>(P2P_LATENCY_STATE_ARG_LAYOUT_MISMATCH);
        x280_fence_ow();
        return 0;
    }

    (void)p2p_latency::run_participant<typename Member::Backend, Member::SHAPE, Member::POLL_MODE, Member::TX_MODE>(
        state, section.common, r);

    Member::backend_shutdown(state);

    /* Make the complete result visible before main returns. */
    x280_fence_ow();
    return 0;
}

}  // namespace p2p_latency

#endif /* X280_P2P_LATENCY_ENTRY_CXX_HPP */
