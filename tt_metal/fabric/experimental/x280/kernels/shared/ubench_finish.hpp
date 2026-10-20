// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* The one ending of a microbenchmark frontend's run. */
#ifndef X280_UBENCH_FINISH_HPP
#define X280_UBENCH_FINISH_HPP

#include <stdint.h>

/**
 * Runs Backend::publish_footer, the run's cleanup, then sets the result's
 * terminal_state to `terminal`, or to `cleanup_failed` when the footer
 * returns false, and publishes it behind Backend::publish_barrier. A host
 * that observes a terminal state therefore also observes the cleanup's
 * outcome. Backend::result(s) exposes a uint32_t terminal_state; State is the
 * suite's state enum.
 */
template <typename Backend, typename State>
inline void ubench_finish(typename Backend::state_t& s, State terminal, State cleanup_failed) {
    const bool cleaned = Backend::publish_footer(s);
    Backend::result(s).terminal_state = static_cast<uint32_t>(cleaned ? terminal : cleanup_failed);
    Backend::publish_barrier(s);
}

#endif
