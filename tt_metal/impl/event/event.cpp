// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>
#include <tt_stl/tt_pause.hpp>
#include <event.hpp>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal {

void Event::wait_until_ready() {
    log_trace(
        tt::LogMetal,
        "Waiting for Event to be ready. (ready: {} cq_id: {} event_id: {})",
        bool(ready.load()),
        cq_id,
        event_id);

    tt::stl::TT_NICE_SPIN_UNTIL<100, 10>([this] { return ready.load(); });

    log_trace(tt::LogMetal, "Event is ready. (ready: {} cq_id: {} event_id: {})", bool(ready.load()), cq_id, event_id);

    TT_ASSERT(device != nullptr, "Event must have initialized device ptr");
    TT_ASSERT(event_id != -1, "Event must have initialized event_id");
    TT_ASSERT(cq_id != -1, "Event must have initialized cq_id");
}

}  // namespace tt::tt_metal
