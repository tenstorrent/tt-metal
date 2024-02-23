// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "command_queue_fixture.hpp"
#include "common/logger.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt::tt_metal;

namespace local_test_functions {

void FinishAllCqs(vector<std::reference_wrapper<CommandQueue>>& cqs) {
    for (uint i = 0; i < cqs.size(); i++) {
        Finish(cqs[i]);
    }
}

}

namespace basic_tests {

// Simplest test to record Event per CQ and wait from host.
TEST_F(MultiCommandQueueSingleDeviceFixture, TestEventsEventSynchronizeSanity) {
    CommandQueue a(this->device_, 0);
    CommandQueue b(this->device_, 1);
    vector<std::reference_wrapper<CommandQueue>> cqs = {a, b};
    std::unordered_map<uint, std::vector<Event>> sync_events;
    size_t num_events = 10;

    for (size_t j = 0; j < num_events; j++) {
        for (uint i = 0; i < cqs.size(); i++) {
            log_debug(tt::LogTest, "Recording and Host Syncing on event for CQ ID: {}", cqs[i].get().id());
            auto &event = sync_events[i].emplace_back(Event());
            EnqueueQueueRecordEvent(cqs[i], event);
            EXPECT_EQ(event.cq_id, cqs[i].get().id());
            EXPECT_EQ(event.event_id, j); // 1 cmd per CQ
            EventSynchronize(event);
        }
    }

    // Sync on earlier events again per CQ just to show it works.
    for (uint i = 0; i < cqs.size(); i++) {
        for (size_t j = 0; j < num_events; j++) {
            EventSynchronize(sync_events.at(i)[j]);
        }
    }

    local_test_functions::FinishAllCqs(cqs);
}


}  // end namespace basic_tests
