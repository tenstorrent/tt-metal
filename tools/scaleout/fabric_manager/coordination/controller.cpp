// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/scaleout/fabric_manager/coordination/controller.hpp"

#include <tt_stl/assert.hpp>

namespace tt::scaleout_tools::fabric_manager {

std::vector<Bytes> Controller::exchange(
    const ScopeKey& scope, uint64_t epoch, int index, int count, const Bytes& payload) {
    TT_FATAL(count > 0, "Controller::exchange requires count > 0 (got {})", count);
    TT_FATAL(index >= 0 && index < count, "Controller::exchange index {} out of range [0,{})", index, count);

    std::unique_lock<std::mutex> lock(mu_);
    const Key key{scope.mesh_id, epoch};
    Phase& phase = phases_[key];

    if (phase.expected == 0) {
        phase.expected = count;
    }
    TT_FATAL(
        phase.expected == count,
        "Controller::exchange participant-count mismatch for the same phase (scope epoch {}): {} vs {}",
        epoch,
        phase.expected,
        count);
    TT_FATAL(
        phase.contributions.find(index) == phase.contributions.end(),
        "Controller::exchange duplicate participant index {} for phase epoch {}",
        index,
        epoch);

    phase.contributions[index] = payload;
    phase.arrived++;

    if (phase.arrived == phase.expected) {
        // Last arrival assembles the shared result (ascending index order) and releases everyone.
        phase.result.reserve(static_cast<std::size_t>(phase.expected));
        for (auto& [_, bytes] : phase.contributions) {
            phase.result.push_back(std::move(bytes));
        }
        phase.released = true;
        cv_.notify_all();
    } else {
        cv_.wait(lock, [&phase] { return phase.released; });
    }

    std::vector<Bytes> result = phase.result;  // copy out under lock

    // Last reader erases the phase so the map does not grow unbounded across epochs.
    if (++phase.retrieved == phase.expected) {
        phases_.erase(key);
    }
    return result;
}

}  // namespace tt::scaleout_tools::fabric_manager
