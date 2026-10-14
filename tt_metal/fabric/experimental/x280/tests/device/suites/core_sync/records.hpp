// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_SUITES_CORE_SYNC_RECORDS_HPP
#define X280_SUITES_CORE_SYNC_RECORDS_HPP

#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"

#include <cstdint>
#include <vector>

namespace x280::suites::core_sync {

// One released worker as the characterization record reports it.
struct ReleasedWorker {
    uint32_t logical_x = 0;
    uint32_t logical_y = 0;
    uint64_t raw_timestamp = 0;  // barrier exit, raw debug-timer ticks
    uint32_t outcome = 0;
};

/**
 * The x280_director_characterization record of one release shape (`release`
 * is "null", "unicast" or "multicast"): every worker's raw exit timestamp and
 * outcome. The timestamps come from different cores' timers with no common
 * origin, so the record carries no duration and stays measurement schema 1.
 * The release has no warm-up.
 */
inline x280::harness::output::RecordSpec characterization_record(
    const char* release, uint32_t run_nonce, uint32_t chip_id, const std::vector<ReleasedWorker>& workers) {
    namespace hout = x280::harness::output;
    std::vector<hout::Value> participants;
    participants.reserve(workers.size());
    for (const auto& w : workers) {
        participants.push_back(hout::Value::object(
            {{"logical_x", w.logical_x},
             {"logical_y", w.logical_y},
             {"raw_timestamp", w.raw_timestamp},
             {"outcome", w.outcome}}));
    }
    return {
        .suite = "core_sync",
        .member = "x280_director_characterization",
        .fields =
            {{"release", release},
             {"run_nonce", run_nonce},
             {"chip_id", chip_id},
             {"origins_comparable", false},
             {"availability_error", "cross_core_spread_has_no_local_duration"},
             {"participants", hout::Value::array(std::move(participants))}},
        .wall_counter = x280::harness::measurement::Counter::TensixLiveTimer,
        .warmup = hout::NoWarmup{},
        .clock = hout::NoConversion{},
    };
}

}  // namespace x280::suites::core_sync

#endif  // X280_SUITES_CORE_SYNC_RECORDS_HPP
