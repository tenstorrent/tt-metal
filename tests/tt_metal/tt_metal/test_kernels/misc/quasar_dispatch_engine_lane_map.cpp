// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Dispatch-engine side of the Quasar FDS lane-mapping experiment. The worker side lives in
// quasar_fds_worker_drive_done.cpp.
//
// No go is sent. Each worker tile drives a done carrying its own group id, unprompted, and this
// kernel reads the receive side to find out which lane each of those values arrived on. The raw
// TENSIX_TO_DISPATCH inbox registers sit before all aggregation, one per lane, so the index of a
// non-zero register is a lane number and its contents name the tile that drove it. That is the
// worker-coordinate-to-lane-bit mapping the dispatch design needs and currently has to guess.
//
// Two independent readings of the same fact are recorded, from opposite sides of the aggregation
// logic. The inbox scan gives lane and value directly. Group 0's status is the map of lanes carrying
// nothing, since group 0 is the idle value on the wire, so a lane that starts carrying a real done
// drops out of it. Agreement between the two is what makes the mapping trustworthy.
//
// The per-group done counts are recorded at the same time and cost nothing extra. Every group is
// enabled on every lane, so a group whose tile never drove must still read zero: that is the
// done-direction isolation check, which the go/done handshake tests cannot make because an
// unsignalled group's workers never drive anything.

#include "risc_attribs.h"
#include "risc_common.h"
#include "api/compile_time_args.h"
#include "overlay/fds_functions.hpp"

#include "quasar_fds_signal_status.h"

namespace {

// A done must be stable for a single cycle to be accepted. This is the register's reset value,
// written explicitly so the scan does not depend on the state a previous program left behind.
constexpr uint32_t kNoDeglitchFilter = 0;

// Group 0 is the idle value on the wire and is never driven, so it is configured but not counted.
constexpr uint32_t kFirstRealGroup = 1;

uint32_t count_driving_lanes() {
    uint32_t driving = 0;
    for (uint32_t lane = 0; lane < quasar_fds_test::lane_map::kNumLanes; lane++) {
        driving += (overlay::FdsDispatch::fds_read_neo_status(lane) != 0) ? 1 : 0;
    }
    return driving;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t l1_address = get_named_compile_time_arg_val("l1_address");
    constexpr uint32_t worker_mask = get_named_compile_time_arg_val("worker_mask");
    // How many tiles are driving, so the scan can wait for all of them rather than sampling early
    // and reporting a partial map as if it were the whole one.
    constexpr uint32_t expected_lanes = get_named_compile_time_arg_val("expected_lanes");
    constexpr uint32_t poll_iterations = get_named_compile_time_arg_val("poll_iterations");

    volatile tt_l1_ptr uint32_t* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_address);
    status[quasar_fds_test::lane_map::kSlotStarted] = quasar_fds_test::kStarted;
    status[quasar_fds_test::lane_map::kSlotResult] = quasar_fds_test::kTimeout;
    flush_l2_cache_range(l1_address, quasar_fds_test::lane_map::kNumSlots * sizeof(uint32_t));

    overlay::FdsDispatch::fds_config_filter_length(kNoDeglitchFilter);
    // Every group watches every lane with a threshold of one, so a done landing in the wrong group
    // would be counted rather than filtered out. Nothing here relies on the counts to find lanes;
    // they exist so that a mis-credited done is visible.
    for (uint32_t group = kFirstRealGroup; group < quasar_fds_test::lane_map::kNumGroups; group++) {
        overlay::FdsDispatch::fds_config_groupid(group, worker_mask, 1);
    }

    // Drop anything held from an earlier epoch, so every value the scan finds was driven this run.
    for (uint32_t lane = 0; lane < quasar_fds_test::lane_map::kNumLanes; lane++) {
        overlay::FdsDispatch::fds_clear_neo_status(lane);
    }

    // A done is a held level, so once a tile drives one it stays up and this wait cannot miss it.
    uint32_t driving = 0;
    for (uint32_t i = 0; i < poll_iterations; i++) {
        driving = count_driving_lanes();
        if (driving >= expected_lanes) {
            break;
        }
    }

    // Recorded whether or not every tile turned up: a partial map still names the lanes that worked.
    status[quasar_fds_test::lane_map::kSlotLanesDriving] = driving;
    status[quasar_fds_test::lane_map::kSlotIdleLaneMap] = overlay::FdsDispatch::fds_read_group_status(0);
    for (uint32_t lane = 0; lane < quasar_fds_test::lane_map::kNumLanes; lane++) {
        status[quasar_fds_test::lane_map::kSlotLaneBase + lane] = overlay::FdsDispatch::fds_read_neo_status(lane);
    }
    for (uint32_t group = 0; group < quasar_fds_test::lane_map::kNumGroups; group++) {
        status[quasar_fds_test::lane_map::kSlotGroupCountBase + group] =
            overlay::FdsDispatch::fds_read_group_count(group);
    }
    status[quasar_fds_test::lane_map::kSlotResult] =
        (driving >= expected_lanes) ? quasar_fds_test::kComplete : quasar_fds_test::kTimeout;

    flush_l2_cache_range(l1_address, quasar_fds_test::lane_map::kNumSlots * sizeof(uint32_t));
}
