// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The streaming profiler's captures: each device's lease with tt-insight, which carries the profiler's records from
// the device to the host, the patch of every program image, and a capture's start and stop
// (tt_metal/tools/profiler/STREAMING_PROFILER.md).
#pragma once

#include <vector>

#include <tt-metalium/vocabulary/core_coord.hpp>
#include <tt-metalium/vocabulary/device_types.hpp>
#include <tt/insight/host/device_lease.hpp>

namespace tt {
class Cluster;
namespace llrt {
class RunTimeOptions;
}
}  // namespace tt

namespace tt::tt_metal::streaming_profiler {

// Lends tt-insight the tiles and L1 that `terms` names on the device. Every processor of the device must be held in
// reset. From here on, every image Metal loads is patched through the leases: `two_eriscs` is Metal's 2-ERISC mode,
// which says on which ERISC the main Ethernet program runs.
void lease_device(ChipId chip_id, const tt::insight::DeviceLease::Terms& terms, bool two_eriscs);

// Whether the tile is lent to tt-insight whole, so that Metal loads and runs nothing on it.
bool is_lent_tile(ChipId chip_id, const CoreCoord& translated);

// Starts one capture of the leased devices among `chip_ids`. Metal calls it when it opens the devices, before their
// fabric routers and dispatch kernels start.
void start_capture(
    const std::vector<ChipId>& chip_ids, const tt::Cluster& cluster, const tt::llrt::RunTimeOptions& rtoptions);

// Stops every capture that has one of the devices. Metal calls it when it closes them, after their fabric routers have
// exited, so that the routers' records are in the capture.
void stop_captures(const std::vector<ChipId>& chip_ids);

// Ends the devices' leases, after their captures. Metal calls it before it resets the devices' processors.
void end_leases(const std::vector<ChipId>& chip_ids);

}  // namespace tt::tt_metal::streaming_profiler
