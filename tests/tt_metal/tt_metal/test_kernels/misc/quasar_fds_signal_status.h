// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Status words shared by the Quasar FDS signalling test kernels
// (quasar_dispatch_engine_signal.cpp, quasar_fds_worker_signal.cpp) and the host test
// that launches them. Each side writes markers into its own L1 so the host can tell a
// completed handshake apart from a side that never ran or gave up waiting on FDS.
namespace quasar_fds_test {

// Word offsets into the status block at the L1 address handed to each kernel.
constexpr uint32_t kSlotStarted = 0;
// Dispatch side: number of enabled NEOs that signalled done. Worker side: the last non-zero go
// value seen on any inbox, which need not name this worker's own group.
constexpr uint32_t kSlotObserved = 1;
constexpr uint32_t kSlotResult = 2;
// Dispatch side only: the done count of a group that was configured but deliberately never
// signalled, which is how a group that should have stayed quiet reports that it did.
constexpr uint32_t kSlotQuietGroupCount = 3;
// Worker side only: this worker's own group status register at the end of its wait. A worker that
// sees another group's go on the wire but never latches it reads a non-zero kSlotObserved and a
// zero here, which is the difference between a shared wire and a broken group filter.
constexpr uint32_t kSlotGroupStatus = 4;
constexpr uint32_t kNumSlots = 5;

constexpr uint32_t kStarted = 0x5A5A0001;
constexpr uint32_t kComplete = 0x5A5A0002;
constexpr uint32_t kTimeout = 0x5A5A0003;

// Status layout for the two-epoch re-arm experiment (quasar_dispatch_engine_rearm.cpp,
// quasar_fds_worker_rearm.cpp). Each side has its own slots at distinct indices rather than
// sharing indices with different meanings, so a dump of either block reads unambiguously.
namespace rearm {

constexpr uint32_t kSlotStarted = 0;
constexpr uint32_t kSlotResult = 1;

// Dispatch side. The two counts either side of the inbox clear are the measurement the whole
// experiment exists for: they say whether clearing a receive inbox holds while the worker is still
// driving the same value into it, or whether it re-latches immediately.
constexpr uint32_t kSlotRound1Count = 2;
constexpr uint32_t kSlotCountAfterClear = 3;
constexpr uint32_t kSlotCountAfterSettle = 4;
constexpr uint32_t kSlotRound2Count = 5;

// Worker side. Each is 1 if that step was reached and 0 if it timed out.
constexpr uint32_t kSlotRound1Go = 6;
constexpr uint32_t kSlotDeassertSeen = 7;
constexpr uint32_t kSlotStatusAfterDeassert = 8;
constexpr uint32_t kSlotRound2Go = 9;

constexpr uint32_t kNumSlots = 10;

}  // namespace rearm

// Status layout for the lane-mapping experiment (quasar_dispatch_engine_lane_map.cpp,
// quasar_fds_worker_drive_done.cpp). Every worker tile drives a done carrying its own group id and
// nothing sends a go, so the dispatch side's raw inbox registers name which lane each tile sits on.
namespace lane_map {

// One TENSIX_TO_DISPATCH inbox register per lane on the dispatch side, and 16 group ids.
constexpr uint32_t kNumLanes = 32;
constexpr uint32_t kNumGroups = 16;

constexpr uint32_t kSlotStarted = 0;
constexpr uint32_t kSlotResult = 1;
// How many lanes were carrying a non-zero value when the scan ran.
constexpr uint32_t kSlotLanesDriving = 2;
// Group 0's status, which is the map of lanes carrying nothing. An independent reading of the same
// fact as the inbox scan, from a register on the other side of the aggregation logic.
constexpr uint32_t kSlotIdleLaneMap = 3;
// The raw value each lane is carrying: kSlotLaneBase + lane. Zero means idle, otherwise the group
// id, which names the tile that drove it.
constexpr uint32_t kSlotLaneBase = 4;
// The done count for each group: kSlotGroupCountBase + group. A group whose tile never drove must
// read zero, which is what makes this the done-direction isolation check.
constexpr uint32_t kSlotGroupCountBase = kSlotLaneBase + kNumLanes;

constexpr uint32_t kNumSlots = kSlotGroupCountBase + kNumGroups;

// The worker side reuses the shared slots above: started, result, and the group it drove in
// kSlotObserved.

}  // namespace lane_map

// Status layout and payload contract for the write-ordering experiment
// (quasar_dispatch_engine_ordered_read.cpp, quasar_fds_worker_ordered_write.cpp). The worker writes a
// payload to the dispatch core's L1 over the NOC and then drives its FDS done; the dispatch engine
// reads that payload the moment it sees the done. Whether the payload is intact is the question a
// completion fence exists to answer.
namespace ordering {

constexpr uint32_t kSlotStarted = 0;
constexpr uint32_t kSlotResult = 1;

// Dispatch side: how many payload words did not hold the expected value when the done was observed,
// and where the first of them was.
constexpr uint32_t kSlotMismatches = 2;
constexpr uint32_t kSlotFirstMismatchIndex = 3;
constexpr uint32_t kSlotFirstMismatchValue = 4;

// Worker side.
constexpr uint32_t kSlotGoSeen = 5;
constexpr uint32_t kSlotBarrierUsed = 6;

// Dispatch side: the last payload word, read as the very first thing after the done is observed.
// The full scan takes long enough that its later words have time to land; this single read is the
// tightest probe available and is the one to watch.
constexpr uint32_t kSlotTailWord = 7;
// 1 if the worker signalled over FDS, 0 if it signalled with a NOC atomic increment.
constexpr uint32_t kSlotSignalledByFds = 8;

constexpr uint32_t kNumSlots = 9;

// Payload word i holds kPayloadSeed + i, so a stale word identifies itself by value as well as by
// position. The host pre-fills the destination with kPrefillWord, which is deliberately outside the
// payload's value range: reading it back means the write had not landed.
constexpr uint32_t kPayloadSeed = 0x0DEF0000;
constexpr uint32_t kPrefillWord = 0xBAADF00D;

// Big enough that the write takes real time on the wire — a single word would land too fast for a
// missing barrier to ever be visible — and small enough to check in a kernel loop. Raised from 4 KB
// after both arms came back clean at that size: a larger transfer keeps more of it in flight.
constexpr uint32_t kPayloadBytes = 32768;
constexpr uint32_t kPayloadWords = kPayloadBytes / sizeof(uint32_t);

// Offset from the dispatch core's unreserved L1 base to the completion counter used by the NOC-atomic
// control arm. On its own cache line, clear of both the status block and the payload, so polling it
// cannot disturb either.
constexpr uint32_t kCounterOffset = 512;

// Offset from each core's unreserved L1 base to the payload, leaving the status block below it.
constexpr uint32_t kPayloadOffset = 1024;

}  // namespace ordering

}  // namespace quasar_fds_test
