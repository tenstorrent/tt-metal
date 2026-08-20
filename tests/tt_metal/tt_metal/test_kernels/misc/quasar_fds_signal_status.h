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

}  // namespace quasar_fds_test
