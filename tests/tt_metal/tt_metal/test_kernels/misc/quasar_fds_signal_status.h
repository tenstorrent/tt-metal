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
// Dispatch side: number of enabled NEOs that signalled done. Worker side: raw go value read back.
constexpr uint32_t kSlotObserved = 1;
constexpr uint32_t kSlotResult = 2;
constexpr uint32_t kNumSlots = 3;

constexpr uint32_t kStarted = 0x5A5A0001;
constexpr uint32_t kComplete = 0x5A5A0002;
constexpr uint32_t kTimeout = 0x5A5A0003;

}  // namespace quasar_fds_test
