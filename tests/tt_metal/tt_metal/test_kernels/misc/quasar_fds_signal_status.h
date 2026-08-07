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

// Both worker-side kernels index their own status block by hardware thread index, so the worker
// block is sized for every processor a Tensix cluster has: indices 0 to 7 are the data-movement
// cores and 8 to 23 are the four TRISCs of each of the four Tensix engines. Processors that never
// run a kernel, such as the two reserved data-movement cores, leave their slots at zero, which is
// how the host tells which ones took part.
constexpr uint32_t kNumWorkerProcessors = 24;

// A dispatch-engine tile has the same eight data-movement cores and reserves none of them, so the
// kernel there runs on all eight and indexes its block the same way.
constexpr uint32_t kNumDispatchProcessors = 8;

// Stride between one processor's status block and the next, in words. Far wider than the three
// words a block uses, so that every block starts on its own cache lines: the data-movement kernels
// write status through the cached path and flush it, while the TRISCs write through the uncached
// alias, and two processors sharing a line could have one write back over the other's slots.
constexpr uint32_t kSlotsPerProcessor = 32;

constexpr uint32_t kStarted = 0x5A5A0001;
constexpr uint32_t kComplete = 0x5A5A0002;
constexpr uint32_t kTimeout = 0x5A5A0003;

}  // namespace quasar_fds_test
