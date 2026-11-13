// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the power-virus host test and its kernels: the per-Tensix L1
// control block and the active-ETH link kernel layout. Constants only, so the
// host can include it too.

#pragma once

#include <cstdint>

namespace pv {

// -- Tensix L1 control block (offsets from the control base) -----------------
// The host writes the control block before launch and sets STOP to end the run.
constexpr uint32_t kStop = 0;         // host writes 1 to stop every generator on the core
constexpr uint32_t kGo = 16;          // 1 = start; preset by host, or set by the BRISC super-sync barrier
constexpr uint32_t kReady = 32;       // TRISC unpack writes 1 once initialized (super-sync waits on it)
constexpr uint32_t kFinal = 48;       // TRISC loop index to stop at, agreed by all three TRISCs
constexpr uint32_t kFpuRecord = 64;   // [elapsed_lo, elapsed_hi, loops_lo, loops_hi]
constexpr uint32_t kDramRecord = 80;  // [elapsed_lo, elapsed_hi, xfers_lo, xfers_hi]
constexpr uint32_t kNocRecord = 96;   // [elapsed_lo, elapsed_hi, xfers_lo, xfers_hi]
constexpr uint32_t kControlBytes = 128;

constexpr uint32_t kFinalUnset = 0xFFFFFFFFu;

// -- Active-ETH L1 layout of the link kernel (offsets from unreserved base) ---
constexpr uint32_t kEthHandshake = 0;  // written by peer
constexpr uint32_t kEthTiming = 16;    // timing staging, persisted to DRAM at the end
constexpr uint32_t kEthFinIn = 32;     // written by peer: [magic, peer_sent]
constexpr uint32_t kEthFinOut = 48;    // staging for our FIN
constexpr uint32_t kEthStop = 64;      // host writes 1 to stop
constexpr uint32_t kEthHeaderBytes = 128;

// Throttle time unit, in wall-clock ticks.
constexpr uint32_t kIdleUnitTicks = 512;

}  // namespace pv
