// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Shared between test_dfb_write_barrier_bench.cpp and its two DM kernels. It measures what a
// kernel-exit write barrier costs on Quasar, to choose between an unconditional and a conditional one.

namespace dfb_wb_bench {

// What the measured kernel does before each timed barrier.
constexpr uint32_t SCENARIO_IDLE = 0;             // S1: no writes outstanding
constexpr uint32_t SCENARIO_EXPLICIT_WRITES = 1;  // S2: n plain noc_async_write calls
constexpr uint32_t SCENARIO_TRID_WRITES = 2;      // S3: n writes on DFB txn ids, as implicit-sync DFB writes are issued
constexpr uint32_t SCENARIO_COUPLED_IDLE = 3;     // S4: no writes here while the streamer on another DM writes
constexpr uint32_t SCENARIO_CROSS_HART_TRID = 4;  // S5: the streamer issues txn-id writes; this DM barriers on them

// The barrier being timed.
constexpr uint32_t V_NONE = 0;         // loop overhead only
constexpr uint32_t V_CONDITIONAL = 1;  // full barrier only if this DM wrote
constexpr uint32_t V_FULL = 2;         // noc_async_write_barrier()
constexpr uint32_t V_TRID = 3;         // noc_async_write_barrier_with_trid over the DFB txn ids
constexpr uint32_t NUM_VARIANTS = 4;

// Result words per variant, written through the uncached L1 alias.
constexpr uint32_t R_MIN = 0;
constexpr uint32_t R_MAX = 1;
constexpr uint32_t R_SUM = 2;
constexpr uint32_t R_ITERS = 3;
constexpr uint32_t R_LEAKED = 4;           // iterations with writes still outstanding right after the barrier
constexpr uint32_t R_MAX_OUTSTANDING = 5;  // largest outstanding count this DM saw (S4/S5: other DM's traffic)
constexpr uint32_t WORDS_PER_VARIANT = 8;

// Handshake words, after the per-variant results.
constexpr uint32_t FLAG_BASE_WORD = NUM_VARIANTS * WORDS_PER_VARIANT;
constexpr uint32_t FLAG_STREAMER_STARTED = FLAG_BASE_WORD + 0;
// S5: sequence number of the streamer's last txn-id burst (issued, no barrier yet), and of the last
// burst the measurer has finished with.
constexpr uint32_t FLAG_ISSUED_SEQ = FLAG_BASE_WORD + 1;
constexpr uint32_t FLAG_CONSUMED_SEQ = FLAG_BASE_WORD + 2;
constexpr uint32_t FLAG_MEASURER_DONE = FLAG_BASE_WORD + 3;
constexpr uint32_t STREAMER_SELF_MAX_OUTSTANDING = FLAG_BASE_WORD + 4;  // S5: the streamer's own view
constexpr uint32_t NUM_RESULT_WORDS = FLAG_BASE_WORD + 5;

// DFB txn ids are allocated top-down from 31 (dataflow_buffer_config.h); the bench uses the same range.
constexpr uint32_t FIRST_TRID = 28;
constexpr uint32_t NUM_TRIDS = 4;

}  // namespace dfb_wb_bench
