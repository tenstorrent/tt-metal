// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// NoC transaction IDs used by the fast dispatch kernels.
//
// Per-trid counters are kept per NIU, not per command buffer, and on WH/BH a trid's outstanding count is shared by
// reads and non-posted writes. On Quasar the counters are shared by every DM on the tile, and prefetch, dispatch and
// dispatch_s all run on the same dispatch engine. Allocate every trid here so the kernels never wait on each other's
// transactions.
namespace cq_trids {

// Untagged transactions. Kernels must leave the read and write cmd bufs on this trid when they exit.
constexpr uint32_t DEFAULT = 0U;

// Prefetch: exec_buf DRAM reads into cmddat_q.
constexpr uint32_t PREFETCH_EXEC_BUF_READ = 1U;
// Prefetch: in-flight fetch_q PCIe reads, one trid per outstanding read.
constexpr uint32_t PREFETCH_FETCH_Q_READ_BASE = 2U;
constexpr uint32_t PREFETCH_FETCH_Q_READ_COUNT = 4U;
// Prefetch: relay_linear scratch double-buffer halves.
constexpr uint32_t PREFETCH_RELAY_LINEAR_READ[2] = {6U, 7U};

static_assert(PREFETCH_EXEC_BUF_READ > DEFAULT);
static_assert(PREFETCH_FETCH_Q_READ_BASE > PREFETCH_EXEC_BUF_READ);
static_assert(PREFETCH_RELAY_LINEAR_READ[0] >= PREFETCH_FETCH_Q_READ_BASE + PREFETCH_FETCH_Q_READ_COUNT);
static_assert(PREFETCH_RELAY_LINEAR_READ[1] == PREFETCH_RELAY_LINEAR_READ[0] + 1U);

// Highest trid in use. Must fit in NOC_MAX_TRANSACTION_ID on every arch.
constexpr uint32_t MAX_USED = PREFETCH_RELAY_LINEAR_READ[1];

}  // namespace cq_trids

#ifdef NOC_MAX_TRANSACTION_ID
static_assert(cq_trids::MAX_USED <= NOC_MAX_TRANSACTION_ID, "dispatch trids exceed this arch's transaction IDs");
#endif
