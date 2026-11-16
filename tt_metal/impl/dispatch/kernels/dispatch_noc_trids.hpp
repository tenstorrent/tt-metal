// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NoC transaction ids of the fast-dispatch kernels, one range per use. The dispatch kernels share their cores
// (cq_prefetch, cq_dispatch and dispatch_s; on a Quasar dispatch engine the Tensor prefetcher kernels too), and the
// count a barrier on an id reads covers every transaction on the core that carries that id. So a dispatch kernel
// tags its transactions only with ids from its own range here, and the ranges are disjoint.
//
// Id 0 is untagged. cq_dispatch, dispatch_s and every transaction no range covers use it, and a barrier on it
// waits on all of them. The command-queue ranges grow up from 1, and the Tensor prefetcher range grows down from
// the top, so either side can grow without moving the other.

#include <stdint.h>

#include "noc/noc_parameters.h"
#if defined(ARCH_QUASAR)
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"
#endif

namespace dispatch_noc_trids {

// The highest id the core's NoC takes. On Quasar, kernels outside the runtime keep to [0, USER_TXN_ID_MAX] and
// dataflow buffers take the rest; fast dispatch's kernels are the runtime and run where no dataflow buffer does.
#if defined(ARCH_QUASAR)
inline constexpr uint32_t kMaxTrid = HW_TXN_ID_MAX;
#else
inline constexpr uint32_t kMaxTrid = NOC_MAX_TRANSACTION_ID;
#endif

// `count` consecutive ids from `first`.
struct TridRange {
    uint32_t first = 0;
    uint32_t count = 0;

    constexpr uint32_t size() const { return count; }
    constexpr uint32_t end() const { return first + count; }
    constexpr uint32_t operator[](uint32_t i) const { return first + i; }
};

// cq_prefetch: exec_buf DRAM reads into cmddat_q.
inline constexpr TridRange kPrefetchExecBuf{.first = 1, .count = 1};
// cq_prefetch: fetch_q reads in flight, one id each. A power of two, so the next id wraps by mask.
inline constexpr TridRange kPrefetchFetchQ{.first = kPrefetchExecBuf.end(), .count = 4};
// cq_prefetch: relay_linear's two scratch halves, one id each.
inline constexpr TridRange kPrefetchRelayLinear{.first = kPrefetchFetchQ.end(), .count = 2};

#if defined(ARCH_QUASAR)
// The Tensor prefetcher kernels on a dispatch engine: one id per kernel in tensor_prefetcher_kernels(), counted
// down from the top. A dispatch engine runs at most tensor_prefetcher_fifo::kMaxKernels of them: its 8
// data-movement cores less the 3 that the kernels of its one command queue take.
inline constexpr uint32_t kTensorPrefetcherMaxKernels = 8 - 3;
inline constexpr TridRange kTensorPrefetcher{
    .first = kMaxTrid + 1 - kTensorPrefetcherMaxKernels, .count = kTensorPrefetcherMaxKernels};
constexpr uint32_t tensor_prefetcher_trid(uint32_t kernel_index) { return kMaxTrid - kernel_index; }
#endif

constexpr bool disjoint(const TridRange& a, const TridRange& b) { return a.end() <= b.first || b.end() <= a.first; }
constexpr bool fits(const TridRange& r) { return r.first > 0 && r.end() <= kMaxTrid + 1; }

static_assert(fits(kPrefetchExecBuf) && fits(kPrefetchFetchQ) && fits(kPrefetchRelayLinear));
static_assert(disjoint(kPrefetchExecBuf, kPrefetchFetchQ) && disjoint(kPrefetchExecBuf, kPrefetchRelayLinear));
static_assert(disjoint(kPrefetchFetchQ, kPrefetchRelayLinear));
#if defined(ARCH_QUASAR)
static_assert(fits(kTensorPrefetcher));
static_assert(
    disjoint(kTensorPrefetcher, kPrefetchExecBuf) && disjoint(kTensorPrefetcher, kPrefetchFetchQ) &&
    disjoint(kTensorPrefetcher, kPrefetchRelayLinear));
#endif

}  // namespace dispatch_noc_trids
