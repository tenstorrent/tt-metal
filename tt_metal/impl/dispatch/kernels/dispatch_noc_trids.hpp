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
#include "tt_metal/impl/dispatch/kernels/cq_prefetch.hpp"
#if defined(ARCH_QUASAR)
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"
#include "tt_metal/impl/buffers/tensor_prefetcher_fifo.hpp"
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
inline constexpr TridRange kPrefetchFetchQ{
    .first = kPrefetchExecBuf.end(), .count = tt::tt_metal::PrefetchConstants::PREFETCH_MAX_OUTSTANDING_PCIE_READS};
// cq_prefetch: relay_linear's two scratch halves, one id each.
inline constexpr TridRange kPrefetchRelayLinear{.first = kPrefetchFetchQ.end(), .count = 2};

#if defined(ARCH_QUASAR)
// The Tensor prefetcher kernels on a dispatch engine: one id per kernel in tensor_prefetcher_kernels(), counted
// down from the top, for the most kernels a dispatch engine runs.
inline constexpr TridRange kTensorPrefetcher{
    .first = kMaxTrid + 1 - tt::tt_metal::tensor_prefetcher_fifo::kMaxKernels,
    .count = tt::tt_metal::tensor_prefetcher_fifo::kMaxKernels};
constexpr uint32_t tensor_prefetcher_trid(uint32_t kernel_index) { return kTensorPrefetcher.end() - 1 - kernel_index; }
#endif

// The command-queue ranges chain up from 1, each starting where the one before it ends, so only the top of the
// chain can run into the ids above it.
static_assert(kPrefetchRelayLinear.end() <= kMaxTrid + 1);
#if defined(ARCH_QUASAR)
static_assert(kPrefetchRelayLinear.end() <= kTensorPrefetcher.first);
#endif

}  // namespace dispatch_noc_trids
