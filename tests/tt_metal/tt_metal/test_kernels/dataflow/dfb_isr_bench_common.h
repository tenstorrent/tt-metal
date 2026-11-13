// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared helpers for the dfb_isr_timing_bench kernels.
//
// Measurement is done from emulator waveforms, so these kernels carry NO timing instrumentation:
// they only run the stream. Each kernel binds up to kMaxBenchDfbs DFBs named "d00".."d11"; the
// generated get_token_if_present<name>() getter tells which are bound, so one source serves any
// count. All DFBs of one kernel have the same access-pattern pair (the host guarantees it); the typed
// binding tokens carry it. The bench uses STRIDED producers and STRIDED or ALL consumers only.

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

#include "api/dataflow/dataflow_buffer.h"
#include "internal/template_string.h"

namespace isr_bench {

constexpr std::size_t kMaxBenchDfbs = 12;

// Fixed-length names so they can sit in one constexpr array and be indexed at compile time.
constexpr ::internal::TemplateString kDfbNames[kMaxBenchDfbs] = {
    "d00", "d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09", "d10", "d11"};

template <std::size_t I>
constexpr bool dfb_bound() {
    return dfb::get_token_if_present<kDfbNames[I]>() != nullptr;
}

// Number of DFBs bound to this kernel: d00..d{N-1} must be contiguous.
template <std::size_t I = 0>
constexpr std::size_t count_bound_dfbs() {
    if constexpr (I < kMaxBenchDfbs) {
        if constexpr (dfb_bound<I>()) {
            return 1 + count_bound_dfbs<I + 1>();
        } else {
            return 0;
        }
    } else {
        return 0;
    }
}

template <std::size_t I>
constexpr auto token() {
    return *dfb::get_token_if_present<kDfbNames[I]>();
}

// One DataflowBuffer per bound DFB, constructed in place (the originals drain in their destructors).
template <std::size_t... I>
auto make_dfbs(std::index_sequence<I...>) {
    return std::tuple<decltype(DataflowBuffer(token<I>()))...>(token<I>()...);
}

// Calls f(dfb, index) for every DFB in the tuple, in index order.
template <typename Tuple, typename F, std::size_t... I>
FORCE_INLINE void for_each_dfb_impl(Tuple& t, F&& f, std::index_sequence<I...>) {
    (f(std::get<I>(t), static_cast<uint32_t>(I)), ...);
}

template <typename Tuple, typename F>
FORCE_INLINE void for_each_dfb(Tuple& t, F&& f) {
    for_each_dfb_impl(t, f, std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

}  // namespace isr_bench
