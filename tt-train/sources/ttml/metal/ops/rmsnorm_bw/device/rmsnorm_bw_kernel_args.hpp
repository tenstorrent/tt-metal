// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Circular buffer indices and kernel argument slots shared by the rmsnorm_bw program factory and its reader, writer
// and compute kernels. Included from both host and device code, so it holds only plain constants.

#pragma once

#include <cstddef>
#include <cstdint>

namespace ttml::metal::ops::rmsnorm_bw::device::kernel_args {

template <std::size_t N>
constexpr bool all_distinct(const uint32_t (&values)[N]) {
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            if (values[i] == values[j]) {
                return false;
            }
        }
    }
    return true;
}

template <std::size_t N>
constexpr bool all_below(const uint32_t (&values)[N], const std::size_t limit) {
    for (std::size_t i = 0; i < N; ++i) {
        if (values[i] >= limit) {
            return false;
        }
    }
    return true;
}

// Every slot in [0, N) is used exactly once.
template <std::size_t N>
constexpr bool fills_slots(const uint32_t (&slots)[N]) {
    return all_below(slots, N) && all_distinct(slots);
}

namespace cb {
// Inputs
constexpr uint32_t kInput = 0U;
constexpr uint32_t kMaskW = 1U;
constexpr uint32_t kScaler = 2U;
constexpr uint32_t kGamma = 3U;
constexpr uint32_t kRms = 4U;
constexpr uint32_t kDLdout = 5U;
constexpr uint32_t kMatMulReduce = 6U;
constexpr uint32_t kZero = 7U;
// Outputs
constexpr uint32_t kDLda = 8U;
constexpr uint32_t kDgammaComponents = 9U;
// Intermediates
constexpr uint32_t kRecipRms = 10U;
constexpr uint32_t kScale = 11U;
constexpr uint32_t kExternalSums = 12U;
constexpr uint32_t kMaskH = 13U;
constexpr uint32_t kDgammaRow = 14U;

// Hardware limit on circular buffers per core.
constexpr uint32_t kMaxCbs = 32U;
constexpr uint32_t kAll[] = {
    kInput,
    kMaskW,
    kScaler,
    kGamma,
    kRms,
    kDLdout,
    kMatMulReduce,
    kZero,
    kDLda,
    kDgammaComponents,
    kRecipRms,
    kScale,
    kExternalSums,
    kMaskH,
    kDgammaRow};
}  // namespace cb

static_assert(all_distinct(cb::kAll), "rmsnorm_bw circular buffer indices must be distinct");
static_assert(all_below(cb::kAll, cb::kMaxCbs), "rmsnorm_bw circular buffer index out of range");

// Reader compile-time args. The input, gamma, rms, dL_dout and external sums tensor accessor args follow, in that
// order, from kNumArgs.
namespace reader_ct {
constexpr uint32_t kPackedScaler = 0U;
constexpr uint32_t kBlockSize = 1U;
constexpr uint32_t kMaskW = 2U;
constexpr uint32_t kWt = 3U;
constexpr uint32_t kNumExternalTiles = 4U;
constexpr uint32_t kMaskH = 5U;
constexpr uint32_t kNumArgs = 6U;
constexpr uint32_t kAll[] = {kPackedScaler, kBlockSize, kMaskW, kWt, kNumExternalTiles, kMaskH};
static_assert(sizeof(kAll) / sizeof(kAll[0]) == kNumArgs && fills_slots(kAll), "rmsnorm_bw reader compile-time args");
}  // namespace reader_ct

namespace reader_rt {
constexpr uint32_t kInputAddr = 0U;
constexpr uint32_t kGammaAddr = 1U;
constexpr uint32_t kRmsAddr = 2U;
constexpr uint32_t kDLdoutAddr = 3U;
constexpr uint32_t kNumRows = 4U;
constexpr uint32_t kStartRow = 5U;
constexpr uint32_t kExternalSumsAddr = 6U;
constexpr uint32_t kNumArgs = 7U;
constexpr uint32_t kAll[] = {kInputAddr, kGammaAddr, kRmsAddr, kDLdoutAddr, kNumRows, kStartRow, kExternalSumsAddr};
static_assert(sizeof(kAll) / sizeof(kAll[0]) == kNumArgs && fills_slots(kAll), "rmsnorm_bw reader runtime args");
}  // namespace reader_rt

// Writer compile-time args. The dL_da and dgamma tensor accessor args follow, in that order, from kNumArgs.
namespace writer_ct {
constexpr uint32_t kBlockSize = 0U;
constexpr uint32_t kWt = 1U;
constexpr uint32_t kNumArgs = 2U;
constexpr uint32_t kAll[] = {kBlockSize, kWt};
static_assert(sizeof(kAll) / sizeof(kAll[0]) == kNumArgs && fills_slots(kAll), "rmsnorm_bw writer compile-time args");
}  // namespace writer_ct

namespace writer_rt {
constexpr uint32_t kDLdaAddr = 0U;
constexpr uint32_t kDgammaAddr = 1U;
constexpr uint32_t kNumRows = 2U;
constexpr uint32_t kStartRow = 3U;
constexpr uint32_t kDgammaRow = 4U;  // row of this core in the on-chip dgamma partials
constexpr uint32_t kNumArgs = 5U;
constexpr uint32_t kAll[] = {kDLdaAddr, kDgammaAddr, kNumRows, kStartRow, kDgammaRow};
static_assert(sizeof(kAll) / sizeof(kAll[0]) == kNumArgs && fills_slots(kAll), "rmsnorm_bw writer runtime args");
}  // namespace writer_rt

namespace compute_ct {
constexpr uint32_t kNumRowsPerCore = 0U;
constexpr uint32_t kBlockSize = 1U;
constexpr uint32_t kMaskW = 2U;
constexpr uint32_t kWt = 3U;
constexpr uint32_t kNumExternalTiles = 4U;
constexpr uint32_t kHt = 5U;  // tile rows per sequence
constexpr uint32_t kNumArgs = 6U;
constexpr uint32_t kAll[] = {kNumRowsPerCore, kBlockSize, kMaskW, kWt, kNumExternalTiles, kHt};
static_assert(sizeof(kAll) / sizeof(kAll[0]) == kNumArgs && fills_slots(kAll), "rmsnorm_bw compute compile-time args");
}  // namespace compute_ct

namespace compute_rt {
constexpr uint32_t kStartRow = 0U;
constexpr uint32_t kNumArgs = 1U;
constexpr uint32_t kAll[] = {kStartRow};
static_assert(sizeof(kAll) / sizeof(kAll[0]) == kNumArgs && fills_slots(kAll), "rmsnorm_bw compute runtime args");
}  // namespace compute_rt

}  // namespace ttml::metal::ops::rmsnorm_bw::device::kernel_args
