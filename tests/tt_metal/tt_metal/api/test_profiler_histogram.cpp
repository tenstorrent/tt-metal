// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "impl/profiler/profiler_analysis.hpp"

namespace {

void expect_all_samples_in_range(
    const std::vector<uint64_t>& samples, uint64_t min_ns, uint64_t max_ns, uint32_t buckets) {
    const auto hist = tt::tt_metal::detail::make_quantized_histogram_ns(samples, min_ns, max_ns, buckets);
    const uint64_t counted = std::accumulate(hist.bucket_counts.begin(), hist.bucket_counts.end(), uint64_t{0});
    EXPECT_EQ(hist.underflow, 0u) << min_ns << ".." << max_ns << " buckets=" << buckets;
    EXPECT_EQ(hist.overflow, 0u) << min_ns << ".." << max_ns << " buckets=" << buckets;
    EXPECT_EQ(counted, samples.size()) << min_ns << ".." << max_ns << " buckets=" << buckets;
    EXPECT_LE(hist.min_ns, min_ns);
    EXPECT_GT(hist.max_ns, max_ns);
}

}  // namespace

constexpr uint32_t kDefaultBuckets = 10;
constexpr uint64_t kFastKernelNs = 1050;
constexpr uint64_t kSlowKernelNs = 2050;
constexpr std::array<uint64_t, 4> kMinValuesNs = {0, 37, 1050, 99999};
constexpr std::array<uint64_t, 7> kSpansNs = {0, 1, 999, 1000, 12345, 1000000, 1000000000000};
constexpr std::array<uint32_t, 3> kBucketCounts = {1, 3, 10};
constexpr uint64_t kCoveredMinNs = 3599;
constexpr uint64_t kCoveredMaxNs = 10089;
constexpr uint64_t kCoveredBucketWidthNs = 700;

TEST(ProfilerHistogram, CPU_AutoRangeCountsTheSlowestSample) {
    expect_all_samples_in_range({kFastKernelNs, kSlowKernelNs}, kFastKernelNs, kSlowKernelNs, kDefaultBuckets);
}

TEST(ProfilerHistogram, CPU_AutoRangeCoversMinAndMax) {
    for (uint64_t min_ns : kMinValuesNs) {
        for (uint64_t span : kSpansNs) {
            for (uint32_t buckets : kBucketCounts) {
                expect_all_samples_in_range({min_ns, min_ns + span}, min_ns, min_ns + span, buckets);
            }
        }
    }
}

TEST(ProfilerHistogram, CPU_AutoRangeKeepsBucketWidthWhenAlreadyCovered) {
    const auto hist = tt::tt_metal::detail::make_quantized_histogram_ns(
        {kCoveredMinNs, kCoveredMaxNs}, kCoveredMinNs, kCoveredMaxNs, kDefaultBuckets);
    ASSERT_EQ(hist.bucket_edges_ns.size(), kDefaultBuckets + 1);
    EXPECT_EQ(hist.bucket_edges_ns[1] - hist.bucket_edges_ns[0], kCoveredBucketWidthNs);
    EXPECT_EQ(hist.overflow, 0u);
}

TEST(ProfilerHistogram, CPU_MaxRangeDoesNotDivideByZero) {
    const uint64_t max_ns = std::numeric_limits<uint64_t>::max();
    const auto hist = tt::tt_metal::detail::make_quantized_histogram_ns({0, max_ns}, 0, max_ns, 1);
    EXPECT_GT(hist.bucket_edges_ns[1], hist.bucket_edges_ns[0]);
}
