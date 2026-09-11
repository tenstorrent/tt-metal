// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "ttnn/operations/experimental/fft/device/kernels/dataflow/bluestein_streaming_common.h"

namespace {

TEST(BluesteinStreaming, PartialAndPaddedChunks) {
    EXPECT_EQ(bluestein_streaming::chunk_count(65537), 65u);
    EXPECT_EQ(bluestein_streaming::chunk_count(262144), 256u);
    EXPECT_EQ(bluestein_streaming::valid_elements(65537, 63), 1024u);
    EXPECT_EQ(bluestein_streaming::valid_elements(65537, 64), 1u);
    EXPECT_EQ(bluestein_streaming::valid_elements(65537, 65), 0u);
    EXPECT_EQ(bluestein_streaming::valid_elements(65537, 255), 0u);
}

TEST(BluesteinStreaming, ExactCoverageAndDmaBounds) {
    constexpr std::array sizes{1u, 1023u, 1024u, 1025u, 16385u, 65532u, 65533u, 65537u, 262144u, 1048576u};
    for (const uint32_t size : sizes) {
        SCOPED_TRACE(size);
        const uint32_t chunks = bluestein_streaming::chunk_count(size);
        uint32_t covered = 0;
        for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
            const uint32_t valid = bluestein_streaming::valid_elements(size, chunk);
            const uint32_t offset = chunk * bluestein_streaming::chunk_elements * sizeof(float);
            EXPECT_GT(valid, 0u);
            EXPECT_LE(valid, bluestein_streaming::chunk_elements);
            EXPECT_EQ(offset % 32u, 0u);
            EXPECT_LE(offset + valid * sizeof(float), size * sizeof(float));
            covered += valid;
        }
        EXPECT_EQ(covered, size);
        EXPECT_EQ(bluestein_streaming::valid_elements(size, chunks), 0u);
    }
}

}  // namespace
