// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>

#pragma push_macro("FORCE_INLINE")
#undef FORCE_INLINE
#define FORCE_INLINE inline
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/sequence_accessor.hpp"
#pragma pop_macro("FORCE_INLINE")

namespace {
template <uint32_t Primary, uint32_t Joint, uint32_t Chunk>
void check_pages() {
    const auto accessor = sequence_accessor<Primary, Joint, Chunk>(0u, 1u);
    const uint32_t segment_rows[] = {Primary, Joint};
    uint32_t page = 0;
    for (uint32_t head = 0; head < 3; ++head) {
        for (uint32_t segment = 0; segment < 2; ++segment) {
            const uint32_t rows = segment_rows[segment];
            const uint32_t tiles = (rows + 31) / 32;
            for (uint32_t tile = 0; tile < tiles; ++tile) {
                for (uint32_t col = 0; col < 4; ++col, ++page) {
                    uint32_t calls = 0;
                    EXPECT_TRUE(accessor.visit(page, [&](uint32_t source, uint32_t address) {
                        ++calls;
                        EXPECT_EQ(source, segment);
                        EXPECT_EQ(address, (head * tiles + tile) * 4 + col);
                    }));
                    EXPECT_EQ(calls, 1u);
                    EXPECT_EQ(accessor.valid_rows(page), std::min(rows - tile * 32, 32u));
                }
            }
        }
        while (page < (head + 1) * accessor.head_pages) {
            EXPECT_FALSE(accessor.visit(page, [](auto, auto) { ADD_FAILURE() << "Visited padding page"; }));
            EXPECT_EQ(accessor.valid_rows(page), 0u);
            ++page;
        }
    }
}

TEST(SDPASequenceAccessor, DenseAndJointPadding) {
    check_pages<512, 0, 512>();
    check_pages<1, 0, 256>();
    check_pages<513, 0, 512>();
    check_pages<1, 1, 256>();
    check_pages<15, 17, 512>();
    check_pages<31, 33, 256>();
    check_pages<255, 1, 512>();
    check_pages<257, 31, 256>();
    check_pages<511, 1, 512>();
    check_pages<512, 17, 256>();
    check_pages<767, 33, 512>();
}
}  // namespace
