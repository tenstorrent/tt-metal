// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/test_utils/int8.hpp"
#include "tt_metal/test_utils/uint8.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

namespace tt::test_utils {
namespace {

TEST(IntegerTestUtils, PacksSignedBytesLeastSignificantFirst) {
    EXPECT_EQ(pack_four_int8_into_uint32(1, 2, 3, 4), 0x04030201u);
    EXPECT_EQ(pack_four_int8_into_uint32(-128, -1, 0, 127), 0x7f00ff80u);
}

TEST(IntegerTestUtils, PacksUnsignedBytesLeastSignificantFirst) {
    EXPECT_EQ(pack_four_uint8_into_uint32(1, 2, 3, 4), 0x04030201u);
    EXPECT_EQ(pack_four_uint8_into_uint32(0, 127, 128, 255), 0xff807f00u);
}

TEST(IntegerTestUtils, CreatesReproduciblePackedSignedBuffers) {
    const auto data = create_random_vector_of_int8(4096, 42);
    ASSERT_EQ(data.size(), 1024u);
    EXPECT_EQ(data, create_random_vector_of_int8(4096, 42));
    EXPECT_NE(data, create_random_vector_of_int8(4096, 43));
    EXPECT_TRUE(create_random_vector_of_int8(0, 42).empty());
    for (uint32_t word : data) {
        for (int byte = 0; byte < 4; ++byte) {
            EXPECT_NE((word >> (8 * byte)) & 0xffu, 0x80u);
        }
    }
}

TEST(IntegerTestUtils, CreatesReproduciblePackedUnsignedBuffers) {
    const auto data = create_random_vector_of_uint8(4096, 42);
    ASSERT_EQ(data.size(), 1024u);
    EXPECT_EQ(data, create_random_vector_of_uint8(4096, 42));
    EXPECT_NE(data, create_random_vector_of_uint8(4096, 43));
    EXPECT_TRUE(create_random_vector_of_uint8(0, 42).empty());
}

TEST(IntegerTestUtils, RejectsIncompleteWords) {
    for (size_t num_bytes : {1u, 2u, 3u, 5u, 6u, 7u}) {
        EXPECT_THROW((void)create_random_vector_of_int8(num_bytes, 42), std::runtime_error);
        EXPECT_THROW((void)create_random_vector_of_uint8(num_bytes, 42), std::runtime_error);
    }
}

}  // namespace
}  // namespace tt::test_utils
