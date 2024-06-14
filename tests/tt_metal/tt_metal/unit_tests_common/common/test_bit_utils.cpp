// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bit_utils.h"
#include <gtest/gtest.h>
#include <cstdint>

TEST(NoFixture, ExtractBitArray) {
    uint32_t src[4] = {0x12345678, 0x9abcdef0, 0x13579bdf, 0x2468ace0};
    // 1. Extract the 20-bit elements from the 32-bit source array.
    uint32_t dest[4];
    extract_bit_array(src, 20, dest, 4);

    EXPECT_EQ(dest[0], 0x45678);
    EXPECT_EQ(dest[1], 0xf0123);
    EXPECT_EQ(dest[2], 0xabcde);
    EXPECT_EQ(dest[3], 0x9bdf9);

    // 2. Extract 16-bit elements
    extract_bit_array(src, 16, dest, 4);
    EXPECT_EQ(dest[0], 0x5678);
    EXPECT_EQ(dest[1], 0x1234);
    EXPECT_EQ(dest[2], 0xdef0);
    EXPECT_EQ(dest[3], 0x9abc);
}

TEST(NoFixture, PackBitArray) {
    uint32_t src[8] = { 1, 2, 3, 4, 5, 6, 7, 7 };
    uint32_t dest[8];

    // 1. Pack as 4-bit elements
    pack_bit_array(src, 4, dest, 8);
    EXPECT_EQ(dest[0], 0x77654321);

    // 2. Pack as 8-bit elements
    pack_bit_array(src, 8, dest, 8);
    EXPECT_EQ(dest[0], 0x04030201);
    EXPECT_EQ(dest[1], 0x07070605);


    // 3. Pack as 16-bit elements
    pack_bit_array(src, 16, dest, 8);
    EXPECT_EQ(dest[0], 0x00020001);
    EXPECT_EQ(dest[1], 0x00040003);
    EXPECT_EQ(dest[2], 0x00060005);
    EXPECT_EQ(dest[3], 0x00070007);


    // 4. Pack as 3-bit elements
    uint32_t expected=0;
    for (int i = 0; i < 8; i++) {
        expected |= (src[i] & 0x7) << (i * 3);
    }
    pack_bit_array(src, 3, dest, 8);
    EXPECT_EQ(dest[0], expected);
}

TEST(NoFixture, PackExtractBitArray) {
    uint32_t src[8] = { 1, 2, 3, 4, 5, 6, 7, 7 };

    for (uint num_pack_bits = 3; num_pack_bits <= 31; num_pack_bits++) {
        uint32_t dest[8];
        pack_bit_array(src, num_pack_bits, dest, 8);
        uint32_t extracted[8];
        extract_bit_array(dest, num_pack_bits, extracted, 8);
        for (int i = 0; i < 8; i++) {
            EXPECT_EQ(src[i], extracted[i]);
        }
    }
}

TEST(NoFixture, ExtractPackBitArray) {
    uint32_t src[4] = { 0x12345678, 0x9abcdef0, 0x13579bdf, 0x2468ace0 };

    // Compute the number of 3-bit elements that can be packed into 4 x 32-bit elements
    const uint32_t num_3_bit_elements = (4 * 32) / 3;
    uint32_t dest[num_3_bit_elements];

    for (uint num_pack_bits = 3; num_pack_bits <= 31; num_pack_bits++) {
        const uint32_t num_dest_elements = (4 * 32) / num_pack_bits;

        extract_bit_array(src, num_pack_bits, dest, num_dest_elements);
        uint32_t packed[4];
        pack_bit_array(dest, num_pack_bits, packed, num_dest_elements);

        // If the bit length of src is not evenly divisible by num_pack_bits
        // then the last element after packing back won't equal the original.
        bool has_partial = (num_pack_bits * num_dest_elements) % 32 != 0;
        for (int i = 0; i < 4 - has_partial; i++) {
            EXPECT_EQ(src[i], packed[i]);
        }
    }
}
