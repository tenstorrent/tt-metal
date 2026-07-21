// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <string_view>

#include <gtest/gtest.h>

#include "common/stable_hash.hpp"

namespace {

constexpr uint64_t kUint64VectorInput = 0x0123456789abcdefULL;
constexpr uint64_t kUint64VectorDigest = 0x56e446078535eb02ULL;

constexpr std::array<uint8_t, 5> kEmbeddedNulInput = {'a', 0, 'b', 0, 'c'};
constexpr uint64_t kEmbeddedNulDigest = 0x344008a172774148ULL;

constexpr std::string_view kMultiPart1 = "part1";
constexpr std::array<uint8_t, 7> kMultiPart2 = {0x00, 0xff, 'p', 'a', 'r', 't', '2'};
constexpr uint64_t kMultiPart3 = 0x0102030405060708ULL;
constexpr uint64_t kMultiUpdateDigest = 0x137a5e70bf72ee9fULL;

}  // namespace

TEST(StableHasherTest, Uint64HashMatchesCanonicalVector) {
    tt::StableHasher hasher;
    hasher.update(kUint64VectorInput);
    EXPECT_EQ(hasher.digest(), kUint64VectorDigest);
}

TEST(StableHasherTest, EmbeddedNulBytesHashMatchesCanonicalVector) {
    tt::StableHasher hasher;
    hasher.update(kEmbeddedNulInput.data(), kEmbeddedNulInput.size());
    EXPECT_EQ(hasher.digest(), kEmbeddedNulDigest);
}

TEST(StableHasherTest, MultiUpdateHashMatchesCanonicalVector) {
    tt::StableHasher hasher;
    hasher.update(kMultiPart1);
    hasher.update(kMultiPart2.data(), kMultiPart2.size());
    hasher.update(kMultiPart3);
    EXPECT_EQ(hasher.digest(), kMultiUpdateDigest);
}

TEST(StableHasherTest, MultiUpdateAndSingleUpdateProduceSameDigest) {
    tt::StableHasher multi_update;
    multi_update.update(kMultiPart1);
    multi_update.update(kMultiPart2.data(), kMultiPart2.size());
    multi_update.update(kMultiPart3);

    constexpr std::array<uint8_t, kMultiPart1.size() + kMultiPart2.size() + sizeof(uint64_t)> kConcatenated = {
        'p', 'a', 'r', 't', '1', 0x00, 0xff, 'p', 'a', 'r', 't', '2', 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01};

    tt::StableHasher single_update;
    single_update.update(kConcatenated.data(), kConcatenated.size());

    EXPECT_EQ(multi_update.digest(), single_update.digest());
}
