// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <string>

#include <tt-metalium/run_id.hpp>

namespace tt::tt_metal::test {
namespace {

TEST(RunIdTest, IsNotEmpty) { EXPECT_FALSE(get_or_create_run_id().empty()); }

// Pairing a memory report with a performance report relies on every artefact from one process
// carrying the same value, so the identifier must be minted once rather than per call.
TEST(RunIdTest, IsStableAcrossCalls) { EXPECT_EQ(get_or_create_run_id(), get_or_create_run_id()); }

TEST(RunIdTest, MintedValueIsLowercaseHex) {
    const char* from_env = std::getenv("TT_METAL_RUN_ID");
    if (from_env != nullptr && *from_env != '\0') {
        GTEST_SKIP() << "TT_METAL_RUN_ID is set, so the identifier is not minted here";
    }

    const std::string& run_id = get_or_create_run_id();
    EXPECT_EQ(run_id.size(), 32u);
    EXPECT_TRUE(std::all_of(run_id.begin(), run_id.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    }));
}

}  // namespace
}  // namespace tt::tt_metal::test
