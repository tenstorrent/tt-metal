// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <string>

#include <tt-metalium/session_id.hpp>

namespace tt::tt_metal::test {
namespace {

TEST(SessionIdTest, IsNotEmpty) { EXPECT_FALSE(get_or_create_session_id().empty()); }

TEST(SessionIdTest, IsStableAcrossCalls) { EXPECT_EQ(get_or_create_session_id(), get_or_create_session_id()); }

TEST(SessionIdTest, MintedValueIsLowercaseHex) {
    const char* from_env = std::getenv("TTNN_RUN_SESSION_ID");
    if (from_env != nullptr && *from_env != '\0') {
        GTEST_SKIP() << "TTNN_RUN_SESSION_ID is set, so the identifier is not minted here";
    }

    const std::string& session_id = get_or_create_session_id();
    EXPECT_EQ(session_id.size(), 32u);
    EXPECT_TRUE(std::all_of(session_id.begin(), session_id.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    }));
}

}  // namespace
}  // namespace tt::tt_metal::test
