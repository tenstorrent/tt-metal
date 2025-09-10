// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <random>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#define MAX_HOPS 100
#include "test_msgs.h"
#include "test_msgs_intf.h"
#include "test_msgs_impl.h"

namespace {

void verify_equal(const ::PacketInfo& expected, types::PacketInfo::ConstView view) {
    EXPECT_EQ(view.size(), sizeof expected);
    EXPECT_EQ(view.len(), expected.len);
    EXPECT_EQ(view.trace_route().size(), MAX_HOPS);
    size_t i = 0;
    for (auto hop : view.trace_route()) {
        const auto& hop_expected = expected.trace_route[i];
        auto ts = hop.timestamp();
        EXPECT_EQ(ts.year(), hop_expected.timestamp.year);
        EXPECT_EQ(ts.month(), hop_expected.timestamp.month);
        EXPECT_EQ(ts.day(), hop_expected.timestamp.day);
        EXPECT_EQ(ts.hour(), hop_expected.timestamp.hour);
        EXPECT_EQ(ts.minute(), hop_expected.timestamp.minute);
        EXPECT_EQ(ts.second(), hop_expected.timestamp.second);
        EXPECT_EQ(ts.timezone_id(), hop_expected.timestamp.timezone_id);
        EXPECT_THAT(hop.addr(), testing::ElementsAreArray(hop_expected.addr));
        ++i;
    }
}

}  // namespace

TEST(CodegenTest, CodegenTest) {
    auto factory = create_factory();
    types::PacketInfo info = factory.create<types::PacketInfo>();
    ASSERT_EQ(info.size(), sizeof(::PacketInfo));
    ::PacketInfo* raw_ptr = reinterpret_cast<::PacketInfo*>(info.data());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis(0, 255);
    std::generate_n(info.data(), info.size(), [&]() { return std::byte{dis(gen)}; });

    // Verify that reading fields with accessor matches raw data
    types::PacketInfo::ConstView const_view = info.view();
    verify_equal(*raw_ptr, const_view);

    // Alter the content using accessor
    auto mutable_view = info.view();
    mutable_view.len() = 42;
    for (auto hop : mutable_view.trace_route()) {
        auto ts = hop.timestamp();
        ts.year() = 2024;
        ts.month() = 6;
        ts.day() = 24;
        ts.timezone_id() = dis(gen);
        auto addr = hop.addr();
        std::fill_n(addr.begin(), addr.size(), dis(gen));
    }
    // Verify that after modification, the view reflects the changes
    verify_equal(*raw_ptr, const_view);
    for (auto hop : mutable_view.trace_route()) {
        auto ts = hop.timestamp();
        EXPECT_EQ(ts.year(), 2024);
        EXPECT_EQ(ts.month(), 6);
        EXPECT_EQ(ts.day(), 24);
    }
}
