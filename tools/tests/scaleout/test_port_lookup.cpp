// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <board/port_lookup.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::scaleout_tools {
namespace {

struct PortLookupCase {
    BoardType board_type;
    uint32_t asic_location;
    uint8_t channel;
    tt::tt_metal::PortType expected_port_type;
    uint32_t expected_port_id;
};

class PortLookupTest : public ::testing::TestWithParam<PortLookupCase> {};

TEST_P(PortLookupTest, ResolvePortTypeMatchesBoardTopology) {
    const auto& test_case = GetParam();
    EXPECT_EQ(
        resolve_port_type(test_case.board_type, test_case.asic_location, test_case.channel),
        test_case.expected_port_type);

    auto board = create_board(test_case.board_type);
    EXPECT_EQ(resolve_port_id(board, test_case.asic_location, test_case.channel), PortId{test_case.expected_port_id});
}

INSTANTIATE_TEST_SUITE_P(
    KnownBoardChannels,
    PortLookupTest,
    ::testing::Values(
        // N150
        PortLookupCase{BoardType::N150, 0, 6, tt::tt_metal::PortType::QSFP_DD, 1},
        PortLookupCase{BoardType::N150, 0, 0, tt::tt_metal::PortType::QSFP_DD, 2},
        PortLookupCase{BoardType::N150, 0, 14, tt::tt_metal::PortType::WARP100, 1},
        // N300
        PortLookupCase{BoardType::N300, 0, 8, tt::tt_metal::PortType::TRACE, 1},
        PortLookupCase{BoardType::N300, 1, 0, tt::tt_metal::PortType::TRACE, 2},
        PortLookupCase{BoardType::N300, 0, 0, tt::tt_metal::PortType::QSFP_DD, 2},
        // UBB_BLACKHOLE (Galaxy tray)
        PortLookupCase{BoardType::UBB_BLACKHOLE, 5, 0, tt::tt_metal::PortType::TRACE, 1},
        PortLookupCase{BoardType::UBB_BLACKHOLE, 1, 2, tt::tt_metal::PortType::QSFP_DD, 2},
        PortLookupCase{BoardType::UBB_BLACKHOLE, 5, 6, tt::tt_metal::PortType::LINKING_BOARD_1, 1},
        PortLookupCase{BoardType::UBB_BLACKHOLE, 8, 4, tt::tt_metal::PortType::LINKING_BOARD_3, 1}));

TEST(PortLookupTest, UnknownChannelReturnsUnknownPortType) {
    EXPECT_EQ(resolve_port_type(BoardType::N150, 0, 42), tt::tt_metal::PortType::UNKNOWN);
    EXPECT_FALSE(try_get_port(BoardType::N150, 0, 42).has_value());
}

TEST(PortLookupTest, ResolvePortIdReturnsZeroForUnknownChannel) {
    auto board = create_board(BoardType::N300);
    EXPECT_EQ(resolve_port_id(board, 0, 42), PortId{0});
    EXPECT_FALSE(try_get_port_id(board, 0, 42).has_value());
}

}  // namespace
}  // namespace tt::scaleout_tools
