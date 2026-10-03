// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "mcast_host_test_common.hpp"

namespace ttnn::kernel_lib::host::test {
namespace {

void attach_for_inspection(const McastImpl& mcast, const McastConfig& cfg = {}) {
    tt::tt_metal::ProgramDescriptor descriptor;
    tt::tt_metal::KernelDescriptor kernel;
    kernel.core_ranges = mcast.participating_cores();
    kernel.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = cfg.noc};
    const std::array targets{std::ref(kernel)};
    mcast.attach(descriptor, "inspection", targets, 0);
}

void check_building_queries(const McastImpl& mcast) {
    EXPECT_NO_THROW(mcast.participating_cores());
    EXPECT_NO_THROW(mcast.sender_only_cores());
    std::vector<uint32_t> unchanged{42};
    EXPECT_ANY_THROW(mcast.append_compile_time_args_to(unchanged));
    EXPECT_ANY_THROW(mcast.append_runtime_args_to(unchanged, {0, 0}));
    EXPECT_EQ(unchanged, std::vector<uint32_t>{42});
}

}  // namespace

TEST_F(McastHostFixture, InvalidGroups) {
    const auto receivers = grid({2, 2}, {3, 2});
    GroupInput fixed(receivers, std::vector<CoreCoord>{{2, 2}});
    GroupInput rotating(grid({2, 3}, {3, 3}), std::vector<CoreCoord>{{2, 3}, {3, 3}});
    GroupInput longer(grid({2, 4}, {3, 4}), std::vector<CoreCoord>{{2, 4}, {3, 4}, {4, 4}});
    // Four isolated mapped destinations cannot be represented within the three-rectangle limit.
    GroupInput fragmented(cores({{0, 0}, {2, 0}, {4, 0}, {6, 0}}), std::vector<CoreCoord>{{0, 0}});
    EXPECT_THROW(compile_args(make_mcast(device_, {fragmented})), std::exception);
    EXPECT_ANY_THROW(McastImpl(*device_).add_group(CoreRangeSet{}, std::vector<CoreCoord>{{2, 2}}));
    EXPECT_ANY_THROW(McastImpl(*device_).add_group(CoreRangeSet{}, std::vector<CoreCoord>{{2, 2}, {3, 2}}));
    EXPECT_ANY_THROW(compile_args(make_mcast(device_, {})));
    EXPECT_ANY_THROW(McastImpl(*device_).add_group(receivers, std::vector<CoreCoord>{}));
    EXPECT_ANY_THROW(McastImpl(*device_).add_group(receivers, std::vector<CoreCoord>{{2, 2}, {2, 2}}));
    EXPECT_ANY_THROW(make_mcast(device_, {fixed, fixed}));
    EXPECT_ANY_THROW(make_mcast(device_, {fixed, rotating}));
    EXPECT_ANY_THROW(make_mcast(device_, {rotating, longer}));
    // Disjoint receivers are insufficient when groups share a sender.
    EXPECT_ANY_THROW(make_mcast(device_, {fixed, GroupInput(grid({4, 4}, {4, 4}), std::vector<CoreCoord>{{2, 2}})}));
    McastConfig cfg;
    EXPECT_ANY_THROW(compile_args(make_mcast(device_, {GroupInput(receivers, {{2, 2}}, 2)}, cfg)));
}

TEST_F(McastHostFixture, CollectionAndArgumentPreparationLifecycle) {
    McastImpl mcast(*device_);
    check_building_queries(mcast);
    EXPECT_TRUE(mcast.participating_cores().empty());
    EXPECT_ANY_THROW(attach_for_inspection(mcast));
    mcast.add_group(grid({2, 2}, {3, 2}), {{2, 2}});
    check_building_queries(mcast);
    EXPECT_ANY_THROW(mcast.add_group(CoreRangeSet{}, {{0, 0}}));
    EXPECT_ANY_THROW(mcast.add_group(grid({2, 3}, {3, 3}), {}));
    EXPECT_ANY_THROW(mcast.add_group(grid({2, 3}, {3, 3}), {{2, 3}, {2, 3}}));
    EXPECT_ANY_THROW(mcast.add_group(grid({2, 3}, {3, 3}), {{2, 3}, {3, 3}}));
    EXPECT_ANY_THROW(mcast.add_group(grid({2, 3}, {3, 3}), {{2, 2}}));
    mcast.add_group(grid({4, 2}, {4, 2}), {{4, 2}});
    EXPECT_EQ(mcast.participating_cores(), grid({2, 2}, {4, 2}));
    attach_for_inspection(mcast);
    EXPECT_EQ(allocated_semaphores(mcast).size(), 2u);
    const auto ct = compile_args(mcast);
    const auto rt = runtime_args(mcast, {2, 2});
    const auto* participants = &mcast.participating_cores();
    attach_for_inspection(mcast);
    EXPECT_EQ(compile_args(mcast), ct);
    EXPECT_EQ(runtime_args(mcast, {2, 2}), rt);
    EXPECT_EQ(&mcast.participating_cores(), participants);
    EXPECT_ANY_THROW(mcast.add_group(grid({6, 2}, {6, 2}), {{6, 2}}));
    EXPECT_EQ(mcast.participating_cores().num_cores(), 3u);
}

TEST_F(McastHostFixture, LateTransportSelection) {
    // A late irregular group changes the common transport of the earlier dense group.
    McastImpl late(*device_, chain_config());
    late.add_group(grid({0, 0}, {2, 0}), {{1, 0}});
    check_building_queries(late);
    late.add_group(cores({{0, 2}, {2, 2}}), {{0, 2}});
    attach_for_inspection(late);
    EXPECT_EQ(wire::transfer_mode(emitted_metadata(compile_args(late)).mcast.flags), TransferMode::ChainUnicast);
    EXPECT_EQ(emitted_metadata(compile_args(late)).mcast.rectangle_capacity, 0u);
    EXPECT_EQ(runtime_args(late, {1, 0}).size(), 8u);
    EXPECT_EQ(runtime_args(late, {2, 2}).size(), 8u);
}

TEST_F(McastHostFixture, McastValueSemanticsAndConfigSnapshot) {
    McastConfig cfg;
    cfg.noc = NOC::NOC_1;
    McastImpl building(*device_, cfg);
    cfg.noc = NOC::NOC_0;
    building.add_group(grid({2, 2}, {2, 2}), {{2, 2}});
    auto copy = building;
    copy.add_group(grid({4, 2}, {5, 2}), {{4, 2}});
    const McastConfig original_cfg{.noc = NOC::NOC_1};
    attach_for_inspection(copy, original_cfg);
    attach_for_inspection(building, original_cfg);
    EXPECT_EQ(building.participating_cores().num_cores(), 1u);
    EXPECT_EQ(copy.participating_cores().num_cores(), 3u);
    EXPECT_EQ(emitted_semaphore(compile_args(copy), wire::DATA_READY), 0u);
    EXPECT_NE(emitted_metadata(compile_args(copy)).mcast.flags & wire::NOC1, 0u);
    EXPECT_EQ(emitted_metadata(compile_args(building)).mcast.has_remote_receivers, 0u);
    EXPECT_EQ(emitted_metadata(compile_args(copy)).mcast.has_remote_receivers, 1u);
    const auto ct = compile_args(copy);
    const auto rt = runtime_args(copy, {4, 2});
    auto moved = [&] {
        auto original = copy;
        return McastImpl(std::move(original));
    }();
    copy = building;
    attach_for_inspection(moved, original_cfg);
    EXPECT_EQ(compile_args(moved), ct);
    EXPECT_EQ(runtime_args(moved, {4, 2}), rt);
    McastImpl assigned(*device_);
    assigned = moved;
    EXPECT_EQ(runtime_args(assigned, {4, 2}), rt);
    auto moving_building = McastImpl(*device_);
    moving_building.add_group(grid({4, 2}, {5, 2}), {{4, 2}});
    assigned = std::move(moving_building);
    check_building_queries(assigned);
    attach_for_inspection(assigned);
    EXPECT_EQ(assigned.participating_cores(), grid({4, 2}, {5, 2}));
}

TEST_F(McastHostFixture, ChainRejectsUnsupportedProtocolsAndGeometry) {
    const auto receivers = cores({{0, 0}, {2, 0}});
    const GroupInput group(receivers, {{0, 0}});
    McastConfig config;
    config.handshake = false;
    EXPECT_ANY_THROW(compile_args(make_mcast(device_, {group}, chain_config(config))));
    config.handshake = true;
    for (uint32_t ack : {0u, 1u}) {
        EXPECT_ANY_THROW(compile_args(make_mcast(device_, {GroupInput(receivers, {{0, 0}}, ack)}, chain_config())));
    }
    EXPECT_ANY_THROW(compile_args(make_mcast(device_, {GroupInput(receivers, {{0, 0}, {2, 0}})}, chain_config())));
    EXPECT_ANY_THROW(compile_args(
        make_mcast(device_, {GroupInput(cores({{0, 0}, {2, 0}, {4, 0}, {6, 0}}), {{0, 0}})}, chain_config())));
    EXPECT_ANY_THROW(make_mcast(device_, {group, group}, chain_config()));
}

}  // namespace ttnn::kernel_lib::host::test
