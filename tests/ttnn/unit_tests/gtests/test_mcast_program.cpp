// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "mcast_host_test_common.hpp"

namespace ttnn::kernel_lib::host::test {
TEST_F(McastHostFixture, ProgramBindingAllocatesOnceAndAppendsResolvedIds) {
    using namespace tt::tt_metal;
    const auto participants = grid({0, 0}, {1, 0});
    auto mcast = make_mcast(device_, {{participants, {{0, 0}}}});
    Program program;
    program.impl().add_semaphore(participants, 0, 7, tt::CoreType::WORKER);
    program.impl().add_semaphore(participants, 2, 9, tt::CoreType::WORKER);
    const auto ct = attach_unplaced(mcast, program, {71});
    const auto rt = attach_on_core(mcast, program, {0, 0}, {73});
    EXPECT_EQ(emitted_semaphore(ct, wire::DATA_READY, 1), 1u);
    EXPECT_EQ(emitted_semaphore(ct, wire::CONSUMER_READY, 1), 3u);
    EXPECT_EQ(rt.front(), 73u);
    EXPECT_GT(rt.size(), 1u);
    ASSERT_EQ(program.impl().semaphores().size(), 4u);
    EXPECT_EQ(program.impl().semaphores()[0].initial_value(), 7u);
    // Later attachments, including through a copy, reuse the bound Program's semaphores.
    attach_unplaced(mcast, program);
    auto copy = mcast;
    attach_unplaced(copy, program);
    EXPECT_EQ(program.impl().semaphores().size(), 4u);
    Program other;
    EXPECT_ANY_THROW(attach_unplaced(copy, other));
    EXPECT_TRUE(other.impl().semaphores().empty());
    Program moved = std::move(program);
    EXPECT_ANY_THROW(attach_unplaced(mcast, moved));
    EXPECT_EQ(moved.impl().semaphores().size(), 4u);
    ProgramDescriptor descriptor;
    KernelDescriptor kernel;
    kernel.core_ranges = participants;
    EXPECT_ANY_THROW(mcast.attach(descriptor, "weights", std::array{std::ref(kernel)}, 0));
    tt::tt_metal::experimental::ProgramSpec spec;
    tt::tt_metal::experimental::ProgramRunArgs run_args;
    EXPECT_ANY_THROW(mcast.attach(spec, run_args, "weights", {}));
}

TEST_F(McastHostFixture, ProgramBindingCoversChainExhaustionAndUnsupportedPrograms) {
    using namespace tt::tt_metal;
    auto chain = make_mcast(device_, {{cores({{0, 0}, {1, 1}, {2, 0}}), {{0, 0}}}}, chain_config());
    Program chain_program;
    const auto ct = attach_unplaced(chain, chain_program);
    ASSERT_EQ(chain_program.impl().semaphores().size(), 3u);
    EXPECT_EQ(emitted_semaphore(ct, wire::SIGNAL_SOURCE), 2u);

    const auto participants = grid({0, 0}, {1, 0});
    auto mcast = make_mcast(device_, {{participants, {{0, 0}}}});
    Program full;
    for (uint32_t id = 0; id < NUM_SEMAPHORES; ++id) {
        full.impl().add_semaphore(participants, id, 0, tt::CoreType::WORKER);
    }
    EXPECT_ANY_THROW(attach_unplaced(mcast, full));
    EXPECT_EQ(full.impl().semaphores().size(), NUM_SEMAPHORES);
    ProgramDescriptor descriptor;
    KernelDescriptor kernel;
    kernel.kernel_source = "void kernel_main() {}";
    kernel.source_type = KernelDescriptor::SourceType::SOURCE_CODE;
    kernel.core_ranges = grid({0, 0}, {0, 0});
    kernel.config = ReaderConfigDescriptor{};
    descriptor.kernels.push_back(kernel);
    Program compiled(descriptor);
    compiled.impl().compile(device_);
    ASSERT_TRUE(compiled.impl().is_compiled());
    EXPECT_ANY_THROW(attach_unplaced(mcast, compiled));
    EXPECT_TRUE(compiled.impl().semaphores().empty());
}

TEST_F(McastHostFixture, ProgramAttachMeasuresRuntimePrefixAndAppendsToEveryCore) {
    using namespace tt::tt_metal;
    const auto receivers = grid({0, 0}, {2, 0});
    // The default fixed sender is the first receiver, core (0, 0).
    Mcast mcast(*device_, McastConfig{.noc = NOC::NOC_0}, receivers, receivers.num_cores());
    Program program;
    DataMovementConfig sender{.noc = NOC::NOC_0, .compile_args = {11, 12}, .named_compile_args = {{"other", 5}}};
    std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> sender_rt{{{0, 0}, {1, 2}}};
    mcast.attach(program, grid({0, 0}, {0, 0}), sender, "weights", sender_rt);
    EXPECT_EQ(sender.named_compile_args.at("weights_ct_offset"), 2u);
    EXPECT_EQ(sender.named_compile_args.at("weights_rt_offset"), 2u);
    EXPECT_EQ(sender.named_compile_args.at("other"), 5u);
    ASSERT_GT(sender.compile_args.size(), 2u);
    EXPECT_EQ(sender.compile_args[0], 11u);
    const auto semaphores = program.impl().semaphores().size();
    EXPECT_GT(semaphores, 0u);

    // The operation prefix is kept and the channel's runtime arguments follow it.
    std::vector<uint32_t> expected{1, 2};
    const auto channel_args =
        runtime_args(Mcast(*device_, McastConfig{.noc = NOC::NOC_0}, receivers, receivers.num_cores()), {0, 0});
    expected.insert(expected.end(), channel_args.begin(), channel_args.end());
    ASSERT_EQ(sender_rt.size(), 1u);
    EXPECT_EQ(sender_rt.front().second, expected);

    // A second kernel on the same Program shares the semaphores and compile-time words. Placed cores without
    // operation arguments get entries holding only the multicast arguments.
    DataMovementConfig receiver{.noc = NOC::NOC_1};
    std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> receiver_rt;
    mcast.attach(program, grid({1, 0}, {2, 0}), receiver, "weights", receiver_rt);
    EXPECT_EQ(program.impl().semaphores().size(), semaphores);
    EXPECT_EQ(receiver.named_compile_args.at("weights_ct_offset"), 0u);
    EXPECT_EQ(receiver.named_compile_args.at("weights_rt_offset"), 0u);
    EXPECT_EQ(receiver.compile_args, std::vector<uint32_t>(sender.compile_args.begin() + 2, sender.compile_args.end()));
    ASSERT_EQ(receiver_rt.size(), 2u);
    EXPECT_FALSE(receiver_rt.front().second.empty());

    // Invalid attachments are fatal and leave the configuration and arguments unchanged.
    const auto expect_rejected =
        [&](Program& target, const CoreRangeSet& cores, std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> rt) {
            DataMovementConfig config{.noc = NOC::NOC_0, .compile_args = {7}};
            const auto original = rt;
            EXPECT_ANY_THROW(mcast.attach(target, cores, config, "rejected", rt));
            EXPECT_EQ(config.compile_args, (std::vector<uint32_t>{7}));
            EXPECT_TRUE(config.named_compile_args.empty());
            EXPECT_EQ(rt, original);
        };
    const auto row = grid({0, 0}, {2, 0});
    expect_rejected(program, row, {{{0, 0}, {1}}, {{1, 0}, {1, 2}}, {{2, 0}, {3}}});  // Different prefix sizes.
    expect_rejected(program, row, {{{0, 0}, {1}}, {{1, 0}, {2}}});                    // Placed core without a prefix.
    expect_rejected(program, grid({1, 0}, {2, 0}), {{{0, 0}, {1}}});                  // Unplaced core.
    expect_rejected(program, row, {{{0, 0}, {1}}, {{0, 0}, {1}}});                    // Duplicate core.
    Program other;
    expect_rejected(other, row, {});  // Another Program.
    EXPECT_TRUE(other.impl().semaphores().empty());

    // Reusing a prefix on one kernel is fatal.
    auto reused = sender;
    std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> reused_rt{{{0, 0}, {1, 2}}};
    EXPECT_ANY_THROW(mcast.attach(program, grid({0, 0}, {0, 0}), reused, "weights", reused_rt));
    EXPECT_EQ(reused.compile_args, sender.compile_args);

    // The sender must run on the multicast NoC; receivers may use another NoC.
    DataMovementConfig sender_on_other_noc{.noc = NOC::NOC_1};
    std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> other_noc_rt;
    EXPECT_ANY_THROW(mcast.attach(program, grid({0, 0}, {0, 0}), sender_on_other_noc, "other_noc", other_noc_rt));
    EXPECT_TRUE(other_noc_rt.empty());
}

TEST_F(McastHostFixture, ProgramAttachChainsChannelsOnOneKernel) {
    using namespace tt::tt_metal;
    const auto receivers = grid({0, 0}, {2, 0});
    Mcast first(*device_, McastConfig{.noc = NOC::NOC_0}, receivers, receivers.num_cores());
    Mcast second(*device_, McastConfig{.noc = NOC::NOC_0}, receivers, receivers.num_cores());
    Program program;
    DataMovementConfig config{.noc = NOC::NOC_0, .compile_args = {11}};
    std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> rt{{{0, 0}, {1}}, {{1, 0}, {2}}, {{2, 0}, {3}}};
    first.attach(program, receivers, config, "first", rt);
    const auto first_ct_words = config.compile_args.size() - 1;
    const auto first_rt_words = rt.front().second.size() - 1;
    second.attach(program, receivers, config, "second", rt);
    EXPECT_EQ(config.named_compile_args.at("first_ct_offset"), 1u);
    EXPECT_EQ(config.named_compile_args.at("first_rt_offset"), 1u);
    EXPECT_EQ(config.named_compile_args.at("second_ct_offset"), 1u + first_ct_words);
    EXPECT_EQ(config.named_compile_args.at("second_rt_offset"), 1u + first_rt_words);
    EXPECT_EQ(rt.front().second.size(), 1u + 2 * first_rt_words);
}

TEST_F(McastHostFixture, ProgramAbsentAttachMarksChannelAbsent) {
    using namespace tt::tt_metal;
    DataMovementConfig config{.compile_args = {3}};
    attach_absent_mcast(config, "bias");
    EXPECT_EQ(config.compile_args, (std::vector<uint32_t>{3, wire::ABSENT}));
    EXPECT_EQ(config.named_compile_args.at("bias_ct_offset"), 1u);
    EXPECT_EQ(config.named_compile_args.at("bias_rt_offset"), 0u);
    EXPECT_ANY_THROW(attach_absent_mcast(config, "bias"));
    EXPECT_EQ(config.compile_args, (std::vector<uint32_t>{3, wire::ABSENT}));
}

TEST_F(McastHostFixture, ProgramAttachRejectsChainForwarderOnAnotherNoc) {
    using namespace tt::tt_metal;
    // An irregular receiver set lowered to a chain makes every receiver a forwarder.
    const auto receivers = cores({{0, 0}, {1, 1}, {2, 0}});
    Mcast mcast(
        *device_,
        chain_config(McastConfig{.noc = NOC::NOC_0}),
        receivers,
        receivers.num_cores(),
        McastExplicitFixedSenderConfig{.senders_per_group = {{0, 0}}});
    Program program;
    const auto forwarder = grid({1, 1}, {1, 1});
    DataMovementConfig same_noc{.noc = NOC::NOC_0};
    std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> accepted;
    mcast.attach(program, forwarder, same_noc, "chain", accepted);
    ASSERT_EQ(accepted.size(), 1u);
    EXPECT_FALSE(accepted.front().second.empty());
    DataMovementConfig other_noc{.noc = NOC::NOC_1};
    std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> rejected;
    EXPECT_ANY_THROW(mcast.attach(program, forwarder, other_noc, "chain_other_noc", rejected));
    EXPECT_TRUE(rejected.empty());
}

}  // namespace ttnn::kernel_lib::host::test
