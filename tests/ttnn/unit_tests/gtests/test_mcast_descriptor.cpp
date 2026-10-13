// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "mcast_host_test_common.hpp"

namespace ttnn::kernel_lib::host::test {
TEST_F(McastHostFixture, CompactPlacementGoldens) {
    using namespace tt::tt_metal;
    for (const auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        for (bool handshake : {false, true}) {
            auto mcast = make_mcast(device_, {{grid({1, 2}, {4, 2}), {{1, 2}}}}, {.noc = noc, .handshake = handshake});
            ProgramDescriptor descriptor;
            KernelDescriptor sender, receiver, inactive;
            sender.core_ranges = cores({{1, 2}});
            receiver.core_ranges = grid({2, 2}, {4, 2});
            inactive.core_ranges = cores({{0, 0}});
            for (auto* kernel : {&sender, &receiver, &inactive}) {
                kernel->config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = noc};
            }
            mcast.attach(
                descriptor, "channel", std::array{std::ref(sender), std::ref(receiver), std::ref(inactive)}, 0);
            const auto lo = device_->worker_core_from_logical_core({1, 2});
            const auto hi = device_->worker_core_from_logical_core({4, 2});
            const std::vector<uint32_t> bounds = noc == NOC::NOC_0 ? std::vector<uint32_t>{lo.x, lo.y, hi.x, hi.y}
                                                                   : std::vector<uint32_t>{hi.x, hi.y, lo.x, lo.y};
            EXPECT_EQ(sender.runtime_args.front().second, bounds);
            for (const auto& [core, args] : receiver.runtime_args) {
                EXPECT_EQ(args, (std::vector<uint32_t>{lo.x, lo.y}));
            }
            EXPECT_TRUE(inactive.runtime_args.front().second.empty());
            EXPECT_EQ(sender.compile_time_args.size() + sender.named_compile_time_args.size(), handshake ? 6u : 5u);
            EXPECT_EQ(receiver.compile_time_args.size() + receiver.named_compile_time_args.size(), handshake ? 5u : 4u);
            EXPECT_EQ(emitted_metadata(sender.compile_time_args).kernel.roles, 1u);
            EXPECT_EQ(emitted_metadata(receiver.compile_time_args).kernel.roles, 2u);
            EXPECT_EQ(emitted_metadata(inactive.compile_time_args).kernel.roles, 0u);
        }
    }
    auto local = make_mcast(device_, {{cores({{2, 3}}), {{2, 3}}}});
    tt::tt_metal::ProgramDescriptor descriptor;
    tt::tt_metal::KernelDescriptor kernel;
    kernel.core_ranges = cores({{2, 3}});
    kernel.config =
        tt::tt_metal::DataMovementConfigDescriptor{.processor = tt::tt_metal::DataMovementProcessor::RISCV_0};
    local.attach(descriptor, "local", std::array{std::ref(kernel)}, 0);
    EXPECT_TRUE(kernel.runtime_args.front().second.empty());
    EXPECT_NE(kernel.compile_time_args[0], wire::ABSENT);
    EXPECT_EQ(emitted_metadata(kernel.compile_time_args).mcast.remote_count_known, 1u);
    tt::tt_metal::KernelDescriptor empty;
    empty.config = kernel.config;
    local.attach(descriptor, "empty", std::array{std::ref(empty)}, local.next_semaphore_id());
    EXPECT_NE(empty.compile_time_args[0], wire::ABSENT);
    EXPECT_EQ(emitted_metadata(empty.compile_time_args).kernel.roles, 0xFFFFFFFFu);
    EXPECT_EQ(emitted_metadata(empty.compile_time_args).kernel.capabilities, 3u);
    EXPECT_TRUE(empty.runtime_args.empty());
}

TEST_F(McastHostFixture, DescriptorAppendPadsPerKernelAndPreservesBindings) {
    using namespace tt::tt_metal;
    const auto participants = grid({0, 0}, {1, 0});
    auto mcast = make_mcast(device_, {GroupInput(participants, {{0, 0}})});
    ProgramDescriptor desc;
    KernelDescriptor kernel;
    kernel.core_ranges = grid({0, 0}, {2, 0});
    kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    kernel.compile_time_args = {17, 19};
    kernel.named_compile_time_args = {{"op", 99}};
    kernel.runtime_args = {{{0, 0}, {21}}, {{1, 0}, {31, 33, 35}}};
    kernel.buffer_bindings = {{.core = {1, 0}, .arg_idx = 2}};
    kernel.common_runtime_args = {71};
    // Direct references work both before and after moving a kernel into the descriptor.
    mcast.attach(desc, "first", std::array{std::ref(kernel)}, 0);
    EXPECT_EQ(
        kernel.named_compile_time_args,
        (KernelDescriptor::NamedCompileTimeArgs{{"op", 99}, {"first_ct_offset", 2}, {"first_rt_offset", 3}}));
    EXPECT_EQ(kernel.buffer_bindings[0].arg_idx, 2u);
    EXPECT_EQ(kernel.common_runtime_args, (std::vector<uint32_t>{71}));
    ASSERT_EQ(kernel.runtime_args.size(), 3u);
    for (size_t i = 0; i < kernel.runtime_args.size(); ++i) {
        const auto& [core, args] = kernel.runtime_args[i];
        std::vector<uint32_t> expected;
        if (i == 0) {
            expected = {21, 0, 0};
        } else if (i == 1) {
            expected = {31, 33, 35};
        } else {
            expected = {0, 0, 0};
        }
        const auto payload = runtime_args(mcast, core);
        expected.insert(expected.end(), payload.begin(), payload.end());
        EXPECT_EQ(args, expected);
    }
    auto expected_ct = std::vector<uint32_t>{17, 19};
    const auto ct = compile_args(mcast);
    expected_ct.insert(expected_ct.end(), ct.begin(), ct.end());
    EXPECT_EQ(kernel.compile_time_args, expected_ct);
    desc.kernels.push_back(std::move(kernel));
    const auto old_ct_size = desc.kernels[0].compile_time_args.size();
    const auto old_rt_size = desc.kernels[0].runtime_args[0].second.size();
    mcast.attach(desc, "second", std::array{std::ref(desc.kernels[0])}, mcast.next_semaphore_id());
    EXPECT_EQ(desc.semaphores.size(), 4u);
    EXPECT_EQ(desc.kernels[0].named_compile_time_args[3].second, old_ct_size);
    EXPECT_EQ(desc.kernels[0].named_compile_time_args[4].second, old_rt_size);
    const auto before = desc.kernels[0].compile_time_args;
    EXPECT_ANY_THROW(mcast.attach(desc, "second", std::array{std::ref(desc.kernels[0])}, mcast.next_semaphore_id()));
    EXPECT_EQ(desc.semaphores.size(), 4u);
    EXPECT_EQ(desc.kernels[0].compile_time_args, before);
    attach_absent_mcast(desc.kernels[0], "absent");
    EXPECT_EQ(desc.kernels[0].compile_time_args.back(), 0u);
    EXPECT_EQ(desc.kernels[0].runtime_args[0].second.size(), old_rt_size + runtime_args(mcast, {0, 0}).size());
}

TEST_F(McastHostFixture, DescriptorAppendFailurePreservesAllTargets) {
    using namespace tt::tt_metal;
    auto mcast = make_mcast(device_, {GroupInput(grid({0, 0}, {1, 0}), {{0, 0}})});
    ProgramDescriptor desc;
    KernelDescriptor first;
    first.core_ranges = grid({0, 0}, {1, 0});
    first.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    first.runtime_args = {{{0, 0}, {7}}, {{1, 0}, {8, 9}}};
    auto second = first;
    // Padding must not turn an out-of-bounds binding into a valid one.
    second.buffer_bindings = {{.core = {0, 0}, .arg_idx = 1}};
    const std::array targets{std::ref(first), std::ref(second)};
    EXPECT_ANY_THROW(mcast.attach(desc, "channel", targets, 0));
    EXPECT_TRUE(desc.semaphores.empty());
    EXPECT_TRUE(first.compile_time_args.empty());
    EXPECT_TRUE(first.named_compile_time_args.empty());
    EXPECT_EQ(first.runtime_args[0].second, (std::vector<uint32_t>{7}));
    EXPECT_TRUE(second.named_compile_time_args.empty());
    const std::array duplicate{std::ref(first), std::ref(first)};
    EXPECT_ANY_THROW(mcast.attach(desc, "channel", duplicate, 0));
}

TEST_F(McastHostFixture, DescriptorAttachAppendsAfterPrefixesAndPreservesBufferBindings) {
    using namespace tt::tt_metal;
    const auto participants = grid({0, 0}, {1, 0});
    auto mcast = make_mcast(device_, {GroupInput(participants, {{0, 0}})});
    ProgramDescriptor desc;
    // The caller's cursor starts after resources already appended by other builders.
    desc.semaphores.push_back({.id = 0, .core_ranges = grid({0, 0}, {0, 0})});
    desc.semaphores.push_back({.id = 2, .core_ranges = grid({1, 0}, {1, 0})});
    KernelDescriptor kernel;
    kernel.core_ranges = grid({0, 0}, {2, 0});
    kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    kernel.compile_time_args = {17, 19};
    kernel.common_runtime_args = {71};
    kernel.runtime_args = {{{0, 0}, {21, 23}}, {{1, 0}, {31, 33, 35}}, {{2, 0}, {41, 43}}};
    kernel.buffer_bindings = {{.core = {0, 0}, .arg_idx = 0}, {.core = {1, 0}, .arg_idx = 1}};
    kernel.common_buffer_bindings = {{.arg_idx = 0}};
    desc.kernels.push_back(kernel);
    const std::array points{std::ref(desc.kernels[0])};
    EXPECT_ANY_THROW(mcast.attach(desc, "mcast", points, 2));
    EXPECT_EQ(desc.semaphores.size(), 2u);
    EXPECT_ANY_THROW(mcast.next_semaphore_id());
    mcast.attach(desc, "mcast", points, 3);
    ASSERT_EQ(desc.semaphores.size(), 4u);
    EXPECT_EQ(desc.semaphores[2].id, 3u);
    EXPECT_EQ(desc.semaphores[3].id, 4u);
    EXPECT_EQ(mcast.next_semaphore_id(), 5u);
    const auto& attached = desc.kernels[0];
    const std::vector<uint32_t> expected_ct{17, 19, 0x0024E2E1u, 3, 4, 1};
    EXPECT_EQ(attached.compile_time_args, expected_ct);
    const auto a = device_->worker_core_from_logical_core({0, 0});
    const auto b = device_->worker_core_from_logical_core({1, 0});
    const std::vector<uint32_t> expected_sender{21, 23, 0, 1, a.x, a.y, a.x, a.y, b.x, b.y};
    const std::vector<uint32_t> expected_receiver{31, 33, 35, 2, a.x, a.y, 0, 0, 0, 0};
    const std::vector<uint32_t> expected_inactive{41, 43, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(attached.runtime_args[0].second, expected_sender);
    EXPECT_EQ(attached.runtime_args[1].second, expected_receiver);
    EXPECT_EQ(attached.runtime_args[2].second, expected_inactive);
    EXPECT_EQ(attached.buffer_bindings[0].arg_idx, 0u);
    EXPECT_EQ(attached.buffer_bindings[1].arg_idx, 1u);
    EXPECT_EQ(attached.common_buffer_bindings[0].arg_idx, 0u);
    EXPECT_EQ(attached.common_runtime_args, std::vector<uint32_t>{71});
    // A separate regular Program attachment still uses Program's automatic allocator.
    EXPECT_EQ(emitted_semaphore(compile_args(mcast), wire::DATA_READY), 0u);
    EXPECT_EQ(emitted_semaphore(compile_args(mcast), wire::CONSUMER_READY), 1u);
}

TEST_F(McastHostFixture, DescriptorAttachValidatesAllKernelsBeforeMutation) {
    using namespace tt::tt_metal;
    auto mcast = make_mcast(device_, {GroupInput(grid({0, 0}, {1, 0}), {{0, 0}})});
    ProgramDescriptor desc;
    KernelDescriptor sender;
    sender.core_ranges = grid({0, 0}, {0, 0});
    sender.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    sender.compile_time_args = {17};
    sender.runtime_args = {{{0, 0}, {23}}};
    KernelDescriptor receiver;
    receiver.core_ranges = grid({1, 0}, {1, 0});
    receiver.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};  // Pure receivers may use the other NoC.
    receiver.compile_time_args = {29};
    desc.kernels = {sender, receiver};
    const std::array bad_points{std::ref(desc.kernels[0]), std::ref(desc.kernels[1])};
    desc.kernels[1].buffer_bindings = {{.core = {1, 0}, .arg_idx = 0}};
    EXPECT_ANY_THROW(mcast.attach(desc, "mcast", bad_points, 0));
    EXPECT_TRUE(desc.semaphores.empty());
    EXPECT_EQ(desc.kernels[0].compile_time_args, sender.compile_time_args);
    EXPECT_EQ(desc.kernels[0].runtime_args, sender.runtime_args);
    EXPECT_TRUE(desc.kernels[1].runtime_args.empty());
    auto points = bad_points;
    desc.kernels[1].buffer_bindings.clear();
    EXPECT_NO_THROW(mcast.attach(desc, "mcast", points, 0));
    EXPECT_EQ(desc.kernels[1].runtime_args.size(), 1u);
}

TEST_F(McastHostFixture, DescriptorAttachAndAbsence) {
    using namespace tt::tt_metal;
    const auto participants = grid({0, 0}, {1, 0});
    auto mcast = make_mcast(device_, {GroupInput(participants, {{0, 0}})}, McastConfig{.handshake = false});
    ProgramDescriptor desc;
    KernelDescriptor kernel;
    kernel.core_ranges = participants;
    kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    kernel.compile_time_args = {31};
    desc.kernels.push_back(kernel);
    const std::array points{std::ref(desc.kernels[0])};
    mcast.attach(desc, "mcast", points, 0);
    EXPECT_EQ(desc.semaphores.size(), 1u);
    EXPECT_EQ(emitted_semaphore(desc.kernels[0].compile_time_args, wire::DATA_READY, 1), 0u);
    EXPECT_EQ(emitted_semaphore(desc.kernels[0].compile_time_args, wire::CONSUMER_READY, 1), UNUSED_SEM_ID);
    EXPECT_EQ(emitted_metadata(desc.kernels[0].compile_time_args, 1).mcast.flags, 0u);
    const auto runtime = desc.kernels[0].runtime_args;
    attach_absent_mcast(desc.kernels[0], "absent_mcast");
    EXPECT_EQ(desc.kernels[0].compile_time_args.back(), 0u);
    EXPECT_EQ(desc.kernels[0].compile_time_args[0], 31u);
    EXPECT_EQ(desc.kernels[0].runtime_args, runtime);
    EXPECT_EQ(desc.semaphores.size(), 1u);
}

TEST_F(McastHostFixture, DescriptorAttachConsumesIdsAndSupportsIndependentMcasts) {
    using namespace tt::tt_metal;
    const auto participants = grid({0, 0}, {1, 0});
    const GroupInput group(participants, {{0, 0}});
    auto first_mcast = make_mcast(device_, {group});
    ProgramDescriptor desc;
    KernelDescriptor kernel;
    kernel.core_ranges = participants;
    kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    kernel.compile_time_args = {97};
    desc.kernels.push_back(kernel);
    const std::array first{std::ref(desc.kernels[0])};
    EXPECT_ANY_THROW(first_mcast.next_semaphore_id());
    first_mcast.attach(desc, "mcast", first, 5);
    EXPECT_EQ(desc.semaphores[0].id, 5u);
    EXPECT_EQ(desc.semaphores[1].id, 6u);
    EXPECT_EQ(first_mcast.next_semaphore_id(), 7u);

    // The second channel allocates its own slots, appending after the first helper block.
    auto second_mcast = make_mcast(device_, {group});
    const std::array second{std::ref(desc.kernels[0])};
    second_mcast.attach(desc, "second_mcast", second, first_mcast.next_semaphore_id());
    ASSERT_EQ(desc.semaphores.size(), 4u);
    EXPECT_EQ(desc.semaphores[2].id, 7u);
    EXPECT_EQ(desc.semaphores[3].id, 8u);
    EXPECT_EQ(second_mcast.next_semaphore_id(), 9u);
    const std::vector<uint32_t> expected{97, 0x0024E2E1u, 5, 6, 1, 0x0024E2E1u, 7, 8, 1};
    EXPECT_EQ(desc.kernels[0].compile_time_args, expected);
    for (const auto& [core, args] : desc.kernels[0].runtime_args) {
        ASSERT_EQ(args.size(), 14u);
        EXPECT_TRUE(std::equal(args.begin(), args.begin() + 7, args.begin() + 7));
    }
}

TEST_F(McastHostFixture, DescriptorAttachChainRequiresForwarderNocAndThreeResources) {
    using namespace tt::tt_metal;
    const auto participants = cores({{0, 0}, {2, 0}});
    auto mcast = make_mcast(device_, {GroupInput(participants, {{0, 0}})}, chain_config());
    ProgramDescriptor desc;
    KernelDescriptor head;
    head.core_ranges = grid({0, 0}, {0, 0});
    head.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    KernelDescriptor forwarder;
    forwarder.core_ranges = grid({2, 0}, {2, 0});
    forwarder.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};
    desc.kernels = {head, forwarder};
    const std::array points{std::ref(desc.kernels[0]), std::ref(desc.kernels[1])};
    EXPECT_ANY_THROW(mcast.attach(desc, "mcast", points, 0));
    EXPECT_TRUE(desc.semaphores.empty());
    EXPECT_TRUE(desc.kernels[0].compile_time_args.empty());
    desc.kernels[1].config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
    mcast.attach(desc, "mcast", points, 0);
    ASSERT_EQ(desc.semaphores.size(), 3u);
    ASSERT_EQ(desc.kernels[0].runtime_args.size(), 1u);
    ASSERT_EQ(desc.kernels[1].runtime_args.size(), 1u);
    EXPECT_EQ(emitted_metadata(desc.kernels[0].compile_time_args).kernel.roles, wire::CAN_SEND);
    EXPECT_EQ(emitted_metadata(desc.kernels[1].compile_time_args).kernel.roles, wire::CAN_RECEIVE);
    EXPECT_EQ(desc.kernels[0].runtime_args[0].second.size(), 5u);
    EXPECT_EQ(desc.kernels[1].runtime_args[0].second.size(), 7u);
}

TEST_F(McastHostFixture, DescriptorAttachUsesNativeReaderWriterNocDefaults) {
    using namespace tt::tt_metal;
    const auto participants = grid({0, 0}, {1, 0});
    const GroupInput group(participants, {{0, 0}});
    for (bool reader : {false, true}) {
        // Resolve through the actual native config constructors, independently of attach.
        const auto native_noc = reader ? ReaderDataMovementConfig{}.noc : WriterDataMovementConfig{}.noc;
        auto mcast = make_mcast(device_, {group}, McastConfig{.noc = native_noc});
        ProgramDescriptor desc;
        KernelDescriptor kernel;
        kernel.core_ranges = participants;
        if (reader) {
            kernel.config = ReaderConfigDescriptor{};
        } else {
            kernel.config = WriterConfigDescriptor{};
        }
        desc.kernels.push_back(kernel);
        const std::array points{std::ref(desc.kernels[0])};
        EXPECT_NO_THROW(mcast.attach(desc, "mcast", points, 0));

        const auto other_noc = native_noc == NOC::NOC_0 ? NOC::NOC_1 : NOC::NOC_0;
        auto mismatched = make_mcast(device_, {group}, McastConfig{.noc = other_noc});
        desc.kernels = {kernel};
        desc.semaphores.clear();
        EXPECT_ANY_THROW(mismatched.attach(desc, "mcast", points, 0));
        EXPECT_TRUE(desc.kernels[0].compile_time_args.empty());
        EXPECT_TRUE(desc.semaphores.empty());
    }
}

}  // namespace ttnn::kernel_lib::host::test

namespace ttnn::kernel_lib::host::test {

TEST_F(McastHostFixture, NoHandshakeLeavesExchangeCreditAcrossConstructionPaths) {
    using namespace tt::tt_metal;
    const auto participants = grid({0, 0}, {1, 0});
    for (const auto signal :
         {dataflow_kernel_lib::DataReadySignal::Flag, dataflow_kernel_lib::DataReadySignal::Counter}) {
        auto mcast = make_mcast(device_, {{participants, {{0, 0}}}}, {.handshake = false, .data_ready = signal});
        Program program;
        ProgramDescriptor descriptor;
        auto spec = spec_pair();
        for (uint32_t id = 0; id < 14; ++id) {
            program.impl().add_semaphore(participants, id, 0, tt::CoreType::WORKER);
            descriptor.semaphores.push_back({.id = id, .core_ranges = participants, .initial_value = 0});
            spec.semaphores.push_back(
                {.unique_id = m2::SemaphoreSpecName{"exchange_" + std::to_string(id)}, .target_nodes = participants});
        }
        KernelDescriptor kernel;
        kernel.core_ranges = participants;
        kernel.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
        const std::array targets{std::ref(kernel)};
        mcast.attach(descriptor, "payload", targets, 14);
        ASSERT_EQ(descriptor.semaphores.size(), 15u);
        EXPECT_EQ(descriptor.semaphores.back().id, 14u);
        EXPECT_EQ(emitted_semaphore(kernel.compile_time_args, wire::CONSUMER_READY), UNUSED_SEM_ID);

        m2::ProgramRunArgs args;
        mcast.attach(spec, args, "payload", spec_targets);
        ASSERT_EQ(spec.semaphores.size(), 15u);
        for (const auto& target : spec.kernels) {
            EXPECT_EQ(target.semaphore_bindings.size(), 1u);
        }
        mcast.append_semaphores(program);
        ASSERT_EQ(program.impl().semaphores().size(), 15u);
        EXPECT_EQ(program.impl().semaphores().back().id(), 14u);
        // The final hardware slot remains available for the operation's exchange credit.
        EXPECT_EQ(CreateSemaphore(program, participants, 0), 15u);
    }
}

}  // namespace ttnn::kernel_lib::host::test
