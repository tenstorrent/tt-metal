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
    std::vector<uint32_t> ct{71}, rt{73};
    EXPECT_ANY_THROW(mcast.append_compile_time_args_to(ct));
    EXPECT_ANY_THROW(mcast.append_runtime_args_to(rt, {0, 0}));
    EXPECT_EQ(ct, (std::vector<uint32_t>{71}));
    EXPECT_EQ(rt, (std::vector<uint32_t>{73}));
    mcast.append_semaphores(program);
    mcast.append_compile_time_args_to(ct);
    mcast.append_runtime_args_to(rt, {0, 0});
    EXPECT_EQ(emitted_semaphore(ct, wire::DATA_READY, 1), 1u);
    EXPECT_EQ(emitted_semaphore(ct, wire::CONSUMER_READY, 1), 3u);
    EXPECT_EQ(rt.front(), 73u);
    EXPECT_GT(rt.size(), 1u);
    ASSERT_EQ(program.impl().semaphores().size(), 4u);
    EXPECT_EQ(program.impl().semaphores()[0].initial_value(), 7u);
    EXPECT_ANY_THROW(mcast.append_semaphores(program));
    auto copy = mcast;
    EXPECT_ANY_THROW(copy.append_semaphores(program));
    EXPECT_EQ(program.impl().semaphores().size(), 4u);
    Program other;
    EXPECT_ANY_THROW(copy.append_semaphores(other));
    EXPECT_TRUE(other.impl().semaphores().empty());
    Program moved = std::move(program);
    EXPECT_ANY_THROW(mcast.append_semaphores(moved));
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
    chain.append_semaphores(chain_program);
    std::vector<uint32_t> ct;
    chain.append_compile_time_args_to(ct);
    ASSERT_EQ(chain_program.impl().semaphores().size(), 3u);
    EXPECT_EQ(emitted_semaphore(ct, wire::SIGNAL_SOURCE), 2u);

    const auto participants = grid({0, 0}, {1, 0});
    auto mcast = make_mcast(device_, {{participants, {{0, 0}}}});
    Program full;
    for (uint32_t id = 0; id < NUM_SEMAPHORES; ++id) {
        full.impl().add_semaphore(participants, id, 0, tt::CoreType::WORKER);
    }
    EXPECT_ANY_THROW(mcast.append_semaphores(full));
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
    EXPECT_ANY_THROW(mcast.append_semaphores(compiled));
    EXPECT_TRUE(compiled.impl().semaphores().empty());
}

}  // namespace ttnn::kernel_lib::host::test
