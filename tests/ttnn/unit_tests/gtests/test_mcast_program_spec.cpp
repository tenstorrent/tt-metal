// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "mcast_host_test_common.hpp"

namespace ttnn::kernel_lib::host::test {
TEST_F(McastHostFixture, SpecAttachPopulatesNamedMetadataResourcesAndRuntimePrefixes) {
    auto mcast = make_mcast(device_, {GroupInput(grid({0, 0}, {1, 0}), {{0, 0}})});
    auto spec = spec_pair(2);
    m2::ProgramRunArgs args;
    for (size_t i = 0; i < spec_targets.size(); ++i) {
        m2::KernelRunArgs values{.kernel = spec_targets[i], .common_runtime_arg_values = {{"kept_common", 97}}};
        for (uint32_t x = i == 0 ? 0 : 1; x <= (i == 0 ? 0 : 2); ++x) {
            values.advanced_options.runtime_varargs[{x, 0}] = {41 + x, 51 + x};
            values.runtime_arg_values["kept_rt"][{x, 0}] = 83 + x;
        }
        args.kernel_run_args.push_back(std::move(values));
    }
    mcast.attach(spec, args, "channel", spec_targets);
    ASSERT_EQ(spec.semaphores.size(), 2u);
    EXPECT_EQ(spec.semaphores[0].unique_id, m2::SemaphoreSpecName{"channel_mcast_data_ready"});
    EXPECT_EQ(spec.semaphores[1].unique_id, m2::SemaphoreSpecName{"channel_mcast_consumer_ready"});
    const auto sender = device_->worker_core_from_logical_core({0, 0});
    const auto end = device_->worker_core_from_logical_core({1, 0});
    for (size_t i = 0; i < spec.kernels.size(); ++i) {
        const auto& kernel = spec.kernels[i];
        EXPECT_EQ(kernel.advanced_options.num_runtime_varargs, i == 0 ? 6u : 5u);
        EXPECT_EQ(kernel.compile_time_args.get("channel_mcast_rt_base").value(), 2u);
        EXPECT_EQ(kernel.compile_time_args.get("channel_mcast_ct_base").value(), 2u);
        const auto decoded = decode_emitted_ct(kernel.advanced_options.compile_time_varargs, 2, false);
        EXPECT_EQ(decoded.metadata.mcast.flags, 1u);
        EXPECT_EQ(decoded.metadata.mcast.rectangle_capacity, 1u);
        EXPECT_EQ(decoded.metadata.mcast.uniform_remote_count, i == 0 ? 1u : 0u);
        EXPECT_EQ(kernel.advanced_options.compile_time_varargs.size(), i == 0 ? 4u : 3u);
        EXPECT_EQ(kernel.advanced_options.compile_time_varargs[0], 101u);
        EXPECT_EQ(kernel.advanced_options.compile_time_varargs[1], 103u);
        EXPECT_EQ(kernel.compile_time_args.size(), 3u);  // Existing arg plus CT/RT offsets.
        EXPECT_EQ(kernel.compile_time_args.get("kept_ct").value(), 73u);
        EXPECT_EQ(
            kernel.compiler_options.defines.get("channel_mcast_data_ready_type").value(),
            "dataflow_kernel_lib::detail::McastSemaphoreToken<sem::channel_mcast_data_ready>");
        EXPECT_EQ(kernel.compiler_options.defines.get("channel_mcast_signal_source_type").value(), "std::nullptr_t");
        EXPECT_EQ(args.kernel_run_args[i].common_runtime_arg_values.get("kept_common").value(), 97u);
    }
    const std::vector<uint32_t> sender_expected{41, 51, sender.x, sender.y, end.x, end.y};
    EXPECT_EQ(args.kernel_run_args[0].advanced_options.runtime_varargs.get({0, 0}).value(), sender_expected);
    EXPECT_EQ(args.kernel_run_args[1].runtime_arg_values.get("kept_rt").value().get({1, 0}).value(), 84u);
    const auto& inactive = args.kernel_run_args[1].advanced_options.runtime_varargs.get({2, 0}).value();
    EXPECT_EQ(inactive, (std::vector<uint32_t>{43, 53, 0, 0, 0}));
}

TEST_F(McastHostFixture, SpecAttachComposesAndNativeRunArgsCopiesKeepPayloads) {
    auto mcast = make_mcast(device_, {GroupInput(grid({0, 0}, {1, 0}), {{0, 0}})});
    auto spec = spec_pair();
    m2::ProgramRunArgs args;
    mcast.attach(spec, args, "first", spec_targets);
    const auto payload = args.kernel_run_args[0].advanced_options.runtime_varargs.get({0, 0}).value();
    mcast.attach(spec, args, "second", spec_targets);
    EXPECT_EQ(spec.semaphores.size(), 4u);
    EXPECT_EQ(spec.kernels[0].compile_time_args.get("second_mcast_rt_base").value(), 4u);
    EXPECT_EQ(spec.kernels[0].compile_time_args.get("first_mcast_ct_base").value(), 2u);
    EXPECT_EQ(spec.kernels[0].compile_time_args.get("second_mcast_ct_base").value(), 4u);
    EXPECT_EQ(spec.kernels[1].compile_time_args.get("second_mcast_ct_base").value(), 3u);
    EXPECT_EQ(spec.kernels[0].advanced_options.num_runtime_varargs, 8u);
    auto copied = args;
    copied.kernel_run_args[0].runtime_arg_values["kept_rt"][{0, 0}] = 123;
    auto expected = payload;
    expected.insert(expected.end(), payload.begin(), payload.end());
    EXPECT_EQ(copied.kernel_run_args[0].advanced_options.runtime_varargs.get({0, 0}).value(), expected);
    EXPECT_TRUE(args.kernel_run_args[0].runtime_arg_values.empty());
    EXPECT_ANY_THROW(mcast.attach(spec, args, "first", spec_targets));
    EXPECT_EQ(spec.semaphores.size(), 4u);
}

TEST_F(McastHostFixture, SpecAttachFailuresLeaveBothObjectsUnchanged) {
    auto mcast = make_mcast(device_, {GroupInput(grid({0, 0}, {1, 0}), {{0, 0}})});
    for (const auto violation :
         {"missing-prefix",
          "trailing-values",
          "duplicate-run-entry",
          "unplaced-run-node",
          "wrong-sender-noc",
          "duplicate-target",
          "duplicate-prefix"}) {
        auto spec = spec_pair(1);
        m2::ProgramRunArgs args;
        for (const auto& name : spec_targets) {
            args.kernel_run_args.push_back({.kernel = name});
        }
        args.kernel_run_args[0].advanced_options.runtime_varargs[{0, 0}] = {19};
        args.kernel_run_args[1].advanced_options.runtime_varargs[{1, 0}] = {23};
        args.kernel_run_args[1].advanced_options.runtime_varargs[{2, 0}] = {29};
        if (std::string_view(violation) == "missing-prefix") {
            args.kernel_run_args[1].advanced_options.runtime_varargs.erase({2, 0});
        }
        if (std::string_view(violation) == "trailing-values") {
            args.kernel_run_args[1].advanced_options.runtime_varargs[{2, 0}].push_back(31);
        }
        if (std::string_view(violation) == "duplicate-run-entry") {
            args.kernel_run_args.push_back(args.kernel_run_args[0]);
        }
        if (std::string_view(violation) == "unplaced-run-node") {
            args.kernel_run_args[1].advanced_options.runtime_varargs[{3, 0}] = {37};
        }
        if (std::string_view(violation) == "wrong-sender-noc") {
            std::get<m2::DataMovementHardwareConfig>(spec.kernels[0].hw_config).config_1xx->noc = NOC::NOC_1;
        }
        if (std::string_view(violation) == "duplicate-prefix") {
            spec.kernels[1].compile_time_args["channel_mcast_ct_base"] = 55;
        }
        auto targets = spec_targets;
        if (std::string_view(violation) == "duplicate-target") {
            targets[1] = targets[0];
        }
        const auto before = args;
        const auto before_spec = spec;
        EXPECT_ANY_THROW(mcast.attach(spec, args, "channel", targets)) << violation;
        EXPECT_TRUE(spec.semaphores.empty()) << violation;
        ASSERT_EQ(args.kernel_run_args.size(), before.kernel_run_args.size());
        for (size_t i = 0; i < spec.kernels.size(); ++i) {
            EXPECT_EQ(spec.kernels[i].compile_time_args, before_spec.kernels[i].compile_time_args) << violation;
            EXPECT_EQ(
                spec.kernels[i].advanced_options.compile_time_varargs,
                before_spec.kernels[i].advanced_options.compile_time_varargs)
                << violation;
            EXPECT_EQ(spec.kernels[i].advanced_options.num_runtime_varargs, 1u) << violation;
            EXPECT_TRUE(spec.kernels[i].semaphore_bindings.empty()) << violation;
            EXPECT_TRUE(spec.kernels[i].compiler_options.defines.empty()) << violation;
        }
        for (size_t i = 0; i < args.kernel_run_args.size(); ++i) {
            EXPECT_EQ(
                args.kernel_run_args[i].advanced_options.runtime_varargs,
                before.kernel_run_args[i].advanced_options.runtime_varargs)
                << violation;
        }
    }
}

TEST_F(McastHostFixture, SpecAttachValidatesUniformVarargSchemaAndNormalizesOverrides) {
    auto mcast = make_mcast(device_, {GroupInput(grid({0, 0}, {1, 0}), {{0, 0}})});
    auto spec = spec_pair();
    spec.kernels[1].advanced_options.num_runtime_varargs_per_node.emplace(CoreCoord{1, 0}, 2);
    m2::ProgramRunArgs args;
    EXPECT_ANY_THROW(mcast.attach(spec, args, "channel", spec_targets));
    EXPECT_TRUE(args.kernel_run_args.empty());
    spec.kernels[1].advanced_options.num_runtime_varargs_per_node.emplace(CoreCoord{2, 0}, 2);
    args.kernel_run_args.push_back({.kernel = spec_targets[1]});
    args.kernel_run_args[0].advanced_options.runtime_varargs[{1, 0}] = {31, 37};
    args.kernel_run_args[0].advanced_options.runtime_varargs[{2, 0}] = {41, 43};
    mcast.attach(spec, args, "channel", spec_targets);
    EXPECT_EQ(spec.kernels[1].advanced_options.num_runtime_varargs, 5u);
    EXPECT_TRUE(spec.kernels[1].advanced_options.num_runtime_varargs_per_node.empty());
    EXPECT_EQ(spec.kernels[1].compile_time_args.get("channel_mcast_rt_base").value(), 2u);
}

TEST_F(McastHostFixture, SpecAbsentNeedsNoResourcesOrRunArgumentObject) {
    auto spec = spec_pair(3);
    attach_absent(spec, "none", spec_targets);
    EXPECT_TRUE(spec.semaphores.empty());
    for (const auto& kernel : spec.kernels) {
        EXPECT_EQ(kernel.compile_time_args.get("none_mcast_ct_base").value(), 2u);
        EXPECT_EQ(kernel.advanced_options.compile_time_varargs, (std::vector<uint32_t>{101, 103, 0}));
        EXPECT_EQ(kernel.advanced_options.num_runtime_varargs, 3u);
        EXPECT_TRUE(kernel.semaphore_bindings.empty());
        EXPECT_EQ(kernel.compiler_options.defines.size(), 3u);
        for (const auto& [name, value] : kernel.compiler_options.defines) {
            EXPECT_EQ(value, "std::nullptr_t");
        }
    }
    EXPECT_ANY_THROW(attach_absent(spec, "none", spec_targets));
}

TEST_F(McastHostFixture, SpecAttachAllowsOtherNocOnlyOnPureMulticastReceivers) {
    const auto participants = cores({{0, 0}, {2, 0}});
    auto multicast = make_mcast(device_, {GroupInput(participants, {{0, 0}})});
    auto chain = make_mcast(device_, {GroupInput(participants, {{0, 0}})}, chain_config());
    auto spec = spec_pair();
    std::get<m2::DataMovementHardwareConfig>(spec.kernels[1].hw_config).config_1xx->noc = NOC::NOC_1;
    m2::ProgramRunArgs args;
    EXPECT_ANY_THROW(chain.attach(spec, args, "chain", spec_targets));
    EXPECT_TRUE(args.kernel_run_args.empty());
    EXPECT_NO_THROW(multicast.attach(spec, args, "multicast", spec_targets));
}
}  // namespace ttnn::kernel_lib::host::test

namespace ttnn::kernel_lib::host::test {

void run_spec_device_contract(
    tt::tt_metal::distributed::MeshDevice& device,
    NOC noc,
    bool counter,
    bool rotating,
    bool control,
    bool chain,
    bool handshake,
    bool local) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;
    SCOPED_TRACE(
        ::testing::Message() << "noc=" << noc << " counter=" << counter << " rotating=" << rotating << " control="
                             << control << " chain=" << chain << " handshake=" << handshake << " local=" << local);
    const std::vector<CoreCoord> active = local   ? std::vector<CoreCoord>{{0, 0}}
                                          : chain ? std::vector<CoreCoord>{{0, 0}, {1, 0}, {0, 1}}
                                                  : std::vector<CoreCoord>{{0, 0}, {1, 0}, {2, 0}};
    auto placed = active;
    placed.push_back({3, 0});  // Placed kernel outside either mcast must get inactive role data.
    const auto participants = cores(active);
    const auto placement = cores(placed);
    const uint32_t rounds = handshake ? 4 : 1;
    const std::array targets{m2::KernelSpecName{"contract"}};
    m2::KernelSpec kernel{
        .unique_id = targets.front(),
        .source = "tests/ttnn/unit_tests/kernel_lib/kernels/mcast_spec.cpp",
        .compile_time_args = {{"rounds", rounds}, {"control", control ? 1u : 0u}},
        .hw_config = m2::DataMovementHardwareConfig{
            .config_1xx = m2::DataMovementHardwareConfig::DataMovement1XXConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = noc}}};
    kernel.advanced_options.num_runtime_varargs = 2;
    kernel.scratchpad_bindings.push_back(
        {.scratchpad_spec_name = m2::ScratchpadSpecName{"pad"}, .accessor_name = "pad"});
    m2::ProgramSpec spec{
        .kernels = {kernel},
        .scratchpads = {{.unique_id = m2::ScratchpadSpecName{"pad"}, .size_per_node = 256}},
        .work_units = {{.name = "contract", .kernels = {targets.front()}, .target_nodes = placement}}};
    m2::ProgramRunArgs populated;
    populated.kernel_run_args.push_back({.kernel = targets.front()});
    for (auto node : placed) {
        populated.kernel_run_args.front().advanced_options.runtime_varargs.emplace(
            node, std::vector<uint32_t>{0x1234, 0x5678});
    }
    McastConfig cfg{
        .noc = noc,
        .handshake = handshake,
        .data_ready =
            counter ? dataflow_kernel_lib::DataReadySignal::Counter : dataflow_kernel_lib::DataReadySignal::Flag,
        .irregular_receiver_set_mode = chain ? TransferMode::ChainUnicast : TransferMode::Multicast};
    std::vector<CoreCoord> senders{{0, 0}};
    if (rotating) {
        senders.push_back({2, 0});
    }
    auto mcast = make_mcast(&device, {GroupInput(participants, senders)}, cfg);
    mcast.attach(spec, populated, "channel", targets);
    auto second = make_mcast(&device, {GroupInput(participants, {{0, 0}})}, McastConfig{.noc = noc});
    second.attach(spec, populated, "second", targets);
    attach_absent(spec, "absent", targets);
    // Named RT values may be declared after attach: generated get_vararg must use
    // the final named-argument offset, preserving both the caller prefix and helper slices.
    spec.kernels.front().runtime_arg_schema.runtime_arg_names = {"seed", "report_addr"};
    auto workload = m2::MakeMeshWorkloadFromSpec(device, spec);
    auto* physical = device.get_devices().front();
    for (uint32_t run = 0; run < 3; ++run) {
        const uint32_t seed = 19 + run * 173;
        const uint32_t report_addr = 100 * 1024 + run * sizeof(uint32_t);
        // Native value copies retain the topology, while operation values change each launch.
        auto invocation = populated;
        for (auto node : placed) {
            m2::AddRuntimeArgsForNode(
                invocation.kernel_run_args.front().runtime_arg_values,
                node,
                {{"seed", seed}, {"report_addr", report_addr}});
            std::vector<uint32_t> zero{0};
            tt::tt_metal::detail::WriteToDeviceL1(physical, node, report_addr, zero);
        }
        for (auto& [region, program] : workload.get_programs()) {
            m2::SetProgramRunArgs(program, invocation);
        }
        EnqueueMeshWorkload(device.mesh_command_queue(), workload, true);
        for (auto node : placed) {
            SCOPED_TRACE(::testing::Message() << "run=" << run << " node=(" << node.x << "," << node.y << ")");
            std::vector<uint32_t> base, result;
            tt::tt_metal::detail::ReadFromDeviceL1(physical, node, report_addr, 4, base);
            ASSERT_EQ(base.size(), 1u);
            ASSERT_NE(base.front(), 0u);
            tt::tt_metal::detail::ReadFromDeviceL1(physical, node, base.front() + 128, 48, result);
            ASSERT_EQ(result.size(), 12u);
            const bool inside = node != CoreCoord{3, 0};
            for (uint32_t round = 0; round < rounds; ++round) {
                const uint32_t expected = control ? (counter ? round + 1 : 1) : 136 * (seed + round * 100) + 1360;
                EXPECT_EQ(result[round], inside ? expected : 0u);
            }
            EXPECT_EQ(result[8], inside ? 136 * (seed + 1000) + 1360 : 0u);
            EXPECT_EQ(result[9], 0x1234u);
            EXPECT_EQ(result[10], 0x5678u);
            EXPECT_EQ(result[11], 0xA55Au);
        }
    }
}

TEST_F(McastHostFixture, SpecDeviceSmoke) {
    run_spec_device_contract(*device_, NOC::NOC_0, false, false, false, false, true, false);
}

TEST_F(McastHostFixture, SpecDeviceOldTag) {
    using namespace tt::tt_metal;
    const std::array targets{m2::KernelSpecName{"old_tag"}};
    m2::ProgramSpec spec{
        .kernels =
            {{.unique_id = targets.front(),
              .source = m2::KernelSpec::SourceCode{R"(
                #include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args_spec.hpp"
                void kernel_main() { constexpr auto channel = MCAST_SPEC_ARGS(channel); }
            )"},
              .hw_config =
                  m2::DataMovementHardwareConfig{
                      .config_1xx =
                          m2::DataMovementHardwareConfig::DataMovement1XXConfig{
                              .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0}}}},
        .work_units = {{.name = "old_tag", .kernels = {targets.front()}, .target_nodes = CoreCoord{0, 0}}}};
    auto mcast = make_mcast(device_, {{grid({0, 0}, {1, 0}), {{0, 0}}}});
    m2::ProgramRunArgs args;
    mcast.attach(spec, args, "channel", targets);
    const auto ct_base = spec.kernels.front().compile_time_args.get("channel_mcast_ct_base").value();
    spec.kernels.front().advanced_options.compile_time_varargs[ct_base] = 2;
    try {
        auto workload = m2::MakeMeshWorkloadFromSpec(*device_, spec);
        for (auto& [region, program] : workload.get_programs()) {
            m2::SetProgramRunArgs(program, args);
        }
        tt::tt_metal::distributed::EnqueueMeshWorkload(device_->mesh_command_queue(), workload, true);
        FAIL() << "The native decoder accepted an obsolete multicast tag";
    } catch (const std::exception& error) {
        EXPECT_NE(std::string(error.what()).find("Unsupported multicast wire tag"), std::string::npos);
    }
}

TEST_F(McastHostFixture, SpecDeviceMatrix) {
    for (auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        for (bool counter : {false, true}) {
            for (bool control : {false, true}) {
                for (bool rotating : {false, true}) {
                    run_spec_device_contract(*device_, noc, counter, rotating, control, false, true, false);
                }
                run_spec_device_contract(*device_, noc, counter, false, control, true, true, false);
                run_spec_device_contract(*device_, noc, counter, false, control, false, false, false);
                run_spec_device_contract(*device_, noc, counter, false, control, false, true, true);
            }
        }
    }
}

}  // namespace ttnn::kernel_lib::host::test
