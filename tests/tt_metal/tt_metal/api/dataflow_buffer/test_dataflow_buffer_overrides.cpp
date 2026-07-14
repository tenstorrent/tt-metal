// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DFB re-entry / entry-size / num-entries override runtime tests (legacy-only).

#include "dfb_test_common.hpp"

namespace tt::tt_metal {

// size/num-entries override + re-entry harness
struct DfbSizeOverride {
    std::optional<uint32_t> entry_size = std::nullopt;
    std::optional<uint32_t> num_entries = std::nullopt;
};

static void run_dfb_size_override_test(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    bool implicit_sync_param,
    uint32_t data_entry_size,   // bound-tensor page size == effective entry_size for every launch
    uint32_t entry_size_spec,   // DFB-declared entry_size
    uint32_t num_entries_spec,  // DFB-declared ring depth
    uint32_t workload,          // entries streamed (CTAs / entries_per_core)
    const std::vector<DfbSizeOverride>& launches,
    uint8_t num_producers = 1,
    uint8_t num_consumers = 1) {
    IDevice* device = mesh_device->get_devices()[0];
    const bool implicit_sync = (device->arch() == ARCH::QUASAR) && implicit_sync_param;

    // Per-thread compile-time loop bounds; the strided kernels split `workload` across the threads
    // (ceiling division, with a runtime entries_per_core bound to skip the tail).
    const uint32_t entries_per_producer = (workload + num_producers - 1) / num_producers;
    const uint32_t entries_per_consumer = (workload + num_consumers - 1) / num_consumers;

    const experimental::DFBSpecName DFB_NAME{"dfb"};
    const experimental::KernelSpecName PRODUCER{"producer"};
    const experimental::KernelSpecName CONSUMER{"consumer"};
    const experimental::TensorParamName IN_TENSOR{"in_tensor"};
    const experimental::TensorParamName OUT_TENSOR{"out_tensor"};

    const auto tensor_spec = make_flat_dram_tensor_spec(data_entry_size, workload);
    MeshTensor in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
    MeshTensor out_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});

    experimental::DataMovementHardwareConfig dm_producer_cfg;
    experimental::DataMovementHardwareConfig dm_consumer_cfg;
    if (device->arch() == ARCH::QUASAR) {
        dm_producer_cfg = experimental::DataMovementGen2Config{};
        dm_consumer_cfg = experimental::DataMovementGen2Config{};
    } else {
        dm_producer_cfg =
            experimental::DataMovementGen1Config{.processor = tt::tt_metal::DataMovementProcessor::RISCV_0};
        dm_consumer_cfg = experimental::DataMovementGen1Config{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::NOC_1};
    }

    experimental::KernelSpec producer_spec{
        .unique_id = PRODUCER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp",
        .num_threads = num_producers,
        .dfb_bindings = {experimental::ProducerOf(DFB_NAME, "out")},
        .tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}},
        .compile_time_args =
            {{"num_entries_per_producer", entries_per_producer},
             {"implicit_sync", static_cast<uint32_t>(implicit_sync ? 1u : 0u)},
             {"num_producers", static_cast<uint32_t>(num_producers)}},
        .runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}},
        .hw_config = dm_producer_cfg,
    };
    experimental::KernelSpec consumer_spec{
        .unique_id = CONSUMER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp",
        .num_threads = num_consumers,
        .dfb_bindings = {{
            .dfb_spec_name = DFB_NAME,
            .accessor_name = "in",
            .endpoint_type = experimental::DFBEndpointType::CONSUMER,
            .access_pattern = experimental::DFBAccessPattern::STRIDED,
        }},
        .tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}},
        .compile_time_args =
            {{"num_entries_per_consumer", entries_per_consumer},
             {"blocked_consumer", 0u},
             {"implicit_sync", static_cast<uint32_t>(implicit_sync ? 1u : 0u)},
             {"num_consumers", static_cast<uint32_t>(num_consumers)}},
        .runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}},
        .hw_config = dm_consumer_cfg,
    };
    // Implicit sync is gen2 only
    if (device->arch() == ARCH::QUASAR && !implicit_sync) {
        auto& producer_hw_config = std::get<experimental::DataMovementGen2Config>(
            std::get<experimental::DataMovementHardwareConfig>(producer_spec.hw_config));
        auto& consumer_hw_config = std::get<experimental::DataMovementGen2Config>(
            std::get<experimental::DataMovementHardwareConfig>(consumer_spec.hw_config));
        producer_hw_config.disable_dfb_implicit_sync_for_all = true;
        consumer_hw_config.disable_dfb_implicit_sync_for_all = true;
    }

    const CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0)));
    experimental::WorkUnitSpec wu{.name = "main", .kernels = {PRODUCER, CONSUMER}, .target_nodes = core_range_set};

    experimental::DataflowBufferSpec dfb_spec{
        .unique_id = DFB_NAME,
        .entry_size = entry_size_spec,
        .num_entries = num_entries_spec,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    experimental::ProgramSpec spec{
        .name = "dfb_size_override",
        .kernels = {producer_spec, consumer_spec},
        .dataflow_buffers = {dfb_spec},
        .tensor_parameters =
            {{.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
             {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()}},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    const auto input =
        tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, workload * data_entry_size / sizeof(uint32_t));

    using NodeRuntimeArgs = experimental::ProgramRunArgs::KernelRunArgs::NodeRuntimeArgs;
    uint32_t eff_entry_size = entry_size_spec;
    uint32_t eff_num_entries = num_entries_spec;

    for (const auto& step : launches) {
        if (step.entry_size.has_value()) {
            eff_entry_size = *step.entry_size;
        }
        if (step.num_entries.has_value()) {
            eff_num_entries = *step.num_entries;
        }
        // The bound tensors' page size is fixed, so the effective entry_size must match it each launch.
        ASSERT_EQ(eff_entry_size, data_entry_size)
            << "test setup error: effective entry_size must equal the bound tensor page size";

        experimental::ProgramRunArgs run_params;
        experimental::ProgramRunArgs::KernelRunArgs producer_params{.kernel = PRODUCER};
        producer_params.runtime_arg_values = {
            NodeRuntimeArgs{experimental::NodeCoord{0, 0}, {{"chunk_offset", 0u}, {"entries_per_core", workload}}}};
        experimental::ProgramRunArgs::KernelRunArgs consumer_params{.kernel = CONSUMER};
        consumer_params.runtime_arg_values = {
            NodeRuntimeArgs{experimental::NodeCoord{0, 0}, {{"chunk_offset", 0u}, {"entries_per_core", workload}}}};
        run_params.kernel_run_args = {producer_params, consumer_params};
        run_params.tensor_args = {
            {IN_TENSOR, experimental::TensorArgument{in_tensor}},
            {OUT_TENSOR, experimental::TensorArgument{out_tensor}}};
        // At most one dfb_run_overrides entry per DFB; carry whichever of entry_size/num_entries is set.
        if (step.entry_size.has_value() || step.num_entries.has_value()) {
            run_params.dfb_run_overrides.push_back(
                {.dfb = DFB_NAME, .entry_size = step.entry_size, .num_entries = step.num_entries});
        }
        experimental::SetProgramRunArgs(program, run_params);

        // Overrides are reflected in host-side state immediately.
        auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle(*DFB_NAME));
        EXPECT_EQ(dfb->config.entry_size, eff_entry_size);
        EXPECT_EQ(dfb->config.num_entries, eff_num_entries);

        detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), input);
        if (device->arch() == ARCH::QUASAR) {
            // TODO #38042: barrier not yet uplifted for Quasar; wait for the DRAM write to land.
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            std::vector<uint32_t> rdback;
            detail::ReadFromBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), rdback);
            tt_driver_atomics::mfence();
            ASSERT_EQ(rdback, input);
        }
        detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

        std::vector<uint32_t> output;
        detail::ReadFromBuffer(*out_tensor.mesh_buffer().get_reference_buffer(), output);
        EXPECT_EQ(output, input);
    }
}

// Gen1-only override / re-entry runtime tests (no 2.0 twin)
TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB_NumEntriesOverride_ReEntry) {
    run_dfb_size_override_test(
        this->devices_.at(0),
        GetParam(),
        /*data_entry_size=*/64,
        /*entry_size_spec=*/64,
        /*num_entries_spec=*/8,
        /*workload=*/8,
        {DfbSizeOverride{}, DfbSizeOverride{.num_entries = 4}, DfbSizeOverride{.num_entries = 16}});
}

// entry_size override: the DFB is declared at entry_size=32, but the bound tensors
// use page size 64; the override raises entry_size to 64 (matching the tensors) before launch.
TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB_EntrySizeOverride) {
    run_dfb_size_override_test(
        this->devices_.at(0),
        GetParam(),
        /*data_entry_size=*/64,
        /*entry_size_spec=*/32,
        /*num_entries_spec=*/8,
        /*workload=*/8,
        {DfbSizeOverride{.entry_size = 64}});
}

// Both parameters at once: entry_size 32->64 (tensors sized to 64) AND ring depth
// 8->4 in a single override, exercising base/limit recompute with simultaneously changed entry_size and
// capacity, plus reallocation for the new total_size.
TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB_BothOverride) {
    run_dfb_size_override_test(
        this->devices_.at(0),
        GetParam(),
        /*data_entry_size=*/64,
        /*entry_size_spec=*/32,
        /*num_entries_spec=*/8,
        /*workload=*/8,
        {DfbSizeOverride{.entry_size = 64, .num_entries = 4}});
}

// Symmetric 3P/3C: ring 6 -> 12 (1 TC per side).
TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB_NumEntriesOverride_ReEntry_3Sx3S) {
    DFB_SKIP_IF_UNSUPPORTED(3, 3);
    run_dfb_size_override_test(
        this->devices_.at(0),
        GetParam(),
        /*data_entry_size=*/64,
        /*entry_size_spec=*/64,
        /*num_entries_spec=*/6,
        /*workload=*/6,
        {DfbSizeOverride{}, DfbSizeOverride{.num_entries = 12}},
        /*num_producers=*/3,
        /*num_consumers=*/3);
}

// Asymmetric 1P/4C: ring 8 -> 16.
TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB_NumEntriesOverride_ReEntry_1Sx4S) {
    DFB_SKIP_IF_UNSUPPORTED(1, 4);
    run_dfb_size_override_test(
        this->devices_.at(0),
        GetParam(),
        /*data_entry_size=*/64,
        /*entry_size_spec=*/64,
        /*num_entries_spec=*/8,
        /*workload=*/8,
        {DfbSizeOverride{}, DfbSizeOverride{.num_entries = 16}},
        /*num_producers=*/1,
        /*num_consumers=*/4);
}

// Asymmetric 4P/1C: ring 8 -> 16.
TEST_P(DFBImplicitSyncParamFixture, DMTest1xDFB_NumEntriesOverride_ReEntry_4Sx1S) {
    DFB_SKIP_IF_UNSUPPORTED(4, 1);
    run_dfb_size_override_test(
        this->devices_.at(0),
        GetParam(),
        /*data_entry_size=*/64,
        /*entry_size_spec=*/64,
        /*num_entries_spec=*/8,
        /*workload=*/8,
        {DfbSizeOverride{}, DfbSizeOverride{.num_entries = 16}},
        /*num_producers=*/4,
        /*num_consumers=*/1);
}

}  // namespace tt::tt_metal
