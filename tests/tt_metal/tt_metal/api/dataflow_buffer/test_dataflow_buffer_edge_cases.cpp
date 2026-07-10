// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Edge-case coverage: counter-wrap, ring-pressure, decoy, long-run (Metal 2.0).

#include "dfb_test_common.hpp"

namespace tt::tt_metal {


// A1: DM->Tensix->DM decoy pipeline
enum class A1Transform { Identity, Relu };

static void run_a1_pipeline(const std::shared_ptr<distributed::MeshDevice>& mesh_device, A1Transform transform) {
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only (Gen2Config)";
    }

    IDevice* device = mesh_device->get_devices()[0];
    constexpr uint32_t entry_size = 2 * 32 * 32;  // bf16 tile = 2048 B
    constexpr uint32_t num_entries = 4;
    const m2::NodeCoord node{0, 0};

    // Tensors
    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, num_entries, DataType::BFLOAT16);
    auto in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
    auto out_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});

    const m2::DFBSpecName DFB_IN{"dfb_in"};
    const m2::DFBSpecName DFB_OUT{"dfb_out"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::KernelSpecName COMPUTE{"compute"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    // DFBs — disable_implicit_sync=true matches kernels' explicit credit-flow path.
    m2::DataflowBufferSpec dfb_in{
        .unique_id = DFB_IN,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    m2::DataflowBufferSpec dfb_out{
        .unique_id = DFB_OUT,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // Producer kernel: writes input → DFB_IN
    auto producer = make_dm_dfb_producer(PRODUCER, DFB_IN, IN_TENSOR, num_entries, /*implicit_sync=*/false);

    // Compute kernel: dfb_in → (relu or identity) → dfb_out
    const std::string compute_source = (transform == A1Transform::Relu)
                                           ? "tests/tt_metal/tt_metal/test_kernels/compute/dfb_eltwise_relu_2_0.cpp"
                                           : "tests/tt_metal/tt_metal/test_kernels/compute/dfb_eltwise_copy_2_0.cpp";
    auto compute = make_compute_kernel(COMPUTE, compute_source);
    compute.dfb_bindings = {
        {.dfb_spec_name = DFB_IN,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
        {.dfb_spec_name = DFB_OUT,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
    };
    compute.compile_time_args = {{"per_core_tile_cnt", num_entries}};

    // Consumer kernel: DFB_OUT → output tensor
    auto consumer = make_dm_dfb_consumer(
        CONSUMER, DFB_OUT, OUT_TENSOR, num_entries, /*blocked_consumer=*/false, /*implicit_sync=*/false);

    // All-pass set dfb_in/dfb_out .disable_implicit_sync = true; #45160 moved that onto the
    // Gen2 DM config, so disable per DM endpoint (the compute stage is Tensix → no DM side).
    disable_implicit_sync_for(producer, DFB_IN);
    disable_implicit_sync_for(consumer, DFB_OUT);

    m2::WorkUnitSpec wu{
        .name = "wu",
        .kernels = {PRODUCER, CONSUMER, COMPUTE},
        .target_nodes = node,
    };

    m2::ProgramSpec spec{
        .name = "a1_2_0",
        .kernels = {producer, consumer, compute},
        .dataflow_buffers = {dfb_in, dfb_out},
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };

    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values = {{.node = node, .args = {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}}},
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values = {{.node = node, .args = {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}}},
        },
        m2::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {
        {IN_TENSOR, std::cref(in_tensor)},
        {OUT_TENSOR, std::cref(out_tensor)},
    };
    m2::SetProgramRunArgs(program, params);

    // Stimulus
    const uint32_t total_bytes = entry_size * num_entries;
    auto input = (transform == A1Transform::Relu)
                     ? create_random_vector_of_bfloat16(total_bytes, 1.0f, 0xA1A1)   // positive only
                     : create_random_vector_of_bfloat16(total_bytes, 2.0f, 0xA1A1);  // [-1,1]
    detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), input);
    m2_writeshard_barrier_uint32(device, in_tensor, input);

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(*out_tensor.mesh_buffer().get_reference_buffer(), output);

    if (transform == A1Transform::Relu) {
        // Positive bf16 inputs → relu identity, allow bf16 tolerance.
        EXPECT_TRUE(
            packed_uint32_t_vector_comparison(output, input, [](float a, float b) { return std::abs(a - b) < 0.01f; }));
    } else {
        EXPECT_EQ(input, output);
    }
}

TEST_F(MeshDeviceFixture, A1_2_0_DMTensixDMTest2xDFB1Sx1S) {
    run_a1_pipeline(this->devices_.at(0), A1Transform::Identity);
}

TEST_F(MeshDeviceFixture, A1_2_0_DMTensixDMTest2xDFB1Sx1S_Relu) {
    run_a1_pipeline(this->devices_.at(0), A1Transform::Relu);
}

// B: implicit-sync edge-case regressions
static void run_dm_dfb_dm_implicit_sync_2_0(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t num_iterations,
    bool implicit_sync,
    uint32_t entry_size = 1024,
    uint32_t num_entries = 16,
    uint32_t total_tiles = 16) {
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Implicit sync is Quasar-only";
    }

    IDevice* device = mesh_device->get_devices()[0];
    const m2::NodeCoord node{0, 0};

    const m2::DFBSpecName DFB{"dfb"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, total_tiles, DataType::UINT32);
    auto in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
    auto out_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});

    // DFB-level implicit_sync must match the kernels' CTA (inverted polarity).
    m2::DataflowBufferSpec dfb{
        .unique_id = DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    auto producer = make_dm_dfb_producer(PRODUCER, DFB, IN_TENSOR, total_tiles, implicit_sync);

    auto consumer =
        make_dm_dfb_consumer(CONSUMER, DFB, OUT_TENSOR, total_tiles, /*blocked_consumer=*/false, implicit_sync);

    // All-pass: dfb.disable_implicit_sync = !implicit_sync (now per-DM-endpoint, post-#45160).
    maybe_disable_implicit_sync(producer, implicit_sync, DFB);
    maybe_disable_implicit_sync(consumer, implicit_sync, DFB);

    m2::WorkUnitSpec wu{.name = "wu", .kernels = {PRODUCER, CONSUMER}, .target_nodes = node};

    m2::ProgramSpec spec{
        .name = "dm_dfb_dm",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb},
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };

    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values = {{.node = node, .args = {{"chunk_offset", 0u}, {"entries_per_core", total_tiles}}}},
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values = {{.node = node, .args = {{"chunk_offset", 0u}, {"entries_per_core", total_tiles}}}},
        },
    };
    params.tensor_args = {
        {IN_TENSOR, std::cref(in_tensor)},
        {OUT_TENSOR, std::cref(out_tensor)},
    };
    m2::SetProgramRunArgs(program, params);

    const uint32_t total_words = entry_size * total_tiles / sizeof(uint32_t);
    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 1000000, total_words);
    detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), input);
    m2_writeshard_barrier_uint32(device, in_tensor, input);

    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);
    }

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(*out_tensor.mesh_buffer().get_reference_buffer(), output);
    EXPECT_EQ(input, output) << "M2 DM→DFB→DM identity mismatch";
}

TEST_F(MeshDeviceFixture, B1_2_0_DM0NoKernel_TensixDMImplicitSync) {
    // B1 = single-iteration implicit-sync DM→DFB→DM. DM0-no-kernel coverage is
    // an internal-state regression that fires whether or not we explicitly
    // skip DM0; the helper produces the same wire-level traffic.
    run_dm_dfb_dm_implicit_sync_2_0(this->devices_.at(0), /*num_iterations=*/1, /*implicit_sync=*/true);
}

TEST_F(MeshDeviceFixture, B1b_2_0_DM0IdleSubordinateRuns_TensixDMImplicitSync) {
    // B1b same shape as B1 with an extra iter to expose stale-credit edge.
    run_dm_dfb_dm_implicit_sync_2_0(this->devices_.at(0), /*num_iterations=*/2, /*implicit_sync=*/true);
}

TEST_F(MeshDeviceFixture, B3_2_0_TailCreditRace_RepeatedImplicitSync_DMDM) {
    // B3: 3 repeated implicit-sync iterations exercise the tail-credit race.
    run_dm_dfb_dm_implicit_sync_2_0(this->devices_.at(0), /*num_iterations=*/3, /*implicit_sync=*/true);
}

// D1: long implicit-sync run past counter wrap
TEST_F(MeshDeviceFixture, D1_2_0_LongImplicitSync_PostCounterWrap) {
    auto& mesh_device = this->devices_.at(0);
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Implicit sync is Quasar-only";
    }

    IDevice* device = mesh_device->get_devices()[0];
    constexpr uint32_t kPreloadValue = 65528;
    constexpr uint32_t kPushTiles = 32;
    constexpr uint32_t kEntrySize = 1024;
    constexpr uint32_t kRingEntries = 16;
    const m2::NodeCoord node{0, 0};

    const m2::DFBSpecName DFB{"dfb"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};
    // Two single-writer semaphores form a producer<->consumer rendezvous so that
    // neither DM kernel enters its data loop until BOTH the producer's posted and
    // the consumer's acked counters have been preloaded (occupancy provably 0).
    const m2::SemaphoreSpecName SEM_PROD_READY{"sem_prod_ready"};
    const m2::SemaphoreSpecName SEM_CONS_READY{"sem_cons_ready"};

    const auto tensor_spec = make_flat_dram_tensor_spec(kEntrySize, kPushTiles, DataType::UINT32);
    auto in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
    auto out_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});

    m2::DataflowBufferSpec dfb{
        .unique_id = DFB,
        .entry_size = kEntrySize,
        .num_entries = kRingEntries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    auto producer =
        make_dm_kernel(PRODUCER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_with_tc_preload_2_0.cpp");
    producer.dfb_bindings = {
        {.dfb_spec_name = DFB,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    producer.tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}};
    producer.compile_time_args = {
        {"num_entries_per_producer", kPushTiles}, {"implicit_sync", 1u}, {"kPreloadPostedValue", kPreloadValue}};
    producer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};
    producer.semaphore_bindings = {
        {.semaphore_spec_name = SEM_PROD_READY, .accessor_name = "prod_ready"},
        {.semaphore_spec_name = SEM_CONS_READY, .accessor_name = "cons_ready"}};

    auto consumer =
        make_dm_kernel(CONSUMER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer_with_tc_preload_2_0.cpp");
    consumer.dfb_bindings = {
        {.dfb_spec_name = DFB,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    consumer.tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}};
    consumer.compile_time_args = {
        {"num_entries_per_consumer", kPushTiles},
        {"blocked_consumer", 0u},
        {"implicit_sync", 1u},
        {"kPreloadAckedValue", kPreloadValue}};
    consumer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};
    consumer.semaphore_bindings = {
        {.semaphore_spec_name = SEM_PROD_READY, .accessor_name = "prod_ready"},
        {.semaphore_spec_name = SEM_CONS_READY, .accessor_name = "cons_ready"}};

    m2::WorkUnitSpec wu{.name = "wu", .kernels = {PRODUCER, CONSUMER}, .target_nodes = node};

    m2::ProgramSpec spec{
        .name = "d1_2_0",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb},
        .semaphores =
            {
                m2::SemaphoreSpec{.unique_id = SEM_PROD_READY, .target_nodes = node},
                m2::SemaphoreSpec{.unique_id = SEM_CONS_READY, .target_nodes = node},
            },
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };

    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values = {{.node = node, .args = {{"chunk_offset", 0u}, {"entries_per_core", kPushTiles}}}},
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values = {{.node = node, .args = {{"chunk_offset", 0u}, {"entries_per_core", kPushTiles}}}},
        },
    };
    params.tensor_args = {
        {IN_TENSOR, std::cref(in_tensor)},
        {OUT_TENSOR, std::cref(out_tensor)},
    };
    m2::SetProgramRunArgs(program, params);

    auto input = create_random_vector_of_bfloat16(kPushTiles * kEntrySize, 1.0f, 0xD1D1);
    detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), input);
    m2_writeshard_barrier_uint32(device, in_tensor, input);

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(*out_tensor.mesh_buffer().get_reference_buffer(), output);

    // Diagnostic: on mismatch, dump first divergent tile + per-tile histogram so
    // we can characterize the failure mode (all-zeros from start, wrap-point only, etc.).
    if (input != output) {
        constexpr size_t kU32PerTile = kEntrySize / sizeof(uint32_t);
        auto mm = std::mismatch(input.begin(), input.end(), output.begin());
        size_t first_diff = mm.first - input.begin();
        size_t mismatch_count = std::transform_reduce(
            input.begin(), input.end(), output.begin(), size_t{0}, std::plus<>{}, std::not_equal_to<>{});
        if (first_diff < input.size()) {
            std::vector<size_t> per_tile_mismatches(kPushTiles, 0);
            for (size_t i = 0; i < input.size(); ++i) {
                if (input[i] != output[i]) {
                    per_tile_mismatches[i / kU32PerTile]++;
                }
            }
            log_info(
                tt::LogTest,
                "D1_2_0 first mismatch at idx {} (tile {}, word {}): input=0x{:x} output=0x{:x}. Total {}/{}.",
                first_diff,
                first_diff / kU32PerTile,
                first_diff % kU32PerTile,
                input[first_diff],
                output[first_diff],
                mismatch_count,
                input.size());
            // Wrap is at tile 8 (posted=65528+8 = 65536 wraps to 0). Print per-tile state.
            for (size_t t = 0; t < kPushTiles; ++t) {
                if (per_tile_mismatches[t] > 0) {
                    log_info(
                        tt::LogTest,
                        "D1_2_0 tile {}: {}/{} words mismatched (input[0]=0x{:x} output[0]=0x{:x}){}",
                        t,
                        per_tile_mismatches[t],
                        kU32PerTile,
                        input[t * kU32PerTile],
                        output[t * kU32PerTile],
                        t == 8 ? "  <-- wrap point" : "");
                }
            }
        }
    }
    EXPECT_EQ(input, output) << "M2 D1: identity copy across uint16 TC-counter wrap point failed";
}

// D2: all-DMs-concurrent ring saturation
static void run_d2_all_dms_concurrent_2_0(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, bool implicit_sync) {
    run_dm_dfb_dm_implicit_sync_2_0(
        mesh_device,
        /*num_iterations=*/1,
        implicit_sync,
        /*entry_size=*/1024,
        /*num_entries=*/24,
        /*total_tiles=*/96);
}

TEST_F(MeshDeviceFixture, D2_2_0_AllDMsConcurrent_6Sx2S_ImplicitOff) {
    run_d2_all_dms_concurrent_2_0(this->devices_.at(0), /*implicit_sync=*/false);
}

TEST_F(MeshDeviceFixture, D2_2_0_AllDMsConcurrent_6Sx2S_ImplicitOn) {
    run_d2_all_dms_concurrent_2_0(this->devices_.at(0), /*implicit_sync=*/true);
}

// D3: multi-core two-groups-via-decoy
TEST_F(MeshDeviceFixture, D3_2_0_MultiCoreDFB_TwoGroupsViaDecoy) {
    auto& mesh_device = this->devices_.at(0);
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "TC-based grouping is Quasar-only";
    }

    IDevice* device = mesh_device->get_devices()[0];
    CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t num_workers = grid.x * grid.y;
    if (num_workers < 2) {
        GTEST_SKIP() << "Need >= 2 Tensix cores; device has " << num_workers
                     << " (single-Tensix emulator?). Run on silicon or a multi-Tensix sim.";
    }

    constexpr uint32_t entries_per_core = 16;
    constexpr uint32_t entry_size = 1024;
    constexpr uint32_t num_entries = 16;
    const m2::NodeCoord core_a{0, 0};
    const m2::NodeCoord core_b{1, 0};

    const m2::DFBSpecName DECOY_DFB{"decoy_dfb"};
    const m2::DFBSpecName SHARED_DFB{"shared_dfb"};
    const m2::KernelSpecName DECOY_PRODUCER{"decoy_producer"};
    const m2::KernelSpecName DECOY_CONSUMER{"decoy_consumer"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, 2 * entries_per_core, DataType::UINT32);
    auto in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
    auto out_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});

    // Decoy DFB: lives on core A only.
    m2::DataflowBufferSpec decoy_dfb{
        .unique_id = DECOY_DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    // Shared DFB: lives on cores A and B.
    m2::DataflowBufferSpec shared_dfb{
        .unique_id = SHARED_DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // Decoy producer/consumer on core A only — no-ops, just claim TC slots.
    auto decoy_producer =
        make_dm_kernel(DECOY_PRODUCER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_2_0.cpp");
    decoy_producer.dfb_bindings = {
        {.dfb_spec_name = DECOY_DFB,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    decoy_producer.tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}};
    decoy_producer.compile_time_args = {{"num_entries_per_producer", 0u}, {"implicit_sync", 0u}};
    decoy_producer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};

    auto decoy_consumer =
        make_dm_kernel(DECOY_CONSUMER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer_2_0.cpp");
    decoy_consumer.dfb_bindings = {
        {.dfb_spec_name = DECOY_DFB,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    decoy_consumer.tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}};
    decoy_consumer.compile_time_args = {
        {"num_entries_per_consumer", 0u}, {"blocked_consumer", 0u}, {"implicit_sync", 0u}};
    decoy_consumer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};

    // Real shared producer/consumer across A and B (uses dfb::shared kernel variant).
    auto producer =
        make_dm_kernel(PRODUCER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_with_id_2_0.cpp");
    producer.dfb_bindings = {
        {.dfb_spec_name = SHARED_DFB,
         .accessor_name = "shared",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    producer.tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}};
    producer.compile_time_args = {{"num_entries_per_producer", entries_per_core}, {"implicit_sync", 0u}};
    producer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};

    auto consumer = make_dm_kernel(CONSUMER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer_2_0.cpp");
    // NOTE: dfb_consumer.cpp uses dfb::in — rebinding it to the SHARED_DFB by
    // accessor_name "in" is valid; the kernel doesn't care about the host
    // DFB's spec name.
    consumer.dfb_bindings = {
        {.dfb_spec_name = SHARED_DFB,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    consumer.tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}};
    consumer.compile_time_args = {
        {"num_entries_per_consumer", entries_per_core}, {"blocked_consumer", 0u}, {"implicit_sync", 0u}};
    consumer.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};

    // All-pass disabled decoy_dfb/shared_dfb implicit sync (now per-DM-endpoint, post-#45160).
    disable_implicit_sync_for(decoy_producer, DECOY_DFB);
    disable_implicit_sync_for(decoy_consumer, DECOY_DFB);
    disable_implicit_sync_for(producer, SHARED_DFB);
    disable_implicit_sync_for(consumer, SHARED_DFB);

    // WUs: decoy on core A only; shared on both. WUs cannot overlap target_nodes,
    // so we put decoy on a single-core WU and shared on a disjoint range.
    m2::WorkUnitSpec decoy_wu{
        .name = "decoy_wu",
        .kernels = {DECOY_PRODUCER, DECOY_CONSUMER},
        .target_nodes = core_a,
    };
    m2::WorkUnitSpec shared_wu{
        .name = "shared_wu",
        .kernels = {PRODUCER, CONSUMER},
        .target_nodes = core_b,  // start with core_b only, since decoy claims core_a
    };

    m2::ProgramSpec spec{
        .name = "d3_2_0",
        .kernels = {decoy_producer, decoy_consumer, producer, consumer},
        .dataflow_buffers = {decoy_dfb, shared_dfb},
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {decoy_wu, shared_wu},
    };

    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = DECOY_PRODUCER,
            .runtime_arg_values = {{.node = core_a, .args = {{"chunk_offset", 0u}, {"entries_per_core", 0u}}}},
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = DECOY_CONSUMER,
            .runtime_arg_values = {{.node = core_a, .args = {{"chunk_offset", 0u}, {"entries_per_core", 0u}}}},
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = PRODUCER,
            .runtime_arg_values =
                {{.node = core_b, .args = {{"chunk_offset", 0u}, {"entries_per_core", entries_per_core}}}},
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values =
                {{.node = core_b, .args = {{"chunk_offset", 0u}, {"entries_per_core", entries_per_core}}}},
        },
    };
    params.tensor_args = {
        {IN_TENSOR, std::cref(in_tensor)},
        {OUT_TENSOR, std::cref(out_tensor)},
    };
    m2::SetProgramRunArgs(program, params);

    auto input = create_constant_vector_of_bfloat16(2 * entries_per_core * entry_size, 1.0f);
    detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), input);
    m2_writeshard_barrier_uint32(device, in_tensor, input);

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(*out_tensor.mesh_buffer().get_reference_buffer(), output);
    // For the m2 version we just verify the shared DFB ran end-to-end on core B.
    // The "two DfbGroups" assertion from the legacy test depends on inspecting
    // program.impl() before LaunchProgram; we keep that for the legacy test and
    // make this m2 variant a simpler end-to-end correctness check.
    //
    // The shared pipeline runs only on core_b (the decoy claims core_a and is a no-op), so it produces
    // only the first entries_per_core of the 2*entries_per_core tensor; the out_tensor's second half is
    // never written. Compare only the produced first half.
    ASSERT_EQ(input.size(), output.size());
    const size_t produced = output.size() / 2;
    input.resize(produced);
    output.resize(produced);
    EXPECT_EQ(input, output) << "M2 D3: shared DFB pipeline mismatch (core_b produced half)";
}

// ring-pressure scenarios (tight rings, heavy wraparound)
TEST_P(DFBImplicitSyncParamFixture_2_0, DMTest1xDFB_RingPressure_1Sx1S_2_0) {
    M2SingleDFBParams params{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::DM,
        .num_producers = 1,
        .num_consumers = 1,
        .implicit_sync = GetParam(),
        .num_entries = 16,
        .num_entries_in_buffer = 32,
    };
    run_single_dfb_program_2_0(this->devices_.at(0), params);
}

TEST_P(DFBImplicitSyncParamFixture_2_0, DMTest1xDFB_RingPressure_3Sx3S_2_0) {
    // M2 caps user DM cores per WU at 6 (legacy 4Sx4S=8 doesn't fit on Gen2).
    M2SingleDFBParams params{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::DM,
        .num_producers = 3,
        .num_consumers = 3,
        .implicit_sync = GetParam(),
        .num_entries = default_num_entries(3, 3),
        .num_entries_in_buffer = 27,
    };
    run_single_dfb_program_2_0(this->devices_.at(0), params);
}

TEST_P(DFBImplicitSyncParamFixture_2_0, TensixDMTest1xDFB_RingPressure_2Sx4S_2_0) {
    M2SingleDFBParams params{
        .producer_type = M2PorCType::TENSIX,
        .consumer_type = M2PorCType::DM,
        .num_producers = 2,
        .num_consumers = 4,
        .implicit_sync = GetParam(),
        .num_entries = 16,
        .num_entries_in_buffer = 32,
    };
    run_single_dfb_program_2_0(this->devices_.at(0), params);
}

// 4 DM producers + 4 Tensix consumers ALL, num_entries=4 → capacity=1: maximum
// ring pressure on the remapper fan-out path (1 DM post → 4 UNPACK TC acks).
TEST_P(DFBImplicitSyncParamFixture_2_0, DMTensixTest1xDFB_RingPressure_4Sx4A_2_0) {
    M2SingleDFBParams params{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::TENSIX,
        .num_producers = 4,
        .num_consumers = 4,
        .pap = m2::DFBAccessPattern::STRIDED,
        .cap = m2::DFBAccessPattern::ALL,
        .implicit_sync = GetParam(),
        .num_entries = 4,
        .num_entries_in_buffer = 64,
    };
    run_single_dfb_program_2_0(this->devices_.at(0), params);
}

}  // namespace tt::tt_metal
