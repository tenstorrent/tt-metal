// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-core and multi-DFB (concurrent/sequential) tests (Metal 2.0).

#include "dfb_test_common.hpp"

namespace tt::tt_metal {


// multi-core + concurrent/sequential multi-DFB harnesses (Metal 2.0)
static void run_single_dfb_multicore_2_0(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t num_producers,
    uint32_t num_consumers,
    m2::DFBAccessPattern pap,
    m2::DFBAccessPattern cap,
    bool implicit_sync) {
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only";
    }
    IDevice* device = mesh_device->get_devices()[0];
    CoreCoord grid = device->compute_with_storage_grid_size();
    if (grid.x * grid.y < 2) {
        GTEST_SKIP() << "Multi-core test requires >= 2 Tensix cores";
    }
    // DM-DM ALL + implicit sync is unsupported (legacy parity); the single-core path
    // (run_single_dfb_program_2_0) skips it too. The multicore helper was missing this guard, so the
    // *_1Sx4A_2_0/ImplicitSyncTrue variant would fail on the device instead of skipping.
    if (cap == m2::DFBAccessPattern::ALL && implicit_sync) {
        GTEST_SKIP() << "DM-DM ALL with implicit_sync not supported (legacy parity)";
    }

    constexpr uint32_t entry_size = 1024;
    const uint32_t num_entries = default_num_entries(num_producers, num_consumers);
    const m2::NodeCoord core_a{0, 0};
    const m2::NodeCoord core_b{1, 0};
    const bool is_all = (cap == m2::DFBAccessPattern::ALL);

    const m2::DFBSpecName DFB{"dfb"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    // Each core owns num_entries slots → total = 2 * num_entries.
    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, 2 * num_entries, DataType::UINT32);
    auto in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
    auto out_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});

    m2::DataflowBufferSpec dfb_spec{
        .unique_id = DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    const uint32_t per_producer = (num_entries + num_producers - 1) / num_producers;
    const uint32_t per_consumer = is_all ? num_entries : (num_entries + num_consumers - 1) / num_consumers;

    auto producer = make_dm_dfb_producer(PRODUCER, DFB, IN_TENSOR, per_producer, implicit_sync, pap, num_producers);

    auto consumer =
        make_dm_dfb_consumer(CONSUMER, DFB, OUT_TENSOR, per_consumer, is_all, implicit_sync, cap, num_consumers);

    // All-pass: dfb.disable_implicit_sync = !implicit_sync (now per-DM-endpoint, post-#45160).
    maybe_disable_implicit_sync(producer, implicit_sync, DFB);
    maybe_disable_implicit_sync(consumer, implicit_sync, DFB);

    // Single WU covering both cores via NodeRange.
    m2::WorkUnitSpec wu{
        .name = "wu",
        .kernels = {PRODUCER, CONSUMER},
        .target_nodes = m2::NodeRange{core_a, core_b},
    };

    m2::ProgramSpec spec{
        .name = "multicore_dfb_2_0",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb_spec},
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
        {.kernel = PRODUCER,
         .runtime_arg_values =
             {{.node = core_a, .args = {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}},
              {.node = core_b, .args = {{"chunk_offset", num_entries}, {"entries_per_core", num_entries}}}}},
        {.kernel = CONSUMER,
         .runtime_arg_values =
             {{.node = core_a, .args = {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}},
              {.node = core_b, .args = {{"chunk_offset", num_entries}, {"entries_per_core", num_entries}}}}},
    };
    params.tensor_args = {
        {IN_TENSOR, std::cref(in_tensor)},
        {OUT_TENSOR, std::cref(out_tensor)},
    };
    m2::SetProgramRunArgs(program, params);

    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(
        0, 1000000, 2 * num_entries * entry_size / sizeof(uint32_t));
    detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), input);
    m2_writeshard_barrier_uint32(device, in_tensor, input);

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(*out_tensor.mesh_buffer().get_reference_buffer(), output);
    EXPECT_EQ(input, output) << "M2 multi-core DFB identity mismatch";
}

static void run_concurrent_dfbs_program_2_0(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t num_dfbs,
    uint32_t entry_size,
    uint32_t entries_per_dfb,
    bool implicit_sync) {
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Concurrent DFB tests require Quasar";
    }
    if (2 * num_dfbs > 6) {
        GTEST_SKIP() << "2*num_dfbs must fit in 6 Quasar DM threads";
    }

    IDevice* device = mesh_device->get_devices()[0];
    const m2::NodeCoord node{0, 0};

    // One big DRAM tensor sliced num_dfbs ways for input + same for output.
    const uint32_t total_entries = num_dfbs * entries_per_dfb;
    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, total_entries, DataType::UINT32);
    auto in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
    auto out_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});

    // Build N DFBs + N producer kernels + N consumer kernels.
    std::vector<m2::DataflowBufferSpec> dfbs;
    std::vector<m2::KernelSpec> kernels;
    std::vector<m2::KernelSpecName> kernel_names;
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        const m2::DFBSpecName dfb_id{"dfb_" + std::to_string(i)};
        const m2::KernelSpecName prod_id{"producer_" + std::to_string(i)};
        const m2::KernelSpecName cons_id{"consumer_" + std::to_string(i)};
        dfbs.push_back({
            .unique_id = dfb_id,
            .entry_size = entry_size,
            .num_entries = entries_per_dfb,
            .data_format_metadata = tt::DataFormat::Float16_b,
        });
        auto prod = make_dm_kernel(prod_id, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_multi_producer_2_0.cpp");
        prod.dfb_bindings = {
            {.dfb_spec_name = dfb_id,
             .accessor_name = "out",
             .endpoint_type = m2::DFBEndpointType::PRODUCER,
             .access_pattern = m2::DFBAccessPattern::STRIDED}};
        prod.tensor_bindings = {
            {.tensor_parameter_name = m2::TensorParamName{"in_tensor"}, .accessor_name = "src_tensor"}};
        prod.compile_time_args = {
            {"num_entries_per_producer", entries_per_dfb},
            {"implicit_sync", implicit_sync ? 1u : 0u},
            {"chunk_offset", i * entries_per_dfb}};
        maybe_disable_implicit_sync(prod, implicit_sync, dfb_id);  // all-pass: !implicit_sync (post-#45160)
        kernels.push_back(prod);
        kernel_names.push_back(prod_id);

        auto cons = make_dm_kernel(cons_id, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_multi_consumer_2_0.cpp");
        cons.dfb_bindings = {
            {.dfb_spec_name = dfb_id,
             .accessor_name = "in",
             .endpoint_type = m2::DFBEndpointType::CONSUMER,
             .access_pattern = m2::DFBAccessPattern::STRIDED}};
        cons.tensor_bindings = {
            {.tensor_parameter_name = m2::TensorParamName{"out_tensor"}, .accessor_name = "dst_tensor"}};
        cons.compile_time_args = {
            {"num_entries_per_consumer", entries_per_dfb},
            {"implicit_sync", implicit_sync ? 1u : 0u},
            {"chunk_offset", i * entries_per_dfb}};
        maybe_disable_implicit_sync(cons, implicit_sync, dfb_id);  // all-pass: !implicit_sync (post-#45160)
        kernels.push_back(cons);
        kernel_names.push_back(cons_id);
    }

    m2::WorkUnitSpec wu{.name = "wu", .kernels = kernel_names, .target_nodes = node};

    m2::ProgramSpec spec{
        .name = "concurrent_dfbs_2_0",
        .kernels = kernels,
        .dataflow_buffers = dfbs,
        .tensor_parameters =
            {
                {.unique_id = m2::TensorParamName{"in_tensor"}, .spec = in_tensor.tensor_spec()},
                {.unique_id = m2::TensorParamName{"out_tensor"}, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };

    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);

    m2::ProgramRunArgs params;
    for (const auto& name : kernel_names) {
        params.kernel_run_args.push_back({.kernel = name});
    }
    params.tensor_args = {
        {m2::TensorParamName{"in_tensor"}, std::cref(in_tensor)},
        {m2::TensorParamName{"out_tensor"}, std::cref(out_tensor)},
    };
    m2::SetProgramRunArgs(program, params);

    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(
        0, 1000000, total_entries * entry_size / sizeof(uint32_t));
    detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), input);
    m2_writeshard_barrier_uint32(device, in_tensor, input);

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(*out_tensor.mesh_buffer().get_reference_buffer(), output);
    EXPECT_EQ(input, output) << "M2 concurrent DFBs mismatch";
}

struct M2SeqDFBSpec {
    m2::DFBAccessPattern cap;
};

static void run_sequential_4_dfbs_2_0(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const std::array<M2SeqDFBSpec, 4>& dfb_specs,
    uint32_t num_producers,
    uint32_t num_consumers,
    bool implicit_sync) {
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Sequential 4-DFB test requires Quasar";
    }
    if (num_producers + num_consumers > 6) {
        GTEST_SKIP() << "num_p + num_c must fit in 6 Quasar DM threads";
    }
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t entry_size = 1024;
    // num_entries must be divisible by both num_producers and num_consumers.
    const uint32_t lcm_pc = (num_producers * num_consumers) / std::gcd(num_producers, num_consumers);
    const uint32_t num_entries = ((16u + lcm_pc - 1u) / lcm_pc) * lcm_pc;
    const uint32_t entries_per_producer = num_entries / num_producers;
    const uint32_t entries_per_consumer_strided = num_entries / num_consumers;
    const uint32_t entries_per_consumer_all = num_entries;
    const m2::NodeCoord node{0, 0};

    constexpr std::array<const char*, 4> DFB_NAMES{"buf_0", "buf_1", "buf_2", "buf_3"};
    constexpr std::array<const char*, 4> SRC_NAMES{"src_0", "src_1", "src_2", "src_3"};
    constexpr std::array<const char*, 4> DST_NAMES{"dst_0", "dst_1", "dst_2", "dst_3"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};

    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, num_entries, DataType::UINT32);
    std::vector<MeshTensor> in_tensors, out_tensors;
    in_tensors.reserve(4);
    out_tensors.reserve(4);
    for (uint32_t i = 0; i < 4; ++i) {
        in_tensors.push_back(MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{}));
        out_tensors.push_back(MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{}));
    }

    std::vector<m2::DataflowBufferSpec> dfbs;
    dfbs.reserve(4);
    for (uint32_t i = 0; i < 4; ++i) {
        dfbs.push_back({
            .unique_id = m2::DFBSpecName{DFB_NAMES[i]},
            .entry_size = entry_size,
            .num_entries = num_entries,
            .data_format_metadata = tt::DataFormat::Float16_b,
        });
    }

    auto producer = make_dm_kernel(
        PRODUCER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_seq_producer_quad_2_0.cpp", num_producers);
    for (uint32_t i = 0; i < 4; ++i) {
        producer.dfb_bindings.push_back(
            {.dfb_spec_name = m2::DFBSpecName{DFB_NAMES[i]},
             .accessor_name = DFB_NAMES[i],
             .endpoint_type = m2::DFBEndpointType::PRODUCER,
             .access_pattern = m2::DFBAccessPattern::STRIDED});
        producer.tensor_bindings.push_back(
            {.tensor_parameter_name = m2::TensorParamName{SRC_NAMES[i]}, .accessor_name = SRC_NAMES[i]});
    }
    producer.compile_time_args = {
        {"num_entries_per_producer", entries_per_producer}, {"implicit_sync", implicit_sync ? 1u : 0u}};

    auto consumer = make_dm_kernel(
        CONSUMER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_seq_consumer_quad_2_0.cpp", num_consumers);
    for (uint32_t i = 0; i < 4; ++i) {
        const auto cap = dfb_specs[i].cap;
        consumer.dfb_bindings.push_back(
            {.dfb_spec_name = m2::DFBSpecName{DFB_NAMES[i]},
             .accessor_name = DFB_NAMES[i],
             .endpoint_type = m2::DFBEndpointType::CONSUMER,
             .access_pattern = cap});
        consumer.tensor_bindings.push_back(
            {.tensor_parameter_name = m2::TensorParamName{DST_NAMES[i]}, .accessor_name = DST_NAMES[i]});
    }
    consumer.compile_time_args = {
        {"implicit_sync", implicit_sync ? 1u : 0u},
        {"entries_per_consumer_strided", entries_per_consumer_strided},
        {"entries_per_consumer_all", entries_per_consumer_all},
        {"is_blocked_0", dfb_specs[0].cap == m2::DFBAccessPattern::ALL ? 1u : 0u},
        {"is_blocked_1", dfb_specs[1].cap == m2::DFBAccessPattern::ALL ? 1u : 0u},
        {"is_blocked_2", dfb_specs[2].cap == m2::DFBAccessPattern::ALL ? 1u : 0u},
        {"is_blocked_3", dfb_specs[3].cap == m2::DFBAccessPattern::ALL ? 1u : 0u},
    };

    // All-pass: each DFB .disable_implicit_sync = !implicit_sync (now per-DM-endpoint, post-#45160).
    for (uint32_t i = 0; i < 4; ++i) {
        maybe_disable_implicit_sync(producer, implicit_sync, m2::DFBSpecName{DFB_NAMES[i]});
        maybe_disable_implicit_sync(consumer, implicit_sync, m2::DFBSpecName{DFB_NAMES[i]});
    }

    std::vector<m2::TensorParameter> tensor_parameters;
    tensor_parameters.reserve(8);
    for (uint32_t i = 0; i < 4; ++i) {
        tensor_parameters.push_back({.unique_id = m2::TensorParamName{SRC_NAMES[i]}, .spec = tensor_spec});
        tensor_parameters.push_back({.unique_id = m2::TensorParamName{DST_NAMES[i]}, .spec = tensor_spec});
    }

    m2::ProgramSpec spec{
        .name = "seq_4dfb_2_0",
        .kernels = {producer, consumer},
        .dataflow_buffers = dfbs,
        .tensor_parameters = tensor_parameters,
        .work_units = {m2::WorkUnitSpec{
            .name = "wu",
            .kernels = {PRODUCER, CONSUMER},
            .target_nodes = node,
        }},
    };

    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);

    m2::ProgramRunArgs params;
    // Kernels with no runtime args still need a KernelRunArgs entry so the
    // framework actually launches them on each node. Without this, the kernels
    // are wired up but never start, and outputs stay at the initial value (0).
    params.kernel_run_args = {
        {.kernel = PRODUCER, .runtime_arg_values = {{.node = node, .args = {}}}},
        {.kernel = CONSUMER, .runtime_arg_values = {{.node = node, .args = {}}}},
    };
    for (uint32_t i = 0; i < 4; ++i) {
        params.tensor_args.insert({m2::TensorParamName{SRC_NAMES[i]}, std::cref(in_tensors[i])});
        params.tensor_args.insert({m2::TensorParamName{DST_NAMES[i]}, std::cref(out_tensors[i])});
    }
    m2::SetProgramRunArgs(program, params);

    std::vector<std::vector<uint32_t>> inputs(4);
    for (uint32_t i = 0; i < 4; ++i) {
        inputs[i] = tt::test_utils::generate_uniform_random_vector<uint32_t>(
            0, 100, num_entries * entry_size / sizeof(uint32_t));
        detail::WriteToBuffer(*in_tensors[i].mesh_buffer().get_reference_buffer(), inputs[i]);
        m2_writeshard_barrier_uint32(device, in_tensors[i], inputs[i]);
    }

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    for (uint32_t i = 0; i < 4; ++i) {
        std::vector<uint32_t> output;
        detail::ReadFromBuffer(*out_tensors[i].mesh_buffer().get_reference_buffer(), output);
        EXPECT_EQ(inputs[i], output) << "M2 sequential 4xDFB[" << i << "] output mismatch";
    }
}

// multi-core tests
TEST_P(DFBImplicitSyncParamFixture_2_0, MultiCoreDMTest2Core_1Sx1S_2_0) {
    run_single_dfb_multicore_2_0(
        this->devices_.at(0), 1, 1, m2::DFBAccessPattern::STRIDED, m2::DFBAccessPattern::STRIDED, GetParam());
}
TEST_P(DFBImplicitSyncParamFixture_2_0, MultiCoreDMTest2Core_2Sx2S_2_0) {
    run_single_dfb_multicore_2_0(
        this->devices_.at(0), 2, 2, m2::DFBAccessPattern::STRIDED, m2::DFBAccessPattern::STRIDED, GetParam());
}
TEST_P(DFBImplicitSyncParamFixture_2_0, MultiCoreDMTest2Core_1Sx4A_2_0) {
    run_single_dfb_multicore_2_0(
        this->devices_.at(0), 1, 4, m2::DFBAccessPattern::STRIDED, m2::DFBAccessPattern::ALL, GetParam());
}

// concurrent / sequential multi-DFB tests
TEST_P(DFBImplicitSyncParamFixture_2_0, DMTest3xDFB_1Sx1S_2_0) {
    run_concurrent_dfbs_program_2_0(
        this->devices_.at(0),
        /*num_dfbs=*/3,
        /*entry_size=*/1024,
        /*entries_per_dfb=*/16,
        GetParam());
}

TEST_P(DFBImplicitSyncParamFixture_2_0, DMTest4xDFB_3Sx3S_2_0) {
    // 4 DFBs × 3P+3C STRIDED — stresses TC allocator across 6 DM threads.
    std::array<M2SeqDFBSpec, 4> dfbs{
        M2SeqDFBSpec{m2::DFBAccessPattern::STRIDED},
        M2SeqDFBSpec{m2::DFBAccessPattern::STRIDED},
        M2SeqDFBSpec{m2::DFBAccessPattern::STRIDED},
        M2SeqDFBSpec{m2::DFBAccessPattern::STRIDED}};
    run_sequential_4_dfbs_2_0(this->devices_.at(0), dfbs, /*num_producers=*/3, /*num_consumers=*/3, GetParam());
}

TEST_P(DFBImplicitSyncParamFixture_2_0, DMTest4xDFB_Mixed_2_0) {
    // 2× STRIDED + 2× ALL — exercises mixed plain/remapper TCs in one program.
    if (GetParam()) {
        // Legacy parity: DM→DM ALL with implicit sync deadlocks (no DM↔DM remapper).
        GTEST_SKIP() << "DM→DM ALL with implicit_sync not supported (legacy parity)";
    }
    std::array<M2SeqDFBSpec, 4> dfbs{
        M2SeqDFBSpec{m2::DFBAccessPattern::STRIDED},
        M2SeqDFBSpec{m2::DFBAccessPattern::STRIDED},
        M2SeqDFBSpec{m2::DFBAccessPattern::ALL},
        M2SeqDFBSpec{m2::DFBAccessPattern::ALL}};
    run_sequential_4_dfbs_2_0(this->devices_.at(0), dfbs, /*num_producers=*/3, /*num_consumers=*/3, GetParam());
}

TEST_P(DFBImplicitSyncParamFixture_2_0, TensixDMTest4xDFB_1Sx1S_2_0) {
    auto& mesh_device = this->devices_.at(0);
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only";
    }
    const bool implicit_sync = GetParam();
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t num_dfbs = 4;
    constexpr uint32_t entry_size = 1024;
    constexpr uint32_t num_entries = 16;
    constexpr uint32_t total_bytes = num_entries * entry_size;
    const m2::NodeCoord node{0, 0};

    constexpr std::array<const char*, 4> DFB_NAMES{"buf_0", "buf_1", "buf_2", "buf_3"};
    constexpr std::array<const char*, 4> DST_NAMES{"dst_0", "dst_1", "dst_2", "dst_3"};
    constexpr std::array<const char*, 4> CONSUMER_NAMES{"consumer_0", "consumer_1", "consumer_2", "consumer_3"};
    const m2::KernelSpecName PRODUCER{"producer"};

    const auto dram_spec = make_flat_dram_tensor_spec(entry_size, num_entries, DataType::UINT32);

    std::vector<MeshTensor> out_tensors;
    out_tensors.reserve(4);
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        out_tensors.push_back(MeshTensor::allocate_on_device(*mesh_device, dram_spec, TensorTopology{}));
    }

    std::vector<m2::DataflowBufferSpec> dfbs;
    dfbs.reserve(num_dfbs);
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        dfbs.push_back({
            .unique_id = m2::DFBSpecName{DFB_NAMES[i]},
            .entry_size = entry_size,
            .num_entries = num_entries,
            .data_format_metadata = tt::DataFormat::Float16_b,
        });
    }

    // Tensix sequential producer: 1 thread, hardcoded 4 DFB bindings. No
    // tensor_bindings — TRISC compute kernels can't include tensor_accessor.h
    // transitively. Host pre-fills each ring's L1 region via uniform_alloc_addr.
    auto producer = make_compute_kernel(
        PRODUCER, "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_seq_producer_quad_2_0.cpp", /*num_threads=*/1);
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        producer.dfb_bindings.push_back(
            {.dfb_spec_name = m2::DFBSpecName{DFB_NAMES[i]},
             .accessor_name = DFB_NAMES[i],
             .endpoint_type = m2::DFBEndpointType::PRODUCER,
             .access_pattern = m2::DFBAccessPattern::STRIDED});
    }
    producer.compile_time_args = {{"num_entries_per_producer", num_entries}};

    // 4 independent DM consumer kernels; each reuses dfb_consumer_2_0.cpp.
    std::vector<m2::KernelSpec> consumers;
    consumers.reserve(num_dfbs);
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        auto c = make_dm_kernel(
            m2::KernelSpecName{CONSUMER_NAMES[i]},
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer_2_0.cpp");
        c.dfb_bindings = {
            {.dfb_spec_name = m2::DFBSpecName{DFB_NAMES[i]},
             .accessor_name = "in",
             .endpoint_type = m2::DFBEndpointType::CONSUMER,
             .access_pattern = m2::DFBAccessPattern::STRIDED}};
        c.tensor_bindings = {
            {.tensor_parameter_name = m2::TensorParamName{DST_NAMES[i]}, .accessor_name = "dst_tensor"}};
        c.compile_time_args = {
            {"num_entries_per_consumer", num_entries},
            {"blocked_consumer", 0u},
            {"implicit_sync", implicit_sync ? 1u : 0u}};
        c.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};
        // All-pass: DFB .disable_implicit_sync = !implicit_sync. Producer is Tensix (no DM
        // side); disable on the DM consumer endpoint (post-#45160).
        maybe_disable_implicit_sync(c, implicit_sync, m2::DFBSpecName{DFB_NAMES[i]});
        consumers.push_back(c);
    }

    std::vector<m2::TensorParameter> tensor_parameters;
    tensor_parameters.reserve(num_dfbs);
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        tensor_parameters.push_back({.unique_id = m2::TensorParamName{DST_NAMES[i]}, .spec = dram_spec});
    }

    std::vector<m2::KernelSpec> all_kernels;
    all_kernels.push_back(producer);
    for (const auto& c : consumers) {
        all_kernels.push_back(c);
    }

    std::vector<m2::KernelSpecName> wu_kernel_names{PRODUCER};
    for (const auto* n : CONSUMER_NAMES) {
        wu_kernel_names.emplace_back(n);
    }

    m2::ProgramSpec spec{
        .name = "tensix_dm_4dfb_2_0",
        .kernels = all_kernels,
        .dataflow_buffers = dfbs,
        .tensor_parameters = tensor_parameters,
        .work_units = {m2::WorkUnitSpec{
            .name = "wu",
            .kernels = wu_kernel_names,
            .target_nodes = node,
        }},
    };

    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);

    m2::ProgramRunArgs params;
    // Producer has no runtime args but still needs a KernelRunArgs entry to be
    // launched. Without it, the kernel is wired up but never runs.
    params.kernel_run_args.push_back({.kernel = PRODUCER, .runtime_arg_values = {{.node = node, .args = {}}}});
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        params.kernel_run_args.push_back(
            {.kernel = m2::KernelSpecName{CONSUMER_NAMES[i]},
             .runtime_arg_values = {
                 {.node = node, .args = {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}}}});
    }
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        params.tensor_args.insert({m2::TensorParamName{DST_NAMES[i]}, std::cref(out_tensors[i])});
    }
    m2::SetProgramRunArgs(program, params);

    // Pre-fill each DFB's L1 ring directly via uniform_alloc_addr after manual
    // finalize+allocate. No borrowed_from / ring tensor needed; the compute
    // kernel is TRISC-only and can't carry tensor bindings.
    detail::CompileProgram(device, program);
    program.impl().finalize_dataflow_buffer_configs();
    program.impl().allocate_dataflow_buffers(device);

    std::vector<std::vector<uint32_t>> inputs(num_dfbs);
    for (uint32_t i = 0; i < num_dfbs; ++i) {
        const uint32_t dfb_l1_addr =
            program.impl().get_dataflow_buffer(program.impl().get_dfb_handle(DFB_NAMES[i]))->uniform_alloc_addr();
        inputs[i] = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, total_bytes / sizeof(uint32_t));
        detail::WriteToDeviceL1(device, CoreCoord(0, 0), dfb_l1_addr, inputs[i]);
    }

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    for (uint32_t i = 0; i < num_dfbs; ++i) {
        std::vector<uint32_t> output;
        detail::ReadFromBuffer(*out_tensors[i].mesh_buffer().get_reference_buffer(), output);
        EXPECT_EQ(inputs[i], output) << "M2 concurrent Tensix→DM 4xDFB[" << i << "] mismatch";
    }
}

// homogeneous-grid multi-core group test
TEST_F(MeshDeviceFixture, MultiCoreDFB_HomogeneousGrid_SingleGroup_2_0) {
    auto& mesh_device = this->devices_.at(0);
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only (Gen2Config)";
    }
    IDevice* device = mesh_device->get_devices()[0];
    CoreCoord grid = device->compute_with_storage_grid_size();
    if (grid.x < 2 || grid.y < 2) {
        GTEST_SKIP() << "Homogeneous-grid test requires >= 2x2 Tensix grid";
    }

    constexpr uint32_t entry_size = 512;
    constexpr uint32_t num_entries = 8;
    const m2::NodeRange grid_2x2{m2::NodeCoord{0, 0}, m2::NodeCoord{1, 1}};  // 4 cores

    const m2::DFBSpecName DFB{"dfb"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    // Each core owns num_entries slots → 4 cores × num_entries pages total.
    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, 4 * num_entries, DataType::UINT32);
    auto in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
    auto out_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});

    m2::DataflowBufferSpec dfb_spec{
        .unique_id = DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    auto producer = make_dm_dfb_producer(PRODUCER, DFB, IN_TENSOR, num_entries, /*implicit_sync=*/false);

    auto consumer = make_dm_dfb_consumer(
        CONSUMER, DFB, OUT_TENSOR, num_entries, /*blocked_consumer=*/false, /*implicit_sync=*/false);

    // All-pass disabled dfb implicit sync (now per-DM-endpoint, post-#45160). Set before the
    // ProgramSpec copies these kernels by value.
    disable_implicit_sync_for(producer, DFB);
    disable_implicit_sync_for(consumer, DFB);

    m2::ProgramSpec spec{
        .name = "homogeneous_grid_2_0",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb_spec},
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {m2::WorkUnitSpec{
            .name = "wu",
            .kernels = {PRODUCER, CONSUMER},
            .target_nodes = grid_2x2,
        }},
    };

    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);
    program.impl().finalize_dataflow_buffer_configs();

    // Identical HW config across the 4 cores → must collapse into 1 DfbGroup.
    CoreCoord first_core{0, 0};
    auto dfbs = program.impl().dataflow_buffers_on_core(first_core);
    ASSERT_EQ(dfbs.size(), 1u) << "Expected exactly 1 DFB on core";
    const auto& dfb = dfbs[0];
    ASSERT_EQ(dfb->groups.size(), 1u) << "Homogeneous 2x2 grid must collapse to 1 DfbGroup";
    EXPECT_EQ(dfb->groups[0].l1_by_core.size(), 4u) << "Single DfbGroup must cover all 4 cores";

    // All 4 cores in the 2x2 grid should appear in the group's l1_by_core map.
    std::set<CoreCoord> accounted_cores;
    for (const auto& [c, _] : dfb->groups[0].l1_by_core) {
        accounted_cores.insert(c);
    }
    for (uint32_t x = 0; x <= 1; ++x) {
        for (uint32_t y = 0; y <= 1; ++y) {
            EXPECT_EQ(accounted_cores.count(CoreCoord(x, y)), 1u)
                << "Core (" << x << "," << y << ") missing from DfbGroup";
        }
    }
}

}  // namespace tt::tt_metal
