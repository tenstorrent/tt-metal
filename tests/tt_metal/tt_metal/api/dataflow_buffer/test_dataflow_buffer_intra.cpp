// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Intra-scope (self-loop) DFB tests.

#include "dfb_test_common.hpp"

namespace tt::tt_metal {

// legacy intra-Tensix self-loop harness + test
static void run_intra_tensix_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t entry_size,
    uint32_t num_entries,
    uint32_t num_threads) {
    IDevice* device = mesh_device->get_devices()[0];

    experimental::dfb::DataflowBufferConfig dfb_config{
        .entry_size = entry_size,
        .num_entries = num_entries,
        .num_producers = num_threads,
        .pap = dfb::AccessPattern::STRIDED,
        .num_consumers = num_threads,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false,
        .tensix_scope = experimental::dfb::TensixScope::INTRA};

    CoreCoord logical_core = CoreCoord(0, 0);
    CoreRangeSet core_range_set(CoreRange(logical_core, logical_core));

    const uint32_t words_per_entry = entry_size / sizeof(uint32_t);

    TT_FATAL(
        num_entries % num_threads == 0,
        "num_entries ({}) must be divisible by num_threads ({}) for intra-tensix block partitioning",
        num_entries,
        num_threads);
    const uint32_t entries_per_neo = num_entries / num_threads;

    const experimental::DFBSpecName INTRA_DFB{"intra_dfb"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    experimental::DataflowBufferSpec intra_dfb_spec{
        .unique_id = INTRA_DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = dfb_config.data_format,
    };

    // Self-looped: register both PRODUCER and CONSUMER bindings on the same kernel.
    // The kernel only references dfb::out; both bindings resolve to the same DFB.
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_intra.cpp",
        .num_threads = num_threads,
        .dfb_bindings =
            {
                {
                    .dfb_spec_name = INTRA_DFB,
                    .accessor_name = "out",
                    .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                    .access_pattern = experimental::DFBAccessPattern::STRIDED,
                },
                {
                    .dfb_spec_name = INTRA_DFB,
                    .accessor_name = "in",
                    .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                    .access_pattern = experimental::DFBAccessPattern::STRIDED,
                },
            },
        .compile_time_args =
            {
                {"entries_per_neo", entries_per_neo},
                {"words_per_entry", words_per_entry},
            },
        .hw_config = experimental::ComputeGen2Config{},
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {COMPUTE},
        .target_nodes = core_range_set,
    };

    experimental::ProgramSpec spec{
        .name = "intra_tensix_dfb",
        .kernels = {compute_spec},
        .dataflow_buffers = {intra_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs run_params;
    run_params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE}};
    experimental::SetProgramRunArgs(program, run_params);

    const uint32_t total_size = num_entries * entry_size;
    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, total_size / sizeof(uint32_t));

    const uint32_t dfb_l1_addr = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));

    detail::WriteToDeviceL1(device, logical_core, dfb_l1_addr, input);

    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    // Packer increments each word by 1, then unpacker increments it by 1 → +2 per word.
    // This holds for every Neo's ring independently, so the entire L1 region is input + 2.
    std::vector<uint32_t> expected(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        expected[i] = input[i] + 2;
    }

    std::vector<uint32_t> l1_data;
    detail::ReadFromDeviceL1(device, logical_core, dfb_l1_addr, total_size, l1_data);
    EXPECT_EQ(expected, l1_data) << "Intra-tensix DFB L1 mismatch";
}

TEST_F(MeshDeviceFixture, TensixIntraTest1xDFB4Sx4S) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping intra-tensix DFB test for WH/BH until DFB is backported";
    }
    run_intra_tensix_dfb_program(this->devices_.at(0), /*entry_size=*/1024, /*num_entries=*/16, /*num_threads=*/4);
}

// Metal 2.0 intra: DM->Trisc self-loop double-relu
TEST_F(MeshDeviceFixture, C2_2_0_DMTriscSelfLoopDM_DoubleRelu) {
    auto& mesh_device = this->devices_.at(0);
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "C2 INTRA-scope DFB self-loop requires Quasar";
    }

    IDevice* device = mesh_device->get_devices()[0];
    constexpr uint32_t entry_size = 2 * 32 * 32;  // bf16 tile = 2048 B
    constexpr uint32_t num_entries = 4;
    const m2::NodeCoord node{0, 0};

    const m2::DFBSpecName DFB_IN{"dfb_in"};
    const m2::DFBSpecName DFB_SELF{"dfb_self"};
    const m2::DFBSpecName DFB_OUT{"dfb_out"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::KernelSpecName COMPUTE{"compute"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, num_entries, DataType::BFLOAT16);
    auto in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});
    auto out_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});

    m2::DataflowBufferSpec dfb_in{
        .unique_id = DFB_IN,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    m2::DataflowBufferSpec dfb_self{
        .unique_id = DFB_SELF,
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

    auto producer = make_dm_dfb_producer(PRODUCER, DFB_IN, IN_TENSOR, num_entries, /*implicit_sync=*/false);

    auto compute = make_compute_kernel(COMPUTE, "tests/tt_metal/tt_metal/test_kernels/compute/dfb_c2_pipeline_2_0.cpp");
    compute.dfb_bindings = {
        {.dfb_spec_name = DFB_IN,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
        {.dfb_spec_name = DFB_SELF,
         .accessor_name = "self",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
        {.dfb_spec_name = DFB_SELF,
         .accessor_name = "self",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
        {.dfb_spec_name = DFB_OUT,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
    };
    compute.compile_time_args = {{"per_core_tile_cnt", num_entries}};

    auto consumer = make_dm_dfb_consumer(
        CONSUMER, DFB_OUT, OUT_TENSOR, num_entries, /*blocked_consumer=*/false, /*implicit_sync=*/false);

    // All-pass disabled dfb_in/dfb_self/dfb_out implicit sync; dfb_self is Tensix-only
    // (compute self-loop, no DM endpoint). Disable the two DM-side endpoints (post-#45160).
    disable_implicit_sync_for(producer, DFB_IN);
    disable_implicit_sync_for(consumer, DFB_OUT);

    m2::WorkUnitSpec wu{.name = "wu", .kernels = {PRODUCER, CONSUMER, COMPUTE}, .target_nodes = node};

    m2::ProgramSpec spec{
        .name = "c2_2_0",
        .kernels = {producer, consumer, compute},
        .dataflow_buffers = {dfb_in, dfb_self, dfb_out},
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
            .runtime_arg_values =
                m2::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{
            .kernel = CONSUMER,
            .runtime_arg_values =
                m2::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}}),
        },
        m2::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {
        {IN_TENSOR, std::cref(in_tensor)},
        {OUT_TENSOR, std::cref(out_tensor)},
    };
    m2::SetProgramRunArgs(program, params);

    // Positive bf16 inputs → double-relu identity.
    const uint32_t total_bytes = entry_size * num_entries;
    auto input = create_random_vector_of_bfloat16(total_bytes, 1.0f, 0xC2C2);
    detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), input);
    m2_writeshard_barrier_uint32(device, in_tensor, input);

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(*out_tensor.mesh_buffer().get_reference_buffer(), output);
    EXPECT_TRUE(packed_uint32_t_vector_comparison(output, input, [](float a, float b) {
        return std::abs(a - b) < 0.01f;
    })) << "M2 C2 double-relu identity mismatch";
}

// Metal 2.0 intra-Tensix self-loop harness + test
static void run_intra_tensix_dfb_program_2_0(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t entry_size, uint32_t num_threads) {
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 INTRA test is Quasar-only";
    }
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t num_entries = 16;  // matches legacy
    TT_FATAL(num_entries % num_threads == 0, "num_entries must be divisible by num_threads");
    const uint32_t entries_per_neo = num_entries / num_threads;
    const uint32_t words_per_entry = entry_size / sizeof(uint32_t);
    const uint32_t total_bytes = num_entries * entry_size;

    const m2::DFBSpecName DFB{"intra_dfb"};
    const m2::KernelSpecName COMPUTE{"compute"};

    m2::DataflowBufferSpec dfb_spec{
        .unique_id = DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // Compute kernel binds the same DFB as both PRODUCER ("out") and CONSUMER
    // ("in"). M2 infers INTRA scope. Names MUST differ: using the same name for
    // both endpoints causes M2 to only wire one Neo's slice, leaving the others'
    // L1 untouched. No tensor_bindings (TRISC compute kernels can't include
    // tensor_accessor.h transitively). Matches the working
    // run_intra_tensix_dfb_program pattern in main's test_dataflow_buffer.cpp.
    auto compute =
        make_compute_kernel(COMPUTE, "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_intra_2_0.cpp", num_threads);
    compute.dfb_bindings = {
        {.dfb_spec_name = DFB,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
        {.dfb_spec_name = DFB,
         .accessor_name = "in",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
    };
    compute.compile_time_args = {{"entries_per_neo", entries_per_neo}, {"words_per_entry", words_per_entry}};

    // target_nodes MUST be a NodeRangeSet (not a bare NodeCoord) to match the
    // working legacy M2 helper on main (run_intra_tensix_dfb_program). With a
    // bare NodeCoord the framework takes a different scheduling path that
    // exposes a PACK/UNPACK write race for some Neos' L1 slices.
    const m2::NodeRangeSet node_set{m2::NodeRange{m2::NodeCoord{0, 0}, m2::NodeCoord{0, 0}}};
    m2::ProgramSpec spec{
        .name = "intra_tensix_2_0",
        .kernels = {compute},
        .dataflow_buffers = {dfb_spec},
        .tensor_parameters = {},
        .work_units = {m2::WorkUnitSpec{.name = "main", .kernels = {COMPUTE}, .target_nodes = node_set}},
    };

    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);

    // Kernel has no runtime args; pass kernel_spec_name only (matches legacy
    // pattern in main's run_intra_tensix_dfb_program).
    m2::ProgramRunArgs params;
    params.kernel_run_args = {{.kernel = COMPUTE}};
    m2::SetProgramRunArgs(program, params);

    // DFB is the first L1 allocation in the program → lands at the base L1
    // allocator address. Same trick the legacy intra test uses.
    const uint32_t dfb_l1_addr = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));

    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, total_bytes / sizeof(uint32_t));
    detail::WriteToDeviceL1(device, CoreCoord(0, 0), dfb_l1_addr, input);

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    std::vector<uint32_t> expected(input.size());
    std::transform(input.begin(), input.end(), expected.begin(), [](uint32_t v) { return v + 2; });
    std::vector<uint32_t> output;
    detail::ReadFromDeviceL1(device, CoreCoord(0, 0), dfb_l1_addr, total_bytes, output);
    EXPECT_EQ(expected, output) << "M2 intra-tensix DFB +2 per word mismatch (num_threads=" << num_threads << ")";
}

TEST_F(MeshDeviceFixture, TensixIntraTest1xDFB1Sx1S_2_0) {
    run_intra_tensix_dfb_program_2_0(this->devices_.at(0), /*entry_size=*/1024, /*num_threads=*/1);
}

// Metal 2.0 intra + remapper coexistence
TEST_F(MeshDeviceFixture, TensixIntraAndRemapperTest_4Neo_DM1Sx4B_2_0) {
    auto& mesh_device = this->devices_.at(0);
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only (Gen2Config)";
    }
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t entry_size = 1024;
    constexpr uint32_t num_entries = 16;
    constexpr uint32_t num_neos = 4;
    constexpr uint32_t entries_per_neo = num_entries / num_neos;  // = 4
    const m2::NodeCoord node{0, 0};

    const m2::DFBSpecName DFB_REMAPPER{"dfb_remapper"};
    const m2::DFBSpecName DFB_INTRA{"dfb_intra"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName COMPUTE{"compute"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};

    // dfb_remapper: DM->Tensix 1S × 4B ALL, implicit_sync=true.
    m2::DataflowBufferSpec dfb_remapper{
        .unique_id = DFB_REMAPPER,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
        // implicit sync default = enabled; do not disable.
    };
    // dfb_intra: PACK->UNPACK self-loop, 4×4 STRIDED, implicit sync off (INTRA requirement).
    m2::DataflowBufferSpec dfb_intra_spec{
        .unique_id = DFB_INTRA,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // DRAM input tensor for the DM producer.
    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, num_entries, DataType::UINT32);
    auto in_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});

    // DM producer: implicit-sync path feeds the remapper ring.
    auto producer = make_dm_dfb_producer(PRODUCER, DFB_REMAPPER, IN_TENSOR, num_entries, /*implicit_sync=*/true);

    // Compute kernel: 4 Neo threads. Binds remapper as CONSUMER (ALL) and intra
    // DFB as PRODUCER+CONSUMER (PACK→UNPACK self-loop, infers INTRA scope).
    auto compute = make_compute_kernel(
        COMPUTE, "tests/tt_metal/tt_metal/test_kernels/compute/dfb_intra_and_consume_all_2_0.cpp", num_neos);
    compute.dfb_bindings = {
        {.dfb_spec_name = DFB_REMAPPER,
         .accessor_name = "consume",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::ALL},
        {.dfb_spec_name = DFB_INTRA,
         .accessor_name = "intra",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
        {.dfb_spec_name = DFB_INTRA,
         .accessor_name = "intra",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
    };
    compute.compile_time_args = {
        {"num_entries_consumer", num_entries},
        {"entries_per_neo", entries_per_neo},
        {"words_per_entry", entry_size / sizeof(uint32_t)},
    };

    m2::ProgramSpec spec{
        .name = "intra_and_remapper_2_0",
        .kernels = {producer, compute},
        .dataflow_buffers = {dfb_remapper, dfb_intra_spec},
        .tensor_parameters =
            {
                {.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
            },
        .work_units = {m2::WorkUnitSpec{
            .name = "wu",
            .kernels = {PRODUCER, COMPUTE},
            .target_nodes = node,
        }},
    };

    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = PRODUCER,
         .runtime_arg_values =
             m2::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}})},
    };
    params.tensor_args = {{IN_TENSOR, std::cref(in_tensor)}};
    m2::SetProgramRunArgs(program, params);

    // Fill DRAM input; DM NOC-reads this into the remapper ring's L1.
    auto input_remapper =
        tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, num_entries * entry_size / sizeof(uint32_t));
    detail::WriteToBuffer(*in_tensor.mesh_buffer().get_reference_buffer(), input_remapper);
    m2_writeshard_barrier_uint32(device, in_tensor, input_remapper);

    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    // Verify remapper ring L1: DM wrote input_remapper; Tensix consumed credits
    // (ALL pattern) but did not overwrite the ring's data.
    const uint32_t remapper_l1_addr =
        program.impl().get_dataflow_buffer(program.impl().get_dfb_handle(*DFB_REMAPPER))->uniform_alloc_addr();
    std::vector<uint32_t> l1_remapper;
    detail::ReadFromDeviceL1(device, CoreCoord(0, 0), remapper_l1_addr, num_entries * entry_size, l1_remapper);
    EXPECT_EQ(input_remapper, l1_remapper) << "M2 DM->Tensix strided x ALL remapper ring L1 mismatch";
}

}  // namespace tt::tt_metal
