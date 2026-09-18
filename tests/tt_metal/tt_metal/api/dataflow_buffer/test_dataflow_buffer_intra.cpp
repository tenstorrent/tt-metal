// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Intra-scope (self-loop) DFB tests.

#include "dfb_test_common.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "impl/program/program_impl.hpp"

#include <array>
#include <map>
#include <string_view>

namespace tt::tt_metal {

// legacy intra-Tensix self-loop harness + test
static void run_intra_tensix_dfb_program(
    distributed::MeshDevice& mesh_device, uint32_t entry_size, uint32_t num_entries, uint32_t num_threads) {
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

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    experimental::ProgramRunArgs run_params;
    run_params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE}};
    experimental::SetProgramRunArgs(program, run_params);

    const uint32_t total_size = num_entries * entry_size;
    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, total_size / sizeof(uint32_t));

    const uint32_t dfb_l1_addr =
        static_cast<uint32_t>(mesh_device.allocator()->get_base_allocator_addr(HalMemType::L1));

    slow_dispatch::WriteToL1(mesh_device, logical_core, dfb_l1_addr, input);

    LaunchProgram(mesh_device, std::move(program), /*wait_until_cores_done=*/true);

    // Packer increments each word by 1, then unpacker increments it by 1 → +2 per word.
    // This holds for every Neo's ring independently, so the entire L1 region is input + 2.
    std::vector<uint32_t> expected(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        expected[i] = input[i] + 2;
    }

    std::vector<uint32_t> l1_data;
    slow_dispatch::ReadFromL1(mesh_device, logical_core, dfb_l1_addr, total_size, l1_data);
    EXPECT_EQ(expected, l1_data) << "Intra-tensix DFB L1 mismatch";
}

TEST_F(UnitMeshFixture, TensixIntraTest1xDFB4Sx4S) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping intra-tensix DFB test for WH/BH until DFB is backported";
    }
    run_intra_tensix_dfb_program(this->device(), /*entry_size=*/1024, /*num_entries=*/16, /*num_threads=*/4);
}

// Metal 2.0 intra: DM->Trisc self-loop double-relu
TEST_F(UnitMeshFixture, C2_2_0_DMTriscSelfLoopDM_DoubleRelu) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "C2 INTRA-scope DFB self-loop requires Quasar";
    }

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
    auto in_tensor = MeshTensor::allocate_on_device(this->device(), tensor_spec);
    auto out_tensor = MeshTensor::allocate_on_device(this->device(), tensor_spec);

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

    Program program = m2::MakeProgramFromSpec(this->device(), spec);

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
    slow_dispatch::WriteToBuffer(in_tensor.mesh_buffer(), input);
    m2_writeshard_barrier_uint32(this->device(), in_tensor, input);

    // Finalize early so host can assert INTRA remapper assignment before launch.
    program.impl().finalize_dataflow_buffer_configs();
    auto self_dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle(*DFB_SELF));
    ASSERT_NE(self_dfb, nullptr);
    EXPECT_EQ(self_dfb->remapper_programmer, experimental::dfb::detail::RemapperProgrammer::TENSIX_PACKER);
    EXPECT_EQ(self_dfb->dm1_remapper_slot_count(), 0u);
    ASSERT_FALSE(self_dfb->groups.empty());
    const auto& self_rc = self_dfb->groups[0].hw_risc_configs[0];
    const uint8_t self_tc = ::dfb::get_counter_id(self_rc.config.packed_tile_counter[0]);
    EXPECT_GE(self_tc, ::dfb::TC_TENSIX_POOL_START);
    EXPECT_GE(self_rc.config.intra_shadow_tc_id, ::dfb::TC_TENSIX_POOL_START);
    EXPECT_NE(self_rc.config.intra_shadow_tc_id, self_tc);
    EXPECT_GE(self_rc.config.remapper_pair_index, ::dfb::REMAPPER_ONE_TO_ONE_PAIR_START);

    LaunchProgram(this->device(), std::move(program), /*wait_until_cores_done=*/true);

    std::vector<uint32_t> output;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), output);
    EXPECT_TRUE(packed_uint32_t_vector_comparison(output, input, [](float a, float b) {
        return std::abs(a - b) < 0.01f;
    })) << "M2 C2 double-relu identity mismatch";

    // Capacity is shared (all DFBs use num_entries=4), so only assert the INTRA ClientL/shadow
    // programming — not overlay isolation by capacity fingerprint.
    const auto live = read_live_tcs(this->device(), CoreCoord(0, 0), /*neo_id=*/0);
    ASSERT_GT(live.capacity.size(), self_rc.config.intra_shadow_tc_id);
    EXPECT_EQ(live.capacity[self_tc], num_entries);
    EXPECT_EQ(live.capacity[self_rc.config.intra_shadow_tc_id], 0u);
}

// Metal 2.0 intra-Tensix self-loop harness + test
static void run_intra_tensix_dfb_program_2_0(
    distributed::MeshDevice& mesh_device, uint32_t entry_size, uint32_t num_threads) {
    if (mesh_device.arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 INTRA test is Quasar-only";
    }

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

    Program program = m2::MakeProgramFromSpec(mesh_device, spec);

    // Kernel has no runtime args; pass kernel_spec_name only (matches legacy
    // pattern in main's run_intra_tensix_dfb_program).
    m2::ProgramRunArgs params;
    params.kernel_run_args = {{.kernel = COMPUTE}};
    m2::SetProgramRunArgs(program, params);

    // DFB is the first L1 allocation in the program → lands at the base L1
    // allocator address. Same trick the legacy intra test uses.
    const uint32_t dfb_l1_addr =
        static_cast<uint32_t>(mesh_device.allocator()->get_base_allocator_addr(HalMemType::L1));

    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, total_bytes / sizeof(uint32_t));
    slow_dispatch::WriteToL1(mesh_device, CoreCoord(0, 0), dfb_l1_addr, input);

    LaunchProgram(mesh_device, std::move(program), /*wait_until_cores_done=*/true);

    std::vector<uint32_t> expected(input.size());
    std::transform(input.begin(), input.end(), expected.begin(), [](uint32_t v) { return v + 2; });
    std::vector<uint32_t> output;
    slow_dispatch::ReadFromL1(mesh_device, CoreCoord(0, 0), dfb_l1_addr, total_bytes, output);
    EXPECT_EQ(expected, output) << "M2 intra-tensix DFB +2 per word mismatch (num_threads=" << num_threads << ")";
}

TEST_F(UnitMeshFixture, TensixIntraTest1xDFB1Sx1S_2_0) {
    run_intra_tensix_dfb_program_2_0(this->device(), /*entry_size=*/1024, /*num_threads=*/1);
}

// Metal 2.0 intra + remapper coexistence
TEST_F(UnitMeshFixture, TensixIntraAndRemapperTest_4Neo_DM1Sx4B_2_0) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only (Gen2Config)";
    }

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
    auto in_tensor = MeshTensor::allocate_on_device(this->device(), tensor_spec);

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

    Program program = m2::MakeProgramFromSpec(this->device(), spec);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = PRODUCER,
         .runtime_arg_values =
             m2::MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", num_entries}})},
    };
    params.tensor_args = {{IN_TENSOR, std::cref(in_tensor)}};
    m2::SetProgramRunArgs(program, params);

    // Finalize before launch so the counters each DFB owns are known while the pre-launch snapshot is taken.
    program.impl().finalize_dataflow_buffer_configs();

    // Remapper (DM1) vs packer (INTRA) pool split + live TC integrity.
    auto remapper_dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle(*DFB_REMAPPER));
    auto intra_dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle(*DFB_INTRA));
    ASSERT_NE(remapper_dfb, nullptr);
    ASSERT_NE(intra_dfb, nullptr);
    EXPECT_EQ(remapper_dfb->remapper_programmer, experimental::dfb::detail::RemapperProgrammer::DM1);
    EXPECT_EQ(intra_dfb->remapper_programmer, experimental::dfb::detail::RemapperProgrammer::TENSIX_PACKER);
    EXPECT_EQ(intra_dfb->dm1_remapper_slot_count(), 0u);

    uint8_t packer_lo = ::dfb::NUM_REMAPPER_PAIRINGS;
    for (const auto& rc : intra_dfb->groups[0].hw_risc_configs) {
        ASSERT_TRUE(rc.is_producer);
        const uint8_t tc_id = ::dfb::get_counter_id(rc.config.packed_tile_counter[0]);
        EXPECT_GE(tc_id, ::dfb::TC_TENSIX_POOL_START);
        EXPECT_GE(rc.config.intra_shadow_tc_id, ::dfb::TC_TENSIX_POOL_START);
        EXPECT_NE(rc.config.intra_shadow_tc_id, tc_id);
        EXPECT_GE(rc.config.remapper_pair_index, ::dfb::REMAPPER_ONE_TO_ONE_PAIR_START);
        packer_lo = std::min(packer_lo, rc.config.remapper_pair_index);
    }
    for (const auto& rc : remapper_dfb->groups[0].hw_risc_configs) {
        if (!rc.is_producer) {
            continue;
        }
        EXPECT_LT(rc.config.remapper_pair_index, ::dfb::NUM_REMAPPER_ONE_TO_MANY_PAIRINGS)
            << "DM1 1:m pairs must stay in [0,16)";
        EXPECT_LT(rc.config.remapper_pair_index, packer_lo) << "DM1 pairs must stay below packer-owned top-down range";
    }

    // Record which Neo0 counter each DFB claims so a collision shows up as two owners of one tc id instead of
    // being inferred from counter values. Distinct capacities (remapper consumers hold num_entries, INTRA
    // ClientL holds entries_per_neo, shadow stays unprogrammed) then let the live readback confirm the owner
    // of every counter this program touched.
    static_assert(num_entries != entries_per_neo, "capacities must differ so a counter names its owner");
    struct TcClaim {
        uint32_t capacity;
        std::string_view owner;
    };
    std::map<uint8_t, TcClaim> claims;  // Neo0 tc id -> expected state after init
    auto claim = [&](uint8_t tc_id, uint32_t capacity, std::string_view owner) {
        const auto [it, inserted] = claims.emplace(tc_id, TcClaim{capacity, owner});
        if (!inserted) {
            EXPECT_EQ(it->second.owner, owner)
                << "Neo0 TC" << (int)tc_id << " claimed by both " << it->second.owner << " and " << owner;
        }
    };

    for (const auto& rc : remapper_dfb->groups[0].hw_risc_configs) {
        if (rc.is_producer) {
            continue;
        }
        const uint8_t tc_id = ::dfb::get_counter_id(rc.config.packed_tile_counter[0]);
        if (::dfb::get_tensix_id(rc.config.packed_tile_counter[0]) != 0) {
            continue;  // host live readback is Neo0-local
        }
        EXPECT_LT(tc_id, ::dfb::TC_TENSIX_POOL_START) << "DM->Tensix consumer must use a DM-visible TC";
        claim(tc_id, num_entries, "remapper consumer");
    }
    for (const auto& rc : intra_dfb->groups[0].hw_risc_configs) {
        if (::dfb::get_tensix_id(rc.config.packed_tile_counter[0]) != 0) {
            continue;
        }
        claim(::dfb::get_counter_id(rc.config.packed_tile_counter[0]), entries_per_neo, "INTRA ClientL");
        claim(rc.config.intra_shadow_tc_id, 0u, "INTRA shadow");
    }

    // DFB init only resets and sizes the counters named in this program's config, so counters it does not
    // claim must come out of the run bit-identical to this snapshot no matter what earlier programs left in
    // them. Comparing against the snapshot is what makes the isolation check exact instead of a guess about
    // which values look like INTRA state.
    const auto before = read_live_tcs(this->device(), CoreCoord(0, 0), /*neo_id=*/0);
    ASSERT_GE(before.capacity.size(), ::dfb::NUM_TILE_COUNTERS_PER_TENSIX);

    // Fill DRAM input; DM NOC-reads this into the remapper ring's L1.
    auto input_remapper =
        tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, num_entries * entry_size / sizeof(uint32_t));
    slow_dispatch::WriteToBuffer(in_tensor.mesh_buffer(), input_remapper);
    m2_writeshard_barrier_uint32(this->device(), in_tensor, input_remapper);

    // Ring L1 address is fixed by finalize, so read it while the program is still owned here: the launch moves
    // the program into the workload, which invalidates program (and remapper_dfb) once the launch returns.
    const uint32_t remapper_l1_addr = remapper_dfb->uniform_alloc_addr();

    LaunchProgram(this->device(), std::move(program), /*wait_until_cores_done=*/true);

    // Verify remapper ring L1: DM wrote input_remapper; Tensix consumed credits
    // (ALL pattern) but did not overwrite the ring's data.
    std::vector<uint32_t> l1_remapper;
    slow_dispatch::ReadFromL1(this->device(), CoreCoord(0, 0), remapper_l1_addr, num_entries * entry_size, l1_remapper);
    EXPECT_EQ(input_remapper, l1_remapper) << "M2 DM->Tensix strided x ALL remapper ring L1 mismatch";

    // space_available is the free-space view (capacity - tiles_available), so a fully drained counter reads
    // tiles_available == 0 and space_available == capacity.
    const auto live = read_live_tcs(this->device(), CoreCoord(0, 0), /*neo_id=*/0);
    ASSERT_GE(live.capacity.size(), ::dfb::NUM_TILE_COUNTERS_PER_TENSIX);
    for (const auto& [tc_id, expected] : claims) {
        EXPECT_EQ(live.capacity[tc_id], expected.capacity) << expected.owner << " TC" << (int)tc_id << " capacity";
        EXPECT_EQ(live.tiles_available[tc_id], 0u) << expected.owner << " TC" << (int)tc_id << " not drained";
        EXPECT_EQ(live.space_available[tc_id], expected.capacity)
            << expected.owner << " TC" << (int)tc_id << " space_available after drain";
    }
    for (uint32_t tc = 0; tc < ::dfb::TC_TENSIX_POOL_START; tc++) {
        if (claims.contains(static_cast<uint8_t>(tc))) {
            continue;
        }
        EXPECT_EQ(live.capacity[tc], before.capacity[tc]) << "unclaimed DM-visible TC" << tc << " capacity changed";
        EXPECT_EQ(live.tiles_available[tc], before.tiles_available[tc])
            << "unclaimed DM-visible TC" << tc << " tiles_available changed";
        EXPECT_EQ(live.space_available[tc], before.space_available[tc])
            << "unclaimed DM-visible TC" << tc << " space_available changed";
    }
}

// T1: pure INTRA must not corrupt DM-visible overlay tile counters 0-15.
TEST_F(UnitMeshFixture, TensixIntraIsolatesOverlayTCs) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Overlay TC isolation is Quasar-only";
    }

    constexpr uint32_t entry_size = 1024;
    constexpr uint32_t num_entries = 8;
    constexpr uint32_t num_threads = 1;
    constexpr uint32_t entries_per_neo = num_entries / num_threads;
    constexpr uint32_t total_bytes = num_entries * entry_size;
    constexpr uint32_t scratch_magic = 0x000A11A5u;
    constexpr uint32_t scratch_done = 0x000D0EEu;
    constexpr uint32_t handshake_ready = 0x000BEE71u;
    constexpr uint32_t handshake_pushes_done = 0x000BEE72u;
    constexpr uint32_t settle_reads = 256;
    constexpr uint32_t num_mirrored_dm_tcs = ::dfb::NUM_TENSIX_TILE_COUNTERS_FOR_DM;  // 16
    constexpr uint32_t words_per_tc = 3;

    // Uneven pushes so a doubled credit cannot be mistaken for the next step's push. Pack issues the
    // whole series before unpack pops, so the sum must fit in the ClientL capacity.
    constexpr std::array<uint32_t, 4> steps = {1, 3, 2, 2};
    constexpr uint32_t num_steps = 4;
    static_assert(steps[0] + steps[1] + steps[2] + steps[3] == entries_per_neo);

    // Intra-tensix allocation is deterministic for the first INTRA DFB on Neo0; the kernel needs the
    // ClientL id at compile time and the host asserts the finalized config still matches.
    constexpr uint32_t client_l_tc_arg = ::dfb::TC_TENSIX_POOL_START;  // 16

    constexpr uint32_t ready_idx = 2;
    constexpr uint32_t pushes_done_idx = 3;
    constexpr uint32_t num_steps_idx = 4;
    constexpr uint32_t posted_base = 5;
    constexpr uint32_t acked_base = posted_base + num_steps;
    constexpr uint32_t baseline_base = acked_base + num_steps;
    constexpr uint32_t final_base = baseline_base + num_mirrored_dm_tcs * words_per_tc;
    constexpr uint32_t done_idx = final_base + num_mirrored_dm_tcs * words_per_tc;
    constexpr uint32_t scratch_words = done_idx + 1;

    const uint32_t dfb_l1_addr =
        static_cast<uint32_t>(this->device().allocator()->get_base_allocator_addr(HalMemType::L1));
    const uint32_t scratch_l1_addr = dfb_l1_addr + total_bytes;

    const m2::DFBSpecName DFB{"intra_dfb"};
    const m2::KernelSpecName COMPUTE{"compute"};

    m2::DataflowBufferSpec dfb_spec{
        .unique_id = DFB,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    auto compute = make_compute_kernel(
        COMPUTE, "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_intra_overlay_check.cpp", num_threads);
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
    compute.compile_time_args = {
        {"scratch_l1_address", scratch_l1_addr},
        {"scratch_magic", scratch_magic},
        {"scratch_done", scratch_done},
        {"handshake_ready", handshake_ready},
        {"handshake_pushes_done", handshake_pushes_done},
        {"settle_reads", settle_reads},
        {"client_l_tc", client_l_tc_arg},
        {"num_overlay_tcs", num_mirrored_dm_tcs},
        {"words_per_tc", words_per_tc},
        {"ready_idx", ready_idx},
        {"pushes_done_idx", pushes_done_idx},
        {"num_steps_idx", num_steps_idx},
        {"posted_base", posted_base},
        {"acked_base", acked_base},
        {"baseline_base", baseline_base},
        {"final_base", final_base},
        {"done_idx", done_idx},
        {"step0", steps[0]},
        {"step1", steps[1]},
        {"step2", steps[2]},
        {"step3", steps[3]},
    };

    const m2::NodeRangeSet node_set{m2::NodeRange{m2::NodeCoord{0, 0}, m2::NodeCoord{0, 0}}};
    m2::ProgramSpec spec{
        .name = "intra_overlay_isolation_2_0",
        .kernels = {compute},
        .dataflow_buffers = {dfb_spec},
        .tensor_parameters = {},
        .work_units = {m2::WorkUnitSpec{.name = "main", .kernels = {COMPUTE}, .target_nodes = node_set}},
    };

    Program program = m2::MakeProgramFromSpec(this->device(), spec);
    program.impl().finalize_dataflow_buffer_configs();

    auto intra_dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle(*DFB));
    ASSERT_NE(intra_dfb, nullptr);
    ASSERT_EQ(intra_dfb->remapper_programmer, experimental::dfb::detail::RemapperProgrammer::TENSIX_PACKER);
    const auto& rc = intra_dfb->groups[0].hw_risc_configs[0];
    const uint8_t client_l_tc = ::dfb::get_counter_id(rc.config.packed_tile_counter[0]);
    EXPECT_GE(client_l_tc, ::dfb::TC_TENSIX_POOL_START);
    EXPECT_GE(rc.config.intra_shadow_tc_id, ::dfb::TC_TENSIX_POOL_START);
    EXPECT_NE(rc.config.intra_shadow_tc_id, client_l_tc);
    EXPECT_GE(rc.config.remapper_pair_index, ::dfb::REMAPPER_ONE_TO_ONE_PAIR_START);
    // The kernel samples ClientL as a compile-time arg, so a reallocation must fail loudly here
    // rather than silently sampling the wrong counter.
    ASSERT_EQ(client_l_tc, client_l_tc_arg);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {{.kernel = COMPUTE}};
    m2::SetProgramRunArgs(program, params);

    // Kernel only exercises TC push/pop credits; L1 payload is untouched.
    std::vector<uint32_t> scratch_zero(scratch_words, 0);
    slow_dispatch::WriteToL1(this->device(), CoreCoord(0, 0), scratch_l1_addr, scratch_zero);

    // This program owns only Tensix-only counters (ClientL + shadow), so every DM-visible counter must survive
    // the run untouched. The kernel's own baseline is sampled after DFB init; snapshotting here additionally
    // covers init-time aliasing, and comparing exact values avoids inferring corruption from a magic capacity.
    const auto before = read_live_tcs(this->device(), CoreCoord(0, 0), /*neo_id=*/0);
    ASSERT_GE(before.capacity.size(), ::dfb::NUM_TILE_COUNTERS_PER_TENSIX);

    LaunchProgram(this->device(), std::move(program), /*wait_until_cores_done=*/true);

    std::vector<uint32_t> scratch;
    slow_dispatch::ReadFromL1(
        this->device(), CoreCoord(0, 0), scratch_l1_addr, scratch_words * sizeof(uint32_t), scratch);
    ASSERT_EQ(scratch.size(), scratch_words);
    EXPECT_EQ(scratch[0], scratch_magic) << "Pack never wrote overlay isolation scratch";
    EXPECT_EQ(scratch[done_idx], scratch_done) << "Pack never finished overlay isolation scratch";
    EXPECT_EQ(scratch[1], entries_per_neo) << "ClientL capacity mismatch";
    EXPECT_EQ(scratch[ready_idx], handshake_ready) << "Pack never published the baseline handshake";
    EXPECT_EQ(scratch[pushes_done_idx], handshake_pushes_done) << "Pack never finished the push series";
    EXPECT_EQ(scratch[num_steps_idx], num_steps) << "Kernel decoded a different step count";

    // Double-count check on ClientL: nothing is popped until every push has been issued, so
    // tiles-available after push step s is the running push total and space-available after pop
    // step s is the running pop total. A doubled credit shows up as 2x the expected value.
    uint32_t tiles_pushed = 0;
    for (uint32_t step = 0; step < num_steps; step++) {
        tiles_pushed += steps[step];
        EXPECT_EQ(scratch[posted_base + step], tiles_pushed)
            << "ClientL TC" << (uint32_t)client_l_tc << " tiles-available after push step " << step << " (pushed "
            << steps[step] << ", expected running total " << tiles_pushed << ") — possible double-counted push";
    }
    uint32_t tiles_popped = 0;
    for (uint32_t step = 0; step < num_steps; step++) {
        tiles_popped += steps[step];
        EXPECT_EQ(scratch[acked_base + step], tiles_popped)
            << "ClientL TC" << (uint32_t)client_l_tc << " space-available after pop step " << step << " (popped "
            << steps[step] << ", expected running total " << tiles_popped << ") — possible double-counted pop";
    }
    EXPECT_EQ(tiles_pushed, entries_per_neo);
    EXPECT_EQ(tiles_popped, tiles_pushed) << "Tiles popped must equal tiles pushed (no lost/double credits)";

    // Mirrored DM TCs (overlay 0-15) must be unchanged by the INTRA push/pop series. The remapper
    // routes ClientL updates to a sacrificial shadow; nothing should leak into the DM-visible pool.
    for (uint32_t tc = 0; tc < num_mirrored_dm_tcs; tc++) {
        for (uint32_t f = 0; f < words_per_tc; f++) {
            const uint32_t baseline = scratch[baseline_base + tc * words_per_tc + f];
            const uint32_t final_v = scratch[final_base + tc * words_per_tc + f];
            EXPECT_EQ(final_v, baseline) << "Mirrored DM TC" << tc << " field " << f
                                         << " changed during INTRA push/pop (baseline=" << baseline
                                         << " final=" << final_v << ")";
        }
    }

    // Host-side NEO mirror readback after the drained series: ClientL must be empty, and every DM-visible
    // counter must match the pre-launch snapshot — init sized and reset only ClientL, so anything else moving
    // means a T6 update aliased into the overlay pool.
    const auto live = read_live_tcs(this->device(), CoreCoord(0, 0), /*neo_id=*/0);
    ASSERT_GT(live.capacity.size(), client_l_tc);
    EXPECT_EQ(live.capacity[client_l_tc], entries_per_neo);
    EXPECT_EQ(live.tiles_available[client_l_tc], 0u)
        << "ClientL tiles_available not drained after equal push/pop — possible double-count or missed pop";
    for (uint32_t tc = 0; tc < ::dfb::NUM_TENSIX_TILE_COUNTERS_FOR_DM; tc++) {
        EXPECT_EQ(live.capacity[tc], before.capacity[tc]) << "Mirrored DM TC" << tc << " capacity changed";
        EXPECT_EQ(live.tiles_available[tc], before.tiles_available[tc])
            << "Mirrored DM TC" << tc << " tiles_available changed — T6->overlay alias leaked a push";
        EXPECT_EQ(live.space_available[tc], before.space_available[tc])
            << "Mirrored DM TC" << tc << " space_available changed — T6->overlay alias leaked a credit";
    }
}

}  // namespace tt::tt_metal
