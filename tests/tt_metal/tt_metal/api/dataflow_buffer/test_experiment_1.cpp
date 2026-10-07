// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dfb_test_common.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

TEST_F(UnitMeshFixture, Experiment1_DmComputeDmCopy) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Quasar only";
    }

    constexpr uint32_t page_size_bytes = 2 * 32 * 32;  // bf16 tile = 2048 B
    constexpr uint32_t num_pages =
        5 * 4 * 8 * 4 - 3;  // five tiles per neo tensix minus 3 to test if the last 3 will be fine

    const auto grid = this->device().compute_with_storage_grid_size();  // 8 x 4 on craq-sim
    const m2::NodeRange all_nodes(m2::NodeCoord{0, 0}, m2::NodeCoord{grid.x - 1, grid.y - 1});

    const TensorSpec tensor_spec = make_flat_dram_tensor_spec(page_size_bytes, num_pages, DataType::BFLOAT16);
    MeshTensor in0_tensor = MeshTensor::allocate_on_device(this->device(), tensor_spec);
    MeshTensor out_tensor = MeshTensor::allocate_on_device(this->device(), tensor_spec);

    const m2::DFBSpecName DFB_IN0{"dfb_in0"};
    const m2::DFBSpecName DFB_OUT{"dfb_out"};

    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::KernelSpecName COMPUTE{"compute"};

    const m2::TensorParamName IN0_TENSOR{"in0_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    const uint32_t dfb_num_entries = 8;
    m2::DataflowBufferSpec dfb_in0{
        .unique_id = DFB_IN0,
        .entry_size = page_size_bytes,
        .num_entries = dfb_num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    m2::DataflowBufferSpec dfb_out{
        .unique_id = DFB_OUT,
        .entry_size = page_size_bytes,
        .num_entries = dfb_num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    auto producer = make_dm_kernel(
        PRODUCER,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/test_experiment_1_producer.cpp",
        /*num_threads=*/4,
        /*disable_implicit_sync_for=*/{});
    producer.dfb_bindings = {
        {.dfb_spec_name = DFB_IN0,
         .accessor_name = "in_0",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    producer.tensor_bindings = {{.tensor_parameter_name = IN0_TENSOR, .accessor_name = "src"}};
    auto consumer = make_dm_kernel(
        CONSUMER,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/test_experiment_1_consumer.cpp",
        /*num_threads=*/2,
        /*disable_implicit_sync_for=*/{});
    consumer.dfb_bindings = {
        {.dfb_spec_name = DFB_OUT,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    consumer.tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst"}};

    const std::string compute_source = "tests/tt_metal/tt_metal/test_kernels/compute/test_experiment_1_compute.cpp";
    auto compute = make_compute_kernel(COMPUTE, compute_source, /*num_threads=*/4);
    compute.dfb_bindings = {
        {.dfb_spec_name = DFB_IN0,
         .accessor_name = "in_0",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
        {.dfb_spec_name = DFB_OUT,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
    };

    producer.runtime_arg_schema = {.runtime_arg_names = {"tiles_per_neo", "start_page"}};
    compute.runtime_arg_schema = {.runtime_arg_names = {"tiles_per_neo"}};
    consumer.runtime_arg_schema = {.runtime_arg_names = {"tiles_per_neo", "start_page"}};

    m2::WorkUnitSpec wu{
        .name = "wu",
        .kernels = {PRODUCER, CONSUMER, COMPUTE},
        .target_nodes = all_nodes,
    };

    m2::ProgramSpec spec{
        .name = "exp1_test",
        .kernels = {producer, consumer, compute},
        .dataflow_buffers = {dfb_in0, dfb_out},
        .tensor_parameters =
            {
                {.unique_id = IN0_TENSOR, .spec = in0_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };

    Program program = m2::MakeProgramFromSpec(this->device(), spec);

    m2::ProgramRunArgs::KernelRunArgs producer_args{.kernel = PRODUCER};
    m2::ProgramRunArgs::KernelRunArgs compute_args{.kernel = COMPUTE};
    m2::ProgramRunArgs::KernelRunArgs consumer_args{.kernel = CONSUMER};

    const uint32_t num_neos = grid.x * grid.y;
    const uint32_t tiles_per_neo_base = num_pages / num_neos;
    uint32_t tiles_per_neo_remainder = num_pages % num_neos;

    uint32_t start_for_neo = 0;
    for (uint32_t x = 0; x < grid.x; x++) {
        for (uint32_t y = 0; y < grid.y; y++) {
            const uint32_t tiles_for_neo = tiles_per_neo_base + (tiles_per_neo_remainder > 0 ? 1 : 0);
            if (tiles_per_neo_remainder > 0) {
                tiles_per_neo_remainder--;
            }
            const m2::NodeCoord n{x, y};
            m2::AddRuntimeArgsForNode(
                producer_args.runtime_arg_values, n, {{"tiles_per_neo", tiles_for_neo}, {"start_page", start_for_neo}});
            m2::AddRuntimeArgsForNode(compute_args.runtime_arg_values, n, {{"tiles_per_neo", tiles_for_neo}});
            m2::AddRuntimeArgsForNode(
                consumer_args.runtime_arg_values, n, {{"tiles_per_neo", tiles_for_neo}, {"start_page", start_for_neo}});
            start_for_neo += tiles_for_neo;
        }
    }
    ASSERT_EQ(start_for_neo, num_pages);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {producer_args, compute_args, consumer_args};
    params.tensor_args = {{IN0_TENSOR, std::cref(in0_tensor)}, {OUT_TENSOR, std::cref(out_tensor)}};
    m2::SetProgramRunArgs(program, params);

    auto input = create_random_vector_of_bfloat16(page_size_bytes * num_pages, 2.0f, 42);
    slow_dispatch::WriteToBuffer(in0_tensor.mesh_buffer(), input);
    m2_writeshard_barrier_uint32(this->device(), in0_tensor, input);
    LaunchProgram(this->device(), std::move(program));

    std::vector<uint32_t> output;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), output);
    EXPECT_EQ(input, output);
}

}  // namespace tt::tt_metal
