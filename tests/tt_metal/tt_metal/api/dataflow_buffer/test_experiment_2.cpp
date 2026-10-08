// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dfb_test_common.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

TEST_F(UnitMeshFixture, Experiment2_DmComputeDmMulAdd) {
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
    MeshTensor in1_tensor = MeshTensor::allocate_on_device(this->device(), tensor_spec);
    MeshTensor in2_tensor = MeshTensor::allocate_on_device(this->device(), tensor_spec);
    MeshTensor out_tensor = MeshTensor::allocate_on_device(this->device(), tensor_spec);

    const m2::DFBSpecName DFB_IN0{"dfb_in0"};
    const m2::DFBSpecName DFB_IN1{"dfb_in1"};
    const m2::DFBSpecName DFB_IN2{"dfb_in2"};
    const m2::DFBSpecName DFB_OUT{"dfb_out"};

    const m2::KernelSpecName PRODUCER_A{"producer_a"};
    const m2::KernelSpecName PRODUCER_B{"producer_b"};
    const m2::KernelSpecName PRODUCER_C{"producer_c"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::KernelSpecName COMPUTE{"compute"};

    const m2::TensorParamName IN0_TENSOR{"in0_tensor"};
    const m2::TensorParamName IN1_TENSOR{"in1_tensor"};
    const m2::TensorParamName IN2_TENSOR{"in2_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    const uint32_t dfb_num_entries = 8;
    m2::DataflowBufferSpec dfb_in0{
        .unique_id = DFB_IN0,
        .entry_size = page_size_bytes,
        .num_entries = dfb_num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    m2::DataflowBufferSpec dfb_in1{
        .unique_id = DFB_IN1,
        .entry_size = page_size_bytes,
        .num_entries = dfb_num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    m2::DataflowBufferSpec dfb_in2{
        .unique_id = DFB_IN2,
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

    auto producer_a = make_dm_kernel(
        PRODUCER_A,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/test_experiment_2_producer.cpp",
        /*num_threads=*/1,
        /*disable_implicit_sync_for=*/{});
    producer_a.dfb_bindings = {
        {.dfb_spec_name = DFB_IN0,
         .accessor_name = "in_0",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
    };

    auto producer_b = make_dm_kernel(
        PRODUCER_B,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/test_experiment_2_producer.cpp",
        /*num_threads=*/1,
        /*disable_implicit_sync_for=*/{});
    producer_b.dfb_bindings = {
        {.dfb_spec_name = DFB_IN1,
         .accessor_name = "in_0",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
    };

    auto producer_c = make_dm_kernel(
        PRODUCER_C,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/test_experiment_2_producer.cpp",
        /*num_threads=*/1,
        /*disable_implicit_sync_for=*/{});
    producer_c.dfb_bindings = {
        {.dfb_spec_name = DFB_IN2,
         .accessor_name = "in_0",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
    };
    producer_a.tensor_bindings = {{.tensor_parameter_name = IN0_TENSOR, .accessor_name = "src"}};
    producer_b.tensor_bindings = {{.tensor_parameter_name = IN1_TENSOR, .accessor_name = "src"}};
    producer_c.tensor_bindings = {{.tensor_parameter_name = IN2_TENSOR, .accessor_name = "src"}};
    auto consumer = make_dm_kernel(
        CONSUMER,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/test_experiment_2_consumer.cpp",
        /*num_threads=*/2,
        /*disable_implicit_sync_for=*/{});
    consumer.dfb_bindings = {
        {.dfb_spec_name = DFB_OUT,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED}};
    consumer.tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst"}};

    const std::string compute_source = "tests/tt_metal/tt_metal/test_kernels/compute/test_experiment_2_compute.cpp";
    auto compute = make_compute_kernel(COMPUTE, compute_source, /*num_threads=*/4);
    compute.dfb_bindings = {
        {.dfb_spec_name = DFB_IN0,
         .accessor_name = "in_0",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
        {.dfb_spec_name = DFB_IN1,
         .accessor_name = "in_1",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
        {.dfb_spec_name = DFB_IN2,
         .accessor_name = "in_2",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
        {.dfb_spec_name = DFB_OUT,
         .accessor_name = "out",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
    };

    producer_a.runtime_arg_schema = {.runtime_arg_names = {"tiles_per_neo", "start_page"}};
    producer_b.runtime_arg_schema = {.runtime_arg_names = {"tiles_per_neo", "start_page"}};
    producer_c.runtime_arg_schema = {.runtime_arg_names = {"tiles_per_neo", "start_page"}};
    compute.runtime_arg_schema = {.runtime_arg_names = {"tiles_per_neo"}};
    consumer.runtime_arg_schema = {.runtime_arg_names = {"tiles_per_neo", "start_page"}};

    m2::WorkUnitSpec wu{
        .name = "wu",
        .kernels = {PRODUCER_A, PRODUCER_B, PRODUCER_C, CONSUMER, COMPUTE},
        .target_nodes = all_nodes,
    };

    m2::ProgramSpec spec{
        .name = "exp2_test",
        .kernels = {producer_a, producer_b, producer_c, consumer, compute},
        .dataflow_buffers = {dfb_in0, dfb_in1, dfb_in2, dfb_out},
        .tensor_parameters =
            {
                {.unique_id = IN0_TENSOR, .spec = in0_tensor.tensor_spec()},
                {.unique_id = IN1_TENSOR, .spec = in1_tensor.tensor_spec()},
                {.unique_id = IN2_TENSOR, .spec = in2_tensor.tensor_spec()},
                {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()},
            },
        .work_units = {wu},
    };

    Program program = m2::MakeProgramFromSpec(this->device(), spec);

    m2::ProgramRunArgs::KernelRunArgs producer_a_args{.kernel = PRODUCER_A};
    m2::ProgramRunArgs::KernelRunArgs producer_b_args{.kernel = PRODUCER_B};
    m2::ProgramRunArgs::KernelRunArgs producer_c_args{.kernel = PRODUCER_C};
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
                producer_a_args.runtime_arg_values,
                n,
                {{"tiles_per_neo", tiles_for_neo}, {"start_page", start_for_neo}});
            m2::AddRuntimeArgsForNode(
                producer_b_args.runtime_arg_values,
                n,
                {{"tiles_per_neo", tiles_for_neo}, {"start_page", start_for_neo}});
            m2::AddRuntimeArgsForNode(
                producer_c_args.runtime_arg_values,
                n,
                {{"tiles_per_neo", tiles_for_neo}, {"start_page", start_for_neo}});
            m2::AddRuntimeArgsForNode(compute_args.runtime_arg_values, n, {{"tiles_per_neo", tiles_for_neo}});
            m2::AddRuntimeArgsForNode(
                consumer_args.runtime_arg_values, n, {{"tiles_per_neo", tiles_for_neo}, {"start_page", start_for_neo}});
            start_for_neo += tiles_for_neo;
        }
    }
    ASSERT_EQ(start_for_neo, num_pages);

    m2::ProgramRunArgs params;
    params.kernel_run_args = {producer_a_args, producer_b_args, producer_c_args, compute_args, consumer_args};
    params.tensor_args = {
        {IN0_TENSOR, std::cref(in0_tensor)},
        {IN1_TENSOR, std::cref(in1_tensor)},
        {IN2_TENSOR, std::cref(in2_tensor)},
        {OUT_TENSOR, std::cref(out_tensor)}};
    m2::SetProgramRunArgs(program, params);

    constexpr uint32_t total_bytes = page_size_bytes * num_pages;
    const auto a = create_random_vector_of_bfloat16(total_bytes, 2, 1, -1.0f);
    const auto b = create_random_vector_of_bfloat16(total_bytes, 2, 2, -1.0f);
    const auto c = create_random_vector_of_bfloat16(total_bytes, 2, 3, -1.0f);
    slow_dispatch::WriteToBuffer(in0_tensor.mesh_buffer(), a);
    m2_writeshard_barrier_uint32(this->device(), in0_tensor, a);
    slow_dispatch::WriteToBuffer(in1_tensor.mesh_buffer(), b);
    m2_writeshard_barrier_uint32(this->device(), in1_tensor, b);
    slow_dispatch::WriteToBuffer(in2_tensor.mesh_buffer(), c);
    m2_writeshard_barrier_uint32(this->device(), in2_tensor, c);
    LaunchProgram(this->device(), std::move(program));

    std::vector<uint32_t> output;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), output);

    const auto a_bf16 = unpack_uint32_vec_into_bfloat16_vec(a);
    const auto b_bf16 = unpack_uint32_vec_into_bfloat16_vec(b);
    const auto c_bf16 = unpack_uint32_vec_into_bfloat16_vec(c);
    std::vector<bfloat16> golden(a_bf16.size());
    for (size_t i = 0; i < golden.size(); ++i) {
        golden[i] = bfloat16(
            static_cast<float>(a_bf16[i]) * static_cast<float>(b_bf16[i]) +
            std::max(static_cast<float>(c_bf16[i]), 0.0f));
    }
    const auto golden_packed = pack_bfloat16_vec_into_uint32_vec(golden);

    constexpr float tolerance = 0.02f;
    EXPECT_TRUE(packed_uint32_t_vector_comparison(
        output, golden_packed, [](float x, float y) { return std::abs(x - y) < tolerance; }));
}

}  // namespace tt::tt_metal
