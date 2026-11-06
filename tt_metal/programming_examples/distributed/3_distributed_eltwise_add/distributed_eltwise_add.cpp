// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

// Make a tensor over the memory of an existing mesh buffer, so that kernels can access the buffer through a tensor
// binding. The tensor does not own the memory: the mesh buffer must stay alive while the tensor is in use.
MeshTensor MakeTensorView(const std::shared_ptr<MeshBuffer>& buffer, const TensorSpec& per_device_spec) {
    std::shared_ptr<MeshBuffer> view =
        MeshBuffer::create(buffer->global_config(), buffer->device_local_config(), buffer->device(), buffer->address());
    return MeshTensor::from_buffer(std::move(*view), per_device_spec);
}

Program CreateEltwiseAddProgram(
    MeshDevice& mesh_device,
    const MeshTensor& a,
    const MeshTensor& b,
    const MeshTensor& c,
    size_t tile_size_bytes,
    uint32_t num_tiles) {
    constexpr auto target_tensix_core = experimental::NodeCoord{0, 0};

    // Names identify the kernels, the dataflow buffers and the tensor parameters within the program spec.
    const experimental::KernelSpecName reader{"reader"};
    const experimental::KernelSpecName writer{"writer"};
    const experimental::KernelSpecName compute{"compute"};
    const experimental::TensorParamName a_tensor_param{"a"};
    const experimental::TensorParamName b_tensor_param{"b"};
    const experimental::TensorParamName c_tensor_param{"c"};

    // Add dataflow buffers for data movement
    const experimental::DFBSpecName src0_dfb{"src0"};
    constexpr uint32_t num_input_tiles = 1;
    experimental::DataflowBufferSpec dfb_src0_spec{
        .unique_id = src0_dfb,
        .entry_size = static_cast<uint32_t>(tile_size_bytes),
        .num_entries = num_input_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    const experimental::DFBSpecName src1_dfb{"src1"};
    experimental::DataflowBufferSpec dfb_src1_spec{
        .unique_id = src1_dfb,
        .entry_size = static_cast<uint32_t>(tile_size_bytes),
        .num_entries = num_input_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    const experimental::DFBSpecName output_dfb{"output"};
    constexpr uint32_t num_output_tiles = 1;
    experimental::DataflowBufferSpec dfb_output_spec{
        .unique_id = output_dfb,
        .entry_size = static_cast<uint32_t>(tile_size_bytes),
        .num_entries = num_output_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // Add data movement kernels
    experimental::KernelSpec reader_spec{
        .unique_id = reader,
        .source = "tt_metal/programming_examples/contributed/vecadd/kernels/interleaved_tile_read.cpp",
        .dfb_bindings = {experimental::ProducerOf(src0_dfb, "in0"), experimental::ProducerOf(src1_dfb, "in1")},
        .tensor_bindings =
            {{.tensor_parameter_name = a_tensor_param, .accessor_name = "a"},
             {.tensor_parameter_name = b_tensor_param, .accessor_name = "b"}},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
            },
    };

    experimental::KernelSpec writer_spec{
        .unique_id = writer,
        .source = "tt_metal/programming_examples/contributed/vecadd/kernels/tile_write.cpp",
        .dfb_bindings = {experimental::ConsumerOf(output_dfb, "out0")},
        .tensor_bindings = {{.tensor_parameter_name = c_tensor_param, .accessor_name = "c"}},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            },
    };

    // Create the eltwise binary kernel
    experimental::KernelSpec compute_spec{
        .unique_id = compute,
        .source = "tt_metal/programming_examples/contributed/vecadd/kernels/add.cpp",
        .compiler_options =
            {.defines = {{"ELTWISE_OP", "add_tiles"}, {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"}},
             .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {experimental::ConsumerOf(src0_dfb, "in0"),
             experimental::ConsumerOf(src1_dfb, "in1"),
             experimental::ProducerOf(output_dfb, "out0")},
        .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                .sfpu_precision_mode = Precision::Precise,
                .enable_32_bit_dest = false,
            },
    };

    experimental::ProgramSpec program_spec{
        .name = "distributed_eltwise_add",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {dfb_src0_spec, dfb_src1_spec, dfb_output_spec},
        .tensor_parameters =
            {{.unique_id = a_tensor_param, .spec = a.tensor_spec()},
             {.unique_id = b_tensor_param, .spec = b.tensor_spec()},
             {.unique_id = c_tensor_param, .spec = c.tensor_spec()}},
        .work_units =
            {{.name = "distributed_eltwise_add",
              .kernels = {reader, writer, compute},
              .target_nodes = target_tensix_core}},
    };
    Program program = experimental::MakeProgramFromSpec(mesh_device, program_spec);

    // Set runtime arguments for each device
    experimental::ProgramRunArgs run_args{
        .kernel_run_args =
            {{.kernel = reader,
              .runtime_arg_values =
                  experimental::MakeRuntimeArgsForSingleNode(target_tensix_core, {{"n_tiles", num_tiles}})},
             {.kernel = writer,
              .runtime_arg_values =
                  experimental::MakeRuntimeArgsForSingleNode(target_tensix_core, {{"n_tiles", num_tiles}})},
             {.kernel = compute,
              .runtime_arg_values =
                  experimental::MakeRuntimeArgsForSingleNode(target_tensix_core, {{"n_tiles", num_tiles}})}},
        .tensor_args =
            {{a_tensor_param, experimental::TensorArgument{a}},
             {b_tensor_param, experimental::TensorArgument{b}},
             {c_tensor_param, experimental::TensorArgument{c}}},
    };
    experimental::SetProgramRunArgs(program, run_args);

    return program;
}

// The example demonstrates distributed element-wise addition across a 2x4 mesh of devices:
//
// 1. Allocating distributed buffers that automatically shard data across the mesh
// 2. Initializing and distributing input data to each device's local memory
// 3. Executing a distributed compute kernel that performs element-wise addition
//    in parallel across all devices
// 4. Gathering and validating the results from the distributed computation
//
// The example showcases TT-Metalium's ability to abstract away the complexity
// of distributed memory management and compute.
int main() {
    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));

    // Define the global buffer shape and shard shape for distributed buffers
    auto shard_shape = Shape2D{32, 32};
    auto distributed_buffer_shape =
        Shape2D{shard_shape.height() * mesh_device->num_rows(), shard_shape.width() * mesh_device->num_cols()};
    auto num_tiles = 1;
    auto tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    auto distributed_buffer_size_bytes = mesh_device->num_rows() * mesh_device->num_cols() * tile_size_bytes;

    // Configure device-local buffer settings
    auto local_buffer_config =
        DeviceLocalBufferConfig{.page_size = tile_size_bytes, .buffer_type = BufferType::DRAM, .bottom_up = false};
    auto distributed_buffer_config = tt::tt_metal::distributed::ShardedBufferConfig{
        .global_size = distributed_buffer_size_bytes,
        .global_buffer_shape = distributed_buffer_shape,
        .shard_shape = shard_shape,
        .shard_orientation = ShardOrientation::ROW_MAJOR};

    // Create distributed buffers for inputs and output
    auto a = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
    auto b = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
    auto c = MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());

    // Create and initialize source data
    constexpr auto val_to_add = 0.5f;
    std::vector<uint32_t> a_data =
        create_random_vector_of_bfloat16(distributed_buffer_size_bytes, 1 /* rand_max_float */, 0 /* seed */);
    std::vector<uint32_t> b_data = create_constant_vector_of_bfloat16(distributed_buffer_size_bytes, val_to_add);

    // Write data to distributed buffers
    auto& cq = mesh_device->mesh_command_queue();
    EnqueueWriteMeshBuffer(cq, a, a_data, false /* blocking */);
    EnqueueWriteMeshBuffer(cq, b, b_data, false /* blocking */);

    // Each device holds one shard of the distributed buffers. The kernels access the shard through a tensor that
    // views the buffer's memory. Each row of a row-major tensor is one page, so a row of one tile's worth of elements
    // gives a page size of one tile.
    TensorSpec shard_tensor_spec(
        Shape({static_cast<uint32_t>(num_tiles), static_cast<uint32_t>(tile_size_bytes / sizeof(bfloat16))}),
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
    MeshTensor a_tensor = MakeTensorView(a, shard_tensor_spec);
    MeshTensor b_tensor = MakeTensorView(b, shard_tensor_spec);
    MeshTensor c_tensor = MakeTensorView(c, shard_tensor_spec);

    // Create program for distributed computation
    auto program = CreateEltwiseAddProgram(*mesh_device, a_tensor, b_tensor, c_tensor, tile_size_bytes, num_tiles);

    // Create mesh workload and broadcast the program across all devices
    auto mesh_workload = MeshWorkload();
    auto device_range = MeshCoordinateRange(mesh_device->shape());

    mesh_workload.add_program(device_range, std::move(program));
    EnqueueMeshWorkload(cq, mesh_workload, false /* blocking */);

    // Read back results
    std::vector<uint32_t> result_data(a_data.size(), 0);
    EnqueueReadMeshBuffer(cq, result_data, c, true /* blocking */);

    // Verify results
    auto transform_to_golden = [](const bfloat16& a) { return bfloat16(static_cast<float>(a) + val_to_add); };
    std::vector<uint32_t> golden_data =
        pack_bfloat16_vec_into_uint32_vec(unpack_uint32_vec_into_bfloat16_vec(a_data, transform_to_golden));

    // Print partial results so we can see the output is correct (plus or minus some error due to BFP16 precision)
    std::cout << "Partial results: (note we are running under BFP16. It's going to be less accurate)\n";
    bfloat16* c_bf16 = reinterpret_cast<bfloat16*>(result_data.data());
    bfloat16* golden_bf16 = reinterpret_cast<bfloat16*>(golden_data.data());

    size_t num_failures = 0;
    auto total_values = result_data.size() * 2;
    for (int i = 0; i < total_values; i++) {
        if (!is_close(static_cast<float>(c_bf16[i]), static_cast<float>(golden_bf16[i]))) {
            num_failures++;
        }
    }

    std::cout << "Total values: " << total_values << "\n";
    std::cout << "Distributed elementwise add verification: " << (total_values - num_failures) << " / " << total_values
              << " passed\n";
    if (num_failures > 0) {
        std::cout << "Distributed elementwise add verification failed with " << num_failures << " failures\n";
        throw std::runtime_error("Distributed elementwise add verification failed");
    }

    return 0;
}
