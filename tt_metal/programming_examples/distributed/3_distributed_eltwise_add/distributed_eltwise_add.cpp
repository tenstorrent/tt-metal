// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

Program CreateEltwiseAddProgram(
    MeshDevice& mesh_device,
    const std::shared_ptr<MeshBuffer>& a,
    const std::shared_ptr<MeshBuffer>& b,
    const std::shared_ptr<MeshBuffer>& c,
    size_t tile_size_bytes,
    uint32_t num_tiles) {
    auto program = CreateProgram(mesh_device);
    auto target_tensix_core = CoreRange(CoreCoord{0, 0});

    // Add circular buffers for data movement
    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t num_input_tiles = 1;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * tile_size_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, tile_size_bytes);
    tt_metal::CreateCircularBuffer(program, target_tensix_core, cb_src0_config);

    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * tile_size_bytes, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, tile_size_bytes);
    tt_metal::CreateCircularBuffer(program, target_tensix_core, cb_src1_config);

    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    constexpr uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * tile_size_bytes, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, tile_size_bytes);
    tt_metal::CreateCircularBuffer(program, target_tensix_core, cb_output_config);

    // Add data movement kernels
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*a->get_reference_buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*b->get_reference_buffer()).append_to(reader_compile_time_args);
    KernelHandle reader = CreateKernel(
        program,
        "tt_metal/programming_examples/contributed/vecadd/kernels/interleaved_tile_read.cpp",
        target_tensix_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*c->get_reference_buffer()).append_to(writer_compile_time_args);
    KernelHandle writer = CreateKernel(
        program,
        "tt_metal/programming_examples/contributed/vecadd/kernels/tile_write.cpp",
        target_tensix_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Create the eltwise binary kernel
    auto compute = CreateKernel(
        program,
        "tt_metal/programming_examples/contributed/vecadd/kernels/add.cpp",
        target_tensix_core,
        ComputeConfig{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {},
            .defines = {{"ELTWISE_OP", "add_tiles"}, {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"}}});

    // Set runtime arguments for each device
    SetRuntimeArgs(program, reader, target_tensix_core, {a->address(), b->address(), num_tiles});
    SetRuntimeArgs(program, writer, target_tensix_core, {c->address(), num_tiles});
    SetRuntimeArgs(program, compute, target_tensix_core, {num_tiles});

    return program;
}

// The example demonstrates distributed element-wise addition across a 2x4 mesh of devices:
//
// 1. Allocating buffers in lock-step: the same buffer, at the same address, on every device
// 2. Distributing input data across the mesh by writing each device its own part of it
// 3. Executing a distributed compute kernel that performs element-wise addition
//    in parallel across all devices
// 4. Gathering and validating the results from the distributed computation
//
// The example showcases TT-Metalium's ability to abstract away the complexity
// of distributed memory management and compute.
//
// TT-NN distributed tensors split data across a mesh and gather it back for you; see "Distributing Tensor to
// MeshDevice" in tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md.
int main() {
    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));

    // Each device holds one 32x32 tile of each input and of the output.
    auto num_tiles = 1;
    auto tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    auto distributed_buffer_size_bytes = mesh_device->num_devices() * tile_size_bytes;
    auto device_range = MeshCoordinateRange(mesh_device->shape());

    // Configure device-local buffer settings
    auto local_buffer_config =
        DeviceLocalBufferConfig{.page_size = tile_size_bytes, .buffer_type = BufferType::DRAM, .bottom_up = false};
    auto buffer_config = ReplicatedBufferConfig{.size = tile_size_bytes};

    // Create buffers for inputs and output on every device
    auto a = MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    auto b = MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    auto c = MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());

    // Create and initialize source data: `a` has a different tile for each device, `b` the same tile on every device
    constexpr auto val_to_add = 0.5f;
    std::vector<uint32_t> a_data =
        create_random_vector_of_bfloat16(distributed_buffer_size_bytes, 1 /* rand_max_float */, 0 /* seed */);
    std::vector<uint32_t> b_data = create_constant_vector_of_bfloat16(tile_size_bytes, val_to_add);

    // Write each device its own tile of `a`, in row-major order of the mesh, and the same tile of `b` to all of them
    auto& cq = mesh_device->mesh_command_queue();
    const size_t words_per_device = tile_size_bytes / sizeof(uint32_t);
    size_t offset = 0;
    for (const auto& coord : device_range) {
        std::vector<uint32_t> a_tile(a_data.begin() + offset, a_data.begin() + offset + words_per_device);
        WriteShard(cq, a, a_tile, coord, true /* blocking */);
        offset += words_per_device;
    }
    EnqueueWriteMeshBuffer(cq, b, b_data, false /* blocking */);

    // Create program for distributed computation
    auto program = CreateEltwiseAddProgram(*mesh_device, a, b, c, tile_size_bytes, num_tiles);

    // Create mesh workload and broadcast the program across all devices
    auto mesh_workload = MeshWorkload();

    mesh_workload.add_program(device_range, std::move(program));
    EnqueueMeshWorkload(cq, mesh_workload, false /* blocking */);

    // Read back each device's result tile, in the same order as the tiles of `a`
    std::vector<uint32_t> result_data;
    result_data.reserve(a_data.size());
    for (const auto& coord : device_range) {
        std::vector<uint32_t> c_shard;
        ReadShard(cq, c_shard, c, coord, true /* blocking */);
        result_data.insert(result_data.end(), c_shard.begin(), c_shard.end());
    }

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
