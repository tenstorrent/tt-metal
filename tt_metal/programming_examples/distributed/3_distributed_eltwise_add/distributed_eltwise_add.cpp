// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

namespace {

DistributedHostBuffer make_host_buffer(
    const MeshShape& shape, const std::function<HostBuffer(const MeshCoordinate&)>& produce) {
    auto host_buffer = DistributedHostBuffer::create(shape);
    for (const auto& coord : MeshCoordinateRange(shape)) {
        host_buffer.emplace_shard(coord, [&]() { return produce(coord); });
    }
    return host_buffer;
}

}  // namespace

Program CreateEltwiseAddProgram(
    const std::shared_ptr<MeshBuffer>& a,
    const std::shared_ptr<MeshBuffer>& b,
    const std::shared_ptr<MeshBuffer>& c,
    size_t tile_size_bytes,
    uint32_t num_tiles) {
    auto program = CreateProgram();
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
            .math_fidelity = MathFidelity::HiFi4,
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
// 1. Allocating a MeshBuffer on every device (ReplicatedBufferConfig) and owning
//    per-device host data through DistributedHostBuffer
// 2. Writing distinct input shards to each device
// 3. Executing a MeshWorkload that performs element-wise addition in parallel
// 4. Reading each device's result shard and validating it
int main() {
    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));

    constexpr uint32_t num_tiles = 1;
    const auto tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    const auto per_device_size_bytes = num_tiles * tile_size_bytes;
    const auto num_uint32s = per_device_size_bytes / sizeof(uint32_t);

    auto local_buffer_config =
        DeviceLocalBufferConfig{.page_size = tile_size_bytes, .buffer_type = BufferType::DRAM, .bottom_up = false};
    const ReplicatedBufferConfig buffer_config{.size = per_device_size_bytes};

    auto a = MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    auto b = MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    auto c = MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());

    constexpr auto val_to_add = 0.5f;
    uint32_t seed = 0;
    auto a_host = make_host_buffer(mesh_device->shape(), [&](const MeshCoordinate&) {
        const uint32_t shard_seed = seed++;
        return HostBuffer(create_random_vector_of_bfloat16(per_device_size_bytes, 1, shard_seed));
    });
    auto b_host = make_host_buffer(mesh_device->shape(), [&](const MeshCoordinate&) {
        return HostBuffer(create_constant_vector_of_bfloat16(per_device_size_bytes, val_to_add));
    });

    auto& cq = mesh_device->mesh_command_queue();
    cq.enqueue_write(a, a_host, /*blocking=*/false);
    cq.enqueue_write(b, b_host, /*blocking=*/false);

    auto program = CreateEltwiseAddProgram(a, b, c, tile_size_bytes, num_tiles);

    auto mesh_workload = MeshWorkload();
    auto device_range = MeshCoordinateRange(mesh_device->shape());
    mesh_workload.add_program(device_range, std::move(program));
    EnqueueMeshWorkload(cq, mesh_workload, false /* blocking */);

    auto result_host = make_host_buffer(
        mesh_device->shape(), [&](const MeshCoordinate&) { return HostBuffer(std::vector<uint32_t>(num_uint32s, 0)); });
    cq.enqueue_read(c, result_host, /*shards=*/std::nullopt, /*blocking=*/true);

    auto transform_to_golden = [](const bfloat16& value) { return bfloat16(static_cast<float>(value) + val_to_add); };

    std::cout << "Partial results: (note we are running under BFP16. It's going to be less accurate)\n";
    size_t num_failures = 0;
    size_t total_values = 0;
    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        const auto a_shard = a_host.get_shard(coord);
        const auto c_shard = result_host.get_shard(coord);
        if (!a_shard.has_value() || !c_shard.has_value()) {
            continue;
        }

        std::vector<uint32_t> a_data(a_shard->view_as<uint32_t>().begin(), a_shard->view_as<uint32_t>().end());
        std::vector<uint32_t> golden_data =
            pack_bfloat16_vec_into_uint32_vec(unpack_uint32_vec_into_bfloat16_vec(a_data, transform_to_golden));
        const auto* result_bf16 = reinterpret_cast<const bfloat16*>(c_shard->view_bytes().data());
        const auto* golden_bf16 = reinterpret_cast<const bfloat16*>(golden_data.data());
        const size_t values_in_shard = golden_data.size() * 2;
        total_values += values_in_shard;
        for (size_t i = 0; i < values_in_shard; ++i) {
            if (!is_close(static_cast<float>(result_bf16[i]), static_cast<float>(golden_bf16[i]))) {
                num_failures++;
            }
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
