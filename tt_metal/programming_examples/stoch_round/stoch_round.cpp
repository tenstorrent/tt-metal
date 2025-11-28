// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>

#include <cstdint>
#include <random>

using namespace tt;
using namespace tt::tt_metal;

std::shared_ptr<distributed::MeshBuffer> MakeMeshBuffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t size, uint32_t page_size, bool sram) {
    distributed::DeviceLocalBufferConfig local_config{
        .page_size = page_size, .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM)};
    distributed::ReplicatedBufferConfig buffer_config{.size = size};
    return distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
}

std::shared_ptr<distributed::MeshBuffer> MakeMeshBufferFP32(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t n_tiles, bool sram) {
    constexpr uint32_t tile_size = sizeof(float) * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    const uint32_t page_tiles = sram ? n_tiles : 1;
    return MakeMeshBuffer(mesh_device, tile_size * n_tiles, page_tiles * tile_size, sram);
}

std::shared_ptr<distributed::MeshBuffer> MakeMeshBufferBF16(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t n_tiles, bool sram) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    const uint32_t page_tiles = sram ? n_tiles : 1;
    return MakeMeshBuffer(mesh_device, tile_size * n_tiles, page_tiles * tile_size, sram);
}

int main() {
    int device_id = 0;

    const uint32_t n_tiles = 640;

    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    const uint32_t tile_size = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    std::map<CoreCoord, uint32_t> core_tile_idx;

    auto src = MakeMeshBufferFP32(mesh_device, n_tiles, false);
    auto dst = MakeMeshBufferBF16(mesh_device, n_tiles, false);

    float large_val = 1.0f;
    float small_delta = 0.002f;
    std::vector<float> src_data(tile_size * n_tiles, large_val + small_delta);

    auto core_grid = mesh_device->compute_with_storage_grid_size();
    uint32_t num_cores_x = core_grid.x;
    uint32_t num_cores_y = core_grid.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    constexpr uint32_t tile_size_bytes = sizeof(float) * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    constexpr uint32_t tile_size_bytes_bf16 = sizeof(bfloat16) * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    constexpr uint32_t cir_buf_num_tiles = 4;

    CreateCircularBuffer(
        program,
        all_device_cores,
        CircularBufferConfig(cir_buf_num_tiles * tile_size_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_0, tile_size_bytes));

    CreateCircularBuffer(
        program,
        all_device_cores,
        CircularBufferConfig(cir_buf_num_tiles * tile_size_bytes_bf16, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_16, tile_size_bytes_bf16));

    constexpr bool row_major = true;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(core_grid, n_tiles, row_major);

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)tt::CBIndex::c_0};
    TensorAccessorArgs(*src).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)tt::CBIndex::c_16};
    TensorAccessorArgs(*dst).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        (std::uint32_t)tt::CBIndex::c_0, (std::uint32_t)tt::CBIndex::c_16};

    auto reader = CreateKernel(
        program,
        "tt_metal/programming_examples/stoch_round/kernels/dataflow/read_tile.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});

    auto writer = CreateKernel(
        program,
        "tt_metal/programming_examples/stoch_round/kernels/dataflow/write_tile.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args});

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;

    tt::tt_metal::ComputeConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = unpack_to_dest_mode,
        .math_approx_mode = false,
        .compile_args = compute_compile_time_args,
        .defines = {},
    };

    auto compute = CreateKernel(
        program, "tt_metal/programming_examples/stoch_round/kernels/compute/round_tile.cpp", all_cores, compute_config);

    auto work_groups = {
        std::make_pair(core_group_1, num_tiles_per_core_group_1),
        std::make_pair(core_group_2, num_tiles_per_core_group_2)};

    uint32_t start_tile_id = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(1, 0xFFFFFFFF);

    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                uint32_t core_seed = dis(gen);
                SetRuntimeArgs(program, compute, core, {core_seed, work_per_core, start_tile_id});
                SetRuntimeArgs(program, reader, core, {src->address(), work_per_core, start_tile_id});
                SetRuntimeArgs(program, writer, core, {dst->address(), work_per_core, start_tile_id});
                core_tile_idx[core] = start_tile_id;

                start_tile_id += work_per_core;
            }
        }
    }

    EnqueueWriteMeshBuffer(cq, src, src_data, false);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    fmt::print("Kernel execution finished\n");

    std::vector<bfloat16> dst_data;
    distributed::EnqueueReadMeshBuffer(cq, dst_data, dst, true);

    int rounded_up_count = 0;
    int rounded_down_count = 0;

    for (size_t i = 0; i < dst_data.size(); i++) {
        float res = static_cast<float>(dst_data[i]);

        if (res == std::bit_cast<float>(0x3F810000)) {  // 0x3F810000 is the bit representation of 1.0078125 in fp32
            rounded_up_count++;
        } else if (res == 1.0f) {
            rounded_down_count++;
        } else {
            fmt::print(stderr, "Invalid result\n");
            mesh_device->close();
            return 1;
        }
    }

    fmt::print("Total Elements: {}\n", dst_data.size());
    fmt::print("Rounded Down (stagnated): {}\n", rounded_down_count);
    fmt::print("Rounded Up (progressed):  {}\n", rounded_up_count);

    if (rounded_up_count == 0) {
        fmt::print(stderr, "FAILED: All values rounded down. This looks like RNE (Round Nearest Even).\n");
        mesh_device->close();
        return 1;
    } else {
        float ratio = (float)rounded_up_count / dst_data.size();
        fmt::print("SUCCESS: Stochastic behavior detected.\n");
        fmt::print("Rounding Ratio: {:.2f}% (Expected 25.6% for 0.002 delta)\n", ratio * 100.0f);  // 128/500 = 25.6%
    }

    mesh_device->close();
    return 0;
}
