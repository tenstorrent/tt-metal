// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
#include "tt-metalium/base_types.hpp"

using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main(int /*argc*/, char** /*argv*/) {
    bool pass = true;

    try {
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::MeshWorkload workload;
        auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();

        constexpr uint32_t M = 512;
        constexpr uint32_t K = 576;
        constexpr uint32_t N = 640;
        constexpr uint32_t tile_size = 32;

        constexpr uint32_t padded_M = ((M + tile_size - 1) / tile_size) * tile_size;
        constexpr uint32_t padded_K = ((K + tile_size - 1) / tile_size) * tile_size;
        constexpr uint32_t padded_N = ((N + tile_size - 1) / tile_size) * tile_size;

        constexpr uint32_t Mt = padded_M / tile_size;  // ---> 16
        constexpr uint32_t Kt = padded_K / tile_size;  // ---> 18
        constexpr uint32_t Nt = padded_N / tile_size;  // ---> 20
        constexpr uint32_t A_total_tiles = Mt * Kt;
        constexpr uint32_t B_total_tiles = Kt * Nt;
        constexpr uint32_t C_total_tiles = Mt * Nt;

        fmt::print("---------------------------------------------\n");
        fmt::print("Mt: {}\n", Mt);
        fmt::print("Kt: {}\n", Kt);
        fmt::print("Nt: {}\n", Nt);
        fmt::print("---------------------------------------------\n");

        uint32_t start_core_x = 0;
        uint32_t start_core_y = 0;
        uint32_t num_cores_x = 5;
        uint32_t num_cores_y = 4;

        uint32_t in0_block_w = 2;
        uint32_t out_subblock_h = 4;
        uint32_t out_subblock_w = 2;

        uint32_t per_core_M = Mt / num_cores_y;
        uint32_t per_core_N = Nt / num_cores_x;

        CoreRange all_cores(
            {(std::size_t)start_core_x, (std::size_t)start_core_y},
            {(std::size_t)num_cores_x - 1, (std::size_t)num_cores_y - 1});
        CoreRange in0_sender_in1_sender(
            {(std::size_t)start_core_x, (std::size_t)start_core_y},
            {(std::size_t)start_core_x, (std::size_t)start_core_y});
        CoreRange in0_sender_in1_reciever(
            {(std::size_t)start_core_x, (std::size_t)start_core_y + 1},
            {(std::size_t)start_core_x, (std::size_t)num_cores_y - 1});
        CoreRange in0_reciever_in1_sender(
            {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
            {(std::size_t)num_cores_x - 1, (std::size_t)start_core_y});
        CoreRange in0_reciever_in1_reciever(
            {(std::size_t)start_core_x + 1, (std::size_t)start_core_y + 1},
            {(std::size_t)num_cores_x - 1, (std::size_t)num_cores_y - 1});

        CoreCoord start_core = {(std::size_t)start_core_x, (std::size_t)start_core_y};
        auto start_core_physical = mesh_device->worker_core_from_logical_core(start_core);
        fmt::print("start core physical address: {}\n", start_core_physical);

        constexpr uint32_t elements_per_tile = tile_size * tile_size;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

        distributed::DeviceLocalBufferConfig dram_config{.page_size = tile_size_bytes, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig A_buffer_config{.size = A_total_tiles * tile_size_bytes};
        distributed::ReplicatedBufferConfig B_buffer_config{.size = B_total_tiles * tile_size_bytes};
        distributed::ReplicatedBufferConfig C_buffer_config{.size = C_total_tiles * tile_size_bytes};

        auto srcA_dram_buffer = distributed::MeshBuffer::create(A_buffer_config, dram_config, mesh_device.get());
        auto srcB_dram_buffer = distributed::MeshBuffer::create(B_buffer_config, dram_config, mesh_device.get());
        auto srcC_dram_buffer = distributed::MeshBuffer::create(C_buffer_config, dram_config, mesh_device.get());
        auto dst_dram_buffer = distributed::MeshBuffer::create(C_buffer_config, dram_config, mesh_device.get());

        std::vector<bfloat16> a_tensor_data(elements_per_tile * A_total_tiles);
        std::vector<bfloat16> b_tensor_data(elements_per_tile * B_total_tiles);
        std::vector<bfloat16> c_tensor_data(elements_per_tile * C_total_tiles);

        std::fill(a_tensor_data.begin(), a_tensor_data.end(), bfloat16(2.0f));
        std::fill(b_tensor_data.begin(), b_tensor_data.end(), bfloat16(3.0f));
        std::fill(c_tensor_data.begin(), c_tensor_data.end(), bfloat16(4.0f));

        distributed::EnqueueWriteMeshBuffer(cq, srcA_dram_buffer, a_tensor_data, false);
        distributed::EnqueueWriteMeshBuffer(cq, srcB_dram_buffer, b_tensor_data, false);
        distributed::EnqueueWriteMeshBuffer(cq, srcC_dram_buffer, c_tensor_data, false);

        tt::CBIndex src0_cb_index = tt::CBIndex::c_0;
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(
                2 * per_core_M * in0_block_w * tile_size_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, tile_size_bytes));

        tt::CBIndex src1_cb_index = tt::CBIndex::c_1;
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(
                2 * per_core_N * in0_block_w * tile_size_bytes, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, tile_size_bytes));

        tt::CBIndex src2_cb_index = tt::CBIndex::c_2;
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(
                per_core_N * per_core_M * tile_size_bytes, {{src2_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src2_cb_index, tile_size_bytes));

        tt::CBIndex dst_cb_index = tt::CBIndex::c_16;
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(per_core_N * per_core_M * tile_size_bytes, {{dst_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(dst_cb_index, tile_size_bytes));

        tt::CBIndex intermediate_cb_index = tt::CBIndex::c_24;
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(
                per_core_N * per_core_M * tile_size_bytes, {{intermediate_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(intermediate_cb_index, tile_size_bytes));

        std::vector<uint32_t> in0_sender_in1_sender_compile_time_args;
        TensorAccessorArgs(*srcA_dram_buffer).append_to(in0_sender_in1_sender_compile_time_args);
        TensorAccessorArgs(*srcB_dram_buffer).append_to(in0_sender_in1_sender_compile_time_args);
        TensorAccessorArgs(*srcC_dram_buffer).append_to(in0_sender_in1_sender_compile_time_args);
        auto read_in0_sender_in1_sender = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "jaksa2/kernels/dataflow/read_in0_sender_in1_sender.cpp",
            in0_sender_in1_sender,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = in0_sender_in1_sender_compile_time_args});

        std::vector<uint32_t> in0_reciever_in1_sender_compile_time_args;
        TensorAccessorArgs(*srcB_dram_buffer).append_to(in0_reciever_in1_sender_compile_time_args);
        TensorAccessorArgs(*srcC_dram_buffer).append_to(in0_reciever_in1_sender_compile_time_args);
        auto read_in0_reciever_in1_sender = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "jaksa2/kernels/dataflow/read_in0_reciever_in1_sender.cpp",
            in0_reciever_in1_sender,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = in0_reciever_in1_sender_compile_time_args});

        std::vector<uint32_t> in0_sender_in1_reciever_compile_time_args;
        TensorAccessorArgs(*srcA_dram_buffer).append_to(in0_sender_in1_reciever_compile_time_args);
        TensorAccessorArgs(*srcC_dram_buffer).append_to(in0_sender_in1_reciever_compile_time_args);
        auto read_in0_sender_in1_reciever = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "jaksa2/kernels/dataflow/read_in0_sender_in1_reciever.cpp",
            in0_sender_in1_reciever,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = in0_sender_in1_reciever_compile_time_args});

        std::vector<uint32_t> in0_reciever_in1_reciever_compile_time_args;
        TensorAccessorArgs(*srcC_dram_buffer).append_to(in0_reciever_in1_reciever_compile_time_args);
        auto read_in0_reciever_in1_reciever = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "jaksa2/kernels/dataflow/read_in0_reciever_in1_reciever.cpp",
            in0_reciever_in1_reciever,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = in0_reciever_in1_reciever_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "jaksa2/kernels/dataflow/write.cpp",
            all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_compile_time_args});

        std::vector<uint32_t> compute_compile_time_args = {
            Kt / in0_block_w,             // how many BLOCKS are in the K dimension
            per_core_M / out_subblock_h,  // number of SUBBLOCKS in a BLOCK vertically
            per_core_N / out_subblock_w,  // number of SUBBLOCKS in a BLOCK horizontally
            out_subblock_h,               // number of TILES in a SUBBLOCK vertically
            out_subblock_w,               // number of TILES in a SUBBLOCK horizontally
            in0_block_w};
        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "jaksa2/kernels/compute/compute.cpp",
            all_cores,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_compile_time_args});

        auto in0_mcast_sender_semaphore_id = CreateSemaphore(program, all_cores, INVALID);
        auto in0_mcast_receiver_semaphore_id = CreateSemaphore(program, all_cores, INVALID);
        auto in1_mcast_sender_semaphore_id = CreateSemaphore(program, all_cores, INVALID);
        auto in1_mcast_receiver_semaphore_id = CreateSemaphore(program, all_cores, INVALID);

        // runtime args
        for (int core_idx_y = 0; core_idx_y < num_cores_y; core_idx_y++) {
            for (int core_idx_x = 0; core_idx_x < num_cores_x; core_idx_x++) {
                CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};

                std::vector<uint32_t> in0_sender_in1_sender_reader_args = {
                    (std::uint32_t)srcA_dram_buffer->address(),
                    (std::uint32_t)srcB_dram_buffer->address(),
                    (std::uint32_t)srcC_dram_buffer->address(),

                    (std::uint32_t)in0_mcast_sender_semaphore_id,
                    (std::uint32_t)in0_mcast_receiver_semaphore_id,
                    (std::uint32_t)in1_mcast_sender_semaphore_id,
                    (std::uint32_t)in1_mcast_receiver_semaphore_id,

                    (std::uint32_t)in0_block_w,
                    (std::uint32_t)Kt / in0_block_w,  // how many BLOCKS are in the K dimension
                    (std::uint32_t)per_core_M,        // how many TILES are in a BLOCK vertically
                    (std::uint32_t)per_core_N,        // how many TILES are in a BLOCK horizontally

                    (std::uint32_t)num_cores_x - 1,  // in0 multicast number of destinations
                    (std::uint32_t)num_cores_y - 1,  // in1 multicast number of destinations

                    (std::uint32_t)Nt,  // number of TILES in N dimension

                    (std::uint32_t)out_subblock_h,  // number of TILES in a SUBBLOCK vertically
                    (std::uint32_t)out_subblock_w,  // number of TILES in a SUBBLOCK horizontally

                    (std::uint32_t)start_core_physical.x,  // physical coordinate x of starting core
                    (std::uint32_t)start_core_physical.y,  // physical coordinate y of starting core
                };

                std::vector<uint32_t> in0_sender_in1_reciever_reader_args = {
                    (std::uint32_t)srcA_dram_buffer->address(),
                    (std::uint32_t)srcC_dram_buffer->address(),

                    (std::uint32_t)in0_mcast_sender_semaphore_id,
                    (std::uint32_t)in0_mcast_receiver_semaphore_id,
                    (std::uint32_t)in1_mcast_sender_semaphore_id,
                    (std::uint32_t)in1_mcast_receiver_semaphore_id,

                    (std::uint32_t)in0_block_w,
                    (std::uint32_t)Kt / in0_block_w,  // how many BLOCKS are in the K dimension
                    (std::uint32_t)per_core_M,        // how many TILES are in a BLOCK vertically
                    (std::uint32_t)per_core_N,        // how many TILES are in a BLOCK horizontally

                    (std::uint32_t)num_cores_x - 1,  // in0 multicast number of destinations
                    (std::uint32_t)num_cores_y - 1,  // in1 multicast number of destinations

                    (std::uint32_t)Nt,  // number of TILES in N dimension

                    (std::uint32_t)core.y,  // this core's y coordinate in the core grid

                    (std::uint32_t)out_subblock_h,  // number of TILES in a SUBBLOCK vertically
                    (std::uint32_t)out_subblock_w,  // number of TILES in a SUBBLOCK horizontally

                    (std::uint32_t)start_core_physical.x,  // physical coordinate x of starting core
                    (std::uint32_t)start_core_physical.y,  // physical coordinate y of starting core
                };

                std::vector<uint32_t> in0_reciever_in1_sender_reader_args = {
                    (std::uint32_t)srcB_dram_buffer->address(),
                    (std::uint32_t)srcC_dram_buffer->address(),

                    (std::uint32_t)in0_mcast_sender_semaphore_id,
                    (std::uint32_t)in0_mcast_receiver_semaphore_id,
                    (std::uint32_t)in1_mcast_sender_semaphore_id,
                    (std::uint32_t)in1_mcast_receiver_semaphore_id,

                    (std::uint32_t)in0_block_w,
                    (std::uint32_t)Kt / in0_block_w,  // how many BLOCKS are in the K dimension
                    (std::uint32_t)per_core_M,        // how many TILES are in a BLOCK vertically
                    (std::uint32_t)per_core_N,        // how many TILES are in a BLOCK horizontally

                    (std::uint32_t)num_cores_x - 1,  // in0 multicast number of destinations
                    (std::uint32_t)num_cores_y - 1,  // in1 multicast number of destinations

                    (std::uint32_t)Nt,  // number of TILES in N dimension

                    (std::uint32_t)core.x,  // this core's x coordinate in the core grid

                    (std::uint32_t)out_subblock_h,  // number of TILES in a SUBBLOCK vertically
                    (std::uint32_t)out_subblock_w,  // number of TILES in a SUBBLOCK horizontally

                    (std::uint32_t)start_core_physical.x,  // physical coordinate x of starting core
                    (std::uint32_t)start_core_physical.y,  // physical coordinate y of starting core
                };

                std::vector<uint32_t> in0_reciever_in1_reciever_reader_args = {
                    (std::uint32_t)srcC_dram_buffer->address(),

                    (std::uint32_t)in0_mcast_sender_semaphore_id,
                    (std::uint32_t)in0_mcast_receiver_semaphore_id,
                    (std::uint32_t)in1_mcast_sender_semaphore_id,
                    (std::uint32_t)in1_mcast_receiver_semaphore_id,

                    (std::uint32_t)in0_block_w,
                    (std::uint32_t)Kt / in0_block_w,  // how many BLOCKS are in the K dimension
                    (std::uint32_t)per_core_M,        // how many TILES are in a BLOCK vertically
                    (std::uint32_t)per_core_N,        // how many TILES are in a BLOCK horizontally

                    (std::uint32_t)Nt,  // number of TILES in N dimension

                    (std::uint32_t)core.x,  // this core's x coordinate in the core grid
                    (std::uint32_t)core.y,  // this core's y coordinate in the core grid

                    (std::uint32_t)out_subblock_h,  // number of TILES in a SUBBLOCK vertically
                    (std::uint32_t)out_subblock_w,  // number of TILES in a SUBBLOCK horizontally

                    (std::uint32_t)start_core_physical.x,  // physical coordinate x of starting core
                    (std::uint32_t)start_core_physical.y,  // physical coordinate y of starting core
                };
                std::vector<uint32_t> mm_writer_args = {
                    (std::uint32_t)dst_dram_buffer->address(),

                    (std::uint32_t)per_core_M,  // how many TILES are in a BLOCK vertically
                    (std::uint32_t)per_core_N,  // how many TILES are in a BLOCK horizontally

                    (std::uint32_t)core.x,  // this core's x coordinate in the core grid
                    (std::uint32_t)core.y,  // this core's y coordinate in the core grid
                    (std::uint32_t)Nt,      // number of TILES in N dimension
                };

                if (core.x == 0 and core.y == 0) {
                    SetRuntimeArgs(program, read_in0_sender_in1_sender, core, in0_sender_in1_sender_reader_args);
                    SetRuntimeArgs(program, compute, core, {2});
                    SetRuntimeArgs(program, writer, core, mm_writer_args);
                } else if (core.x == 0 and core.y != 0) {
                    SetRuntimeArgs(program, read_in0_sender_in1_reciever, core, in0_sender_in1_reciever_reader_args);
                    SetRuntimeArgs(program, compute, core, {2});
                    SetRuntimeArgs(program, writer, core, mm_writer_args);
                } else if (core.x != 0 and core.y == 0) {
                    SetRuntimeArgs(program, read_in0_reciever_in1_sender, core, in0_reciever_in1_sender_reader_args);
                    SetRuntimeArgs(program, compute, core, {2});
                    SetRuntimeArgs(program, writer, core, mm_writer_args);
                } else {
                    SetRuntimeArgs(
                        program, read_in0_reciever_in1_reciever, core, in0_reciever_in1_reciever_reader_args);
                    SetRuntimeArgs(program, compute, core, {2});
                    SetRuntimeArgs(program, writer, core, mm_writer_args);
                }
            }
        }

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        fmt::print("FINISHED ALL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

        std::vector<bfloat16> result_vec;
        distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

        TT_FATAL(result_vec.size() == c_tensor_data.size(), "Result vector size mismatch");
        fmt::print("result size: {}\n", result_vec.size());
        fmt::print("result tile 1: {}\n", static_cast<float>(result_vec[0]));

        float total_expected = 0.0f;
        float total_actual = 0.0f;
        const float expected_val = 2.0f * 3.0f * K;
        for (size_t i = 0; i < result_vec.size(); ++i) {
            float actual = static_cast<float>(result_vec[i]);
            total_expected += expected_val;
            total_actual += actual;
        }
        fmt::print("expected / actual: {}\n", total_expected / total_actual);

        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());
        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
