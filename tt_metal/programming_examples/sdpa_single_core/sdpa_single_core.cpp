// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <bmm_op.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/core_coord.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

// Reference implementation of matrix multiplication.
// Array A is of size MxK, Array B is of size KxN, and the output C is of size MxN.
// The implementation is bare bones and does not include optimizations such as tiling or vectorization.
// This is intended to be used as a golden reference for testing the Metalium implementation.
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// void golden_sdpa(
//     const std::vector<bfloat16>& a,
//     const std::vector<bfloat16>& b,
//     std::vector<bfloat16>& output,
//     uint32_t M,
//     uint32_t N,
//     uint32_t K) {
//     std::vector<bfloat16> c_bf(M * N, 0);

//     for (int i = 0; i < M; i++) {
//         for (int j = 0; j < N; j++) {
//             std::uint32_t idx_c = j + (i * N);
//             std::uint32_t idx_a = i * K;
//             std::uint32_t idx_b = j;
//             float c_f = 0;
//             for (int k_m = 0; k_m < K; k_m++) {
//                 c_f += static_cast<float>(a[idx_a]) * static_cast<float>(b[idx_b]);
//                 idx_a += 1;
//                 idx_b += N;
//             }
//             output.at(idx_c) = bfloat16(c_f);
//         }
//     }
// }

void sdpa_single_core(
    const uint32_t Sq_chunk_t,
    const uint32_t Sk_chunk_t,
    const uint32_t head_dim_t,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // Set up mesh command queue, workload, device range, and program. This is a single-core example using core {0,0}.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};
    // Core range from x: [0, 0] to y: [0, 0] (single core at {0, 0})
    CoreCoord core({0, 0});

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    const MathFidelity math_fidelity = MathFidelity::HiFi2;
    const uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    
    const uint32_t q_buffer_size = 2 * Sq_chunk_t * head_dim_t * single_tile_size;
    const uint32_t kt_buffer_size = 2 * Sk_chunk_t * head_dim_t * single_tile_size;
    const uint32_t out_buffer_size = 2 * Sq_chunk_t * head_dim_t * single_tile_size;

    const uint32_t num_iter = 16;

    // We'll use these later on to check for correctness.
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    distributed::ReplicatedBufferConfig buffer_config_Q{.size = q_buffer_size};
    distributed::ReplicatedBufferConfig buffer_config_KT{.size = kt_buffer_size};
    distributed::ReplicatedBufferConfig buffer_config_out{.size = out_buffer_size};

    auto q_dram_buffer = distributed::MeshBuffer::create(buffer_config_Q, dram_config, mesh_device.get());
    auto kt_dram_buffer = distributed::MeshBuffer::create(buffer_config_KT, dram_config, mesh_device.get());
    auto out_dram_buffer = distributed::MeshBuffer::create(buffer_config_out, dram_config, mesh_device.get());
    
    const uint32_t q_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_q_config =
        CircularBufferConfig(q_buffer_size, {{q_cb_index, cb_data_format}})
            .set_page_size(q_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_q_config);

    const uint32_t kt_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_kt_config =
        CircularBufferConfig(kt_buffer_size, {{kt_cb_index, cb_data_format}})
            .set_page_size(kt_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_kt_config);
    
    const uint32_t qkt_cb_index = CBIndex::c_2;
    const uint32_t qkt_tiles = Sq_chunk_t * Sk_chunk_t;
    auto cb_qkt_config = CircularBufferConfig(qk_tiles * single_tile_size, {{qkt_cb_index, cb_data_format}})
                                  .set_page_size(qkt_cb_index, single_tile_size);
    CreateCircularBuffer(program, core_grid, cb_qkt_config);

    // Create the data movement kernels and the compute kernel
    std::vector<uint32_t> reader_compile_time_args = {
        Sq_chunk_t,
        Sk_chunk_t,
        head_dim_t,
        num_iter,
    };
    TensorAccessorArgs(*q_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*kt_dram_buffer).append_to(reader_compile_time_args);
    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sdpa_single_core/kernels/dataflow/reader.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args = {
        Sq_chunk_t,
        Sk_chunk_t,
        head_dim_t,
        num_iter,
    };
    TensorAccessorArgs(*out_dram_buffer).append_to(writer_compile_time_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sdpa_single_core/kernels/dataflow/writer.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    std::vector<uint32_t> compute_compile_time_args = {
        Sq_chunk_t,
        Sk_chunk_t,
        head_dim_t,
        num_iter,
    };
    tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "sdpa_single_core/kernels/compute/sdpa.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_compile_time_args});

    // Set kernel arguments
    uint32_t q_addr = q_dram_buffer->address();
    uint32_t kt_addr = kt_dram_buffer->address();
    uint32_t out_addr = out_dram_buffer->address();
    tt_metal::SetRuntimeArgs(program, reader_id, core, {q_addr, kt_addr});

    tt_metal::SetRuntimeArgs(program, writer_id, core, {out_addr});
    // NOTE: Note that we never set the runtime arguments for the compute kernel. This is because everything needed has
    // been set at compile time. The compute kernel does not need any runtime arguments to execute. And so we can skip
    // this step.

    // Upload the input data to the DRAM buffers, execute the kernels, wait for the result to be read into the output
    // buffer
    //distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, a, false);
    //distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, b, false);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    //distributed::EnqueueReadMeshBuffer(cq, output, dst_dram_buffer, true);
}

///////////////////////////////////////

int main() {
    bool pass = true;

    try {
        // Open device
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        constexpr uint32_t Sq_chunk_t = 8;
        constexpr uint32_t Sk_chunk_t = 16;
        constexpr uint32_t head_dim_t = 128 / TILE_WIDTH;

        // // input vectors with various ranges of values
        // std::mt19937 rng(std::random_device{}());
        // std::uniform_real_distribution<float> dist(0.f, 1.0f);
        // std::vector<bfloat16> src0_vec(M * K);
        // std::vector<bfloat16> src1_vec(K * N);

        // for (bfloat16& v : src0_vec) {
        //     v = bfloat16(dist(rng));
        // }
        // for (bfloat16& v : src1_vec) {
        //     v = bfloat16(dist(rng));
        // }

        // // Golden Matmul running on CPU so we can compare later
        // std::vector<bfloat16> golden_vec(M * N, 0);
        // golden_sdpa(src0_vec, src1_vec, golden_vec, M, N, K);

        // Tilize the input vectors to match the expected tiled layout for the device
        // The Tenstorrent hardware operates on data in 32x32 tiles rather than standard row-major format.
        // tilize_nfaces() converts the input matrices from row-major layout to the tiled layout expected by the device.
        // This transformation groups elements into 32x32 blocks and reorders them in memory so that each tile
        // (32x32 elements) is stored contiguously. This matches the native data access patterns of the matrix engine
        // and enables efficient operations on the accelerator.
        // src0_vec = tilize_nfaces(src0_vec, M, K);
        // src1_vec = tilize_nfaces(src1_vec, K, N);

        // Invoke the matrix multiplication on the device
        //std::vector<bfloat16> result_vec(M * N, 0);
        
        //sdpa_single_core(src0_vec, src1_vec, result_vec, false, M, N, K, mesh_device);
        sdpa_single_core(Sq_chunk_t, Sk_chunk_t, head_dim_t, mesh_device);

        // // Reverse the tilization to get the result in the row-major format that the CPU expects
        // result_vec = untilize_nfaces(result_vec, M, N);

        // fmt::print("Output vector of size {}\n", result_vec.size());

        // // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector
        // // This is a measure of how similar the two vectors are.
        // // A PCC close to 1 indicates that the two vectors are very similar.
        // float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        // fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
        // TT_FATAL(pearson > 0.97, "PCC not high enough. Result PCC: {}, Expected PCC: 0.97", pearson);

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

    TT_ASSERT(pass);

    return 0;
}
