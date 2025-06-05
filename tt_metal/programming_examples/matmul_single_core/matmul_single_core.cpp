// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/command_queue.hpp>
#include <matmul_common/bmm_op.hpp>
#include <tt-metalium/device.hpp>
#include "tt-metalium/core_coord.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

// Reference implementation of matrix multiplication.
// Array A is of size MxK, Array B is of size KxN, and the output C is of size MxN.
// The implementation is bare bones and does not include optimizations such as tiling or vectorization.
// This is intended to be used as a golden reference for testing the Metalium implementation.
void golden_matmul(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K) {
    std::vector<bfloat16> c_bf(M * N, 0);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::uint32_t idx_c = j + (i * N);
            std::uint32_t idx_a = i * K;
            std::uint32_t idx_b = j;
            float c_f = 0;
            for (int k_m = 0; k_m < K; k_m++) {
                c_f += a[idx_a].to_float() * b[idx_b].to_float();
                idx_a += 1;
                idx_b += N;
            }
            output.at(idx_c) = bfloat16(c_f);
        }
    }
}

// Matrix multiplication using the accelerator.
// Input a and b as well as output are vectors of bfloat16. But in the tiled layout.
// The input a is of size MxK, input b is of size KxN, and the output c is of size MxN.
// For this function, M, N and N must be divisible by TILE_HEIGHT and TILE_WIDTH respectively as that is the native unit
// of computation on the accelerator.
void matmul_single_core(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    bool bcast_batch,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    IDevice* device) {
    // Setup the device and command queue. This is a single cored example, so we will use the first core {0, 0}.
    CommandQueue& cq = device->command_queue();
    Program program{};
    // Core range from x: [0, 0] to y: [0, 0] (single core at {0, 0})
    CoreCoord core({0, 0});

    // Calcaulate the number of tiles for each dimension.
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    // Create DRAM buffers for the input and output data.
    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    // We allocate DRAM buffers for the input and output data.
    // Setting page_size to single_tile_size is the most common configuration for memory buffers in Metalium
    // as it is generic, works for most cases and achieves good performance.
    tt_metal::InterleavedBufferConfig dram_config_A{
        .device = device,
        .size = sizeof(bfloat16) * a.size(),
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig dram_config_B{
        .device = device,
        .size = sizeof(bfloat16) * b.size(),
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig dram_config_C{
        .device = device,
        .size = sizeof(bfloat16) * output.size(),
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config_A);
    std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config_B);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config_C);

    // Create circular buffers for the input and output data.
    // Using 2 tiles per circular buffer to allow for double buffering (data movement can be reading from one tile while
    // the compute kernel is using the other tile). This number can be adjusted based on the use case. But geberally
    // diminishing returns observed after several tiles.
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
            .set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    // Create the data movement kernels and the compute kernel
    auto reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Compile time arguments for the kernels
    // Note that these take effect at the kernel's compile time. Chaning these values will require recompilation of the
    // kernel. Having arguments at compile time allows the compiler to optimize the kernel for the specific use case.
    // Like applying loop unrolling, constant folding, etc.. resulting in a more efficient kernel.
    std::vector<uint32_t> compute_compile_time_args = {
        Mt,  // Mt
        Kt,  // Kt
        Nt   // Nt
    };
    auto matmul_single_core_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_single_core/kernels/compute/mm.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_compile_time_args});

    // Set kernel arguments
    uint32_t src0_addr = src0_dram_buffer->address();
    uint32_t src1_addr = src1_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();
    tt_metal::SetRuntimeArgs(program, reader_id, core, {src0_addr, src1_addr, Mt, Kt, Nt});

    tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_addr, Mt, Kt, Nt});
    // NOTE: Note that we never set the runtime arguments for the compute kernel. This is because everything needed has
    // been set at compile time. The compute kernel does not need any runtime arguments to execute. And so we can skip
    // this step.

    // Upload the input data to the DRAM buffers, execute the kernels, wait for the result to be read into the output
    // buffer
    EnqueueWriteBuffer(cq, src0_dram_buffer, a.data(), false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b.data(), false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_dram_buffer, output.data(), true);
}

///////////////////////////////////////

int main() {
    bool pass = true;

    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    try {
        // Open device
        constexpr int device_id = 0;
        IDevice* device = CreateDevice(device_id);

        // parameters for the matrix multiplication
        constexpr uint32_t M = 640;  // user-defined
        constexpr uint32_t N = 640;  // user-defined
        constexpr uint32_t K = 640;  // user-defined

        static_assert(M % TILE_HEIGHT == 0, "M must be divisible by TILE_HEIGHT");
        static_assert(N % TILE_WIDTH == 0, "N must be divisible by TILE_WIDTH");
        static_assert(K % TILE_WIDTH == 0, "K must be divisible by TILE_WIDTH");

        // input vectors with various ranges of values
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.f, 1.0f);
        std::vector<bfloat16> src0_vec(M * K);
        std::vector<bfloat16> src1_vec(K * N);

        for (bfloat16& v : src0_vec) {
            v = bfloat16(dist(rng));
        }
        for (bfloat16& v : src1_vec) {
            v = bfloat16(dist(rng));
        }

        // Golden Matmul running on CPU so we can compare later
        std::vector<bfloat16> golden_vec(M * N, 0);
        golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K);

        // Tilize the input vectors to match the expected tiled layout for the device
        // The Tenstorrent hardware operates on data in 32x32 tiles rather than standard row-major format.
        // tilize_nfaces() converts the input matrices from row-major layout to the tiled layout expected by the device.
        // This transformation groups elements into 32x32 blocks and reorders them in memory so that each tile
        // (32x32 elements) is stored contiguously. This matches the native data access patterns of the matrix engine
        // and enables efficient operations on the accelerator.
        src0_vec = tilize_nfaces(src0_vec, M, K);
        src1_vec = tilize_nfaces(src1_vec, K, N);

        // Invoke the matrix multiplication on the device
        std::vector<bfloat16> result_vec(M * N, 0);
        matmul_single_core(src0_vec, src1_vec, result_vec, false, M, N, K, device);
        // Reverse the tilization to get the result in the row-major format that the CPU expects
        result_vec = untilize_nfaces(result_vec, M, N);

        log_info(tt::LogVerif, "Output vector of size {}", result_vec.size());

        // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector
        // This is a measure of how similar the two vectors are.
        // A PCC close to 1 indicates that the two vectors are very similar.
        float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        log_info(tt::LogVerif, "Metalium vs Golden -- PCC = {}", pearson);
        TT_FATAL(pearson > 0.97, "PCC not high enough. Result PCC: {}, Expected PCC: 0.97", pearson);

        pass &= CloseDevice(device);

    } catch (const std::exception& e) {
        log_error(tt::LogTest, "Test failed with exception!");
        log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
