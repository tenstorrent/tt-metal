// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/work_split.hpp>
#include <bmm_op.hpp>
#include <tt-metalium/device.hpp>
#include <fmt/core.h>

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
// Reference implementation of matrix multiplication.
// Array A is of size MxK, Array B is of size KxN, and the output C is of size MxN.
// The implementation is bare bones and does not include optimizations such as tiling or vectorization.
// This is intended to be used as a golden reference for testing the Metalium implementation.
void golden_matmul(
    std::vector<bfloat16>& a,
    std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K) {
    std::uint32_t idx_c = 0;
    std::uint32_t idx_a = 0;
    std::uint32_t idx_b = 0;

    float c_f;
    float float_tmp;
    std::vector<bfloat16> c_bf(M * N, 0);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            idx_c = j + (i * N);
            idx_a = i * K;
            idx_b = j;
            c_f = 0;
            for (int k_m = 0; k_m < K; k_m++) {
                float_tmp = a[idx_a].to_float() * b[idx_b].to_float();
                c_f += float_tmp;
                idx_a += 1;
                idx_b += N;
            }
            output.at(idx_c) = bfloat16(c_f);
        }
    }
}

/**
 * @brief Multi-core matrix multiplication using SPMD (Single Program, Multiple Data) parallelization.
 *
 * Performs C = A * B matrix multiplication by distributing output tiles across multiple cores.
 * Each core runs the same program but works on different portions of the output matrix,
 * making this a simple and efficient parallelization scheme.
 *
 * The function uses three types of kernels running in parallel:
 * - Reader: Loads input matrix tiles from DRAM into circular buffers
 * - Compute: Performs tile-wise matrix multiplication (A_tile * B_tile = C_tile)
 * - Writer: Stores computed output tiles back to DRAM
 *
 * Work distribution is handled automatically - if output tiles don't divide evenly
 * across cores, some cores get one extra tile to balance the workload.
 *
 * @param a Input matrix A in row-major format (bfloat16 elements)
 * @param b Input matrix B in row-major format (bfloat16 elements)
 * @param output Output matrix C to store A*B result (bfloat16 elements)
 * @param M Number of rows in matrix A and output matrix C
 * @param N Number of columns in matrix B and output matrix C
 * @param K Number of columns in matrix A and rows in matrix B
 * @param device Target device for computation
 *
 * @note Matrix dimensions must be divisible by tile size (32x32) for this implementation
 * @note Uses circular buffers with 2 tiles for double-buffering to overlap compute and data movement
 */
void matmul_multi_core(
    std::vector<bfloat16>& a,
    std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    IDevice* device) {
    // Check if the configuration is valid - matrices must be divisible by tile dimensions
    TT_ASSERT(
        (M * N) % TILE_HW == 0,
        "Matrix dimensions M={} and N={} must be divisible by TILE_HW={} to use this matmul implementation",
        M,
        N,
        TILE_HW);

    // Setup the device and command queue for multi-core execution
    CommandQueue& cq = device->command_queue();
    Program program{};

    // Get the compute grid size to determine how many cores are available
    auto core_grid = device->compute_with_storage_grid_size();
    auto num_output_tiles_total = (M * N) / TILE_HW;

    // Use the split_work_to_cores utility function to distribute matrix multiplication work
    // across available cores for efficient SPMD (Single Program, Multiple Data) execution.
    // This function takes the total number of output tiles and available cores, then calculates
    // how to divide the work when it cannot be evenly distributed. It returns two groups of cores:
    // - Primary group: handles more tiles per core
    // - Secondary group: handles fewer tiles per core
    // The secondary group is empty if the work can be evenly distributed across all cores. This
    // approach minimizes workload imbalance between cores for optimal performance.
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
        split_work_to_cores(core_grid, num_output_tiles_total);

    // Extracting Matrix dimensions from input/output vectors and converting to tile coordinates.
    // The accelerator works with 32x32 tiles, so we need to convert from element dimensions
    // to tile dimensions for proper addressing and computation.
    const uint32_t Mt = M / TILE_HEIGHT;  // Number of tiles in M dimension
    const uint32_t Kt = K / TILE_WIDTH;   // Number of tiles in K dimension
    const uint32_t Nt = N / TILE_WIDTH;   // Number of tiles in N dimension

    // Create DRAM Buffers for input and output vectors.
    // We allocate DRAM buffers for the input matrices and output matrix.
    // Setting page_size to single_tile_size is the most common configuration for memory buffers in Metalium
    // as it is generic, works for most cases and achieves good performance.
    // Writing data from input vectors to source buffers.
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;  // 2 * 32 * 32 = 2048 bytes

    tt_metal::InterleavedBufferConfig dram_config_A{
        .device = device,
        .size = single_tile_size * Mt * Kt,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig dram_config_B{
        .device = device,
        .size = single_tile_size * Nt * Kt,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig dram_config_C{
        .device = device,
        .size = single_tile_size * Mt * Nt,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    auto src0_dram_buffer = CreateBuffer(dram_config_A);
    auto src1_dram_buffer = CreateBuffer(dram_config_B);
    auto dst_dram_buffer = CreateBuffer(dram_config_C);

    // Configure Circular Buffers
    // Circular buffers act as staging areas for data movement between DRAM and compute units.
    // Using 2 tiles per circular buffer to allow for double buffering (data movement can be reading from one tile while
    // the compute kernel is using the other tile). This number can be adjusted based on the use case, but generally
    // diminishing returns are observed after several tiles.
    // input tiles count is = 2 so one tile can be read while the other is being processed
    const auto cb_data_format = tt::DataFormat::Float16_b;
    uint32_t num_input_tiles = 2;
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,  // create on all cores
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,  // create on all cores
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,  // create on all cores
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, single_tile_size));

    // Create Kernels (Reader, Writer, Compute)
    // - Reader kernel: Handles reading input data from DRAM into circular buffers
    // - Writer kernel: Handles writing output data from circular buffers back to DRAM
    // - Compute kernel: Performs the actual matrix multiplication computation
    // All kernels run across all cores to enable parallel execution
    MathFidelity math_fidelity = MathFidelity::HiFi4;  // High fidelity math for accurate results
    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_multi_core/kernels/dataflow/reader_mm_output_tiles_partitioned.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {}});

    auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_multi_core/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = {}});

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_multi_core/kernels/compute/mm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = {}});

    // Set Runtime Arguments for Kernels
    // Each core needs to know which portion of the work it's responsible for. We are parallelizing across output
    // tiles - each core computes different output tiles. Runtime arguments can be changed between program executions
    // without recompilation.
    uint32_t work_offset = 0;
    auto work_groups = {std::make_pair(core_group_1, work_per_core1), std::make_pair(core_group_2, work_per_core2)};

    // Iterate through each work group and assign work to cores
    for (const auto& [ranges, work_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                // Set arguments for the reader kernel (data input)
                tt_metal::SetRuntimeArgs(
                    program,
                    reader_id,
                    core,
                    {src0_dram_buffer->address(),  // Address of matrix A in DRAM
                     src1_dram_buffer->address(),  // Address of matrix B in DRAM
                     Mt,                           // Number of tiles in M dimension
                     Kt,                           // Number of tiles in K dimension
                     Nt,                           // Number of tiles in N dimension
                     work_offset,                  // Starting offset for this core's work
                     work_per_core});              // Amount of work for this core

                // Set arguments for the writer kernel (data output)
                tt_metal::SetRuntimeArgs(
                    program, writer_id, core, {dst_dram_buffer->address(), work_per_core, work_offset});

                // Set arguments for the compute kernel
                tt_metal::SetRuntimeArgs(
                    program,
                    compute_kernel_id,
                    core,
                    {work_per_core,            // Amount of work for this core
                     Kt});                     // Number of tiles in K dimension for dot product
                work_offset += work_per_core;  // Update offset for next core
            }
        }
    }

    // Launch program & read in output buffer result into the host vector
    // 1. Upload input data to DRAM buffers
    // 2. Execute the program (all kernels run in parallel across cores)
    // 3. Read back the result from DRAM to host memory
    // The 'true' parameter in EnqueueReadBuffer ensures we wait for completion (so when the function
    // returns, the output vector is fully populated).
    EnqueueWriteBuffer(cq, src0_dram_buffer, a.data(), false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b.data(), false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_dram_buffer, output.data(), true);
}

///////////////////////////////////////

int main() {
    bool pass = true;

    try {
        constexpr int device_id = 0;
        IDevice* device = CreateDevice(device_id);

        // Create source data with specified matrix dimensions
        constexpr uint32_t M = 640;  // Number of rows in matrix A (user-defined)
        constexpr uint32_t N = 640;  // Number of columns in matrix B (user-defined)
        constexpr uint32_t K = 640;  // Inner dimension for multiplication (user-defined)

        // Ensure that the matrix dimensions are compatible with the tile size
        static_assert(M % TILE_HEIGHT == 0, "M must be divisible by TILE_HEIGHT");
        static_assert(N % TILE_WIDTH == 0, "N must be divisible by TILE_WIDTH");
        static_assert(K % TILE_WIDTH == 0, "K must be divisible by TILE_WIDTH");

        // Calculate matrix dimensions in tiles for the accelerator
        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t Nt = N / TILE_WIDTH;

        // Calculate buffer sizes needed for each matrix in bytes
        constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;  // 2 * 32 * 32 = 2048 bytes
        uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt;  // num_tiles of FP16_B

        // Create random input vectors for matrices A and B
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        std::vector<bfloat16> src0_vec(M * K, 0);  // Matrix A (MxK)
        std::vector<bfloat16> src1_vec(K * N, 0);  // Matrix B (KxN)
        // // Fill with random bfloat16 values
        for (bfloat16& v : src0_vec) {
            v = bfloat16(dist(rng));
        }
        for (bfloat16& v : src1_vec) {
            v = bfloat16(dist(rng));
        }

        // Golden Matmul running on CPU (Float) - reference implementation for verification
        std::vector<bfloat16> golden_vec(M * N, 0);
        golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K);

        // Input vector tilizing to match device expected tiled layout
        // The Tenstorrent hardware operates on data in 32x32 tiles rather than standard row-major format.
        // tilize_nfaces() converts the input matrices from row-major layout to the tiled layout expected by the device.
        // This transformation groups elements into 32x32 blocks and reorders them in memory so that each tile
        // (32x32 elements) is stored contiguously. This matches the native data access patterns of the matrix engine
        // and enables efficient operations on the accelerator.
        src0_vec = tilize_nfaces(src0_vec, M, K);
        src1_vec = tilize_nfaces(src1_vec, K, N);

        /* Calling the MatMul host program. Read in result into a host vector */
        std::vector<bfloat16> result_vec(dram_buffer_C_size / sizeof(bfloat16));
        matmul_multi_core(src0_vec, src1_vec, result_vec, M, N, K, device);
        // Reverse the tilization to get the result in the row-major format that the CPU expects
        result_vec = untilize_nfaces(result_vec, M, N);

        fmt::print("Output vector of size {}\n", result_vec.size());

        // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector
        // This is a measure of how similar the two vectors are.
        // A PCC close to 1 indicates that the two vectors are very similar.
        float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
        TT_FATAL(pearson > 0.97, "PCC not high enough. Result PCC: {}, Expected PCC: 0.97", pearson);

        pass &= CloseDevice(device);

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
