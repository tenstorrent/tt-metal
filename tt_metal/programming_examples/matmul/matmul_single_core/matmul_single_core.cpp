// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/ez/ez.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <bmm_op.hpp>

#include <cstdint>
#include <vector>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::ez;

// Reference implementation of matrix multiplication.
// Array A is of size MxK, Array B is of size KxN, and the output C is of size MxN.
// The implementation is bare bones and does not include optimizations such as tiling or vectorization.
// This is intended to be used as a golden reference for testing the Metalium implementation.
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

void golden_matmul(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float c_f = 0;
            for (uint32_t k_m = 0; k_m < K; k_m++) {
                c_f += static_cast<float>(a[i * K + k_m]) * static_cast<float>(b[k_m * N + j]);
            }
            output.at(j + i * N) = bfloat16(c_f);
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
    uint32_t M,
    uint32_t N,
    uint32_t K,
    DeviceContext& ctx) {
    // This is a single-core example using core {0,0}.
    CoreCoord core({0, 0});

    // Calculate the number of tiles for each dimension.
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    // Create DRAM buffers for the input and output data.
    // A tile on Tenstorrent is 32x32 elements; with BFloat16, each tile occupies 2048 bytes.
    // We allocate DRAM buffers for the input and output data (replicated per device across the mesh).
    // Setting page_size to single_tile_size is the most common configuration for memory buffers in Metalium
    // as it is generic, works for most cases and achieves good performance.
    auto src0_dram_buffer = ctx.dram_tile_buffer(Mt * Kt);
    auto src1_dram_buffer = ctx.dram_tile_buffer(Kt * Nt);
    auto dst_dram_buffer = ctx.dram_tile_buffer(Mt * Nt);

    // Create circular buffers for the input and output data.
    // Using 2 tiles per circular buffer to allow for double buffering (data movement can be reading from one tile while
    // the compute kernel is using the other tile). This number can be adjusted based on the use case. But generally
    // diminishing returns observed after several tiles.
    constexpr uint32_t num_input_tiles = 2;
    constexpr uint32_t num_output_tiles = 2;

    auto builder = ProgramBuilder(core);
    builder.cb(tt::CBIndex::c_0, num_input_tiles)
        .cb(tt::CBIndex::c_1, num_input_tiles)
        .cb(tt::CBIndex::c_16, num_output_tiles);

    // Create the data movement kernels and the compute kernel.
    // Reader: loads A and B tiles from DRAM into circular buffers c_0 and c_1.
    // Writer: stores computed output tiles from c_16 back to DRAM.
    // The EZ API auto-generates TensorAccessorArgs from the buffer lists.
    auto& reader_ref = builder.reader(
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp",
        {src0_dram_buffer, src1_dram_buffer});
    auto& writer_ref = builder.writer(
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp",
        {dst_dram_buffer});

    // Compile time arguments for the compute kernel.
    // Note that these take effect at the kernel's compile time. Changing these values will require recompilation of the
    // kernel. Having arguments at compile time allows the compiler to optimize the kernel for the specific use case.
    // Like applying loop unrolling, constant folding, etc.. resulting in a more efficient kernel.
    builder.compute(
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/compute/mm.cpp",
        MathFidelity::HiFi4,
        {Mt, Kt, Nt});

    // Set kernel runtime arguments.
    reader_ref.runtime_args({src0_dram_buffer->address(), src1_dram_buffer->address(), Mt, Kt, Nt});
    writer_ref.runtime_args({dst_dram_buffer->address(), Mt, Kt, Nt});
    // NOTE: We never set the runtime arguments for the compute kernel. Everything needed has
    // been set at compile time. The compute kernel does not need any runtime arguments to execute.

    // Upload the input data to the DRAM buffers, execute the kernels, read back the result.
    ctx.write(src0_dram_buffer, a);
    ctx.write(src1_dram_buffer, b);
    ctx.run(builder.build());
    output = ctx.read<bfloat16>(dst_dram_buffer);
}

///////////////////////////////////////

int main() {
    bool pass = true;

    try {
        // Open device.
        // DeviceContext wraps MeshDevice creation, command queue, and teardown in RAII.
        constexpr int device_id = 0;
        DeviceContext ctx(device_id);

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

        // Tilize the input vectors to match the expected tiled layout for the device.
        // The Tenstorrent hardware operates on data in 32x32 tiles rather than standard row-major format.
        // tilize_nfaces() converts the input matrices from row-major layout to the tiled layout expected by the device.
        // This transformation groups elements into 32x32 blocks and reorders them in memory so that each tile
        // (32x32 elements) is stored contiguously. This matches the native data access patterns of the matrix engine
        // and enables efficient operations on the accelerator.
        src0_vec = tilize_nfaces(src0_vec, M, K);
        src1_vec = tilize_nfaces(src1_vec, K, N);

        // Invoke the matrix multiplication on the device
        std::vector<bfloat16> result_vec(M * N, 0);
        matmul_single_core(src0_vec, src1_vec, result_vec, M, N, K, ctx);
        // Reverse the tilization to get the result in the row-major format that the CPU expects
        result_vec = untilize_nfaces(result_vec, M, N);

        fmt::print("Output vector of size {}\n", result_vec.size());

        // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector.
        // This is a measure of how similar the two vectors are.
        // A PCC close to 1 indicates that the two vectors are very similar.
        float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
        TT_FATAL(pearson > 0.97, "PCC not high enough. Result PCC: {}, Expected PCC: 0.97", pearson);

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
