// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;
using namespace ttnn;

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
    std::vector<bfloat16> c_bf(M * N, 0);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::uint32_t idx_c = j + (i * N);
            std::uint32_t idx_a = i * K;
            std::uint32_t idx_b = j;
            float c_f = 0;
            for (int k_m = 0; k_m < K; k_m++) {
                c_f += static_cast<float>(a[idx_a]) * static_cast<float>(b[idx_b]);
                idx_a += 1;
                idx_b += N;
            }
            output.at(idx_c) = static_cast<bfloat16>(c_f);
        }
    }
}

// Matrix multiplication using the accelerator.
// Input a and b as well as output are vectors of bfloat16. But in the tiled layout.
// The input a is of size MxK, input b is of size KxN, and the output c is of size MxN.
// For this function, M, N and N must be divisible by TILE_HEIGHT and TILE_WIDTH respectively as that is the native unit
// of computation on the accelerator.
// This version uses ttnn::Tensor instead of distributed::MeshBuffer.
void matmul_single_core_tensor(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // Set up mesh command queue, workload, device range, and program. This is a single-core example using core {0,0}.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};
    // Core range from x: [0, 0] to y: [0, 0] (i.e. single core at {0, 0})
    tt::tt_metal::CoreCoord core({0, 0});

    // Calcaulate the number of tiles for each dimension.
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    // Create ttnn::Tensor objects for the input and output data.
    // We use TILE layout as that's what the hardware expects for matmul operations.
    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    // Create tensors in device DRAM for src0 (MxK), src1 (KxN), and dst (MxN)
    TensorLayout tile_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig(BufferType::DRAM));
    TensorSpec src0_spec(Shape({M, K}), tile_layout);
    TensorSpec src1_spec(Shape({K, N}), tile_layout);
    TensorSpec dst_spec(Shape({M, N}), tile_layout);

    // Create device tensors from input data using Tensor::from_vector
    // This creates the tensors and transfers data to device in one step
    auto src0_tensor = Tensor::from_vector<bfloat16>(std::vector<bfloat16>(a), src0_spec, mesh_device.get());
    auto src1_tensor = Tensor::from_vector<bfloat16>(std::vector<bfloat16>(b), src1_spec, mesh_device.get());
    // Allocate output tensor on device (no initialization needed - hardware will zero it via tile_regs_acquire)
    auto dst_tensor = allocate_tensor_on_device(dst_spec, mesh_device.get());

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
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
            .set_page_size(output_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    // Get mesh buffers from tensors for use with TensorAccessorArgs and runtime args
    auto src0_mesh_buffer = src0_tensor.mesh_buffer();
    auto src1_mesh_buffer = src1_tensor.mesh_buffer();
    auto dst_mesh_buffer = dst_tensor.mesh_buffer();

    // Create the data movement kernels and the compute kernel
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_mesh_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_mesh_buffer).append_to(reader_compile_time_args);
    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_mesh_buffer).append_to(writer_compile_time_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Compile time arguments for the kernels
    // Note that these take effect at the kernel's compile time. Chaning these values will require recompilation of the
    // kernel. Having arguments at compile time allows the compiler to optimize the kernel for the specific use case.
    // Like applying loop unrolling, constant folding, etc.. resulting in a more efficient kernel.
    std::vector<uint32_t> compute_compile_time_args = {
        Mt,  // Mt
        Kt,  // Kt
        Nt   // Nt
    };
    tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/compute/mm.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_compile_time_args});

    // Set kernel arguments
    uint32_t src0_addr = src0_mesh_buffer->address();
    uint32_t src1_addr = src1_mesh_buffer->address();
    uint32_t dst_addr = dst_mesh_buffer->address();
    tt_metal::SetRuntimeArgs(program, reader_id, core, {src0_addr, src1_addr, Mt, Kt, Nt});

    tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_addr, Mt, Kt, Nt});
    // NOTE: Note that we never set the runtime arguments for the compute kernel. This is because everything needed has
    // been set at compile time. The compute kernel does not need any runtime arguments to execute. And so we can skip
    // this step.

    // Execute the kernels (data is already on device from Tensor::from_vector)
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);  // Wait for workload to complete

    // Read the result back from device using Tensor::to_vector
    output = dst_tensor.to_vector<bfloat16>();
}

///////////////////////////////////////

int main() {
    bool pass = true;

    try {
        // Open device
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

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

        // Invoke the matrix multiplication on the Tensix device
        std::vector<bfloat16> result_vec(M * N, 0);
        matmul_single_core_tensor(src0_vec, src1_vec, result_vec, M, N, K, mesh_device);

        fmt::print("Output vector of size {}\n", result_vec.size());

        // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector
        // This is a measure of how similar the two vectors are.
        // A PCC close to 1 indicates that the two vectors are very similar.
        float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
        TT_FATAL(pearson > 0.97, "PCC not high enough. Result PCC: {}, Expected PCC: 0.97", pearson);

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
