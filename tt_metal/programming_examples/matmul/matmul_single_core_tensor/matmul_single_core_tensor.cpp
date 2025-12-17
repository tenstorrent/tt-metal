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

// Define prefix path for kernel files if not set in the makefile.
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// clang-format off
/**
 * Reference implementation of matrix multiplication.
 * Array A is of size MxK, Array B is of size KxN, and the output C is of size MxN.
 * The implementation is bare bones and does not include optimizations such as tiling or vectorization.
 * This is intended to be used as a golden reference for testing the Metalium implementation.
 *
 * | Argument | Description                                                         |
 * |----------|---------------------------------------------------------------------|
 * | a        | Input matrix A in row-major format, size MxK                        |
 * | b        | Input matrix B in row-major format, size KxN                        |
 * | output   | Output matrix C in row-major format, size MxN (will be overwritten) |
 * | M        | Number of rows in matrix A and output matrix C                      |
 * | N        | Number of columns in matrix B and output matrix C                   |
 * | K        | Number of columns in matrix A and rows in matrix B                  |
 */
// clang-format on
void golden_matmul(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    const uint32_t M,
    const uint32_t N,
    const uint32_t K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // Compute C[i * N + j] += A[i * K + k] * B[k * N + j];
            std::uint32_t idx_c = (i * N) + j;
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

// clang-format off
/**
 * Structure to hold program-related state including device, program, workload, and execution context.
 *
 * | Member        | Description                                               |
 * |---------------|-----------------------------------------------------------|
 * | mesh_device   | Shared pointer to the mesh device                         |
 * | program       | Program object containing kernels, circular buffers, etc. |
 * | core          | Core coordinate where the program executes                |
 * | workload      | Workload object that bundles programs for execution       |
 * | device_range  | Range of devices where the program should execute         |
 * | cq            | Command queue for ordering operations on the mesh         |
 */
// clang-format on
struct ProgramState {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    Program program;
    tt::tt_metal::CoreCoord core;
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range;
    distributed::MeshCommandQueue& cq;

    ProgramState(
        std::shared_ptr<distributed::MeshDevice> mesh_device,
        Program program,
        tt::tt_metal::CoreCoord core,
        distributed::MeshWorkload workload,
        distributed::MeshCoordinateRange device_range,
        distributed::MeshCommandQueue& cq) :
        mesh_device(std::move(mesh_device)),
        program(std::move(program)),
        core(core),
        workload(std::move(workload)),
        device_range(std::move(device_range)),
        cq(cq) {}
};

// clang-format off
/**
 * Initialize program state for single-core execution.
 * Creates a unit mesh device, sets up command queue, workload, device range, and program.
 * This program uses only a single Tensix core at [0, 0].
 *
 * Return value: ProgramState
 */
// clang-format on
ProgramState init_program() {
    // Open device
    constexpr int device_id = 0;
    // In TT-Metal, all operations use a mesh abstraction - even a single device is represented as a 1x1 mesh.
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    // Ordering of operations in the mesh is managed by a command queue.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // MeshWorkload represents a collection of programs to be executed across the mesh.
    distributed::MeshWorkload workload;
    // Each program in the workload is associated with a range of devices where it should run.
    // In our case, we have a single program running on our entire (unit) mesh.
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    // This program uses only a single Tensix core at [0, 0].
    tt::tt_metal::CoreCoord core({0, 0});
    // Create a program object. A program is a collection of kernels that are executed on the device.
    // Kernels will be specified later.
    Program program = CreateProgram();

    return ProgramState(
        std::move(mesh_device), std::move(program), core, std::move(workload), std::move(device_range), cq);
}

// clang-format off
/**
 * Helper function to create a circular buffer with the specified number of tiles and CB index.
 * Internalizes the calculation of single_tile_size based on bfloat16 tile dimensions.
 *
 * Return value: void
 *
 * | Argument  | Description                                               |
 * |-----------|-----------------------------------------------------------|
 * | program   | The program to which the circular buffer will be added    |
 * | core      | Core coordinate where the circular buffer will be created |
 * | num_tiles | Number of tiles to allocate in the circular buffer        |
 * | cb_index  | Circular buffer index (c_0 to c_31)                       |
 */
// clang-format on
void create_cb(Program& program, const tt::tt_metal::CoreCoord& core, uint32_t num_tiles, tt::CBIndex cb_index) {
    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    CircularBufferConfig cb_config = CircularBufferConfig(num_tiles * single_tile_size, {{cb_index, cb_data_format}})
                                         .set_page_size(cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_config);
}

// clang-format off
/**
 * Matrix multiplication using the Tensix device.
 * The input a is of size MxK, input b is of size KxN, and the output c is of size MxN.
 * For this function, M, N and K must be divisible by TILE_HEIGHT and TILE_WIDTH respectively as that is the native unit
 * of computation on the accelerator.
 *
 * | Argument  | Description                                                         |
 * |-----------|---------------------------------------------------------------------|
 * | a         | Input matrix A in row-major format, size MxK                        |
 * | b         | Input matrix B in row-major format, size KxN                        |
 * | output    | Output matrix C in row-major format, size MxN (will be overwritten) |
 * | M         | Number of rows in matrix A and output matrix C                      |
 * | N         | Number of columns in matrix B and output matrix C                   |
 * | K         | Number of columns in matrix A and rows in matrix B                  |
 * | prog_state| Program state containing device, program, and execution context     |
 */
// clang-format on
void matmul_single_core_tensor(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    ProgramState& prog_state) {
    // Calculate the number of tiles for each dimension.
    const uint32_t Mt = M / TILE_HEIGHT;
    const uint32_t Kt = K / TILE_WIDTH;
    const uint32_t Nt = N / TILE_WIDTH;

    // Create ttnn::Tensor objects for the input and output data.
    // We use TILE layout as that's what the hardware expects for matmul operations.
    // Tensors allocated in device DRAM for src0 (MxK), src1 (KxN), and dst (MxN)
    TensorLayout tile_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig(BufferType::DRAM));
    TensorSpec src0_spec(Shape({M, K}), tile_layout);
    TensorSpec src1_spec(Shape({K, N}), tile_layout);
    TensorSpec dst_spec(Shape({M, N}), tile_layout);

    // Create device tensors from input data.
    // This creates the tensors and transfers data to device in one step.
    Tensor src0_tensor =
        Tensor::from_vector<bfloat16>(std::vector<bfloat16>(a), src0_spec, prog_state.mesh_device.get());
    Tensor src1_tensor =
        Tensor::from_vector<bfloat16>(std::vector<bfloat16>(b), src1_spec, prog_state.mesh_device.get());
    // Allocate output tensor on device (no initialization needed - kernel will zero it via tile_regs_acquire).
    Tensor dst_tensor = allocate_tensor_on_device(dst_spec, prog_state.mesh_device.get());

    // Create circular buffers for the input and output data.
    // Using 2 tiles per circular buffer to allow for double buffering (data movement can be reading from one tile while
    // the compute kernel is using the other tile). This number can be adjusted based on the use case, But generally
    // there are diminishing returns observed after several tiles.
    constexpr uint32_t num_input_tiles = 2;
    // There are 32 circular buffers (c_0 - c_31) on the device. We can use any of them, as long as they are not already
    // in use. Kernel code is responsible for using the correct circular buffer for the input and output data (e.g.
    // reader kernel reads data into c_0 and c_1, while the compute kernel reads data from these same buffers).
    create_cb(prog_state.program, prog_state.core, num_input_tiles, CBIndex::c_0);
    create_cb(prog_state.program, prog_state.core, num_input_tiles, CBIndex::c_1);

    constexpr uint32_t num_output_tiles = 2;
    // Compute kernel will write output data to c_16, which will be consumed by the writer kernel.
    // c_16 chosen arbitrarily (e.g. to leave c_2-c_15 free for other potential inputs when code is extended in the
    // future).
    create_cb(prog_state.program, prog_state.core, num_output_tiles, tt::CBIndex::c_16);

    // Get mesh buffers from tensors for use with TensorAccessorArgs and runtime args.
    auto src0_mesh_buffer = src0_tensor.mesh_buffer();
    auto src1_mesh_buffer = src1_tensor.mesh_buffer();
    auto dst_mesh_buffer = dst_tensor.mesh_buffer();

    // Create a reader kernel to read data from DRAM into circular buffers.
    std::vector<uint32_t> reader_compile_time_args;
    // The TensorAccessor object abstracts away physical details of data distribution across banks.
    // Kernels can use TensorAccessorArgs to access the data in a unified way, regardless of the physical distribution.
    TensorAccessorArgs(*src0_mesh_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_mesh_buffer).append_to(reader_compile_time_args);
    // There are two data movement processors (RISCVs) in each Tensix core. We pick one to read data from DRAM into
    // circular buffers and the other to write result from circular buffer to DRAM. Which one is used for what doesn't
    // impact functionality or performance.
    KernelHandle reader_id = tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp",
        prog_state.core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});

    // Create a writer kernel to write result from circular buffers to DRAM.
    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_mesh_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_id = tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp",
        prog_state.core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args});

    // Compile time arguments for the compute kernel.
    // Note that these take effect at the kernel's compile time, which is JIT compile done at program creation time.
    // Having arguments at compile time allows the compiler to optimize the kernel for the specific use case
    // (e.g. apply loop unrolling, constant folding, etc.), resulting in a more efficient kernel.
    std::vector<uint32_t> compute_compile_time_args = {
        Mt,  // Mt
        Kt,  // Kt
        Nt   // Nt
    };
    // Observe that the compute kernel is
    tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/compute/mm.cpp",
        prog_state.core,
        tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    // Set kernel arguments
    uint32_t src0_addr = src0_mesh_buffer->address();
    uint32_t src1_addr = src1_mesh_buffer->address();
    uint32_t dst_addr = dst_mesh_buffer->address();
    tt_metal::SetRuntimeArgs(prog_state.program, reader_id, prog_state.core, {src0_addr, src1_addr, Mt, Kt, Nt});

    tt_metal::SetRuntimeArgs(prog_state.program, writer_id, prog_state.core, {dst_addr, Mt, Kt, Nt});
    // NOTE: Note that we never set the runtime arguments for the compute kernel. This is because everything needed has
    // been set at compile time. The compute kernel does not need any runtime arguments to execute, so we can skip
    // this step.

    // Execute the kernels (data is already on device from Tensor::from_vector)
    prog_state.workload.add_program(prog_state.device_range, std::move(prog_state.program));
    distributed::EnqueueMeshWorkload(prog_state.cq, prog_state.workload, true);  // Wait for workload to complete

    // Read the result back from device using Tensor::to_vector
    output = dst_tensor.to_vector<bfloat16>();
}

///////////////////////////////////////

// clang-format off
/**
 * Main function that demonstrates single-core matrix multiplication using ttnn::Tensor API.
 * Creates test data, runs golden reference implementation on CPU, executes matmul on Tensix device,
 * and verifies results using Pearson correlation coefficient.
 *
 * Return value: int (0 on success, non-zero on failure)
 */
// clang-format on
int main() {
    bool pass = true;

    try {
        // Initialize program state (includes device creation)
        ProgramState prog_state = init_program();

        // parameters for the matrix multiplication
        constexpr uint32_t M = 640;  // user-defined
        constexpr uint32_t N = 640;  // user-defined
        constexpr uint32_t K = 640;  // user-defined

        static_assert(M % TILE_HEIGHT == 0, "M must be divisible by TILE_HEIGHT");
        static_assert(N % TILE_WIDTH == 0, "N must be divisible by TILE_WIDTH");
        static_assert(K % TILE_WIDTH == 0, "K must be divisible by TILE_WIDTH");

        // Use a fixed seed for reproducible results. Change this value to get different random sequences.
        constexpr uint32_t rng_seed = 42;
        std::mt19937 rng(rng_seed);
        // Input vectors with random values in the range [0, 1).
        std::uniform_real_distribution<float> dist(0.f, 1.0f);
        std::vector<bfloat16> src0_vec(M * K);
        std::vector<bfloat16> src1_vec(K * N);

        for (bfloat16& v : src0_vec) {
            v = static_cast<bfloat16>(dist(rng));
        }
        for (bfloat16& v : src1_vec) {
            v = static_cast<bfloat16>(dist(rng));
        }

        // Golden Matmul running on CPU so we can verify Tensix result.
        std::vector<bfloat16> golden_vec(M * N, 0);
        golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K);

        // Invoke the matrix multiplication on the Tensix device
        std::vector<bfloat16> result_vec(M * N);
        matmul_single_core_tensor(src0_vec, src1_vec, result_vec, M, N, K, prog_state);

        fmt::print("Output vector of size {}\n", result_vec.size());

        // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector
        // This is a measure of how similar the two vectors are.
        // A PCC close to 1 indicates that the two vectors are very similar.
        float pcc = check_bfloat16_vector_pcc(golden_vec, result_vec);
        fmt::print("Metalium vs Golden -- PCC = {}\n", pcc);
        constexpr float expected_pcc = 0.97;
        TT_FATAL(pcc >= expected_pcc, "PCC not high enough. Result PCC: {}, Expected PCC: {}", pcc, expected_pcc);

        pass &= prog_state.mesh_device->close();

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
