// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
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
 * Reference implementation of element-wise addition.
 * This is intended to be used as a golden reference for testing the Metalium implementation.
 *
 * | Argument | Description                         |
 * |----------|-------------------------------------|
 * | a        | Input vector A                      |
 * | b        | Input vector B                      |
 * | output   | Output vector (will be overwritten) |
 */
// clang-format on
void golden_add(const std::vector<bfloat16>& a, const std::vector<bfloat16>& b, std::vector<bfloat16>& output) {
    TT_FATAL(a.size() == b.size(), "Input vectors must have the same size");
    TT_FATAL(output.size() == a.size(), "Output vector must have the same size as input vectors");
    for (size_t i = 0; i < a.size(); i++) {
        output[i] = static_cast<bfloat16>(static_cast<float>(a[i]) + static_cast<float>(b[i]));
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
    const uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    const tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    CircularBufferConfig cb_config = CircularBufferConfig(num_tiles * single_tile_size, {{cb_index, cb_data_format}})
                                         .set_page_size(cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_config);
}

// clang-format off
/**
 * Element-wise addition using the Tensix device.
 * Performs element-wise addition of two input vectors and stores the result in the output vector.
 * All vectors must have the same size, which must be divisible by TILE_HEIGHT * TILE_WIDTH.
 *
 * | Argument  | Description                                                         |
 * |-----------|---------------------------------------------------------------------|
 * | a         | Input vector A                                                      |
 * | b         | Input vector B                                                      |
 * | output    | Output vector (will be overwritten)                                 |
 * | prog_state| Program state containing device, program, and execution context     |
 */
// clang-format on
void eltwise_add_tensor(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    ProgramState& prog_state) {
    TT_FATAL(a.size() == b.size(), "Input vectors must have the same size");
    TT_FATAL(output.size() == a.size(), "Output vector must have the same size as input vectors");

    const uint32_t total_elements = static_cast<uint32_t>(a.size());
    const uint32_t elements_per_tile = TILE_HEIGHT * TILE_WIDTH;
    TT_FATAL(total_elements % elements_per_tile == 0, "Total elements must be divisible by elements per tile");
    const uint32_t n_tiles = total_elements / elements_per_tile;

    // Create ttnn::Tensor objects for the input and output data.
    // We use TILE layout as that's what the hardware expects for element-wise operations.
    // For TILE layout with sequential tiles, we need a 2D shape where each tile is 32x32.
    // Reshape as (n_tiles * TILE_HEIGHT, TILE_WIDTH) so tiles are laid out sequentially.
    TensorLayout tile_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig(BufferType::DRAM));
    TensorSpec src0_spec(Shape({n_tiles * TILE_HEIGHT, TILE_WIDTH}), tile_layout);
    TensorSpec src1_spec(Shape({n_tiles * TILE_HEIGHT, TILE_WIDTH}), tile_layout);
    TensorSpec dst_spec(Shape({n_tiles * TILE_HEIGHT, TILE_WIDTH}), tile_layout);

    // Create device tensors from input data.
    // This creates the tensors and transfers data to device in one step.
    Tensor src0_tensor =
        Tensor::from_vector<bfloat16>(std::vector<bfloat16>(a), src0_spec, prog_state.mesh_device.get());
    Tensor src1_tensor =
        Tensor::from_vector<bfloat16>(std::vector<bfloat16>(b), src1_spec, prog_state.mesh_device.get());
    // Allocate output tensor on device (no initialization needed - kernel will handle it).
    Tensor dst_tensor = allocate_tensor_on_device(dst_spec, prog_state.mesh_device.get());

    // Create circular buffers for the input and output data.
    // Using 2 tiles per circular buffer to allow for double buffering (data movement can be reading from one tile while
    // the compute kernel is using the other tile). This number can be adjusted based on the use case, but generally
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

    // Create the reader, writer and compute kernels. The kernels do the following:
    // * Reader: Reads data from the DRAM buffer and pushes it into the circular buffer.
    // * Compute: Waits for data to be available in the circular buffer, pops it, adds the two inputs together and
    //            pushes the result into the output circular buffer.
    // * Writer: Waits for data to be available in the output circular buffer, pops it and writes it back into DRAM.
    // These kernels work together to form a pipeline. The reader reads data from the DRAM buffer and makes them
    // available in the compute kernel. The compute kernel does math and pushes the result into the writer kernel. The
    // writer kernel writes the result back to DRAM.
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_mesh_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_mesh_buffer).append_to(reader_compile_time_args);
    KernelHandle reader_id = tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "lab_eltwise_binary/kernels/dataflow/read_tiles.cpp",
        prog_state.core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_mesh_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_id = tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "lab_eltwise_binary/kernels/dataflow/write_tiles.cpp",
        prog_state.core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args});

    KernelHandle compute_id = tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "lab_eltwise_binary/kernels/compute/tiles_add.cpp",
        prog_state.core,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4});  // There's different math fidelity modes (for the tensor engine)
    // that trade off performance for accuracy. HiFi4 is the most accurate
    // mode. The other modes are HiFi3, HiFi2, HiFi1 and LoFi. The
    // difference between them is the number of bits used during computation.

    // Set the runtime arguments for the kernels.
    tt_metal::SetRuntimeArgs(
        prog_state.program,
        reader_id,
        prog_state.core,
        {src0_mesh_buffer->address(), src1_mesh_buffer->address(), n_tiles});
    tt_metal::SetRuntimeArgs(prog_state.program, writer_id, prog_state.core, {dst_mesh_buffer->address(), n_tiles});
    tt_metal::SetRuntimeArgs(prog_state.program, compute_id, prog_state.core, {n_tiles});

    // Execute the kernels (data is already on device from Tensor::from_vector)
    prog_state.workload.add_program(prog_state.device_range, std::move(prog_state.program));
    distributed::EnqueueMeshWorkload(prog_state.cq, prog_state.workload, true);  // Wait for workload to complete

    // Read the result back from device using Tensor::to_vector
    output = dst_tensor.to_vector<bfloat16>();
}

///////////////////////////////////////

// clang-format off
/**
 * Main function that demonstrates single-core element-wise addition using ttnn::Tensor API.
 * Creates test data, runs golden reference implementation on CPU, executes addition on Tensix device,
 * and verifies results.
 *
 * Return value: int (0 on success, non-zero on failure)
 */
// clang-format on
int main() {
    bool pass = true;

    try {
        // Initialize program state (includes device creation)
        ProgramState prog_state = init_program();

        // Define some constants that will be used throughout the program.
        // * Processing 64 tiles
        // * Each tile is 32x32 elements
        // * Each element is a bfloat16 (2 bytes)
        constexpr uint32_t n_tiles = 64;
        constexpr uint32_t elements_per_tile = TILE_WIDTH * TILE_HEIGHT;
        constexpr uint32_t total_elements = n_tiles * elements_per_tile;

        // Use a fixed seed for reproducible results. Change this value to get different random sequences.
        constexpr uint32_t rng_seed = 42;
        std::mt19937 rng(rng_seed);
        // Input vectors with random values in the range [0, 1).
        std::uniform_real_distribution<float> dist(0.f, 1.0f);
        std::vector<bfloat16> src0_vec(total_elements);
        for (bfloat16& v : src0_vec) {
            v = static_cast<bfloat16>(dist(rng));
        }

        // src1 is a vector of bfloat16 values initialized to -1.0f.
        constexpr float val_to_add = -1.0f;
        std::vector<bfloat16> src1_vec(total_elements, static_cast<bfloat16>(val_to_add));

        // Golden addition running on CPU so we can verify Tensix result.
        std::vector<bfloat16> golden_vec(total_elements);
        golden_add(src0_vec, src1_vec, golden_vec);

        // Invoke the element-wise addition on the Tensix device
        std::vector<bfloat16> result_vec(total_elements);
        eltwise_add_tensor(src0_vec, src1_vec, result_vec, prog_state);

        fmt::print("Output vector of size {}\n", result_vec.size());

        // Validate results
        constexpr float eps = 1e-2f;  // loose tolerance because of the nature of bfloat16
        TT_FATAL(result_vec.size() == golden_vec.size(), "Result vector size mismatch");
        for (size_t i = 0; i < result_vec.size(); ++i) {
            const float expected = static_cast<float>(golden_vec[i]);
            const float actual = static_cast<float>(result_vec[i]);

            if (std::abs(expected - actual) > eps) {
                pass = false;
                fmt::print(stderr, "Result mismatch at index {}: expected {}, got {}\n", i, expected, actual);
            }
        }

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
