// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
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
                float_tmp = static_cast<float>(a[idx_a]) * static_cast<float>(b[idx_b]);
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
 * - Reader: Loads input matrix tiles from DRAM into dataflow buffers
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
 * @param mesh_device Target mesh device (1x1 or larger) for computation
 *
 * @note Matrix dimensions must be divisible by tile size (32x32) for this implementation
 * @note Uses dataflow buffers with 2 tiles for double-buffering to overlap compute and data movement
 */
void matmul_multi_core(
    std::vector<bfloat16>& a,
    std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // Check if the configuration is valid - matrices must be divisible by tile dimensions
    TT_ASSERT(
        (M * N) % TILE_HW == 0,
        "Matrix dimensions M={} and N={} must be divisible by TILE_HW={} to use this matmul implementation",
        M,
        N,
        TILE_HW);

    // Set up mesh command queue, workload and device range for multi-core execution
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // Get the compute grid size to determine how many cores are available
    auto core_grid = mesh_device->compute_with_storage_grid_size();
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

    // Create DRAM tensors for input and output matrices (replicated per device across the mesh).
    // We allocate DRAM tensors for the input matrices and output matrix.
    // Setting the page size to single_tile_size is the most common configuration for memory buffers in Metalium
    // as it is generic, works for most cases and achieves good performance.
    // Each row of a row-major tensor is one page, so a row of one tile's worth of elements gives a page size of one
    // tile.
    // Writing data from input vectors to source tensors.
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;  // 2 * 32 * 32 = 2048 bytes
    constexpr uint32_t elements_per_tile = single_tile_size / sizeof(bfloat16);

    auto make_dram_tensor_spec = [&](uint32_t num_tiles) {
        return TensorSpec(
            Shape({num_tiles, elements_per_tile}),
            TensorLayout(
                DataType::BFLOAT16,
                PageConfig(Layout::ROW_MAJOR),
                MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
    };
    TensorSpec src0_tensor_spec = make_dram_tensor_spec(Mt * Kt);
    TensorSpec src1_tensor_spec = make_dram_tensor_spec(Nt * Kt);
    TensorSpec dst_tensor_spec = make_dram_tensor_spec(Mt * Nt);

    MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, src0_tensor_spec);
    MeshTensor src1_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, src1_tensor_spec);
    MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dst_tensor_spec);
    // Each tensor is a mesh-wide replicated allocation; on a unit mesh this is a single device buffer

    // Names identify the kernels, the dataflow buffers and the tensor parameters within the program spec.
    const tt_metal::experimental::KernelSpecName reader_kernel{"reader"};
    const tt_metal::experimental::KernelSpecName writer_kernel{"writer"};
    const tt_metal::experimental::KernelSpecName compute_kernel{"compute"};
    const tt_metal::experimental::DFBSpecName src0_dfb{"src0"};
    const tt_metal::experimental::DFBSpecName src1_dfb{"src1"};
    const tt_metal::experimental::DFBSpecName output_dfb{"output"};
    const tt_metal::experimental::TensorParamName src0_tensor_param{"src0"};
    const tt_metal::experimental::TensorParamName src1_tensor_param{"src1"};
    const tt_metal::experimental::TensorParamName dst_tensor_param{"dst"};

    // Configure Dataflow Buffers
    // Dataflow buffers act as staging areas for data movement between DRAM and compute units.
    // Using 2 tiles per dataflow buffer to allow for double buffering (data movement can be reading from one tile while
    // the compute kernel is using the other tile). This number can be adjusted based on the use case, but generally
    // diminishing returns are observed after several tiles.
    // input tiles count is = 2 so one tile can be read while the other is being processed
    // A dataflow buffer is created on every core that runs its kernels (here: all cores).
    const auto cb_data_format = tt::DataFormat::Float16_b;
    uint32_t num_input_tiles = 2;
    tt_metal::experimental::DataflowBufferSpec src0_dfb_spec{
        .unique_id = src0_dfb,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = cb_data_format,
    };

    tt_metal::experimental::DataflowBufferSpec src1_dfb_spec{
        .unique_id = src1_dfb,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = cb_data_format,
    };

    tt_metal::experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = output_dfb,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = cb_data_format,
    };

    // Create Kernels (Reader, Writer, Compute)
    // - Reader kernel: Handles reading input data from DRAM into dataflow buffers
    // - Writer kernel: Handles writing output data from dataflow buffers back to DRAM
    // - Compute kernel: Performs the actual matrix multiplication computation
    // All kernels run across all cores to enable parallel execution
    tt::tt_metal::MathFidelity math_fidelity =
        tt::tt_metal::MathFidelity::HiFi4;  // High fidelity math for accurate results
    tt_metal::experimental::KernelSpec reader_kernel_spec{
        .unique_id = reader_kernel,
        .source =
            OVERRIDE_KERNEL_PREFIX "matmul/matmul_multi_core/kernels/dataflow/reader_mm_output_tiles_partitioned.cpp",
        .dfb_bindings =
            {tt_metal::experimental::ProducerOf(src0_dfb, "in0"), tt_metal::experimental::ProducerOf(src1_dfb, "in1")},
        .tensor_bindings =
            {{.tensor_parameter_name = src0_tensor_param, .accessor_name = "a"},
             {.tensor_parameter_name = src1_tensor_param, .accessor_name = "b"}},
        .runtime_arg_schema = {.runtime_arg_names = {"Mt", "Kt", "Nt", "output_tile_start_id", "num_output_tiles"}},
        .hw_config =
            tt_metal::experimental::DataMovementHardwareConfig{
                .config_1xx =
                    tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
            },
    };

    tt_metal::experimental::KernelSpec writer_kernel_spec{
        .unique_id = writer_kernel,
        .source =
            OVERRIDE_KERNEL_PREFIX "matmul/matmul_multi_core/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        .dfb_bindings = {tt_metal::experimental::ConsumerOf(output_dfb, "out0")},
        .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "c"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config =
            tt_metal::experimental::DataMovementHardwareConfig{
                .config_1xx =
                    tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            },
    };

    tt_metal::experimental::KernelSpec compute_kernel_spec{
        .unique_id = compute_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_multi_core/kernels/compute/mm.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {tt_metal::experimental::ConsumerOf(src0_dfb, "in0"),
             tt_metal::experimental::ConsumerOf(src1_dfb, "in1"),
             tt_metal::experimental::ProducerOf(output_dfb, "out0")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_output_tiles", "Kt"}},
        .hw_config =
            tt_metal::experimental::ComputeHardwareConfig{
                .fpu_math_fidelity = math_fidelity,
                .sfpu_precision_mode = Precision::Precise,
            },
    };

    // A program spec describes the program: its kernels, the dataflow buffers between them, the tensors they
    // operate on, and the cores the kernels run on (the work units).
    tt_metal::experimental::ProgramSpec program_spec{
        .name = "matmul_multi_core",
        .kernels = {reader_kernel_spec, writer_kernel_spec, compute_kernel_spec},
        .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec, output_dfb_spec},
        .tensor_parameters =
            {{.unique_id = src0_tensor_param, .spec = src0_tensor_spec},
             {.unique_id = src1_tensor_param, .spec = src1_tensor_spec},
             {.unique_id = dst_tensor_param, .spec = dst_tensor_spec}},
        .work_units =
            {{.name = "matmul_multi_core",
              .kernels = {reader_kernel, writer_kernel, compute_kernel},
              .target_nodes = all_cores}},
    };
    Program program = tt_metal::experimental::MakeProgramFromSpec(*mesh_device, program_spec);

    // Set Runtime Arguments for Kernels
    // Each core needs to know which portion of the work it's responsible for. We are parallelizing across output
    // tiles - each core computes different output tiles. Runtime arguments can be changed between program executions
    // without recompilation.
    tt_metal::experimental::KernelRunArgs reader_run_args{.kernel = reader_kernel};
    tt_metal::experimental::KernelRunArgs writer_run_args{.kernel = writer_kernel};
    tt_metal::experimental::KernelRunArgs compute_run_args{.kernel = compute_kernel};
    uint32_t work_offset = 0;
    auto work_groups = {std::make_pair(core_group_1, work_per_core1), std::make_pair(core_group_2, work_per_core2)};

    // Iterate through each work group and assign work to cores
    for (const auto& [ranges, work_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                // Set arguments for the reader kernel (data input)
                tt_metal::experimental::AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values,
                    core,
                    {{"Mt", Mt},                             // Number of tiles in M dimension
                     {"Kt", Kt},                             // Number of tiles in K dimension
                     {"Nt", Nt},                             // Number of tiles in N dimension
                     {"output_tile_start_id", work_offset},  // Starting offset for this core's work
                     {"num_output_tiles", work_per_core}});  // Amount of work for this core

                // Set arguments for the writer kernel (data output)
                tt_metal::experimental::AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {{"num_tiles", work_per_core}, {"start_id", work_offset}});

                // Set arguments for the compute kernel
                tt_metal::experimental::AddRuntimeArgsForNode(
                    compute_run_args.runtime_arg_values,
                    core,
                    {{"num_output_tiles", work_per_core},  // Amount of work for this core
                     {"Kt", Kt}});                         // Number of tiles in K dimension for dot product
                work_offset += work_per_core;              // Update offset for next core
            }
        }
    }

    // The matrices A, B and C are passed to the kernels through the tensors bound to the tensor parameters.
    tt_metal::experimental::ProgramRunArgs run_args{
        .kernel_run_args = {reader_run_args, writer_run_args, compute_run_args},
        .tensor_args =
            {{src0_tensor_param, tt_metal::experimental::TensorArgument{src0_dram_tensor}},
             {src1_tensor_param, tt_metal::experimental::TensorArgument{src1_dram_tensor}},
             {dst_tensor_param, tt_metal::experimental::TensorArgument{dst_dram_tensor}}},
    };
    tt_metal::experimental::SetProgramRunArgs(program, run_args);

    // Launch program & read in output tensor result into the host vector
    // 1. Upload input data to DRAM tensors
    // 2. Execute the program (all kernels run in parallel across cores)
    // 3. Read back the result from DRAM to host memory
    // The 'true' parameter in enqueue_read_tensor ensures we wait for completion (so when the function
    // returns, the output vector is fully populated).
    HostTensor src0_host_tensor = HostTensor::from_vector(a, src0_tensor_spec);
    HostTensor src1_host_tensor = HostTensor::from_vector(b, src1_tensor_spec);
    cq.enqueue_write_tensor(src0_host_tensor, src0_dram_tensor);
    cq.enqueue_write_tensor(src1_host_tensor, src1_dram_tensor);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    // Blocking read waits for completion before returning and resizes 'output' as needed
    output = cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();
}

///////////////////////////////////////

int main() {
    bool pass = true;

    try {
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

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
        uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt;                           // num_tiles of FP16_B

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
        matmul_multi_core(src0_vec, src1_vec, result_vec, M, N, K, mesh_device);
        // Reverse the tilization to get the result in the row-major format that the CPU expects
        result_vec = untilize_nfaces(result_vec, M, N);

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
