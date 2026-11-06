// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
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
    bool /*bcast_batch*/,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // Set up mesh command queue, workload and device range. This is a single-core example using core {0,0}.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    // Core range from x: [0, 0] to y: [0, 0] (single core at {0, 0})
    tt_metal::experimental::NodeCoord core({0, 0});

    // Calcaulate the number of tiles for each dimension.
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    // Create DRAM tensors for the input and output data.
    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    uint32_t elements_per_tile = single_tile_size / sizeof(bfloat16);

    // We allocate DRAM tensors for the input and output data (replicated per device across the mesh).
    // Setting the page size to single_tile_size is the most common configuration for memory buffers in Metalium
    // as it is generic, works for most cases and achieves good performance. Each row of a row-major tensor is one
    // page, so a row of one tile's worth of elements gives a page size of one tile.
    auto make_dram_tensor_spec = [&](size_t num_elements) {
        return TensorSpec(
            Shape({static_cast<uint32_t>(num_elements / elements_per_tile), elements_per_tile}),
            TensorLayout(
                DataType::BFLOAT16,
                PageConfig(Layout::ROW_MAJOR),
                MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
    };
    TensorSpec src0_tensor_spec = make_dram_tensor_spec(a.size());
    TensorSpec src1_tensor_spec = make_dram_tensor_spec(b.size());
    TensorSpec dst_tensor_spec = make_dram_tensor_spec(output.size());

    MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, src0_tensor_spec);
    MeshTensor src1_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, src1_tensor_spec);
    MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dst_tensor_spec);

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

    // Create dataflow buffers for the input and output data.
    // Using 2 tiles per dataflow buffer to allow for double buffering (data movement can be reading from one tile while
    // the compute kernel is using the other tile). This number can be adjusted based on the use case. But geberally
    // diminishing returns observed after several tiles.
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
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

    uint32_t num_output_tiles = 2;
    tt_metal::experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = output_dfb,
        .entry_size = single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = cb_data_format,
    };

    // Describe the data movement kernels and the compute kernel
    tt_metal::experimental::KernelSpec reader_kernel_spec{
        .unique_id = reader_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp",
        .dfb_bindings =
            {tt_metal::experimental::ProducerOf(src0_dfb, "in0"), tt_metal::experimental::ProducerOf(src1_dfb, "in1")},
        .tensor_bindings =
            {{.tensor_parameter_name = src0_tensor_param, .accessor_name = "in0"},
             {.tensor_parameter_name = src1_tensor_param, .accessor_name = "in1"}},
        .runtime_arg_schema = {.runtime_arg_names = {"Mt", "Kt", "Nt"}},
        .hw_config =
            tt_metal::experimental::DataMovementHardwareConfig{
                .config_1xx =
                    tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
            },
    };

    tt_metal::experimental::KernelSpec writer_kernel_spec{
        .unique_id = writer_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp",
        .dfb_bindings = {tt_metal::experimental::ConsumerOf(output_dfb, "out0")},
        .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "out0"}},
        .runtime_arg_schema = {.runtime_arg_names = {"Mt", "Nt"}},
        .hw_config =
            tt_metal::experimental::DataMovementHardwareConfig{
                .config_1xx =
                    tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            },
    };

    // Compile time arguments for the kernels
    // Note that these take effect at the kernel's compile time. Changing these values will require recompilation of the
    // kernel. Having arguments at compile time allows the compiler to optimize the kernel for the specific use case.
    // Like applying loop unrolling, constant folding, etc.. resulting in a more efficient kernel.
    tt_metal::experimental::KernelSpec compute_kernel_spec{
        .unique_id = compute_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/compute/mm.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {tt_metal::experimental::ConsumerOf(src0_dfb, "in0"),
             tt_metal::experimental::ConsumerOf(src1_dfb, "in1"),
             tt_metal::experimental::ProducerOf(output_dfb, "out0")},
        .compile_time_args = {{"Mt", Mt}, {"Kt", Kt}, {"Nt", Nt}},
        .hw_config =
            tt_metal::experimental::ComputeHardwareConfig{
                .fpu_math_fidelity = math_fidelity,
                .sfpu_precision_mode = Precision::Precise,
            },
    };

    // A program spec describes the program: its kernels, the dataflow buffers between them, the tensors they
    // operate on, and the cores the kernels run on (the work units).
    tt_metal::experimental::ProgramSpec program_spec{
        .name = "matmul_single_core",
        .kernels = {reader_kernel_spec, writer_kernel_spec, compute_kernel_spec},
        .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec, output_dfb_spec},
        .tensor_parameters =
            {{.unique_id = src0_tensor_param, .spec = src0_tensor_spec},
             {.unique_id = src1_tensor_param, .spec = src1_tensor_spec},
             {.unique_id = dst_tensor_param, .spec = dst_tensor_spec}},
        .work_units =
            {{.name = "matmul_single_core",
              .kernels = {reader_kernel, writer_kernel, compute_kernel},
              .target_nodes = core}},
    };
    Program program = tt_metal::experimental::MakeProgramFromSpec(*mesh_device, program_spec);

    // Set kernel arguments, and the tensors bound to the tensor parameters.
    tt_metal::experimental::ProgramRunArgs run_args{
        .kernel_run_args =
            {{.kernel = reader_kernel,
              .runtime_arg_values =
                  tt_metal::experimental::MakeRuntimeArgsForSingleNode(core, {{"Mt", Mt}, {"Kt", Kt}, {"Nt", Nt}})},
             {.kernel = writer_kernel,
              .runtime_arg_values =
                  tt_metal::experimental::MakeRuntimeArgsForSingleNode(core, {{"Mt", Mt}, {"Nt", Nt}})}},
        .tensor_args =
            {{src0_tensor_param, tt_metal::experimental::TensorArgument{src0_dram_tensor}},
             {src1_tensor_param, tt_metal::experimental::TensorArgument{src1_dram_tensor}},
             {dst_tensor_param, tt_metal::experimental::TensorArgument{dst_dram_tensor}}},
    };
    tt_metal::experimental::SetProgramRunArgs(program, run_args);
    // NOTE: Note that we never set the runtime arguments for the compute kernel. This is because everything needed has
    // been set at compile time. The compute kernel does not need any runtime arguments to execute. And so we can skip
    // this step.

    // Upload the input data to the DRAM tensors, execute the kernels, wait for the result to be read into the output
    // buffer
    HostTensor src0_host_tensor = HostTensor::from_vector(a, src0_tensor_spec);
    HostTensor src1_host_tensor = HostTensor::from_vector(b, src1_tensor_spec);
    cq.enqueue_write_tensor(src0_host_tensor, src0_dram_tensor);
    cq.enqueue_write_tensor(src1_host_tensor, src1_dram_tensor);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    output = cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();
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
        matmul_single_core(src0_vec, src1_vec, result_vec, false, M, N, K, mesh_device);
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
