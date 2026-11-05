// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

using namespace tt;
using namespace tt::tt_metal;

/*
 * 1. Host creates one vector of data.
 * 2. Device eltwise performs a unary SFPU operation on the data.
 * 3. Read result back and compare to golden.
 * */
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main() {
    bool pass = true;

    try {
        // Initialize the device (here we use the 1st device, but you can use any device)
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        // In Metalium, submitting operations to the device is done through a command queue. This includes
        // uploading/downloading data to/from the device, and executing programs.
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        // A MeshWorkload is a collection of programs that will be executed on the mesh.
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

        // This example program will only use 1 Tensix core. So we set the core to {0, 0}.
        constexpr experimental::NodeCoord core = {0, 0};

        constexpr uint32_t n_tiles = 64;
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

        // Allocate DRAM tensors for the input and output data. Each row of a row-major tensor is one page, so a row
        // of one tile's worth of elements gives a page size of one tile.
        TensorSpec dram_tensor_spec(
            Shape({n_tiles, elements_per_tile}),
            TensorLayout(
                DataType::BFLOAT16,
                PageConfig(Layout::ROW_MAJOR),
                MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));

        // Allocate an input and output tensor on DRAM. We will perform a unary operation on the input tensor and write
        // the result.
        MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
        MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
        // Each tensor is replicated across the mesh; on a 1x1 mesh it's a single device buffer.

        // Names identify the kernels, the dataflow buffers and the tensor parameters within the program spec.
        const experimental::KernelSpecName unary_reader_kernel{"unary_reader"};
        const experimental::KernelSpecName unary_writer_kernel{"unary_writer"};
        const experimental::KernelSpecName eltwise_sfpu_kernel{"eltwise_sfpu"};
        const experimental::DFBSpecName src0_dfb{"src0"};
        const experimental::DFBSpecName output_dfb{"output"};
        const experimental::TensorParamName src0_tensor_param{"src0"};
        const experimental::TensorParamName dst_tensor_param{"dst"};

        // Describe 2 dataflow buffers for input and output.
        constexpr uint32_t num_input_tiles = 2;
        experimental::DataflowBufferSpec src0_dfb_spec{
            .unique_id = src0_dfb,
            .entry_size = tile_size_bytes,
            .num_entries = num_input_tiles,
            .data_format_metadata = tt::DataFormat::Float16_b,
        };
        experimental::DataflowBufferSpec output_dfb_spec{
            .unique_id = output_dfb,
            .entry_size = tile_size_bytes,
            .num_entries = num_input_tiles,
            .data_format_metadata = tt::DataFormat::Float16_b,
        };

        // Describe the 2 data movement kernels and the compute kernel.
        experimental::KernelSpec unary_reader_kernel_spec{
            .unique_id = unary_reader_kernel,
            .source = OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/dataflow/read_tile.cpp",
            .dfb_bindings = {experimental::ProducerOf(src0_dfb, "in0")},
            .tensor_bindings = {{.tensor_parameter_name = src0_tensor_param, .accessor_name = "in0"}},
            .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
            .hw_config =
                experimental::DataMovementHardwareConfig{
                    .config_1xx =
                        experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
                },
        };

        experimental::KernelSpec unary_writer_kernel_spec{
            .unique_id = unary_writer_kernel,
            .source = OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/dataflow/write_tile.cpp",
            .dfb_bindings = {experimental::ConsumerOf(output_dfb, "out0")},
            .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "out0"}},
            .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
            .hw_config =
                experimental::DataMovementHardwareConfig{
                    .config_1xx =
                        experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
                },
        };

        experimental::KernelSpec eltwise_sfpu_kernel_spec{
            .unique_id = eltwise_sfpu_kernel,
            .source = OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/compute/eltwise_sfpu.cpp",
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings = {experimental::ConsumerOf(src0_dfb, "in0"), experimental::ProducerOf(output_dfb, "out0")},
            .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
            .hw_config =
                experimental::ComputeHardwareConfig{
                    .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                    .sfpu_precision_mode = Precision::Precise,
                },
        };

        // A program is a collection of kernels. Note that unlike OpenCL/CUDA where every core must run the
        // same kernel at a given time. Metalium allows you to run different kernels on different cores
        // simultaneously.
        // A program spec describes the program: its kernels, the dataflow buffers between them, the tensors they
        // operate on, and the cores the kernels run on (the work units).
        experimental::ProgramSpec program_spec{
            .name = "eltwise_sfpu",
            .kernels = {unary_reader_kernel_spec, unary_writer_kernel_spec, eltwise_sfpu_kernel_spec},
            .dataflow_buffers = {src0_dfb_spec, output_dfb_spec},
            .tensor_parameters =
                {{.unique_id = src0_tensor_param, .spec = dram_tensor_spec},
                 {.unique_id = dst_tensor_param, .spec = dram_tensor_spec}},
            .work_units =
                {{.name = "eltwise_sfpu",
                  .kernels = {unary_reader_kernel, unary_writer_kernel, eltwise_sfpu_kernel},
                  .target_nodes = core}},
        };
        Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

        // Initialize the input data with random values and use as the input to the kernel.
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.f, 1.0f);
        std::vector<bfloat16> src0_vec(n_tiles * elements_per_tile);
        for (bfloat16& v : src0_vec) {
            v = bfloat16(dist(rng));
        }

        // Write the data on host to the input tensor on the device.
        // The write does not block, which allows us to overlap the data movement and following host operations (
        // setting kerenel args) in this case
        HostTensor src0_host_tensor = HostTensor::from_vector(src0_vec, dram_tensor_spec);
        cq.enqueue_write_tensor(src0_host_tensor, src0_dram_tensor);

        // Set up the runtime arguments for the kernels, and the tensors bound to the tensor parameters.
        experimental::ProgramRunArgs run_args{
            .kernel_run_args =
                {{.kernel = eltwise_sfpu_kernel,
                  .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
                 {.kernel = unary_reader_kernel,
                  .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
                 {.kernel = unary_writer_kernel,
                  .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})}},
            .tensor_args =
                {{src0_tensor_param, experimental::TensorArgument{src0_dram_tensor}},
                 {dst_tensor_param, experimental::TensorArgument{dst_dram_tensor}}},
        };
        experimental::SetProgramRunArgs(program, run_args);

        // Enqueue the program as a mesh workload (non-blocking) and wait for completion before reading results.
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        // Read the result (from shard at mesh coordinate {0,0} on a unit mesh) and compare to our expected result.
        std::vector<bfloat16> result_vec =
            cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();

        // Compute the same thing on CPU for comparison

        // Compare the result to the golden vector. Loose tolerance of 2e-2f because bfloat16 is not as accurate as 32
        // bit float.
        constexpr float eps = 5e-2f;
        for (uint32_t i = 0; i < result_vec.size(); ++i) {
            float expected = static_cast<float>(bfloat16(std::exp(static_cast<float>(src0_vec[i]))));
            float result = static_cast<float>(result_vec[i]);
            if (std::abs(expected - result) > eps) {
                pass = false;
                fmt::print(stderr, "Result mismatch at index {}: {} != {}\n", i, expected, result);
            }
        }

        // Finally, close the device.
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
