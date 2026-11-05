// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/ostream.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
#include "tt-metalium/base_types.hpp"

using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    bool pass = true;

    try {
        // Create a 1x1 mesh on device 0 (same API scales to multi-device meshes)
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        // Submit work via the mesh command queue: uploads/downloads and program execution.
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        // Define some constants that will be used throughout the program.
        // * Processing 64 tiles
        // * Each tile is 32x32 elements
        // * Each element is a bfloat16 (2 bytes)
        constexpr uint32_t n_tiles = 64;
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

        // Configure mesh tensors. Use single-tile page size so transfers operate tile-by-tile: each row of a
        // row-major tensor is one page (number of bytes when round-robin between banks), so a row of one tile's
        // worth of elements gives a page size of one tile. The tensors live in DRAM.
        TensorSpec dram_tensor_spec(
            Shape({n_tiles, elements_per_tile}),
            TensorLayout(
                DataType::BFLOAT16,
                PageConfig(Layout::ROW_MAJOR),
                MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));

        // Allocate the tensors (replicated across mesh; on unit mesh ⇒ single device allocation)
        // src0, src1 are input tensors; dst is output tensor
        MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
        MeshTensor src1_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
        MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);

        // A MeshWorkload is a collection of programs that will be executed on the mesh. Each workload is
        // local to a single device. Here we create a workload for our single-device mesh.
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

        // This example program will only use 1 Tensix core. So we set the core to {0, 0} (the most top-left core).
        constexpr experimental::NodeCoord core = {0, 0};

        // Names identify the kernels, the dataflow buffers and the tensor parameters within the program spec.
        const experimental::KernelSpecName reader{"reader"};
        const experimental::KernelSpecName writer{"writer"};
        const experimental::KernelSpecName compute{"compute"};
        const experimental::DFBSpecName src0_dfb{"src0"};
        const experimental::DFBSpecName src1_dfb{"src1"};
        const experimental::DFBSpecName dst_dfb{"dst"};
        const experimental::TensorParamName src0_tensor_param{"src0"};
        const experimental::TensorParamName src1_tensor_param{"src1"};
        const experimental::TensorParamName dst_tensor_param{"dst"};

        // Describe 3 dataflow buffers. Think of them as pipes moving data from one core to another.
        // dfb_src0 and dfb_src1 are used to move data from the reader kernel to the compute kernel.
        // dfb_dst is used to move data from the compute kernel to the writer kernel.
        // Each dataflow buffer is made up of 2 tiles. Thus when one tile is pushed and being used by the receiving end,
        // the sending end can get the next piece of data ready to be pushed. Overlapping the operations leads to better
        // performance. However, there is a trade off: the more tiles in a dataflow buffer, the more memory is used.
        // Dataflow buffers are backed by L1(SRAM) memory and L1 is a precious resource. The hardware supports up to 32
        // dataflow buffers and they all act the same.
        constexpr uint32_t tiles_per_cb = 2;
        experimental::DataflowBufferSpec src0_dfb_spec{
            .unique_id = src0_dfb,
            .entry_size = tile_size_bytes,
            .num_entries = tiles_per_cb,
            .data_format_metadata = tt::DataFormat::Float16_b,
        };
        experimental::DataflowBufferSpec src1_dfb_spec{
            .unique_id = src1_dfb,
            .entry_size = tile_size_bytes,
            .num_entries = tiles_per_cb,
            .data_format_metadata = tt::DataFormat::Float16_b,
        };
        experimental::DataflowBufferSpec dst_dfb_spec{
            .unique_id = dst_dfb,
            .entry_size = tile_size_bytes,
            .num_entries = tiles_per_cb,
            .data_format_metadata = tt::DataFormat::Float16_b,
        };

        // Describe the reader, writer and compute kernels. The kernels do the following:
        // * Reader: Reads data from the DRAM buffer and pushes it into the dataflow buffer.
        // * Compute: Waits for data to be available in the dataflow buffer, pops it, adds the two inputs together and
        // pushes the result
        //   into the output dataflow buffer.
        // * Writer: Waits for data to be available in the output dataflow buffer, pops it and writes it back into DRAM.
        // These kernels work together to form a pipeline. The reader reads data from the DRAM buffer and makes them
        // available in the compute kernel. The compute kernel does math and pushes the result into the writer kernel.
        // The writer kernel writes the result back to DRAM.
        experimental::KernelSpec reader_spec{
            .unique_id = reader,
            .source = OVERRIDE_KERNEL_PREFIX "custom_sfpi_add/kernels/dataflow/read_tiles.cpp",
            .dfb_bindings = {experimental::ProducerOf(src0_dfb, "in0"), experimental::ProducerOf(src1_dfb, "in1")},
            .tensor_bindings =
                {{.tensor_parameter_name = src0_tensor_param, .accessor_name = "in0"},
                 {.tensor_parameter_name = src1_tensor_param, .accessor_name = "in1"}},
            .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
            .hw_config =
                experimental::DataMovementHardwareConfig{
                    .config_1xx =
                        experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
                },
        };
        experimental::KernelSpec writer_spec{
            .unique_id = writer,
            .source = OVERRIDE_KERNEL_PREFIX "custom_sfpi_add/kernels/dataflow/write_tile.cpp",
            .dfb_bindings = {experimental::ConsumerOf(dst_dfb, "out0")},
            .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "out0"}},
            .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
            .hw_config =
                experimental::DataMovementHardwareConfig{
                    .config_1xx =
                        experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
                },
        };
        experimental::KernelSpec compute_spec{
            .unique_id = compute,
            .source = OVERRIDE_KERNEL_PREFIX "custom_sfpi_add/kernels/compute/tiles_add.cpp",
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {experimental::ConsumerOf(src0_dfb, "in0"),
                 experimental::ConsumerOf(src1_dfb, "in1"),
                 experimental::ProducerOf(dst_dfb, "out0")},
            .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
            .hw_config =
                experimental::ComputeHardwareConfig{
                    .enable_32_bit_dest =
                        false,  // We don't need the destination accumulator to be FP32 as input and output are BFP16
                },
        };

        // A program is a collection of kernels. Note that unlike OpenCL/CUDA where every core must run the
        // same kernel at a given time. Metalium allows you to run different kernels on different cores
        // simultaneously.
        // A program spec describes the program: its kernels, the dataflow buffers between them, the tensors they
        // operate on, and the cores the kernels run on (the work units).
        experimental::ProgramSpec program_spec{
            .name = "custom_sfpi_add",
            .kernels = {reader_spec, writer_spec, compute_spec},
            .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec, dst_dfb_spec},
            .tensor_parameters =
                {{.unique_id = src0_tensor_param, .spec = dram_tensor_spec},
                 {.unique_id = src1_tensor_param, .spec = dram_tensor_spec},
                 {.unique_id = dst_tensor_param, .spec = dram_tensor_spec}},
            .work_units = {{.name = "custom_sfpi_add", .kernels = {reader, writer, compute}, .target_nodes = core}},
        };
        Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

        // Initialize the input buffers with random data. For this example, src0 is a random vector of bfloat16 values
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        std::vector<bfloat16> a_data(elements_per_tile * n_tiles);
        for (auto& val : a_data) {
            val = bfloat16(distribution(rng));
        }

        // ... and src1 is a vector of bfloat16 values initialized to -1.0f.
        constexpr float val_to_add = -1.0f;
        std::vector<bfloat16> b_data(elements_per_tile * n_tiles, bfloat16(val_to_add));

        // Upload the data from host to the device. The tensor write is non-blocking: an upload will be launched, but
        // the function will return immediately, before the upload is complete. This is useful for performance reasons,
        // as it allows the host to continue while the upload is in progress. Note that the host is responsible for
        // ensuring that the upload is complete before the memory holding the data is freed.
        HostTensor a_host_tensor = HostTensor::from_vector(a_data, dram_tensor_spec);
        HostTensor b_host_tensor = HostTensor::from_vector(b_data, dram_tensor_spec);
        cq.enqueue_write_tensor(a_host_tensor, src0_dram_tensor);
        cq.enqueue_write_tensor(b_host_tensor, src1_dram_tensor);

        // Set the runtime arguments for the kernels, and the tensors bound to the tensor parameters.
        experimental::ProgramRunArgs run_args{
            .kernel_run_args =
                {{.kernel = reader,
                  .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
                 {.kernel = writer,
                  .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
                 {.kernel = compute,
                  .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})}},
            .tensor_args =
                {{src0_tensor_param, experimental::TensorArgument{src0_dram_tensor}},
                 {src1_tensor_param, experimental::TensorArgument{src1_dram_tensor}},
                 {dst_tensor_param, experimental::TensorArgument{dst_dram_tensor}}},
        };
        experimental::SetProgramRunArgs(program, run_args);

        // Add the program to the workload for the mesh.
        workload.add_program(device_range, std::move(program));
        // Enqueue the workload for execution on the mesh (non-blocking) and wait for completion before reading back.
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        distributed::Finish(cq);
        // NOTE: The above is equivalent to a blocking enqueue of the workload.

        // Read the result back from the shard at mesh coordinate {0,0}. Use blocking=true to wait for completion.
        // The vector is automatically resized to fit the data.
        std::vector<bfloat16> result_vec =
            cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();

        // Compare the result with the input. The result should be the input plus val_to_add.
        constexpr float eps = 1e-2f; // loose tolerance because of the nature of bfloat16
        TT_FATAL(result_vec.size() == a_data.size(), "Result vector size mismatch");
        for (size_t i = 0; i < result_vec.size(); ++i) {
            const float expected = static_cast<float>(a_data[i]) + val_to_add;
            const float actual = static_cast<float>(result_vec[i]);
            if (std::abs(expected - actual) > eps) {
                pass = false;
                fmt::print(stderr, "Result mismatch at index {}: expected {}, got {}\n", i, expected, actual);
            }
        }

        // Close the device
        if (!mesh_device->close()) {
            pass = false;
        }

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
