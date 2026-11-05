// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

#include <cstdint>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

int main() {
    // Create a 1x1 mesh device (Mesh API). For multi-device setups, create a larger mesh shape.
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    // Mesh command queue and program setup
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // Core range setup
    constexpr experimental::NodeCoord core0 = {0, 0};
    constexpr experimental::NodeCoord core1 = {0, 1};
    const auto core0_physical_coord = mesh_device->worker_core_from_logical_core(core0);
    const auto core1_physical_coord = mesh_device->worker_core_from_logical_core(core1);

    experimental::NodeRange sem_core_range = experimental::NodeRange(core0, core1);

    // Check if the environment variable for kernels print is set
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print(
            stderr,
            "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to (0,0),(0,1) to see the output of "
            "the Data Movement kernels. Command: export TT_METAL_DPRINT_CORES=(0,0),(0,1)\n");
    }

    // Input data preparation
    constexpr uint32_t single_tile_size = sizeof(uint16_t) * tt::constants::TILE_HW;
    // A row-major tensor with one row of one tile's worth of elements is a single DRAM page of one tile.
    TensorSpec dram_tensor_spec(
        Shape({1, tt::constants::TILE_HW}),
        TensorLayout(
            DataType::UINT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));

    MeshTensor src_dram_tensor =
        MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);  // replicated per device
    MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);

    // Names identify the kernels, the dataflow buffers, the semaphore and the tensor parameters within the program
    // spec.
    const experimental::KernelSpecName core0_reader_kernel{"core0_reader"};
    const experimental::KernelSpecName core0_writer_kernel{"core0_writer"};
    const experimental::KernelSpecName core1_reader_kernel{"core1_reader"};
    const experimental::KernelSpecName core1_writer_kernel{"core1_writer"};
    const experimental::DFBSpecName src0_dfb{"src0"};
    const experimental::DFBSpecName src1_dfb{"src1"};
    const experimental::SemaphoreSpecName sem{"core_sync"};
    const experimental::TensorParamName src_tensor_param{"src"};
    const experimental::TensorParamName dst_tensor_param{"dst"};

    // Core synchronization semaphore setup
    experimental::SemaphoreSpec sem_spec{
        .unique_id = sem,
        .target_nodes = sem_core_range,
    };

    // Source data preparation and DRAM transfer
    const uint16_t input_data = 14;  // Example input data
    std::vector<uint16_t> src_vec(single_tile_size / sizeof(uint16_t), input_data);
    HostTensor src_host_tensor = HostTensor::from_vector(src_vec, dram_tensor_spec);
    cq.enqueue_write_tensor(src_host_tensor, src_dram_tensor);

    // L1 dataflow buffer setup
    experimental::DataflowBufferSpec src0_dfb_spec{
        .unique_id = src0_dfb,
        .entry_size = single_tile_size,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::UInt16,
    };

    // src1 is bound on both cores, so it has the same L1 address on both cores.
    // Writer 0 uses its own src1 address as the destination address on core 1.
    experimental::DataflowBufferSpec src1_dfb_spec{
        .unique_id = src1_dfb,
        .entry_size = single_tile_size,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::UInt16,
    };

    // Kernels setup
    // Core 0 kernels
    experimental::KernelSpec core0_reader_kernel_spec{
        .unique_id = core0_reader_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "NoC_tile_transfer/kernels/dataflow/reader0.cpp",
        .dfb_bindings = {experimental::ProducerOf(src0_dfb, "src0"), experimental::ProducerOf(src1_dfb, "src1")},
        .tensor_bindings = {{.tensor_parameter_name = src_tensor_param, .accessor_name = "input"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_0},
            },
    };
    experimental::KernelSpec core0_writer_kernel_spec{
        .unique_id = core0_writer_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "NoC_tile_transfer/kernels/dataflow/writer0.cpp",
        .dfb_bindings = {experimental::ConsumerOf(src0_dfb, "src0"), experimental::ConsumerOf(src1_dfb, "src1")},
        .semaphore_bindings = {{.semaphore_spec_name = sem, .accessor_name = "core_sync"}},
        .runtime_arg_schema = {.runtime_arg_names = {"core_1_x", "core_1_y"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_1},
            },
    };

    // Core 1 kernels
    experimental::KernelSpec core1_reader_kernel_spec{
        .unique_id = core1_reader_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "NoC_tile_transfer/kernels/dataflow/reader1.cpp",
        .dfb_bindings = {experimental::ProducerOf(src1_dfb, "src1")},
        .semaphore_bindings = {{.semaphore_spec_name = sem, .accessor_name = "core_sync"}},
        .runtime_arg_schema = {.runtime_arg_names = {"core_0_x", "core_0_y"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_0},
            },
    };
    experimental::KernelSpec core1_writer_kernel_spec{
        .unique_id = core1_writer_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "NoC_tile_transfer/kernels/dataflow/writer1.cpp",
        .dfb_bindings = {experimental::ConsumerOf(src1_dfb, "src1")},
        .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "output"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_1},
            },
    };

    // The program spec places the core 0 kernels and the core 1 kernels in separate work units.
    experimental::ProgramSpec program_spec{
        .name = "noc_tile_transfer",
        .kernels =
            {core0_reader_kernel_spec, core0_writer_kernel_spec, core1_reader_kernel_spec, core1_writer_kernel_spec},
        .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec},
        .semaphores = {sem_spec},
        .tensor_parameters =
            {{.unique_id = src_tensor_param, .spec = dram_tensor_spec},
             {.unique_id = dst_tensor_param, .spec = dram_tensor_spec}},
        .work_units =
            {{.name = "core0", .kernels = {core0_reader_kernel, core0_writer_kernel}, .target_nodes = core0},
             {.name = "core1", .kernels = {core1_reader_kernel, core1_writer_kernel}, .target_nodes = core1}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

    // Runtime args setup
    experimental::ProgramRunArgs run_args{
        .kernel_run_args =
            {{.kernel = core0_writer_kernel,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                  core0, {{"core_1_x", core1_physical_coord.x}, {"core_1_y", core1_physical_coord.y}})},
             {.kernel = core1_reader_kernel,
              .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                  core1, {{"core_0_x", core0_physical_coord.x}, {"core_0_y", core0_physical_coord.y}})}},
        .tensor_args =
            {{src_tensor_param, experimental::TensorArgument{src_dram_tensor}},
             {dst_tensor_param, experimental::TensorArgument{dst_dram_tensor}}},
    };
    experimental::SetProgramRunArgs(program, run_args);

    // Program enqueue (non-blocking). Wait for completion before reading back.
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Data transfer back to host machine
    std::vector<uint16_t> result_vec = cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<uint16_t>();

    fmt::print("Result = {} : Expected = {}\n", result_vec[0], input_data);

    mesh_device->close();
}
