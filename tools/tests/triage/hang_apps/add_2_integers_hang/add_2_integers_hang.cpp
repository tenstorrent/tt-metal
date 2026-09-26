// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tensor/tensor_apis.hpp>

#include "stop_simulation_on_termination.hpp"

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {

const experimental::DFBSpecName IN0_DFB{"in0_dfb"};
const experimental::DFBSpecName IN1_DFB{"in1_dfb"};
const experimental::DFBSpecName OUT_DFB{"out_dfb"};
const experimental::TensorParamName IN0_T{"in0_tensor"};
const experimental::TensorParamName IN1_T{"in1_tensor"};
const experimental::TensorParamName OUT_T{"out_tensor"};
const experimental::KernelSpecName READER{"reader"};
const experimental::KernelSpecName WRITER{"writer"};
const experimental::KernelSpecName COMPUTE{"compute"};

// A single BFloat16 tile in interleaved DRAM. HostTensor::from_vector / to_vector take and
// return row-major data and handle the tilization against this spec, so the host code below
// works with plain row-major vectors.
TensorSpec single_tile_dram_spec() {
    return TensorSpec(
        Shape{tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::TILE),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
}

}  // namespace

int main() {
    // Killed while hung -- which is the whole point of this app -- we still have to release the
    // simulator, so exit rather than die on SIGTERM.
    triage_hang_apps::stop_simulation_on_termination();

    // A MeshDevice is a software concept that allows developers to virtualize a cluster of connected devices as a
    // single object, maintaining uniform memory and runtime state across all physical devices. A UnitMesh is a 1x1
    // MeshDevice that allows users to interface with a single physical device.
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    // In Metalium, submitting operations to the device is done through a command queue. This includes
    // uploading/downloading data to/from the device, and executing programs.
    // A MeshCommandQueue is a software concept that allows developers to submit operations to a MeshDevice.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    // We will only be using one Tensix node for this particular example. As Tenstorrent processors are a 2D grid of
    // nodes we can specify the node coordinates as (0, 0).
    constexpr experimental::NodeCoord node = {0, 0};

    // Most data on Tensix is stored in tiles. A tile is a 2D array of (usually) 32x32 values. And the Tensix uses
    // BFloat16 as the most well supported data type. Thus the tile size is 32x32x2 = 2048 bytes.
    constexpr uint32_t n_elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_WIDTH;
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * n_elements_per_tile;

    // Tensor Creation:
    // Create 3 tensors in DRAM to hold the 2 input tiles and 1 output tile. A MeshTensor is a
    // user-managed device memory resource; the kernels reach it through the TensorAccessor bound
    // to the matching TensorParameter below.
    const TensorSpec tile_spec = single_tile_dram_spec();
    auto src0_tensor = MeshTensor::allocate_on_device(*mesh_device, tile_spec);
    auto src1_tensor = MeshTensor::allocate_on_device(*mesh_device, tile_spec);
    auto dst_tensor = MeshTensor::allocate_on_device(*mesh_device, tile_spec);

    // Create 3 dataflow buffers. Think of them like pipes moving data from one kernel to another. in0 and in1 are
    // used to move data from the reader kernel to the compute kernel. out is used to move data from the compute
    // kernel to the writer kernel. Each one is made up of 1 tile here. A larger number of entries lets the sending
    // end get the next piece of data ready while the receiving end is still using the current one, overlapping the
    // operations and leading to better performance. However there is a trade off: the more entries, the more memory
    // is used, and dataflow buffers are backed by L1 (SRAM) memory, which is a precious resource.
    constexpr uint32_t num_tiles = 1;
    auto make_dfb_spec = [&](const experimental::DFBSpecName& name) {
        return experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = single_tile_size,
            .num_entries = num_tiles,
            .data_format_metadata = tt::DataFormat::Float16_b,
        };
    };

    // The kernels below carry both a Gen1 and a Gen2 hardware config so that the same program runs on Wormhole /
    // Blackhole and on Quasar; the runtime selects the one matching the active architecture. Quasar has no
    // per-processor data movement kernel assignment (it has 8 data movement cores per cluster, allocated by the
    // implementation), which is why the Gen2 config carries no processor / NOC selection.
    const bool is_quasar = mesh_device->arch() == ARCH::QUASAR;
    experimental::DataMovementHardwareConfig reader_dm_config;
    experimental::DataMovementHardwareConfig writer_dm_config;
    experimental::ComputeHardwareConfig compute_hw_config;
    if (is_quasar) {
        reader_dm_config = experimental::DataMovementGen2Config{};
        writer_dm_config = experimental::DataMovementGen2Config{};
        compute_hw_config = experimental::ComputeGen2Config{.fpu_math_fidelity = MathFidelity::HiFi4};
    } else {
        // The conventional Gen1 placement: reader on RISCV_1, writer on RISCV_0.
        reader_dm_config = experimental::CreateReaderGen1DataMovementConfig();
        writer_dm_config = experimental::CreateWriterGen1DataMovementConfig();
        compute_hw_config = experimental::ComputeGen1Config{.fpu_math_fidelity = MathFidelity::HiFi4};
    }

    // Describe the reader, writer and compute kernels. The sibling ttnn_add_integers_hang app runs
    // these same three kernel sources through its own TTNN program factory.
    //
    // The kernels do the following:
    // * Reader: Reads data from the input tensors and pushes it into the input dataflow buffers.
    // * Compute: Waits for data to be available in the input dataflow buffers, pops it, adds the two inputs together
    //   and pushes the result into the output dataflow buffer.
    // * Writer: Waits for data to be available in the output dataflow buffer, pops it and writes it back to DRAM.
    // These kernels work together to form a pipeline. The reader reads data from DRAM and makes it available to the
    // compute kernel. The compute kernel does math and pushes the result to the writer kernel. The writer kernel
    // writes the result back to DRAM.
    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = OVERRIDE_KERNEL_PREFIX "add_2_integers_hang/kernels/dataflow/reader_binary_1_tile.cpp",
        .dfb_bindings = {experimental::ProducerOf(IN0_DFB, "in0"), experimental::ProducerOf(IN1_DFB, "in1")},
        .tensor_bindings =
            {{.tensor_parameter_name = IN0_T, .accessor_name = "in0"},
             {.tensor_parameter_name = IN1_T, .accessor_name = "in1"}},
        .hw_config = reader_dm_config,
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = OVERRIDE_KERNEL_PREFIX "add_2_integers_hang/kernels/dataflow/writer_1_tile.cpp",
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {{.tensor_parameter_name = OUT_T, .accessor_name = "out"}},
        .hw_config = writer_dm_config,
    };

    // This kernel performs the actual addition of the two input tiles
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = OVERRIDE_KERNEL_PREFIX "add_2_integers_hang/kernels/compute/add_2_tiles_hang.cpp",
        // Metal 2.0's type-agnostic default opt level is O2; a compute kernel used to get O3, so
        // state it explicitly to keep the generated code the same as before.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {experimental::ConsumerOf(IN0_DFB, "in0"),
             experimental::ConsumerOf(IN1_DFB, "in1"),
             experimental::ProducerOf(OUT_DFB, "out")},
        .hw_config = compute_hw_config,
    };

    experimental::ProgramSpec spec{
        .name = "add_2_integers_hang",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {make_dfb_spec(IN0_DFB), make_dfb_spec(IN1_DFB), make_dfb_spec(OUT_DFB)},
        .tensor_parameters =
            {{.unique_id = IN0_T, .spec = tile_spec},
             {.unique_id = IN1_T, .spec = tile_spec},
             {.unique_id = OUT_T, .spec = tile_spec}},
        .work_units = {{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node}},
    };

    // A MeshWorkload is a collection of programs that are executed on a MeshDevice. Building it from the ProgramSpec
    // compiles the kernels; the specific physical devices the workload runs on are determined by the mesh shape.
    distributed::MeshWorkload workload = experimental::MakeMeshWorkloadFromSpec(*mesh_device, spec);
    Program& program = workload.get_programs().begin()->second;

    // Bind the tensors the kernels operate on. None of the three kernels declares runtime args of its own: the
    // reader and writer get their addresses from these tensor arguments, and the compute kernel needs none.
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = READER},
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = WRITER},
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {
        {IN0_T, experimental::TensorArgument{src0_tensor}},
        {IN1_T, experimental::TensorArgument{src1_tensor}},
        {OUT_T, experimental::TensorArgument{dst_tensor}},
    };
    experimental::SetProgramRunArgs(program, params);

    // Create the data that will be used as input to the kernels.
    // src0 is a vector of bfloat16 values initialized to random values between 0.0f and 14.0f.
    // src1 is a vector of bfloat16 values initialized to random values between 0.0f and 8.0f.
    std::vector<bfloat16> src0_vec(n_elements_per_tile);
    std::vector<bfloat16> src1_vec(n_elements_per_tile);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist1(0.0f, 14.0f);
    std::uniform_real_distribution<float> dist2(0.0f, 8.0f);
    for (size_t i = 0; i < n_elements_per_tile; ++i) {
        src0_vec[i] = bfloat16(dist1(rng));
        src1_vec[i] = bfloat16(dist2(rng));
    }

    // Upload the data from host to the device.
    cq.enqueue_write_tensor(HostTensor::from_vector(src0_vec, tile_spec), src0_tensor);
    cq.enqueue_write_tensor(HostTensor::from_vector(src1_vec, tile_spec), src1_tensor);

    // Execute the workload. The compute kernel hangs on purpose, so on a device with a dispatch timeout configured
    // this throws rather than returning; on the RTL simulator in slow dispatch the wait is unbounded and the process
    // stays blocked here, which is what lets an external debug tool inspect the hung device.
    try {
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
    } catch (std::runtime_error& e) {
        // Being torn down: teardown has closed the link to the simulator, so this failure is the
        // expected end of the wait, not a fault to report.
        if (triage_hang_apps::termination_requested()) {
            triage_hang_apps::park_until_process_exits();
        }
        std::string error_msg = e.what();
        if (error_msg.find("device timeout") != std::string::npos || error_msg.find("Timeout (") != std::string::npos) {
            printf("Device timeout detected as expected.\n");
            std::_Exit(0);
        } else {
            throw;
        }
    }

    // Data can be read back from a MeshTensor through the command queue.
    std::vector<bfloat16> result_vec = cq.enqueue_read_tensor(dst_tensor).to_vector<bfloat16>();

    // compare the results with the expected values.
    bool success = true;
    for (size_t i = 0; i < n_elements_per_tile; ++i) {
        float expected = static_cast<float>(src0_vec[i]) + static_cast<float>(src1_vec[i]);
        if (std::abs(expected - static_cast<float>(result_vec[i])) > 3e-1f) {
            fmt::print(
                stderr, "Mismatch at index {}: expected {}, got {}\n", i, expected, static_cast<float>(result_vec[i]));
            success = false;
        }
    }
    if (!success) {
        fmt::print("Error: Result does not match expected value!\n");
    } else {
        fmt::print("Success: Result matches expected value!\n");
    }
    mesh_device->close();
}
