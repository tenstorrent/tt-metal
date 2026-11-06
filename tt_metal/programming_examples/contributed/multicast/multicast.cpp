// SPDX-FileCopyrightText: © 2025 Ryan Barton
//
// SPDX-License-Identifier: Apache-2.0

/*
 This example demonstrates how to create a 32x32 data tile and multicast it from a sender (coordinator) core to multiple
 receiver cores.  It covers the setup of semaphores and multicore addressing on Tenstorrent hardware.  In the original
 configuration, (0,0) is the sender core, while (1,0), (2,0), and (3,0) are the receiver cores.  The user can modify
 these coordinates as desired.
*/

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <cstdint>
#include <vector>
#include <memory>
#include <iostream>
#include <tt-metalium/tt_metal.hpp>

// Optional: For verbose host-side tile verification prints.
#include <optional>

// Optional: For a delay between device kernel's termination and host kernel's tile verification prints (clean output).
#include <thread>
#include <chrono>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;
using namespace std;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// Each row of a row-major tensor is one page, so a row of page_tiles tiles gives a page size of page_tiles tiles.
TensorSpec MakeTensorSpecBFP16(uint32_t n_tiles, bool sram) {
    constexpr uint32_t tile_elements = TILE_WIDTH * TILE_HEIGHT;
    const uint32_t page_tiles = sram ? n_tiles : 1;
    return TensorSpec(
        Shape({n_tiles / page_tiles, page_tiles * tile_elements}),
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, (sram ? BufferType::L1 : BufferType::DRAM)}));
}

// The buffer is allocated from the top of memory (bottom_up = false) and then wrapped in a tensor.
MeshTensor MakeTensorBFP16(distributed::MeshDevice& mesh_device, uint32_t n_tiles, bool sram) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_WIDTH * TILE_HEIGHT;
    const uint32_t page_tiles = sram ? n_tiles : 1;
    const distributed::DeviceLocalBufferConfig device_local_config{
        .page_size = page_tiles * tile_size,
        .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM),
        .bottom_up = false};
    const distributed::ReplicatedBufferConfig buffer_config{.size = tile_size * n_tiles};
    return MeshTensor::from_buffer(
        std::move(*distributed::MeshBuffer::create(buffer_config, device_local_config, &mesh_device)),
        MakeTensorSpecBFP16(n_tiles, sram));
}

tt_metal::experimental::DataflowBufferSpec MakeDataflowBufferSpecBFP16(
    const tt_metal::experimental::DFBSpecName& dfb, uint32_t n_tiles) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_WIDTH * TILE_HEIGHT;
    return tt_metal::experimental::DataflowBufferSpec{
        .unique_id = dfb,
        .entry_size = tile_size,
        .num_entries = n_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
}

void VerifyDPRINTEnvironment(const std::string& program_path) {
    fmt::print("\n=========== DPRINT ENVIRONMENT CHECK ===========\n");
    const char* dprint_env = std::getenv("TT_METAL_DPRINT_CORES");
    if (!dprint_env || std::string(dprint_env).empty()) {
        fmt::print(stderr, "[WARNING] TT_METAL_DPRINT_CORES is not set.\n");
        fmt::print(stderr, "          Device-side DPRINT output will not appear.\n");
        fmt::print(stderr, "          To enable output, run:\n");
        fmt::print(stderr, "              export TT_METAL_DPRINT_CORES='(0,0)-(3,0)'\n");
    } else {
        fmt::print("[INFO] TT_METAL_DPRINT_CORES is set to: {}\n", dprint_env);
    }
    const char* prepend_env = std::getenv("TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC");
    if (!prepend_env || std::string(prepend_env).empty()) {
        fmt::print(stderr, "[WARNING] TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC is not set.\n");
        fmt::print(stderr, "          DPRINT output may be auto-prefixed with <device>:<core>:<risc>.\n");
        fmt::print(stderr, "          To disable prefixing, run:\n");
        fmt::print(stderr, "              export TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=0\n");
    } else {
        fmt::print("[INFO] TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC is set to: {}\n", prepend_env);
    }
    if (isatty(fileno(stderr))) {
        fmt::print(stderr, "[INFO] You are viewing DPRINT in the terminal.\n");
        fmt::print(stderr, "       For a clean and organized viewing experience, consider running with redirection:\n");
        fmt::print(stderr, "           {} &> multicast_dprint_out.txt\n", program_path);
    }
    fmt::print("================================================\n\n");
}

void verify_tiles(
    const std::vector<bfloat16>& golden_tile,
    const std::vector<bfloat16>& received_tiles,
    int num_tiles,
    bool verbose_verify) {
    fmt::print("\n=========== MULTICASTED TILE VERIFICATION ===========\n");

    ////////// LAMBDAS FOR ITERATING AND VERBOSE PRINTING TILES //////////
    auto tile_elem_idx = [](int tile, int i, int j) {
        return (tile * TILE_WIDTH * TILE_HEIGHT) + (i * TILE_WIDTH) + j;
    };

    auto print_tile =
        [&](const std::string& label, const std::vector<bfloat16>& data, std::optional<int> tile_num = std::nullopt) {
            if (tile_num.has_value()) {
                fmt::print("\n[{} TILE {}]\n", label, tile_num.value() + 1);
            } else {
                fmt::print("\n[{} TILE]\n", label);
            }

            for (int i = 0; i < TILE_HEIGHT; i++) {
                for (int j = 0; j < TILE_WIDTH; j++) {
                    int idx = tile_num.has_value() ? tile_elem_idx(tile_num.value(), i, j) : (i * TILE_WIDTH) + j;
                    fmt::print("{} ", static_cast<float>(data[idx]));
                }
                fmt::print("\n");
            }
        };

    ////////// VERIFICATION FLOW //////////
    if (verbose_verify) {
        print_tile("ORIGINAL", golden_tile);
    }

    bool all_match = true;  // verification loop primer.

    // iterate over all received tiles.
    for (int tile = 0; tile < num_tiles; tile++) {
        bool tile_match = true;
        if (verbose_verify) {
            print_tile("RECEIVED", received_tiles, tile);
        }
        // iterate over ith tile's elements, check whether they match with golden's elements.
        for (int i = 0; i < TILE_HEIGHT && tile_match; i++) {
            for (int j = 0; j < TILE_WIDTH; j++) {
                int idx = tile_elem_idx(tile, i, j);
                float received = static_cast<float>(received_tiles[idx]);
                float golden = static_cast<float>(golden_tile[(i * TILE_WIDTH) + j]);
                if (received != golden) {
                    tile_match = false;
                    break;
                }
            }
        }
        if (tile_match) {
            fmt::print("[✅ PASS] Receiver tile {} matches the golden tile.\n", tile + 1);
        } else {
            fmt::print("[❌ FAIL] Receiver tile {} does not match the golden tile.\n", tile + 1);
        }

        all_match &= tile_match;
    }
    if (all_match) {
        fmt::print("[✅ PASS] All {} receiver tiles match the golden tile.\n", num_tiles);
    } else {
        fmt::print("[❌ FAIL] One or more tiles did not match the golden tile.\n");
    }
    fmt::print("=====================================================\n\n");
}

int main(int /*argc*/, char** argv) {
    ////////// DEVICE SETUP //////////
    // A MeshDevice is a software concept that allows developers to virtualize a cluster of connected devices as a
    // single object, maintaining uniform memory and runtime state across all physical devices. A UnitMesh is a 1x1
    // MeshDevice that allows users to interface with a single physical device.
    int device_id = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    ////////// PROGRAM BLOCK //////////
    {
        distributed::MeshWorkload workload;
        const auto device_coord = distributed::MeshCoordinate(0, 0);

        ////////// TENSIX CORE SETUP //////////
        // Define logical sender core and receiver core range (for the kernel specs on the host).
        tt_metal::experimental::NodeRange all_cores_logical = tt_metal::experimental::NodeRange({0, 0}, {3, 0});
        tt_metal::experimental::NodeCoord sender_core_logical = {0, 0};
        tt_metal::experimental::NodeRange receiver_cores_logical = tt_metal::experimental::NodeRange({1, 0}, {3, 0});
        // Convert logical coordinates to physical coordinates (necessary for multicasting).
        CoreCoord sender_core_physical = mesh_device->worker_core_from_logical_core(sender_core_logical);
        CoreRange receiver_cores_physical = CoreRange(
            mesh_device->worker_core_from_logical_core(receiver_cores_logical.start_coord),
            mesh_device->worker_core_from_logical_core(receiver_cores_logical.end_coord));
        // Define physical sender core and receiver core range (for runtime arguments on the device).
        CoreCoord sender_core = sender_core_physical;
        CoreCoord receiver_core_start = receiver_cores_physical.start_coord;
        CoreCoord receiver_core_end = receiver_cores_physical.end_coord;
        // Grab the number of destinations, which will act as our "atomic counter" for semaphores, and as a general
        // reference for num of receivers.
        size_t num_dests = receiver_cores_logical.size();

        // Names identify the kernels, the dataflow buffers, the semaphores and the tensor parameter within the
        // program spec.
        const tt_metal::experimental::KernelSpecName coordinator_kernel{"coordinator"};
        const tt_metal::experimental::KernelSpecName inbound_kernel{"inbound"};
        const tt_metal::experimental::KernelSpecName outbound_kernel{"outbound"};
        const tt_metal::experimental::KernelSpecName void_compute_kernel{"void_compute"};
        const tt_metal::experimental::DFBSpecName in0_dfb{"in0"};
        const tt_metal::experimental::DFBSpecName out0_dfb{"out0"};
        const tt_metal::experimental::SemaphoreSpecName sender{"sender"};
        const tt_metal::experimental::SemaphoreSpecName receiver{"receiver"};
        const tt_metal::experimental::TensorParamName output_tensor_param{"output"};

        ////////// SEMAPHORE SETUP //////////
        tt_metal::experimental::SemaphoreSpec sender_semaphore_spec{
            .unique_id = sender, .target_nodes = all_cores_logical};
        tt_metal::experimental::SemaphoreSpec receiver_semaphore_spec{
            .unique_id = receiver, .target_nodes = all_cores_logical};

        ////////// DRAM & SRAM BUFFERS SETUP //////////
        constexpr uint32_t num_tiles = 1;
        uint32_t dram_bank_id = 0;
        TensorSpec src0_dram_tensor_spec = MakeTensorSpecBFP16(num_tiles, false);
        TensorSpec output_dram_tensor_spec = MakeTensorSpecBFP16(static_cast<uint32_t>(num_dests * num_tiles), false);
        MeshTensor src0_dram_tensor = MakeTensorBFP16(*mesh_device, num_tiles, false);
        MeshTensor output_dram_tensor =
            MakeTensorBFP16(*mesh_device, static_cast<uint32_t>(num_dests * num_tiles), false);
        tt_metal::experimental::DataflowBufferSpec in0_dfb_spec = MakeDataflowBufferSpecBFP16(in0_dfb, num_tiles);
        tt_metal::experimental::DataflowBufferSpec out0_dfb_spec = MakeDataflowBufferSpecBFP16(out0_dfb, num_tiles);

        ////////// DATA MOVEMENT CONFIG SETUP //////////
        tt_metal::experimental::DataMovementHardwareConfig DataMovementConfigIn{
            .config_1xx =
                tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
        };
        tt_metal::experimental::DataMovementHardwareConfig DataMovementConfigOut{
            .config_1xx =
                tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
        };

        ////////// COORDINATOR KERNEL SETUP //////////
        // The coordinator kernel both fills and reads the "in0" dataflow buffer on the sender core.
        tt_metal::experimental::KernelSpec coordinator_kernel_spec{
            .unique_id = coordinator_kernel,
            .source = "tt_metal/programming_examples/contributed/multicast/kernels/dataflow/coordinator_kernel.cpp",
            .dfb_bindings =
                {tt_metal::experimental::ProducerOf(in0_dfb, "in0"),
                 tt_metal::experimental::ConsumerOf(in0_dfb, "in0")},
            .semaphore_bindings =
                {{.semaphore_spec_name = sender, .accessor_name = "sender"},
                 {.semaphore_spec_name = receiver, .accessor_name = "receiver"}},
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"start_x",
                      "start_y",
                      "end_x",
                      "end_y",
                      "dram_bank_id",
                      "src0_dram",
                      "single_tile_size",
                      "num_dests"}},
            .hw_config = DataMovementConfigIn,
        };

        ////////// DATAFLOW KERNELS SETUP //////////
        // The inbound kernel both fills and reads the "in0" dataflow buffer on each receiver core.
        tt_metal::experimental::KernelSpec inbound_kernel_spec{
            .unique_id = inbound_kernel,
            .source = "tt_metal/programming_examples/contributed/multicast/kernels/dataflow/inbound_kernel.cpp",
            .dfb_bindings =
                {tt_metal::experimental::ProducerOf(in0_dfb, "in0"),
                 tt_metal::experimental::ConsumerOf(in0_dfb, "in0"),
                 tt_metal::experimental::ProducerOf(out0_dfb, "out0")},
            .semaphore_bindings =
                {{.semaphore_spec_name = sender, .accessor_name = "sender"},
                 {.semaphore_spec_name = receiver, .accessor_name = "receiver"}},
            .runtime_arg_schema = {.runtime_arg_names = {"start_x", "start_y"}},
            .hw_config = DataMovementConfigIn,
        };
        tt_metal::experimental::KernelSpec outbound_kernel_spec{
            .unique_id = outbound_kernel,
            .source = "tt_metal/programming_examples/contributed/multicast/kernels/dataflow/outbound_kernel.cpp",
            .dfb_bindings = {tt_metal::experimental::ConsumerOf(out0_dfb, "out0")},
            .tensor_bindings = {{.tensor_parameter_name = output_tensor_param, .accessor_name = "dst"}},
            .runtime_arg_schema = {.runtime_arg_names = {"tile_offset"}},
            .hw_config = DataMovementConfigOut,
        };

        ////////// COMPUTE KERNEL SETUP //////////
        tt_metal::experimental::KernelSpec void_compute_kernel_spec{
            .unique_id = void_compute_kernel,
            .source = "tt_metal/programming_examples/contributed/multicast/kernels/compute/void_compute_kernel.cpp",
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .hw_config =
                tt_metal::experimental::ComputeHardwareConfig{
                    .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                    .sfpu_precision_mode = Precision::Precise,
                    .enable_32_bit_dest = false,
                },
        };

        // A program spec describes the program: its kernels, the dataflow buffers and semaphores they use, the
        // tensor they write, and the cores the kernels run on (the work units).
        tt_metal::experimental::ProgramSpec program_spec{
            .name = "multicast",
            .kernels = {coordinator_kernel_spec, inbound_kernel_spec, outbound_kernel_spec, void_compute_kernel_spec},
            .dataflow_buffers = {in0_dfb_spec, out0_dfb_spec},
            .semaphores = {sender_semaphore_spec, receiver_semaphore_spec},
            .tensor_parameters = {{.unique_id = output_tensor_param, .spec = output_dram_tensor_spec}},
            .work_units =
                {{.name = "coordinator", .kernels = {coordinator_kernel}, .target_nodes = sender_core_logical},
                 {.name = "receivers",
                  .kernels = {inbound_kernel, outbound_kernel, void_compute_kernel},
                  .target_nodes = receiver_cores_logical}},
        };
        Program program = tt_metal::experimental::MakeProgramFromSpec(*mesh_device, program_spec);

        ////////// IDENTITY MATRIX TILE SETUP //////////
        std::vector<bfloat16> identity_tile = create_identity_matrix(TILE_WIDTH, TILE_HEIGHT, TILE_WIDTH);
        HostTensor src0_host_tensor = HostTensor::from_vector(identity_tile, src0_dram_tensor_spec);
        cq.enqueue_write_tensor(src0_host_tensor, src0_dram_tensor);

        ////////// RUNTIME ARGS SETUP //////////
        // Args for the sender core to multicast tile.
        // They must have access to coordinates of all receiver cores, to execute multicast operation.
        tt_metal::experimental::KernelRunArgs coordinator_run_args{
            .kernel = coordinator_kernel,
            .runtime_arg_values = tt_metal::experimental::MakeRuntimeArgsForSingleNode(
                sender_core_logical,
                {{"start_x", (uint32_t)(receiver_core_start.x)},
                 {"start_y", (uint32_t)(receiver_core_start.y)},
                 {"end_x", (uint32_t)(receiver_core_end.x)},
                 {"end_y", (uint32_t)(receiver_core_end.y)},
                 {"dram_bank_id", dram_bank_id},
                 {"src0_dram", static_cast<uint32_t>(src0_dram_tensor.address())},
                 {"single_tile_size", static_cast<uint32_t>(sizeof(bfloat16) * TILE_HW)},
                 {"num_dests", static_cast<uint32_t>(num_dests)}}),
        };
        // Args for the receiver cores to receive tile.
        // They must have access to coordinates of the sender core, to listen for multicast operation.
        tt_metal::experimental::KernelRunArgs inbound_run_args{.kernel = inbound_kernel};
        for (const tt_metal::experimental::NodeCoord& core : receiver_cores_logical) {
            tt_metal::experimental::AddRuntimeArgsForNode(
                inbound_run_args.runtime_arg_values,
                core,
                {{"start_x", (uint32_t)(sender_core.x)}, {"start_y", (uint32_t)(sender_core.y)}});
        }

        // Args for the receiver cores to send tile back to DRAM.
        tt_metal::experimental::KernelRunArgs outbound_run_args{.kernel = outbound_kernel};
        int tile_index = 0;
        for (const tt_metal::experimental::NodeCoord& core : receiver_cores_logical) {
            tt_metal::experimental::AddRuntimeArgsForNode(
                outbound_run_args.runtime_arg_values, core, {{"tile_offset", static_cast<uint32_t>(tile_index)}});
            tile_index++;
        }

        // The output tensor is bound to the outbound kernel's tensor parameter.
        tt_metal::experimental::ProgramRunArgs run_args{
            .kernel_run_args = {coordinator_run_args, inbound_run_args, outbound_run_args},
            .tensor_args = {{output_tensor_param, tt_metal::experimental::TensorArgument{output_dram_tensor}}},
        };
        tt_metal::experimental::SetProgramRunArgs(program, run_args);

        ////////// PROGRAM LAUNCH AND CLOSE //////////
        VerifyDPRINTEnvironment(argv[0]);
        fmt::print("Launching program\n");
        fmt::print(
            "Hello, Core ({}, {}) on Device {}, please multicast the tile to your neighbors.\n",
            sender_core_logical.x,
            sender_core_logical.y,
            device_id);
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        fmt::print("Waiting until program finishes\n");
        distributed::Finish(cq);
        fmt::print(
            "Thank you, Core ({}, {}) on Device {}, for the multicast.\n",
            sender_core_logical.x,
            sender_core_logical.y,
            device_id);

        // introduce artificial delay to avoid jumbling host-side tile verification prints with potentially echoing
        // device-side DPRINTs.  Delay may be adjusted or removed as needed.
        this_thread::sleep_for(chrono::milliseconds(200));

        ////////// TILE MULTICAST VERIFICATION //////////
        // We're reading from a shard allocated on Device Coordinate 0, 0, since this is a 1x1
        //  When the MeshDevice is 2 dimensional, the MeshBuffer read API can be used to target specific physical
        //  devices
        std::vector<bfloat16> received_tiles =
            cq.enqueue_read_tensor(output_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();
        bool verbose_verify =
            false;  // if enabled, the original and all multicast-received tiles are printed in full (32x32).
        verify_tiles(identity_tile, received_tiles, num_dests, verbose_verify);
    }

    mesh_device->close();
    return 0;
}
