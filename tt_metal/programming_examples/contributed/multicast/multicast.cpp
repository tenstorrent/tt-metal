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
using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

std::shared_ptr<distributed::MeshBuffer> MakeBufferBFP16(
    std::shared_ptr<distributed::MeshDevice> mesh_device, uint32_t n_tiles, bool sram) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_WIDTH * TILE_HEIGHT;
    const uint32_t page_tiles = sram ? n_tiles : 1;
    const distributed::DeviceLocalBufferConfig device_local_config{
        .page_size = page_tiles * tile_size,
        .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM),
        .bottom_up = false};
    const distributed::ReplicatedBufferConfig buffer_config{.size = tile_size * n_tiles};
    return distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());
}

CBHandle MakeCircularBufferBFP16(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_WIDTH * TILE_HEIGHT;
    return CreateCircularBuffer(program, core, CircularBufferConfig(n_tiles * tile_size, {{cb, tt::DataFormat::Float16_b}}).set_page_size(cb, tile_size));
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
    auto tile_elem_idx = [](int tile, int i, int j) { return tile * TILE_WIDTH * TILE_HEIGHT + i * TILE_WIDTH + j; };

    auto print_tile =
        [&](const std::string& label, const std::vector<bfloat16>& data, std::optional<int> tile_num = std::nullopt) {
            if (tile_num.has_value()) {
                fmt::print("\n[{} TILE {}]\n", label, tile_num.value() + 1);
            } else {
                fmt::print("\n[{} TILE]\n", label);
            }

            for (int i = 0; i < TILE_HEIGHT; i++) {
                for (int j = 0; j < TILE_WIDTH; j++) {
                    int idx = tile_num.has_value() ? tile_elem_idx(tile_num.value(), i, j) : i * TILE_WIDTH + j;
                    fmt::print("{} ", data[idx].to_float());
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
                float received = received_tiles[idx].to_float();
                float golden = golden_tile[i * TILE_WIDTH + j].to_float();
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

int main(int argc, char **argv) {

    ////////// DEVICE SETUP //////////
    //A MeshDevice is a software concept that allows developers to virtualize a cluster of connected devices as a single object,
    // maintaining uniform memory and runtime state across all physical devices.
    //A UnitMesh is a 1x1 MeshDevice that allows users to interface with a single physical device.
    int device_id = 0;
    auto enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        device_id, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreType::WORKER);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    ////////// PROGRAM BLOCK //////////
    {
        distributed::MeshWorkload workload;
        Program program = CreateProgram();
        const auto device_coord = distributed::MeshCoordinate(0, 0);

        ////////// TENSIX CORE SETUP //////////
        // Define logical sender core and receiver core range (for kernel creation on the host).
        CoreRange all_cores_logical = CoreRange({0, 0}, {3, 0});
        CoreCoord sender_core_logical = {0, 0};
        CoreRange receiver_cores_logical = CoreRange({1, 0}, {3, 0});
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

        ////////// DATA MOVEMENT CONFIG SETUP //////////
        DataMovementConfig DataMovementConfigIn = {.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};
        DataMovementConfig DataMovementConfigOut = {.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default};

        ////////// COORDINATOR KERNEL SETUP //////////
        KernelHandle coordinator_kernel_id = CreateKernel(
            program,
            "tt_metal/programming_examples/contributed/multicast/kernels/dataflow/coordinator_kernel.cpp",
            sender_core_logical,
            DataMovementConfigIn);

        ////////// DATAFLOW KERNELS SETUP //////////
        KernelHandle inbound_kernel_id = CreateKernel(
            program,
            "tt_metal/programming_examples/contributed/multicast/kernels/dataflow/inbound_kernel.cpp",
            receiver_cores_logical,
            DataMovementConfigIn);
        KernelHandle outbound_kernel_id = CreateKernel(
            program,
            "tt_metal/programming_examples/contributed/multicast/kernels/dataflow/outbound_kernel.cpp",
            receiver_cores_logical,
            DataMovementConfigOut);

        ////////// COMPUTE KERNEL SETUP //////////
        vector<uint32_t> compute_kernel_args = {};
        KernelHandle comp_kernel_id = CreateKernel(
            program,
            "tt_metal/programming_examples/contributed/multicast/kernels/compute/void_compute_kernel.cpp",
            receiver_cores_logical,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = compute_kernel_args});

        ////////// SEMAPHORE SETUP //////////
        uint32_t sender = CreateSemaphore(program, all_cores_logical, 0);
        uint32_t receiver = CreateSemaphore(program, all_cores_logical, 0);

        ////////// DRAM & SRAM BUFFERS SETUP //////////
        constexpr uint32_t num_tiles = 1;
        uint32_t dram_bank_id = 0;
        auto src0_dram_buffer = MakeBufferBFP16(mesh_device, num_tiles, false);
        auto output_dram_buffer = MakeBufferBFP16(mesh_device, num_dests * num_tiles, false);
        auto cb_src0 = MakeCircularBufferBFP16(program, all_cores_logical, tt::CBIndex::c_0, num_tiles);
        auto cb_output = MakeCircularBufferBFP16(program, all_cores_logical, tt::CBIndex::c_16, num_tiles);

        ////////// IDENTITY MATRIX TILE SETUP //////////
        std::vector<bfloat16> identity_tile = create_identity_matrix(TILE_WIDTH, TILE_HEIGHT, TILE_WIDTH);
        distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, identity_tile);

        ////////// RUNTIME ARGS SETUP //////////
        // Args for the sender core to multicast tile.
        // They must have access to coordinates of all receiver cores, to execute multicast operation.
        SetRuntimeArgs(
            program,
            coordinator_kernel_id,
            sender_core_logical,
            {(uint32_t)(receiver_core_start.x),
             (uint32_t)(receiver_core_start.y),
             (uint32_t)(receiver_core_end.x),
             (uint32_t)(receiver_core_end.y),
             sender,
             receiver,
             dram_bank_id,
             src0_dram_buffer->address(),
             sizeof(bfloat16) * TILE_HW,
             num_dests});
        // Args for the receiver cores to receive tile.
        // They must have access to coordinates of the sender core, to listen for multicast operation.
        SetRuntimeArgs(
            program,
            inbound_kernel_id,
            receiver_cores_logical,
            {(uint32_t)(sender_core.x), (uint32_t)(sender_core.y), sender, receiver});

        // Args for the receiver cores to send tile back to DRAM.
        int tile_index = 0;
        for (const CoreCoord& core : receiver_cores_logical) {
            SetRuntimeArgs(
                program, outbound_kernel_id, core, {output_dram_buffer->address(), static_cast<uint32_t>(tile_index)});
            tile_index++;
        }

        ////////// PROGRAM LAUNCH AND CLOSE //////////
        VerifyDPRINTEnvironment(argv[0]);
        fmt::print("Launching program\n");
        fmt::print(
            "Hello, Core ({}, {}) on Device {}, please multicast the tile to your neighbors.\n",
            sender_core_logical.x,
            sender_core_logical.y,
            device_id);
        distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
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
        std::vector<bfloat16> received_tiles(num_dests * TILE_HW);

        // We're reading from a shard allocated on Device Coordinate 0, 0, since this is a 1x1
        //  When the MeshDevice is 2 dimensional, this API can be used to target specific physical devices
        distributed::ReadShard(cq, received_tiles, output_dram_buffer, device_coord);
        bool verbose_verify =
            false;  // if enabled, the original and all multicast-received tiles are printed in full (32x32).
        verify_tiles(identity_tile, received_tiles, num_dests, verbose_verify);
    }

    mesh_device.reset();
    return 0;
}
