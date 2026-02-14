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
#include <tt-metalium/experimental/ez/ez.hpp>

#include <cstdint>
#include <vector>
#include <iostream>

// Optional: For verbose host-side tile verification prints.
#include <optional>

// Optional: For a delay between device kernel's termination and host kernel's tile verification prints (clean output).
#include <thread>
#include <chrono>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;
using namespace tt::tt_metal::experimental::ez;
using namespace std;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

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
    // DeviceContext wraps MeshDevice creation, command queue, and teardown in RAII.
    // A UnitMesh is a 1x1 MeshDevice that interfaces with a single physical device.
    int device_id = 0;
    DeviceContext ctx(device_id);

    ////////// TENSIX CORE SETUP //////////
    // Define logical sender core and receiver core range (for kernel creation on the host).
    CoreRange all_cores_logical = CoreRange({0, 0}, {3, 0});
    CoreCoord sender_core_logical = {0, 0};
    CoreRange receiver_cores_logical = CoreRange({1, 0}, {3, 0});
    // Convert logical coordinates to physical coordinates (necessary for multicasting).
    // Multicast NoC operations require physical addresses, not logical core coordinates.
    CoreCoord sender_core_physical = ctx.physical_core(sender_core_logical);
    CoreCoord receiver_start_physical = ctx.physical_core(receiver_cores_logical.start_coord);
    CoreCoord receiver_end_physical = ctx.physical_core(receiver_cores_logical.end_coord);
    // Grab the number of destinations, which will act as our "atomic counter" for semaphores, and as a general
    // reference for num of receivers.
    size_t num_dests = receiver_cores_logical.size();

    ////////// DRAM & SRAM BUFFERS SETUP //////////
    constexpr uint32_t num_tiles = 1;
    uint32_t dram_bank_id = 0;
    auto src0_dram_buffer = ctx.dram_tile_buffer(num_tiles);
    auto output_dram_buffer = ctx.dram_tile_buffer(num_dests * num_tiles);

    ////////// PROGRAM SETUP //////////
    // Build the program with circular buffers, semaphores, and kernels.
    // Circular buffers c_0 and c_16 are created on all cores (sender + receivers).
    auto builder = ProgramBuilder(all_cores_logical);
    builder.cb(tt::CBIndex::c_0, num_tiles)
        .cb(tt::CBIndex::c_16, num_tiles);

    ////////// SEMAPHORE SETUP //////////
    // Semaphores coordinate the multicast handshake between sender and receivers.
    // The sender signals when data is ready; receivers signal when they've consumed it.
    uint32_t sender_sem = builder.semaphore();
    uint32_t receiver_sem = builder.semaphore();

    ////////// COORDINATOR KERNEL SETUP //////////
    // The coordinator (sender) reads a tile from DRAM into cb_0, then multicasts it to all receiver cores.
    // We use .kernel() with explicit RISCV_0/NOC_0 to match the original: multicast is NOC-specific,
    // and the coordinator does not use TensorAccessorArgs (it reads DRAM via direct bank addressing).
    builder.on(sender_core_logical)
        .kernel(
            OVERRIDE_KERNEL_PREFIX "contributed/multicast/kernels/dataflow/coordinator_kernel.cpp",
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default})
        .runtime_args(
            {(uint32_t)(receiver_start_physical.x),
             (uint32_t)(receiver_start_physical.y),
             (uint32_t)(receiver_end_physical.x),
             (uint32_t)(receiver_end_physical.y),
             sender_sem,
             receiver_sem,
             dram_bank_id,
             src0_dram_buffer->address(),
             sizeof(bfloat16) * TILE_HW,
             (uint32_t)num_dests})
        .done()

        ////////// DATAFLOW KERNELS SETUP //////////
        // Inbound kernel: receiver cores wait for the multicast from the sender core.
        .on(receiver_cores_logical)
        .reader(
            OVERRIDE_KERNEL_PREFIX "contributed/multicast/kernels/dataflow/inbound_kernel.cpp")
        .runtime_args(
            {(uint32_t)(sender_core_physical.x),
             (uint32_t)(sender_core_physical.y),
             sender_sem,
             receiver_sem})
        .done();

    // Outbound kernel: receiver cores write the received tile from cb_16 back to DRAM.
    // Per-core runtime args: each receiver writes to a different tile index.
    auto& outbound_ref = builder.on(receiver_cores_logical)
        .writer(
            OVERRIDE_KERNEL_PREFIX "contributed/multicast/kernels/dataflow/outbound_kernel.cpp",
            {output_dram_buffer});
    uint32_t tile_index = 0;
    for (const CoreCoord& core : receiver_cores_logical) {
        outbound_ref.runtime_args_at(core, {output_dram_buffer->address(), tile_index});
        tile_index++;
    }

    ////////// COMPUTE KERNEL SETUP //////////
    // Void compute kernel on receiver cores — placeholder for the compute pipeline stage.
    builder.on(receiver_cores_logical)
        .compute(
            OVERRIDE_KERNEL_PREFIX "contributed/multicast/kernels/compute/void_compute_kernel.cpp",
            MathFidelity::HiFi4);

    ////////// IDENTITY MATRIX TILE SETUP //////////
    std::vector<bfloat16> identity_tile = create_identity_matrix(TILE_WIDTH, TILE_HEIGHT, TILE_WIDTH);
    ctx.write(src0_dram_buffer, identity_tile);

    ////////// PROGRAM LAUNCH AND CLOSE //////////
    VerifyDPRINTEnvironment(argv[0]);
    fmt::print("Launching program\n");
    fmt::print(
        "Hello, Core ({}, {}) on Device {}, please multicast the tile to your neighbors.\n",
        sender_core_logical.x,
        sender_core_logical.y,
        device_id);
    ctx.launch(builder.build());
    fmt::print("Waiting until program finishes\n");
    ctx.finish();
    fmt::print(
        "Thank you, Core ({}, {}) on Device {}, for the multicast.\n",
        sender_core_logical.x,
        sender_core_logical.y,
        device_id);

    // introduce artificial delay to avoid jumbling host-side tile verification prints with potentially echoing
    // device-side DPRINTs.  Delay may be adjusted or removed as needed.
    this_thread::sleep_for(chrono::milliseconds(200));

    ////////// TILE MULTICAST VERIFICATION //////////
    auto received_tiles = ctx.read<bfloat16>(output_dram_buffer);
    bool verbose_verify =
        false;  // if enabled, the original and all multicast-received tiles are printed in full (32x32).
    verify_tiles(identity_tile, received_tiles, num_dests, verbose_verify);

    return 0;
}
