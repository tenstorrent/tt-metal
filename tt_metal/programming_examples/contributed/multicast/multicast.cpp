// SPDX-FileCopyrightText: © 2025 Ryan Barton
//
// SPDX-License-Identifier: Apache-2.0

/*
 This example demonstrates how to create a 32x32 data tile and multicast it from a sender (coordinator) core to multiple receiver cores.
 It covers the setup of semaphores and multicore addressing on Tenstorrent hardware.

 To view & verify the multicasted tile output on the device, set the following environment variable before execution:

     export TT_METAL_DPRINT_CORES='(0,0)-(3,0)'

 In the original configuration, (0,0) is the sender core, while (1,0), (2,0), and (3,0) are the receiver cores.
 The user can modify these coordinates as desired.
*/

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device_impl.hpp>
#include <cstdint>
#include <vector>
#include <memory>
#include <iostream>

using namespace tt;
using namespace tt::tt_metal;
using namespace std;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

std::shared_ptr<Buffer> MakeBufferBFP16(IDevice* device, uint32_t n_tiles, bool sram) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    const uint32_t page_tiles = sram ? n_tiles : 1;
    return CreateBuffer({.device = device, .size = tile_size * n_tiles, .page_size = page_tiles * tile_size, .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM)});
}

CBHandle MakeCircularBufferBFP16(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    return CreateCircularBuffer(program, core, CircularBufferConfig(n_tiles * tile_size, {{cb, tt::DataFormat::Float16_b}}).set_page_size(cb, tile_size));
}

int main(int argc, char **argv) {

    ////////// DEVICE SETUP //////////
    int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    
    ////////// PROGRAM BLOCK //////////
    { 
        Program program = CreateProgram();

        ////////// TENSIX CORE SETUP //////////
        // Define logical sender core and receiver core range (for kernel creation on the host)
        CoreRange all_cores_logical({0, 0}, {3, 0});
        CoreCoord sender_core_logical = {0, 0};
        CoreRange receiver_cores_logical({1, 0}, {3, 0});
        // Convert logical coordinates to physical coordinates (necessary for multicasting)
        CoreCoord sender_core_physical = 
            device->worker_core_from_logical_core(sender_core_logical);
        CoreRange receiver_cores_physical(
            device->worker_core_from_logical_core(receiver_cores_logical.start_coord),
            device->worker_core_from_logical_core(receiver_cores_logical.end_coord)
        );
        // Define physical sender core and receiver core range (for runtime arguments on the device)
        CoreCoord sender_core = sender_core_physical;
        CoreCoord receiver_core_start = receiver_cores_physical.start_coord;
        CoreCoord receiver_core_end = receiver_cores_physical.end_coord;
        // Grab the number of destinations, which will act as our "atomic counter" for semaphores
        size_t num_dests = receiver_cores_logical.size();

        ////////// DATA MOVEMENT CONFIG SETUP //////////
        DataMovementConfig DataMovementConfigIn = {.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};
        DataMovementConfig DataMovementConfigOut = {.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default};

        ////////// COORDINATOR KERNEL SETUP //////////
        KernelHandle coordinator_kernel_id = CreateKernel(
            program,
            "tt_metal/programming_examples/contributed/multicast/kernels/dataflow/coordinator_kernel.cpp",
            sender_core_logical,
            DataMovementConfigIn
        );

        ////////// DATAFLOW KERNELS SETUP //////////
        KernelHandle inbound_kernel_id = CreateKernel(
            program,
            "tt_metal/programming_examples/contributed/multicast/kernels/dataflow/inbound_kernel.cpp",
            receiver_cores_logical,
            DataMovementConfigIn
        );
        KernelHandle outbound_kernel_id = CreateKernel(
            program,
            "tt_metal/programming_examples/contributed/multicast/kernels/dataflow/outbound_kernel.cpp",
            receiver_cores_logical,
            DataMovementConfigOut
        );

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
                .compile_args = compute_kernel_args
            }
        );

        ////////// SEMAPHORE SETUP //////////
        uint32_t sender = CreateSemaphore(program, all_cores_logical, 0);
        uint32_t receiver = CreateSemaphore(program, all_cores_logical, 0);

        ////////// DRAM & SRAM BUFFERS SETUP //////////
        const uint32_t num_tiles = 1;
        uint32_t dram_bank_id = 0;
        auto src0_dram_buffer = MakeBufferBFP16(device, num_tiles, false);
        auto cb_src0 = MakeCircularBufferBFP16(program, all_cores_logical, tt::CBIndex::c_0, num_tiles);
        auto cb_output = MakeCircularBufferBFP16(program, all_cores_logical, tt::CBIndex::c_16, num_tiles);

        ////////// IDENTITY MATRIX TILE SETUP //////////
        std::vector<bfloat16> identity_tile = create_identity_matrix(32, 32, 32);
        EnqueueWriteBuffer(cq, src0_dram_buffer, identity_tile.data(), false);
        // Verification
        std::vector<bfloat16> identity_tile_reload;
        EnqueueReadBuffer(cq, src0_dram_buffer, identity_tile_reload, true);
        std::cout << "-VERIFIED- 32x32 identity tile is stored in DRAM:\n";
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                std::cout << identity_tile_reload[i * 32 + j].to_float() << " ";
            }
            std::cout << std::endl;
        }

        ////////// RUNTIME ARGS SETUP //////////
        // Args for the sender core. 
        // They must have access to coordinates of all receiver cores, to execute multicast operation.
        SetRuntimeArgs(program, coordinator_kernel_id, sender_core_logical, {
            (uint32_t)(receiver_core_start.x),
            (uint32_t)(receiver_core_start.y),
            (uint32_t)(receiver_core_end.x),
            (uint32_t)(receiver_core_end.y),
            sender, receiver,
            dram_bank_id,
            src0_dram_buffer->address(),
            sizeof(bfloat16) * 32 * 32,
            num_dests
        });
        // Args for the receiver cores.
        // They must have access to coordinates of the sender core, to listen for multicast operation.
        SetRuntimeArgs(program, inbound_kernel_id, receiver_cores_logical, {
            (uint32_t)(sender_core.x),
            (uint32_t)(sender_core.y),
            sender, receiver
        });

        ////////// PROGRAM LAUNCH AND CLOSE //////////
        printf("Launching program\n");
        EnqueueProgram(cq, program, false);
        printf("Hello, Core {0, 0} on Device 0, please synchronize your neighbor cores\n");
        printf("Waiting until program finishes\n");
        Finish(cq);
        printf("Thank you, Core {0, 0} on Device 0, for your synchronization.\n");
    }

    CloseDevice(device);
    
    return 0;
}
