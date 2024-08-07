// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include <stdint.h>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;
using namespace std;

int main(int argc, char **argv) {

    ////////// DEVICE SETUP //////////
    int device_id = 0;
    Device *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    ////////// TENSIX CORE SETUP //////////
    constexpr CoreCoord coordinator_core_logical = {0, 0};
    CoreRange all_cores({0, 0}, {3, 0});
    CoreRange coordinate_range_logical({1, 0}, {3, 0});
    std::vector<CoreCoord> coordinate_range_logical_vec = {{0,0}, {1, 0}, {2, 0}, {3, 0}};
    std::vector<CoreCoord> coordinate_range_physical = device->worker_cores_from_logical_cores(coordinate_range_logical_vec);

    ////////// DATA MOVEMENT CONFIG SETUP //////////
    DataMovementConfig DataMovementConfigIn = {.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};
    DataMovementConfig DataMovementConfigOut = {.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default};

    ////////// COORDINATOR KERNEL SETUP //////////
    KernelHandle coordinator_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/multicast/kernels/dataflow/coordinator_kernel.cpp",
        coordinator_core_logical,
        DataMovementConfigIn);

    ////////// DATAFLOW KERNELS SETUP //////////
    KernelHandle inbound_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/multicast/kernels/dataflow/inbound_kernel.cpp",
        coordinate_range_logical,
        DataMovementConfigIn);

    KernelHandle outbound_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/multicast/kernels/dataflow/outbound_kernel.cpp",
        coordinate_range_logical,
        DataMovementConfigOut);

    ////////// COMPUTE KERNEL SETUP //////////
    vector<uint32_t> compute_kernel_args = {};
    KernelHandle comp_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/multicast/kernels/compute/void_compute_kernel.cpp",
        coordinate_range_logical,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args
        }
    );

    ////////// SEMAPHORE SETUP //////////
    uint32_t kernel_start_ac_initial_val = 0;

    uint32_t sender = CreateSemaphore(
        program,
        all_cores,
        kernel_start_ac_initial_val
    );

    uint32_t receiver = CreateSemaphore(
        program,
        all_cores,
        kernel_start_ac_initial_val
    );

    ////////// DRAM & SRAM (CB) BUFFERS SETUP //////////
    // Define single tile size and create identity tile
    uint32_t single_tile_size = 2 * 32 * 32; // bytes
    std::vector<bfloat16> identity_tile = create_identity_matrix(32, 32, 32);

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };
    std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);

    EnqueueWriteBuffer(cq, src0_dram_buffer, identity_tile.data(), false);

    // Setup circular buffer parameters
    uint32_t num_input_tiles = 1;
    uint32_t num_output_tiles = 1;
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;

    // Create input circular buffers
    uint32_t src0_cb_index = CB::c_in0; // 0
    CircularBufferConfig config_cb_in = CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
        .set_page_size(src0_cb_index, single_tile_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, config_cb_in);

    // Create output circular buffer (not used in this example)
    uint32_t output_cb_index = CB::c_out0; // output operands start at index 16
    CircularBufferConfig config_cb_out = CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
        .set_page_size(output_cb_index, single_tile_size);
    CBHandle cb_output = tt_metal::CreateCircularBuffer(program, all_cores, config_cb_out);

    ////////// RUNTIME ARGS SETUP //////////
    // Set runtime arguments for coordinator kernel
    tt_metal::SetRuntimeArgs(
        program,
        coordinator_kernel_id,
        coordinator_core_logical,
        {
            static_cast<uint32_t>(coordinate_range_physical[1].x), // worker start_x
            static_cast<uint32_t>(coordinate_range_physical[1].y), // worker start_y
            static_cast<uint32_t>(coordinate_range_physical[3].x), // worker end_x
            static_cast<uint32_t>(coordinate_range_physical[3].y), // worker end_y
            sender,
            receiver,
            src0_dram_buffer->address(),
            single_tile_size
        }
    );

    // Set runtime arguments for inbound kernels
    tt_metal::SetRuntimeArgs(
        program,
        inbound_kernel_id,
        coordinate_range_logical,
        {
            static_cast<uint32_t>(coordinate_range_physical[0].x), // coordinator start_x
            static_cast<uint32_t>(coordinate_range_physical[0].y), // coordinator start_y
            sender,
            receiver
        }
    );

    ////////// PROGRAM LAUNCH AND CLOSE //////////
    printf("Launching program\n");
    EnqueueProgram(cq, program, false);

    printf("Hello, Core {0, 0} on Device 0, please synchronize your neighbor cores\n");

    printf("Waiting until program finishes\n");

    Finish(cq);

    printf("Thank you, Core {0, 0} on Device 0, for your synchronization.\n");

    printf("Closing device\n");
    CloseDevice(device);

    return 0;
}
