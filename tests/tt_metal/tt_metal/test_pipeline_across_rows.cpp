#include <algorithm>
#include <functional>
#include <random>
#include <cstdlib> // for putenv()

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include "hostdevcommon/common_values.hpp"

using namespace tt;

// helper l1 allocator (same addrs across all cores, real alloc to be per-core)
uint32_t l1_alloc(uint32_t size_in_bytes) {
    constexpr uint32_t L1_SIZE = 1024*1024;
    static uint32_t l1_alloc_addr = UNRESERVED_BASE;

    uint32_t addr = l1_alloc_addr;
    TT_ASSERT(addr % 32 == 0); // 32-byte aligned to allow NOC transfers to any allocated address
    TT_ASSERT(addr + size_in_bytes <= L1_SIZE); // need to fit into L1

    l1_alloc_addr += size_in_bytes;

    return addr;
}

int main(int argc, char **argv) {
    bool pass = true;

    // Once this test is uplifted to use fast dispatch, this can be removed.
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    try {
        std::string env_var = "TT_PCI_DMA_BUF_SIZE=1048576";
        int result = putenv(const_cast<char*>(env_var.c_str()));

        ////////////////////////////////////////////////////////////////////////////
        //                      Initial Runtime Args Parse
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> input_args(argv, argv + argc);
        string arch_name = "";
        try {
            std::tie(arch_name, input_args) =
                test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
        } catch (const std::exception& e) {
            log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
        }
        const tt::ARCH arch = tt::get_arch_from_string(arch_name);
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(arch, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();

        // set up the program

        // saturate DRAM
        uint32_t num_cores = 12;
        uint32_t num_tiles = 64 * 1024;
        uint32_t block_size_tiles = 16;
        uint32_t num_blocks_in_CB = 2;
        uint32_t IO_data_in_dram = true;
        uint32_t num_repetitions = 1;

        // saturate L1
        // uint32_t num_cores = 10;
        // uint32_t num_tiles = 384;
        // uint32_t block_size_tiles = 16;
        // uint32_t num_blocks_in_CB = 2;
        // uint32_t IO_data_in_dram = false;
        // uint32_t num_repetitions = 64;

        // test #1
        // uint32_t num_cores = 12;
        // uint32_t num_tiles = 384;
        // uint32_t block_size_tiles = 1;
        // uint32_t num_blocks_in_CB = 16;
        // uint32_t IO_data_in_dram = false;
        // uint32_t num_repetitions = 128;

        // test #2
        // uint32_t num_cores = 12;
        // uint32_t num_tiles = 384;
        // uint32_t block_size_tiles = 2;
        // uint32_t num_blocks_in_CB = 16;
        // uint32_t IO_data_in_dram = false;
        // uint32_t num_repetitions = 128;

        // test #3
        // uint32_t num_cores = 12;
        // uint32_t num_tiles = 384;
        // uint32_t block_size_tiles = 4;
        // uint32_t num_blocks_in_CB = 16;
        // uint32_t IO_data_in_dram = false;
        // uint32_t num_repetitions = 128;

        // test #5
        // uint32_t num_cores = 12;
        // uint32_t num_tiles = 384;
        // uint32_t block_size_tiles = 8;
        // uint32_t num_blocks_in_CB = 8;
        // uint32_t IO_data_in_dram = false;
        // uint32_t num_repetitions = 128;

        // test #6
        // uint32_t num_cores = 12;
        // uint32_t num_tiles = 384;
        // uint32_t block_size_tiles = 16;
        // uint32_t num_blocks_in_CB = 4;
        // uint32_t IO_data_in_dram = false;
        // uint32_t num_repetitions = 128;

        // test #7
        // uint32_t num_cores = 12;
        // // uint32_t num_tiles = 256;
        // // uint32_t block_size_tiles = 32;
        // // uint32_t num_blocks_in_CB = 4;
        // // uint32_t IO_data_in_dram = false;
        // // uint32_t num_repetitions = 256;

        // test #8
        // uint32_t num_cores = 12;
        // uint32_t num_tiles = 128;
        // uint32_t block_size_tiles = 64;
        // uint32_t num_blocks_in_CB = 4;
        // uint32_t IO_data_in_dram = false;
        // uint32_t num_repetitions = 512;

        // test #9
        // uint32_t num_cores = 12;
        // uint32_t num_tiles = 128;
        // uint32_t block_size_tiles = 128;
        // uint32_t num_blocks_in_CB = 2;
        // uint32_t IO_data_in_dram = false;
        // uint32_t num_repetitions = 512;


        TT_ASSERT(num_cores >= 2 && num_cores <= 12); // grayskull
        TT_ASSERT(num_tiles % block_size_tiles == 0);

        std::vector<CoreCoord> cores;
        for (uint32_t i = 0; i < num_cores; i++) {
            cores.push_back({i, 0});
        }

        log_info(LogTest, "num_cores: {}", num_cores);
        log_info(LogTest, "num_tiles: {}", num_tiles);
        log_info(LogTest, "block_size_tiles: {}", block_size_tiles);
        log_info(LogTest, "num_blocks_in_CB: {}", num_blocks_in_CB);
        log_info(LogTest, "IO_data_in_DRAM: {}", IO_data_in_dram);
        log_info(LogTest, "num_repetitions: {}", num_repetitions);

        uint32_t single_tile_size = 2 * 1024;
        uint32_t block_size_bytes = block_size_tiles * single_tile_size;
        log_info(LogTest, "block_size_bytes: {}", block_size_bytes);
        log_info(LogTest, "CB size: {}", block_size_bytes * num_blocks_in_CB);

        // source and destination buffers
        uint32_t buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        uint32_t total_bytes_moved = buffer_size * num_repetitions;
        log_info(LogTest, "total_bytes_moved: {}", total_bytes_moved);

        // semaphores in L1, 32B aligned for NOC transfers
        uint32_t sender_semaphore_addr = l1_alloc(32); // we need 4B, use 32B for NOC alignemnt
        uint32_t receiver_semaphore_addr = l1_alloc(32);
        uint32_t l1_valid_value_addr = l1_alloc(32);

        // circular buffers in L1
        uint32_t cb_index = 8;
        uint32_t cb_size_tiles = num_blocks_in_CB * block_size_tiles;
        uint32_t cb_size_bytes = cb_size_tiles * single_tile_size;
        uint32_t cb_addr = l1_alloc(cb_size_bytes);

        for (auto core : cores) {
            auto cb = tt_metal::CreateCircularBuffer(
                program,
                cb_index,
                core,
                cb_size_tiles,
                cb_size_bytes,
                tt::DataFormat::Float16_b,
                cb_addr
            );
        }

        /// used only if IO data in DRAM
        tt_metal::Buffer src_buffer;
        tt_metal::Buffer dst_buffer;

        uint32_t src_address;
        CoreCoord src_noc_xy;
        uint32_t dst_address;
        CoreCoord dst_noc_xy;

        if (IO_data_in_dram) {
            uint32_t dram_buffer_addr = 0;
            TT_ASSERT(dram_buffer_addr + buffer_size <= 1024 * 1024 * 1024); // 1GB

            src_buffer = tt_metal::Buffer(device, buffer_size, dram_buffer_addr, buffer_size, tt_metal::BufferType::DRAM);
            dst_buffer = tt_metal::Buffer(device, buffer_size, dram_buffer_addr + buffer_size, buffer_size, tt_metal::BufferType::DRAM);

            src_address = src_buffer.address();
            src_noc_xy = src_buffer.noc_coordinates();
            dst_address = dst_buffer.address();
            dst_noc_xy = dst_buffer.noc_coordinates();
        } else {
            uint32_t l1_buffer_addr = l1_alloc(buffer_size);

            src_buffer = tt_metal::Buffer(device, buffer_size, l1_buffer_addr, buffer_size, tt_metal::BufferType::L1);
            dst_buffer = tt_metal::Buffer(device, buffer_size, l1_buffer_addr + buffer_size, buffer_size, tt_metal::BufferType::L1);

            src_address = src_buffer.address();
            src_noc_xy = src_buffer.noc_coordinates();
            dst_address = dst_buffer.address();
            dst_noc_xy = dst_buffer.noc_coordinates();
        }


        // create kernels
        vector<tt_metal::KernelID> receiver_kernels;
        vector<tt_metal::KernelID> sender_kernels;
        for (int core_id = 0; core_id < num_cores; core_id++) {

            string receiver_kernel_name;
            if (core_id == 0) {
                receiver_kernel_name = "tt_metal/kernels/dataflow/reader_first_stage.cpp";
            } else {
                receiver_kernel_name = "tt_metal/kernels/dataflow/receiver_intermediate_stage.cpp";
            }

            std::vector<uint32_t> receiver_kernel_compile_time_args = {cb_index, block_size_tiles};
            receiver_kernels.push_back(tt_metal::CreateDataMovementKernel(
                program,
                receiver_kernel_name,
                cores[core_id],
                tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = receiver_kernel_compile_time_args}));

            string sender_kernel_name;
            if (core_id == num_cores - 1) {
                sender_kernel_name = "tt_metal/kernels/dataflow/writer_last_stage.cpp";
            } else {
                sender_kernel_name = "tt_metal/kernels/dataflow/sender_intermediate_stage.cpp";
            }
            std::vector<uint32_t> sender_kernel_compile_time_args = {cb_index, block_size_tiles};
            sender_kernels.push_back(tt_metal::CreateDataMovementKernel(
                program,
                sender_kernel_name,
                cores[core_id],
                tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = sender_kernel_compile_time_args}));
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////

        pass &= tt_metal::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        // send input data to the device
        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        log_info("src_vec[0] = {}", src_vec[0]);

        tt_metal::WriteToBuffer(src_buffer, src_vec);
        // host initializes only the sender's semaphores, reciver's semaphores are initialized by the kernel
        std::vector<uint32_t> invalid = {INVALID};
        for (auto core : cores) {
            tt_metal::detail::WriteToDeviceL1(device, core, sender_semaphore_addr, invalid);
        }

        // send run-time kernel arguments
        for (int core_id = 0; core_id < num_cores; core_id++) {
            if (core_id == 0) {
                tt_metal::SetRuntimeArgs(
                    program,
                    receiver_kernels[core_id],
                    cores[core_id],
                    {src_address,
                    (uint32_t)src_noc_xy.x,
                    (uint32_t)src_noc_xy.y,
                    (uint32_t)num_tiles,
                    (uint32_t)num_repetitions});
            } else {
                tt_metal::SetRuntimeArgs(
                    program,
                    receiver_kernels[core_id],
                    cores[core_id],
                    {(uint32_t)device->worker_core_from_logical_core(cores[core_id-1]).x,
                    (uint32_t)device->worker_core_from_logical_core(cores[core_id-1]).y,
                    (uint32_t)num_tiles,
                    (uint32_t)sender_semaphore_addr,
                    (uint32_t)receiver_semaphore_addr,
                    (uint32_t)num_repetitions});
            }

            if (core_id == num_cores - 1) {
                tt_metal::SetRuntimeArgs(
                    program,
                    sender_kernels[core_id],
                    cores[core_id],
                    {dst_address,
                    (uint32_t)dst_noc_xy.x,
                    (uint32_t)dst_noc_xy.y,
                    (uint32_t)num_tiles,
                    (uint32_t)num_repetitions});
            } else {
                tt_metal::SetRuntimeArgs(
                    program,
                    sender_kernels[core_id],
                    cores[core_id],
                    {(uint32_t)device->worker_core_from_logical_core(cores[core_id+1]).x,
                    (uint32_t)device->worker_core_from_logical_core(cores[core_id+1]).y,
                    (uint32_t)num_tiles,
                    (uint32_t)sender_semaphore_addr,
                    (uint32_t)receiver_semaphore_addr,
                    (uint32_t)l1_valid_value_addr,
                    (uint32_t)num_repetitions});
            }
        }

        tt_metal::WriteRuntimeArgsToDevice(device, program);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
        log_info(LogTest, "Launching kernels...");
        pass &= tt_metal::LaunchKernels(device, program);
        log_info(LogTest, "Kernels done.");

        log_info(LogTest, "Reading results from device...");
        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= (src_vec == result_vec);

        pass &= tt_metal::CloseDevice(device);;

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
