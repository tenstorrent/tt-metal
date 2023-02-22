#include <algorithm>
#include <functional>
#include <random>
#include <cstdlib> // for putenv()

#include "ll_buda/host_api.hpp"
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

    try {
        std::string env_var = "TT_PCI_DMA_BUF_SIZE=1048576";
        int result = putenv(const_cast<char*>(env_var.c_str()));

        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = new ll_buda::Program();

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

        std::vector<tt_xy_pair> cores;
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
            auto cb = ll_buda::CreateCircularBuffer(
                program,
                device,
                cb_index,
                core,
                cb_size_tiles,
                cb_size_bytes,
                cb_addr,
                tt::DataFormat::Float16_b
            );
        }

        /// used only if IO data in DRAM
        ll_buda::DramBuffer* src_dram_buffer;
        ll_buda::DramBuffer* dst_dram_buffer;

        // used only if IO data in L1
        ll_buda::L1Buffer* src_l1_buffer;
        ll_buda::L1Buffer* dst_l1_buffer;

        uint32_t src_address;
        tt_xy_pair src_noc_xy;
        uint32_t dst_address;
        tt_xy_pair dst_noc_xy;

        if (IO_data_in_dram) {
            uint32_t dram_buffer_addr = 0;
            TT_ASSERT(dram_buffer_addr + buffer_size <= 1024 * 1024 * 1024); // 1GB

            src_dram_buffer = ll_buda::CreateDramBuffer(device, 0, buffer_size, dram_buffer_addr);
            dst_dram_buffer = ll_buda::CreateDramBuffer(device, 7, buffer_size, dram_buffer_addr);

            src_address = src_dram_buffer->address();
            src_noc_xy = src_dram_buffer->noc_coordinates();
            dst_address = dst_dram_buffer->address();
            dst_noc_xy = dst_dram_buffer->noc_coordinates();
        } else {
            uint32_t l1_buffer_addr = l1_alloc(buffer_size); // same address on src / dst cores

            src_l1_buffer = ll_buda::CreateL1Buffer(program, device, cores[0],           buffer_size, l1_buffer_addr);
            dst_l1_buffer = ll_buda::CreateL1Buffer(program, device, cores[num_cores-1], buffer_size, l1_buffer_addr);

            src_address = src_l1_buffer->address();
            src_noc_xy = device->worker_core_from_logical_core(src_l1_buffer->logical_core());
            dst_address = dst_l1_buffer->address();
            dst_noc_xy = device->worker_core_from_logical_core(dst_l1_buffer->logical_core());
        }


        // create kernels
        vector<ll_buda::DataMovementKernel*> receiver_kernels;
        vector<ll_buda::DataMovementKernel*> sender_kernels;
        for (int core_id = 0; core_id < num_cores; core_id++) {

            string receiver_kernel_name;
            if (core_id == 0) {
                receiver_kernel_name = "kernels/dataflow/reader_first_stage.cpp";
            } else {
                receiver_kernel_name = "kernels/dataflow/receiver_intermediate_stage.cpp";
            }

            ll_buda::DataMovementKernelArgs* receiver_kernel_compile_time_args = ll_buda::InitializeCompileTimeDataMovementKernelArgs(cores[core_id], {cb_index, block_size_tiles});
            receiver_kernels.push_back(ll_buda::CreateDataMovementKernel(
                program,
                receiver_kernel_name,
                cores[core_id],
                receiver_kernel_compile_time_args,
                ll_buda::DataMovementProcessor::RISCV_1,
                ll_buda::NOC::RISCV_1_default));

            string sender_kernel_name;
            if (core_id == num_cores - 1) {
                sender_kernel_name = "kernels/dataflow/writer_last_stage.cpp";
            } else {
                sender_kernel_name = "kernels/dataflow/sender_intermediate_stage.cpp";
            }
            ll_buda::DataMovementKernelArgs* sender_kernel_compile_time_args = ll_buda::InitializeCompileTimeDataMovementKernelArgs(cores[core_id], {cb_index, block_size_tiles});
            sender_kernels.push_back(ll_buda::CreateDataMovementKernel(
                program,
                sender_kernel_name,
                cores[core_id],
                sender_kernel_compile_time_args,
                ll_buda::DataMovementProcessor::RISCV_0,
                ll_buda::NOC::RISCV_0_default));
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        constexpr bool profile_kernel = true;
        pass &= ll_buda::CompileProgram(device, program, skip_hlkc, profile_kernel);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        // send input data to the device
        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        log_info("src_vec[0] = {}", src_vec[0]);

        if (IO_data_in_dram) {
            pass &= ll_buda::WriteToDeviceDRAM(src_dram_buffer, src_vec);
        } else {
            pass &= ll_buda::WriteToDeviceL1(device, src_l1_buffer->logical_core(), src_vec, src_l1_buffer->address());
        }
        // host initializes only the sender's semaphores, reciver's semaphores are initialized by the kernel
        std::vector<uint32_t> invalid = {INVALID};
        for (auto core : cores) {
            ll_buda::WriteToDeviceL1(device, core, invalid, sender_semaphore_addr);
        }

        // send run-time kernel arguments
        for (int core_id = 0; core_id < num_cores; core_id++) {
            if (core_id == 0) {
                ll_buda::WriteRuntimeArgsToDevice(
                    device,
                    receiver_kernels[core_id],
                    cores[core_id],
                    {src_address,
                    (uint32_t)src_noc_xy.x,
                    (uint32_t)src_noc_xy.y,
                    (uint32_t)num_tiles,
                    (uint32_t)num_repetitions});
            } else {
                ll_buda::WriteRuntimeArgsToDevice(
                    device,
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
                ll_buda::WriteRuntimeArgsToDevice(
                    device,
                    sender_kernels[core_id],
                    cores[core_id],
                    {dst_address,
                    (uint32_t)dst_noc_xy.x,
                    (uint32_t)dst_noc_xy.y,
                    (uint32_t)num_tiles,
                    (uint32_t)num_repetitions});
            } else {
                ll_buda::WriteRuntimeArgsToDevice(
                    device,
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

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program, profile_kernel);
        log_info(LogTest, "Launching kernels...");
        pass &= ll_buda::LaunchKernels(device, program);
        log_info(LogTest, "Kernels done.");

        log_info(LogTest, "Reading results from device...");
        std::vector<uint32_t> result_vec;
        if (IO_data_in_dram) {
            ll_buda::ReadFromDeviceDRAM(dst_dram_buffer, result_vec);
        } else {
            ll_buda::ReadFromDeviceL1(device, dst_l1_buffer->logical_core(), dst_l1_buffer->address(), result_vec, dst_l1_buffer->size());
        }
            ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= (src_vec == result_vec);

        pass &= ll_buda::CloseDevice(device);;

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
