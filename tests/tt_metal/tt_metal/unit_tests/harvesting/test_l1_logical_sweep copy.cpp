#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: Test for testing dram multi-casting to all the L1 cores
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

int main(int argc, char **argv) {
    bool pass = true;

    try {
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

        pass &= tt_metal::InitializeDevice(device);;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();

        CoreCoord core = {0, 0};
        size_t numel = 32*32;
        size_t num_bytes_per_elem = 2;
        uint32_t single_tile_size = num_bytes_per_elem * numel;
        uint32_t num_tiles = 1;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_addr = 0;
        int dram_channel_id = 0;
        uint32_t local_buffer_addr = 200 * 1024;

        // same address as local_buffer
        // Note: src will NOT write into its dst buffer address
        // since we are not setting NOC_CMD_BRCST_SRC_INCLUDE
        uint32_t dest_buffer_addr = 200 * 1024;

        auto dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_addr, dram_channel_id, dram_buffer_size, tt_metal::BufferType::DRAM);

        auto dram_noc_xy = dram_buffer.noc_coordinates();

        CoreCoord core_start = {0, 0};
        CoreCoord grid_size = device->logical_grid_size();
        CoreCoord core_end = {core_start.x + (grid_size.x - 1), core_start.y + (grid_size.y - 1)};
        auto core_start_physical = device->worker_core_from_logical_core(core_start);
        auto core_end_physical = device->worker_core_from_logical_core(core_end);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        pass &= tt_metal::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        auto inputs = tt::test_utils::generate_uniform_int_random_vector<uint32_t>(0, UINT32_MAX, single_tile_size);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        log_debug(LogTest, "Launching kernels");
        pass &= tt_metal::LaunchKernels(device, program);
        log_debug(LogTest, "Kernels done");

        for(int i = 0 ; i < grid_size.y; i++) {
            for(int j = 0 ; j < grid_size.x; j++) {
                CoreCoord dest_core = {(std::size_t) core_start.x + j, (std::size_t) core_start.y + i};
                tt_metal::WriteToDeviceL1(device, dest_core, dest_buffer_addr, inputs);
            }
        }

        for(int i = 0 ; i < grid_size.y; i++) {
            for(int j = 0 ; j < grid_size.x; j++) {
                CoreCoord dest_core = {(std::size_t) core_start.x + j, (std::size_t) core_start.y + i};
                std::vector<uint32_t> dest_core_data;
                tt_metal::ReadFromDeviceL1(device, dest_core, dest_buffer_addr, dram_buffer_size, dest_core_data);
                pass &= (dest_core_data == inputs);
                if(not (dest_core_data == inputs)) {
                    tt::test_utils::print_vec(dest_core_data);
                    log_fatal(LogTest, "Mismatch on core {}, {}", dest_core.x, dest_core.y);
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
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
