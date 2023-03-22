#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program *program = new tt_metal::Program();

        tt_xy_pair core = {0, 0};
        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 1;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_addr = 0;
        int dram_channel_id = 0;
        uint32_t local_buffer_addr = 200 * 1024;

        // same address as local_buffer
        // Note: src will NOT write into its dst buffer address
        // since we are not setting NOC_CMD_BRCST_SRC_INCLUDE
        uint32_t dest_buffer_addr = 200 * 1024;

        auto dram_buffer = tt_metal::CreateDramBuffer(device, dram_channel_id, dram_buffer_size, dram_buffer_addr);
        auto dram_noc_xy = dram_buffer->noc_coordinates();

        tt_xy_pair core_start = {0, 0};
        std::size_t num_cores_x = 12;
        std::size_t num_cores_y = 10;
        tt_xy_pair core_end = {core_start.x + (num_cores_x - 1), core_start.y + (num_cores_y - 1)};
        auto core_start_physical = device->worker_core_from_logical_core(core_start);
        auto core_end_physical = device->worker_core_from_logical_core(core_end);
        std::vector<uint32_t> mcast_reader_args = {
            (std::uint32_t)dram_buffer_addr,
            (std::uint32_t)dram_noc_xy.x,
            (std::uint32_t)dram_noc_xy.y,
            (std::uint32_t)dram_buffer_size,
            (std::uint32_t)local_buffer_addr,
            (std::uint32_t)dest_buffer_addr,
            (std::uint32_t)core_end_physical.x,
            (std::uint32_t)core_end_physical.y,
            (std::uint32_t)core_start_physical.x,
            (std::uint32_t)core_start_physical.y,
            (std::uint32_t)(num_cores_x * num_cores_y) - 1}; // Note: exclude src from acks, since we are not setting NOC_CMD_BRCST_SRC_INCLUDE

        log_info(LogTest, "Start = {}, {}", core_start_physical.x, core_start_physical.y);
        log_info(LogTest, "End = {}, {}", core_end_physical.x, core_end_physical.y);
        auto mcast_reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/dram_to_l1_multicast.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_1,
            tt_metal::NOC::RISCV_1_default);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        pass &= tt_metal::CompileProgram(device, program, skip_hlkc);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        SHAPE shape = {1, 1, 32, 32};
        tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto activations = pack_bfloat16_vec_into_uint32_vec(tensor.get_values());
        pass &= tt_metal::WriteToDeviceDRAM(dram_buffer, activations);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
        pass &= tt_metal::WriteRuntimeArgsToDevice(device, mcast_reader_kernel, core, mcast_reader_args);

        log_info(LogTest, "Launching kernels");
        pass &= tt_metal::LaunchKernels(device, program);
        log_info(LogTest, "Kernels done");

        for(int i = 0 ; i < num_cores_y; i++) {
            for(int j = 0 ; j < num_cores_x; j++) {
                tt_xy_pair dest_core = {(std::size_t) core_start.x + j, (std::size_t) core_start.y + i};
                std::vector<uint32_t> dest_core_data;
                tt_metal::ReadFromDeviceL1(device, dest_core, dest_buffer_addr, dest_core_data, dram_buffer_size);
                auto dest_core_data_unpacked = unpack_uint32_vec_into_bfloat16_vec(dest_core_data);
                pass &= (dest_core_data_unpacked == tensor.get_values());
                if(not (dest_core_data_unpacked == tensor.get_values())) {
                    log_info(LogTest, "Mismatch on core {}, {}", dest_core.x, dest_core.y);
                    print_vec_of_bfloat16(dest_core_data_unpacked, 1, "Result");
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
