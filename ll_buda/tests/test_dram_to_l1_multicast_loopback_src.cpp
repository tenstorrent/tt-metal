#include <algorithm>
#include <functional>
#include <random>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tensor/tensor.hpp"

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
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = new ll_buda::Program();

        tt_xy_pair core = {0, 0};
        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 1;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        
        uint32_t dram_buffer_addr = 0;
        int dram_channel_id = 0;
        uint32_t local_buffer_addr = 200 * 1024;

        uint32_t dest_buffer_addr = 500 * 1024;

        auto dram_buffer = ll_buda::CreateDramBuffer(dram_channel_id, dram_buffer_size, dram_buffer_addr);
        auto dram_noc_xy = dram_buffer->noc_coordinates(device);
        
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
            (std::uint32_t)(num_cores_x * num_cores_y)
        };
        log_info(LogTest, "Start = {}, {}", core_start_physical.x, core_start_physical.y);
        log_info(LogTest, "End = {}, {}", core_end_physical.x, core_end_physical.y);
        auto mcast_reader_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/dram_to_l1_multicast_include_src.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        pass &= ll_buda::CompileProgram(device, program, skip_hlkc);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        SHAPE shape = {1, 1, 32, 32};
        tt::Tensor<bfloat16> tensor = tt::initialize_tensor<bfloat16>(shape, tt::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto activations = pack_bfloat16_vec_into_uint32_vec(tensor.get_values());
        pass &= ll_buda::WriteToDeviceDRAM(device, dram_buffer, activations);

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);
        pass &= ll_buda::WriteRuntimeArgsToDevice(device, mcast_reader_kernel, core, mcast_reader_args);

        log_info(LogTest, "Launching kernels");
        pass &= ll_buda::LaunchKernels(device, program);
        log_info(LogTest, "Kernels done");
        
        for(int i = 0 ; i < num_cores_y; i++) {
            for(int j = 0 ; j < num_cores_x; j++) {
                tt_xy_pair dest_core = {(std::size_t) core_start.x + j, (std::size_t) core_start.y + i};
                std::vector<uint32_t> dest_core_data;
                ll_buda::ReadFromDeviceL1(device, dest_core, dest_buffer_addr, dest_core_data, dram_buffer_size);
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