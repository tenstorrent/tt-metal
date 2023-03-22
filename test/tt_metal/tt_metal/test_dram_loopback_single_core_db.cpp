#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
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
        uint32_t num_tiles = 256;
        uint32_t dram_buffer_size_bytes = single_tile_size * num_tiles;

        uint32_t input_dram_buffer_addr = 0;
        uint32_t output_dram_buffer_addr = 512 * 1024;
        int dram_channel = 0;

        // L1 buffer is double buffered
        // We read and write total_l1_buffer_size_tiles / 2 tiles from and to DRAM
        uint32_t l1_buffer_addr = 400 * 1024;
        uint32_t total_l1_buffer_size_tiles = num_tiles / 2;
        TT_ASSERT(total_l1_buffer_size_tiles % 2 == 0);
        uint32_t total_l1_buffer_size_bytes = total_l1_buffer_size_tiles * single_tile_size;

        auto input_dram_buffer = tt_metal::CreateDramBuffer(device, dram_channel, dram_buffer_size_bytes, input_dram_buffer_addr);

        auto l1_b0 = tt_metal::CreateL1Buffer(program, device, core, total_l1_buffer_size_bytes, l1_buffer_addr);

        auto output_dram_buffer = tt_metal::CreateDramBuffer(device, dram_channel, dram_buffer_size_bytes, output_dram_buffer_addr);

        auto input_dram_noc_xy = input_dram_buffer->noc_coordinates();
        auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();

        auto dram_copy_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/dram_copy_db.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_0,
            tt_metal::NOC::RISCV_0_default);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        pass &= tt_metal::CompileProgram(device, program, skip_hlkc);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
            dram_buffer_size_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count());
        pass &= tt_metal::WriteToDeviceDRAM(input_dram_buffer, input_vec);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        tt_metal::WriteRuntimeArgsToDevice(
            device,
            dram_copy_kernel,
            core,
            {input_dram_buffer_addr,
            (std::uint32_t)input_dram_noc_xy.x,
            (std::uint32_t)input_dram_noc_xy.y,
            output_dram_buffer_addr,
            (std::uint32_t)output_dram_noc_xy.x,
            (std::uint32_t)output_dram_noc_xy.y,
            dram_buffer_size_bytes,
            num_tiles,
            l1_buffer_addr,
            total_l1_buffer_size_tiles,
            total_l1_buffer_size_bytes});

        pass &= tt_metal::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromDeviceDRAM(output_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass = (input_vec == result_vec);

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
