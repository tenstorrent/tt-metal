#include <algorithm>
#include <functional>
#include <random>

#include "ll_buda/host_api.hpp"
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
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = new ll_buda::Program();

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

        auto input_dram_buffer = ll_buda::CreateDramBuffer(device, dram_channel, dram_buffer_size_bytes, input_dram_buffer_addr);

        auto l1_b0 = ll_buda::CreateL1Buffer(program, core, total_l1_buffer_size_bytes, l1_buffer_addr);

        auto output_dram_buffer = ll_buda::CreateDramBuffer(device, dram_channel, dram_buffer_size_bytes, output_dram_buffer_addr);

        auto input_dram_noc_xy = input_dram_buffer->noc_coordinates();
        auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();

        auto dram_copy_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/dram_copy_db.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        pass &= ll_buda::CompileProgram(device, program, skip_hlkc);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
            dram_buffer_size_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count());
        pass &= ll_buda::WriteToDeviceDRAM(device, input_dram_buffer, input_vec);

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);

        ll_buda::WriteRuntimeArgsToDevice(
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

        pass &= ll_buda::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        ll_buda::ReadFromDeviceDRAM(device, output_dram_buffer, result_vec, output_dram_buffer->size());

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass = (input_vec == result_vec);

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
