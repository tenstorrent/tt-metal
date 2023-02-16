#include <algorithm>
#include <functional>
#include <random>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"

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
        uint32_t num_tiles = 2048;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        
        uint32_t dram_buffer_src_addr = 0;
        int dram_src_channel_id = 0;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto src_dram_buffer = ll_buda::CreateDramBuffer(dram_src_channel_id, dram_buffer_size, dram_buffer_src_addr);
        auto dst_dram_buffer = ll_buda::CreateDramBuffer(dram_dst_channel_id, dram_buffer_size, dram_buffer_dst_addr);

        auto dram_src_noc_xy = src_dram_buffer->noc_coordinates(device);
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates(device);

        int num_cbs = 1; // works at the moment
        assert(num_tiles % num_cbs == 0);
        int num_tiles_per_cb = num_tiles / num_cbs;

        uint32_t cb0_index = 0;
        uint32_t cb0_addr = 200 * 1024;
        uint32_t num_cb_tiles = 8;
        auto cb0 = ll_buda::CreateCircularBuffer(
            program,
            cb0_index,
            core,
            num_cb_tiles,
            num_cb_tiles * single_tile_size,
            cb0_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t cb1_index = 8;
        uint32_t cb1_addr = 250 * 1024;
        auto cb1 = ll_buda::CreateCircularBuffer(
            program,
            cb1_index,
            core,
            num_cb_tiles,
            num_cb_tiles * single_tile_size,
            cb1_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t cb2_index = 16;
        uint32_t cb2_addr = 300 * 1024;
        auto cb2 = ll_buda::CreateCircularBuffer(
            program,
            cb2_index,
            core,
            num_cb_tiles,
            num_cb_tiles * single_tile_size,
            cb2_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t cb3_index = 24;
        uint32_t cb3_addr = 350 * 1024;
        auto cb3 = ll_buda::CreateCircularBuffer(
            program,
            cb3_index,
            core,
            num_cb_tiles,
            num_cb_tiles * single_tile_size,
            cb3_addr,
            tt::DataFormat::Float16_b
        );
        
        auto reader_cb_kernel_args = ll_buda::InitializeCompileTimeDataMovementKernelArgs(core, {8, 2});
        auto writer_cb_kernel_args = ll_buda::InitializeCompileTimeDataMovementKernelArgs(core, {8, 4});

        auto reader_cb_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/reader_cb_test.cpp",
            core,
            reader_cb_kernel_args,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        auto writer_cb_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/writer_cb_test.cpp",
            core,
            writer_cb_kernel_args,
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
        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        pass &= ll_buda::WriteToDeviceDRAM(device, src_dram_buffer, src_vec);

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            reader_cb_kernel,
            core,
            {dram_buffer_src_addr,
            (uint32_t)dram_src_noc_xy.x,
            (uint32_t)dram_src_noc_xy.y,
            (uint32_t)num_tiles_per_cb});

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            writer_cb_kernel,
            core,
            {dram_buffer_dst_addr,
            (uint32_t)dram_dst_noc_xy.x,
            (uint32_t)dram_dst_noc_xy.y,
            (uint32_t)num_tiles_per_cb});

        pass &= ll_buda::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        ll_buda::ReadFromDeviceDRAM(device, dst_dram_buffer, result_vec, dst_dram_buffer->size());
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
