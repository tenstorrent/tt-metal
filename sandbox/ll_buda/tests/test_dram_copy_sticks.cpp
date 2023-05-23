#include <algorithm>
#include <functional>
#include <random>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"
// #include "tt_gdb/tt_gdb.hpp"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

namespace unary_datacopy {
//#include "hlks/eltwise_copy.cpp"
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::int32_t per_core_tile_cnt;
};
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = new ll_buda::Program();

        CoreCoord core = {0, 0};

        int num_sticks = 4;
        int num_elements_in_stick = 512;
        int stick_size = num_elements_in_stick * 2;
        int num_elements_in_stick_as_packed_uint32 = num_elements_in_stick / 2;
        uint32_t dram_buffer_size =  num_sticks * stick_size; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_src_addr = 0;
        int dram_src_channel_id = 0;

        auto src_dram_buffer = ll_buda::CreateDramBuffer(device, dram_src_channel_id, dram_buffer_size, dram_buffer_src_addr);

        auto dram_src_noc_xy = src_dram_buffer->noc_coordinates();
        uint32_t l1_buffer_addr = 400 * 1024;
        auto l1_b0 = ll_buda::CreateL1Buffer(program, device, core, dram_buffer_size, l1_buffer_addr);
        auto unary_reader_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/dram_copy_sticks.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool profile_kernel = true;
        pass &= ll_buda::CompileProgram(device, program, profile_kernel);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        pass &= ll_buda::WriteToDeviceDRAM(src_dram_buffer, src_vec);

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program, profile_kernel);
        ll_buda::WriteRuntimeArgsToDevice(
            device,
            unary_reader_kernel,
            core,
            {l1_buffer_addr,
            dram_buffer_src_addr,
            (std::uint32_t)dram_src_noc_xy.x,
            (std::uint32_t)dram_src_noc_xy.y,
            (std::uint32_t) num_sticks,
            (std::uint32_t) stick_size});

        pass &= ll_buda::LaunchKernels(device, program);

        //std::vector<uint32_t> result_vec;
        //ll_buda::ReadFromDeviceDRAM(dst_dram_buffer, result_vec);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        //pass &= (src_vec == result_vec);

        pass &= ll_buda::CloseDevice(device);

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
