#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
// #include "tt_gdb/tt_gdb.hpp"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

int main(int argc, char **argv) {
    bool pass = true;

    // Once this test is uplifted to use fast dispatch, this can be removed.
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

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

        pass &= tt_metal::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();

        //CoreCoord core = {0, 0};

        uint32_t num_tiles = 10;
        uint32_t tile_size_bytes = 1024 * 2;
        uint32_t total_tiles_size_bytes = num_tiles * tile_size_bytes;
        uint32_t dram_buffer_src_addr = 0;
        uint32_t dram_buffer_size = total_tiles_size_bytes;

        uint32_t l1_buffer_addr = 400 * 1024;
        for(uint32_t i = 0; i < 10; i++) {
            for(uint32_t j = 0; j < i; j++) {
                CoreCoord core = {(size_t) j, (size_t) i};
                CoreCoord dst_soc_core = {(size_t) i+1, (size_t) j+1};
                if(j > 5) {
                    dst_soc_core.y += 1;
                }
                std::cout << "Sending from " << j+1 << "," << i+1 << " to " << i+1 << "," << j+1 << std::endl;
                auto l1_b0 = tt_metal::Buffer(device, dram_buffer_size, l1_buffer_addr, dram_buffer_size, tt_metal::BufferType::L1);

                std::vector<uint32_t> src_vec = create_constant_vector_of_bfloat16(
                    dram_buffer_size, i * 10 + j);
                auto src_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_src_addr, dram_buffer_size, tt_metal::BufferType::DRAM);
                auto dram_src_noc_xy = src_dram_buffer.noc_coordinates();
                tt_metal::WriteToBuffer(src_dram_buffer, src_vec);

                auto l1_to_l1_kernel = tt_metal::CreateDataMovementKernel(
                        program,
                        "tt_metal/kernels/dataflow/l1_to_l1.cpp",
                        core,
                        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

                tt_metal::SetRuntimeArgs(
                        program,
                        l1_to_l1_kernel,
                        core,
                        {dram_buffer_src_addr,
                        (std::uint32_t)dram_src_noc_xy.x,
                        (std::uint32_t)dram_src_noc_xy.y,
                        l1_buffer_addr,
                        l1_buffer_addr,
                        (uint32_t)dst_soc_core.x,
                        (uint32_t)dst_soc_core.y,
                        num_tiles,
                        tile_size_bytes,
                        total_tiles_size_bytes});
                dram_buffer_src_addr += dram_buffer_size;
            }
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        pass &= tt_metal::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::WriteRuntimeArgsToDevice(device, program);
        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        pass &= tt_metal::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        for(uint32_t i = 0; i < 10; i++) {
            for(uint32_t j = i+1; j < 10; j++) {
                CoreCoord core = {(size_t) j, (size_t) i};
                tt_metal::detail::ReadFromDeviceL1(device, core, l1_buffer_addr, total_tiles_size_bytes, result_vec);
                std::vector<uint32_t> src_vec = create_constant_vector_of_bfloat16(
                    dram_buffer_size, j * 10 + i);
                if(src_vec != result_vec) {
                    std::cout << "      Failed on core " << j+1 << "," << i+1 << std::endl;
                }
                else {
                    std::cout << "Passed on core " << j+1 << "," << i+1 << std::endl;
                }
                pass &= (src_vec == result_vec);
            }
        }

        //std::vector<uint32_t> result_vec;
        //tt_metal::ReadFromDeviceDRAM(dst_dram_buffer, result_vec);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        //pass &= (src_vec == result_vec);

        pass &= tt_metal::CloseDevice(device);

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
