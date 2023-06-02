#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/device/tt_memory.h"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
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

        pass &= tt_metal::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();

        CoreCoord core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 2048;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_src_addr = 0;
        int dram_src_channel_id = 0;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto src_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_src_addr, dram_src_channel_id, dram_buffer_size, tt_metal::BufferType::DRAM);
        auto dst_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_dst_addr, dram_dst_channel_id, dram_buffer_size, tt_metal::BufferType::DRAM);

        auto dram_src_noc_xy = src_dram_buffer.noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
        // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_input_tiles = 8;
        auto cb_src0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            src0_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 300 * 1024;
        uint32_t num_output_tiles = 1;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        auto unary_reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_unary_push_4.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_1,
            tt_metal::NOC::RISCV_1_default);

        auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_0,
            tt_metal::NOC::RISCV_0_default);

        vector<uint32_t> compute_kernel_args = {
            uint(num_tiles) // per_core_tile_cnt
        };

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/eltwise_copy_3m.cpp",
            core,
            compute_kernel_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        // Check that binary memory objects in the kernel match the ones obtained from the persistent cache
        auto kernel_group = program.kernels_on_core(core);

        int num_compiles = 3;
        // kernel->binaries() returns 32B aligned binaries
        std::vector<ll_api::memory> compute_binaries;
        std::vector<ll_api::memory> brisc_binaries;
        std::vector<ll_api::memory> ncrisc_binaries;
        for (int i = 0; i < num_compiles; i++) {
            pass &= tt_metal::CompileProgram(device, program);
            if (i == 0) {
                compute_binaries = kernel_group.compute->binaries();
                TT_ASSERT(compute_binaries.size() == 3, "Expected 3 Compute binaries!");
                brisc_binaries = kernel_group.riscv_0->binaries();
                TT_ASSERT(brisc_binaries.size() == 1, "Expected 1 BRISC binary!");
                ncrisc_binaries = kernel_group.riscv_1->binaries();
                TT_ASSERT(ncrisc_binaries.size() == 1, "Expected 1 NCRISC binary!");
            } else {
                TT_ASSERT(kernel_group.compute->binaries() == compute_binaries);
                TT_ASSERT(kernel_group.riscv_0->binaries() == brisc_binaries);
                TT_ASSERT(kernel_group.riscv_1->binaries() == ncrisc_binaries);
            }
            std::string brisc_hex_path = kernel_group.riscv_0->binary_path() + "/brisc/brisc.hex";
            ll_api::memory brisc_binary = llrt::get_risc_binary(brisc_hex_path);
            TT_ASSERT(brisc_binary == brisc_binaries.at(0), "Expected saved BRISC binary to be the same as binary in persistent cache");
            std::string ncrisc_hex_path = kernel_group.riscv_1->binary_path() + "/ncrisc/ncrisc.hex";
            ll_api::memory ncrisc_binary = llrt::get_risc_binary(ncrisc_hex_path);
            TT_ASSERT(ncrisc_binary == ncrisc_binaries.at(0), "Expected saved NCRISC binary to be the same as binary in persistent cache");
            for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
                std::string trisc_id_str = std::to_string(trisc_id);
                std::string trisc_hex_path = kernel_group.compute->binary_path() + "/tensix_thread" + trisc_id_str + "/tensix_thread" + trisc_id_str + ".hex";
                ll_api::memory trisc_binary = llrt::get_risc_binary(trisc_hex_path);
                TT_ASSERT(trisc_binary == compute_binaries.at(trisc_id), "Expected saved TRISC binary for " + trisc_id_str + " to be the same as binary in persistent cache");
            }
        }

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
