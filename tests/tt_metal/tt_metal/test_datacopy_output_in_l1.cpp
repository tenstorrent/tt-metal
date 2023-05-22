#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/tools/tt_gdb/tt_gdb.hpp"


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
        tt_metal::Device *device =
            tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device, tt_metal::MemoryAllocator::L1_BANKING);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();

        tt_xy_pair core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 32;
        uint32_t buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        int dram_src_channel_id = 0;
        auto l1_bank_ids = device->bank_ids_from_logical_core(core);
        TT_ASSERT(not l1_bank_ids.empty());
        auto l1_bank_id = l1_bank_ids.at(0);

        auto src_dram_buffer = tt_metal::Buffer(device, buffer_size, dram_src_channel_id, buffer_size, tt_metal::BufferType::DRAM);

        auto dst_l1_buffer = tt_metal::Buffer(device, buffer_size, l1_bank_id, buffer_size, tt_metal::BufferType::L1);

        auto dram_src_noc_xy = src_dram_buffer.noc_coordinates();
        auto l1_dst_noc_xy = dst_l1_buffer.noc_coordinates();

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
        // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = 8;
        auto cb_src0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t num_output_tiles = 1;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
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
        tt_metal::KernelArgs eltwise_unary_args = tt_metal::KernelArgs(core, compute_kernel_args);

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/eltwise_copy_3m.cpp",
            core,
            eltwise_unary_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        pass &= tt_metal::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        tt_metal::WriteToBuffer(src_dram_buffer, src_vec);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        tt_metal::WriteRuntimeArgsToDevice(
            device,
            unary_reader_kernel,
            core,
            {src_dram_buffer.address(),
            (std::uint32_t)dram_src_noc_xy.x,
            (std::uint32_t)dram_src_noc_xy.y,
            num_tiles});

        tt_metal::WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            {dst_l1_buffer.address(),
            (std::uint32_t)l1_dst_noc_xy.x,
            (std::uint32_t)l1_dst_noc_xy.y,
            num_tiles});

        // tt::tt_metal::tt_gdb(device, 0, program->logical_cores(), program->cores_to_ops());
        pass &= tt_metal::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_l1_buffer, result_vec);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        pass &= (src_vec == result_vec);

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
