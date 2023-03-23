#include <algorithm>
#include <functional>
#include <random>
#include <math.h>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"

#include "llrt/llrt.hpp"
#include "tt_metal/llrt/test_libs/debug_mailbox.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

inline vector<uint32_t> gold_standard_tilize(std::vector<uint32_t> src_vec, vector<uint32_t> shape) {
    vector<uint32_t> dst_vec;

    int num_rows = shape.at(0);
    int num_cols = shape.at(1) / 2;
    for (int x = 0; x < num_rows; x += 32) {
        for (int y = 0; y < num_cols; y += 16) {
            int start = x * num_cols + y;

            // Top faces
            for (int j = 0; j < 2; j++) {
                int start_ = start + 8 * j;
                for (int k = 0; k < 16; k++) {
                    for (int i = 0; i < 8; i++) {
                        int idx = start_ + num_cols * k + i;
                        dst_vec.push_back(src_vec.at(idx));
                    }
                }
            }

            // Bottom faces
            start += 16 * num_cols;
            for (int j = 0; j < 2; j++) {
                int start_ = start + 8 * j;
                for (int k = 0; k < 16; k++) {
                    for (int i = 0; i < 8; i++) {
                        int idx = start_ + num_cols * k + i;
                        dst_vec.push_back(src_vec.at(idx));
                    }
                }
            }
        }
    }

    return dst_vec;
}

bool test_write_interleaved_sticks_and_then_read_interleaved_sticks() {
    /*
        This test just writes sticks in a interleaved fashion to DRAM and then reads back to ensure
        they were written correctly
    */
    bool pass = true;

    try {
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);

        int num_sticks = 256;
        int num_elements_in_stick = 1024;
        int stick_size = num_elements_in_stick * 2;
        int num_elements_in_stick_as_packed_uint32 = num_elements_in_stick / 2;
        uint32_t dram_buffer_src_addr = 0;
        uint32_t dram_buffer_size =  num_sticks * stick_size; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(
            dram_buffer_size, false);

        pass &= tt_metal::WriteToDeviceDRAMChannelsInterleaved(
            device, src_vec, dram_buffer_src_addr, num_sticks, num_elements_in_stick_as_packed_uint32, 4);

        vector<uint32_t> dst_vec;
        tt_metal::ReadFromDeviceDRAMChannelsInterleaved(
            device, dst_vec, dram_buffer_src_addr, num_sticks, num_elements_in_stick_as_packed_uint32, 4);

        pass &= (src_vec == dst_vec);
    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    return pass;
}

bool interleaved_stick_reader_single_bank_tilized_writer_datacopy_test() {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program *program = new tt_metal::Program();

        tt_xy_pair core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;

        int num_sticks = 256;
        int num_elements_in_stick = 1024;
        int stick_size = num_elements_in_stick * 2;
        int num_elements_in_stick_as_packed_uint32 = num_elements_in_stick / 2;

        int num_tiles_c = stick_size / 64;

        // In this test, we are reading in sticks, but tilizing on the fly
        // and then writing tiles back to DRAM
        int num_output_tiles = (num_sticks * num_elements_in_stick) / 1024;

        uint32_t dram_buffer_size =  num_sticks * stick_size; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_src_addr = 0;
        int dram_src_channel_id = 0;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto dst_dram_buffer = tt_metal::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size, dram_buffer_dst_addr);
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
        // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_input_tiles = num_tiles_c;
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
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            1,
            single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        auto unary_reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
            core,
            tt_metal::InitializeCompileTimeDataMovementKernelArgs(core, {1}),
            tt_metal::DataMovementProcessor::RISCV_1,
            tt_metal::NOC::RISCV_1_default);

        auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_0,
            tt_metal::NOC::RISCV_0_default);

        vector<uint32_t> compute_kernel_args = {
            uint(num_output_tiles)
        };
        tt_metal::ComputeKernelArgs *eltwise_unary_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_kernel_args);

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/eltwise_copy.cpp",
            core,
            eltwise_unary_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        pass &= tt_metal::CompileProgram(device, program, skip_hlkc);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(
            dram_buffer_size, false);

        pass &= tt_metal::WriteToDeviceDRAMChannelsInterleaved(
            device, src_vec, dram_buffer_src_addr, num_sticks, num_elements_in_stick_as_packed_uint32, 4);
        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        tt_metal::WriteRuntimeArgsToDevice(
            device,
            unary_reader_kernel,
            core,
            {dram_buffer_src_addr,
            (uint32_t) num_sticks,
            (uint32_t) stick_size,
            (uint32_t) log2(stick_size)});

        tt_metal::WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            {dram_buffer_dst_addr,
            (uint32_t) dram_dst_noc_xy.x,
            (uint32_t) dram_dst_noc_xy.y,
            (uint32_t) num_output_tiles});

        tt_xy_pair debug_core = {1,1};
        pass &= tt_metal::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromDeviceDRAM(dst_dram_buffer, result_vec);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        pass &= (src_vec == result_vec);

        if (not pass) {
            std::cout << "GOLDEN" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(src_vec, num_output_tiles);

            std::cout << "RESULT" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(result_vec, num_output_tiles);
        }

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    return pass;
}

bool interleaved_stick_reader_interleaved_tilized_writer_datacopy_test() {
    bool pass = true;

    /*
        Placeholder to not forget to write this test
    */

    return pass;
}

bool interleaved_tilized_reader_single_bank_stick_writer_datacopy_test() {
    bool pass = true;

    /*
        Placeholder to not forget to write this test
    */

    return pass;
}


bool interleaved_tilized_reader_interleaved_stick_writer_datacopy_test() {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program *program = new tt_metal::Program();

        tt_xy_pair core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;

        int num_sticks = 256;
        int num_elements_in_stick = 1024;
        int stick_size = num_elements_in_stick * 2;
        int num_elements_in_stick_as_packed_uint32 = num_elements_in_stick / 2;

        int num_tiles_c = stick_size / 64;

        // In this test, we are reading in sticks, but tilizing on the fly
        // and then writing tiles back to DRAM
        int num_output_tiles = (num_sticks * num_elements_in_stick) / 1024;

        uint32_t dram_buffer_size =  num_sticks * stick_size; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_src_addr = 0;
        int dram_src_channel_id = 0;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto dst_dram_buffer = tt_metal::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size, dram_buffer_dst_addr);
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
        // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_input_tiles = num_tiles_c;
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
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        auto unary_reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
            core,
            tt_metal::InitializeCompileTimeDataMovementKernelArgs(core, {1}),
            tt_metal::DataMovementProcessor::RISCV_1,
            tt_metal::NOC::RISCV_1_default);

        auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_0,
            tt_metal::NOC::RISCV_0_default);

        vector<uint32_t> compute_kernel_args = {
            uint(num_output_tiles)
        };
        tt_metal::ComputeKernelArgs *eltwise_unary_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_kernel_args);

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/eltwise_copy.cpp",
            core,
            eltwise_unary_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        pass &= tt_metal::CompileProgram(device, program, skip_hlkc);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(
            dram_buffer_size, false);

        pass &= tt_metal::WriteToDeviceDRAMChannelsInterleaved(
            device, src_vec, dram_buffer_src_addr, num_sticks, num_elements_in_stick_as_packed_uint32, 4);
        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        tt_metal::WriteRuntimeArgsToDevice(
            device,
            unary_reader_kernel,
            core,
            {dram_buffer_src_addr,
            (uint32_t) num_sticks,
            (uint32_t) stick_size,
            (uint32_t) log2(stick_size)});

        tt_metal::WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            {dram_buffer_dst_addr,
            (uint32_t) num_sticks,
            (uint32_t) stick_size,
            (uint32_t) log2(stick_size)});

        tt_xy_pair debug_core = {1,1};
        read_trisc_debug_mailbox(device->cluster(), 0, debug_core, 0);
        read_trisc_debug_mailbox(device->cluster(), 0, debug_core, 1);
        pass &= tt_metal::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromDeviceDRAMChannelsInterleavedTiles(device, dram_buffer_dst_addr, result_vec, dst_dram_buffer->size());
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        pass &= (src_vec == result_vec);

        if (not pass) {
            std::cout << "GOLDEN" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(src_vec, num_output_tiles);

            std::cout << "RESULT" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(result_vec, num_output_tiles);
        }

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    pass &= test_write_interleaved_sticks_and_then_read_interleaved_sticks();
    pass &= interleaved_stick_reader_single_bank_tilized_writer_datacopy_test();
    pass &= interleaved_tilized_reader_interleaved_stick_writer_datacopy_test();

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }
}
