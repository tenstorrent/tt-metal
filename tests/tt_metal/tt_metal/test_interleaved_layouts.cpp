#include <algorithm>
#include <functional>
#include <random>
#include <math.h>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"

#include "llrt/llrt.hpp"
#include "tt_metal/llrt/test_libs/debug_mailbox.hpp"

#include "llrt/tt_debug_print_server.hpp"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

bool test_write_interleaved_sticks_and_then_read_interleaved_sticks(const tt::ARCH& arch) {
    /*
        This test just writes sticks in a interleaved fashion to DRAM and then reads back to ensure
        they were written correctly
    */
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(arch, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);

        int num_sticks = 256;
        int num_elements_in_stick = 1024;
        int stick_size = num_elements_in_stick * 2;
        int num_elements_in_stick_as_packed_uint32 = num_elements_in_stick / 2;
        uint32_t dram_buffer_src_addr = 0;
        uint32_t dram_buffer_size =  num_sticks * stick_size; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(
            dram_buffer_size, false);

        auto sticks_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_src_addr, stick_size, tt_metal::BufferType::DRAM);

        tt_metal::WriteToBuffer(sticks_buffer, src_vec);

        vector<uint32_t> dst_vec;
        tt_metal::ReadFromBuffer(sticks_buffer, dst_vec);

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

bool interleaved_stick_reader_single_bank_tilized_writer_datacopy_test(const tt::ARCH& arch) {
    bool pass = true;

    try {
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
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)

        auto src_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_src_addr, stick_size, tt_metal::BufferType::DRAM);

        auto dst_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_dst_addr, dram_buffer_size, tt_metal::BufferType::DRAM);
        auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
        // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_input_tiles = num_tiles_c;
        auto cb_src0 = tt_metal::CreateCircularBuffer(
            program,
            src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            src0_cb_addr
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 300 * 1024;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            ouput_cb_index,
            core,
            1,
            single_tile_size,
            tt::DataFormat::Float16_b,
            output_cb_addr
        );

        auto unary_reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = {1}});

        auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {
            uint(num_output_tiles)
        };

        auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/eltwise_copy.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        pass &= tt_metal::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(
            dram_buffer_size, false);

        tt_metal::WriteToBuffer(src_dram_buffer, src_vec);
        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel,
            core,
            {dram_buffer_src_addr,
            (uint32_t) num_sticks,
            (uint32_t) stick_size,
            (uint32_t) log2(stick_size)});

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dram_buffer_dst_addr,
            (uint32_t) dram_dst_noc_xy.x,
            (uint32_t) dram_dst_noc_xy.y,
            (uint32_t) num_output_tiles});

        CoreCoord debug_core = {1,1};
        tt_metal::WriteRuntimeArgsToDevice(device, program);
        pass &= tt_metal::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_dram_buffer, result_vec);
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

        DeallocateBuffer(dst_dram_buffer);
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


bool interleaved_tilized_reader_interleaved_stick_writer_datacopy_test(const tt::ARCH& arch) {
    bool pass = true;

    try {
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
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)

        auto src_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_src_addr, stick_size, tt_metal::BufferType::DRAM);

        auto dst_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_dst_addr, stick_size, tt_metal::BufferType::DRAM);
        auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
        // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_input_tiles = num_tiles_c;
        auto cb_src0 = tt_metal::CreateCircularBuffer(
            program,
            src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            src0_cb_addr
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 300 * 1024;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            ouput_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            output_cb_addr
        );

        auto unary_reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = {1}});

        auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {
            uint(num_output_tiles)
        };

        auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/eltwise_copy.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        pass &= tt_metal::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(
            dram_buffer_size, false);

        tt_metal::WriteToBuffer(src_dram_buffer, src_vec);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel,
            core,
            {dram_buffer_src_addr,
            (uint32_t) num_sticks,
            (uint32_t) stick_size,
            (uint32_t) log2(stick_size)});

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dram_buffer_dst_addr,
            (uint32_t) num_sticks,
            (uint32_t) stick_size,
            (uint32_t) log2(stick_size)});

        CoreCoord debug_core = {1,1};
        read_trisc_debug_mailbox(device->cluster(), 0, debug_core, 0);
        read_trisc_debug_mailbox(device->cluster(), 0, debug_core, 1);
        tt_metal::WriteRuntimeArgsToDevice(device, program);
        pass &= tt_metal::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_dram_buffer, result_vec);
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


template <bool src_is_in_l1, bool dst_is_in_l1>
bool test_interleaved_l1_datacopy(const tt::ARCH& arch) {

    uint num_pages = 256;
    uint num_bytes_per_page = 2048;
    uint num_entries_per_page = 512;
    uint num_bytes_per_entry = 4;
    uint buffer_size = num_pages * num_bytes_per_page;

    uint num_l1_banks = 128;
    uint num_dram_banks = 8;

    bool pass = true;

    int pci_express_slot = 0;
    tt_metal::Device *device =
        tt_metal::CreateDevice(arch, pci_express_slot);

    pass &= tt_metal::InitializeDevice(device);

    tt_metal::Program program = tt_metal::Program();
    CoreCoord core = {0, 0};

    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        0,
        core,
        2,
        2 * num_bytes_per_page,
        tt::DataFormat::Float16_b
    );

    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        16,
        core,
        2,
        2 * num_bytes_per_page,
        tt::DataFormat::Float16_b
    );

    auto unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = {not src_is_in_l1}});

    auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = {not dst_is_in_l1}});


    vector<uint32_t> compute_kernel_args = { num_pages };
    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_copy.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
    );

    std::vector<uint32_t> host_buffer = create_random_vector_of_bfloat16(
        buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    tt_metal::Buffer src;
    tt_metal::Buffer dst;
    if constexpr (src_is_in_l1) {
        TT_ASSERT((buffer_size % num_l1_banks) == 0);

        src = tt_metal::Buffer(device, buffer_size, num_bytes_per_page, tt_metal::BufferType::L1);
        tt_metal::WriteToBuffer(src, host_buffer);

        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel,
            core,
            {src.address(), 0, 0, num_pages});

    } else {
        TT_ASSERT((buffer_size % num_dram_banks) == 0);

        src = tt_metal::Buffer(device, buffer_size, num_bytes_per_page, tt_metal::BufferType::DRAM);
        tt_metal::WriteToBuffer(src, host_buffer);

        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel,
            core,
            {src.address(), 0, 0, num_pages});
    }

    std::vector<uint32_t> readback_buffer;
    if constexpr (dst_is_in_l1) {
        dst = tt_metal::Buffer(device, buffer_size, num_bytes_per_page, tt_metal::BufferType::L1);

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dst.address(), 0, 0, num_pages});

        pass &= tt_metal::CompileProgram(device, program);
        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
        tt_metal::WriteRuntimeArgsToDevice(device, program);

        pass &= tt_metal::LaunchKernels(device, program);

        tt_metal::ReadFromBuffer(dst, readback_buffer);

    } else {
         dst = tt_metal::Buffer(device, buffer_size, num_bytes_per_page, tt_metal::BufferType::DRAM);

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dst.address(), 0, 0, num_pages});

        pass &= tt_metal::CompileProgram(device, program);
        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
        tt_metal::WriteRuntimeArgsToDevice(device, program);

        pass &= tt_metal::LaunchKernels(device, program);

        tt_metal::ReadFromBuffer(dst, readback_buffer);
    }

    pass = (host_buffer == readback_buffer);

    TT_ASSERT(pass);

    return pass;
}

int main(int argc, char **argv) {

    // Once this test is uplifted to use fast dispatch, this can be removed.
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    bool pass = true;
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

    // DRAM row/tile interleaved layout tests
    pass &= test_write_interleaved_sticks_and_then_read_interleaved_sticks(arch);
    pass &= interleaved_stick_reader_single_bank_tilized_writer_datacopy_test(arch);
    pass &= interleaved_tilized_reader_interleaved_stick_writer_datacopy_test(arch);

    // L1 tile-interleaved tests
    pass &= test_interleaved_l1_datacopy<true, true>(arch);
    pass &= test_interleaved_l1_datacopy<false, true>(arch);
    pass &= test_interleaved_l1_datacopy<true, false>(arch);
    pass &= test_interleaved_l1_datacopy<false, false>(arch);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }
}
