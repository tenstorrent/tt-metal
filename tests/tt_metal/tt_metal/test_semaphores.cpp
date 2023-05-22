#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

void initialize_program(tt_metal::Device *device, tt_metal::Program &program, const tt_metal::CoreRange &core_range) {
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;

    // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
    // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = 8;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src0_cb_index,
        core_range,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        src0_cb_addr,
        tt::DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t output_cb_addr = 300 * 1024;
    uint32_t num_output_tiles = 1;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        device,
        ouput_cb_index,
        core_range,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        tt::DataFormat::Float16_b
    );

    auto unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_push_4.cpp",
        core_range,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core_range,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles) // per_core_tile_cnt
    };
    tt_metal::KernelArgs eltwise_unary_args = tt_metal::KernelArgs(core_range, compute_kernel_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_copy_3m.cpp",
        core_range,
        eltwise_unary_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    tt_metal::CompileProgram(device, program);
}

bool check_addresses_are_same(const std::vector<tt_metal::Semaphore *> &semaphores) {
    bool pass = true;
    auto first_addr = semaphores.at(0)->address();
    for (auto s : semaphores) {
        pass &= s->address() == first_addr;
    }
    TT_ASSERT(pass && "Expected all semaphore addresses to be the same!");
    return pass;
}

bool test_initialize_semaphores(tt_metal::Device *device, tt_metal::Program &program, tt_metal::CoreRange &core_range) {
    bool pass = true;

    auto size_per_semaphore = SEMAPHORE_SIZE / NUM_SEMAPHORES;

    std::vector<uint32_t> golden;
    for (uint32_t i = 0; i < NUM_SEMAPHORES; i++) {
        uint32_t initial_value = i;
        auto semaphores = tt_metal::CreateSemaphores(program, device, core_range, initial_value);
        golden.push_back(initial_value);
        pass &= check_addresses_are_same(semaphores);
        pass &= semaphores.at(0)->address() == SEMAPHORE_BASE + (size_per_semaphore * i);
    }

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

    for (auto x = core_range.first.x; x <= core_range.second.x; x++) {
        for (auto y = core_range.first.y; y <= core_range.second.y; y++) {
            auto logical_core = tt_xy_pair(x, y);
            std::vector<uint32_t> res;
            tt_metal::ReadFromDeviceL1(device, logical_core, SEMAPHORE_BASE, SEMAPHORE_SIZE, res);
            pass &= res == golden;
        }
    }

    return pass;
}

bool test_more_than_max_num_semaphores_(tt_metal::Device *device, tt_metal::Program &program, tt_metal::CoreRange &core_range) {
    bool pass = true;

    try {
        uint32_t val = 5;
        auto fourth_semaphores = tt_metal::CreateSemaphores(program, device, core_range, val);
    } catch (const std::exception &e) {
        pass = true;
    }

    return pass;
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

        pass &= tt_metal::InitializeDevice(device);

        tt_metal::Program program = tt_metal::Program();
        tt_metal::CoreRange core_range = {{0, 0}, {1, 1}};
        initialize_program(device, program, core_range);

        pass &= test_initialize_semaphores(device, program, core_range);

        pass &= test_more_than_max_num_semaphores_(device, program, core_range);

        ////////////////////////////////////////////////////////////////////////////
        //                              Teardown
        ////////////////////////////////////////////////////////////////////////////
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
