// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include <filesystem>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/jit_build/build_env_manager.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;

bool test_compile_args(std::vector<uint32_t> compile_args_vec, tt_metal::IDevice* device) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    tt_metal::KernelHandle unary_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/test_compile_args.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = compile_args_vec});

    tt_metal::KernelHandle unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_args = {
        0  // dummy
    };

    auto eltwise_unary_kernel = tt_metal::CreateKernel(
        program, "tt_metal/kernels/compute/blank.cpp", core, tt_metal::ComputeConfig{.compile_args = compute_args});

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::detail::CompileProgram(device, program);

    return true;
}

int main(int argc, char** argv) {
    bool pass = true;

    try {
        int device_id = 0;

        tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);
        // Remove old compiled kernels
        static const std::string kernel_name = "test_compile_args";
        auto binary_path_str =
            kernel
                ->binaries(
                    tt::tt_metal::BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_env)
                .get_out_kernel_root_path() +
            kernel_name;
        std::filesystem::remove_all(binary_path_str);

        pass &= test_compile_args({0, 68, 0, 124}, device);
        pass &= test_compile_args({1, 5, 0, 124}, device);

        TT_FATAL(std::filesystem::exists(binary_path_str), "Expected kernel to be compiled!");

        std::filesystem::path binary_path{binary_path_str};
        auto num_built_kernels =
            std::distance(std::filesystem::directory_iterator(binary_path), std::filesystem::directory_iterator{});
        TT_FATAL(num_built_kernels == 2, "Expected compute kernel test_compile_args to be compiled twice!");

        if (tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled()) {
            // Test that the kernel_args.csv file was generated for both kernels
            log_info(LogTest, "Test kernel args logging");
            auto kernel_args_path = binary_path.parent_path() / "kernel_args.csv";
            TT_FATAL(
                std::filesystem::exists(kernel_args_path),
                "Expected kernel_args.csv to be generated in path {}",
                kernel_args_path);

            std::ifstream compile_args_file(kernel_args_path);
            std::string line;
            int num_found = 0;
            while (std::getline(compile_args_file, line)) {
                if (line.find("test_compile_args") != std::string::npos) {
                    if (line.find("ARG_0=0,ARG_1=68,ARG_2=0,ARG_3=124") != std::string::npos) {
                        num_found++;
                    } else if (line.find("ARG_0=1,ARG_1=5,ARG_2=0,ARG_3=124") != std::string::npos) {
                        num_found++;
                    } else {
                        TT_THROW("Expected kernel_args.csv to contain the compile args for test_compile_args");
                    }
                }
            }
            TT_FATAL(
                num_found == 2,
                "Expected kernel_args.csv to contain the compile args for both kernels. Instead, found {} entries",
                num_found);
        }

        CloseDevice(device);

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
