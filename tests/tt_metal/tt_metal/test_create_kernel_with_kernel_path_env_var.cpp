#include <gtest/gtest.h>

#include <filesystem>

#include "assert.hpp"
#include "core_coord.h"
#include "detail/tt_metal.hpp"
#include "host_api.hpp"
#include "impl/kernels/data_types.hpp"
#include "impl/program/program.hpp"
#include "llrt/rtoptions.hpp"
#include "logger.hpp"

using namespace tt;
using namespace tt::tt_metal;

void validate_env_vars_are_set() {
    TT_FATAL(llrt::OptionsG.is_root_dir_specified(), "TT_METAL_HOME must be set for this test");
    TT_FATAL(llrt::OptionsG.is_kernel_dir_specified(), "TT_METAL_KERNEL_PATH must be set for this test");
}

bool does_path_exist(const string &path) {
    const std::filesystem::path &file_path(path);
    return std::filesystem::exists(file_path);
}

bool is_path_a_directory(const string &path) {
    TT_FATAL(does_path_exist(path));
    const std::filesystem::path &file_path(path);
    return std::filesystem::is_directory(file_path);
}

bool is_dir_empty(const string &path) {
    TT_FATAL(does_path_exist(path));
    TT_FATAL(is_path_a_directory(path));
    const std::filesystem::path &file_path(path);
    return std::filesystem::is_empty(file_path);
}

void validate_kernel_dir_is_valid() {
    const string &kernel_dir = llrt::OptionsG.get_kernel_dir();
    if (!does_path_exist(kernel_dir) || !is_path_a_directory(kernel_dir) || !is_dir_empty(kernel_dir)) {
        TT_THROW("TT_METAL_KERNEL_PATH must be an existing, empty directory for this test");
    }
}

void validate_preconditions() {
    validate_env_vars_are_set();
    validate_kernel_dir_is_valid();
}

void setup_kernel_dir(const string &orig_kernel_file, const string &new_kernel_file) {
    const string &kernel_dir = llrt::OptionsG.get_kernel_dir();
    const std::filesystem::path &kernel_file_path_under_kernel_dir(kernel_dir + orig_kernel_file);
    std::filesystem::create_directories(kernel_file_path_under_kernel_dir);

    const string &metal_root = llrt::OptionsG.get_root_dir();
    const std::filesystem::path &kernel_file_path_under_metal_root(metal_root + new_kernel_file);
    std::filesystem::copy(kernel_file_path_under_metal_root, kernel_file_path_under_kernel_dir);
}

void cleanup_kernel_dir() {
    const string &kernel_dir = llrt::OptionsG.get_kernel_dir();
    std::filesystem::remove_all(kernel_dir);
}

void create_kernel(tt_metal::Program &program, const string &kernel_file) {
    CoreCoord core(0, 0);
    tt_metal::CreateKernel(
        program,
        kernel_file,
        core,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default})
}

bool test_compile_program_with_kernel_under_metal_root_dir(Device *device) {
    const string &kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";
    tt_metal::Program program = CreateProgram();
    create_kernel(program, kernel_file);
    detail::CompileProgram(device, program);
    return true;
}

bool test_compile_program_with_kernel_under_kernel_root_dir(Device *device) {
    const string &orig_kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";
    const string &new_kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/new_kernel.cpp";
    setup_kernel_dir(orig_kernel_file, new_kernel_file);
    tt_metal::Program program = CreateProgram();
    create_kernel(program, new_kernel_file);
    detail::CompileProgram(device, program);
    cleanup_kernel_dir();
    return true;
}

bool test_compile_program_with_kernel_under_metal_root_dir_and_kernel_root_dir(Device *device) {
    const string &kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";
    setup_kernel_dir(kernel_file, kernel_file);
    tt_metal::Program program = CreateProgram();
    create_kernel(program, kernel_file);
    detail::CompileProgram(device, program);
    cleanup_kernel_dir();
    return true;
}

bool test_compile_program_with_non_existent_kernel(Device *device) {
    bool pass = false;
    const string &kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/non_existent_kernel.cpp";
    tt_metal::Program program = CreateProgram();
    create_kernel(program, kernel_file);
    try {
        detail::CompileProgram(device, program);
    } catch (const std::exception &e) {
        pass = true;
    }
    return pass;
}

int main(int argc, char **argv) {
    validate_preconditions();

    bool pass = true;

    try {
        const int device_id = 0;
        Device *device = CreateDevice(device_id);

        pass &= test_compile_program_with_kernel_under_metal_root_dir(device);

        pass &= test_compile_program_with_kernel_under_kernel_root_dir(device);

        pass &= test_compile_program_with_kernel_under_metal_root_dir_and_kernel_root_dir(device);

        pass &= test_compile_program_with_non_existent_kernel(device);

        pass &= CloseDevice(device);
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
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass);
    return 0;
}
