// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <exception>
#include <filesystem>

#include "assert.hpp"
#include "core_coord.h"
#include "detail/tt_metal.hpp"
#include "host_api.hpp"
#include "impl/kernels/data_types.hpp"
#include "impl/program/program.hpp"
#include "impl/program/program_pool.hpp"
#include "llrt/rtoptions.hpp"
#include "tt_cluster_descriptor_types.h"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::llrt;

class CompileProgramWithKernelPathEnvVarFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        this->validate_preconditions();

        const chip_id_t device_id = 0;
        this->device_ = CreateDevice(device_id);
        this->program_handle_ = CreateProgram();
        this->program_ = ProgramPool::instance().get_program(this->program_handle_);
    }

    void TearDown() override {
        CloseDevice(this->device_);
        CloseProgram(this->program_handle_);
    }

    void create_kernel(const string &kernel_file) {
        CoreCoord core(0, 0);
        tt_metal::CreateKernel(
            this->program_handle_,
            kernel_file,
            core,
            tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    }

    void setup_kernel_dir(const string &orig_kernel_file, const string &new_kernel_file) {
        const string &kernel_dir = OptionsG.get_kernel_dir();
        const std::filesystem::path &kernel_file_path_under_kernel_dir(kernel_dir + new_kernel_file);
        const std::filesystem::path &dirs_under_kernel_dir = kernel_file_path_under_kernel_dir.parent_path();
        std::filesystem::create_directories(dirs_under_kernel_dir);

        const string &metal_root = OptionsG.get_root_dir();
        const std::filesystem::path &kernel_file_path_under_metal_root(metal_root + orig_kernel_file);
        std::filesystem::copy(kernel_file_path_under_metal_root, kernel_file_path_under_kernel_dir);
    }

    void cleanup_kernel_dir() {
        const string &kernel_dir = OptionsG.get_kernel_dir();
        for (const std::filesystem::directory_entry &entry : std::filesystem::directory_iterator(kernel_dir)) {
            std::filesystem::remove_all(entry);
        }
    }

    Device *device_;
    ProgramHandle program_handle_;
    Program* program_;

   private:
    void validate_preconditions() {
        this->validate_env_vars_are_set();
        this->validate_kernel_dir_is_valid();
    }

    void validate_env_vars_are_set() {
        if (!OptionsG.is_root_dir_specified()) {
            GTEST_SKIP() << "Skipping test: TT_METAL_HOME must be set";
        }
        if (!OptionsG.is_kernel_dir_specified()) {
            GTEST_SKIP() << "Skipping test: TT_METAL_KERNEL_PATH must be set";
        }
    }

    void validate_kernel_dir_is_valid() {
        const string &kernel_dir = llrt::OptionsG.get_kernel_dir();
        if (!this->does_path_exist(kernel_dir) || !this->is_path_a_directory(kernel_dir) ||
            !this->is_dir_empty(kernel_dir)) {
            GTEST_SKIP() << "Skipping test: TT_METAL_KERNEL_PATH must be an existing, empty directory";
        }
    }

    bool does_path_exist(const string &path) {
        const std::filesystem::path &file_path(path);
        return std::filesystem::exists(file_path);
    }

    bool is_path_a_directory(const string &path) {
        TT_FATAL(this->does_path_exist(path), "{} does not exist", path);
        const std::filesystem::path &file_path(path);
        return std::filesystem::is_directory(file_path);
    }

    bool is_dir_empty(const string &path) {
        TT_FATAL(this->does_path_exist(path), "{} does not exist", path);
        TT_FATAL(this->is_path_a_directory(path), "{} is not a directory", path);
        const std::filesystem::path &file_path(path);
        return std::filesystem::is_empty(file_path);
    }
};

TEST_F(CompileProgramWithKernelPathEnvVarFixture, KernelUnderMetalRootDir) {
    const string &kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";
    create_kernel(kernel_file);
    detail::CompileProgram(this->device_, *this->program_);
}

TEST_F(CompileProgramWithKernelPathEnvVarFixture, KernelUnderKernelRootDir) {
    const string &orig_kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";
    const string &new_kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/new_kernel.cpp";
    this->setup_kernel_dir(orig_kernel_file, new_kernel_file);
    this->create_kernel(new_kernel_file);
    detail::CompileProgram(this->device_, *this->program_);
    this->cleanup_kernel_dir();
}

TEST_F(CompileProgramWithKernelPathEnvVarFixture, KernelUnderMetalRootDirAndKernelRootDir) {
    const string &kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";
    this->setup_kernel_dir(kernel_file, kernel_file);
    this->create_kernel(kernel_file);
    detail::CompileProgram(this->device_, *this->program_);
    this->cleanup_kernel_dir();
}

TEST_F(CompileProgramWithKernelPathEnvVarFixture, NonExistentKernel) {
    const string &kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/non_existent_kernel.cpp";
    this->create_kernel(kernel_file);
    EXPECT_THROW(detail::CompileProgram(this->device_, *this->program_), std::exception);
}
