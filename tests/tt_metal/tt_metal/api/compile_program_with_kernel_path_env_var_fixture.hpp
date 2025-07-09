// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#include <gtest/gtest.h>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"

using namespace tt;

class CompileProgramWithKernelPathEnvVarFixture : public ::testing::Test {
protected:
    void SetUp() override {
        if (!this->are_preconditions_satisfied()) {
            GTEST_SKIP();
        }

        const chip_id_t device_id = 0;
        this->device_ = tt::tt_metal::CreateDevice(device_id);
        this->program_ = tt::tt_metal::CreateProgram();
    }

    void TearDown() override {
        if (!IsSkipped()) {
            tt::tt_metal::CloseDevice(this->device_);
        }
    }

    void create_kernel(const std::string& kernel_file) {
        CoreCoord core(0, 0);
        tt_metal::CreateKernel(
            this->program_,
            kernel_file,
            core,
            tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::RISCV_1_default});
    }

    void setup_kernel_dir(const std::string& orig_kernel_file, const std::string& new_kernel_file) {
        const std::string& kernel_dir = tt_metal::MetalContext::instance().rtoptions().get_kernel_dir();
        const std::filesystem::path& kernel_file_path_under_kernel_dir(kernel_dir + new_kernel_file);
        const std::filesystem::path& dirs_under_kernel_dir = kernel_file_path_under_kernel_dir.parent_path();
        std::filesystem::create_directories(dirs_under_kernel_dir);

        const std::string& metal_root = tt_metal::MetalContext::instance().rtoptions().get_root_dir();
        const std::filesystem::path& kernel_file_path_under_metal_root(metal_root + orig_kernel_file);
        std::filesystem::copy(kernel_file_path_under_metal_root, kernel_file_path_under_kernel_dir);
    }

    void cleanup_kernel_dir() {
        const std::string& kernel_dir = tt_metal::MetalContext::instance().rtoptions().get_kernel_dir();
        for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(kernel_dir)) {
            std::filesystem::remove_all(entry);
        }
    }

    tt::tt_metal::IDevice* device_;
    tt::tt_metal::Program program_;

private:
    bool are_preconditions_satisfied() { return this->are_env_vars_set() && this->is_kernel_dir_valid(); }

    bool are_env_vars_set() {
        bool are_set = true;
        if (!tt_metal::MetalContext::instance().rtoptions().is_root_dir_specified()) {
            log_info(LogTest, "Skipping test: TT_METAL_HOME must be set");
            are_set = false;
        }
        if (!tt_metal::MetalContext::instance().rtoptions().is_kernel_dir_specified()) {
            log_info(LogTest, "Skipping test: TT_METAL_KERNEL_PATH must be set");
            are_set = false;
        }
        return are_set;
    }

    bool is_kernel_dir_valid() {
        bool is_valid = true;
        const std::string& kernel_dir = tt_metal::MetalContext::instance().rtoptions().get_kernel_dir();
        if (!this->does_path_exist(kernel_dir) || !this->is_path_a_directory(kernel_dir) ||
            !this->is_dir_empty(kernel_dir)) {
            log_info(LogTest, "Skipping test: TT_METAL_KERNEL_PATH must be an existing, empty directory");
            is_valid = false;
        }
        return is_valid;
    }

    bool does_path_exist(const std::string& path) {
        const std::filesystem::path& file_path(path);
        return std::filesystem::exists(file_path);
    }

    bool is_path_a_directory(const std::string& path) {
        TT_FATAL(this->does_path_exist(path), "{} does not exist", path);
        const std::filesystem::path& file_path(path);
        return std::filesystem::is_directory(file_path);
    }

    bool is_dir_empty(const std::string& path) {
        TT_FATAL(this->does_path_exist(path), "{} does not exist", path);
        TT_FATAL(this->is_path_a_directory(path), "{} is not a directory", path);
        const std::filesystem::path& file_path(path);
        return std::filesystem::is_empty(file_path);
    }
};
