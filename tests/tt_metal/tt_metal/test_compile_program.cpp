// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>
#include <cstdlib>
#include <filesystem>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/llrt/tt_memory.h"
#include "tt_metal/detail/kernel_cache.hpp"
#include "tt_metal/detail/tt_metal.hpp"

#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"

using namespace tt;
using namespace tt::tt_metal;

struct KernelCacheStatus {
    std::unordered_map<std::string, std::string> kernel_name_to_hash_str;
    std::unordered_map<std::string, bool> kernel_name_to_cache_hit;
};

void ClearKernelCache (uint32_t build_key){
    std::filesystem::remove_all(jit_build_get_kernel_compile_outpath(build_key));
    detail::HashLookup::inst().clear();
}

// This assumes binaries are written to specific location: kernel_compile_outpath / kernel_name / hash
std::unordered_map<std::string, std::string> get_last_program_binary_path(const Program &program, int build_key) {
    std::unordered_map<std::string, std::string> kernel_name_to_last_compiled_dir;
    auto root_dir = jit_build_get_kernel_compile_outpath(build_key);
    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        auto kernel = detail::GetKernel(program, kernel_id);
        if (not std::filesystem::exists(root_dir + kernel->name())) {
            continue;
        }

        std::filesystem::path kernel_path{root_dir + kernel->name()};
        std::filesystem::file_time_type ftime = std::filesystem::last_write_time(*kernel_path.begin());
        std::string latest_hash;
        for (auto const& dir_entry : std::filesystem::directory_iterator{kernel_path}) {
            auto kbtime = std::filesystem::last_write_time(dir_entry.path());
            if (kbtime > ftime) {
                ftime = kbtime;
                latest_hash = dir_entry.path().filename().string();
            }
        }
        TT_FATAL(not latest_hash.empty());
        kernel_name_to_last_compiled_dir.insert({kernel->name(), latest_hash});
    }
    return kernel_name_to_last_compiled_dir;
}

// TODO: Replace this when we have debug/test hooks (GH: #964) to inspect inside CompileProgram
KernelCacheStatus CompileProgramTestWrapper(Device *device, Program &program, bool profile_kernel=false) {
    // Check
    auto root_dir = jit_build_get_kernel_compile_outpath(device->build_key());
    std::unordered_map<std::string, std::string> pre_compile_kernel_to_hash_str = get_last_program_binary_path(program, device->build_key());

    detail::CompileProgram(device, program);

    std::unordered_map<std::string, std::string> post_compile_kernel_to_hash_str = get_last_program_binary_path(program, device->build_key());

    KernelCacheStatus kernel_cache_status;
    for (const auto&[kernel_name, hash_str] : post_compile_kernel_to_hash_str) {
        if (pre_compile_kernel_to_hash_str.find(kernel_name) == pre_compile_kernel_to_hash_str.end()) {
            kernel_cache_status.kernel_name_to_cache_hit.insert({kernel_name, false});
        } else {
            auto prev_hash_str = pre_compile_kernel_to_hash_str.at(kernel_name);
            bool cache_hit = hash_str == prev_hash_str;
            kernel_cache_status.kernel_name_to_cache_hit.insert({kernel_name, cache_hit});
        }
        kernel_cache_status.kernel_name_to_hash_str.insert({kernel_name, hash_str});
    }
    return kernel_cache_status;
}

struct ProgramAttributes {
    uint32_t num_tiles=2048;
    MathFidelity math_fidelity=MathFidelity::HiFi4;
    bool fp32_dest_acc_en=false;
    bool math_approx_mode=false;
    DataMovementProcessor reader_processor = DataMovementProcessor::RISCV_1;
    DataMovementProcessor writer_processor = DataMovementProcessor::RISCV_0;
    NOC reader_noc = NOC::RISCV_1_default;
    NOC writer_noc = NOC::RISCV_0_default;
    tt::DataFormat data_format = tt::DataFormat::Float16_b;
    uint32_t src_cb_index = 0;
    uint32_t output_cb_index = 16;
};

Program create_program(Device *device, const ProgramAttributes &program_attributes) {

    CoreCoord core = {0, 0};
    tt_metal::Program program = tt_metal::CreateProgram();

    uint32_t single_tile_size = 2 * 1024;

    // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
    // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
    uint32_t num_input_tiles = 8;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{program_attributes.src_cb_index, program_attributes.data_format}})
        .set_page_size(program_attributes.src_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    // output operands start at index 16
    uint32_t num_output_tiles = 1;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{program_attributes.output_cb_index, program_attributes.data_format}})
        .set_page_size(program_attributes.output_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    auto unary_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = program_attributes.reader_processor, .noc = program_attributes.reader_noc});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = program_attributes.writer_processor, .noc = program_attributes.writer_noc});

    vector<uint32_t> compute_kernel_args = {
        uint(program_attributes.num_tiles) // per_core_tile_cnt
    };

    auto eltwise_unary_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = program_attributes.math_fidelity,
            .fp32_dest_acc_en = program_attributes.fp32_dest_acc_en,
            .math_approx_mode = program_attributes.math_approx_mode,
            .compile_args = compute_kernel_args
        }
    );

    return std::move(program);
}

void assert_kernel_binary_path_exists(const Program &program, int build_key, const KernelCacheStatus &kernel_cache_status) {
    auto kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;
    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        auto kernel = detail::GetKernel(program, kernel_id);
        auto hash = kernel_name_to_hash.at(kernel->name());
        auto kernel_binary_path = jit_build_get_kernel_compile_outpath(build_key) + kernel->name() + "/" + hash;
        TT_FATAL(std::filesystem::exists(kernel_binary_path), "Expected " + kernel_binary_path + " folder to exist!");
    }
}

void assert_program_cache_hit_status(const Program &program, bool hit_expected, const KernelCacheStatus &kernel_cache_status) {
    auto kernel_name_to_cache_hit_status = kernel_cache_status.kernel_name_to_cache_hit;
    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        auto kernel = detail::GetKernel(program, kernel_id);
        auto hit_status = kernel_name_to_cache_hit_status.at(kernel->name());
        TT_FATAL(hit_status == hit_expected, "Did not get expected cache status " + std::to_string(hit_expected) + " for kernel " + kernel->name());
    }
}

void assert_kernel_hash_matches(const std::unordered_map<std::string, std::string> &golden_kernel_name_to_hash, const KernelCacheStatus &kernel_cache_status) {
    for (const auto &[kernel_name, hash] : kernel_cache_status.kernel_name_to_hash_str) {
        auto expected_hash = golden_kernel_name_to_hash.at(kernel_name);
        TT_FATAL(hash == expected_hash, "Expected hash for " + kernel_name + " " + expected_hash + " but got " + hash);
    }
}

bool test_compile_program_in_loop(Device *device) {
    bool pass = true;

    ClearKernelCache(device->build_key());
    ProgramAttributes default_attributes;
    auto program = create_program(device, default_attributes);

    static constexpr int num_compiles = 10;
    std::unordered_map<std::string, std::string> kernel_name_to_hash;
    for (int compile_idx = 0; compile_idx < num_compiles; compile_idx++) {
        auto kernel_cache_status = CompileProgramTestWrapper(device, program);
        if (compile_idx == 0) {
            assert_kernel_binary_path_exists(program, device->build_key(), kernel_cache_status);
            assert_program_cache_hit_status(program, /*hit_expected=*/false, kernel_cache_status);
            kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;
        } else {
            assert_program_cache_hit_status(program, /*hit_expected=*/true, kernel_cache_status);
            assert_kernel_hash_matches(kernel_name_to_hash, kernel_cache_status);
        }
    }

    return pass;
}

bool test_compile_program_after_clean_kernel_binary_directory(Device *device) {
    bool pass = true;

    ClearKernelCache(device->build_key());

    ProgramAttributes default_attributes;
    auto program = create_program(device, default_attributes);

    auto kernel_cache_status = CompileProgramTestWrapper(device, program);

    assert_kernel_binary_path_exists(program, device->build_key(), kernel_cache_status);
    assert_program_cache_hit_status(program, /*hit_expected=*/false, kernel_cache_status);
    std::unordered_map<std::string, std::string> kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;

    ClearKernelCache(device->build_key());
    program.invalidate_compile();
    auto second_kernel_cache_status = CompileProgramTestWrapper(device, program);
    assert_program_cache_hit_status(program, /*hit_expected=*/false, second_kernel_cache_status);
    assert_kernel_hash_matches(kernel_name_to_hash, second_kernel_cache_status);

    return pass;
}

void assert_hash_comparison_for_kernel_type(
    const Program &program,
    const std::unordered_map<std::string, std::string> &prev_kernel_name_to_hash,
    const std::unordered_map<tt::RISCV, bool> &type_to_same_hash_expected,
    const KernelCacheStatus &kernel_cache_status
) {
    auto curr_kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;
    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        auto kernel = detail::GetKernel(program, kernel_id);
        auto prev_hash = prev_kernel_name_to_hash.at(kernel->name());
        auto curr_hash = curr_kernel_name_to_hash.at(kernel->name());
        bool same_hash_expected = type_to_same_hash_expected.at(kernel->processor());
        if (same_hash_expected) {
            TT_FATAL(prev_hash == curr_hash, "Expected same hashes for " + kernel->name());
        } else {
            TT_FATAL(prev_hash != curr_hash, "Expected different hashes for " + kernel->name());
        }
    }
}

void assert_cache_hit_status_for_kernel_type(const Program &program, const std::unordered_map<tt::RISCV, bool> &type_to_cache_hit_status, const KernelCacheStatus &kernel_cache_status) {
    auto kernel_name_to_cache_hit_status = kernel_cache_status.kernel_name_to_cache_hit;
    for (size_t kernel_id = 0; kernel_id < program.num_kernels(); kernel_id++) {
        auto kernel = detail::GetKernel(program, kernel_id);
        bool hit_expected = type_to_cache_hit_status.at(kernel->processor());
        auto hit_status = kernel_name_to_cache_hit_status.at(kernel->name());
        TT_FATAL(hit_status == hit_expected, "Did not get expected cache status " + std::to_string(hit_expected) + " for kernel " + kernel->name());
    }
}

std::unordered_map<std::string, std::string> compile_program_with_modified_kernel(
    Device *device,
    const ProgramAttributes &attributes,
    const std::unordered_map<std::string, std::string> &prev_kernel_name_to_hash,
    const std::unordered_map<tt::RISCV, bool> &kernel_type_to_cache_hit_status
) {
    auto program = create_program(device, attributes);
    auto kernel_cache_status = CompileProgramTestWrapper(device, program);
    assert_kernel_binary_path_exists(program, device->build_key(), kernel_cache_status);
    assert_cache_hit_status_for_kernel_type(program, kernel_type_to_cache_hit_status, kernel_cache_status);
    assert_hash_comparison_for_kernel_type(program, prev_kernel_name_to_hash, kernel_type_to_cache_hit_status, kernel_cache_status);
    std::unordered_map<std::string, std::string> kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;
    return kernel_name_to_hash;
}

bool test_compile_program_with_modified_program(Device *device) {
    bool pass = true;

    const static std::unordered_map<tt::RISCV, bool> compute_miss_data_movement_hit = {
        {tt::RISCV::COMPUTE, false},
        {tt::RISCV::BRISC, true},
        {tt::RISCV::NCRISC, true}
    };

    const static std::unordered_map<tt::RISCV, bool> compute_hit_data_movement_miss = {
        {tt::RISCV::COMPUTE, true},
        {tt::RISCV::BRISC, false},
        {tt::RISCV::NCRISC, false}
    };

    const static std::unordered_map<tt::RISCV, bool> compute_hit_data_movement_hit = {
        {tt::RISCV::COMPUTE, true},
        {tt::RISCV::BRISC, true},
        {tt::RISCV::NCRISC, true}
    };

    const static std::unordered_map<tt::RISCV, bool> compute_miss_data_movement_miss = {
        {tt::RISCV::COMPUTE, false},
        {tt::RISCV::BRISC, false},
        {tt::RISCV::NCRISC, false}
    };

    ClearKernelCache(device->build_key());

    ProgramAttributes attributes;
    auto program = create_program(device, attributes);
    auto kernel_cache_status = CompileProgramTestWrapper(device, program);
    assert_kernel_binary_path_exists(program, device->build_key(), kernel_cache_status);
    assert_program_cache_hit_status(program, /*hit_expected=*/false, kernel_cache_status);
    std::unordered_map<std::string, std::string> kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;

    // Modify compute kernel compile time args - expect cache miss for compute kernel
    attributes.num_tiles = 1024;
    kernel_name_to_hash = compile_program_with_modified_kernel(device, attributes, kernel_name_to_hash, compute_miss_data_movement_hit);

    // Modify compute kernel math fidelity - expect cache miss for compute kernel
    attributes.math_fidelity = MathFidelity::LoFi;
    kernel_name_to_hash = compile_program_with_modified_kernel(device, attributes, kernel_name_to_hash, compute_miss_data_movement_hit);

    // Modify compute kernel fp32_dest_acc_en - expect cache miss for compute kernel
    // Grayskull does not support fp32 accumulation
    if (device->arch() != ARCH::GRAYSKULL) {
        attributes.fp32_dest_acc_en = true;
        kernel_name_to_hash = compile_program_with_modified_kernel(device, attributes, kernel_name_to_hash, compute_miss_data_movement_hit);
    }

    // Modify compute kernel math_approx_mode - expect cache miss for compute kernel
    attributes.math_approx_mode = true;
    kernel_name_to_hash = compile_program_with_modified_kernel(device, attributes, kernel_name_to_hash, compute_miss_data_movement_hit);

    // Modify data movement kernel noc - expect cache miss for data movement kernels
    attributes.reader_noc = NOC::RISCV_0_default;
    attributes.writer_noc = NOC::RISCV_1_default;
    kernel_name_to_hash = compile_program_with_modified_kernel(device, attributes, kernel_name_to_hash, compute_hit_data_movement_miss);

    // Modify data movement kernel processor - expect cache hit
    attributes.reader_processor = DataMovementProcessor::RISCV_1;
    attributes.writer_processor = DataMovementProcessor::RISCV_0;
    kernel_name_to_hash = compile_program_with_modified_kernel(device, attributes, kernel_name_to_hash, compute_hit_data_movement_hit);

    // Modify circular buffer data format - expect cache miss for all kernels
    attributes.data_format = tt::DataFormat::Bfp8_b;
    kernel_name_to_hash = compile_program_with_modified_kernel(device, attributes, kernel_name_to_hash, compute_miss_data_movement_miss);

    // Modify circular buffer index - expect cache miss for all kernels
    attributes.src_cb_index = attributes.src_cb_index + 1;
    attributes.output_cb_index = attributes.output_cb_index + 1;
    kernel_name_to_hash = compile_program_with_modified_kernel(device, attributes, kernel_name_to_hash, compute_miss_data_movement_miss);

    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        int device_id = 0;
        Device *device = CreateDevice(device_id);

        constexpr bool profile_device = true;


        pass &= test_compile_program_in_loop(device);

        pass &= test_compile_program_after_clean_kernel_binary_directory(device);

        pass &= test_compile_program_with_modified_program(device);

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
