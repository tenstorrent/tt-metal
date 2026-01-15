// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/mesh_dispatch_fixture.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "hal_types.hpp"
#include "jit_build/build.hpp"
#include "tt_metal/detail/kernel_cache.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include <umd/device/types/arch.hpp>

// Access to internal API: ProgramImpl::num_kernel, get_kernel
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace {

struct KernelCacheStatus {
    std::unordered_map<std::string, std::string> kernel_name_to_hash_str;
    std::unordered_map<std::string, bool> kernel_name_to_cache_hit;
};

void ClearKernelCache(const std::string& kernel_root_path) {
    std::filesystem::remove_all(kernel_root_path);
    detail::HashLookup::inst().clear();
}

// This assumes binaries are written to specific location: kernel_compile_outpath / kernel_name / hash
std::unordered_map<std::string, std::string> get_last_program_binary_path(
    const Program& program, const std::string& kernel_root_path) {
    std::unordered_map<std::string, std::string> kernel_name_to_last_compiled_dir;
    for (size_t kernel_id = 0; kernel_id < program.impl().num_kernels(); kernel_id++) {
        auto kernel = program.impl().get_kernel(kernel_id);
        if (not std::filesystem::exists(kernel_root_path + kernel->name())) {
            continue;
        }

        std::filesystem::path kernel_path{kernel_root_path + kernel->name()};
        std::filesystem::file_time_type ftime = std::filesystem::last_write_time(*kernel_path.begin());
        std::string latest_hash;
        for (const auto& dir_entry : std::filesystem::directory_iterator{kernel_path}) {
            auto kbtime = std::filesystem::last_write_time(dir_entry.path());
            if (kbtime > ftime) {
                ftime = kbtime;
                latest_hash = dir_entry.path().filename().string();
            }
        }
        TT_FATAL(not latest_hash.empty(), "Error");
        kernel_name_to_last_compiled_dir.insert({kernel->name(), latest_hash});
    }
    return kernel_name_to_last_compiled_dir;
}

KernelCacheStatus CompileProgramTestWrapper(IDevice* device, Program& program, bool /*profile_kernel*/ = false) {
    std::unordered_map<std::string, std::string> pre_compile_kernel_to_hash_str = get_last_program_binary_path(
        program,
        BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_env.get_out_kernel_root_path());

    detail::CompileProgram(device, program);

    std::unordered_map<std::string, std::string> post_compile_kernel_to_hash_str = get_last_program_binary_path(
        program,
        BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_env.get_out_kernel_root_path());

    KernelCacheStatus kernel_cache_status;
    for (const auto& [kernel_name, hash_str] : post_compile_kernel_to_hash_str) {
        if (!pre_compile_kernel_to_hash_str.contains(kernel_name)) {
            kernel_cache_status.kernel_name_to_cache_hit.insert({kernel_name, false});
        } else {
            const auto& prev_hash_str = pre_compile_kernel_to_hash_str.at(kernel_name);
            bool cache_hit = hash_str == prev_hash_str;
            kernel_cache_status.kernel_name_to_cache_hit.insert({kernel_name, cache_hit});
        }
        kernel_cache_status.kernel_name_to_hash_str.insert({kernel_name, hash_str});
    }
    return kernel_cache_status;
}

struct ProgramAttributes {
    uint32_t num_tiles = 2048;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    DataMovementProcessor reader_processor = DataMovementProcessor::RISCV_1;
    DataMovementProcessor writer_processor = DataMovementProcessor::RISCV_0;
    NOC reader_noc = NOC::RISCV_1_default;
    NOC writer_noc = NOC::RISCV_0_default;
    tt::DataFormat data_format = tt::DataFormat::Float16_b;
    uint32_t src_cb_index = tt::CBIndex::c_0;
    uint32_t output_cb_index = tt::CBIndex::c_16;
};

Program create_program(IDevice* /*device*/, const ProgramAttributes& program_attributes) {
    CoreCoord core = {0, 0};
    Program program = CreateProgram();

    uint32_t single_tile_size = 2 * 1024;

    uint32_t num_input_tiles = 8;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(
            num_input_tiles * single_tile_size, {{program_attributes.src_cb_index, program_attributes.data_format}})
            .set_page_size(program_attributes.src_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(
            num_output_tiles * single_tile_size, {{program_attributes.output_cb_index, program_attributes.data_format}})
            .set_page_size(program_attributes.output_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        core,
        DataMovementConfig{.processor = program_attributes.reader_processor, .noc = program_attributes.reader_noc});

    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = program_attributes.writer_processor, .noc = program_attributes.writer_noc});

    vector<uint32_t> compute_kernel_args = {uint(program_attributes.num_tiles)};

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
        core,
        ComputeConfig{
            .math_fidelity = program_attributes.math_fidelity,
            .fp32_dest_acc_en = program_attributes.fp32_dest_acc_en,
            .math_approx_mode = program_attributes.math_approx_mode,
            .compile_args = compute_kernel_args});

    return program;
}

void assert_kernel_binary_path_exists(
    const Program& program, const std::string& kernel_root_path, const KernelCacheStatus& kernel_cache_status) {
    auto kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;
    for (size_t kernel_id = 0; kernel_id < program.impl().num_kernels(); kernel_id++) {
        auto kernel = program.impl().get_kernel(kernel_id);
        auto hash = kernel_name_to_hash.at(kernel->name());
        auto kernel_binary_path = kernel_root_path + kernel->name() + "/" + hash;
        TT_FATAL(std::filesystem::exists(kernel_binary_path), "Expected {} folder to exist!", kernel_binary_path);
    }
}

void assert_program_cache_hit_status(
    const Program& program, bool hit_expected, const KernelCacheStatus& kernel_cache_status) {
    auto kernel_name_to_cache_hit_status = kernel_cache_status.kernel_name_to_cache_hit;
    for (size_t kernel_id = 0; kernel_id < program.impl().num_kernels(); kernel_id++) {
        auto kernel = program.impl().get_kernel(kernel_id);
        auto hit_status = kernel_name_to_cache_hit_status.at(kernel->name());
        TT_FATAL(
            hit_status == hit_expected,
            "Did not get expected cache status {} for kernel {}",
            hit_expected,
            kernel->name());
    }
}

void assert_kernel_hash_matches(
    const std::unordered_map<std::string, std::string>& golden_kernel_name_to_hash,
    const KernelCacheStatus& kernel_cache_status) {
    for (const auto& [kernel_name, hash] : kernel_cache_status.kernel_name_to_hash_str) {
        const auto& expected_hash = golden_kernel_name_to_hash.at(kernel_name);
        TT_FATAL(hash == expected_hash, "Expected hash for {} {} but got {}", kernel_name, expected_hash, hash);
    }
}

void assert_hash_comparison_for_kernel_type(
    const Program& program,
    const std::unordered_map<std::string, std::string>& prev_kernel_name_to_hash,
    const std::unordered_map<HalProcessorClassType, bool>& type_to_same_hash_expected,
    const KernelCacheStatus& kernel_cache_status) {
    auto curr_kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;
    for (size_t kernel_id = 0; kernel_id < program.impl().num_kernels(); kernel_id++) {
        auto kernel = program.impl().get_kernel(kernel_id);
        auto prev_hash = prev_kernel_name_to_hash.at(kernel->name());
        auto curr_hash = curr_kernel_name_to_hash.at(kernel->name());
        bool same_hash_expected = type_to_same_hash_expected.at(kernel->get_kernel_processor_class());
        if (same_hash_expected) {
            TT_FATAL(prev_hash == curr_hash, "Expected same hashes for {}", kernel->name());
        } else {
            TT_FATAL(prev_hash != curr_hash, "Expected different hashes for {}", kernel->name());
        }
    }
}

void assert_cache_hit_status_for_kernel_type(
    const Program& program,
    const std::unordered_map<HalProcessorClassType, bool>& type_to_cache_hit_status,
    const KernelCacheStatus& kernel_cache_status) {
    auto kernel_name_to_cache_hit_status = kernel_cache_status.kernel_name_to_cache_hit;
    for (size_t kernel_id = 0; kernel_id < program.impl().num_kernels(); kernel_id++) {
        auto kernel = program.impl().get_kernel(kernel_id);
        bool hit_expected = type_to_cache_hit_status.at(kernel->get_kernel_processor_class());
        auto hit_status = kernel_name_to_cache_hit_status.at(kernel->name());
        TT_FATAL(
            hit_status == hit_expected,
            "Did not get expected cache status {} for kernel {}",
            hit_expected,
            kernel->name());
    }
}

std::unordered_map<std::string, std::string> compile_program_with_modified_kernel(
    IDevice* device,
    const ProgramAttributes& attributes,
    const std::unordered_map<std::string, std::string>& prev_kernel_name_to_hash,
    const std::unordered_map<HalProcessorClassType, bool>& kernel_type_to_cache_hit_status) {
    auto program = create_program(device, attributes);
    auto kernel_cache_status = CompileProgramTestWrapper(device, program);
    assert_kernel_binary_path_exists(
        program,
        BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_env.get_out_kernel_root_path(),
        kernel_cache_status);
    assert_cache_hit_status_for_kernel_type(program, kernel_type_to_cache_hit_status, kernel_cache_status);
    assert_hash_comparison_for_kernel_type(
        program, prev_kernel_name_to_hash, kernel_type_to_cache_hit_status, kernel_cache_status);
    std::unordered_map<std::string, std::string> kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;
    return kernel_name_to_hash;
}

}  // namespace

TEST_F(MeshDispatchFixture, CompileProgramInLoop) {
    IDevice* dev = devices_[0]->get_devices()[0];

    ClearKernelCache(
        BuildEnvManager::get_instance().get_device_build_env(dev->build_id()).build_env.get_out_kernel_root_path());
    ProgramAttributes default_attributes;
    auto program = create_program(dev, default_attributes);

    static constexpr int num_compiles = 10;
    std::unordered_map<std::string, std::string> kernel_name_to_hash;
    for (int compile_idx = 0; compile_idx < num_compiles; compile_idx++) {
        auto kernel_cache_status = CompileProgramTestWrapper(dev, program);
        if (compile_idx == 0) {
            assert_kernel_binary_path_exists(
                program,
                BuildEnvManager::get_instance()
                    .get_device_build_env(dev->build_id())
                    .build_env.get_out_kernel_root_path(),
                kernel_cache_status);
            assert_program_cache_hit_status(program, /*hit_expected=*/false, kernel_cache_status);
            kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;
        } else {
            assert_program_cache_hit_status(program, /*hit_expected=*/true, kernel_cache_status);
            assert_kernel_hash_matches(kernel_name_to_hash, kernel_cache_status);
        }
    }
}

TEST_F(MeshDispatchFixture, CompileProgramAfterCleanKernelBinaryDirectory) {
    IDevice* dev = devices_[0]->get_devices()[0];

    ClearKernelCache(
        BuildEnvManager::get_instance().get_device_build_env(dev->build_id()).build_env.get_out_kernel_root_path());

    ProgramAttributes default_attributes;
    auto program = create_program(dev, default_attributes);

    auto kernel_cache_status = CompileProgramTestWrapper(dev, program);

    assert_kernel_binary_path_exists(
        program,
        BuildEnvManager::get_instance().get_device_build_env(dev->build_id()).build_env.get_out_kernel_root_path(),
        kernel_cache_status);
    assert_program_cache_hit_status(program, /*hit_expected=*/false, kernel_cache_status);
    std::unordered_map<std::string, std::string> kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;

    ClearKernelCache(
        BuildEnvManager::get_instance().get_device_build_env(dev->build_id()).build_env.get_out_kernel_root_path());
    auto second_program = create_program(dev, default_attributes);
    auto second_kernel_cache_status = CompileProgramTestWrapper(dev, second_program);
    assert_program_cache_hit_status(second_program, /*hit_expected=*/false, second_kernel_cache_status);
    assert_kernel_hash_matches(kernel_name_to_hash, second_kernel_cache_status);
}

TEST_F(MeshDispatchFixture, CompileProgramWithModifiedProgram) {
    IDevice* dev = devices_[0]->get_devices()[0];

    const static std::unordered_map<HalProcessorClassType, bool> compute_miss_data_movement_hit = {
        {HalProcessorClassType::COMPUTE, false}, {HalProcessorClassType::DM, true}};

    const static std::unordered_map<HalProcessorClassType, bool> compute_hit_data_movement_miss = {
        {HalProcessorClassType::COMPUTE, true}, {HalProcessorClassType::DM, false}};

    const static std::unordered_map<HalProcessorClassType, bool> compute_hit_data_movement_hit = {
        {HalProcessorClassType::COMPUTE, true}, {HalProcessorClassType::DM, true}};

    const static std::unordered_map<HalProcessorClassType, bool> compute_miss_data_movement_miss = {
        {HalProcessorClassType::COMPUTE, false}, {HalProcessorClassType::DM, false}};

    ClearKernelCache(
        BuildEnvManager::get_instance().get_device_build_env(dev->build_id()).build_env.get_out_kernel_root_path());

    ProgramAttributes attributes;
    auto program = create_program(dev, attributes);
    auto kernel_cache_status = CompileProgramTestWrapper(dev, program);
    assert_kernel_binary_path_exists(
        program,
        BuildEnvManager::get_instance().get_device_build_env(dev->build_id()).build_env.get_out_kernel_root_path(),
        kernel_cache_status);
    assert_program_cache_hit_status(program, /*hit_expected=*/false, kernel_cache_status);
    std::unordered_map<std::string, std::string> kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;

    // Modify compute kernel compile time args - expect cache miss for compute kernel
    attributes.num_tiles = 1024;
    kernel_name_to_hash =
        compile_program_with_modified_kernel(dev, attributes, kernel_name_to_hash, compute_miss_data_movement_hit);

    // Modify compute kernel math fidelity - expect cache miss for compute kernel
    attributes.math_fidelity = MathFidelity::LoFi;
    kernel_name_to_hash =
        compile_program_with_modified_kernel(dev, attributes, kernel_name_to_hash, compute_miss_data_movement_hit);

    // Modify compute kernel fp32_dest_acc_en - expect cache miss for compute kernel
    // Grayskull does not support fp32 accumulation
    if (dev->arch() != ARCH::GRAYSKULL) {
        attributes.fp32_dest_acc_en = true;
        kernel_name_to_hash =
            compile_program_with_modified_kernel(dev, attributes, kernel_name_to_hash, compute_miss_data_movement_hit);
    }

    // Modify compute kernel math_approx_mode - expect cache miss for compute kernel
    attributes.math_approx_mode = true;
    kernel_name_to_hash =
        compile_program_with_modified_kernel(dev, attributes, kernel_name_to_hash, compute_miss_data_movement_hit);

    // Modify data movement kernel noc - expect cache miss for data movement kernels
    attributes.reader_noc = NOC::RISCV_0_default;
    attributes.writer_noc = NOC::RISCV_1_default;
    kernel_name_to_hash =
        compile_program_with_modified_kernel(dev, attributes, kernel_name_to_hash, compute_hit_data_movement_miss);

    // Modify data movement kernel processor - expect cache hit
    attributes.reader_processor = DataMovementProcessor::RISCV_1;
    attributes.writer_processor = DataMovementProcessor::RISCV_0;
    kernel_name_to_hash =
        compile_program_with_modified_kernel(dev, attributes, kernel_name_to_hash, compute_hit_data_movement_hit);

    // Modify circular buffer data format - expect cache miss for all kernels
    attributes.data_format = tt::DataFormat::Bfp8_b;
    kernel_name_to_hash =
        compile_program_with_modified_kernel(dev, attributes, kernel_name_to_hash, compute_miss_data_movement_miss);

    // Modify circular buffer index - expect cache miss for all kernels
    attributes.src_cb_index = attributes.src_cb_index + 1;
    attributes.output_cb_index = attributes.output_cb_index + 1;
    kernel_name_to_hash =
        compile_program_with_modified_kernel(dev, attributes, kernel_name_to_hash, compute_miss_data_movement_miss);
}
