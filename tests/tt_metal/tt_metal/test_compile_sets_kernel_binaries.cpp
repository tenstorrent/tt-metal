// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <enchantum/enchantum.hpp>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-logger/tt-logger.hpp>

#include "jit_build/build.hpp"
#include "llrt.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "tt_memory.h"
#include "tt_metal/detail/kernel_cache.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include <umd/device/types/arch.hpp>

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace {

std::string get_latest_kernel_binary_path(const std::string& kernel_root_path, const std::shared_ptr<Kernel>& kernel) {
    TT_FATAL(kernel != nullptr, "Error");
    TT_FATAL(std::filesystem::exists(kernel_root_path + kernel->name()), "Error");

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
    return kernel->name() + "/" + latest_hash;
}

void construct_program(Program& program, IDevice* device, CoreCoord& core) {
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig buff_config{
        .device = device, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};

    auto src_dram_buffer = CreateBuffer(buff_config);
    auto dst_dram_buffer = CreateBuffer(buff_config);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 8;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {uint(num_tiles)};

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args});
}

}  // namespace

// Custom fixture for multi-device test
class CompileSetsKernelBinariesFixture : public ::testing::Test {
protected:
    void SetUp() override {
        num_devices_ = GetNumAvailableDevices();
        std::vector<int> ids;
        ids.reserve(num_devices_);
        for (int id = 0; id < num_devices_; id++) {
            ids.push_back(id);
        }
        devices_ = detail::CreateDevices(ids);
    }

    void TearDown() override {
        if (!devices_.empty()) {
            detail::CloseDevices(devices_);
        }
    }

    std::map<int, IDevice*> devices_;
    int num_devices_ = 0;
};

TEST_F(CompileSetsKernelBinariesFixture, CompileSetsKernelBinaries) {
    CoreCoord core = {0, 0};
    std::vector<Program> programs;
    std::map<uint64_t, std::vector<const ll_api::memory*>> compute_binaries;
    std::map<uint64_t, std::vector<const ll_api::memory*>> brisc_binaries;
    std::map<uint64_t, std::vector<const ll_api::memory*>> ncrisc_binaries;

    for (int i = 0; i < num_devices_; i++) {
        auto* device = devices_[i];

        programs.push_back(Program());
        Program& program = programs.back();

        construct_program(program, device, core);

        uint32_t programmable_core_index =
            MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
        const KernelGroup* kernel_group = program.impl().kernels_on_core(core, programmable_core_index);
        TT_FATAL(kernel_group != nullptr, "Error");
        std::shared_ptr<Kernel> compute_kernel = nullptr;
        std::shared_ptr<Kernel> riscv0_kernel = nullptr;
        std::shared_ptr<Kernel> riscv1_kernel = nullptr;
        for (auto kernel_id : kernel_group->kernel_ids) {
            auto kernel = program.impl().get_kernel(kernel_id);
            switch (kernel->get_kernel_processor_class()) {
                case HalProcessorClassType::DM:
                    switch (kernel->get_kernel_processor_type(0)) {
                        case 0: riscv0_kernel = kernel; break;
                        case 1: riscv1_kernel = kernel; break;
                        default: TT_THROW("Error");
                    }
                    break;
                case HalProcessorClassType::COMPUTE: compute_kernel = kernel; break;
                default: TT_THROW("Error");
            }
        }
        TT_FATAL(compute_kernel != nullptr && riscv0_kernel != nullptr && riscv1_kernel != nullptr, "Error");

        auto mask = BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key();
        detail::CompileProgram(device, program);
        compute_binaries.insert({mask, compute_kernel->binaries(mask)});
        TT_FATAL(compute_binaries.at(mask).size() == 3, "Expected 3 Compute binaries!");
        brisc_binaries.insert({mask, riscv0_kernel->binaries(mask)});
        TT_FATAL(brisc_binaries.at(mask).size() == 1, "Expected 1 BRISC binary!");
        ncrisc_binaries.insert({mask, riscv1_kernel->binaries(mask)});
        TT_FATAL(ncrisc_binaries.at(mask).size() == 1, "Expected 1 NCRISC binary!");
    }

    int num_compiles = 3;
    for (int iter = 0; iter < 3; iter++) {
        std::vector<std::string> kernel_names = {"reader_unary_push_4", "writer_unary", "eltwise_copy_3m"};
        for (int i = 0; i < num_devices_; i++) {
            for (const auto& kernel_name : kernel_names) {
                std::filesystem::remove_all(
                    BuildEnvManager::get_instance()
                        .get_device_build_env(devices_[i]->id())
                        .build_env.get_out_kernel_root_path() +
                    kernel_name);
            }
        }
        detail::ClearKernelCache();
        std::vector<Program> new_programs;
        for (int i = 0; i < num_devices_; i++) {
            auto& device = devices_[i];
            new_programs.push_back(Program());
            Program& program = new_programs.back();
            construct_program(program, device, core);
        }

        std::vector<std::thread> ths;
        ths.reserve(num_devices_);
        uint32_t dm_class_idx = enchantum::to_underlying(HalProcessorClassType::DM);
        uint32_t compute_class_idx = enchantum::to_underlying(HalProcessorClassType::COMPUTE);
        for (int i = 0; i < num_devices_; i++) {
            auto& device = devices_[i];
            auto& program = new_programs[i];
            ths.emplace_back([&] {
                for (int j = 0; j < num_compiles; j++) {
                    auto mask = BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key();
                    detail::CompileProgram(device, program);
                    uint32_t programmable_core_index = MetalContext::instance().hal().get_programmable_core_type_index(
                        HalProgrammableCoreType::TENSIX);
                    const KernelGroup* kernel_group = program.impl().kernels_on_core(core, programmable_core_index);
                    std::shared_ptr<Kernel> compute_kernel = nullptr;
                    std::shared_ptr<Kernel> riscv0_kernel = nullptr;
                    std::shared_ptr<Kernel> riscv1_kernel = nullptr;
                    for (auto kernel_id : kernel_group->kernel_ids) {
                        auto kernel = program.impl().get_kernel(kernel_id);
                        switch (kernel->get_kernel_processor_class()) {
                            case HalProcessorClassType::DM:
                                switch (kernel->get_kernel_processor_type(0)) {
                                    case 0: riscv0_kernel = kernel; break;
                                    case 1: riscv1_kernel = kernel; break;
                                    default: TT_THROW("Error");
                                }
                                break;
                            case HalProcessorClassType::COMPUTE: compute_kernel = kernel; break;
                            default: TT_THROW("Error");
                        }
                    }
                    TT_FATAL(
                        compute_kernel != nullptr && riscv0_kernel != nullptr && riscv1_kernel != nullptr, "Error");
                    TT_FATAL(compute_kernel->binaries(mask) == compute_binaries.at(mask), "Error");
                    TT_FATAL(riscv0_kernel->binaries(mask) == brisc_binaries.at(mask), "Error");
                    TT_FATAL(riscv1_kernel->binaries(mask) == ncrisc_binaries.at(mask), "Error");

                    std::string kernel_name = get_latest_kernel_binary_path(
                        BuildEnvManager::get_instance()
                            .get_device_build_env(device->build_id())
                            .build_env.get_out_kernel_root_path(),
                        riscv0_kernel);
                    std::string brisc_hex_path =
                        BuildEnvManager::get_instance()
                            .get_kernel_build_state(device->build_id(), programmable_core_index, dm_class_idx, 0)
                            .get_target_out_path(kernel_name);
                    const ll_api::memory& brisc_binary =
                        llrt::get_risc_binary(brisc_hex_path, ll_api::memory::Loading::CONTIGUOUS_XIP);
                    TT_FATAL(
                        brisc_binary == *brisc_binaries.at(mask).at(0),
                        "Expected saved BRISC binary to be the same as binary in persistent cache");
                    kernel_name = get_latest_kernel_binary_path(
                        BuildEnvManager::get_instance()
                            .get_device_build_env(device->build_id())
                            .build_env.get_out_kernel_root_path(),
                        riscv1_kernel);
                    std::string ncrisc_hex_path =
                        BuildEnvManager::get_instance()
                            .get_kernel_build_state(device->build_id(), programmable_core_index, dm_class_idx, 1)
                            .get_target_out_path(kernel_name);
                    auto load_type = (device->arch() == tt::ARCH::GRAYSKULL || device->arch() == tt::ARCH::WORMHOLE_B0)
                                         ? ll_api::memory::Loading::CONTIGUOUS
                                         : ll_api::memory::Loading::CONTIGUOUS_XIP;
                    const ll_api::memory& ncrisc_binary = llrt::get_risc_binary(ncrisc_hex_path, load_type);
                    TT_FATAL(
                        ncrisc_binary == *ncrisc_binaries.at(mask).at(0),
                        "Expected saved NCRISC binary to be the same as binary in persistent cache");
                    for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
                        kernel_name = get_latest_kernel_binary_path(
                            BuildEnvManager::get_instance()
                                .get_device_build_env(device->build_id())
                                .build_env.get_out_kernel_root_path(),
                            compute_kernel);
                        std::string trisc_id_str = std::to_string(trisc_id);
                        std::string trisc_hex_path =
                            BuildEnvManager::get_instance()
                                .get_kernel_build_state(
                                    device->build_id(), programmable_core_index, compute_class_idx, trisc_id)
                                .get_target_out_path(kernel_name);
                        const ll_api::memory& trisc_binary =
                            llrt::get_risc_binary(trisc_hex_path, ll_api::memory::Loading::CONTIGUOUS_XIP);
                        TT_FATAL(
                            trisc_binary == *compute_binaries.at(mask).at(trisc_id),
                            "Expected saved TRISC binary for {} to be the same as binary in persistent cache",
                            trisc_id_str);
                    }
                }
            });
        }
        for (auto& th : ths) {
            th.join();
        }
    }
}
