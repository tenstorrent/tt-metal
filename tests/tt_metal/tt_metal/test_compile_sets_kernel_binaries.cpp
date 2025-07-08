// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <errno.h>
#include <fmt/base.h>
#include <magic_enum/magic_enum.hpp>
#include <stdint.h>
#include <sys/types.h>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <cstring>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "dev_msgs.h"
#include <tt-metalium/device.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/hal_types.hpp>
#include "hostdevcommon/common_values.hpp"
#include "hostdevcommon/kernel_structs.h"
#include "jit_build/build.hpp"
#include <tt-metalium/kernel_types.hpp>
#include "llrt.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel_impl.hpp"
#include "tt_memory.h"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/detail/kernel_cache.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include "umd/device/types/arch.h"
#include <tt-metalium/utils.hpp>

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;

std::string get_latest_kernel_binary_path(
    const std::string& kernel_root_path, const std::shared_ptr<tt_metal::Kernel>& kernel) {
    TT_FATAL(kernel != nullptr, "Error");
    TT_FATAL(std::filesystem::exists(kernel_root_path + kernel->name()), "Error");

    std::filesystem::path kernel_path{kernel_root_path + kernel->name()};
    std::filesystem::file_time_type ftime = std::filesystem::last_write_time(*kernel_path.begin());
    std::string latest_hash;
    for (auto const& dir_entry : std::filesystem::directory_iterator{kernel_path}) {
        auto kbtime = std::filesystem::last_write_time(dir_entry.path());
        if (kbtime > ftime) {
            ftime = kbtime;
            latest_hash = dir_entry.path().filename().string();
        }
    }
    TT_FATAL(not latest_hash.empty(), "Error");
    return kernel->name() + "/" + latest_hash;
}

void construct_program(tt_metal::Program& program, tt_metal::IDevice* device, CoreCoord& core) {
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size =
        single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    tt_metal::InterleavedBufferConfig buff_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    auto src_dram_buffer = CreateBuffer(buff_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();
    auto dst_dram_buffer = CreateBuffer(buff_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();


    // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the
    // input CB CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to
    // math kernel, input CB and reader
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 8;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 1;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    auto unary_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles)  // per_core_tile_cnt
    };

    auto eltwise_unary_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args});
}

int main(int argc, char** argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        CoreCoord core = {0, 0};
        int num_devices = tt::tt_metal::GetNumAvailableDevices();
        std::vector<int> ids;
        for (unsigned int id = 0; id < num_devices; id++) {
            ids.push_back(id);
        }
        tt::DevicePool::initialize(
            ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, tt_metal::DispatchCoreConfig{});
        auto devices = tt::DevicePool::instance().get_all_active_devices();
        std::vector<tt_metal::Program> programs;
        // kernel->binaries() returns 32B aligned binaries
        std::map<uint32_t, std::vector<ll_api::memory const*>> compute_binaries;
        std::map<uint32_t, std::vector<ll_api::memory const*>> brisc_binaries;
        std::map<uint32_t, std::vector<ll_api::memory const*>> ncrisc_binaries;

        for (int i = 0; i < num_devices; i++) {
            auto device = devices[i];

            ////////////////////////////////////////////////////////////////////////////
            //                      Application Setup
            ////////////////////////////////////////////////////////////////////////////
            programs.push_back(tt_metal::Program());
            tt_metal::Program& program = programs.back();

            construct_program(program, device, core);

            ////////////////////////////////////////////////////////////////////////////
            //                      Compile Application
            ////////////////////////////////////////////////////////////////////////////
            // Check that binary memory objects in the kernel match the ones obtained from the persistent cache
            uint32_t programmable_core_index =
                tt_metal::MetalContext::instance().hal().get_programmable_core_type_index(
                    tt_metal::HalProgrammableCoreType::TENSIX);
            const tt_metal::KernelGroup* kernel_group = program.impl().kernels_on_core(core, programmable_core_index);
            TT_FATAL(
                kernel_group != nullptr && kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE].has_value() and
                    kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM0].has_value() and
                    kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM1].has_value(),
                "Error");
            auto compute_kernel =
                tt_metal::detail::GetKernel(program, kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE].value());
            auto riscv0_kernel =
                tt_metal::detail::GetKernel(program, kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM0].value());
            auto riscv1_kernel =
                tt_metal::detail::GetKernel(program, kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM1].value());

            // Run iteration to get golden
            uint32_t mask =
                tt_metal::BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key;
            tt_metal::detail::CompileProgram(device, program);
            compute_binaries.insert({mask, tt_metal::KernelImpl::from(*compute_kernel).binaries(mask)});
            TT_FATAL(compute_binaries.at(mask).size() == 3, "Expected 3 Compute binaries!");
            brisc_binaries.insert({mask, tt_metal::KernelImpl::from(*riscv0_kernel).binaries(mask)});
            TT_FATAL(brisc_binaries.at(mask).size() == 1, "Expected 1 BRISC binary!");
            ncrisc_binaries.insert({mask, tt_metal::KernelImpl::from(*riscv1_kernel).binaries(mask)});
            TT_FATAL(ncrisc_binaries.at(mask).size() == 1, "Expected 1 NCRISC binary!");
        }

        int num_compiles = 3;
        for (int i = 0; i < 3; i++) {
            std::vector<std::string> kernel_names = {"reader_unary_push_4", "writer_unary", "eltwise_copy_3m"};
            for (int i = 0; i < num_devices; i++) {
                for (const auto& kernel_name : kernel_names) {
                    std::filesystem::remove_all(
                        tt_metal::BuildEnvManager::get_instance()
                            .get_device_build_env(devices[i]->id())
                            .build_env.get_out_kernel_root_path() +
                        kernel_name);
                }
            }
            tt_metal::detail::ClearKernelCache();
            std::vector<tt_metal::Program> new_programs;
            for (int i = 0; i < num_devices; i++) {
                auto& device = devices[i];
                new_programs.push_back(tt_metal::Program());
                tt_metal::Program& program = new_programs.back();
                construct_program(program, device, core);
            }

            std::vector<std::thread> ths;
            ths.reserve(num_devices);
            uint32_t dm_class_idx = magic_enum::enum_integer(tt_metal::HalProcessorClassType::DM);
            uint32_t compute_class_idx = magic_enum::enum_integer(tt_metal::HalProcessorClassType::COMPUTE);
            for (int i = 0; i < num_devices; i++) {
                auto& device = devices[i];
                auto& program = new_programs[i];
                ths.emplace_back([&] {
                    for (int j = 0; j < num_compiles; j++) {
                        uint32_t mask = tt_metal::BuildEnvManager::get_instance()
                                            .get_device_build_env(device->build_id())
                                            .build_key;
                        tt_metal::detail::CompileProgram(device, program);
                        uint32_t programmable_core_index =
                            tt_metal::MetalContext::instance().hal().get_programmable_core_type_index(
                                tt_metal::HalProgrammableCoreType::TENSIX);
                        const tt_metal::KernelGroup* kernel_group =
                            program.impl().kernels_on_core(core, programmable_core_index);
                        auto compute_kernel = tt_metal::detail::GetKernel(
                            program, kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE].value());
                        auto riscv0_kernel = tt_metal::detail::GetKernel(
                            program, kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM0].value());
                        auto riscv1_kernel = tt_metal::detail::GetKernel(
                            program, kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM1].value());
                        TT_FATAL(
                            tt_metal::KernelImpl::from(*compute_kernel).binaries(mask) == compute_binaries.at(mask),
                            "Error");
                        TT_FATAL(
                            tt_metal::KernelImpl::from(*riscv0_kernel).binaries(mask) == brisc_binaries.at(mask),
                            "Error");
                        TT_FATAL(
                            tt_metal::KernelImpl::from(*riscv1_kernel).binaries(mask) == ncrisc_binaries.at(mask),
                            "Error");

                        std::string kernel_name = get_latest_kernel_binary_path(
                            tt_metal::BuildEnvManager::get_instance()
                                .get_device_build_env(device->build_id())
                                .build_env.get_out_kernel_root_path(),
                            riscv0_kernel);
                        std::string brisc_hex_path =
                            tt_metal::BuildEnvManager::get_instance()
                                .get_kernel_build_state(device->build_id(), programmable_core_index, dm_class_idx, 0)
                                .get_target_out_path(kernel_name);
                        ll_api::memory const& brisc_binary =
                            llrt::get_risc_binary(brisc_hex_path, ll_api::memory::Loading::CONTIGUOUS_XIP);
                        TT_FATAL(
                            brisc_binary == *brisc_binaries.at(mask).at(0),
                            "Expected saved BRISC binary to be the same as binary in persistent cache");
                        kernel_name = get_latest_kernel_binary_path(
                            tt_metal::BuildEnvManager::get_instance()
                                .get_device_build_env(device->build_id())
                                .build_env.get_out_kernel_root_path(),
                            riscv1_kernel);
                        std::string ncrisc_hex_path =
                            tt_metal::BuildEnvManager::get_instance()
                                .get_kernel_build_state(device->build_id(), programmable_core_index, dm_class_idx, 1)
                                .get_target_out_path(kernel_name);
                        auto load_type =
                            (device->arch() == tt::ARCH::GRAYSKULL || device->arch() == tt::ARCH::WORMHOLE_B0)
                                ? ll_api::memory::Loading::CONTIGUOUS
                                : ll_api::memory::Loading::CONTIGUOUS_XIP;
                        ll_api::memory const& ncrisc_binary = llrt::get_risc_binary(ncrisc_hex_path, load_type);
                        TT_FATAL(
                            ncrisc_binary == *ncrisc_binaries.at(mask).at(0),
                            "Expected saved NCRISC binary to be the same as binary in persistent cache");
                        for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
                            kernel_name = get_latest_kernel_binary_path(
                                tt_metal::BuildEnvManager::get_instance()
                                    .get_device_build_env(device->build_id())
                                    .build_env.get_out_kernel_root_path(),
                                compute_kernel);
                            std::string trisc_id_str = std::to_string(trisc_id);
                            std::string trisc_hex_path =
                                tt_metal::BuildEnvManager::get_instance()
                                    .get_kernel_build_state(
                                        device->build_id(), programmable_core_index, compute_class_idx, trisc_id)
                                    .get_target_out_path(kernel_name);
                            ll_api::memory const& trisc_binary =
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
        for (auto dev : devices) {
            pass &= tt_metal::CloseDevice(dev);
        }

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
