// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/llrt/tt_memory.h"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/detail/kernel_cache.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/device/device_pool.hpp"
#include "llrt/hal.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

std::string get_latest_kernel_binary_path(uint32_t mask, const std::shared_ptr<Kernel> kernel) {
    auto root_dir = jit_build_get_kernel_compile_outpath(mask);
    TT_FATAL(kernel != nullptr, "Error");
    TT_FATAL(std::filesystem::exists(root_dir + kernel->name()), "Error");

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
    TT_FATAL(not latest_hash.empty(), "Error");
    return kernel->name() + "/" + latest_hash;
}

void construct_program(Program& program, Device * device, CoreCoord& core) {
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

    auto dram_src_noc_xy = src_dram_buffer->noc_coordinates();
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the
    // input CB CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to
    // math kernel, input CB and reader
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 8;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(
            num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = 16;  // output operands start at index 16
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
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles)  // per_core_tile_cnt
    };

    auto eltwise_unary_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args});
}

int main(int argc, char **argv) {
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
        tt::DevicePool::initialize(ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, DispatchCoreType::WORKER);
        auto devices = tt::DevicePool::instance().get_all_active_devices();
        std::vector<Program> programs;
        std::set<uint32_t> build_keys;
        // kernel->binaries() returns 32B aligned binaries
        std::map<uint32_t, std::vector<ll_api::memory>> compute_binaries;
        std::map<uint32_t, std::vector<ll_api::memory>> brisc_binaries;
        std::map<uint32_t, std::vector<ll_api::memory>> ncrisc_binaries;

        for (int i = 0; i < num_devices; i++) {
            auto device = devices[i];
            build_keys.insert(device->build_key());

            ////////////////////////////////////////////////////////////////////////////
            //                      Application Setup
            ////////////////////////////////////////////////////////////////////////////
            programs.push_back(Program());
            Program& program = programs.back();

            construct_program(program, device, core);

            ////////////////////////////////////////////////////////////////////////////
            //                      Compile Application
            ////////////////////////////////////////////////////////////////////////////
            // Check that binary memory objects in the kernel match the ones obtained from the persistent cache
            uint32_t programmable_core_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
            const KernelGroup* kernel_group = program.kernels_on_core(core, programmable_core_index);
            TT_FATAL(
                kernel_group != nullptr && kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE].has_value() and
                kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM0].has_value() and kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM1].has_value(),
                "Error");
            auto compute_kernel = tt_metal::detail::GetKernel(program, kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE].value());
            auto riscv0_kernel = tt_metal::detail::GetKernel(program, kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM0].value());
            auto riscv1_kernel = tt_metal::detail::GetKernel(program, kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM1].value());

            // Run iteration to get golden
            uint32_t mask = device->build_key();
            tt_metal::detail::CompileProgram(device, program);
            compute_binaries.insert({mask, compute_kernel->binaries(mask)});
            TT_FATAL(compute_binaries.at(mask).size() == 3, "Expected 3 Compute binaries!");
            brisc_binaries.insert({mask, riscv0_kernel->binaries(mask)});
            TT_FATAL(brisc_binaries.at(mask).size() == 1, "Expected 1 BRISC binary!");
            ncrisc_binaries.insert({mask, riscv1_kernel->binaries(mask)});
            TT_FATAL(ncrisc_binaries.at(mask).size() == 1, "Expected 1 NCRISC binary!");
        }

        int num_compiles = 3;
        for (int i = 0; i < 3; i++) {
            std::vector<string> kernel_names = {"reader_unary_push_4", "writer_unary", "eltwise_copy_3m"};
            for (auto build_key : build_keys) {
                for (auto kernel_name : kernel_names) {
                    std::filesystem::remove_all(jit_build_get_kernel_compile_outpath(build_key) + kernel_name);
                }
            }
            tt_metal::detail::ClearKernelCache();
            std::vector<Program> new_programs;
            for (int i = 0; i < num_devices; i++) {
                auto& device = devices[i];
                new_programs.push_back(Program());
                Program& program = new_programs.back();
                construct_program(program, device, core);
            }

            std::vector<std::thread> ths;
            ths.reserve(num_devices);
            for (int i = 0; i < num_devices; i++) {
                auto& device = devices[i];
                auto& program = new_programs[i];
                ths.emplace_back([&] {
                    for (int j = 0; j < num_compiles; j++) {
                        uint32_t mask = device->build_key();
                        tt_metal::detail::CompileProgram(device, program);
                        uint32_t programmable_core_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
                        const KernelGroup* kernel_group = program.kernels_on_core(core, programmable_core_index);
                        auto compute_kernel = tt_metal::detail::GetKernel(program, kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_COMPUTE].value());
                        auto riscv0_kernel = tt_metal::detail::GetKernel(program, kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM0].value());
                        auto riscv1_kernel = tt_metal::detail::GetKernel(program, kernel_group->kernel_ids[DISPATCH_CLASS_TENSIX_DM1].value());
                        TT_FATAL(compute_kernel->binaries(mask) == compute_binaries.at(mask), "Error");
                        TT_FATAL(riscv0_kernel->binaries(mask) == brisc_binaries.at(mask), "Error");
                        TT_FATAL(riscv1_kernel->binaries(mask) == ncrisc_binaries.at(mask), "Error");

                        std::string brisc_hex_path = device->build_kernel_target_path(
                            JitBuildProcessorType::DATA_MOVEMENT,
                            0,
                            get_latest_kernel_binary_path(mask, riscv0_kernel));
                        ll_api::memory brisc_binary = llrt::get_risc_binary(brisc_hex_path, 0, llrt::PackSpans::PACK);
                        TT_FATAL(
                            brisc_binary == brisc_binaries.at(mask).at(0),
                            "Expected saved BRISC binary to be the same as binary in persistent cache");
                        std::string ncrisc_hex_path = device->build_kernel_target_path(
                            JitBuildProcessorType::DATA_MOVEMENT,
                            1,
                            get_latest_kernel_binary_path(mask, riscv1_kernel));
                        ll_api::memory ncrisc_binary = llrt::get_risc_binary(ncrisc_hex_path, 1, llrt::PackSpans::PACK);
                        TT_FATAL(
                            ncrisc_binary == ncrisc_binaries.at(mask).at(0),
                            "Expected saved NCRISC binary to be the same as binary in persistent cache");
                        for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
                            std::string trisc_id_str = std::to_string(trisc_id);
                            std::string trisc_hex_path = device->build_kernel_target_path(
                                JitBuildProcessorType::COMPUTE,
                                trisc_id,
                                get_latest_kernel_binary_path(mask, compute_kernel));
                            ll_api::memory trisc_binary = llrt::get_risc_binary(trisc_hex_path, 2, llrt::PackSpans::PACK);
                            TT_FATAL(
                                trisc_binary == compute_binaries.at(mask).at(trisc_id),
                                "Expected saved TRISC binary for {} to be the same as binary in persistent cache", trisc_id_str);
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

    TT_FATAL(pass, "Error");

    return 0;
}
