// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reusing programs in this context means enqueuing them across multiple devices, without creating a copy per device.
// This requires the program object to track state that is universal across devices (in our case through
// virtualization). The tests in this file create a single program and enqueue it across multiple devices. These tests
// are skipped for architectures that don't have Coordinate Virtualization enabled (i.e. GS and BH). These will be
// enabled on BH once FW is updated to support Virtual Coordinates. These tests are different from Single Device
// EnqueueProgram tests, since they validate state (Binaries, Semaphores and RTAs) across multiple devices, by comparing
// it with a single program on host. These tests are also structurally different from single device program tests. While
// those tests follow the pattern below: for device ... devices:
//      p = CreateProgram()
//      EnqueueProgram(p, device)
// The ones below follow this pattern:
// p = CreateProgram()
// for device ... devices:
//      EnqueueProgram(p, device)
// This makes it non-trivial to share the host-setup code across tests.

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/allocator.hpp>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/circular_buffer_config.hpp>
#include "command_queue_fixture.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/runtime_args_data.hpp>
#include <tt-metalium/semaphore.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_metal.hpp>
#include "umd/device/tt_core_coordinates.h"
#include "umd/device/types/arch.h"
#include <tt-metalium/util.hpp>
#include <tt-metalium/utils.hpp>

namespace tt::tt_metal {
struct CBConfig {
    uint32_t cb_id;
    uint32_t num_pages;
    uint32_t page_size;
    DataFormat data_format;
};

struct DummyProgramConfig {
    CoreRangeSet cr_set;
    CBConfig cb_config;
    uint32_t num_cbs;
    uint32_t num_sems;
};

struct DummyProgramMultiCBConfig {
    CoreRangeSet cr_set;
    std::vector<CBConfig> cb_config_vector;
    uint32_t num_sems;
};

std::shared_ptr<Program> create_program_multi_core_rta(
    const DummyProgramConfig& rta_program_config,
    const DummyProgramConfig& sem_program_config,
    const std::vector<uint32_t>& dummy_cr0_args,
    const std::vector<uint32_t>& dummy_cr1_args,
    const std::vector<uint32_t>& sem_values,
    const DummyProgramMultiCBConfig& cb_configs,
    uint32_t base_addr) {
    std::shared_ptr<Program> program = std::make_shared<Program>();
    CoreRangeSet rta_cr_set = rta_program_config.cr_set;
    uint32_t rta_base_dm0 = base_addr;
    uint32_t rta_base_dm1 = rta_base_dm0 + 1024 * sizeof(uint32_t);
    uint32_t rta_base_compute = rta_base_dm1 + 2048 * sizeof(uint32_t);

    uint32_t coord_base_dm0 =
        tt::align(base_addr + 4096 * sizeof(uint32_t), MetalContext::instance().hal().get_alignment(HalMemType::L1));
    uint32_t coord_base_dm1 = coord_base_dm0 + 1024 * sizeof(uint32_t);

    std::map<std::string, std::string> dm_defines0 = {
        {"DATA_MOVEMENT", "1"},
        {"NUM_RUNTIME_ARGS", std::to_string(256)},
        {"RESULTS_ADDR", std::to_string(rta_base_dm0)},
        {"COORDS_ADDR", std::to_string(coord_base_dm0)}};
    std::map<std::string, std::string> dm_defines1 = {
        {"DATA_MOVEMENT", "1"},
        {"NUM_RUNTIME_ARGS", std::to_string(256)},
        {"RESULTS_ADDR", std::to_string(rta_base_dm1)},
        {"COORDS_ADDR", std::to_string(coord_base_dm1)}};
    std::map<std::string, std::string> compute_defines = {
        {"COMPUTE", "1"},
        {"NUM_RUNTIME_ARGS", std::to_string(256)},
        {"RESULTS_ADDR", std::to_string(rta_base_compute)}};

    auto dummy_kernel0 = CreateKernel(
        *program,
        "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
        rta_cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = dm_defines0});

    auto dummy_kernel1 = CreateKernel(
        *program,
        "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
        rta_cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = dm_defines1});

    auto dummy_compute_kernel = CreateKernel(
        *program,
        "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
        rta_cr_set,
        ComputeConfig{.defines = compute_defines});

    auto it = rta_program_config.cr_set.ranges().begin();
    CoreRange core_range_0 = *it;
    std::advance(it, 1);
    CoreRange core_range_1 = *it;

    for (const CoreCoord& core_coord : core_range_0) {
        SetRuntimeArgs(*program, dummy_kernel0, core_coord, dummy_cr0_args);
        SetRuntimeArgs(*program, dummy_kernel1, core_coord, dummy_cr0_args);
        SetRuntimeArgs(*program, dummy_compute_kernel, core_coord, dummy_cr0_args);
    }

    for (const CoreCoord& core_coord : core_range_1) {
        SetRuntimeArgs(*program, dummy_kernel0, core_coord, dummy_cr1_args);
        SetRuntimeArgs(*program, dummy_kernel1, core_coord, dummy_cr1_args);
        SetRuntimeArgs(*program, dummy_compute_kernel, core_coord, dummy_cr1_args);
    }
    for (auto sem_value : sem_values) {
        CreateSemaphore(*program, sem_program_config.cr_set, sem_value);
    }
    for (auto cb_config : cb_configs.cb_config_vector) {
        const uint32_t cb_id = cb_config.cb_id;
        const uint32_t cb_num_pages = cb_config.num_pages;
        const uint32_t page_size = cb_config.page_size;
        const uint32_t cb_size = cb_num_pages * page_size;
        const DataFormat data_format = cb_config.data_format;
        const CircularBufferConfig circular_buffer_config =
            CircularBufferConfig(cb_size, {{cb_id, data_format}}).set_page_size(cb_id, page_size);
        const CBHandle cb_handle = CreateCircularBuffer(*program, cb_configs.cr_set, circular_buffer_config);
    }

    return program;
}

TEST_F(CommandQueueMultiDeviceFixture, TestProgramReuseSanity) {
    // Sanity test: Create a program with Semaphores, CBs, RTAs, Core Coords, and Kernel Binaries.
    // Enqueue Program across all devices.
    // Read L1 directly to ensure that all program attributes are correctly present.
    if (devices_[0]->arch() != ARCH::WORMHOLE_B0) {
        GTEST_SKIP();
    }
    CoreCoord worker_grid_size = devices_[0]->compute_with_storage_grid_size();
    CoreRange cr0({0, 0}, {worker_grid_size.x - 1, 3});
    CoreRange cr1({0, 4}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet rta_cr_set(std::vector{cr0, cr1});
    DummyProgramConfig rta_program_config = {.cr_set = rta_cr_set};
    CoreRange sem_cr_range({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet sem_cr_set = CoreRangeSet(sem_cr_range);
    DummyProgramConfig sem_program_config = {.cr_set = sem_cr_set};

    std::vector<uint32_t> dummy_cr0_args;
    std::vector<uint32_t> dummy_cr1_args;
    std::vector<uint32_t> dummy_sems;
    uint32_t num_runtime_args_for_cr0 = 32;
    uint32_t num_runtime_args_for_cr1 = 35;
    // Initialize RTA data across core_ranges
    uint32_t start_idx = 25;
    for (uint32_t i = 0; i < num_runtime_args_for_cr0; i++) {
        dummy_cr0_args.push_back(start_idx++);
    }
    for (uint32_t i = 0; i < num_runtime_args_for_cr1; i++) {
        dummy_cr1_args.push_back(start_idx++);
    }
    // Initialize Semaphore values
    for (uint32_t i = 0; i < NUM_SEMAPHORES; i++) {
        dummy_sems.push_back(i + 1);
    }

    // Initialize CB Configs
    CBConfig cb_config_0 = {.cb_id = 0, .num_pages = 1, .page_size = 2048, .data_format = DataFormat::Float16_b};
    CBConfig cb_config_1 = {.cb_id = 1, .num_pages = 2, .page_size = 4096, .data_format = DataFormat::Float16_b};
    CBConfig cb_config_2 = {.cb_id = 2, .num_pages = 2, .page_size = 2048, .data_format = DataFormat::Float16_b};
    CBConfig cb_config_3 = {.cb_id = 3, .num_pages = 4, .page_size = 2048, .data_format = DataFormat::Float16_b};

    DummyProgramMultiCBConfig cb_config = {
        .cr_set = sem_cr_set, .cb_config_vector = {cb_config_0, cb_config_1, cb_config_2, cb_config_3}};

    uint32_t rta_base_addr = devices_[0]->allocator()->get_base_allocator_addr(HalMemType::L1);
    auto program = create_program_multi_core_rta(
        rta_program_config, sem_program_config, dummy_cr0_args, dummy_cr1_args, dummy_sems, cb_config, rta_base_addr);

    constexpr uint32_t coordinate_readback_size = 2 * 3 * sizeof(uint32_t);  // (X,Y) x (Virtual, Logical, Relative) = 6
    // Below addresses are copied from create_program_multi_core_rta
    uint32_t rta_base_dm0 = rta_base_addr;
    uint32_t rta_base_dm1 = rta_base_dm0 + 1024 * sizeof(uint32_t);
    uint32_t rta_base_compute = rta_base_dm1 + 2048 * sizeof(uint32_t);

    // Put the coordinates way above the maximum RTAs
    uint32_t coord_base_dm0 = tt::align(
        rta_base_addr + 4096 * sizeof(uint32_t), MetalContext::instance().hal().get_alignment(HalMemType::L1));
    uint32_t coord_base_dm1 = coord_base_dm0 + 1024 * sizeof(uint32_t);
    uint32_t semaphore_buffer_size = dummy_sems.size() * MetalContext::instance().hal().get_alignment(HalMemType::L1);
    uint32_t cb_config_buffer_size =
        NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);
    for (auto device : devices_) {
        log_info(LogTest, "Running test on {}", device->id());
        EnqueueProgram(device->command_queue(), *program, false);
        Finish(device->command_queue());
        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());

        for (const CoreCoord& core_coord : cr0) {
            const auto& virtual_core_coord = device->worker_core_from_logical_core(core_coord);
            std::vector<uint32_t> expected_core_coordinates_dm{
                virtual_core_coord.x, virtual_core_coord.y, core_coord.x, core_coord.y, core_coord.x, core_coord.y};
            std::vector<uint32_t> dummy_kernel0_args_readback;
            detail::ReadFromDeviceL1(
                device,
                core_coord,
                rta_base_dm0,
                num_runtime_args_for_cr0 * sizeof(uint32_t),
                dummy_kernel0_args_readback);
            EXPECT_EQ(dummy_cr0_args, dummy_kernel0_args_readback);

            std::vector<uint32_t> dummy_kernel0_coords_readback;
            detail::ReadFromDeviceL1(
                device, core_coord, coord_base_dm0, coordinate_readback_size, dummy_kernel0_coords_readback);
            EXPECT_EQ(expected_core_coordinates_dm, dummy_kernel0_coords_readback);

            std::vector<uint32_t> dummy_kernel1_args_readback;
            detail::ReadFromDeviceL1(
                device,
                core_coord,
                rta_base_dm1,
                num_runtime_args_for_cr0 * sizeof(uint32_t),
                dummy_kernel1_args_readback);
            EXPECT_EQ(dummy_cr0_args, dummy_kernel1_args_readback);

            std::vector<uint32_t> dummy_kernel1_coords_readback;
            detail::ReadFromDeviceL1(
                device, core_coord, coord_base_dm1, coordinate_readback_size, dummy_kernel1_coords_readback);
            EXPECT_EQ(expected_core_coordinates_dm, dummy_kernel1_coords_readback);

            std::vector<uint32_t> dummy_compute_args_readback;
            detail::ReadFromDeviceL1(
                device,
                core_coord,
                rta_base_compute,
                num_runtime_args_for_cr0 * sizeof(uint32_t),
                dummy_compute_args_readback);
            EXPECT_EQ(dummy_cr0_args, dummy_compute_args_readback);
        }

        for (const CoreCoord& core_coord : cr1) {
            std::vector<uint32_t> dummy_kernel0_args_readback;
            detail::ReadFromDeviceL1(
                device,
                core_coord,
                rta_base_dm0,
                num_runtime_args_for_cr1 * sizeof(uint32_t),
                dummy_kernel0_args_readback);
            EXPECT_EQ(dummy_cr1_args, dummy_kernel0_args_readback);

            std::vector<uint32_t> dummy_kernel1_args_readback;
            detail::ReadFromDeviceL1(
                device,
                core_coord,
                rta_base_dm1,
                num_runtime_args_for_cr1 * sizeof(uint32_t),
                dummy_kernel1_args_readback);
            EXPECT_EQ(dummy_cr1_args, dummy_kernel1_args_readback);

            std::vector<uint32_t> dummy_compute_args_readback;
            detail::ReadFromDeviceL1(
                device,
                core_coord,
                rta_base_compute,
                num_runtime_args_for_cr1 * sizeof(uint32_t),
                dummy_compute_args_readback);
            EXPECT_EQ(dummy_cr1_args, dummy_compute_args_readback);
        }
        std::vector<uint32_t> semaphore_vals;
        for (const CoreCoord& core_coord : sem_cr_range) {
            uint32_t expected_sem_idx = 0;
            uint32_t sem_base_addr = program->get_sem_base_addr(devices_[0], core_coord, CoreType::WORKER);
            detail::ReadFromDeviceL1(device, core_coord, sem_base_addr, semaphore_buffer_size, semaphore_vals);
            for (uint32_t i = 0; i < semaphore_vals.size();
                 i += (MetalContext::instance().hal().get_alignment(HalMemType::L1) / sizeof(uint32_t))) {
                EXPECT_EQ(semaphore_vals[i], dummy_sems[expected_sem_idx]) << expected_sem_idx;
                expected_sem_idx++;
            }
        }
        std::vector<uint32_t> cb_config_vector;
        for (const CoreCoord& core_coord : sem_cr_range) {
            detail::ReadFromDeviceL1(
                device,
                core_coord,
                program->get_cb_base_addr(device, core_coord, CoreType::WORKER),
                cb_config_buffer_size,
                cb_config_vector);
            uint32_t cb_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
            for (uint32_t i = 0; i < cb_config.cb_config_vector.size(); i++) {
                const uint32_t index = cb_config.cb_config_vector[i].cb_id * sizeof(uint32_t);
                const uint32_t cb_num_pages = cb_config.cb_config_vector[i].num_pages;
                const uint32_t cb_size = cb_num_pages * cb_config.cb_config_vector[i].page_size;
                EXPECT_EQ(cb_config_vector.at(index), cb_addr);
                EXPECT_EQ(cb_config_vector.at(index + 1), cb_size);
                EXPECT_EQ(cb_config_vector.at(index + 2), cb_num_pages);

                cb_addr += cb_size;
            }
        }
    }
}

TEST_F(CommandQueueMultiDeviceFixture, TestDataCopyComputeProgramReuse) {
    // End to End full-grid test. Creates a Program with the same kernel running on the full logical grid.
    // Each core reads from a buffer, performs a simple math operation on each datum in the buffer, and writes
    // outputs in dedicated DRAM memory.
    // The same program is enqueued across all devices, and results are validated.
    if (devices_[0]->arch() != ARCH::WORMHOLE_B0) {
        GTEST_SKIP();
    }
    CoreCoord worker_grid_size = this->devices_[0]->compute_with_storage_grid_size();
    std::vector<std::shared_ptr<Buffer>> input_buffers = {};
    std::vector<std::shared_ptr<Buffer>> output_buffers = {};
    uint32_t single_tile_size = detail::TileSize(DataFormat::Float16_b);

    uint32_t num_tiles = 1;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    for (auto device : this->devices_) {
        InterleavedBufferConfig dram_config{
            .device = device, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                input_buffers.push_back(CreateBuffer(dram_config));
                output_buffers.push_back(CreateBuffer(dram_config));
            }
        }
    }

    Program program = CreateProgram();
    auto full_grid = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    auto reader_writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/full_grid_eltwise_device_reuse.cpp",
        full_grid,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto sem_scaling_factor = 2;
    auto scaling_sem_idx = CreateSemaphore(program, full_grid, sem_scaling_factor);
    uint32_t scaling_height_toggle = 16;
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(dram_buffer_size, {{src0_cb_index, DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);

    uint32_t add_factor = 64;
    for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
        for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
            CoreCoord curr_core = {col_idx, row_idx};
            SetRuntimeArgs(
                program,
                reader_writer_kernel,
                curr_core,
                {input_buffers.at(col_idx * worker_grid_size.y + row_idx)->address(),
                 output_buffers.at(col_idx * worker_grid_size.y + row_idx)->address(),
                 0, /* src_bank_id */
                 0, /* dst_bank_id */
                 add_factor,
                 constants::TILE_HEIGHT,
                 constants::TILE_WIDTH,
                 scaling_sem_idx,
                 scaling_height_toggle});
            CBHandle cb_src0 = CreateCircularBuffer(program, curr_core, cb_src0_config);
        }
    }

    std::vector<uint32_t> src_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 1);
    std::size_t buffer_idx = 0;
    // Write constant inputs once
    for (auto device : devices_) {
        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                EnqueueWriteBuffer(device->command_queue(), input_buffers.at(buffer_idx), src_vec, false);
                buffer_idx++;
            }
        }
    }
    // Run program multiple times with different RTAs and validate for each iteration
    for (int iter = 0; iter < 100; iter++) {
        log_info(LogTest, "Run iter {}", iter);
        if (iter) {
            auto& rtas = GetRuntimeArgs(program, reader_writer_kernel);
            for (auto core : full_grid) {
                rtas[core.x][core.y].at(4) = ((iter % 2) + 1) * add_factor;
            }
        }
        for (auto device : devices_) {
            EnqueueProgram(device->command_queue(), program, false);
        }

        buffer_idx = 0;
        for (auto device : devices_) {
            for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
                for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                    std::vector<bfloat16> dst_vec = {};
                    EnqueueReadBuffer(device->command_queue(), output_buffers.at(buffer_idx), dst_vec, true);
                    buffer_idx++;
                    for (int i = 0; i < dst_vec.size(); i++) {
                        float ref_val = std::pow(2, (iter % 2) + 1);
                        if (i >= 512) {
                            ref_val = std::pow(2, 2 * ((iter % 2) + 1));
                        }
                        EXPECT_EQ(dst_vec[i].to_float(), ref_val);
                    }
                }
            }
        }
    }
}

}  // namespace tt::tt_metal
