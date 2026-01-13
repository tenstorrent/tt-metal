// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "env_lib.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include "mesh_dispatch_fixture.hpp"
#include "tt_metal/impl/context/metal_context.hpp"

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;

void build_and_run_program(
    const std::shared_ptr<distributed::MeshDevice>& device,
    bool /*slow_dispatch*/,
    uint32_t NUM_PROGRAMS,
    uint32_t MAX_LOOP,
    uint32_t page_size,
    bool mix_noc_mode) {
    // Make random
    auto random_seed = 0; // (unsigned int)time(NULL);
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);

    CoreCoord worker_grid_size = {1,1};
    CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set(cr);

    log_info(tt::LogTest, "Starting compile of {} programs now.", NUM_PROGRAMS);

    Program program1;
    Program program2;

    CircularBufferConfig cb_config =
        CircularBufferConfig(page_size, {{0, tt::DataFormat::Float16_b}}).set_page_size(0, page_size);
    CreateCircularBuffer(program1, cr_set, cb_config);
    CreateCircularBuffer(program2, cr_set, cb_config);

    // Add 2 semaphores initialized to 0 for each program, 1 per risc
    uint32_t program1_semaphore0 = CreateSemaphore(program1, cr_set, 0);
    uint32_t program1_semaphore1 = CreateSemaphore(program1, cr_set, 0);
    uint32_t program2_semaphore0 = CreateSemaphore(program2, cr_set, 0);
    uint32_t program2_semaphore1 = CreateSemaphore(program2, cr_set, 0);

    vector<uint32_t> compile_args = {MAX_LOOP, page_size, 2};
    tt_metal::TensorAccessorArgs::create_l1_interleaved().append_to(compile_args);

    auto brisc_kernel1 = CreateKernel(
        program1,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dynamic_noc_writer.cpp",
        cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = compile_args});

    auto ncrisc_kernel1 = CreateKernel(
        program1,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dynamic_noc_writer.cpp",
        cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = compile_args});

    auto brisc_kernel2 = CreateKernel(
        program2,
        mix_noc_mode ? "tests/tt_metal/tt_metal/test_kernels/dataflow/dedicated_noc_writer.cpp"
                     : "tests/tt_metal/tt_metal/test_kernels/dataflow/dynamic_noc_writer.cpp",
        cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .noc_mode = mix_noc_mode ? tt_metal::NOC_MODE::DM_DEDICATED_NOC : tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = compile_args});

    auto ncrisc_kernel2 = CreateKernel(
        program2,
        mix_noc_mode ? "tests/tt_metal/tt_metal/test_kernels/dataflow/dedicated_noc_writer.cpp"
                     : "tests/tt_metal/tt_metal/test_kernels/dataflow/dynamic_noc_writer.cpp",
        cr_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .noc_mode = mix_noc_mode ? tt_metal::NOC_MODE::DM_DEDICATED_NOC : tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
            .compile_args = compile_args});

    for (int core_idx_y = 0; core_idx_y < worker_grid_size.y; core_idx_y++) {
        for (int core_idx_x = 0; core_idx_x < worker_grid_size.x; core_idx_x++) {
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};
            CoreCoord neighbour_core = {core_idx_x == worker_grid_size.x - 1 ? 0 : core_idx_x + 1, core_idx_y};
            CoreCoord neighbour_core_physical = device->worker_core_from_logical_core(neighbour_core);
            // mcast
            auto device_grid = device->compute_with_storage_grid_size();
            CoreCoord top_left_core = {0, 0};
            CoreCoord top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
            CoreCoord bottom_right_core = {device_grid.x - 1, device_grid.y - 1};
            CoreCoord bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);
            std::vector<uint32_t> rt_args = {
                (std::uint32_t)neighbour_core_physical.x,
                (std::uint32_t)neighbour_core_physical.y,
                // mcast
                core_idx_x == 0 && core_idx_y == 0,
                top_left_core_physical.x,
                top_left_core_physical.y,
                bottom_right_core_physical.x,
                bottom_right_core_physical.y,
                device_grid.x * device_grid.y,
                0,  // risc index
                // semaphore IDs for program1
                program1_semaphore0,   // semaphore 0
                program1_semaphore1};  // semaphore 1

            tt::tt_metal::SetRuntimeArgs(program1, brisc_kernel1, core, rt_args);
            rt_args[8] = 1;  // risc index
            tt::tt_metal::SetRuntimeArgs(program1, ncrisc_kernel1, core, rt_args);

            rt_args[8] = 0;  // risc index
            // Override semaphore IDs for program2
            rt_args[9] = program2_semaphore0;   // semaphore 0
            rt_args[10] = program2_semaphore1;  // semaphore 1

            tt::tt_metal::SetRuntimeArgs(program2, brisc_kernel2, core, rt_args);
            rt_args[8] = 1;  // risc index
            tt::tt_metal::SetRuntimeArgs(program2, ncrisc_kernel2, core, rt_args);
        }
    }
    distributed::MeshWorkload workload1;
    distributed::MeshWorkload workload2;
    workload1.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program1));
    workload2.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program2));

    // This loop caches program1 and runs
    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        log_info(tt::LogTest, "Running program1 {} of {}", i + 1, NUM_PROGRAMS);
        if (i % 2 == 0) {
            distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload1, false);
        } else {
            distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload2, false);
        }
    }
    distributed::Finish(device->mesh_command_queue());
}

void build_and_run_program_ethernet(
    const std::shared_ptr<distributed::MeshDevice>& device,
    bool /*slow_dispatch*/,
    uint32_t NUM_PROGRAMS,
    uint32_t MAX_LOOP,
    uint32_t page_size,
    bool mix_noc_mode) {
    // Make random
    auto random_seed = 0;  // (unsigned int)time(NULL);
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);

    auto* device_0 = device->get_devices()[0];

    // Query the number of ethernet ERISCs
    const auto erisc_count = tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);

    // We need at least 2 ERISCs (ERISC0 and ERISC1) to run this test
    if (erisc_count < 2) {
        GTEST_SKIP() << "Skipping test as this test requires at least 2 ERISCs, but only " << erisc_count
                     << " available.";
        return;
    }

    // Get active ethernet cores
    const auto& active_eth_cores = device_0->get_active_ethernet_cores(true);
    if (active_eth_cores.empty()) {
        GTEST_SKIP() << "Skipping test as this test requires at least 1 active ethernet core, but none are available.";
        return;
    }

    // Pick the first active ethernet core
    CoreCoord eth_core = *active_eth_cores.begin();
    log_info(tt::LogTest, "Using ethernet core {} for testing", eth_core.str());

    log_info(tt::LogTest, "Starting compile of {} ethernet programs now.", NUM_PROGRAMS);

    Program program1;
    Program program2;

    // Add 2 semaphores initialized to 0 for each program, 1 per erisc
    CoreRangeSet eth_cr_set(CoreRange(eth_core, eth_core));
    uint32_t program1_semaphore0 = CreateSemaphore(program1, eth_cr_set, 0, CoreType::ETH);
    uint32_t program1_semaphore1 = CreateSemaphore(program1, eth_cr_set, 0, CoreType::ETH);
    uint32_t program2_semaphore0 = CreateSemaphore(program2, eth_cr_set, 0, CoreType::ETH);
    uint32_t program2_semaphore1 = CreateSemaphore(program2, eth_cr_set, 0, CoreType::ETH);

    vector<uint32_t> compile_args = {MAX_LOOP, page_size, 2};
    tt_metal::TensorAccessorArgs::create_l1_interleaved().append_to(compile_args);

    // Create ERISC0 kernel for program1 (Dynamic NOC)
    auto eth_erisc0_kernel1 = CreateKernel(
        program1,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dynamic_noc_writer_eth.cpp",
        eth_core,
        tt_metal::EthernetConfig{
            .noc = tt_metal::NOC::NOC_0,
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .compile_args = compile_args,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC});

    // Create ERISC1 kernel for program1 (Dynamic NOC)
    auto eth_erisc1_kernel1 = CreateKernel(
        program1,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dynamic_noc_writer_eth.cpp",
        eth_core,
        tt_metal::EthernetConfig{
            .noc = tt_metal::NOC::NOC_1,
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .compile_args = compile_args,
            .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC});

    // Note: dedicated_noc_writer.cpp uses circular buffers which are not supported on ethernet cores,
    // so we always use dynamic_noc_writer_eth.cpp for both programs regardless of mix_noc_mode.
    // We can test dedicated NOC mode with the dynamic NOC writer kernel by changing the noc_mode.
    auto eth_erisc0_kernel2 = CreateKernel(
        program2,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dynamic_noc_writer_eth.cpp",
        eth_core,
        tt_metal::EthernetConfig{
            .noc = tt_metal::NOC::NOC_0,
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .compile_args = compile_args,
            .noc_mode = mix_noc_mode ? tt_metal::NOC_MODE::DM_DEDICATED_NOC : tt_metal::NOC_MODE::DM_DYNAMIC_NOC});

    auto eth_erisc1_kernel2 = CreateKernel(
        program2,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dynamic_noc_writer_eth.cpp",
        eth_core,
        tt_metal::EthernetConfig{
            .noc = tt_metal::NOC::NOC_1,
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .compile_args = compile_args,
            .noc_mode = mix_noc_mode ? tt_metal::NOC_MODE::DM_DEDICATED_NOC : tt_metal::NOC_MODE::DM_DYNAMIC_NOC});

    log_info(
        tt::LogTest,
        "Created ethernet kernels: ERISC0 NOC0, ERISC1 NOC1 for both programs (program1: Dynamic NOC, program2: {})",
        mix_noc_mode ? "Dedicated NOC" : "Dynamic NOC");

    // Get physical coordinates for ethernet core
    CoreCoord eth_core_physical = device_0->virtual_core_from_logical_core(eth_core, CoreType::ETH);

    // For ethernet cores, we'll use a worker core as the NOC target
    CoreCoord worker_core = {0, 0};
    CoreCoord worker_core_physical = device_0->worker_core_from_logical_core(worker_core);

    // Get worker grid for multicast parameters
    auto device_grid = device_0->compute_with_storage_grid_size();
    CoreCoord top_left_core = {0, 0};
    CoreCoord top_left_core_physical = device_0->worker_core_from_logical_core(top_left_core);
    CoreCoord bottom_right_core = {device_grid.x - 1, device_grid.y - 1};
    CoreCoord bottom_right_core_physical = device_0->worker_core_from_logical_core(bottom_right_core);
    uint32_t num_dests = device_grid.x * device_grid.y;

    // Get safe L1 address from HAL for ethernet cores
    uint32_t l1_unreserved_base =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    log_info(tt::LogTest, "Using L1 unreserved base address: 0x{:x}", l1_unreserved_base);
    log_info(
        tt::LogTest,
        "Multicast grid: ({},{}) to ({},{}), {} destinations",
        top_left_core_physical.x,
        top_left_core_physical.y,
        bottom_right_core_physical.x,
        bottom_right_core_physical.y,
        num_dests);

    // Runtime arguments for dynamic_noc_writer_eth.cpp kernel:
    // Arg 0-1: noc_x, noc_y (target core)
    // Arg 2: risc_index
    // Arg 3: mcast_enable
    // Arg 4-7: multicast range (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    // Arg 8: num_dests
    // Arg 9-10: semaphores (for num_riscs=2)
    // Arg 11: l1_addr (safe L1 address from HAL)
    constexpr bool mcast_enable = true;
    std::vector<uint32_t> erisc0_rt_args_prog1 = {
        (std::uint32_t)worker_core_physical.x,
        (std::uint32_t)worker_core_physical.y,
        0,             // risc index (ERISC0)
        mcast_enable,  // mcast_enable
        top_left_core_physical.x,
        top_left_core_physical.y,
        bottom_right_core_physical.x,
        bottom_right_core_physical.y,
        num_dests,
        program1_semaphore0,  // semaphore 0
        program1_semaphore1,  // semaphore 1
        l1_unreserved_base};  // safe L1 address

    std::vector<uint32_t> erisc1_rt_args_prog1 = {
        (std::uint32_t)worker_core_physical.x,
        (std::uint32_t)worker_core_physical.y,
        1,             // risc index (ERISC1)
        mcast_enable,  // mcast_enable
        top_left_core_physical.x,
        top_left_core_physical.y,
        bottom_right_core_physical.x,
        bottom_right_core_physical.y,
        num_dests,
        program1_semaphore0,  // semaphore 0
        program1_semaphore1,  // semaphore 1
        l1_unreserved_base};  // safe L1 address

    std::vector<uint32_t> erisc0_rt_args_prog2 = {
        (std::uint32_t)worker_core_physical.x,
        (std::uint32_t)worker_core_physical.y,
        0,             // risc index (ERISC0)
        mcast_enable,  // mcast_enable
        top_left_core_physical.x,
        top_left_core_physical.y,
        bottom_right_core_physical.x,
        bottom_right_core_physical.y,
        num_dests,
        program2_semaphore0,  // semaphore 0
        program2_semaphore1,  // semaphore 1
        l1_unreserved_base};  // safe L1 address

    std::vector<uint32_t> erisc1_rt_args_prog2 = {
        (std::uint32_t)worker_core_physical.x,
        (std::uint32_t)worker_core_physical.y,
        1,             // risc index (ERISC1)
        mcast_enable,  // mcast_enable
        top_left_core_physical.x,
        top_left_core_physical.y,
        bottom_right_core_physical.x,
        bottom_right_core_physical.y,
        num_dests,
        program2_semaphore0,  // semaphore 0
        program2_semaphore1,  // semaphore 1
        l1_unreserved_base};  // safe L1 address

    // Set runtime args for program1 ethernet kernels
    tt::tt_metal::SetRuntimeArgs(program1, eth_erisc0_kernel1, eth_core, erisc0_rt_args_prog1);
    tt::tt_metal::SetRuntimeArgs(program1, eth_erisc1_kernel1, eth_core, erisc1_rt_args_prog1);

    // Set runtime args for program2 ethernet kernels
    tt::tt_metal::SetRuntimeArgs(program2, eth_erisc0_kernel2, eth_core, erisc0_rt_args_prog2);
    tt::tt_metal::SetRuntimeArgs(program2, eth_erisc1_kernel2, eth_core, erisc1_rt_args_prog2);

    log_info(
        tt::LogTest, "Ethernet kernels configured on core {} (physical: {})", eth_core.str(), eth_core_physical.str());

    distributed::MeshWorkload workload1;
    distributed::MeshWorkload workload2;
    workload1.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program1));
    workload2.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program2));

    // This loop caches program1 and runs
    for (uint32_t i = 0; i < NUM_PROGRAMS; i++) {
        log_info(tt::LogTest, "Running ethernet program {} of {}", i + 1, NUM_PROGRAMS);
        if (i % 2 == 0) {
            distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload1, false);
        } else {
            distributed::EnqueueMeshWorkload(device->mesh_command_queue(), workload2, false);
        }
    }
    distributed::Finish(device->mesh_command_queue());
}

TEST_F(MeshDispatchFixture, TestDynamicNoCOneProgram) {
    uint32_t NUM_PROGRAMS = 1;
    uint32_t MAX_LOOP = 65536;
    uint32_t page_size = 1024;
    bool mix_noc_mode = false;

    build_and_run_program(this->devices_[0], this->slow_dispatch_, NUM_PROGRAMS, MAX_LOOP, page_size, mix_noc_mode);
}

TEST_F(MeshDispatchFixture, TestDynamicNoCMutlipleProgram) {
    uint32_t NUM_PROGRAMS = 3;
    uint32_t MAX_LOOP = 65536;
    uint32_t page_size = 1024;
    bool mix_noc_mode = false;

    build_and_run_program(this->devices_[0], this->slow_dispatch_, NUM_PROGRAMS, MAX_LOOP, page_size, mix_noc_mode);
}

TEST_F(MeshDispatchFixture, TestDynamicNoCMutlipleProgramMixedMode) {
    uint32_t NUM_PROGRAMS = 5;
    uint32_t MAX_LOOP = 65536;
    uint32_t page_size = 1024;
    bool mix_noc_mode = true;

    build_and_run_program(this->devices_[0], this->slow_dispatch_, NUM_PROGRAMS, MAX_LOOP, page_size, mix_noc_mode);
}

TEST_F(MeshDispatchFixture, TestDynamicNoCEthernetOneProgram) {
    uint32_t NUM_PROGRAMS = 1;
    uint32_t MAX_LOOP = 65536;
    uint32_t page_size = 1024;
    bool mix_noc_mode = false;

    build_and_run_program_ethernet(
        this->devices_[0], this->slow_dispatch_, NUM_PROGRAMS, MAX_LOOP, page_size, mix_noc_mode);
}

TEST_F(MeshDispatchFixture, TestDynamicNoCEthernetMultipleProgram) {
    uint32_t NUM_PROGRAMS = 3;
    uint32_t MAX_LOOP = 65536;
    uint32_t page_size = 1024;
    bool mix_noc_mode = false;

    build_and_run_program_ethernet(
        this->devices_[0], this->slow_dispatch_, NUM_PROGRAMS, MAX_LOOP, page_size, mix_noc_mode);
}

TEST_F(MeshDispatchFixture, TestDynamicNoCEthernetMultipleProgramMixedMode) {
    uint32_t NUM_PROGRAMS = 5;
    uint32_t MAX_LOOP = 65536;
    uint32_t page_size = 1024;
    bool mix_noc_mode = true;

    build_and_run_program_ethernet(
        this->devices_[0], this->slow_dispatch_, NUM_PROGRAMS, MAX_LOOP, page_size, mix_noc_mode);
}
}  // namespace tt::tt_metal
