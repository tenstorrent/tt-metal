#include "doctest.h"
#include "multi_device_fixture.hpp"
#include "single_device_fixture.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "common/bfloat16.hpp"


using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::create_pipeline {


void create_and_run_row_pipeline(tt_metal::Device* device, u32 num_cores) {
    CommandQueue cq(device);

    tt_metal::Program program = tt_metal::Program();

    u32 num_tiles = 32;
    u32 block_size_tiles = 16;
    u32 num_blocks_in_CB = 2;
    u32 num_repetitions = 1;

    TT_ASSERT(num_cores >= 2 && num_cores <= 12); // grayskull
    TT_ASSERT(num_tiles % block_size_tiles == 0);

    std::vector<CoreCoord> cores;
    for (u32 i = 0; i < num_cores; i++) {
        cores.push_back({i, 0});
    }

    log_info(LogTest, "num_cores: {}", num_cores);
    log_info(LogTest, "num_tiles: {}", num_tiles);
    log_info(LogTest, "block_size_tiles: {}", block_size_tiles);
    log_info(LogTest, "num_blocks_in_CB: {}", num_blocks_in_CB);
    log_info(LogTest, "num_repetitions: {}", num_repetitions);

    u32 single_tile_size = 2 * 1024;
    u32 block_size_bytes = block_size_tiles * single_tile_size;
    log_info(LogTest, "block_size_bytes: {}", block_size_bytes);
    log_info(LogTest, "CB size: {}", block_size_bytes * num_blocks_in_CB);

    // source and destination buffers
    u32 buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    u32 total_bytes_moved = buffer_size * num_repetitions;
    log_info(LogTest, "total_bytes_moved: {}", total_bytes_moved);

    // circular buffers in L1
    u32 cb_index = 8;
    u32 cb_size_tiles = num_blocks_in_CB * block_size_tiles;
    u32 cb_size_bytes = cb_size_tiles * single_tile_size;

    for (auto core : cores) {
        auto cb = tt_metal::CreateCircularBuffer(
            program,
            device,
            cb_index,
            core,
            cb_size_tiles,
            cb_size_bytes,
            tt::DataFormat::Float16_b
        );
    }

    /// used only if IO data in DRAM
    tt_metal::Buffer src_buffer;
    tt_metal::Buffer dst_buffer;

    u32 src_address;
    CoreCoord src_noc_xy;
    u32 dst_address;
    CoreCoord dst_noc_xy;

    u32 dram_buffer_addr = 0;
    TT_ASSERT(dram_buffer_addr + buffer_size <= device->dram_bank_size());

    src_buffer = tt_metal::Buffer(device, buffer_size, dram_buffer_addr, 0, buffer_size, tt_metal::BufferType::DRAM);
    dst_buffer = tt_metal::Buffer(device, buffer_size, dram_buffer_addr, 7, buffer_size, tt_metal::BufferType::DRAM);

    src_address = src_buffer.address();
    src_noc_xy = src_buffer.noc_coordinates();
    dst_address = dst_buffer.address();
    dst_noc_xy = dst_buffer.noc_coordinates();

    // create kernels
    vector<tt_metal::DataMovementKernel*> receiver_kernels;
    vector<tt_metal::DataMovementKernel*> sender_kernels;
    for (int core_id = 0; core_id < num_cores; core_id++) {
        string receiver_kernel_name;
        if (core_id == 0) {
            receiver_kernel_name = "tt_metal/kernels/dataflow/reader_first_stage.cpp";
        } else {
            receiver_kernel_name = "tt_metal/kernels/dataflow/receiver_intermediate_stage.cpp";
        }

        std::vector<u32> receiver_kernel_compile_time_args = {cb_index, block_size_tiles};
        receiver_kernels.push_back(tt_metal::CreateDataMovementKernel(
            program,
            receiver_kernel_name,
            cores[core_id],
            receiver_kernel_compile_time_args,
            tt_metal::DataMovementProcessor::RISCV_1,
            tt_metal::NOC::RISCV_1_default));

        string sender_kernel_name;
        if (core_id == num_cores - 1) {
            sender_kernel_name = "tt_metal/kernels/dataflow/writer_last_stage.cpp";
        } else {
            sender_kernel_name = "tt_metal/kernels/dataflow/sender_intermediate_stage.cpp";
        }
        std::vector<u32> sender_kernel_compile_time_args = {cb_index, block_size_tiles};
        sender_kernels.push_back(tt_metal::CreateDataMovementKernel(
            program,
            sender_kernel_name,
            cores[core_id],
            sender_kernel_compile_time_args,
            tt_metal::DataMovementProcessor::RISCV_0,
            tt_metal::NOC::RISCV_0_default));

        sender_kernels.at(sender_kernels.size() - 1)->add_define("DEVICE_DISPATCH_MODE", "1");

        // Add blank compute kernel
        tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/blank.cpp",
            cores[core_id],
            {},
            MathFidelity::HiFi4,
            false,
            false
        );
    }

    // TODO(agrebenisan): Once semaphores are properly allocated at 16B-aligned addresses, then
    // will make proper sems. For now, using the original code.
    map<CoreCoord, vector<Semaphore*>> sems;
    for (auto core : cores) {
        CoreRange cr = {.start = core, .end = core};

        auto sender_semaphore = tt_metal::CreateSemaphore(program, device, cr, INVALID);
        auto receiver_semaphore = tt_metal::CreateSemaphore(program, device, cr, INVALID);
        auto l1_valid_value_semaphore = tt_metal::CreateSemaphore(program, device, cr, VALID);

        tt::log_debug("SENDER SEM ADDR {}", sender_semaphore->address());
        tt::log_debug("RECEIVER SEM ADDR {}", receiver_semaphore->address());
        tt::log_debug("L1 VALID VALUE SEM ADDR {}", l1_valid_value_semaphore->address());

        vector<Semaphore*> init_vec;
        sems.emplace(core, init_vec);
        sems.at(core).push_back(sender_semaphore);
        sems.at(core).push_back(receiver_semaphore);
        sems.at(core).push_back(l1_valid_value_semaphore);
    }

    // Store runtime args
    RuntimeArgs rt_args;
    for (int core_id = 0; core_id < num_cores; core_id++) {

        // TODO(agrebenisan):  Once semaphores are properly allocated at 16B-aligned addresses, then
        // will make proper sems. For now, using the original code.
        CoreCoord core = cores[core_id];
        auto sender_semaphore_addr = sems[core].at(0)->address();
        auto receiver_semaphore_addr = sems[core].at(1)->address();
        auto l1_valid_value_addr = sems[core].at(2)->address();

        map<RISCV, vector<u32>> worker_core_rt_args;
        if (core_id == 0) {
            worker_core_rt_args[RISCV::NCRISC] = {src_address,
                (u32)src_noc_xy.x,
                (u32)src_noc_xy.y,
                (u32)num_tiles,
                (u32)num_repetitions};
        } else {
            worker_core_rt_args[RISCV::NCRISC] = {(u32)device->worker_core_from_logical_core(cores[core_id-1]).x,
                (u32)device->worker_core_from_logical_core(cores[core_id-1]).y,
                (u32)num_tiles,
                (u32)sender_semaphore_addr,
                (u32)receiver_semaphore_addr,
                (u32)num_repetitions};
        }

        if (core_id == num_cores - 1) {
            worker_core_rt_args[RISCV::BRISC] = {dst_address,
                (u32)dst_noc_xy.x,
                (u32)dst_noc_xy.y,
                (u32)num_tiles,
                (u32)num_repetitions};
        } else {
                worker_core_rt_args[RISCV::BRISC] = {(u32)device->worker_core_from_logical_core(cores[core_id+1]).x,
                (u32)device->worker_core_from_logical_core(cores[core_id+1]).y,
                (u32)num_tiles,
                (u32)sender_semaphore_addr,
                (u32)receiver_semaphore_addr,
                (u32)l1_valid_value_addr,
                (u32)num_repetitions};
        }

        rt_args[cores[core_id]] = worker_core_rt_args;
    }

    constexpr bool profile_device = false;
    tt_metal::CompileProgram(device, program, profile_device);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    // send input data to the device
    std::vector<u32> src_vec = create_random_vector_of_bfloat16(
        buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    EnqueueWriteBuffer(cq, src_buffer, src_vec, false);

    EnqueueProgram(cq, program, rt_args, false);
    Finish(cq);

    if (profile_device){
        tt_metal::DumpDeviceProfileResults(device, program);
    }
    log_info(LogTest, "Kernels done.");

    log_info(LogTest, "Reading results from device...");
    std::vector<u32> result_vec;
    EnqueueReadBuffer(cq, dst_buffer, result_vec, true);

    ////////////////////////////////////////////////////////////////////////////
    //                      Validation & Teardown
    ////////////////////////////////////////////////////////////////////////////
    REQUIRE(src_vec == result_vec);
}

} // namespace unit_tests::create_pipeline

TEST_SUITE(
    "Pipelining tests" *
    doctest::description("Pipelining unit tests") *
    doctest::timeout(5)
) {
    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "Initialize device") {
        SUBCASE("Create a pipeline of size 2 along row") {
            auto arch = this->arch_;
            if (arch != tt::ARCH::GRAYSKULL) {
                return; // Figure out how to properly skip
            }

            unit_tests::create_pipeline::create_and_run_row_pipeline(this->device_, 2);
        }
        SUBCASE("Create a pipeline of size 12 along row") {
            auto arch = this->arch_;
            if (arch != tt::ARCH::GRAYSKULL) {
                return; // Figure out how to properly skip
            }

            unit_tests::create_pipeline::create_and_run_row_pipeline(this->device_, 12);
        }
    }
}
