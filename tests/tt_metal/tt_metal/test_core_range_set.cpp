// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <chrono>
#include <cstdint>
#include <array>
#include <map>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include "impl/buffers/semaphore.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/core_coordinates.hpp>

// Access to internal API: ProgramImpl::get_sem_base_addr, get_sem_size, num_kernels, get_kernel
#include "impl/program/program_impl.hpp"
#include "impl/buffers/circular_buffer.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/context/metal_context.hpp"

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace {

void check_program_is_mapped_to_correct_cores(
    const Program& program, const CoreRangeSet& core_range_set, const std::vector<uint32_t>& compute_kernel_args) {
    for (auto core_range : core_range_set.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                auto logical_core = CoreCoord{x, y};
                for (size_t kernel_id = 0; kernel_id < program.impl().num_kernels(); kernel_id++) {
                    auto kernel = program.impl().get_kernel(kernel_id);
                    TT_FATAL(kernel->is_on_logical_core(logical_core), "Error");
                    if (kernel->get_kernel_processor_class() == HalProcessorClassType::COMPUTE) {
                        auto kernel_compile_time_args = kernel->compile_time_args();
                        TT_FATAL(kernel_compile_time_args == compute_kernel_args, "Error");
                    }
                }
                for (const auto& cb : program.impl().circular_buffers()) {
                    TT_FATAL(cb->is_on_logical_core(logical_core), "Error");
                }
                for (const auto& semaphore : program.impl().semaphores()) {
                    TT_FATAL(semaphore.initialized_on_logical_core(logical_core), "Error");
                }
            }
        }
    }
}

void check_semaphores_are_initialized(
    IDevice* device,
    Program& program,
    const CoreRangeSet& core_range_set,
    const std::vector<uint32_t>& golden_sem_values) {
    for (auto core_range : core_range_set.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                auto logical_core = CoreCoord{x, y};
                std::vector<uint32_t> res;
                auto sem_base_addr = program.impl().get_sem_base_addr(device, logical_core, CoreType::WORKER);
                detail::ReadFromDeviceL1(
                    device,
                    logical_core,
                    sem_base_addr,
                    program.impl().get_sem_size(device, logical_core, CoreType::WORKER),
                    res);
                std::vector<uint32_t> filtered_res;
                static uint32_t num_u32_to_skip =
                    MetalContext::instance().hal().get_alignment(HalMemType::L1) / sizeof(uint32_t);
                for (size_t i = 0; i < res.size(); i += num_u32_to_skip) {
                    filtered_res.push_back(res.at(i));
                }

                TT_FATAL(filtered_res == golden_sem_values, "Error");
            }
        }
    }
}

}  // namespace

TEST_F(MeshDeviceSingleCardFixture, CoreRangeSet) {
    IDevice* dev = devices_[0]->get_devices()[0];
    Program program = CreateProgram();

    CoreRange core_range_one({0, 0}, {1, 1});
    CoreRange core_range_two({2, 2}, {3, 3});
    CoreRangeSet core_ranges = CoreRangeSet(std::vector{core_range_one, core_range_two});

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 4;
    uint32_t buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = buffer_size, .page_size = buffer_size, .buffer_type = BufferType::DRAM};

    auto src_dram_buffer = CreateBuffer(dram_config);

    std::map<CoreCoord, std::shared_ptr<Buffer>> core_to_l1_buffer;
    for (auto core_range : core_ranges.ranges()) {
        auto start = core_range.start_coord;
        auto end = core_range.end_coord;
        for (auto x = start.x; x <= end.x; x++) {
            for (auto y = start.y; y <= end.y; y++) {
                CoreCoord logical_core(x, y);
                InterleavedBufferConfig l1_config{
                    .device = dev, .size = buffer_size, .page_size = buffer_size, .buffer_type = BufferType::L1};
                auto dst_l1_buffer = CreateBuffer(l1_config);
                core_to_l1_buffer.emplace(logical_core, dst_l1_buffer);
            }
        }
    }

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 8;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core_ranges, cb_src0_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    CreateCircularBuffer(program, core_ranges, cb_output_config);

    auto unary_reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        core_ranges,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto unary_writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_1.cpp",
        core_ranges,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {uint(num_tiles)};

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
        core_ranges,
        ComputeConfig{.compile_args = compute_kernel_args});

    std::vector<uint32_t> golden_sem_values;
    for (uint32_t i = 0; i < NUM_SEMAPHORES; i++) {
        uint32_t initial_value = i;
        CreateSemaphore(program, core_ranges, initial_value);
        golden_sem_values.push_back(initial_value);
    }

    check_program_is_mapped_to_correct_cores(program, core_ranges, compute_kernel_args);

    detail::CompileProgram(dev, program);

    std::vector<uint32_t> src_vec =
        create_random_vector_of_bfloat16(buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    detail::WriteToBuffer(src_dram_buffer, src_vec);

    const std::array reader_rt_args = {src_dram_buffer->address(), uint(0), num_tiles};
    for (const auto& [core, dst_l1_buffer] : core_to_l1_buffer) {
        SetRuntimeArgs(program, unary_reader_kernel, core, reader_rt_args);

        auto l1_dst_noc_xy = dev->virtual_core_from_logical_core(
            dst_l1_buffer->allocator()->get_logical_core_from_bank_id(0), CoreType::WORKER);

        SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dst_l1_buffer->address(), (std::uint32_t)l1_dst_noc_xy.x, (std::uint32_t)l1_dst_noc_xy.y, num_tiles});
    }

    detail::LaunchProgram(dev, program);

    check_semaphores_are_initialized(dev, program, core_ranges, golden_sem_values);

    for (const auto& [core, dst_l1_buffer] : core_to_l1_buffer) {
        std::vector<uint32_t> result_vec;
        detail::ReadFromBuffer(dst_l1_buffer, result_vec);
        EXPECT_EQ(src_vec, result_vec) << "Mismatch on core " << core.x << "," << core.y;
    }
}
