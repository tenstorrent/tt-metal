// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdint.h>
#include <sys/types.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/util.hpp>
#include <map>
#include <set>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/semaphore.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "umd/device/tt_core_coordinates.h"

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

using std::vector;
using namespace tt;

namespace unit_tests::initialize_semaphores {

void initialize_program(tt_metal::IDevice* device, tt_metal::Program& program, const CoreRange& core_range) {
    uint32_t single_tile_size = tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    uint32_t num_tiles = 2048;

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 8;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core_range, cb_src0_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 1;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core_range, cb_output_config);

    auto unary_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles)  // per_core_tile_cnt
    };

    auto eltwise_unary_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
        core_range,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args});
}

void create_and_read_max_num_semaphores(
    tt_metal::IDevice* device, tt_metal::Program& program, const CoreRange& core_range) {
    std::vector<uint32_t> golden;
    for (uint32_t i = 0; i < tt::tt_metal::NUM_SEMAPHORES; i++) {
        uint32_t initial_value = i;
        auto semaphore_id = tt_metal::CreateSemaphore(program, core_range, initial_value);
        golden.push_back(initial_value);
        ASSERT_TRUE(semaphore_id == i);
    }

    tt_metal::detail::CompileProgram(device, program);
    program.finalize_offsets(device);

    ASSERT_TRUE(tt_metal::detail::ConfigureDeviceWithProgram(device, program));

    for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
            auto logical_core = CoreCoord{x, y};
            std::vector<uint32_t> res;
            for (uint32_t i = 0; i < tt::tt_metal::NUM_SEMAPHORES; i++) {
                std::vector<uint32_t> single_val;
                uint32_t semaphore_addr =
                    program.get_sem_base_addr(device, logical_core, CoreType::WORKER) +
                    (tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::L1) * i);
                uint32_t semaphore_size = sizeof(uint32_t);
                tt_metal::detail::ReadFromDeviceL1(device, logical_core, semaphore_addr, semaphore_size, single_val);
                ASSERT_TRUE(single_val.size() == 1);
                res.push_back(single_val.at(0));
            }
            ASSERT_TRUE(res == golden);
        }
    }
}

void try_creating_more_than_max_num_semaphores(
    tt_metal::IDevice* device, tt_metal::Program& program, const CoreRange& core_range) {
    ASSERT_TRUE(program.num_semaphores() == 0);
    create_and_read_max_num_semaphores(device, program, core_range);
    constexpr static uint32_t val = 5;
    ASSERT_ANY_THROW(tt_metal::CreateSemaphore(program, core_range, val));
}

}  // namespace unit_tests::initialize_semaphores

namespace tt::tt_metal {

TEST_F(DeviceFixture, TensixInitializeLegalSemaphores) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        tt_metal::Program program = tt_metal::CreateProgram();
        CoreRange core_range({0, 0}, {1, 1});
        unit_tests::initialize_semaphores::initialize_program(devices_.at(id), program, core_range);
        unit_tests::initialize_semaphores::create_and_read_max_num_semaphores(devices_.at(id), program, core_range);
    }
}

TEST_F(DeviceFixture, TensixInitializeIllegalSemaphores) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        tt_metal::Program program = tt_metal::CreateProgram();
        CoreRange core_range({0, 0}, {1, 1});
        unit_tests::initialize_semaphores::initialize_program(devices_.at(id), program, core_range);
        unit_tests::initialize_semaphores::try_creating_more_than_max_num_semaphores(
            devices_.at(id), program, core_range);
    }
}

TEST_F(DeviceFixture, TensixCreateMultipleSemaphoresOnSameCore) {
    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord core0(0, 0);
    uint32_t sem0_id = tt_metal::CreateSemaphore(program, core0, 0);

    CoreCoord core1(4, 0);
    uint32_t sem1_id = tt_metal::CreateSemaphore(program, core1, 1);

    CoreRange core_range({1, 0}, {3, 0});
    CoreRangeSet core_range_set({core_range});
    CoreRangeSet core_range_set2 = core_range_set.merge(std::set<CoreRange>{core1});
    std::set<CoreRange> set_of_cores({CoreRange({2, 0}, {2, 0}), CoreRange({3, 0}, {3, 0}), CoreRange({5, 0}, {5, 0})});
    CoreRangeSet core_range_set3(set_of_cores);
    CoreRangeSet core_range_set4({CoreRange({5, 0}, {6, 0})});

    uint32_t sem2_id = tt_metal::CreateSemaphore(program, core_range_set, 2);
    uint32_t sem3_id = tt_metal::CreateSemaphore(program, core_range_set2, 3);
    uint32_t sem4_id = tt_metal::CreateSemaphore(program, core_range_set2, 4);
    uint32_t sem5_id = tt_metal::CreateSemaphore(program, core_range_set3, 5);
    uint32_t sem6_id = tt_metal::CreateSemaphore(program, core_range_set4, 6);

    EXPECT_EQ(sem0_id, 0);
    EXPECT_EQ(sem1_id, 0);
    EXPECT_EQ(sem2_id, 0);
    EXPECT_EQ(sem3_id, 1);
    EXPECT_EQ(sem4_id, 2);
    EXPECT_EQ(sem5_id, 3);
    EXPECT_EQ(sem6_id, 0);
}

}  // namespace tt::tt_metal
