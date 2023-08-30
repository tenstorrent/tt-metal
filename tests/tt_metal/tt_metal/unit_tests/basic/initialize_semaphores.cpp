#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "single_device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this

using namespace tt;

namespace unit_tests::initialize_semaphores {

void initialize_and_compile_program(tt_metal::Device *device, tt_metal::Program &program, const CoreRange &core_range) {
    uint32_t single_tile_size = tt_metal::detail::TileSize(tt::DataFormat::Float16_b);
    uint32_t num_tiles = 2048;

    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = 8;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        core_range,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        tt::DataFormat::Float16_b,
        src0_cb_addr);

    uint32_t ouput_cb_index = 16;  // output operands start at index 16
    uint32_t output_cb_addr = 300 * 1024;
    uint32_t num_output_tiles = 1;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        ouput_cb_index,
        core_range,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        tt::DataFormat::Float16_b,
        output_cb_addr);

    auto unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_push_4.cpp",
        core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles)  // per_core_tile_cnt
    };

    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_copy_3m.cpp",
        core_range,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

    tt_metal::CompileProgram(device, program);
}

void create_and_read_max_num_semaphores(
    tt_metal::Device *device, tt_metal::Program &program, const CoreRange &core_range) {
    std::vector<uint32_t> golden;
    for (uint32_t i = 0; i < NUM_SEMAPHORES; i++) {
        uint32_t initial_value = i;
        auto semaphore_addr = tt_metal::CreateSemaphore(program, core_range, initial_value);
        golden.push_back(initial_value);
        ASSERT_TRUE(semaphore_addr == SEMAPHORE_BASE + (ALIGNED_SIZE_PER_SEMAPHORE * i));
    }

    ASSERT_TRUE(tt_metal::ConfigureDeviceWithProgram(device, program));

    for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
        for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
            auto logical_core = CoreCoord{x, y};
            std::vector<uint32_t> res;
            for (uint32_t i = 0; i < NUM_SEMAPHORES; i++) {
                std::vector<uint32_t> single_val;
                uint32_t semaphore_addr = SEMAPHORE_BASE + (ALIGNED_SIZE_PER_SEMAPHORE * i);
                uint32_t semaphore_size = UINT32_WORDS_PER_SEMAPHORE * sizeof(uint32_t);
                tt_metal::detail::ReadFromDeviceL1(device, logical_core, semaphore_addr, semaphore_size, single_val);
                ASSERT_TRUE(single_val.size() == 1);
                res.push_back(single_val.at(0));
            }
            ASSERT_TRUE(res == golden);
        }
    }
}

void try_creating_more_than_max_num_semaphores(
    tt_metal::Device *device, tt_metal::Program &program, const CoreRange &core_range) {
    ASSERT_TRUE(program.num_semaphores() == 0);
    create_and_read_max_num_semaphores(device, program, core_range);
    constexpr static uint32_t val = 5;
    ASSERT_ANY_THROW(tt_metal::CreateSemaphore(program, core_range, val));
}

}  // namespace unit_tests::initialize_semaphores

TEST_F(SingleDeviceFixture, InitializeLegalSemaphores) {
    tt_metal::Program program = tt_metal::Program();
    CoreRange core_range = {.start = {0, 0}, .end = {1, 1}};
    unit_tests::initialize_semaphores::initialize_and_compile_program(device_, program, core_range);
    unit_tests::initialize_semaphores::create_and_read_max_num_semaphores(device_, program, core_range);
}

TEST_F(SingleDeviceFixture, InitializeIllegalSemaphores) {
    tt_metal::Program program = tt_metal::Program();
    CoreRange core_range = {.start = {0, 0}, .end = {1, 1}};
    unit_tests::initialize_semaphores::initialize_and_compile_program(device_, program, core_range);
    unit_tests::initialize_semaphores::try_creating_more_than_max_num_semaphores(device_, program, core_range);
}
