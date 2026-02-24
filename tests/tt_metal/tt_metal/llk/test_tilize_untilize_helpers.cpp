// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "device_fixture.hpp"
#include "test_golden_impls.hpp"
#include "impl/data_format/bfloat16_utils.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::compute::helpers {

// Inline writer kernel: just drains a CB (wait_front + pop_front).
// Args: {num_pages, cb_id, pages_per_pop}
static const std::string writer_drain_kernel_src = R"(
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    uint32_t num_pages     = get_arg_val<uint32_t>(0);
    uint32_t cb_id         = get_arg_val<uint32_t>(1);
    uint32_t pages_per_pop = get_arg_val<uint32_t>(2);

    experimental::CircularBuffer cb(cb_id);

    for (uint32_t i = 0; i < num_pages; i += pages_per_pop) {
        uint32_t curr_pop = (num_pages - i < pages_per_pop) ? (num_pages - i) : pages_per_pop;
        cb.wait_front(curr_pop);
        cb.pop_front(curr_pop);
    }
}
)";

// ── Tilize helper test runner ───────────────────────────────────────────────
//
// Exercises compute_kernel_lib::tilize<> with real CBs and L1 output buffer.
//
// block_width  — tiles per block (width dimension, passed as compile arg 0)
// num_blocks   — number of tile-rows to process (height dimension, passed as compile arg 1)
// rows = 0: symmetric mode — input CB has tile-sized pages
// rows > 0: asymmetric mode — input CB has row-sized pages, rows = total row pages
void run_tilize_helper_test(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t num_blocks,
    uint32_t block_width,
    const std::map<std::string, std::string>& extra_defines = {},
    uint32_t rows = 0) {
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = tt::tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];

    CoreCoord core = {0, 0};

    constexpr uint32_t single_tile_size = 2 * 1024;  // bfloat16
    constexpr uint32_t tile_width = 32;
    constexpr uint32_t tile_height = 32;
    uint32_t total_tiles = num_blocks * block_width;
    uint32_t dram_buffer_size = single_tile_size * total_tiles;

    bool asymmetric = (rows > 0);
    uint32_t row_page_size = tile_width * block_width * 2;
    uint32_t input_page_size = asymmetric ? row_page_size : single_tile_size;
    uint32_t input_num_pages = asymmetric ? (tile_height * num_blocks) : total_tiles;
    uint32_t input_pages_per_push = asymmetric ? tile_height : block_width;

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig l1_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::L1};

    auto src_dram_buffer = CreateBuffer(dram_config);
    auto input_l1_buffer = CreateBuffer(l1_config);
    auto output_l1_buffer = CreateBuffer(l1_config);

    // CB 0 — input (row-major), page size depends on mode, backed by L1 buffer
    uint32_t input_cb = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig cb_input_config =
        tt_metal::CircularBufferConfig(dram_buffer_size, {{input_cb, tt::DataFormat::Float16_b}})
            .set_page_size(input_cb, input_page_size)
            .set_globally_allocated_address(*input_l1_buffer);
    tt_metal::CreateCircularBuffer(program_, core, cb_input_config);

    // CB 16 — output (tiled, always tile-sized pages), backed by L1 buffer
    uint32_t output_cb = tt::CBIndex::c_16;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(dram_buffer_size, {{output_cb, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb, single_tile_size)
            .set_globally_allocated_address(*output_l1_buffer);
    tt_metal::CreateCircularBuffer(program_, core, cb_output_config);

    // Reader: pushes input_pages_per_push pages at a time
    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_push_pages.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    // Writer: drains output CB (wait_front + pop_front)
    auto writer_kernel = tt_metal::CreateKernelFromString(
        program_,
        writer_drain_kernel_src,
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    // Compute: tilize_helper.cpp
    // compile arg 0 = block_width, compile arg 1 = num_blocks, 2 = input_cb, 3 = output_cb
    auto compute_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/tilize_helper.cpp",
        core,
        tt_metal::ComputeConfig{
            .compile_args = {block_width, num_blocks, input_cb, output_cb}, .defines = extra_defines});

    // Write input data — zero-pad to full tiles so padding rows are zero in DRAM
    uint32_t src_data_size = asymmetric ? rows * row_page_size : dram_buffer_size;
    std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(src_data_size, false);
    src_vec.resize(dram_buffer_size / sizeof(uint32_t), 0);
    tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);

    // Reader args: {src_addr, bank_id, num_pages, cb_id, pages_per_push, page_size}
    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {src_dram_buffer->address(), (uint32_t)0, input_num_pages, input_cb, input_pages_per_push, input_page_size});

    // Writer args: {num_pages, cb_id, pages_per_pop}
    tt_metal::SetRuntimeArgs(program_, writer_kernel, core, {total_tiles, output_cb, block_width});

    // Compute runtime arg 0: rows (0 = symmetric, >0 = asymmetric total_input_pages)
    tt_metal::SetRuntimeArgs(program_, compute_kernel, core, {rows});

    // Execute
    distributed::EnqueueMeshWorkload(cq, workload, false);

    // Read results from output CB's L1 backing buffer on core (0,0)
    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromDeviceL1(device, core, output_l1_buffer->address(), dram_buffer_size, result_vec);

    ::unit_tests::compute::GoldenConfig config = {
        .num_tiles_r_dim = (int)num_blocks,
        .num_tiles_c_dim = (int)block_width,
    };
    vector<uint32_t> golden = ::unit_tests::compute::gold_standard_tilize(src_vec, config);

    bool pass = (golden == result_vec);
    if (!pass) {
        std::cout << "GOLDEN " << std::endl;
        print_vector(unpack_vector<bfloat16, uint32_t>(golden));
        std::cout << "RESULTS " << std::endl;
        print_vector(unpack_vector<bfloat16, uint32_t>(result_vec));
    }
    log_info(
        tt::LogTest,
        "KernelLib tilize: num_blocks={}, block_width={}, rows={}, pass={}",
        num_blocks,
        block_width,
        rows,
        pass);
    ASSERT_TRUE(pass);
}

// ── Untilize helper test runner ─────────────────────────────────────────────
//
// Exercises compute_kernel_lib::untilize<> with real CBs and L1 output buffer.
//
// block_width  — tiles per block (width dimension, set via UNTILIZE_BLOCK_WIDTH define)
// num_blocks   — number of tile-rows to process (height dimension, passed as compile arg 0)
// rows = 0: symmetric mode — output CB has tile-sized pages
// rows > 0: asymmetric mode — output CB has row-sized pages, rows = total row pages
void run_untilize_helper_test(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t num_blocks,
    uint32_t block_width,
    const std::map<std::string, std::string>& extra_defines = {},
    uint32_t rows = 0) {
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = tt::tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];

    CoreCoord core = {0, 0};

    constexpr uint32_t single_tile_size = 2 * 1024;  // bfloat16
    constexpr uint32_t tile_width = 32;
    constexpr uint32_t tile_height = 32;
    uint32_t total_tiles = num_blocks * block_width;
    uint32_t dram_buffer_size = single_tile_size * total_tiles;

    bool asymmetric = (rows > 0);
    uint32_t row_page_size = tile_width * block_width * 2;
    uint32_t output_page_size = asymmetric ? row_page_size : single_tile_size;
    uint32_t output_num_pages = asymmetric ? rows : total_tiles;
    uint32_t output_pages_per_pop = asymmetric ? tile_height : block_width;

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig l1_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::L1};

    auto src_dram_buffer = CreateBuffer(dram_config);
    auto input_l1_buffer = CreateBuffer(l1_config);
    auto output_l1_buffer = CreateBuffer(l1_config);

    // CB 0 — input (tiled, always tile-sized pages), backed by L1 buffer
    uint32_t input_cb = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig cb_input_config =
        tt_metal::CircularBufferConfig(dram_buffer_size, {{input_cb, tt::DataFormat::Float16_b}})
            .set_page_size(input_cb, single_tile_size)
            .set_globally_allocated_address(*input_l1_buffer);
    tt_metal::CreateCircularBuffer(program_, core, cb_input_config);

    // CB 16 — output (row-major), page size depends on mode, backed by L1 buffer
    uint32_t output_cb = tt::CBIndex::c_16;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(dram_buffer_size, {{output_cb, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb, output_page_size)
            .set_globally_allocated_address(*output_l1_buffer);
    tt_metal::CreateCircularBuffer(program_, core, cb_output_config);

    // Reader: pushes block_width tile-pages at a time into input CB
    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_push_pages.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    // Writer: drains output CB (wait_front + pop_front)
    auto writer_kernel = tt_metal::CreateKernelFromString(
        program_,
        writer_drain_kernel_src,
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    // Compute: untilize_helper.cpp
    // compile arg 0 = block_width, compile arg 1 = num_blocks, 2 = input_cb, 3 = output_cb
    auto compute_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/untilize_helper.cpp",
        core,
        tt_metal::ComputeConfig{
            .compile_args = {block_width, num_blocks, input_cb, output_cb}, .defines = extra_defines});

    // Write input data
    std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(dram_buffer_size, false);
    tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);

    // Reader args: {src_addr, bank_id, num_pages, cb_id, pages_per_push, page_size}
    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {src_dram_buffer->address(), (uint32_t)0, total_tiles, input_cb, block_width, single_tile_size});

    // Writer args: {num_pages, cb_id, pages_per_pop}
    tt_metal::SetRuntimeArgs(program_, writer_kernel, core, {output_num_pages, output_cb, output_pages_per_pop});

    // Compute runtime arg 0: rows (0 = symmetric, >0 = asymmetric total_output_pages)
    tt_metal::SetRuntimeArgs(program_, compute_kernel, core, {rows});

    // Execute
    distributed::EnqueueMeshWorkload(cq, workload, false);

    // Read results from output CB's L1 backing buffer on core (0,0)
    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromDeviceL1(device, core, output_l1_buffer->address(), dram_buffer_size, result_vec);

    ::unit_tests::compute::GoldenConfig config = {
        .num_tiles_r_dim = (int)num_blocks,
        .num_tiles_c_dim = (int)block_width,
    };
    vector<uint32_t> golden = ::unit_tests::compute::gold_standard_untilize(src_vec, config);

    // For partial rows, only compare valid output rows (row-major, contiguous)
    uint32_t compare_size = asymmetric ? (rows * row_page_size / sizeof(uint32_t)) : golden.size();
    bool pass = std::equal(golden.begin(), golden.begin() + compare_size, result_vec.begin());
    if (!pass) {
        std::cout << "GOLDEN (first " << compare_size << " u32)" << std::endl;
        auto g = unpack_vector<bfloat16, uint32_t>(golden);
        g.resize(compare_size * 2);
        print_vector(g);
        std::cout << "RESULTS (first " << compare_size << " u32)" << std::endl;
        auto r = unpack_vector<bfloat16, uint32_t>(result_vec);
        r.resize(compare_size * 2);
        print_vector(r);
    }
    log_info(
        tt::LogTest,
        "KernelLib untilize: num_blocks={}, block_width={}, rows={}, pass={}",
        num_blocks,
        block_width,
        rows,
        pass);
    ASSERT_TRUE(pass);
}

}  // namespace unit_tests::compute::helpers

// Shape parameter: {num_blocks, block_width}
using ShapeParam = std::tuple<uint32_t, uint32_t>;

static std::string ShapeName(const testing::TestParamInfo<ShapeParam>& info) {
    return std::to_string(std::get<0>(info.param)) + "x" + std::to_string(std::get<1>(info.param));
}

static const auto kStandardShapes = testing::Values(
    ShapeParam{1, 1}, ShapeParam{1, 2}, ShapeParam{2, 1}, ShapeParam{1, 4}, ShapeParam{2, 2}, ShapeParam{4, 1});

// ═══════════════════════════════════════════════════════════════════════════
// Tilize helper tests
// ═══════════════════════════════════════════════════════════════════════════

class TilizeBasicTest : public MeshDeviceFixture, public testing::WithParamInterface<ShapeParam> {};
TEST_P(TilizeBasicTest, Run) {
    auto [num_blocks, block_width] = GetParam();
    unit_tests::compute::helpers::run_tilize_helper_test(this->devices_.at(0), num_blocks, block_width);
}
INSTANTIATE_TEST_SUITE_P(TensixKernelLib, TilizeBasicTest, kStandardShapes, ShapeName);

class TilizeWaitUpfrontTest : public MeshDeviceFixture, public testing::WithParamInterface<ShapeParam> {};
TEST_P(TilizeWaitUpfrontTest, Run) {
    auto [num_blocks, block_width] = GetParam();
    unit_tests::compute::helpers::run_tilize_helper_test(
        this->devices_.at(0), num_blocks, block_width, {{"TILIZE_WAIT_MODE", "WaitMode::WaitUpfront"}});
}
INSTANTIATE_TEST_SUITE_P(TensixKernelLib, TilizeWaitUpfrontTest, kStandardShapes, ShapeName);

class TilizeNoReconfigureTest : public MeshDeviceFixture, public testing::WithParamInterface<ShapeParam> {};
TEST_P(TilizeNoReconfigureTest, Run) {
    auto [num_blocks, block_width] = GetParam();
    unit_tests::compute::helpers::run_tilize_helper_test(
        this->devices_.at(0),
        num_blocks,
        block_width,
        {{"TILIZE_RECONFIG_MODE", "ReconfigureRegisterDatatypeMode::NoReconfigure"}});
}
INSTANTIATE_TEST_SUITE_P(TensixKernelLib, TilizeNoReconfigureTest, kStandardShapes, ShapeName);

class TilizeRowsTest : public MeshDeviceFixture, public testing::WithParamInterface<ShapeParam> {};
TEST_P(TilizeRowsTest, Run) {
    auto [num_blocks, block_width] = GetParam();
    constexpr uint32_t tile_height = 32;
    uint32_t rows = tile_height * num_blocks;
    unit_tests::compute::helpers::run_tilize_helper_test(this->devices_.at(0), num_blocks, block_width, {}, rows);
}
INSTANTIATE_TEST_SUITE_P(TensixKernelLib, TilizeRowsTest, kStandardShapes, ShapeName);

// Partial rows — rows not divisible by tile_height (32)
using PartialRowsParam = std::tuple<uint32_t, uint32_t>;  // {rows, block_width}

static std::string PartialRowsName(const testing::TestParamInfo<PartialRowsParam>& info) {
    return "r" + std::to_string(std::get<0>(info.param)) + "_w" + std::to_string(std::get<1>(info.param));
}

class TilizePartialRowsTest : public MeshDeviceFixture, public testing::WithParamInterface<PartialRowsParam> {};
TEST_P(TilizePartialRowsTest, Run) {
    auto [rows, block_width] = GetParam();
    constexpr uint32_t tile_height = 32;
    uint32_t num_blocks = (rows + tile_height - 1) / tile_height;
    unit_tests::compute::helpers::run_tilize_helper_test(this->devices_.at(0), num_blocks, block_width, {}, rows);
}
INSTANTIATE_TEST_SUITE_P(
    TensixKernelLib,
    TilizePartialRowsTest,
    testing::Values(
        PartialRowsParam{20, 1},
        PartialRowsParam{20, 2},
        PartialRowsParam{48, 1},
        PartialRowsParam{48, 2},
        PartialRowsParam{1, 1},
        PartialRowsParam{1, 4}),
    PartialRowsName);

// ═══════════════════════════════════════════════════════════════════════════
// Untilize helper tests
// ═══════════════════════════════════════════════════════════════════════════

class UntilizeBasicTest : public MeshDeviceFixture, public testing::WithParamInterface<ShapeParam> {};
TEST_P(UntilizeBasicTest, Run) {
    auto [num_blocks, block_width] = GetParam();
    unit_tests::compute::helpers::run_untilize_helper_test(this->devices_.at(0), num_blocks, block_width);
}
INSTANTIATE_TEST_SUITE_P(
    TensixKernelLib,
    UntilizeBasicTest,
    testing::Values(
        ShapeParam{1, 1},
        ShapeParam{1, 2},
        ShapeParam{2, 1},
        ShapeParam{1, 4},
        ShapeParam{2, 2},
        ShapeParam{4, 1},
        ShapeParam{1, 10},
        ShapeParam{2, 16},
        ShapeParam{1, 40}),
    ShapeName);

class UntilizeWaitUpfrontTest : public MeshDeviceFixture, public testing::WithParamInterface<ShapeParam> {};
TEST_P(UntilizeWaitUpfrontTest, Run) {
    auto [num_blocks, block_width] = GetParam();
    unit_tests::compute::helpers::run_untilize_helper_test(
        this->devices_.at(0), num_blocks, block_width, {{"UNTILIZE_WAIT_MODE", "WaitMode::WaitUpfront"}});
}
INSTANTIATE_TEST_SUITE_P(TensixKernelLib, UntilizeWaitUpfrontTest, kStandardShapes, ShapeName);

class UntilizeNoReconfigureTest : public MeshDeviceFixture, public testing::WithParamInterface<ShapeParam> {};
TEST_P(UntilizeNoReconfigureTest, Run) {
    auto [num_blocks, block_width] = GetParam();
    unit_tests::compute::helpers::run_untilize_helper_test(
        this->devices_.at(0),
        num_blocks,
        block_width,
        {{"UNTILIZE_RECONFIG_MODE", "ReconfigureRegisterDatatypeMode::NoReconfigure"}});
}
INSTANTIATE_TEST_SUITE_P(TensixKernelLib, UntilizeNoReconfigureTest, kStandardShapes, ShapeName);

class UntilizeRowsTest : public MeshDeviceFixture, public testing::WithParamInterface<ShapeParam> {};
TEST_P(UntilizeRowsTest, Run) {
    auto [num_blocks, block_width] = GetParam();
    constexpr uint32_t tile_height = 32;
    uint32_t rows = tile_height * num_blocks;
    unit_tests::compute::helpers::run_untilize_helper_test(this->devices_.at(0), num_blocks, block_width, {}, rows);
}
INSTANTIATE_TEST_SUITE_P(TensixKernelLib, UntilizeRowsTest, kStandardShapes, ShapeName);

class UntilizePartialRowsTest : public MeshDeviceFixture, public testing::WithParamInterface<PartialRowsParam> {};
TEST_P(UntilizePartialRowsTest, Run) {
    auto [rows, block_width] = GetParam();
    constexpr uint32_t tile_height = 32;
    uint32_t num_blocks = (rows + tile_height - 1) / tile_height;
    unit_tests::compute::helpers::run_untilize_helper_test(this->devices_.at(0), num_blocks, block_width, {}, rows);
}
INSTANTIATE_TEST_SUITE_P(
    TensixKernelLib,
    UntilizePartialRowsTest,
    testing::Values(
        PartialRowsParam{20, 1},
        PartialRowsParam{20, 2},
        PartialRowsParam{48, 1},
        PartialRowsParam{48, 2},
        PartialRowsParam{1, 1},
        PartialRowsParam{1, 4}),
    PartialRowsName);

}  // namespace tt::tt_metal
