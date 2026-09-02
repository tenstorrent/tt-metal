// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/span.hpp>
#include <umd/device/types/arch.hpp>

#include "impl/program/program_impl.hpp"
#include "llk_device_fixture.hpp"
#include "test_golden_impls.hpp"
#include "tt_metal/test_utils/packing.hpp"

// Metal-layer coverage for the Quasar SFPU reduce, driven through the public Compute API rather
// than the LLK directly. The LLK sweep in tt-llk covers the kernel itself; this covers the layer
// above -- that a normal metal program can reach the op and gets the right answer back.
//
// The kernel writes only the axis it collapses and leaves the rest of the tile holding fold
// leftovers, so the comparison reads just that axis, which is what a reduce's consumers read:
//   * REDUCE_COL folds each tile's 32 rows onto its row 0; tiles reduce independently.
//   * REDUCE_ROW folds a tile row's columns onto column 0 of that row, spanning every tile in it.

namespace tt::tt_metal {

using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::compute::sfpu_reduce {

constexpr uint32_t kTileHeight = 32;
constexpr uint32_t kTileWidth = 32;

enum class ReduceAxis { Column, Row };
enum class ReducePool { Sum, Avg, Max, Min };

struct SfpuReduceConfig {
    // Tiles per tile row, and tile rows per block. A row reduce needs its whole block resident in
    // Dest, so these two also bound how much Dest the program asks for.
    uint32_t block_ct_dim = 1;
    uint32_t block_rt_dim = 1;
    uint32_t num_blocks = 1;
    ReduceAxis axis = ReduceAxis::Column;
    ReducePool pool = ReducePool::Sum;
    tt::DataFormat format = tt::DataFormat::Float16_b;
};

std::string pool_type_define(ReducePool pool) {
    switch (pool) {
        case ReducePool::Sum: return "PoolType::SUM";
        case ReducePool::Avg: return "PoolType::AVG";
        case ReducePool::Max: return "PoolType::MAX";
        case ReducePool::Min: return "PoolType::MIN";
    }
    TT_THROW("unreachable pool type");
}

std::string format_define(tt::DataFormat format) {
    switch (format) {
        case tt::DataFormat::Float32: return "DataFormat::Float32";
        case tt::DataFormat::Float16_b: return "DataFormat::Float16_b";
        case tt::DataFormat::Int32: return "DataFormat::Int32";
        default: TT_THROW("unsupported sfpu_reduce format");
    }
}

std::string pool_name(ReducePool pool) {
    switch (pool) {
        case ReducePool::Sum: return "SUM";
        case ReducePool::Avg: return "AVG";
        case ReducePool::Max: return "MAX";
        case ReducePool::Min: return "MIN";
    }
    TT_THROW("unreachable pool type");
}

bool is_int_format(tt::DataFormat format) { return format == tt::DataFormat::Int32; }

// How the operand reaches Dest, which is what decides whether a multi-tile block can be staged.
//
// UnpackToSrc goes through SrcA and the math-side datacopy, whose per-tile loop advances the Dest
// index, so copy_block gives each tile its own Dest tile. UnpackToDest skips that loop
// (llk_math_eltwise_unary_datacopy_block: "math is a sync-only forwarder") and every tile lands on
// Dest tile 0, overwriting the last. Int32 must unpack to Dest, so Int32 cannot use a multi-tile block.
tt::tt_metal::UnpackMode unpack_mode_for(tt::DataFormat format) {
    return is_int_format(format) ? tt::tt_metal::UnpackMode::UnpackToDest : tt::tt_metal::UnpackMode::UnpackToSrc;
}

// Whether a block wider than one tile can be staged for this format.
bool supports_multi_tile_block(tt::DataFormat format) { return !is_int_format(format); }

// A 32-bit format needs a 32-bit Dest; nothing widens a narrow datum on the way in.
bool needs_fp32_dest_acc(tt::DataFormat format) {
    return format == tt::DataFormat::Float32 || format == tt::DataFormat::Int32;
}

uint32_t datum_bytes(tt::DataFormat format) { return format == tt::DataFormat::Float16_b ? 2 : 4; }

std::vector<uint32_t> encode_elements(const std::vector<double>& values, tt::DataFormat format) {
    if (format == tt::DataFormat::Float16_b) {
        std::vector<bfloat16> elements(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            elements[i] = bfloat16(static_cast<float>(values[i]));
        }
        return pack_vector<uint32_t, bfloat16>(elements);
    }

    std::vector<uint32_t> packed(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        if (format == tt::DataFormat::Int32) {
            packed[i] = static_cast<uint32_t>(static_cast<int32_t>(values[i]));
        } else {
            packed[i] = std::bit_cast<uint32_t>(static_cast<float>(values[i]));
        }
    }
    return packed;
}

std::vector<double> decode_elements(const std::vector<uint32_t>& packed, tt::DataFormat format) {
    if (format == tt::DataFormat::Float16_b) {
        const auto elements = unpack_vector<bfloat16, uint32_t>(packed);
        std::vector<double> values(elements.size());
        for (size_t i = 0; i < elements.size(); ++i) {
            values[i] = static_cast<float>(elements[i]);
        }
        return values;
    }

    std::vector<double> values(packed.size());
    for (size_t i = 0; i < packed.size(); ++i) {
        if (format == tt::DataFormat::Int32) {
            values[i] = static_cast<double>(std::bit_cast<int32_t>(packed[i]));
        } else {
            values[i] = static_cast<double>(std::bit_cast<float>(packed[i]));
        }
    }
    return values;
}

// Stimulus in a range the accumulating pools can hold. A row reduce folds block_ct_dim * 32 terms,
// so the bound is chosen against the widest fold the config asks for rather than a fixed span.
std::vector<double> generate_stimulus(const SfpuReduceConfig& config, uint32_t rows, uint32_t cols, uint32_t seed) {
    const uint32_t fold_width = (config.axis == ReduceAxis::Row) ? cols : kTileHeight;
    std::mt19937 gen(seed);

    std::vector<double> values(static_cast<size_t>(rows) * cols);
    if (is_int_format(config.format)) {
        // Keep |sum| well inside Int32 so SUM stays exact and the test measures the fold, not
        // overflow. Signed, so MIN and MAX both see negatives.
        const int32_t bound = static_cast<int32_t>(1'000'000 / fold_width);
        std::uniform_int_distribution<int32_t> dist(-bound, bound);
        for (auto& v : values) {
            v = static_cast<double>(dist(gen));
        }
    } else {
        std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
        for (auto& v : values) {
            v = dist(gen);
        }
    }
    return values;
}

// Fold `values` (row-major rows x cols) and return only the lane the kernel writes:
//   Column -> one value per column, for each tile row       (block_rt_dim * num_blocks * cols)
//   Row    -> one value per row                             (rows)
std::vector<double> gold_sfpu_reduce(
    const std::vector<double>& values, const SfpuReduceConfig& config, uint32_t rows, uint32_t cols) {
    const auto fold = [&config](double acc, double x, bool first) {
        if (first) {
            return x;
        }
        switch (config.pool) {
            case ReducePool::Sum:
            case ReducePool::Avg: return acc + x;
            case ReducePool::Max: return std::max(acc, x);
            case ReducePool::Min: return std::min(acc, x);
        }
        TT_THROW("unreachable pool type");
    };

    std::vector<double> golden;

    if (config.axis == ReduceAxis::Column) {
        const uint32_t tile_rows = rows / kTileHeight;
        golden.reserve(static_cast<size_t>(tile_rows) * cols);
        for (uint32_t tr = 0; tr < tile_rows; ++tr) {
            for (uint32_t col = 0; col < cols; ++col) {
                double acc = 0.0;
                for (uint32_t r = 0; r < kTileHeight; ++r) {
                    const double x = values[static_cast<size_t>(tr * kTileHeight + r) * cols + col];
                    acc = fold(acc, x, r == 0);
                }
                if (config.pool == ReducePool::Avg) {
                    // A column AVG always divides by the tile's 32 rows. The integer path does it
                    // with a shift that truncates toward zero, so the golden has to as well.
                    acc /= static_cast<double>(kTileHeight);
                    if (is_int_format(config.format)) {
                        acc = std::trunc(acc);
                    }
                }
                golden.push_back(acc);
            }
        }
        return golden;
    }

    golden.reserve(rows);
    for (uint32_t r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (uint32_t col = 0; col < cols; ++col) {
            acc = fold(acc, values[static_cast<size_t>(r) * cols + col], col == 0);
        }
        if (config.pool == ReducePool::Avg) {
            acc /= static_cast<double>(cols);
        }
        golden.push_back(acc);
    }
    return golden;
}

// Pull the same lanes out of the untilized device output that gold_sfpu_reduce returns.
std::vector<double> extract_written_lane(
    const std::vector<double>& values, const SfpuReduceConfig& config, uint32_t rows, uint32_t cols) {
    std::vector<double> lane;
    if (config.axis == ReduceAxis::Column) {
        const uint32_t tile_rows = rows / kTileHeight;
        lane.reserve(static_cast<size_t>(tile_rows) * cols);
        for (uint32_t tr = 0; tr < tile_rows; ++tr) {
            for (uint32_t col = 0; col < cols; ++col) {
                lane.push_back(values[static_cast<size_t>(tr * kTileHeight) * cols + col]);
            }
        }
        return lane;
    }

    lane.reserve(rows);
    for (uint32_t r = 0; r < rows; ++r) {
        lane.push_back(values[static_cast<size_t>(r) * cols]);
    }
    return lane;
}

// Absolute tolerance for the accumulating pools. Summing N terms of magnitude M in a
// low-precision float accumulates error like sqrt(N) * M * eps, and where terms nearly cancel that
// error dwarfs the tiny true total — so a fixed tolerance would fail a correct reduction. MAX/MIN
// and the integer formats reduce exactly.
double reduce_atol(const SfpuReduceConfig& config, uint32_t cols) {
    if (is_int_format(config.format)) {
        return 0.0;
    }
    if (config.pool == ReducePool::Max || config.pool == ReducePool::Min) {
        return 0.0;
    }

    const double eps = (config.format == tt::DataFormat::Float16_b) ? 0.0078125      // bf16: 2^-7
                                                                    : 1.1920929e-7;  // fp32: 2^-23
    const double max_term = 4.0;                                                     // matches generate_stimulus
    const double num_terms = (config.axis == ReduceAxis::Row) ? cols : kTileHeight;

    double atol = 2.0 * max_term * eps * std::sqrt(num_terms);
    if (config.pool == ReducePool::Avg) {
        atol /= num_terms;
    }
    return std::max(0.05, atol);
}

void run_single_core_sfpu_reduce(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SfpuReduceConfig& config) {
    auto& cq = mesh_device->mesh_command_queue();
    const experimental::NodeCoord node{0, 0};

    const uint32_t elem_bytes = datum_bytes(config.format);
    const uint32_t single_tile_size = kTileWidth * kTileHeight * elem_bytes;
    const uint32_t tiles_per_block = config.block_ct_dim * config.block_rt_dim;
    const uint32_t num_tiles = tiles_per_block * config.num_blocks;
    const uint32_t buffer_size = single_tile_size * num_tiles;

    // Blocks stack along rows, so the buffer is one strip block_ct_dim tiles wide. That keeps each
    // block's tiles contiguous in tile order, which is what the compute kernel walks.
    const uint32_t cols = config.block_ct_dim * kTileWidth;
    const uint32_t rows = config.num_blocks * config.block_rt_dim * kTileHeight;

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    auto src_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = buffer_size}, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = buffer_size}, dram_config, mesh_device.get());

    const experimental::DFBSpecName IN_DFB{"in_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    // Room for two blocks, not one. A block has to be resident all at once (a row reduce reaches
    // across every tile in its tile row), and sizing to exactly one block leaves no slack: the
    // reader cannot fill the next block while this one is in flight, and the writer cannot drain
    // the last one while the next is being packed.
    experimental::DataflowBufferSpec in_dfb_spec{
        .unique_id = IN_DFB,
        .entry_size = single_tile_size,
        .num_entries = 2 * tiles_per_block,
        .data_format_metadata = config.format,
    };
    experimental::DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = 2 * tiles_per_block,
        .data_format_metadata = config.format,
    };

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(IN_DFB, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec::CompilerOptions::Defines defines{
        {"REDUCE_POOL_TYPE", pool_type_define(config.pool)},
        {"REDUCE_FORMAT", format_define(config.format)},
    };
    if (config.axis == ReduceAxis::Row) {
        defines["REDUCE_AXIS_ROW"] = "1";
    }

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/sfpu_reduce_quasar.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = defines},
        .dfb_bindings = {experimental::ConsumerOf(IN_DFB, "in"), experimental::ProducerOf(OUT_DFB, "out")},
        .compile_time_args =
            {{"block_ct_dim", config.block_ct_dim},
             {"block_rt_dim", config.block_rt_dim},
             {"num_blocks", config.num_blocks}},
        // A 32-bit Dest makes the unpack mode a mandatory choice; see unpack_mode_for.
        .hw_config =
            experimental::ComputeGen2Config{
                .enable_32_bit_dest = needs_fp32_dest_acc(config.format),
                .unpack_modes = {{IN_DFB, unpack_mode_for(config.format)}}},
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "sfpu_reduce",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {in_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    const std::vector<double> input = generate_stimulus(
        config, rows, cols, static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count()));

    const ::unit_tests::compute::GoldenConfig golden_config{
        .num_tiles_r_dim = static_cast<int>(config.num_blocks * config.block_rt_dim),
        .num_tiles_c_dim = static_cast<int>(config.block_ct_dim),
        .datum_bytes = elem_bytes};

    const auto input_tilized =
        ::unit_tests::compute::gold_standard_tilize(encode_elements(input, config.format), golden_config);

    distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, input_tilized, /*blocking=*/true);

    const auto src_page_stride = static_cast<uint32_t>(src_dram_buffer->get_reference_buffer()->aligned_page_size());
    const auto dst_page_stride = static_cast<uint32_t>(dst_dram_buffer->get_reference_buffer()->aligned_page_size());

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src_addr", src_dram_buffer->address()},
                 {"src_bank_id", 0u},
                 {"num_tiles", num_tiles},
                 {"dram_page_stride", src_page_stride}})},
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", dst_dram_buffer->address()},
                 {"dst_bank_id", 0u},
                 {"num_tiles", num_tiles},
                 {"dram_page_stride", dst_page_stride}})},
    };
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(*mesh_device, std::move(program), /*wait_until_cores_done=*/true);

    std::vector<uint32_t> output_tilized;
    distributed::EnqueueReadMeshBuffer(cq, output_tilized, dst_dram_buffer, /*blocking=*/true);

    const auto output =
        decode_elements(::unit_tests::compute::gold_standard_untilize(output_tilized, golden_config), config.format);

    const auto golden = gold_sfpu_reduce(input, config, rows, cols);
    const auto device = extract_written_lane(output, config, rows, cols);

    log_info(
        tt::LogTest,
        "sfpu_reduce {} {} ct_dim={} rt_dim={} blocks={} format={}",
        config.axis == ReduceAxis::Row ? "row" : "column",
        pool_name(config.pool),
        config.block_ct_dim,
        config.block_rt_dim,
        config.num_blocks,
        format_define(config.format));

    ASSERT_EQ(golden.size(), device.size());

    const double atol = reduce_atol(config, cols);
    const double rtol = is_int_format(config.format) ? 0.0 : 0.05;

    for (size_t i = 0; i < golden.size(); ++i) {
        const double diff = std::abs(golden[i] - device[i]);
        const bool ok = diff <= atol + rtol * std::abs(golden[i]);
        ASSERT_TRUE(ok) << fmt::format(
            "mismatch at {}: golden={} device={} (atol={} rtol={})", i, golden[i], device[i], atol, rtol);
    }
}

}  // namespace unit_tests::compute::sfpu_reduce

using namespace unit_tests::compute::sfpu_reduce;

// Column reduce: every pool, including the integer AVG that divides by 32 with a shift.
//
// Width is swept with num_blocks rather than block_ct_dim, i.e. one tile resident at a time. A
// column never leaves its tile, so this covers the same arithmetic; holding a multi-tile block in
// Dest and reducing each tile in place is a separate path that does not work from here yet (see
// TensixComputeSfpuReduceRow).
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, TensixComputeSfpuReduceColumn) {
    for (auto format : {tt::DataFormat::Float16_b, tt::DataFormat::Float32}) {
        for (auto pool : {ReducePool::Sum, ReducePool::Avg, ReducePool::Max, ReducePool::Min}) {
            for (uint32_t num_blocks : {1u, 4u}) {
                run_single_core_sfpu_reduce(
                    this->devices_.at(0),
                    SfpuReduceConfig{
                        .block_ct_dim = 1,
                        .block_rt_dim = 1,
                        .num_blocks = num_blocks,
                        .axis = ReduceAxis::Column,
                        .pool = pool,
                        .format = format});
            }
            // Two tiles wide. Every tile keeps its own result on this axis, so the golden checks
            // all of them -- making this the one case that would catch a block whose later tiles
            // never got written out. The row cases cannot see that: a row result lives in the
            // first tile alone, so they pass even if the rest of the block comes back empty.
            if (supports_multi_tile_block(format)) {
                run_single_core_sfpu_reduce(
                    this->devices_.at(0),
                    SfpuReduceConfig{
                        .block_ct_dim = 2,
                        .block_rt_dim = 1,
                        .num_blocks = 1,
                        .axis = ReduceAxis::Column,
                        .pool = pool,
                        .format = format});
            }
        }
    }
}

// Row reduce: SUM, MAX and MIN. AVG is excluded because sfpu_reduce's static_assert allows
// REDUCE_ROW only for those three, for every format which takes a float row AVG and makes
// only the integer one column-only.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, TensixComputeSfpuReduceRow) {
    for (auto format : {tt::DataFormat::Float16_b, tt::DataFormat::Float32}) {
        for (auto pool : {ReducePool::Sum, ReducePool::Max, ReducePool::Min}) {
            // Int32 must unpack to Dest, which cannot stage a block, so it stays single-tile.
            const uint32_t widest = supports_multi_tile_block(format) ? 2u : 1u;
            for (uint32_t block_ct_dim = 1; block_ct_dim <= widest; ++block_ct_dim) {
                run_single_core_sfpu_reduce(
                    this->devices_.at(0),
                    SfpuReduceConfig{
                        .block_ct_dim = block_ct_dim,
                        .block_rt_dim = 1,
                        .num_blocks = 1,
                        .axis = ReduceAxis::Row,
                        .pool = pool,
                        .format = format});
            }
        }
    }
}

TEST_F(LLKQuasarMeshDeviceSingleCardFixture, TensixComputeSfpuReduceInt32) {
    for (auto pool : {ReducePool::Sum, ReducePool::Avg, ReducePool::Max, ReducePool::Min}) {
        for (uint32_t num_blocks : {1u, 4u}) {
            run_single_core_sfpu_reduce(
                this->devices_.at(0),
                SfpuReduceConfig{
                    .block_ct_dim = 1,
                    .block_rt_dim = 1,
                    .num_blocks = num_blocks,
                    .axis = ReduceAxis::Column,
                    .pool = pool,
                    .format = tt::DataFormat::Int32});
        }
    }
    for (auto pool : {ReducePool::Sum, ReducePool::Max, ReducePool::Min}) {
        run_single_core_sfpu_reduce(
            this->devices_.at(0),
            SfpuReduceConfig{
                .block_ct_dim = 1,
                .block_rt_dim = 1,
                .num_blocks = 1,
                .axis = ReduceAxis::Row,
                .pool = pool,
                .format = tt::DataFormat::Int32});
    }
}
}  // namespace tt::tt_metal
