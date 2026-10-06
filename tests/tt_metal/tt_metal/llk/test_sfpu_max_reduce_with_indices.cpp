// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <bit>
#include <limits>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/arch.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/program/program_impl.hpp"
#include "llk_device_fixture.hpp"
#include "test_golden_impls.hpp"
#include "tt_metal/test_utils/packing.hpp"

// Metal-layer coverage for max_reduce_with_indices: the public Compute API the pool compute_mpwi
// kernels call, driven with a values tile and an indices tile of different formats, the way those
// kernels load them. The tt-llk sweep covers the SFPU kernel itself; this checks the API plumbing,
// the mixed-format unpack / pack around it, and the accumulate chain across calls.

namespace tt::tt_metal {

using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::compute::max_reduce_with_indices {

constexpr std::uint32_t kTileDim = 32;
constexpr std::uint32_t kTileElems = kTileDim * kTileDim;

// Rows past a 9-row window hold this, above every in-window value, so a reduction that reads them
// changes the result.
constexpr float kOutOfWindowValue = 1000.0f;

enum class MpwiLayout { Tile, RowMajor };

struct MaxReduceWithIndicesConfig {
    int num_rows = 9;  // 9-versus-32 network selector, as in the Compute API
    MpwiLayout layout = MpwiLayout::RowMajor;
    bool accumulate = false;
    std::uint32_t num_chunks = 1;
    tt::DataFormat values_format = tt::DataFormat::Float16_b;
    bool wide_indices = false;  // 32-bit index tile and a 32-bit Dest, as for inputs with H*W > 65535
};

std::uint32_t window_rows(const MaxReduceWithIndicesConfig& config) { return config.num_rows <= 9 ? 9 : kTileDim; }

// The index format pool uses: UInt16, or UInt32 for inputs with H*W > 65535.
//
// Quasar has no integer index format that works here yet: it has no typed UInt16 / UInt32 DFB
// formats, Int16 fails the JIT's rule that a bf16 program's inputs share an exponent class,
// RawUInt16 unpacks as zeros, and Int32 has to unpack straight to Dest, which is kernel-wide on
// Quasar and lands every copy_tile on Dest tile 0. So on Quasar the index tile uses a float format
// of the same width; the kernel only moves index bits (SFPLOAD/SFPSTORE as UINT16 / INT32), and
// every index code here is an integer that the float format holds exactly.
tt::DataFormat index_format(tt::ARCH arch, bool wide) {
    if (arch == tt::ARCH::QUASAR) {
        return wide ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    }
    return wide ? tt::DataFormat::UInt32 : tt::DataFormat::UInt16;
}

bool is_float_format(tt::DataFormat format) {
    return format == tt::DataFormat::Float16_b || format == tt::DataFormat::Float32;
}

bool needs_32_bit_dest(const MaxReduceWithIndicesConfig& config) {
    return config.wide_indices || config.values_format == tt::DataFormat::Float32;
}

std::uint32_t datum_bytes(tt::DataFormat format) {
    switch (format) {
        case tt::DataFormat::Float16_b:
        case tt::DataFormat::UInt16: return 2;
        case tt::DataFormat::Float32:
        case tt::DataFormat::UInt32: return 4;
        default: TT_THROW("unsupported max_reduce_with_indices format {}", format);
    }
}

std::string layout_define(MpwiLayout layout) {
    return layout == MpwiLayout::Tile ? "ckernel::DataLayout::TILE" : "ckernel::DataLayout::ROW_MAJOR";
}

std::vector<std::uint32_t> encode_values(const std::vector<float>& values, tt::DataFormat format) {
    if (format == tt::DataFormat::Float16_b) {
        std::vector<bfloat16> elements(values.begin(), values.end());
        return pack_vector<std::uint32_t, bfloat16>(elements);
    }
    std::vector<std::uint32_t> packed(values.size());
    std::transform(
        values.begin(), values.end(), packed.begin(), [](float v) { return std::bit_cast<std::uint32_t>(v); });
    return packed;
}

std::vector<float> decode_values(const std::vector<std::uint32_t>& packed, tt::DataFormat format) {
    if (format == tt::DataFormat::Float16_b) {
        const auto elements = unpack_vector<bfloat16, std::uint32_t>(packed);
        return std::vector<float>(elements.begin(), elements.end());
    }
    std::vector<float> values(packed.size());
    std::transform(
        packed.begin(), packed.end(), values.begin(), [](std::uint32_t v) { return std::bit_cast<float>(v); });
    return values;
}

// An index code as the element bits the index tile stores; the kernel carries these bits unchanged.
std::uint32_t index_bits(std::uint32_t code, tt::DataFormat format) {
    if (!is_float_format(format)) {
        return code;
    }
    if (format == tt::DataFormat::Float32) {
        return std::bit_cast<std::uint32_t>(static_cast<float>(code));
    }
    return std::bit_cast<std::uint16_t>(bfloat16(static_cast<float>(code)));
}

std::vector<std::uint32_t> encode_indices(const std::vector<std::uint32_t>& codes, tt::DataFormat format) {
    std::vector<std::uint32_t> bits(codes.size());
    std::transform(
        codes.begin(), codes.end(), bits.begin(), [format](std::uint32_t c) { return index_bits(c, format); });
    if (datum_bytes(format) == 4) {
        return bits;
    }
    std::vector<std::uint16_t> elements(bits.begin(), bits.end());
    return pack_vector<std::uint32_t, std::uint16_t>(elements);
}

// The element bits of every entry of a packed index tile.
std::vector<std::uint32_t> decode_index_bits(const std::vector<std::uint32_t>& packed, tt::DataFormat format) {
    if (datum_bytes(format) == 4) {
        return packed;
    }
    const auto elements = unpack_vector<std::uint16_t, std::uint32_t>(packed);
    return std::vector<std::uint32_t>(elements.begin(), elements.end());
}

// One 32x32 tile, row-major, as the kernel reads it for this layout: ROW_MAJOR operands are
// row-major sticks (what pool's reader hands compute_mpwi), TILE operands are tilized.
std::vector<std::uint32_t> to_device_tile(
    const std::vector<std::uint32_t>& packed, MpwiLayout layout, std::uint32_t bytes) {
    if (layout == MpwiLayout::RowMajor) {
        return packed;
    }
    const ::unit_tests::compute::GoldenConfig config{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = bytes};
    return ::unit_tests::compute::gold_standard_tilize(packed, config);
}

std::vector<std::uint32_t> from_device_tile(
    const std::vector<std::uint32_t>& packed, MpwiLayout layout, std::uint32_t bytes) {
    if (layout == MpwiLayout::RowMajor) {
        return packed;
    }
    const ::unit_tests::compute::GoldenConfig config{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = bytes};
    return ::unit_tests::compute::gold_standard_untilize(packed, config);
}

struct Chunk {
    std::vector<float> values;           // row-major 32x32
    std::vector<std::uint32_t> indices;  // row-major 32x32 codes, distinct within each column
};

// Each column of a window holds distinct multiples of 1/8 (exact in bf16 and in the TF32 a
// Float32 operand unpacks to), so the arg-max is unique except in the tie columns, which repeat
// the maximum on a second row. Accumulating chains alternate which chunk wins per column, so the
// running max must both survive and be overtaken.
std::vector<Chunk> generate_chunks(const MaxReduceWithIndicesConfig& config, std::uint32_t seed) {
    std::mt19937 gen(seed);
    const std::uint32_t rows = window_rows(config);
    std::vector<Chunk> chunks(config.num_chunks);
    for (std::uint32_t c = 0; c < config.num_chunks; ++c) {
        auto& chunk = chunks[c];
        chunk.values.assign(kTileElems, kOutOfWindowValue);
        chunk.indices.resize(kTileElems);
        // The check only matches an index against its own column, so a code needs to tell apart
        // the rows of every chunk. Narrow codes stay at or below 256, exact in bf16. Wide codes sit
        // above 16 bits, so a path that drops the upper half fails; spaced 64 apart in [2^16, 2^17)
        // they also stay exact in the TF32 a Quasar Float32 index operand unpacks to.
        for (std::uint32_t i = 0; i < kTileElems; ++i) {
            const std::uint32_t row_code = c * kTileDim + i / kTileDim + 1;
            chunk.indices[i] = config.wide_indices ? (1u << 16) + row_code * 64 : row_code;
        }
        for (std::uint32_t col = 0; col < kTileDim; ++col) {
            std::vector<int> steps(64);
            std::iota(steps.begin(), steps.end(), -32);
            std::shuffle(steps.begin(), steps.end(), gen);
            // In an accumulating chain, chunk (col % num_chunks) wins this column.
            const float bias = (config.num_chunks > 1 && col % config.num_chunks == c) ? 16.0f : 0.0f;
            for (std::uint32_t r = 0; r < rows; ++r) {
                chunk.values[r * kTileDim + col] = static_cast<float>(steps[r]) / 8.0f + bias;
            }
            if (col % 7 == 3) {  // tie: copy this column's maximum onto another row
                std::uint32_t top = 0;
                for (std::uint32_t r = 1; r < rows; ++r) {
                    if (chunk.values[r * kTileDim + col] > chunk.values[top * kTileDim + col]) {
                        top = r;
                    }
                }
                const std::uint32_t other = (top + rows / 2) % rows;
                chunk.values[other * kTileDim + col] = chunk.values[top * kTileDim + col];
            }
        }
    }
    return chunks;
}

void run_max_reduce_with_indices(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const MaxReduceWithIndicesConfig& config) {
    const tt::ARCH arch = mesh_device->arch();
    const bool is_quasar = arch == tt::ARCH::QUASAR;
    const tt::DataFormat idx_format = index_format(arch, config.wide_indices);
    const std::uint32_t value_bytes = datum_bytes(config.values_format);
    const std::uint32_t idx_bytes = datum_bytes(idx_format);
    const std::uint32_t value_tile_size = kTileElems * value_bytes;
    const std::uint32_t idx_tile_size = kTileElems * idx_bytes;
    constexpr std::uint32_t kOutTiles = 2;  // operand tile + the running tile above it

    auto& cq = mesh_device->mesh_command_queue();
    const experimental::NodeCoord node{0, 0};

    const auto make_dram_buffer = [&](std::uint32_t size) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = size},
            distributed::DeviceLocalBufferConfig{.page_size = size, .buffer_type = tt_metal::BufferType::DRAM},
            mesh_device.get());
    };
    auto values_src = make_dram_buffer(value_tile_size * config.num_chunks);
    auto indices_src = make_dram_buffer(idx_tile_size * config.num_chunks);
    auto values_dst = make_dram_buffer(value_tile_size * kOutTiles);
    auto indices_dst = make_dram_buffer(idx_tile_size * kOutTiles);

    const experimental::DFBSpecName VALUES_IN{"values_in"};
    const experimental::DFBSpecName INDICES_IN{"indices_in"};
    const experimental::DFBSpecName VALUES_OUT{"values_out"};
    const experimental::DFBSpecName INDICES_OUT{"indices_out"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    // Two entries: room for both chunks of an accumulating chain, and for an output's operand tile
    // and running tile.
    const auto dfb = [](const experimental::DFBSpecName& id, std::uint32_t entry_size, tt::DataFormat format) {
        return experimental::DataflowBufferSpec{
            .unique_id = id, .entry_size = entry_size, .num_entries = 2, .data_format_metadata = format};
    };
    const auto binding = [](const experimental::DFBSpecName& id, const char* accessor, bool producer) {
        return experimental::DFBBinding{
            .dfb_spec_name = id,
            .accessor_name = accessor,
            .endpoint_type =
                producer ? experimental::DFBEndpointType::PRODUCER : experimental::DFBEndpointType::CONSUMER,
            .access_pattern = experimental::DFBAccessPattern::STRIDED,
        };
    };
    // The reader and writer push / pop their DFBs explicitly.
    const auto dm_config = [&](bool reader) {
        if (is_quasar) {
            return experimental::DataMovementHardwareConfig{
                .config_2xx = experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = true}};
        }
        return experimental::DataMovementHardwareConfig{
            .config_1xx = experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                .processor = reader ? DataMovementProcessor::RISCV_1 : DataMovementProcessor::RISCV_0,
                .noc = reader ? NOC::RISCV_1_default : NOC::RISCV_0_default}};
    };

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {binding(VALUES_IN, "in0", true), binding(INDICES_IN, "in1", true)},
        .runtime_arg_schema =
            {.runtime_arg_names = {"src0_addr", "src0_bank_id", "src1_addr", "src1_bank_id", "num_tiles"}},
        .hw_config = dm_config(true),
    };
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_binary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {binding(VALUES_OUT, "out0", false), binding(INDICES_OUT, "out1", false)},
        .runtime_arg_schema =
            {.runtime_arg_names = {"dst0_addr", "dst0_bank_id", "dst1_addr", "dst1_bank_id", "num_tiles"}},
        .hw_config = dm_config(false),
    };

    // A 32-bit Dest makes the unpack mode a mandatory choice. Both operands go through SrcA, as in
    // compute_mpwi; on Quasar unpacking to Dest is also kernel-wide and ignores copy_tile's Dest index.
    experimental::ComputeHardwareConfig::ComputeUnpackModes unpack_modes{};
    if (needs_32_bit_dest(config)) {
        unpack_modes = {{VALUES_IN, UnpackMode::UnpackToSrc}, {INDICES_IN, UnpackMode::UnpackToSrc}};
    }
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/max_reduce_with_indices.cpp",
        .num_threads = 1,
        .compiler_options =
            {.defines =
                 {{"MPWI_NUM_ROWS", std::to_string(config.num_rows)},
                  {"MPWI_LAYOUT", layout_define(config.layout)},
                  {"MPWI_ACCUMULATE", config.accumulate ? "true" : "false"}}},
        .dfb_bindings =
            {binding(VALUES_IN, "values_in", false),
             binding(INDICES_IN, "indices_in", false),
             binding(VALUES_OUT, "values_out", true),
             binding(INDICES_OUT, "indices_out", true)},
        .compile_time_args = {{"num_chunks", config.num_chunks}},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .enable_32_bit_dest = needs_32_bit_dest(config), .unpack_modes = unpack_modes},
    };

    experimental::ProgramSpec spec{
        .name = "max_reduce_with_indices",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers =
            {dfb(VALUES_IN, value_tile_size, config.values_format),
             dfb(INDICES_IN, idx_tile_size, idx_format),
             dfb(VALUES_OUT, value_tile_size, config.values_format),
             dfb(INDICES_OUT, idx_tile_size, idx_format)},
        .work_units = {experimental::WorkUnitSpec{
            .name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    constexpr std::uint32_t seed = 0x5eed;
    const auto chunks = generate_chunks(config, seed);
    std::vector<std::uint32_t> values_l1;
    std::vector<std::uint32_t> indices_l1;
    for (const auto& chunk : chunks) {
        const auto v = to_device_tile(encode_values(chunk.values, config.values_format), config.layout, value_bytes);
        const auto i = to_device_tile(encode_indices(chunk.indices, idx_format), config.layout, idx_bytes);
        values_l1.insert(values_l1.end(), v.begin(), v.end());
        indices_l1.insert(indices_l1.end(), i.begin(), i.end());
    }
    distributed::EnqueueWriteMeshBuffer(cq, values_src, values_l1, /*blocking=*/true);
    distributed::EnqueueWriteMeshBuffer(cq, indices_src, indices_l1, /*blocking=*/true);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src0_addr", values_src->address()},
                 {"src0_bank_id", 0u},
                 {"src1_addr", indices_src->address()},
                 {"src1_bank_id", 0u},
                 {"num_tiles", config.num_chunks}})},
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst0_addr", values_dst->address()},
                 {"dst0_bank_id", 0u},
                 {"dst1_addr", indices_dst->address()},
                 {"dst1_bank_id", 0u},
                 {"num_tiles", kOutTiles}})},
    };
    experimental::SetProgramRunArgs(program, params);
    LaunchProgram(*mesh_device, std::move(program));

    std::vector<std::uint32_t> values_out;
    std::vector<std::uint32_t> indices_out;
    distributed::EnqueueReadMeshBuffer(cq, values_out, values_dst, /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(cq, indices_out, indices_dst, /*blocking=*/true);

    log_info(
        tt::LogTest,
        "max_reduce_with_indices rows={} layout={} accumulate={} chunks={} values={} indices={}",
        config.num_rows,
        config.layout == MpwiLayout::Tile ? "TILE" : "ROW_MAJOR",
        config.accumulate,
        config.num_chunks,
        config.values_format,
        idx_format);

    // Row 0 of the operand tiles holds the result; when accumulating, so does row 0 of the running
    // tiles above them.
    const std::uint32_t value_words = value_tile_size / sizeof(std::uint32_t);
    const std::uint32_t idx_words = idx_tile_size / sizeof(std::uint32_t);
    const std::uint32_t checked_tiles = config.accumulate ? kOutTiles : 1;
    const std::uint32_t rows = window_rows(config);
    for (std::uint32_t t = 0; t < checked_tiles; ++t) {
        const std::vector<std::uint32_t> v_tile(
            values_out.begin() + t * value_words, values_out.begin() + (t + 1) * value_words);
        const std::vector<std::uint32_t> i_tile(
            indices_out.begin() + t * idx_words, indices_out.begin() + (t + 1) * idx_words);
        const auto got_values =
            decode_values(from_device_tile(v_tile, config.layout, value_bytes), config.values_format);
        const auto got_indices = decode_index_bits(from_device_tile(i_tile, config.layout, idx_bytes), idx_format);

        for (std::uint32_t col = 0; col < kTileDim; ++col) {
            float best = -std::numeric_limits<float>::infinity();
            for (const auto& chunk : chunks) {
                for (std::uint32_t r = 0; r < rows; ++r) {
                    best = std::max(best, chunk.values[r * kTileDim + col]);
                }
            }
            // Any window entry holding the maximum is a valid arg-max.
            bool index_ok = false;
            for (const auto& chunk : chunks) {
                for (std::uint32_t r = 0; r < rows; ++r) {
                    index_ok |= chunk.values[r * kTileDim + col] == best &&
                                index_bits(chunk.indices[r * kTileDim + col], idx_format) == got_indices[col];
                }
            }
            ASSERT_EQ(got_values[col], best) << fmt::format("value mismatch: tile {} column {}", t, col);
            ASSERT_TRUE(index_ok) << fmt::format(
                "index {} at tile {} column {} does not hold the maximum {}", got_indices[col], t, col, best);
        }
    }
}

}  // namespace unit_tests::compute::max_reduce_with_indices

using namespace unit_tests::compute::max_reduce_with_indices;

namespace {

// The index widths pool pairs with bf16 inputs, plus a Float32 values tile.
struct FormatCase {
    tt::DataFormat values_format;
    bool wide_indices;
};
constexpr FormatCase kFormatCases[] = {
    {tt::DataFormat::Float16_b, false},
    {tt::DataFormat::Float16_b, true},
    {tt::DataFormat::Float32, true},
};

void run_all_formats(const std::shared_ptr<distributed::MeshDevice>& device, MaxReduceWithIndicesConfig config) {
    for (const auto& fc : kFormatCases) {
        // ttsim flags a Wormhole SrcA unpack of a UInt32 tile as undefined behaviour; Wormhole
        // silicon handles it (compute_mpwi loads its 32-bit indices the same way, and pool's
        // test_mpwi_32_bit_index covers that on hardware).
        if (fc.wide_indices && device->arch() == tt::ARCH::WORMHOLE_B0 &&
            MetalContext::instance().rtoptions().get_simulator_enabled()) {
            log_info(tt::LogTest, "Skipping 32-bit indices on the Wormhole simulator: ttsim rejects a UInt32 unpack");
            continue;
        }
        config.values_format = fc.values_format;
        config.wide_indices = fc.wide_indices;
        run_max_reduce_with_indices(device, config);
    }
}

}  // namespace

// The 9-row network on a tilized tile pair, the Compute API default layout.
TEST_F(LLKMeshDeviceFixture, TensixComputeMaxReduceWithIndicesTile9) {
    run_all_formats(this->devices_.at(0), {.num_rows = 9, .layout = MpwiLayout::Tile});
}

// The 9-row network on row-major sticks, as pool uses it for windows of up to 9 elements.
TEST_F(LLKMeshDeviceFixture, TensixComputeMaxReduceWithIndicesRowMajor9) {
    run_all_formats(this->devices_.at(0), {.num_rows = 9, .layout = MpwiLayout::RowMajor});
}

// The 32-row network on row-major sticks, as pool uses it for windows of 10 to 32 elements.
TEST_F(LLKMeshDeviceFixture, TensixComputeMaxReduceWithIndicesRowMajor32) {
    run_all_formats(this->devices_.at(0), {.num_rows = 32, .layout = MpwiLayout::RowMajor});
}

// Large-window pooling: two chunks folded through the running max under one Dest acquire. Chunk 0
// seeds the running tiles, chunk 1 wins some columns and loses others.
TEST_F(LLKMeshDeviceFixture, TensixComputeMaxReduceWithIndicesRowMajor32Accumulate) {
    run_all_formats(
        this->devices_.at(0), {.num_rows = 32, .layout = MpwiLayout::RowMajor, .accumulate = true, .num_chunks = 2});
}

}  // namespace tt::tt_metal
