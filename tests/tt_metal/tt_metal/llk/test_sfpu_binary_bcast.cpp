// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The SFPU binary broadcast family of the compute API (sfpu_binary_bcast.h) on one 32x32 tile per program:
//   - the whole-tile column and row broadcasts (sfpu_*_bcast_{col,row}) for ADD / SUB / MUL, fp32 and bf16;
//   - the column broadcast on rows 0..15 only (VectorMode::R) and on one 4-row band (sfpu_*_bcast_col_band), where
//     the rows the form does not write must stay as they were and the parts of the column tile the form does not
//     read are planted with NaN / Inf that must not leak into the output;
//   - the parked scalar (sfpu_bcast_scalar_load, sfpu_mul_bcast_scalar): bit-identical to the column broadcast with
//     a uniform column and to the host product, reads only element [0, 0], copies special bit patterns, survives a
//     broadcast op and is re-loaded after exp_tile_init, over the (ITERATIONS, VectorMode) extents and the DEST
//     slots of both the immediate and the generic address paths (fp32 full sync for slots 4..7).
// Goldens are host-side; fp32 compares bit-exact (Blackhole unpacks fp32 straight to DEST; Wormhole's TF32 SrcA
// path is modelled by quantising the inputs), bf16 at about one ulp.

#include <fmt/base.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-umd/types/arch.hpp>

#include "device_fixture.hpp"
#include "test_golden_impls.hpp"

namespace tt::tt_metal {

namespace unit_tests::compute::sfpu_binary_bcast {

enum class BcastDim : uint8_t {
    COL = 0,  // replicate col 0 of bcast tile across all 32 cols
    ROW = 1,  // replicate row 0 of bcast tile across all 32 rows
};

enum class BinOp : uint8_t {
    ADD = 0,
    SUB = 1,
    MUL = 2,
};

// Which part of the data tile the column broadcast writes.
enum class Form : uint8_t {
    FULL = 0,       // the whole tile (sfpu_*_bcast_{col,row})
    ROWS_0_15 = 1,  // rows 0..15 (sfpu_*_bcast_col with VectorMode::R)
    BAND = 2,       // rows 4 * data_band .. +3 from column 0 rows 4 * bcast_band .. +3 (sfpu_*_bcast_col_band)
};

struct SfpuBcastConfig {
    BcastDim bcast_dim;
    BinOp binop;
    tt::DataFormat df;  // Float32 or Float16_b
    Form form = Form::FULL;
    uint32_t data_band = 0;
    uint32_t bcast_band = 0;
};

// Format a BcastDim/BinOp/DataFormat for test log output only.
const std::map<BcastDim, std::string> bcast_dim_name = {
    {BcastDim::COL, "BCAST_COL"},
    {BcastDim::ROW, "BCAST_ROW"},
};
const std::map<BinOp, std::string> binop_name = {
    {BinOp::ADD, "ADD"},
    {BinOp::SUB, "SUB"},
    {BinOp::MUL, "MUL"},
};
const std::map<tt::DataFormat, std::string> df_name = {
    {tt::DataFormat::Float32, "FP32"},
    {tt::DataFormat::Float16_b, "BF16"},
};
const std::map<Form, std::string> form_name = {
    {Form::FULL, "FULL"},
    {Form::ROWS_0_15, "R"},
    {Form::BAND, "BAND"},
};

constexpr uint32_t kTileHW = 32 * 32;
constexpr uint32_t kNumInTiles = 1;  // one tile per input
constexpr uint32_t kQuietNaNBits = 0x7FC00000u;

// Bytes per element for the CB data format.
constexpr uint32_t elem_bytes(tt::DataFormat df) { return (df == tt::DataFormat::Float32) ? 4u : 2u; }

// -----------------------------------------------------------------------------
// Stimulus + golden
// -----------------------------------------------------------------------------

std::vector<float> generate_random_tile(uint64_t seed, float lo, float hi) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> out(kTileHW);
    for (auto& v : out) {
        v = dist(rng);
    }
    return out;
}

// Row-major 32x32 FP32 -> reinterpreted as uint32 vector (bitcast, no conversion).
std::vector<uint32_t> reinterpret_fp32_as_u32(const std::vector<float>& in) {
    std::vector<uint32_t> out(in.size());
    static_assert(sizeof(float) == sizeof(uint32_t));
    std::memcpy(out.data(), in.data(), in.size() * sizeof(float));
    return out;
}

// Quantize each float to bfloat16 (tie-to-even) and return a parallel float vector
// holding the exactly-representable-as-bfloat16 values. Used both for feeding the
// device and for computing a host-side golden that matches the device's input
// precision.
std::vector<float> quantize_to_bf16_float(const std::vector<float>& in) {
    std::vector<float> out(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = static_cast<float>(bfloat16(in[i]));
    }
    return out;
}

constexpr uint32_t TF32_MASK = 0xffffe000u;
// Wormhole's supported FP32-to-SrcA path converts FP32 inputs to TF32 by
// dropping the low 13 mantissa bits.
std::vector<float> quantize_to_tf32_float(const std::vector<float>& in) {
    std::vector<float> out(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        const auto bits = std::bit_cast<uint32_t>(in[i]);
        out[i] = std::bit_cast<float>(bits & TF32_MASK);
    }
    return out;
}
// Row-major 32x32 vector of floats (already quantized to bfloat16) -> packed uint32
// vector with two bfloat16 lanes per uint32 (low 16 = even idx, high 16 = odd idx).
// This is the layout gold_standard_tilize expects when datum_bytes == 2.
std::vector<uint32_t> pack_bf16_as_u32(const std::vector<float>& in) {
    assert(in.size() % 2 == 0);
    std::vector<uint32_t> out(in.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = pack_two_bfloat16_into_uint32({bfloat16(in[2 * i]), bfloat16(in[2 * i + 1])});
    }
    return out;
}

// Does the form write data row r?
bool form_writes_row(const SfpuBcastConfig& cfg, uint32_t r) {
    switch (cfg.form) {
        case Form::FULL: return true;
        case Form::ROWS_0_15: return r < 16;
        case Form::BAND: return r / 4 == cfg.data_band;
    }
    return true;
}

// The bcast-tile row that feeds data row r of a column broadcast.
uint32_t form_bcast_row(const SfpuBcastConfig& cfg, uint32_t r) {
    return cfg.form == Form::BAND ? 4 * cfg.bcast_band + (r - 4 * cfg.data_band) : r;
}

// Does the form read element (r, c) of the bcast tile?
bool form_reads_bcast(const SfpuBcastConfig& cfg, uint32_t r, uint32_t c) {
    if (cfg.bcast_dim == BcastDim::ROW) {
        return r == 0;
    }
    if (c != 0) {
        return false;
    }
    switch (cfg.form) {
        case Form::FULL: return true;
        case Form::ROWS_0_15: return r < 16;
        case Form::BAND: return r / 4 == cfg.bcast_band;
    }
    return true;
}

// NaN and +-Inf into every element of the bcast tile the partial forms never read: the masked column broadcast
// must not let them reach the output.
void plant_unread_bcast(const SfpuBcastConfig& cfg, std::vector<float>& b_rm) {
    if (cfg.form == Form::FULL) {
        return;
    }
    const float plants[3] = {
        std::bit_cast<float>(kQuietNaNBits),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            if (!form_reads_bcast(cfg, r, c)) {
                b_rm[r * 32 + c] = plants[(r + c) % 3];
            }
        }
    }
}

// Compute the row-major 32x32 golden for the selected broadcast + binop: the rows the form writes get the binop
// against their broadcast value, the others keep the data.
std::vector<float> compute_golden(
    const std::vector<float>& a_rm, const std::vector<float>& b_rm, const SfpuBcastConfig& cfg) {
    std::vector<float> golden(kTileHW);
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            const float a = a_rm[r * 32 + c];
            if (!form_writes_row(cfg, r)) {
                golden[r * 32 + c] = a;
                continue;
            }
            const float b = (cfg.bcast_dim == BcastDim::COL) ? b_rm[form_bcast_row(cfg, r) * 32 + 0] : b_rm[0 * 32 + c];
            float g = 0.0f;
            switch (cfg.binop) {
                case BinOp::ADD: g = a + b; break;
                case BinOp::SUB: g = a - b; break;
                case BinOp::MUL: g = a * b; break;
            }
            golden[r * 32 + c] = g;
        }
    }
    return golden;
}

// -----------------------------------------------------------------------------
// Test driver
// -----------------------------------------------------------------------------

// Device output tiles (tile order) widened to one float per element, for a format-agnostic compare.
std::vector<float> widen_to_floats(const std::vector<uint32_t>& tiled, bool is_fp32) {
    std::vector<float> out;
    out.reserve(tiled.size() * (is_fp32 ? 1 : 2));
    if (is_fp32) {
        for (uint32_t w : tiled) {
            out.push_back(std::bit_cast<float>(w));
        }
    } else {
        for (uint32_t w : tiled) {
            const bfloat16 lo = std::bit_cast<bfloat16>(static_cast<uint16_t>(w & 0xffff));
            const bfloat16 hi = std::bit_cast<bfloat16>(static_cast<uint16_t>(w >> 16));
            out.push_back(static_cast<float>(lo));
            out.push_back(static_cast<float>(hi));
        }
    }
    return out;
}

// One program on core (0, 0): c_0 = tile A, c_1 = tile B (one tile each, format `df`), the compute kernel built
// with `defines`, c_16 = `num_out_tiles` output tiles read back and widened. fp32 unpacks straight to DEST on
// Blackhole; Wormhole routes fp32 through its TF32 SrcA path (the caller quantises the goldens accordingly).
std::vector<float> run_bcast_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    tt::DataFormat df,
    const std::map<std::string, std::string>& defines,
    uint32_t num_out_tiles,
    std::vector<uint32_t> src_a_tiled,  // by value: WriteShard takes a mutable vector
    std::vector<uint32_t> src_b_tiled,
    bool dst_full_sync) {
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    Program program = tt_metal::CreateProgram(*mesh_device);
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    CoreCoord core = {0, 0};
    const uint32_t tile_bytes = kTileHW * elem_bytes(df);
    const uint32_t in_bytes = tile_bytes * kNumInTiles;
    const uint32_t out_bytes = tile_bytes * num_out_tiles;
    const bool is_fp32 = (df == tt::DataFormat::Float32);

    // DRAM buffers
    distributed::DeviceLocalBufferConfig in_dram_config{
        .page_size = in_bytes, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig in_buffer_config{.size = in_bytes};
    distributed::DeviceLocalBufferConfig out_dram_config{
        .page_size = out_bytes, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig out_buffer_config{.size = out_bytes};

    auto src_data_buffer = distributed::MeshBuffer::create(in_buffer_config, in_dram_config, mesh_device.get());
    auto src_bcast_buffer = distributed::MeshBuffer::create(in_buffer_config, in_dram_config, mesh_device.get());
    auto dst_buffer = distributed::MeshBuffer::create(out_buffer_config, out_dram_config, mesh_device.get());

    // CBs: c_0 = data, c_1 = bcast, c_16 = out, all in `df`.
    tt_metal::CircularBufferConfig cb_data_cfg =
        tt_metal::CircularBufferConfig(in_bytes, {{tt::CBIndex::c_0, df}}).set_page_size(tt::CBIndex::c_0, tile_bytes);
    tt_metal::CreateCircularBuffer(program_, core, cb_data_cfg);

    tt_metal::CircularBufferConfig cb_bcast_cfg =
        tt_metal::CircularBufferConfig(in_bytes, {{tt::CBIndex::c_1, df}}).set_page_size(tt::CBIndex::c_1, tile_bytes);
    tt_metal::CreateCircularBuffer(program_, core, cb_bcast_cfg);

    tt_metal::CircularBufferConfig cb_out_cfg = tt_metal::CircularBufferConfig(out_bytes, {{tt::CBIndex::c_16, df}})
                                                    .set_page_size(tt::CBIndex::c_16, tile_bytes);
    tt_metal::CreateCircularBuffer(program_, core, cb_out_cfg);

    // Reader: 2 DRAM buffers -> c_0, c_1 (one tile each, single-bank)
    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    // Writer: c_16 -> DRAM
    auto writer_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    // Blackhole can unpack FP32 directly to DEST. Wormhole must use its supported
    // FP32-to-TF32 SrcA path: its unpack-to-DEST cleanup issues a final
    // unpack-to-SrcA, for which FP32-to-FP32 is undefined by the unpacker contract.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    const bool use_fp32_unpack_to_dest = is_fp32 && mesh_device->arch() == tt::ARCH::BLACKHOLE;
    if (use_fp32_unpack_to_dest) {
        unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[tt::CBIndex::c_1] = UnpackToDestMode::UnpackToDestFp32;
    }

    tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/sfpu_binary_bcast.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = dst_full_sync,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = false,
            .compile_args = {},
            .defines = defines});

    distributed::WriteShard(cq, src_data_buffer, src_a_tiled, zero_coord);
    distributed::WriteShard(cq, src_bcast_buffer, src_b_tiled, zero_coord);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {
            static_cast<uint32_t>(src_data_buffer->address()),
            0u,  // src0 bank id
            static_cast<uint32_t>(src_bcast_buffer->address()),
            0u,  // src1 bank id
            kNumInTiles,
        });
    tt_metal::SetRuntimeArgs(
        program_,
        writer_kernel,
        core,
        {
            static_cast<uint32_t>(dst_buffer->address()),
            0u,  // dst bank id
            num_out_tiles,
        });

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> device_tiled;
    distributed::ReadShard(cq, device_tiled, dst_buffer, zero_coord);
    return widen_to_floats(device_tiled, is_fp32);
}

// Whether the fp32 inputs reach DEST as they are (Blackhole) or TF32-quantised (Wormhole's SrcA path).
bool fp32_unpacks_to_dest(const std::shared_ptr<distributed::MeshDevice>& mesh_device, tt::DataFormat df) {
    return df == tt::DataFormat::Float32 && mesh_device->arch() == tt::ARCH::BLACKHOLE;
}

// golden vs device, element by element: fp32 bit-exact (a relative 1e-6 guard for the TF32 path), bf16 at about
// one ulp (2^-8 relative) with a small absolute floor near zero.
bool compare_tiles(
    const std::vector<float>& golden_f, const std::vector<float>& device_f, bool is_fp32, const char* label) {
    if (golden_f.size() != device_f.size()) {
        log_error(
            tt::LogTest,
            "{}: size mismatch: golden={} words, device={} words",
            label,
            golden_f.size(),
            device_f.size());
        return false;
    }
    const float atol = is_fp32 ? 0.0f : 1.0f / 128.0f;
    const float rtol = is_fp32 ? 1e-6f : 1.0f / 128.0f;

    size_t num_mismatches = 0;
    constexpr size_t max_report = 8;
    for (size_t i = 0; i < device_f.size(); ++i) {
        const float g = golden_f[i];
        const float d = device_f[i];
        const float diff = std::fabs(g - d);
        const float tol = std::max(atol, rtol * std::fabs(g));
        if (!(diff <= tol)) {  // a NaN on either side is a mismatch
            if (num_mismatches < max_report) {
                log_error(
                    tt::LogTest, "{}: mismatch at flat idx {}: golden={} device={} diff={}", label, i, g, d, diff);
            }
            ++num_mismatches;
        }
    }
    if (num_mismatches != 0) {
        log_error(tt::LogTest, "{}: total mismatches: {}/{}", label, num_mismatches, device_f.size());
        return false;
    }
    return true;
}

bool run_sfpu_binary_bcast(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SfpuBcastConfig& cfg) {
    const tt::DataFormat df = cfg.df;
    const bool is_fp32 = (df == tt::DataFormat::Float32);
    const bool use_fp32_unpack_to_dest = fp32_unpacks_to_dest(mesh_device, df);

    std::map<std::string, std::string> compute_defines = {
        {"FORM_VAL", std::to_string(static_cast<int>(cfg.form))},
        {"BCAST_DIM_VAL", std::to_string(static_cast<int>(cfg.bcast_dim))},
        {"BINOP_VAL", std::to_string(static_cast<int>(cfg.binop))},
        {"DATA_BAND_VAL", std::to_string(cfg.data_band)},
        {"BCAST_BAND_VAL", std::to_string(cfg.bcast_band)},
    };

    // Stimulus (row-major FP32). For bfloat16 we quantize inputs up-front so the host-side
    // golden sees the same bit-exact values the device will see after the unpacker.
    const uint64_t seed_a = std::chrono::system_clock::now().time_since_epoch().count();
    const uint64_t seed_b = seed_a ^ 0x9E3779B97F4A7C15ull;
    auto src_a_rm = generate_random_tile(seed_a, -2.0f, 2.0f);
    auto src_b_rm = generate_random_tile(seed_b, -2.0f, 2.0f);
    plant_unread_bcast(cfg, src_b_rm);
    if (!is_fp32) {
        src_a_rm = quantize_to_bf16_float(src_a_rm);
        src_b_rm = quantize_to_bf16_float(src_b_rm);
    }

    // Golden (row-major), including the input precision of the selected unpack
    // path, then tilize to match device-side tile layout.
    const auto golden_a_rm = is_fp32 && !use_fp32_unpack_to_dest ? quantize_to_tf32_float(src_a_rm) : src_a_rm;
    const auto golden_b_rm = is_fp32 && !use_fp32_unpack_to_dest ? quantize_to_tf32_float(src_b_rm) : src_b_rm;
    auto golden_rm = compute_golden(golden_a_rm, golden_b_rm, cfg);

    ::unit_tests::compute::GoldenConfig gc{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = elem_bytes(df)};

    auto to_tile_input = [&](const std::vector<float>& rm) {
        return is_fp32 ? reinterpret_fp32_as_u32(rm) : pack_bf16_as_u32(rm);
    };
    auto src_a_tiled = ::unit_tests::compute::gold_standard_tilize(to_tile_input(src_a_rm), gc);
    auto src_b_tiled = ::unit_tests::compute::gold_standard_tilize(to_tile_input(src_b_rm), gc);
    auto golden_tiled = ::unit_tests::compute::gold_standard_tilize(to_tile_input(golden_rm), gc);

    const auto device_f = run_bcast_program(mesh_device, df, compute_defines, 1, src_a_tiled, src_b_tiled, false);
    const auto golden_f = widen_to_floats(golden_tiled, is_fp32);
    return compare_tiles(golden_f, device_f, is_fp32, "bcast");
}

// Every configuration of the column / row broadcast family: the whole-tile forms, the rows-0..15 form and the 16
// band pairs, each for ADD / SUB / MUL in fp32 and bf16.
std::vector<SfpuBcastConfig> all_bcast_configs() {
    std::vector<SfpuBcastConfig> out;
    for (tt::DataFormat df : {tt::DataFormat::Float32, tt::DataFormat::Float16_b}) {
        for (BinOp op : {BinOp::ADD, BinOp::SUB, BinOp::MUL}) {
            out.push_back({BcastDim::COL, op, df});
            out.push_back({BcastDim::ROW, op, df});
            out.push_back({BcastDim::COL, op, df, Form::ROWS_0_15});
            for (uint32_t data_band = 0; data_band < 4; ++data_band) {
                for (uint32_t bcast_band = 0; bcast_band < 4; ++bcast_band) {
                    out.push_back({BcastDim::COL, op, df, Form::BAND, data_band, bcast_band});
                }
            }
        }
    }
    return out;
}

std::string bcast_config_name(const SfpuBcastConfig& cfg) {
    std::string name = bcast_dim_name.at(cfg.bcast_dim) + "_" + binop_name.at(cfg.binop) + "_" + df_name.at(cfg.df);
    if (cfg.form == Form::ROWS_0_15) {
        name += "_R";
    } else if (cfg.form == Form::BAND) {
        name += "_BAND" + std::to_string(cfg.data_band) + "_FROM" + std::to_string(cfg.bcast_band);
    }
    return name;
}

// -----------------------------------------------------------------------------
// The parked scalar
// -----------------------------------------------------------------------------

// How the scalar tile holds the scalar: alone at [0, 0] with NaN everywhere else (the load must read nothing else),
// or in every row of column 0 with NaN elsewhere (so the column broadcast of the same tile is the reference).
enum class ScalarLayout : uint8_t { LONE, COLUMN };

struct ScalarCase {
    std::string name;
    uint32_t variant;  // the kernel's SCALAR_VARIANT_VAL: 0 scalar form, 1 plus the column form, 2 survive, 3 re-init
    int iterations;    // SFPU vectors per face, 8 or 2 (4 is fine too; odd counts cover one column parity)
    uint32_t mode;     // 0 RC, 1 R, 2 C
    uint32_t dst_data;
    uint32_t dst_scalar;
    uint32_t dst_coldata;
    uint32_t scalar_bits;
    ScalarLayout layout;
    bool full_sync;     // fp32 full sync: eight DEST slots, so slots 4..7 take the generic address path
    bool host_compare;  // the scalar is a normal fp32 the host product is defined for (bit-exact on the device)
};

// Does sfpu_mul_bcast_scalar<iterations>(dst, mode) write element (r, c)?
bool scalar_writes(uint32_t mode, int iterations, uint32_t r, uint32_t c) {
    const bool face_upper = r < 16, face_left = c < 16;
    bool face_selected = true;
    if (mode == 1) {
        face_selected = face_upper;  // R: faces 0 and 1
    } else if (mode == 2) {
        face_selected = face_left;  // C: faces 0 and 2
    }
    // Iteration k of a face is face rows 4 * (k / 2) .. +3; with an even count both column parities are covered.
    return face_selected && static_cast<int>(2 * ((r % 16) / 4)) < iterations;
}

bool run_sfpu_bcast_scalar(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const ScalarCase& sc) {
    const tt::DataFormat df = tt::DataFormat::Float32;
    const bool use_fp32_unpack_to_dest = fp32_unpacks_to_dest(mesh_device, df);
    const uint32_t num_out_tiles = (sc.variant == 1 || sc.variant == 2) ? 2 : 1;

    std::map<std::string, std::string> compute_defines = {
        {"FORM_VAL", "3"},
        {"SCALAR_VARIANT_VAL", std::to_string(sc.variant)},
        {"SCALAR_ITER_VAL", std::to_string(sc.iterations)},
        {"SCALAR_MODE_VAL", std::to_string(sc.mode)},
        {"DST_DATA_VAL", std::to_string(sc.dst_data)},
        {"DST_SCALAR_VAL", std::to_string(sc.dst_scalar)},
        {"DST_COLDATA_VAL", std::to_string(sc.dst_coldata)},
        {"NUM_OUT_TILES_VAL", std::to_string(num_out_tiles)},
    };

    // Fixed seeds: a failure must reproduce.
    auto src_a_rm = generate_random_tile(0x5CA1A7u + sc.dst_data, -2.0f, 2.0f);
    std::vector<float> src_b_rm(kTileHW, std::bit_cast<float>(kQuietNaNBits));
    const float s = std::bit_cast<float>(sc.scalar_bits);
    src_b_rm[0] = s;
    if (sc.layout == ScalarLayout::COLUMN) {
        for (uint32_t r = 0; r < 32; ++r) {
            src_b_rm[r * 32] = s;
        }
    }

    // The inputs as DEST sees them (TF32 on the Wormhole SrcA path; the scalar bits included).
    const auto a_rm = use_fp32_unpack_to_dest ? src_a_rm : quantize_to_tf32_float(src_a_rm);
    const float s_seen = use_fp32_unpack_to_dest ? s : std::bit_cast<float>(sc.scalar_bits & TF32_MASK);

    ::unit_tests::compute::GoldenConfig gc{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = 4};
    auto src_a_tiled = ::unit_tests::compute::gold_standard_tilize(reinterpret_fp32_as_u32(src_a_rm), gc);
    auto src_b_tiled = ::unit_tests::compute::gold_standard_tilize(reinterpret_fp32_as_u32(src_b_rm), gc);
    auto a_tiled_f =
        widen_to_floats(::unit_tests::compute::gold_standard_tilize(reinterpret_fp32_as_u32(a_rm), gc), true);

    const auto device_f =
        run_bcast_program(mesh_device, df, compute_defines, num_out_tiles, src_a_tiled, src_b_tiled, sc.full_sync);
    if (device_f.size() != num_out_tiles * kTileHW) {
        log_error(tt::LogTest, "{}: device returned {} elements", sc.name, device_f.size());
        return false;
    }

    // The tile-order index of element (r, c): the face, then the position in it.
    auto tile_index = [](uint32_t r, uint32_t c) { return ((r / 16) * 2 + (c / 16)) * 256 + (r % 16) * 16 + (c % 16); };
    auto bits = [](float f) { return std::bit_cast<uint32_t>(f); };

    size_t mismatches = 0;
    constexpr size_t max_report = 8;
    auto report = [&](const char* what, uint32_t r, uint32_t c, float expected, float got) {
        if (mismatches < max_report) {
            log_error(
                tt::LogTest,
                "{}: {} at ({}, {}): expected {:#x} ({}) got {:#x} ({})",
                sc.name,
                what,
                r,
                c,
                bits(expected),
                expected,
                bits(got),
                got);
        }
        ++mismatches;
    };

    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t i = tile_index(r, c);
            const float a = a_tiled_f[i];
            const float d = device_f[i];
            const bool written = scalar_writes(sc.mode, sc.iterations, r, c);
            if (!written) {
                if (bits(d) != bits(a)) {
                    report("untouched element changed", r, c, a, d);
                }
                continue;
            }
            if (num_out_tiles == 2) {
                // The column form of the same scalar on the same data: the same multiply, so the same bits.
                const float col = device_f[kTileHW + i];
                if (bits(d) != bits(col)) {
                    report("scalar form differs from the column form", r, c, col, d);
                }
            }
            if (sc.host_compare) {
                const float g = a * s_seen;
                if (bits(d) != bits(g)) {
                    report("not the host product", r, c, g, d);
                }
            } else if (std::isnan(s_seen)) {
                if (!std::isnan(d)) {
                    report("a NaN scalar did not propagate", r, c, s_seen, d);
                }
            } else if (std::isinf(s_seen)) {
                const float g = a * s_seen;
                if (!(d == g)) {
                    report("not the host product (inf)", r, c, g, d);
                }
            }
        }
    }
    if (mismatches != 0) {
        log_error(tt::LogTest, "{}: {} mismatches", sc.name, mismatches);
        return false;
    }
    return true;
}

std::vector<ScalarCase> all_scalar_cases() {
    constexpr uint32_t kNegZero = 0x80000000u;
    constexpr uint32_t kInf = 0x7F800000u;
    constexpr uint32_t kDenormal = 0x00000001u;
    auto f = [](float v) { return std::bit_cast<uint32_t>(v); };
    // name, variant, iterations, mode, dst_data, dst_scalar, dst_coldata, scalar, layout, full_sync, host_compare
    return {
        {"identity_rc8", 1, 8, 0, 0, 1, 2, f(0.75f), ScalarLayout::COLUMN, false, true},
        {"identity_rc8_negative", 1, 8, 0, 0, 1, 2, f(-2.5f), ScalarLayout::COLUMN, false, true},
        {"lone_rc8", 0, 8, 0, 0, 1, 2, f(1.25f), ScalarLayout::LONE, false, true},
        {"lone_rc2", 0, 2, 0, 0, 1, 2, f(1.25f), ScalarLayout::LONE, false, true},
        {"lone_r8", 0, 8, 1, 0, 1, 2, f(1.25f), ScalarLayout::LONE, false, true},
        {"lone_r2", 0, 2, 1, 0, 1, 2, f(1.25f), ScalarLayout::LONE, false, true},
        {"lone_c8", 0, 8, 2, 0, 1, 2, f(1.25f), ScalarLayout::LONE, false, true},
        {"lone_rc4", 0, 4, 0, 0, 1, 2, f(0.5f), ScalarLayout::LONE, false, true},
        // The SFPU multiply gives a positive zero where the host product is -0.0, so this case is device against
        // device only: the parked -0.0 scales exactly as the column broadcast of -0.0 does.
        {"negative_zero", 1, 8, 0, 0, 1, 2, kNegZero, ScalarLayout::COLUMN, false, false},
        {"nan", 1, 8, 0, 0, 1, 2, kQuietNaNBits, ScalarLayout::COLUMN, false, false},
        {"inf", 1, 8, 0, 0, 1, 2, kInf, ScalarLayout::COLUMN, false, false},
        {"denormal", 1, 8, 0, 0, 1, 2, kDenormal, ScalarLayout::COLUMN, false, false},
        {"survives_bcast_op", 2, 8, 0, 0, 1, 2, f(0.5f), ScalarLayout::COLUMN, false, true},
        {"survives_bcast_op_r2", 2, 2, 1, 0, 1, 2, f(0.5f), ScalarLayout::COLUMN, false, true},
        {"reload_after_exp_init", 3, 8, 0, 0, 1, 2, f(1.5f), ScalarLayout::LONE, false, true},
        {"slots_3_2", 0, 8, 0, 3, 2, 0, f(1.25f), ScalarLayout::LONE, false, true},
        {"slots_7_4_full_sync", 1, 8, 0, 7, 4, 5, f(0.75f), ScalarLayout::COLUMN, true, true},
        {"slots_6_1_full_sync_r2", 0, 2, 1, 6, 1, 0, f(1.25f), ScalarLayout::LONE, true, true},
    };
}

}  // namespace unit_tests::compute::sfpu_binary_bcast

using unit_tests::compute::sfpu_binary_bcast::bcast_config_name;
using unit_tests::compute::sfpu_binary_bcast::ScalarCase;
using unit_tests::compute::sfpu_binary_bcast::SfpuBcastConfig;

class SfpuBinaryBcastFixture : public MeshDeviceFixture, public ::testing::WithParamInterface<SfpuBcastConfig> {};

TEST_P(SfpuBinaryBcastFixture, TensixSfpuBinaryBcast) {
    // These SFPU bcast LLKs are only defined on Wormhole B0 / Blackhole.
    if (this->arch_ != tt::ARCH::WORMHOLE_B0 && this->arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "SFPU binary-bcast kernels are only supported on Wormhole B0 and Blackhole";
    }

    const SfpuBcastConfig cfg = GetParam();
    log_info(tt::LogTest, "Testing SFPU binary bcast: {}", bcast_config_name(cfg));

    for (auto& device : this->devices_) {
        ASSERT_TRUE(unit_tests::compute::sfpu_binary_bcast::run_sfpu_binary_bcast(device, cfg));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SfpuBinaryBcastAllVariants,
    SfpuBinaryBcastFixture,
    ::testing::ValuesIn(unit_tests::compute::sfpu_binary_bcast::all_bcast_configs()),
    [](const ::testing::TestParamInfo<SfpuBcastConfig>& info) { return bcast_config_name(info.param); });

class SfpuBcastScalarFixture : public MeshDeviceFixture, public ::testing::WithParamInterface<ScalarCase> {};

TEST_P(SfpuBcastScalarFixture, TensixSfpuBcastScalar) {
    if (this->arch_ != tt::ARCH::WORMHOLE_B0 && this->arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "The SFPU scalar broadcast is only supported on Wormhole B0 and Blackhole";
    }

    const ScalarCase sc = GetParam();
    log_info(tt::LogTest, "Testing SFPU bcast scalar: {}", sc.name);

    for (auto& device : this->devices_) {
        ASSERT_TRUE(unit_tests::compute::sfpu_binary_bcast::run_sfpu_bcast_scalar(device, sc));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SfpuBcastScalarAllVariants,
    SfpuBcastScalarFixture,
    ::testing::ValuesIn(unit_tests::compute::sfpu_binary_bcast::all_scalar_cases()),
    [](const ::testing::TestParamInfo<ScalarCase>& info) { return info.param.name; });

}  // namespace tt::tt_metal
