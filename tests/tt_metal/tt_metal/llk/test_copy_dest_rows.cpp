// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// copy_dest_rows / copy_dest_row (api/compute/copy_dest_rows.h) on one core, against the device's own view of the
// operands: the kernel packs the two input tiles straight from DEST first (the reference bits, whatever the unpack path
// made of the host's values), then runs every case on freshly reloaded copies and packs both slots, so a golden is the
// reference with one band or row moved and everything else untouched, compared bit for bit.
//   LaneMap   the sfpi (vector, lane) -> (row, col) map of a DEST tile that the bodies assume (fp32 DEST)
//   bands     all 64 (in_band, out_band) pairs, within one tile and from one tile into another
//   rows      all 1024 (from_row, to_row) pairs, within one tile and from one tile into another: every lane-group pair
// Inputs carry NaN payloads, both signs of zero, infinities and a denormal, so "bit-exact" is checked on those too.
// Configs: fp32 CBs on an fp32 DEST (unpacked straight to DEST on Blackhole), bf16 CBs on an fp32 DEST, bf16 CBs on a
// 16-bit DEST. The kernel is test_kernels/compute/copy_dest_rows.cpp.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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

#include "llk_device_fixture.hpp"

namespace tt::tt_metal {
namespace unit_tests::compute::copy_dest_rows {

constexpr uint32_t kElems = 32 * 32;
constexpr uint32_t kBankId = 0;
constexpr uint32_t kBandCases = 8 * 8;   // in_band x out_band, per target
constexpr uint32_t kRowCases = 32 * 32;  // from_row x to_row, per target
constexpr uint32_t kModeLaneMap = 0, kModeBands = 1, kModeRows = 2;

struct Config {
    tt::DataFormat df;  // the CBs' format: Float32 or Float16_b
    bool fp32_dest;     // fp32_dest_acc_en
    uint32_t mode;      // kModeBands or kModeRows
};

std::string config_name(const Config& c) {
    return std::string(c.mode == kModeBands ? "Bands" : "Rows") + "_" +
           (c.df == tt::DataFormat::Float32 ? "FP32" : "BF16") + "in_" + (c.fp32_dest ? "FP32" : "BF16") + "dest";
}

constexpr uint32_t elem_bytes(tt::DataFormat df) { return df == tt::DataFormat::Float32 ? 4u : 2u; }

// Element (r, c) of a 32x32 tile in device tile order: faces 0..3 of 16x16, row-major within a face.
constexpr uint32_t tile_index(uint32_t r, uint32_t c) { return ((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + c % 16; }

// Raw element bits, indexed r * 32 + c; bf16 in the low 16 bits.
using Elems = std::vector<uint32_t>;

uint32_t to_bf16_bits(uint32_t w) {
    if ((w & 0x7F800000u) == 0x7F800000u && (w & 0x007FFFFFu) != 0u) {
        return ((w >> 16) | 0x40u) & 0xFFFFu;  // a quiet NaN of the same sign
    }
    return ((w + 0x7FFFu + ((w >> 16) & 1u)) >> 16) & 0xFFFFu;
}

Elems quantize(const Elems& fp32, tt::DataFormat df) {
    if (df == tt::DataFormat::Float32) {
        return fp32;
    }
    Elems out(kElems);
    for (uint32_t i = 0; i < kElems; i++) {
        out[i] = to_bf16_bits(fp32[i]);
    }
    return out;
}

// A: 32 * r + c + 0.5 with special values planted; B: -(32 * r + c) - 0.25 with a few of its own. Row-major fp32 bits.
Elems pattern_a() {
    Elems e(kElems);
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t c = 0; c < 32; c++) {
            e[r * 32 + c] = std::bit_cast<uint32_t>(static_cast<float>(32 * r + c) + 0.5f);
        }
    }
    e[0 * 32 + 0] = 0x7FC00123u;   // NaN with a payload
    e[1 * 32 + 5] = 0x80000000u;   // -0.0
    e[2 * 32 + 31] = 0x7F800000u;  // +inf
    e[3 * 32 + 16] = 0xFF800000u;  // -inf
    e[5 * 32 + 7] = 0x00010000u;   // a denormal (bf16-representable)
    e[16 * 32 + 0] = 0x7F800000u;
    e[17 * 32 + 3] = 0xFFC05555u;  // -NaN with a payload
    e[31 * 32 + 31] = 0x80000000u;
    return e;
}

Elems pattern_b() {
    Elems e(kElems);
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t c = 0; c < 32; c++) {
            e[r * 32 + c] = std::bit_cast<uint32_t>(-static_cast<float>(32 * r + c) - 0.25f);
        }
    }
    e[8 * 32 + 8] = 0x7FC00042u;
    e[20 * 32 + 2] = 0x80000000u;
    e[9 * 32 + 30] = 0xFF800000u;
    return e;
}

// Row-major elements -> the device's tile words (fp32: one per element; bf16: two per word, the even element low).
std::vector<uint32_t> to_tile_words(const Elems& e, tt::DataFormat df) {
    std::vector<uint32_t> tiled(kElems);
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t c = 0; c < 32; c++) {
            tiled[tile_index(r, c)] = e[r * 32 + c];
        }
    }
    if (df == tt::DataFormat::Float32) {
        return tiled;
    }
    std::vector<uint32_t> words(kElems / 2);
    for (uint32_t i = 0; i < kElems / 2; i++) {
        words[i] = (tiled[2 * i] & 0xFFFFu) | ((tiled[2 * i + 1] & 0xFFFFu) << 16);
    }
    return words;
}

// One tile of device words -> row-major element bits.
Elems from_tile_words(const uint32_t* words, tt::DataFormat df) {
    std::vector<uint32_t> tiled(kElems);
    if (df == tt::DataFormat::Float32) {
        std::copy(words, words + kElems, tiled.begin());
    } else {
        for (uint32_t i = 0; i < kElems / 2; i++) {
            tiled[2 * i] = words[i] & 0xFFFFu;
            tiled[2 * i + 1] = words[i] >> 16;
        }
    }
    Elems e(kElems);
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t c = 0; c < 32; c++) {
            e[r * 32 + c] = tiled[tile_index(r, c)];
        }
    }
    return e;
}

// The program: A and B in, `out_tiles` tiles out, as the kernel's MODE describes; `pair` selects the target (0: A
// itself, 1: B) of the band and row modes.
std::vector<Elems> run(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    tt::DataFormat df,
    bool fp32_dest,
    uint32_t mode,
    uint32_t num_cases,
    uint32_t pair,
    const Elems& a,
    const Elems& b) {
    auto& cq = mesh_device->mesh_command_queue();
    const auto zero_coord = distributed::MeshCoordinate(0, 0);
    const auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    const CoreCoord core = {0, 0};

    const uint32_t tile_bytes = kElems * elem_bytes(df);
    const uint32_t out_tiles = mode == kModeLaneMap ? 1u : 2u + 2u * num_cases;
    auto make_dram = [&](uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            distributed::DeviceLocalBufferConfig{
                .page_size = bytes, .buffer_type = BufferType::DRAM, .bottom_up = false},
            mesh_device.get());
    };
    auto make_cb = [&](uint32_t idx, uint32_t tiles) {
        const auto cb = static_cast<CBIndex>(idx);
        CreateCircularBuffer(
            program_, core, CircularBufferConfig(tiles * tile_bytes, {{cb, df}}).set_page_size(cb, tile_bytes));
    };
    make_cb(0, 1);
    make_cb(1, 1);
    make_cb(16, 4);
    auto a_buf = make_dram(tile_bytes);
    auto b_buf = make_dram(tile_bytes);
    auto out_buf = make_dram(out_tiles * tile_bytes);
    std::vector<uint32_t> a_words = to_tile_words(a, df), b_words = to_tile_words(b, df);
    distributed::WriteShard(cq, a_buf, a_words, zero_coord);
    distributed::WriteShard(cq, b_buf, b_words, zero_coord);

    auto reader = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto writer = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    // Blackhole unpacks fp32 straight to DEST (an exact copy); Wormhole routes fp32 through SrcA as tf32. The checks
    // below compare against the device's own packed reference, so both are fine.
    std::vector<UnpackToDestMode> utd(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (df == tt::DataFormat::Float32 && mesh_device->arch() == tt::ARCH::BLACKHOLE) {
        utd[0] = UnpackToDestMode::UnpackToDestFp32;
        utd[1] = UnpackToDestMode::UnpackToDestFp32;
    }
    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/copy_dest_rows.cpp",
        core,
        ComputeConfig{
            .fp32_dest_acc_en = fp32_dest,
            .unpack_to_dest_mode = utd,
            .math_approx_mode = false,
            .compile_args = {mode, num_cases, pair}});
    SetRuntimeArgs(
        program_, reader, core, {uint32_t(a_buf->address()), kBankId, uint32_t(b_buf->address()), kBankId, 1u});
    SetRuntimeArgs(program_, writer, core, {uint32_t(out_buf->address()), kBankId, out_tiles});
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> out;
    distributed::ReadShard(cq, out, out_buf, zero_coord);
    const uint32_t words_per_tile = tile_bytes / 4;
    std::vector<Elems> tiles;
    tiles.reserve(out_tiles);
    for (uint32_t t = 0; t < out_tiles; t++) {
        tiles.push_back(from_tile_words(out.data() + t * words_per_tile, df));
    }
    return tiles;
}

// The golden of one case: the reference tiles with one band (rows [r0, r0 + n)) or row moved.
struct Expected {
    Elems slot0, slot1;
};

Expected expected_rows(
    const Elems& ref_a, const Elems& ref_b, uint32_t pair, uint32_t from_r0, uint32_t to_r0, uint32_t n) {
    Expected e{ref_a, ref_b};
    Elems& target = pair == 0 ? e.slot0 : e.slot1;
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t c = 0; c < 32; c++) {
            target[(to_r0 + i) * 32 + c] = ref_a[(from_r0 + i) * 32 + c];
        }
    }
    return e;
}

// Compares one case's two packed slots against the expectation; logs the first mismatches. Returns the mismatch count.
uint32_t check_case(
    const std::string& what, const Elems& got0, const Elems& got1, const Expected& exp, uint32_t& reported) {
    uint32_t mismatches = 0;
    for (uint32_t slot = 0; slot < 2; slot++) {
        const Elems& got = slot == 0 ? got0 : got1;
        const Elems& want = slot == 0 ? exp.slot0 : exp.slot1;
        for (uint32_t i = 0; i < kElems; i++) {
            if (got[i] != want[i]) {
                mismatches++;
                if (reported < 12) {
                    reported++;
                    log_error(
                        tt::LogTest,
                        "{}: slot {} row {} col {}: got {:#010x}, expected {:#010x}",
                        what,
                        slot,
                        i / 32,
                        i % 32,
                        got[i],
                        want[i]);
                }
            }
        }
    }
    return mismatches;
}

bool run_cases(const std::shared_ptr<distributed::MeshDevice>& dev, const Config& cfg, uint32_t pair) {
    const Elems a = quantize(pattern_a(), cfg.df), b = quantize(pattern_b(), cfg.df);
    const uint32_t num_cases = cfg.mode == kModeBands ? kBandCases : kRowCases;
    const auto tiles = run(dev, cfg.df, cfg.fp32_dest, cfg.mode, num_cases, pair, a, b);
    if (tiles.size() != 2 + 2 * num_cases) {
        log_error(tt::LogTest, "expected {} tiles, got {}", 2 + 2 * num_cases, tiles.size());
        return false;
    }
    const Elems& ref_a = tiles[0];
    const Elems& ref_b = tiles[1];
    // The reference is what the device holds; on the exact paths it is the host pattern itself.
    uint32_t ref_diff = 0;
    for (uint32_t i = 0; i < kElems; i++) {
        for (uint32_t which = 0; which < 2; which++) {
            const uint32_t host = which == 0 ? a[i] : b[i], dev = which == 0 ? ref_a[i] : ref_b[i];
            if (host != dev) {
                if (ref_diff < 8) {
                    log_info(
                        tt::LogTest,
                        "  the unpack path changed {} row {} col {}: host {:#010x}, in DEST {:#010x}",
                        which == 0 ? "A" : "B",
                        i / 32,
                        i % 32,
                        host,
                        dev);
                }
                ref_diff++;
            }
        }
    }
    log_info(
        tt::LogTest,
        "{}: reference tiles differ from the host patterns in {} of {} elements (0 on an exact unpack path)",
        config_name(cfg),
        ref_diff,
        2 * kElems);
    uint32_t mismatches = 0, reported = 0, bad_cases = 0;
    for (uint32_t k = 0; k < num_cases; k++) {
        Expected exp;
        std::string what;
        if (cfg.mode == kModeBands) {
            const uint32_t in_band = k / 8, out_band = k % 8;
            exp = expected_rows(ref_a, ref_b, pair, 4 * in_band, 4 * out_band, 4);
            what =
                "band " + std::to_string(in_band) + " -> band " + std::to_string(out_band) + (pair ? " of B" : " of A");
        } else {
            const uint32_t from_row = k / 32, to_row = k % 32;
            exp = expected_rows(ref_a, ref_b, pair, from_row, to_row, 1);
            what = "row " + std::to_string(from_row) + " -> row " + std::to_string(to_row) + (pair ? " of B" : " of A");
        }
        const uint32_t m = check_case(what, tiles[2 + 2 * k], tiles[3 + 2 * k], exp, reported);
        mismatches += m;
        bad_cases += m != 0;
    }
    log_info(
        tt::LogTest,
        "{}, target {}: {} cases, {} with mismatches, {} mismatching elements",
        config_name(cfg),
        pair ? "B" : "A itself",
        num_cases,
        bad_cases,
        mismatches);
    return mismatches == 0;
}

}  // namespace unit_tests::compute::copy_dest_rows

using unit_tests::compute::copy_dest_rows::Config;

class CopyDestRowsFixture : public LLKMeshDeviceFixture, public ::testing::WithParamInterface<Config> {};

// The DEST geometry the bodies assume, on record: value = 1000 * vector + 2 * lane written into every lane of slot 0,
// read back as (row, col) -> (vector, lane). Face f = (r / 16) * 2 + c / 16 at vectors 8f..8f+7; vector
// 8f + 2 * ((r % 16) / 4) + (c % 2); lane 8 * (r % 4) + (c % 16) / 2.
TEST_F(LLKMeshDeviceFixture, TensixCopyDestRowsLaneMap) {
    using namespace unit_tests::compute::copy_dest_rows;
    if (this->arch_ != tt::ARCH::WORMHOLE_B0 && this->arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "copy_dest_rows is defined for Wormhole B0 and Blackhole";
    }
    for (auto& dev : this->devices_) {
        const auto tiles = run(dev, tt::DataFormat::Float32, true, kModeLaneMap, 0, 0, pattern_a(), pattern_b());
        ASSERT_EQ(tiles.size(), 1u);
        const Elems& t = tiles[0];
        uint32_t ok = 0;
        for (uint32_t r = 0; r < 32; r++) {
            for (uint32_t c = 0; c < 32; c++) {
                const uint32_t v = static_cast<uint32_t>(std::bit_cast<float>(t[r * 32 + c]));
                const uint32_t f = (r / 16) * 2 + c / 16;
                const uint32_t vec = 8 * f + 2 * ((r % 16) / 4) + (c % 2), lane = 8 * (r % 4) + (c % 16) / 2;
                ok += (v / 1000 == vec) && ((v % 1000) / 2 == lane);
            }
        }
        log_info(
            tt::LogTest, "DEST lane map: {} / {} elements at the 4-row-band / even-odd-column position", ok, kElems);
        EXPECT_EQ(ok, kElems) << "the DEST geometry is not the 4-row-band / even-odd-column one copy_dest_rows assumes";
    }
}

TEST_P(CopyDestRowsFixture, TensixCopyDestRows) {
    if (this->arch_ != tt::ARCH::WORMHOLE_B0 && this->arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "copy_dest_rows is defined for Wormhole B0 and Blackhole";
    }
    const Config cfg = GetParam();
    for (auto& dev : this->devices_) {
        EXPECT_TRUE(unit_tests::compute::copy_dest_rows::run_cases(dev, cfg, 0)) << "within one tile";
        EXPECT_TRUE(unit_tests::compute::copy_dest_rows::run_cases(dev, cfg, 1)) << "from one tile into another";
    }
}

INSTANTIATE_TEST_SUITE_P(
    CopyDestRowsAllVariants,
    CopyDestRowsFixture,
    ::testing::Values(
        Config{tt::DataFormat::Float32, true, unit_tests::compute::copy_dest_rows::kModeBands},
        Config{tt::DataFormat::Float32, true, unit_tests::compute::copy_dest_rows::kModeRows},
        Config{tt::DataFormat::Float16_b, true, unit_tests::compute::copy_dest_rows::kModeBands},
        Config{tt::DataFormat::Float16_b, true, unit_tests::compute::copy_dest_rows::kModeRows},
        Config{tt::DataFormat::Float16_b, false, unit_tests::compute::copy_dest_rows::kModeBands},
        Config{tt::DataFormat::Float16_b, false, unit_tests::compute::copy_dest_rows::kModeRows}),
    [](const ::testing::TestParamInfo<Config>& info) {
        return unit_tests::compute::copy_dest_rows::config_name(info.param);
    });

}  // namespace tt::tt_metal
