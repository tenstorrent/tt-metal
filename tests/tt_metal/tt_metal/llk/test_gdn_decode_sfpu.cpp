// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GDN decode (two-pass form), the SFPU side: the DEST-level helpers of fused_recurrent_gdn_sfpu.hpp on Blackhole,
// one core, fp32 DEST in half-sync, HiFi4. In order of importance:
//   DestRowMap      the sfpi (vector, lane) -> (row, col) map of an fp32 DEST tile, and the row moves built on it
//   DestPersistence does a DEST slot that is never packed keep its contents across acquires (both halves)?
//   VectorPass      u = beta (v - e r_k), o = e r_q + c u for one V tile, vs a double reference; NaN in the padding
//   ExpExact        exp in exact mode on a column tile vs std::exp, and what mul_col_bcast then applies
// Cycle counts are the MATH thread's wall clock around REPS repetitions; the output CB holds every repetition so the
// pack thread never waits for the test writer. The kernel is test_kernels/compute/gdn_decode_sfpu.cpp.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "llk_device_fixture.hpp"

namespace tt::tt_metal {
namespace unit_tests::compute::gdn_decode_sfpu {

constexpr uint32_t VT = 4;
constexpr uint32_t kNumIn = 5;
constexpr uint32_t kDramBankId = 0;
constexpr uint32_t kStatsBytes = 64;
constexpr uint32_t kReps = 16;
const float kNaN = std::numeric_limits<float>::quiet_NaN();

struct SfpuTileSpec {
    uint32_t r, c;
    uint32_t fr() const { return std::min<uint32_t>(r, 16); }
    uint32_t fc() const { return std::min<uint32_t>(c, 16); }
    uint32_t elems() const { return r * c; }
    uint32_t bytes() const { return r * c * 4; }
};
constexpr SfpuTileSpec kT32{32, 32}, kT16x32{16, 32};

// Row-major [r x c] -> device tile order: faces row-major over the face grid, rows within a face.
std::vector<uint32_t> sfpu_tilize(const std::vector<float>& rm, SfpuTileSpec t) {
    std::vector<uint32_t> out;
    out.reserve(t.elems());
    for (uint32_t fy = 0; fy < t.r / t.fr(); fy++) {
        for (uint32_t fx = 0; fx < t.c / t.fc(); fx++) {
            for (uint32_t rr = 0; rr < t.fr(); rr++) {
                for (uint32_t cc = 0; cc < t.fc(); cc++) {
                    out.push_back(std::bit_cast<uint32_t>(rm[(fy * t.fr() + rr) * t.c + fx * t.fc() + cc]));
                }
            }
        }
    }
    return out;
}

std::vector<float> sfpu_untilize(const uint32_t* tile, SfpuTileSpec t) {
    std::vector<float> rm(t.elems());
    uint32_t n = 0;
    for (uint32_t fy = 0; fy < t.r / t.fr(); fy++) {
        for (uint32_t fx = 0; fx < t.c / t.fc(); fx++) {
            for (uint32_t rr = 0; rr < t.fr(); rr++) {
                for (uint32_t cc = 0; cc < t.fc(); cc++) {
                    rm[(fy * t.fr() + rr) * t.c + fx * t.fc() + cc] = std::bit_cast<float>(tile[n++]);
                }
            }
        }
    }
    return rm;
}

struct SfpuOperand {
    SfpuTileSpec spec{32, 32};
    std::vector<uint32_t> data;  // tilized tiles, concatenated
    uint32_t ntiles = 0;
    void add(const std::vector<float>& rm) {
        auto t = sfpu_tilize(rm, spec);
        data.insert(data.end(), t.begin(), t.end());
        ntiles++;
    }
};

SfpuOperand one_tile(const std::vector<float>& rm, SfpuTileSpec spec) {
    SfpuOperand op{spec};
    op.add(rm);
    return op;
}

// ---- the program --------------------------------------------------------------------------------------------------
struct SfpuRunConfig {
    uint32_t variant;
    uint32_t mode = 0;
    std::vector<SfpuOperand> inputs;  // CBs c_0 .. c_4
    SfpuTileSpec out_spec;
    uint32_t out_tiles_per_rep;
    uint32_t reps = kReps;
};

struct SfpuRunResult {
    std::vector<std::vector<float>> tiles;  // the last rep's output tiles, row-major each
    std::vector<std::vector<float>> first;  // the first rep's output tiles (repeatability)
    uint32_t cycles = 0;
};

SfpuRunResult run(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SfpuRunConfig& c) {
    auto& cq = mesh_device->mesh_command_queue();
    const auto zero_coord = distributed::MeshCoordinate(0, 0);
    const auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    const CoreCoord core = {0, 0};

    auto make_dram = [&](uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            distributed::DeviceLocalBufferConfig{
                .page_size = bytes, .buffer_type = BufferType::DRAM, .bottom_up = false},
            mesh_device.get());
    };
    auto make_cb = [&](uint32_t idx, SfpuTileSpec spec, uint32_t tiles) {
        const auto cb = static_cast<CBIndex>(idx);
        CreateCircularBuffer(
            program_,
            core,
            CircularBufferConfig(tiles * spec.bytes(), {{cb, tt::DataFormat::Float32}})
                .set_page_size(cb, spec.bytes())
                .set_tile_dims(cb, Tile({spec.r, spec.c})));
    };

    std::vector<std::shared_ptr<distributed::MeshBuffer>> in_buffers;
    std::vector<uint32_t> reader_args;
    for (uint32_t i = 0; i < kNumIn; i++) {
        const SfpuOperand& op = i < c.inputs.size() ? c.inputs[i] : SfpuOperand{kT32};
        const uint32_t ntiles = std::max<uint32_t>(op.ntiles, 1);
        make_cb(i, op.spec, ntiles);
        auto buf = make_dram(std::max<uint32_t>(op.data.size() * 4, op.spec.bytes()));
        if (!op.data.empty()) {
            std::vector<uint32_t> data = op.data;
            distributed::WriteShard(cq, buf, data, zero_coord);
        }
        in_buffers.push_back(buf);
        reader_args.insert(reader_args.end(), {uint32_t(buf->address()), kDramBankId, op.ntiles});
    }
    const uint32_t out_tiles = c.reps * c.out_tiles_per_rep;
    make_cb(16, c.out_spec, out_tiles);
    auto out_buffer = make_dram(out_tiles * c.out_spec.bytes());
    auto stats_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kStatsBytes},
        distributed::DeviceLocalBufferConfig{.page_size = kStatsBytes, .buffer_type = BufferType::L1},
        mesh_device.get());

    auto reader = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_gdn_decode_sfpu.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {kNumIn}});
    auto writer = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    // Every input CB is only ever read by copy_tile into DEST: exact fp32 copies.
    std::vector<UnpackToDestMode> utd(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t i = 0; i < kNumIn; i++) {
        utd[i] = UnpackToDestMode::UnpackToDestFp32;
    }
    auto compute = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/gdn_decode_sfpu.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = utd,
            .math_approx_mode = false,
            .compile_args = {c.variant, c.mode, c.reps, VT}});

    SetRuntimeArgs(program_, reader, core, reader_args);
    SetRuntimeArgs(program_, writer, core, {uint32_t(out_buffer->address()), kDramBankId, out_tiles});
    SetRuntimeArgs(program_, compute, core, {uint32_t(stats_buffer->address())});
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> out;
    distributed::ReadShard(cq, out, out_buffer, zero_coord);
    SfpuRunResult res;
    const uint32_t first = (c.reps - 1) * c.out_tiles_per_rep;
    for (uint32_t t = first; t < out_tiles; t++) {
        res.tiles.push_back(sfpu_untilize(out.data() + t * c.out_spec.elems(), c.out_spec));
    }
    for (uint32_t t = 0; t < c.out_tiles_per_rep; t++) {
        res.first.push_back(sfpu_untilize(out.data() + t * c.out_spec.elems(), c.out_spec));
    }
    std::vector<uint32_t> stats;
    detail::ReadFromDeviceL1(
        mesh_device->get_devices().at(0), core, uint32_t(stats_buffer->address()), kStatsBytes, stats);
    res.cycles = stats.size() >= 2 ? stats[1] - stats[0] : 0;
    return res;
}

// ---- checks -------------------------------------------------------------------------------------------------------
bool bits_equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (std::bit_cast<uint32_t>(a[i]) != std::bit_cast<uint32_t>(b[i])) {
            return false;
        }
    }
    return true;
}

bool bits_equal(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!bits_equal(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

bool all_finite(const std::vector<float>& v) {
    return std::all_of(v.begin(), v.end(), [](float f) { return std::isfinite(f); });
}

// rows [r0, r1) of a row-major tile are all +0.0 bitwise
bool rows_are_pos_zero(const std::vector<float>& t, uint32_t cols, uint32_t r0, uint32_t r1) {
    for (uint32_t r = r0; r < r1; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            if (std::bit_cast<uint32_t>(t[r * cols + c]) != 0u) {
                return false;
            }
        }
    }
    return true;
}

std::vector<float> row_of(const std::vector<float>& t, uint32_t cols, uint32_t r) {
    return {t.begin() + r * cols, t.begin() + (r + 1) * cols};
}

double max_rel_err(const std::vector<float>& got, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        num = std::max(num, std::fabs(double(got[i]) - ref[i]));
        den = std::max(den, std::fabs(ref[i]));
    }
    return den > 0 ? num / den : num;
}

int32_t ulp_diff(float a, float b) {
    const auto to_ord = [](float f) {
        const int32_t i = std::bit_cast<int32_t>(f);
        return i < 0 ? std::numeric_limits<int32_t>::min() - i : i;
    };
    return std::abs(to_ord(a) - to_ord(b));
}

// value = 100 * row + col over a 16x32 tile (the probe input)
std::vector<float> ramp_tile(SfpuTileSpec spec) {
    std::vector<float> rm(spec.elems());
    for (uint32_t r = 0; r < spec.r; r++) {
        for (uint32_t c = 0; c < spec.c; c++) {
            rm[r * spec.c + c] = float(100 * r + c);
        }
    }
    return rm;
}

std::vector<float> const_tile(SfpuTileSpec spec, float v) { return std::vector<float>(spec.elems(), v); }

std::vector<float> random_tile(SfpuTileSpec spec, std::mt19937& gen, float scale) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> rm(spec.elems());
    for (auto& v : rm) {
        v = scale * nd(gen);
    }
    return rm;
}

// ---- (a) the DEST map and the row moves ----------------------------------------------------------------------------
TEST_F(LLKBlackholeSingleCardFixture, GdnDecodeSfpuDestRowMap) {
    // 1. The raw (vector, lane) identity of every element, for a 32x32 and a 16x32 tile.
    for (const SfpuTileSpec spec : {kT32, kT16x32}) {
        auto res =
            run(this->devices_.at(0),
                {.variant = 1, .inputs = {one_tile(ramp_tile(spec), spec)}, .out_spec = spec, .out_tiles_per_rep = 1});
        const auto& t = res.tiles.at(0);
        // decode: value = 1000 * vector + 2 * lane
        auto vec_of = [&](uint32_t r, uint32_t c) { return uint32_t(t[r * spec.c + c]) / 1000; };
        auto lane_of = [&](uint32_t r, uint32_t c) { return (uint32_t(t[r * spec.c + c]) % 1000) / 2; };
        // Hypothesis H (LLK column-broadcast geometry): face f = (r / 16) * (c_faces) + c / 16 at vectors 8f..8f+7;
        // vector 8f + 2 * ((r % 16) / 4) + (c % 2); lane 8 * (r % 4) + (c % 16) / 2.
        // Alternative A (the contract's): vector 8f + (r % 16) / 2; lane 16 * (r % 2) + c % 16.
        uint32_t ok_h = 0, ok_a = 0;
        const uint32_t c_faces = spec.c / 16;
        for (uint32_t r = 0; r < spec.r; r++) {
            for (uint32_t c = 0; c < spec.c; c++) {
                const uint32_t f = (r / 16) * c_faces + c / 16;
                const uint32_t vh = 8 * f + 2 * ((r % 16) / 4) + (c % 2), lh = 8 * (r % 4) + (c % 16) / 2;
                const uint32_t va = 8 * f + (r % 16) / 2, la = 16 * (r % 2) + c % 16;
                ok_h += (vec_of(r, c) == vh && lane_of(r, c) == lh);
                ok_a += (vec_of(r, c) == va && lane_of(r, c) == la);
            }
        }
        log_info(
            tt::LogTest,
            "DEST map, {}x{} fp32 tile: elements matching the 4-row-band / even-odd-column geometry {} / {}, matching "
            "the contract's 2-rows-per-vector geometry {} / {}",
            spec.r,
            spec.c,
            ok_h,
            spec.elems(),
            ok_a,
            spec.elems());
        // Print where rows 0..3 of face 0 live (vector, lane range per column parity) so the mapping is on record.
        for (uint32_t r = 0; r < 4; r++) {
            log_info(
                tt::LogTest,
                "  row {}: col 0 -> (vector {}, lane {}), col 1 -> (vector {}, lane {}), col 2 -> (vector {}, lane "
                "{}), col 15 -> (vector {}, lane {}), col 16 -> (vector {}, lane {})",
                r,
                vec_of(r, 0),
                lane_of(r, 0),
                vec_of(r, 1),
                lane_of(r, 1),
                vec_of(r, 2),
                lane_of(r, 2),
                vec_of(r, 15),
                lane_of(r, 15),
                vec_of(r, 16),
                lane_of(r, 16));
        }
        EXPECT_EQ(ok_h, spec.elems()) << "the fp32 DEST geometry is not the 4-row-band / even-odd-column one";
    }

    // 2. The row moves built on that map: 100 * row + col in a 16x32 tile.
    const auto in = ramp_tile(kT16x32);
    auto res =
        run(this->devices_.at(0),
            {.variant = 2, .inputs = {one_tile(in, kT16x32)}, .out_spec = kT16x32, .out_tiles_per_rep = 4});
    ASSERT_EQ(res.tiles.size(), 4u);
    const auto& moved = res.tiles[0];
    const auto& z1 = res.tiles[1];
    const auto& z2 = res.tiles[2];
    const auto& copied = res.tiles[3];
    const auto in_row2 = row_of(in, 32, 2), in_row0 = row_of(in, 32, 0), in_row1 = row_of(in, 32, 1);
    EXPECT_TRUE(bits_equal(row_of(moved, 32, 0), in_row2)) << "move_row2_to_row0: row 0 is not the old row 2";
    EXPECT_TRUE(rows_are_pos_zero(moved, 32, 1, 16)) << "move_row2_to_row0: rows 1..15 are not zero";
    EXPECT_TRUE(bits_equal(row_of(z1, 32, 0), in_row0)) << "zero_rows_from_1: row 0 changed";
    EXPECT_TRUE(rows_are_pos_zero(z1, 32, 1, 16)) << "zero_rows_from_1: rows 1..15 are not zero";
    EXPECT_TRUE(bits_equal(row_of(z2, 32, 0), in_row0) && bits_equal(row_of(z2, 32, 1), in_row1))
        << "zero_rows_from_2: rows 0, 1 changed";
    EXPECT_TRUE(rows_are_pos_zero(z2, 32, 2, 16)) << "zero_rows_from_2: rows 2..15 are not zero";
    EXPECT_TRUE(bits_equal(copied, in)) << "copy_dst<16> is not bit-exact";
    EXPECT_TRUE(bits_equal(res.first, res.tiles)) << "row-move results differ between the first and the last rep";
    log_info(
        tt::LogTest,
        "row moves on a 16x32 tile (value 100*row+col): move_row2_to_row0 row 0 = {} {} {} .. {}, rows 1..15 zero {}; "
        "zero_rows_from_1 ok {}; zero_rows_from_2 ok {}; copy_dst bit-exact {}; cycles per acquire (3 copies + 4 "
        "sfpi passes + 4 packs) {:.0f}",
        moved[0],
        moved[1],
        moved[2],
        moved[31],
        rows_are_pos_zero(moved, 32, 1, 16),
        bits_equal(row_of(z1, 32, 0), in_row0) && rows_are_pos_zero(z1, 32, 1, 16),
        bits_equal(row_of(z2, 32, 0), in_row0) && bits_equal(row_of(z2, 32, 1), in_row1) &&
            rows_are_pos_zero(z2, 32, 2, 16),
        bits_equal(copied, in),
        double(res.cycles) / kReps);
}

// ---- (b) DEST persistence ---------------------------------------------------------------------------------------
TEST_F(LLKBlackholeSingleCardFixture, GdnDecodeSfpuDestPersistence) {
    std::mt19937 gen(7);
    const auto A = random_tile(kT32, gen, 1.0f), B = random_tile(kT32, gen, 1.0f);
    auto res = run(
        this->devices_.at(0),
        {.variant = 3, .inputs = {one_tile(A, kT32), one_tile(B, kT32)}, .out_spec = kT32, .out_tiles_per_rep = 11});
    ASSERT_EQ(res.tiles.size(), 11u);
    const bool half0 = bits_equal(res.tiles[6], A), half1 = bits_equal(res.tiles[7], A);
    const bool packed_slot_kept = bits_equal(res.tiles[10], A);
    const bool zero6 = rows_are_pos_zero(res.tiles[6], 32, 0, 32), zero7 = rows_are_pos_zero(res.tiles[7], 32, 0, 32);
    const bool zero10 = rows_are_pos_zero(res.tiles[10], 32, 0, 32);
    const bool slot0_fine = bits_equal(res.tiles[0], B) && bits_equal(res.tiles[5], B) && bits_equal(res.tiles[9], B);
    const bool same_acquire = bits_equal(res.tiles[8], A);
    EXPECT_TRUE(slot0_fine) << "the control slot (slot 0 <- B, packed each acquire) is wrong";
    EXPECT_TRUE(same_acquire) << "a slot written and packed in the same acquire is wrong";
    log_info(
        tt::LogTest,
        "DEST persistence (fp32, half-sync): slot 3 loaded once in each half, four acquires later it packs as the "
        "tile: half of acquire #0 {}, half of acquire #1 {} (as all zeros: {}, {}); a slot packed in acquire #8 packs "
        "as its tile again in acquire #10 {} (as all zeros: {}); written-and-packed in one acquire {}; results "
        "identical in the first and the last rep {}",
        half0,
        half1,
        zero6,
        zero7,
        packed_slot_kept,
        zero10,
        same_acquire,
        bits_equal(res.first, res.tiles));
    // The answer is no: tile_regs_release() clears the released half (the LLK pack release's ZEROACC CLR_HALF,
    // clear_dest = true by default in llk_pack_dest_section_done), so a slot that is not rewritten in the acquire
    // that packs it comes back as zeros, in both halves. (The clear invalidates rows; the SFPU can still read the
    // stale words, which is why a "resident constant" variant of the vector pass appeared to work. Not a feature.)
    EXPECT_TRUE(zero6 && zero7 && zero10) << "a slot not rewritten in its acquire did not pack as zeros";
    EXPECT_FALSE(half0 || half1 || packed_slot_kept) << "DEST persistence across acquires changed: re-check the LLK";
}

// ---- (c) the vector pass -----------------------------------------------------------------------------------------
struct VectorInputs {
    std::vector<float> rk, rq, v, e, beta_v, c_v;  // rk, rq, v: [VT*32]; e, beta, c scalars repeated for clarity
    float beta, c;
    std::vector<float> g;  // [32] the g column (for the log only)
};

VectorInputs make_vector_inputs(uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    VectorInputs x;
    const uint32_t VD = VT * 32;
    x.rk.resize(VD);
    x.rq.resize(VD);
    x.v.resize(VD);
    for (uint32_t i = 0; i < VD; i++) {
        x.rk[i] = 0.1f * nd(gen);
        x.rq[i] = 0.1f * nd(gen);
        x.v[i] = nd(gen);
    }
    const float g = -3.0f * ud(gen);
    x.e = {std::exp(g)};
    x.beta = 0.1f + 0.9f * ud(gen);
    x.c = 0.5f * nd(gen);
    return x;
}

struct VectorOperands {
    SfpuOperand r16{kT16x32}, v16{kT16x32}, ecol32{kT32}, bcol16{kT16x32}, ccol32{kT32};
    SfpuOperand ebc16{kT16x32};  // one constants tile: column 0 rows 0..3 = e, rows 4..7 = beta, rows 8..11 = c
};

VectorOperands vector_operands(const VectorInputs& x, float pad) {
    VectorOperands ops;
    for (uint32_t j = 0; j < VT; j++) {
        std::vector<float> r(kT16x32.elems(), pad), v(kT16x32.elems(), pad);
        for (uint32_t c = 0; c < 32; c++) {
            r[0 * 32 + c] = x.rk[j * 32 + c];
            r[2 * 32 + c] = x.rq[j * 32 + c];
            v[0 * 32 + c] = x.v[j * 32 + c];
        }
        ops.r16.add(r);
        ops.v16.add(v);
    }
    std::vector<float> e(kT32.elems(), 1.0f);
    for (uint32_t r = 0; r < 32; r++) {
        e[r * 32 + 0] = x.e[0];
    }
    ops.ecol32.add(e);
    std::vector<float> b(kT16x32.elems(), 0.0f);
    for (uint32_t r = 0; r < 16; r++) {
        b[r * 32 + 0] = x.beta;
    }
    ops.bcol16.add(b);
    ops.ccol32.add(const_tile(kT32, x.c));
    std::vector<float> ebc(kT16x32.elems(), 0.0f);
    for (uint32_t r = 0; r < 4; r++) {
        ebc[(r + 0) * 32 + 0] = x.e[0];
        ebc[(r + 4) * 32 + 0] = x.beta;
        ebc[(r + 8) * 32 + 0] = x.c;
    }
    ops.ebc16.add(ebc);
    return ops;
}

// mode 3 reads its only constants tile (ebc16) from c_3 in place of bcol16; c_2 / c_4 are then unused
SfpuRunConfig vector_config(const VectorOperands& ops, uint32_t mode) {
    return {
        .variant = 4,
        .mode = mode,
        .inputs = {ops.r16, ops.v16, ops.ecol32, mode == 3 ? ops.ebc16 : ops.bcol16, ops.ccol32},
        .out_spec = kT16x32,
        .out_tiles_per_rep = 2 * VT};
}

TEST_F(LLKBlackholeSingleCardFixture, GdnDecodeSfpuVectorPass) {
    const VectorInputs x = make_vector_inputs(11);
    const auto ops = vector_operands(x, 0.0f);
    const auto ops_nan = vector_operands(x, kNaN);
    std::vector<double> ref_u, ref_o;
    for (uint32_t i = 0; i < VT * 32; i++) {
        const double e = x.e[0];
        const double u = double(x.beta) * (double(x.v[i]) - e * double(x.rk[i]));
        ref_u.push_back(u);
        ref_o.push_back(e * double(x.rq[i]) + double(x.c) * u);
    }

    auto per_j = run(this->devices_.at(0), vector_config(ops, 0));
    auto band0 = run(this->devices_.at(0), vector_config(ops, 1));
    auto full_tile = run(this->devices_.at(0), vector_config(ops, 2));
    auto one_const = run(this->devices_.at(0), vector_config(ops, 3));
    auto poisoned = run(this->devices_.at(0), vector_config(ops_nan, 0));
    ASSERT_EQ(per_j.tiles.size(), 2 * VT);
    EXPECT_TRUE(bits_equal(one_const.tiles, per_j.tiles))
        << "the one-constants-tile variant (mul_col_bcast_band0) differs bitwise";

    auto rows0 = [](const std::vector<std::vector<float>>& tiles, uint32_t parity) {
        std::vector<float> out;
        for (uint32_t j = 0; j < VT; j++) {
            const auto& t = tiles[2 * j + parity];
            out.insert(out.end(), t.begin(), t.begin() + 32);
        }
        return out;
    };
    const auto u = rows0(per_j.tiles, 0), o = rows0(per_j.tiles, 1);
    const double err_u = max_rel_err(u, ref_u), err_o = max_rel_err(o, ref_o);
    bool u_pad_zero = true, o_pad_zero = true;
    for (uint32_t j = 0; j < VT; j++) {
        u_pad_zero = u_pad_zero && rows_are_pos_zero(per_j.tiles[2 * j], 32, 1, 16);
        o_pad_zero = o_pad_zero && rows_are_pos_zero(per_j.tiles[2 * j + 1], 32, 1, 16);
    }
    const bool nan_u_row0_clean = bits_equal(rows0(poisoned.tiles, 0), u);
    const bool nan_o_row0_clean = bits_equal(rows0(poisoned.tiles, 1), o);
    bool nan_pad_zero = true;
    for (uint32_t j = 0; j < VT; j++) {
        nan_pad_zero = nan_pad_zero && rows_are_pos_zero(poisoned.tiles[2 * j], 32, 1, 16) &&
                       rows_are_pos_zero(poisoned.tiles[2 * j + 1], 32, 1, 16);
    }

    EXPECT_LT(err_u, 5e-3);
    EXPECT_LT(err_o, 5e-3);
    EXPECT_TRUE(u_pad_zero) << "u16 rows 1..15 are not exactly zero";
    EXPECT_TRUE(o_pad_zero) << "out16 rows 1..15 are not zero";
    EXPECT_TRUE(bits_equal(full_tile.tiles, per_j.tiles)) << "16-row helpers differ bitwise from the full-tile API";
    EXPECT_TRUE(bits_equal(band0.tiles, per_j.tiles)) << "4-row (band 0) helpers differ bitwise from the 16-row ones";
    EXPECT_TRUE(bits_equal(per_j.first, per_j.tiles)) << "the vector pass is not repeatable across reps";
    EXPECT_TRUE(nan_u_row0_clean && nan_o_row0_clean) << "NaN in the padding rows of r16 / v16 reached row 0";
    EXPECT_TRUE(nan_pad_zero) << "NaN in the padding rows of r16 / v16 survived into the output padding";
    log_info(
        tt::LogTest,
        "vector pass, VT {}: max rel err vs double u {:.2e} o {:.2e}; u16 rows 1..15 exactly zero {}; out16 rows "
        "1..15 zero {}; 16-row helpers bit-equal to the full-tile API {}; 4-row (band 0) helpers bit-equal {}; "
        "repeatable {}; NaN in rows 1, 3..15 of r16 and v16: u row 0 unchanged {}, o row 0 unchanged {}, output "
        "padding still zero {}",
        VT,
        err_u,
        err_o,
        u_pad_zero,
        o_pad_zero,
        bits_equal(full_tile.tiles, per_j.tiles),
        bits_equal(band0.tiles, per_j.tiles),
        bits_equal(per_j.first, per_j.tiles),
        nan_u_row0_clean,
        nan_o_row0_clean,
        nan_pad_zero);
    log_info(
        tt::LogTest,
        "vector pass cycles per token (VT {} tiles, 2 packs per tile; gate <= 2400): 5 copy_tiles per tile with "
        "16-row helpers {:.0f} ({:.0f} per tile), with 4-row (band 0) helpers {:.0f} ({:.0f} per tile); 3 copy_tiles "
        "per tile (one constants tile, mul_col_bcast_band0) {:.0f} ({:.0f} per tile); full-tile (32-row) compute-API "
        "helpers {:.0f} ({:.0f} per tile)",
        VT,
        double(per_j.cycles) / kReps,
        double(per_j.cycles) / (kReps * VT),
        double(band0.cycles) / kReps,
        double(band0.cycles) / (kReps * VT),
        double(one_const.cycles) / kReps,
        double(one_const.cycles) / (kReps * VT),
        double(full_tile.cycles) / kReps,
        double(full_tile.cycles) / (kReps * VT));
}

// ---- (d) exact exp on a column tile ------------------------------------------------------------------------------
TEST_F(LLKBlackholeSingleCardFixture, GdnDecodeSfpuExpExact) {
    std::mt19937 gen(23);
    std::uniform_real_distribution<float> ud(-6.0f, 0.0f);
    std::vector<float> g(kT32.elems(), 0.0f), gcol(32);
    for (uint32_t r = 0; r < 32; r++) {
        gcol[r] = ud(gen);
        g[r * 32 + 0] = gcol[r];
    }
    const auto data = random_tile(kT32, gen, 1.0f);
    std::vector<float> col0_ref;  // column 0 of the whole-tile form, for the bit-equality of the other forms
    for (uint32_t mode : {0u, 1u, 2u}) {
        auto res =
            run(this->devices_.at(0),
                {.variant = 5,
                 .mode = mode,
                 .inputs = {one_tile(g, kT32), one_tile(data, kT32)},
                 .out_spec = kT32,
                 .out_tiles_per_rep = 2});
        ASSERT_EQ(res.tiles.size(), 2u);
        const auto& e = res.tiles[0];
        const auto& scaled = res.tiles[1];
        int32_t max_ulp = 0;
        uint32_t ones_elsewhere = 0, elsewhere = 0;
        std::vector<float> col0(32);
        for (uint32_t r = 0; r < 32; r++) {
            col0[r] = e[r * 32 + 0];
            max_ulp = std::max(max_ulp, ulp_diff(e[r * 32 + 0], std::exp(gcol[r])));
            for (uint32_t c = 1; c < 32; c++) {
                // mode 0 computes every column (exp(0) = 1), mode 1 faces 0 / 2 only (c < 16); mode 2 runs the
                // eight column-0 vectors, which also carry the other even columns of faces 0 / 2 (exp(0) = 1 there),
                // and leaves the odd columns and faces 1 / 3 as they were (g's zeros)
                if (mode == 2) {
                    elsewhere++;
                    ones_elsewhere += e[r * 32 + c] == ((c < 16 && c % 2 == 0) ? 1.0f : 0.0f);
                } else if (mode == 0 || c < 16) {
                    elsewhere++;
                    ones_elsewhere += e[r * 32 + c] == 1.0f;
                }
            }
        }
        if (mode == 0) {
            col0_ref = col0;
        }
        const bool col0_same = bits_equal(col0, col0_ref);
        // what mul_col_bcast applied: scaled[r][c] == fp32(data[r][c] * e[r][0])
        uint32_t exact = 0;
        int32_t max_ulp_mul = 0;
        for (uint32_t r = 0; r < 32; r++) {
            for (uint32_t c = 0; c < 32; c++) {
                const float want = data[r * 32 + c] * e[r * 32 + 0];
                exact += std::bit_cast<uint32_t>(scaled[r * 32 + c]) == std::bit_cast<uint32_t>(want);
                max_ulp_mul = std::max(max_ulp_mul, ulp_diff(scaled[r * 32 + c], want));
            }
        }
        EXPECT_LE(max_ulp, 2) << "exact exp differs from std::exp by more than 2 ulp";
        EXPECT_EQ(ones_elsewhere, elsewhere) << "the other columns are not what the form should leave there";
        EXPECT_LE(max_ulp_mul, 1) << "mul_col_bcast did not apply column 0 of the exp tile";
        EXPECT_TRUE(col0_same) << "column 0 differs bitwise between the exp forms";
        EXPECT_TRUE(bits_equal(res.first, res.tiles));
        log_info(
            tt::LogTest,
            "exp exact, {}: column 0 (g in [-6, 0], 32 rows) vs std::exp max {} ulp (gate <= 2), bit-equal to the "
            "whole-tile form {}; other columns as expected ({}) on {} / {}; data (.) col(exp) equals fp32(data * e_r) "
            "bitwise on {} / 1024 (max {} ulp); repeatable {}; cycles per rep (2 copies + exp + col bcast + 2 packs) "
            "{:.0f}",
            mode == 0   ? "whole tile (VectorMode::RC)"
            : mode == 1 ? "faces 0 and 2 (VectorMode::C)"
                        : "column 0 only (exp_col0_exact)",
            max_ulp,
            col0_same,
            mode == 2 ? "exp(0) == 1 on the even columns of faces 0 / 2, untouched zeros elsewhere" : "exp(0) == 1",
            ones_elsewhere,
            elsewhere,
            exact,
            max_ulp_mul,
            bits_equal(res.first, res.tiles),
            double(res.cycles) / kReps);
    }
}

}  // namespace unit_tests::compute::gdn_decode_sfpu
}  // namespace tt::tt_metal
