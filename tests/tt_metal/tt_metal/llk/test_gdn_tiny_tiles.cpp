// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Spike A of the GDN decode v2 plan: do the tiny-tile matmul shapes the two-pass decode kernel rests on work on
// Blackhole from a hand-written compute kernel, with the same bits as the padded 32x32 forms, without NaN leaking out
// of the padding, and in how many cycles?
//   two-row read   [q; k] (2x32 tiles) x S (32x32)          vs  [q; k] padded to 32x32
//   rank-1 update  S (DEST seed) + k^T (32x16) x u (16x32)  vs  k^T, u padded to 32x32
//   scalar         [q; k] (2x32) x k in column 0 of 32x32   ->  (q.k, k.k)
//   copy seed      copy_tile of an fp32 S tile into DEST, with and without UnpackToDestFp32
// One core, K = V = 128 (4 x 4 tiles), HiFi4, fp32 DEST. The kernel is test_kernels/compute/gdn_tiny_tiles.cpp.

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
namespace unit_tests::compute::gdn_tiny {

constexpr uint32_t KT = 4, VT = 4, KD = KT * 32, VD = VT * 32;
constexpr uint32_t kDramBankId = 0;
constexpr uint32_t kStatsBytes = 64;
constexpr uint32_t kReps = 16;
const float kNaN = std::numeric_limits<float>::quiet_NaN();

struct TileSpec {
    uint32_t r, c;
    uint32_t fr() const { return std::min<uint32_t>(r, 16); }
    uint32_t fc() const { return std::min<uint32_t>(c, 16); }
    uint32_t elems() const { return r * c; }
    uint32_t bytes() const { return r * c * 4; }
};
constexpr TileSpec T32{32, 32}, T2x32{2, 32}, T32x16{32, 16}, T16x32{16, 32};

// Row-major [r x c] -> device tile order: faces row-major over the face grid, rows within a face.
std::vector<uint32_t> tilize(const std::vector<float>& rm, TileSpec t) {
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

std::vector<float> untilize(const uint32_t* tile, TileSpec t) {
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

struct Operand {
    TileSpec spec{32, 32};
    std::vector<uint32_t> data;  // tilized tiles, concatenated
    uint32_t ntiles = 0;
    void add(const std::vector<float>& rm) {
        auto t = tilize(rm, spec);
        data.insert(data.end(), t.begin(), t.end());
        ntiles++;
    }
};

struct Inputs {
    std::vector<float> S;  // [KD x VD]
    std::vector<float> q, k, u;
};

Inputs make_inputs(uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    Inputs x;
    x.S.resize(KD * VD);
    for (auto& v : x.S) {
        v = 0.05f * nd(gen);
    }
    x.q.resize(KD);
    x.k.resize(KD);
    x.u.resize(VD);
    double nq = 0, nk = 0;
    for (uint32_t i = 0; i < KD; i++) {
        x.q[i] = nd(gen);
        x.k[i] = nd(gen);
        nq += double(x.q[i]) * x.q[i];
        nk += double(x.k[i]) * x.k[i];
    }
    for (uint32_t i = 0; i < KD; i++) {
        x.q[i] = float(x.q[i] / std::sqrt(nq) / std::sqrt(double(KD)));
        x.k[i] = float(x.k[i] / std::sqrt(nk));
    }
    for (auto& v : x.u) {
        v = 0.1f * nd(gen);
    }
    return x;
}

// ---- operand builders ---------------------------------------------------------------------------------------------
Operand s_tiles(const Inputs& x) {
    Operand op{T32};
    for (uint32_t i = 0; i < KT; i++) {
        for (uint32_t j = 0; j < VT; j++) {
            std::vector<float> rm(32 * 32);
            for (uint32_t r = 0; r < 32; r++) {
                for (uint32_t c = 0; c < 32; c++) {
                    rm[r * 32 + c] = x.S[(i * 32 + r) * VD + j * 32 + c];
                }
            }
            op.add(rm);
        }
    }
    return op;
}

Operand qk_tiles(const Inputs& x, uint32_t rows, float fill) {
    Operand op{TileSpec{rows, 32}};
    for (uint32_t i = 0; i < KT; i++) {
        std::vector<float> rm(op.spec.elems(), fill);
        for (uint32_t c = 0; c < 32; c++) {
            rm[0 * 32 + c] = x.q[i * 32 + c];
            rm[1 * 32 + c] = x.k[i * 32 + c];
        }
        op.add(rm);
    }
    return op;
}

Operand kT_tiles(const Inputs& x, bool padded, float fill) {
    Operand op{padded ? T32 : T32x16};
    for (uint32_t i = 0; i < KT; i++) {
        std::vector<float> rm(op.spec.elems(), fill);
        for (uint32_t r = 0; r < 32; r++) {
            rm[r * op.spec.c + 0] = x.k[i * 32 + r];
        }
        op.add(rm);
    }
    return op;
}

Operand u_tiles(const Inputs& x, bool padded, float fill) {
    Operand op{padded ? T32 : T16x32};
    for (uint32_t j = 0; j < VT; j++) {
        std::vector<float> rm(op.spec.elems(), fill);
        for (uint32_t c = 0; c < 32; c++) {
            rm[0 * 32 + c] = x.u[j * 32 + c];
        }
        op.add(rm);
    }
    return op;
}

Operand empty_operand() {
    Operand op{T32};
    return op;
}

// ---- the program --------------------------------------------------------------------------------------------------
struct RunConfig {
    uint32_t variant;
    std::vector<Operand> inputs;  // CBs c_0, c_1, c_2
    TileSpec out_spec;
    uint32_t out_tiles_per_rep;
    bool utd_seed = true;
    uint32_t reps = kReps;
};

struct RunResult {
    std::vector<std::vector<float>> tiles;  // the last rep's output tiles, row-major each
    uint32_t cycles = 0;
};

RunResult run(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const RunConfig& c) {
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
    auto make_cb = [&](uint32_t idx, TileSpec spec, uint32_t tiles) {
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
    for (uint32_t i = 0; i < 3; i++) {
        const Operand& op = i < c.inputs.size() ? c.inputs[i] : Operand{T32};
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
    make_cb(16, c.out_spec, c.out_tiles_per_rep);
    auto out_buffer = make_dram(out_tiles * c.out_spec.bytes());
    auto stats_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kStatsBytes},
        distributed::DeviceLocalBufferConfig{.page_size = kStatsBytes, .buffer_type = BufferType::L1},
        mesh_device.get());

    auto reader = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_gdn_tiny_tiles.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {3u}});
    auto writer = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    std::vector<UnpackToDestMode> utd(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (c.utd_seed) {
        utd[2] = UnpackToDestMode::UnpackToDestFp32;
    }
    auto compute = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/gdn_tiny_tiles.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = utd,
            .math_approx_mode = false,
            .compile_args = {c.variant, c.reps, KT, VT}});

    SetRuntimeArgs(program_, reader, core, reader_args);
    SetRuntimeArgs(program_, writer, core, {uint32_t(out_buffer->address()), kDramBankId, out_tiles});
    SetRuntimeArgs(program_, compute, core, {uint32_t(stats_buffer->address())});
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> out;
    distributed::ReadShard(cq, out, out_buffer, zero_coord);
    RunResult res;
    const uint32_t first = (c.reps - 1) * c.out_tiles_per_rep;
    for (uint32_t t = first; t < out_tiles; t++) {
        res.tiles.push_back(untilize(out.data() + t * c.out_spec.elems(), c.out_spec));
    }
    std::vector<uint32_t> stats;
    detail::ReadFromDeviceL1(
        mesh_device->get_devices().at(0), core, uint32_t(stats_buffer->address()), kStatsBytes, stats);
    res.cycles = stats.size() >= 2 ? stats[1] - stats[0] : 0;
    return res;
}

// ---- checks -------------------------------------------------------------------------------------------------------
bool all_finite(const std::vector<float>& v) {
    return std::all_of(v.begin(), v.end(), [](float f) { return std::isfinite(f); });
}

bool bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
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

double max_rel_err(const std::vector<float>& got, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        num = std::max(num, std::fabs(double(got[i]) - ref[i]));
        den = std::max(den, std::fabs(ref[i]));
    }
    return den > 0 ? num / den : num;
}

// the two live rows of the padded two-row read, concatenated over the VT output tiles
std::vector<float> rows01(const std::vector<std::vector<float>>& tiles, TileSpec spec) {
    std::vector<float> out;
    for (const auto& t : tiles) {
        out.insert(out.end(), t.begin(), t.begin() + 2 * spec.c);
    }
    return out;
}

std::vector<float> flatten(const std::vector<std::vector<float>>& tiles) {
    std::vector<float> out;
    for (const auto& t : tiles) {
        out.insert(out.end(), t.begin(), t.end());
    }
    return out;
}

TEST_F(LLKBlackholeSingleCardFixture, GdnTinyTilesTwoRowRead) {
    const Inputs x = make_inputs(1);
    const Operand S = s_tiles(x);
    // reference r[2][VD] = [q; k] . S in double
    std::vector<double> ref;
    for (uint32_t j = 0; j < VT; j++) {
        for (uint32_t row = 0; row < 2; row++) {
            for (uint32_t c = 0; c < 32; c++) {
                double acc = 0;
                for (uint32_t kk = 0; kk < KD; kk++) {
                    acc += double(row == 0 ? x.q[kk] : x.k[kk]) * x.S[kk * VD + j * 32 + c];
                }
                ref.push_back(acc);
            }
        }
    }
    auto padded =
        run(this->devices_.at(0),
            {.variant = 2, .inputs = {qk_tiles(x, 32, 0.0f), S}, .out_spec = T32, .out_tiles_per_rep = VT});
    const auto p_rows = rows01(padded.tiles, T32);
    const double err_p = max_rel_err(p_rows, ref);
    EXPECT_LT(err_p, 1e-2);
    log_info(
        tt::LogTest,
        "two-row read, padded 32x32 in0: max rel err vs double {:.2e}; cycles per 32x32-by-32x32 product {:.1f}, per "
        "output tile (KT products + pack) {:.0f}",
        err_p,
        double(padded.cycles) / (kReps * VT * KT),
        double(padded.cycles) / (kReps * VT));
    for (uint32_t rows : {2u, 4u, 8u, 16u}) {
        const TileSpec spec{rows, 32};
        auto tiny =
            run(this->devices_.at(0),
                {.variant = 1, .inputs = {qk_tiles(x, rows, 0.0f), S}, .out_spec = spec, .out_tiles_per_rep = VT});
        auto poisoned =
            run(this->devices_.at(0),
                {.variant = 1, .inputs = {qk_tiles(x, rows, kNaN), S}, .out_spec = spec, .out_tiles_per_rep = VT});
        const auto t_rows = rows01(tiny.tiles, spec);
        const auto n_rows = rows01(poisoned.tiles, spec);
        const bool ok = bit_equal(t_rows, p_rows);
        if (rows == 16) {
            EXPECT_TRUE(ok) << "16x32 in0 (the SDPA-decode shape) differs bitwise from the padded 32x32 in0";
        }
        log_info(
            tt::LogTest,
            "two-row read, {}x32 in0: bit-equal to padded {}; finite {}; NaN in the padding rows leaves the live rows "
            "unchanged {}; max rel err vs double {:.2e}; cycles per product {:.1f} (gate <= 24), per output tile "
            "{:.0f}",
            rows,
            ok,
            all_finite(t_rows),
            rows > 2 && bit_equal(t_rows, n_rows),
            max_rel_err(t_rows, ref),
            double(tiny.cycles) / (kReps * VT * KT),
            double(tiny.cycles) / (kReps * VT));
    }
}

TEST_F(LLKBlackholeSingleCardFixture, GdnTinyTilesRankOneUpdate) {
    const Inputs x = make_inputs(2);
    const Operand S = s_tiles(x);
    std::vector<double> ref;  // S + k^T u, tile order (i, j), row-major within the tile
    for (uint32_t i = 0; i < KT; i++) {
        for (uint32_t j = 0; j < VT; j++) {
            for (uint32_t r = 0; r < 32; r++) {
                for (uint32_t c = 0; c < 32; c++) {
                    const uint32_t kk = i * 32 + r, vv = j * 32 + c;
                    ref.push_back(double(x.S[kk * VD + vv]) + double(x.k[kk]) * x.u[vv]);
                }
            }
        }
    }
    auto tiny =
        run(this->devices_.at(0),
            {.variant = 3,
             .inputs = {kT_tiles(x, false, 0.0f), u_tiles(x, false, 0.0f), S},
             .out_spec = T32,
             .out_tiles_per_rep = KT * VT});
    auto padded =
        run(this->devices_.at(0),
            {.variant = 4,
             .inputs = {kT_tiles(x, true, 0.0f), u_tiles(x, true, 0.0f), S},
             .out_spec = T32,
             .out_tiles_per_rep = KT * VT});
    auto tiny_nan_k =
        run(this->devices_.at(0),
            {.variant = 3,
             .inputs = {kT_tiles(x, false, kNaN), u_tiles(x, false, 0.0f), S},
             .out_spec = T32,
             .out_tiles_per_rep = KT * VT});
    auto padded_nan =
        run(this->devices_.at(0),
            {.variant = 4,
             .inputs = {kT_tiles(x, true, kNaN), u_tiles(x, true, kNaN), S},
             .out_spec = T32,
             .out_tiles_per_rep = KT * VT});
    auto tiny_tf32_seed =
        run(this->devices_.at(0),
            {.variant = 3,
             .inputs = {kT_tiles(x, false, 0.0f), u_tiles(x, false, 0.0f), S},
             .out_spec = T32,
             .out_tiles_per_rep = KT * VT,
             .utd_seed = false});
    auto mm_only =
        run(this->devices_.at(0),
            {.variant = 7,
             .inputs = {kT_tiles(x, false, 0.0f), u_tiles(x, false, 0.0f), S},
             .out_spec = T32,
             .out_tiles_per_rep = KT * VT});
    auto mm_only_padded =
        run(this->devices_.at(0),
            {.variant = 7,
             .inputs = {kT_tiles(x, true, 0.0f), u_tiles(x, true, 0.0f), S},
             .out_spec = T32,
             .out_tiles_per_rep = KT * VT});
    auto copy_only =
        run(this->devices_.at(0),
            {.variant = 8,
             .inputs = {kT_tiles(x, false, 0.0f), u_tiles(x, false, 0.0f), S},
             .out_spec = T32,
             .out_tiles_per_rep = KT * VT});
    std::vector<double> ref_mm;  // k^T u alone
    for (uint32_t i = 0; i < KT; i++) {
        for (uint32_t j = 0; j < VT; j++) {
            for (uint32_t r = 0; r < 32; r++) {
                for (uint32_t c = 0; c < 32; c++) {
                    ref_mm.push_back(double(x.k[i * 32 + r]) * x.u[j * 32 + c]);
                }
            }
        }
    }
    const auto t = flatten(tiny.tiles), p = flatten(padded.tiles);
    log_info(
        tt::LogTest,
        "rank-1 decomposition, cycles per S tile in groups of 4: matmul alone tiny {:.1f} (err {:.2e}) padded {:.1f}; "
        "copy seed alone {:.1f}; copy + matmul {:.1f}; poisoned runs bit-equal to the clean one: k^T padding {}, both "
        "paddings {}",
        double(mm_only.cycles) / (kReps * KT * VT),
        max_rel_err(flatten(mm_only.tiles), ref_mm),
        double(mm_only_padded.cycles) / (kReps * KT * VT),
        double(copy_only.cycles) / (kReps * KT * VT),
        double(tiny.cycles) / (kReps * KT * VT),
        bit_equal(flatten(tiny_nan_k.tiles), t),
        bit_equal(flatten(padded_nan.tiles), p));
    EXPECT_TRUE(all_finite(t)) << "tiny rank-1 update produced non-finite values";
    EXPECT_TRUE(bit_equal(t, p)) << "tiny 32x16 x 16x32 differs bitwise from the padded 32x32 x 32x32";
    const double err_t = max_rel_err(t, ref), err_p = max_rel_err(p, ref);
    const double err_tf32 = max_rel_err(flatten(tiny_tf32_seed.tiles), ref);
    EXPECT_LT(err_t, 1e-2);
    log_info(
        tt::LogTest,
        "rank-1 update: tiny vs padded bit-equal {}; max rel err vs double tiny {:.2e} padded {:.2e}, with a tf32 seed "
        "(no UnpackToDestFp32) {:.2e}; NaN in k^T's padding columns (u's padding rows zero) leaks: {}; NaN in both "
        "paddings leaks: {}; cycles per S tile (copy seed + rank-1 + fp32 pack) tiny {:.1f} (gate <= 48 for the matmul "
        "alone) padded {:.1f}",
        bit_equal(t, p),
        err_t,
        err_p,
        err_tf32,
        !all_finite(flatten(tiny_nan_k.tiles)),
        !all_finite(flatten(padded_nan.tiles)),
        double(tiny.cycles) / (kReps * KT * VT),
        double(padded.cycles) / (kReps * KT * VT));
}

TEST_F(LLKBlackholeSingleCardFixture, GdnTinyTilesScalar) {
    const Inputs x = make_inputs(3);
    double qk = 0, kk = 0;
    for (uint32_t i = 0; i < KD; i++) {
        qk += double(x.q[i]) * x.k[i];
        kk += double(x.k[i]) * x.k[i];
    }
    for (uint32_t rows : {2u, 16u, 32u}) {
        auto res =
            run(this->devices_.at(0),
                {.variant = 5,
                 .inputs = {qk_tiles(x, rows, 0.0f), kT_tiles(x, true, 0.0f)},
                 .out_spec = TileSpec{rows, 32},
                 .out_tiles_per_rep = 1});
        const auto& t = res.tiles.at(0);
        const double e_qk = std::fabs(double(t[0]) - qk), e_kk = std::fabs(double(t[32]) - kk);
        if (rows >= 16) {
            EXPECT_LT(e_qk, 1e-2 * std::max(1.0, std::fabs(qk)));
            EXPECT_LT(e_kk, 1e-2 * std::max(1.0, std::fabs(kk)));
        }
        log_info(
            tt::LogTest,
            "scalar q.k, {}x32 in0 x (k in column 0 of 32x32): q.k {:.6f} (ref {:.6f}), k.k {:.6f} (ref {:.6f}); "
            "cycles for KT products + pack {:.0f}",
            rows,
            t[0],
            qk,
            t[32],
            kk,
            double(res.cycles) / kReps);
    }
}

TEST_F(LLKBlackholeSingleCardFixture, GdnTinyTilesCopySeed) {
    const Inputs x = make_inputs(4);
    const Operand S = s_tiles(x);
    auto exact =
        run(this->devices_.at(0),
            {.variant = 6,
             .inputs = {empty_operand(), empty_operand(), S},
             .out_spec = T32,
             .out_tiles_per_rep = KT * VT,
             .utd_seed = true});
    auto tf32 =
        run(this->devices_.at(0),
            {.variant = 6,
             .inputs = {empty_operand(), empty_operand(), S},
             .out_spec = T32,
             .out_tiles_per_rep = KT * VT,
             .utd_seed = false});
    const auto e = flatten(exact.tiles), f = flatten(tf32.tiles);
    uint32_t mism_e = 0, mism_f = 0;
    for (uint32_t i = 0; i < KD * VD; i++) {
        const uint32_t i_tile = i / (32 * 32), rem = i % (32 * 32);
        const uint32_t ti = i_tile / VT, tj = i_tile % VT, r = rem / 32, c = rem % 32;
        const float src = x.S[(ti * 32 + r) * VD + tj * 32 + c];
        mism_e += std::bit_cast<uint32_t>(e[i]) != std::bit_cast<uint32_t>(src);
        mism_f += std::bit_cast<uint32_t>(f[i]) != std::bit_cast<uint32_t>(src);
    }
    EXPECT_EQ(mism_e, 0u) << "copy_tile with UnpackToDestFp32 is not bit-exact";
    log_info(
        tt::LogTest,
        "copy seed: words changed by copy_tile with UnpackToDestFp32 {} / {}, without it {} / {} (tf32 truncation); "
        "cycles per tile (copy + fp32 pack) {:.1f} / {:.1f}",
        mism_e,
        KD * VD,
        mism_f,
        KD * VD,
        double(exact.cycles) / (kReps * KT * VT),
        double(tf32.cycles) / (kReps * KT * VT));
}

}  // namespace unit_tests::compute::gdn_tiny
}  // namespace tt::tt_metal
