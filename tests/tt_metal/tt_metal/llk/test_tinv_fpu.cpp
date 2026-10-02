// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
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
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "llk_device_fixture.hpp"
#include "test_golden_impls.hpp"

// Micro-benchmark of the GDN WY inverse T_inv = (I - negN)^-1 of one 32x32 fp32 tile (Blackhole only):
// today's invert_block (LLK rounds through L1), the SFPU triangle_solve_tile, and the fused FPU inverse of
// chunk_gdn_tinv_fpu.hpp in its Horner and squaring forms, plus the MOVD2A/MOVD2B probes the fused inverse
// rests on. One core runs reader_tinv_fpu -> tinv_fpu -> writer_unary; the MATH thread writes wall-clock
// timestamps per repetition into an L1 buffer the host reads back.

namespace tt::tt_metal {

namespace unit_tests::compute::tinv_fpu {

constexpr uint32_t kTileDim = 32;
constexpr uint32_t kTileHW = kTileDim * kTileDim;
constexpr uint32_t kTileBytes = kTileHW * sizeof(float);
constexpr uint32_t kDramBankId = 0;
constexpr uint32_t kKDim = 128;  // key width of the GDN prep inputs the regimes model
constexpr uint32_t kStatsBytes = 8192;
constexpr double kClockGHz = 1.35;  // Blackhole AICLK assumed for the microsecond columns

enum class Variant : uint32_t {
    InvertBlock = 0,
    Sfpu = 1,
    FpuHorner = 2,
    FpuSquare = 3,
    ProbeD2B = 4,
    ProbeD2A = 5,
    ProbeUnpackB = 6,
    ProbeUnpackA = 7,
    ProbeDummyValid = 8,
    ProbeSetDvalid = 9,
    ProbeBothMoved = 10,
    ProbeD2BFace = 11,
    ProbeD2BThenA = 12,
    ProbeFaceMm = 13,
    LlkRoundsHorner = 14,
    LlkRoundsHornerR = 15,
    FpuHornerR = 16,
};

const char* variant_name(Variant v) {
    switch (v) {
        case Variant::InvertBlock: return "invert_block";
        case Variant::Sfpu: return "sfpu_solve";
        case Variant::FpuHorner: return "fpu_horner";
        case Variant::FpuSquare: return "fpu_square";
        case Variant::ProbeD2B: return "probe_movd2b";
        case Variant::ProbeD2A: return "probe_movd2a";
        case Variant::ProbeUnpackB: return "probe_unpack_srcb";
        case Variant::ProbeUnpackA: return "probe_unpack_srca";
        case Variant::ProbeDummyValid: return "probe_dummy_valid";
        case Variant::ProbeSetDvalid: return "probe_setdvalid";
        case Variant::ProbeBothMoved: return "probe_both_moved";
        case Variant::ProbeD2BFace: return "probe_movd2b_face0";
        case Variant::ProbeD2BThenA: return "probe_movd2b_then_a_face0";
        case Variant::ProbeFaceMm: return "probe_face_mm";
        case Variant::LlkRoundsHorner: return "llk_rounds_horner";
        case Variant::LlkRoundsHornerR: return "llk_rounds_hornerR";
        case Variant::FpuHornerR: return "fpu_hornerR";
    }
    return "?";
}

enum class Regime { Typical, Hard, Adversarial, RandomFp32 };

const char* regime_name(Regime r) {
    switch (r) {
        case Regime::Typical: return "typical";
        case Regime::Hard: return "hard";
        case Regime::Adversarial: return "adversarial";
        case Regime::RandomFp32: return "random_fp32";
    }
    return "?";
}

struct RunConfig {
    Variant variant = Variant::FpuSquare;
    uint32_t num_in = 1;
    uint32_t reps = 1;
    bool stall = false;
    uint32_t nsrc = 0;
    bool split = false;
    bool hoist = true;  // GDN_HOIST_RECONFIG for invert_block, as the fused producer builds it
    uint32_t fmt = 3;   // Src format mask around the moves: 1 pin SrcA tf32, 2 pin SrcB tf32, 4 keep zero flag
};

struct RunResult {
    std::vector<std::vector<float>> tiles;  // row-major output tiles, num_in * reps
    std::vector<uint32_t> stamps;           // 4 timestamps per repetition
};

// --- stimulus ----------------------------------------------------------------------------------------------

std::vector<double> normal_vec(std::mt19937& rng, size_t n) {
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> v(n);
    for (auto& x : v) {
        x = nd(rng);
    }
    return v;
}

void normalize_rows(std::vector<double>& k, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        double s = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            s += k[r * cols + c] * k[r * cols + c];
        }
        const double inv = 1.0 / std::max(std::sqrt(s), 1e-12);
        for (size_t c = 0; c < cols; ++c) {
            k[r * cols + c] *= inv;
        }
    }
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double softplus(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }

// negN = -tril(beta_i (k_i . k_j) exp(decay_i - decay_j), -1), row-major fp32, after test_chunk_gdn_prims._tinv_inputs.
std::vector<float> make_negn(Regime regime, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<float> out(kTileHW, 0.0f);
    if (regime == Regime::RandomFp32) {
        // random mantissas and exponents: the move/unpack probes compare bits
        std::uniform_int_distribution<uint32_t> mant(0, (1u << 23) - 1);
        std::uniform_int_distribution<int> expo(-8, 8);
        std::bernoulli_distribution sign(0.5);
        for (auto& x : out) {
            const uint32_t bits =
                (sign(rng) ? 0x80000000u : 0u) | (static_cast<uint32_t>(127 + expo(rng)) << 23) | mant(rng);
            x = std::bit_cast<float>(bits);
        }
        return out;
    }
    std::vector<double> k(kTileDim * kKDim), beta(kTileDim), g(kTileDim);
    if (regime == Regime::Typical) {
        k = normal_vec(rng, k.size());
        normalize_rows(k, kTileDim, kKDim);
        for (uint32_t i = 0; i < kTileDim; ++i) {
            beta[i] = sigmoid(normal_vec(rng, 1)[0]);
        }
        for (uint32_t i = 0; i < kTileDim; ++i) {
            g[i] = -softplus(normal_vec(rng, 1)[0]) * 0.5;
        }
    } else if (regime == Regime::Hard) {
        std::vector<double> shared = normal_vec(rng, kKDim);
        normalize_rows(shared, 1, kKDim);
        std::vector<double> noise = normal_vec(rng, k.size());
        normalize_rows(noise, kTileDim, kKDim);
        for (uint32_t i = 0; i < kTileDim; ++i) {
            for (uint32_t c = 0; c < kKDim; ++c) {
                k[i * kKDim + c] = 2.0 * shared[c] + 0.5 * noise[i * kKDim + c];
            }
        }
        normalize_rows(k, kTileDim, kKDim);
        for (uint32_t i = 0; i < kTileDim; ++i) {
            beta[i] = sigmoid(normal_vec(rng, 1)[0] * 0.5 + 2.2);
        }
        for (uint32_t i = 0; i < kTileDim; ++i) {
            g[i] = -softplus(normal_vec(rng, 1)[0]) * 0.002;
        }
    } else {
        std::vector<double> shared = normal_vec(rng, kKDim);
        normalize_rows(shared, 1, kKDim);
        for (uint32_t i = 0; i < kTileDim; ++i) {
            for (uint32_t c = 0; c < kKDim; ++c) {
                k[i * kKDim + c] = shared[c];
            }
        }
        std::fill(beta.begin(), beta.end(), 0.999);
        std::fill(g.begin(), g.end(), 0.0);
    }
    std::vector<double> decay(kTileDim);
    double acc = 0.0;
    for (uint32_t i = 0; i < kTileDim; ++i) {
        acc += g[i];
        decay[i] = acc;
    }
    for (uint32_t i = 0; i < kTileDim; ++i) {
        for (uint32_t j = 0; j < i; ++j) {
            double dot = 0.0;
            for (uint32_t c = 0; c < kKDim; ++c) {
                dot += k[i * kKDim + c] * k[j * kKDim + c];
            }
            out[i * kTileDim + j] = static_cast<float>(-(beta[i] * dot * std::exp(decay[i] - decay[j])));
        }
    }
    return out;
}

// (I - negN)^-1 by forward substitution in double, on the fp32 negN the device receives.
std::vector<double> inverse_ref(const std::vector<float>& negn) {
    std::vector<double> x(kTileHW, 0.0);
    for (uint32_t r = 0; r < kTileDim; ++r) {
        for (uint32_t j = 0; j < kTileDim; ++j) {
            double a = (r == j) ? 1.0 : 0.0;
            for (uint32_t c = 0; c < r; ++c) {
                a += static_cast<double>(negn[r * kTileDim + c]) * x[c * kTileDim + j];
            }
            x[r * kTileDim + j] = a;
        }
    }
    return x;
}

struct Accuracy {
    double max_abs = 0.0, mean_abs = 0.0, max_ref = 0.0, mean_ref = 0.0;
    double res_max = 0.0, res_mean = 0.0;  // T (I - negN) - I
    bool finite = true;
    double rel_max() const { return max_abs / std::max(max_ref, 1e-300); }
    double rel_mean() const { return mean_abs / std::max(mean_ref, 1e-300); }
};

Accuracy measure(const std::vector<float>& negn, const std::vector<float>& t_dev) {
    Accuracy a;
    const auto ref = inverse_ref(negn);
    double sum_abs = 0.0, sum_ref = 0.0, sum_res = 0.0;
    for (uint32_t i = 0; i < kTileHW; ++i) {
        if (!std::isfinite(t_dev[i])) {
            a.finite = false;
        }
        const double e = std::fabs(static_cast<double>(t_dev[i]) - ref[i]);
        a.max_abs = std::max(a.max_abs, e);
        sum_abs += e;
        a.max_ref = std::max(a.max_ref, std::fabs(ref[i]));
        sum_ref += std::fabs(ref[i]);
    }
    for (uint32_t r = 0; r < kTileDim; ++r) {
        for (uint32_t j = 0; j < kTileDim; ++j) {
            // R[r][j] = sum_c T[r][c] (I - negN)[c][j] - I[r][j]
            double s = -((r == j) ? 1.0 : 0.0);
            for (uint32_t c = 0; c < kTileDim; ++c) {
                const double l = (c == j ? 1.0 : 0.0) - static_cast<double>(negn[c * kTileDim + j]);
                s += static_cast<double>(t_dev[r * kTileDim + c]) * l;
            }
            a.res_max = std::max(a.res_max, std::fabs(s));
            sum_res += std::fabs(s);
        }
    }
    a.mean_abs = sum_abs / kTileHW;
    a.mean_ref = sum_ref / kTileHW;
    a.res_mean = sum_res / kTileHW;
    return a;
}

std::vector<uint32_t> fp32_as_u32(const std::vector<float>& in) {
    std::vector<uint32_t> out(in.size());
    std::memcpy(out.data(), in.data(), in.size() * sizeof(float));
    return out;
}

std::vector<float> identity_tile() {
    std::vector<float> eye(kTileHW, 0.0f);
    for (uint32_t i = 0; i < kTileDim; ++i) {
        eye[i * kTileDim + i] = 1.0f;
    }
    return eye;
}

// Qtl, Qbr, Q10 (bottom-left), as chunk_gated_delta_rule.cpp make_quadrant_masks.
std::vector<std::vector<float>> mask_tiles() {
    std::vector<std::vector<float>> m(3, std::vector<float>(kTileHW, 0.0f));
    for (uint32_t i = 0; i < kTileDim; ++i) {
        for (uint32_t j = 0; j < kTileDim; ++j) {
            const bool lo_i = i < 16, lo_j = j < 16;
            m[0][i * kTileDim + j] = (lo_i && lo_j) ? 1.0f : 0.0f;
            m[1][i * kTileDim + j] = (!lo_i && !lo_j) ? 1.0f : 0.0f;
            m[2][i * kTileDim + j] = (!lo_i && lo_j) ? 1.0f : 0.0f;
        }
    }
    return m;
}

// --- device run --------------------------------------------------------------------------------------------

RunResult run(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const RunConfig& c,
    const std::vector<std::vector<float>>& negn_tiles) {
    TT_FATAL(negn_tiles.size() == c.num_in, "run: {} negN tiles for num_in {}", negn_tiles.size(), c.num_in);
    // The move probes copy X (4, 5, 8, 9) and I (8, 9) to DST bit-exact through unpack-to-dest; a CB in that mode
    // must not be unpacked for a matmul, so the flags follow the probe.
    const uint32_t vi = static_cast<uint32_t>(c.variant);
    const bool utd_n = vi == 4 || vi == 5 || (vi >= 8 && vi <= 13);
    const bool utd_eye = vi >= 8 && vi <= 13;
    const bool llk_rounds = c.variant == Variant::LlkRoundsHorner || c.variant == Variant::LlkRoundsHornerR;
    const uint32_t tiles_per_in = llk_rounds ? 4 : 1;  // negN, N00, N11, N10
    const uint32_t outs_per_in = llk_rounds ? 3 : 1;   // Bi00, Bi11, off
    const uint32_t num_out = c.num_in * c.reps * outs_per_in;
    const ::unit_tests::compute::GoldenConfig cfg{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = 4};

    std::vector<uint32_t> negn_tiled, mask_tiled;
    for (const auto& t : negn_tiles) {
        const auto tiled = ::unit_tests::compute::gold_standard_tilize(fp32_as_u32(t), cfg);
        negn_tiled.insert(negn_tiled.end(), tiled.begin(), tiled.end());
        if (llk_rounds) {
            for (uint32_t q = 0; q < 3; ++q) {  // N00, N11, N10
                std::vector<float> quad(kTileHW, 0.0f);
                for (uint32_t i = 0; i < kTileHW; ++i) {
                    const bool lo_i = (i / kTileDim) < 16, lo_j = (i % kTileDim) < 16;
                    const bool keep = q == 0 ? (lo_i && lo_j) : q == 1 ? (!lo_i && !lo_j) : (!lo_i && lo_j);
                    quad[i] = keep ? t[i] : 0.0f;
                }
                const auto qt = ::unit_tests::compute::gold_standard_tilize(fp32_as_u32(quad), cfg);
                negn_tiled.insert(negn_tiled.end(), qt.begin(), qt.end());
            }
        }
    }
    for (const auto& t : mask_tiles()) {
        const auto tiled = ::unit_tests::compute::gold_standard_tilize(fp32_as_u32(t), cfg);
        mask_tiled.insert(mask_tiled.end(), tiled.begin(), tiled.end());
    }
    auto eye_tiled = ::unit_tests::compute::gold_standard_tilize(fp32_as_u32(identity_tile()), cfg);

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
    auto negn_buffer = make_dram(c.num_in * tiles_per_in * kTileBytes);
    auto eye_buffer = make_dram(kTileBytes);
    auto mask_buffer = make_dram(3 * kTileBytes);
    auto out_buffer = make_dram(num_out * kTileBytes);
    auto stats_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kStatsBytes},
        distributed::DeviceLocalBufferConfig{.page_size = kStatsBytes, .buffer_type = BufferType::L1},
        mesh_device.get());

    auto make_cb = [&](CBIndex idx, uint32_t tiles) {
        CreateCircularBuffer(
            program_,
            core,
            CircularBufferConfig(tiles * kTileBytes, {{idx, tt::DataFormat::Float32}}).set_page_size(idx, kTileBytes));
    };
    make_cb(CBIndex::c_0, 2 * tiles_per_in);
    make_cb(CBIndex::c_1, 1);
    make_cb(CBIndex::c_2, 3);
    for (uint32_t i = 3; i <= 8; ++i) {
        make_cb(static_cast<CBIndex>(i), 1);
    }
    make_cb(CBIndex::c_16, 2);

    auto reader_kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_tinv_fpu.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto writer_kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (utd_n) {
        unpack_to_dest_mode[CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    }
    if (utd_eye) {
        unpack_to_dest_mode[CBIndex::c_1] = UnpackToDestMode::UnpackToDestFp32;
    }
    std::map<std::string, std::string> defines = {{"GDN_TINV_FPU", "1"}};
    if (c.hoist) {
        defines["GDN_HOIST_RECONFIG"] = "1";
    }
    auto compute_kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/tinv_fpu.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = false,
            .compile_args =
                {static_cast<uint32_t>(c.variant),
                 c.num_in,
                 c.reps,
                 c.stall ? 1u : 0u,
                 c.nsrc,
                 c.split ? 1u : 0u,
                 c.fmt},
            .defines = defines});

    distributed::WriteShard(cq, negn_buffer, negn_tiled, zero_coord);
    distributed::WriteShard(cq, eye_buffer, eye_tiled, zero_coord);
    distributed::WriteShard(cq, mask_buffer, mask_tiled, zero_coord);

    SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {static_cast<uint32_t>(negn_buffer->address()),
         kDramBankId,
         static_cast<uint32_t>(eye_buffer->address()),
         kDramBankId,
         static_cast<uint32_t>(mask_buffer->address()),
         kDramBankId,
         c.num_in * tiles_per_in});
    SetRuntimeArgs(program_, writer_kernel, core, {static_cast<uint32_t>(out_buffer->address()), kDramBankId, num_out});
    SetRuntimeArgs(program_, compute_kernel, core, {static_cast<uint32_t>(stats_buffer->address())});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> out_tiled;
    distributed::ReadShard(cq, out_tiled, out_buffer, zero_coord);
    RunResult res;
    for (uint32_t t = 0; t < num_out && (t + 1) * kTileHW <= out_tiled.size(); ++t) {
        const std::vector<uint32_t> tile(out_tiled.begin() + t * kTileHW, out_tiled.begin() + (t + 1) * kTileHW);
        std::vector<float> rm;
        rm.reserve(kTileHW);
        for (const uint32_t w : ::unit_tests::compute::gold_standard_untilize(tile, cfg)) {
            rm.push_back(std::bit_cast<float>(w));
        }
        res.tiles.push_back(std::move(rm));
    }
    std::vector<uint32_t> stats;
    detail::ReadFromDeviceL1(
        mesh_device->get_devices().at(0), core, static_cast<uint32_t>(stats_buffer->address()), kStatsBytes, stats);
    const uint32_t n = std::min<uint32_t>(stats.empty() ? 0 : stats[0], (kStatsBytes / 4 - 1) / 4);
    res.stamps.assign(stats.begin() + 1, stats.begin() + 1 + 4 * n);
    return res;
}

// --- reporting ---------------------------------------------------------------------------------------------

struct Timing {
    uint32_t n = 0;
    double first = 0.0, median = 0.0, min = 0.0, mean = 0.0;
    double ph_setup = 0.0, ph_chain = 0.0, ph_tail = 0.0;  // split medians: acquire+loads, chain, commit..release
};

double median_of(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

Timing summarize(const RunResult& r, bool split) {
    Timing t;
    const uint32_t n = r.stamps.size() / 4;
    t.n = n;
    if (n == 0) {
        return t;
    }
    std::vector<double> total, s, ch, tl;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t* q = &r.stamps[4 * i];
        total.push_back(static_cast<double>(q[3] - q[0]));
        if (split) {
            s.push_back(static_cast<double>(q[1] - q[0]));
            ch.push_back(static_cast<double>(q[2] - q[1]));
            tl.push_back(static_cast<double>(q[3] - q[2]));
        }
    }
    t.first = total[0];
    std::vector<double> warm(total.begin() + (n > 1 ? 1 : 0), total.end());
    t.median = median_of(warm);
    t.min = *std::min_element(warm.begin(), warm.end());
    t.mean = std::accumulate(warm.begin(), warm.end(), 0.0) / warm.size();
    if (split) {
        t.ph_setup = median_of(s);
        t.ph_chain = median_of(ch);
        t.ph_tail = median_of(tl);
    }
    return t;
}

void log_timing(const std::string& label, const Timing& t, bool split) {
    log_info(
        tt::LogTest,
        "TINV_TIMING {:<34} n={:<4} first={:>7.0f}  median={:>7.0f} cyc ({:.3f} us @{}GHz)  min={:>7.0f}  "
        "mean={:>8.1f}",
        label,
        t.n,
        t.first,
        t.median,
        t.median / (kClockGHz * 1000.0),
        kClockGHz,
        t.min,
        t.mean);
    if (split) {
        log_info(
            tt::LogTest,
            "TINV_SPLIT  {:<34} setup={:>6.0f}  chain={:>6.0f}  tail={:>6.0f} cyc (medians)",
            label,
            t.ph_setup,
            t.ph_chain,
            t.ph_tail);
    }
}

uint32_t trunc_tf32(uint32_t bits) { return bits & 0xFFFFE000u; }
uint32_t rne_tf32(uint32_t bits) {
    const uint32_t lsb = (bits >> 13) & 1u;
    const uint32_t rounded = bits + 0xFFFu + lsb;
    return rounded & 0xFFFFE000u;
}
uint32_t trunc_bf16(uint32_t bits) { return bits & 0xFFFF0000u; }

uint32_t trunc_bits(uint32_t bits, uint32_t k) { return bits & (0xFFFFFFFFu << k); }

struct ProbeStats {
    uint32_t exact = 0, eq_trunc_tf32 = 0, eq_rne_tf32 = 0, eq_trunc_bf16 = 0, other = 0;
    uint32_t eq_trunc9 = 0;  // 9 explicit mantissa bits: what the MVMUL takes from SrcA
};

ProbeStats probe_stats(const std::vector<float>& x, const std::vector<float>& y) {
    ProbeStats s;
    for (uint32_t i = 0; i < kTileHW; ++i) {
        const uint32_t xb = std::bit_cast<uint32_t>(x[i]);
        const uint32_t yb = std::bit_cast<uint32_t>(y[i]);
        if (yb == xb) {
            s.exact++;
        }
        if (yb == trunc_bits(xb, 14)) {
            s.eq_trunc9++;
        }
        if (yb == trunc_tf32(xb)) {
            s.eq_trunc_tf32++;
        } else if (yb == rne_tf32(xb)) {
            s.eq_rne_tf32++;
        } else if (yb == trunc_bf16(xb)) {
            s.eq_trunc_bf16++;
        } else {
            s.other++;
        }
    }
    return s;
}

// Largest k such that y == x with the low k bits cleared, per element; logs the histogram and a few raw pairs.
void log_probe_bits(const char* name, uint32_t tile, const std::vector<float>& x, const std::vector<float>& y) {
    std::map<int, uint32_t> hist;
    uint32_t zeros = 0;
    for (uint32_t i = 0; i < kTileHW; ++i) {
        const uint32_t xb = std::bit_cast<uint32_t>(x[i]);
        const uint32_t yb = std::bit_cast<uint32_t>(y[i]);
        if (yb == 0) {
            zeros++;
        }
        int best = -1;
        for (int k = 0; k <= 24; ++k) {
            const uint32_t mask = k == 0 ? 0xFFFFFFFFu : (0xFFFFFFFFu << k);
            if (yb == (xb & mask)) {
                best = k;
                break;
            }
        }
        hist[best]++;
    }
    std::string h;
    for (const auto& [k, n] : hist) {
        h += fmt::format(" k{}:{}", k, n);
    }
    log_info(tt::LogTest, "TINV_PROBE_BITS {:<28} tile {}: zeros {} lowbits-cleared histogram{}", name, tile, zeros, h);
    for (uint32_t i : {0u, 1u, 33u, 100u, 527u, 1023u}) {
        log_info(
            tt::LogTest,
            "TINV_PROBE_BITS {:<28} tile {} [{:>4}] x={:08x} ({: .6e}) y={:08x} ({: .6e})",
            name,
            tile,
            i,
            std::bit_cast<uint32_t>(x[i]),
            x[i],
            std::bit_cast<uint32_t>(y[i]),
            y[i]);
    }
}

std::vector<std::vector<float>> regime_tiles(Regime regime, uint32_t n, uint32_t seed0) {
    std::vector<std::vector<float>> tiles;
    for (uint32_t i = 0; i < n; ++i) {
        tiles.push_back(make_negn(regime, seed0 + i));
    }
    return tiles;
}

}  // namespace unit_tests::compute::tinv_fpu

using namespace unit_tests::compute::tinv_fpu;

// Facts (a)/(c): what MOVD2B / MOVD2A carry from an fp32 DST into SrcB / SrcA, against the unpacker's own
// fp32 -> Src conversion. Each probe multiplies X by the identity at HiFi4 and returns the product bits.
TEST_F(LLKBlackholeSingleCardFixture, TensixTinvFpuProbeMoves) {
    const auto x_tiles = regime_tiles(Regime::RandomFp32, 2, 101);
    // expect: 13 = every element is X truncated to tf32 (10 mantissa bits, the SrcB operand), 14 = X truncated to
    // 9 mantissa bits (what the MVMUL takes from SrcA), 0 = report only.
    struct Case {
        Variant v;
        uint32_t fmt;
        uint32_t expect;
    };
    const Case cases[] = {
        {Variant::ProbeUnpackB, 0, 13},    {Variant::ProbeUnpackA, 0, 14},      {Variant::ProbeUnpackB, 3, 13},
        {Variant::ProbeUnpackA, 3, 14},    {Variant::ProbeD2B, 0, 0},           {Variant::ProbeD2A, 0, 0},
        {Variant::ProbeDummyValid, 0, 0},  {Variant::ProbeD2B, 1, 0},           {Variant::ProbeD2A, 1, 14},
        {Variant::ProbeDummyValid, 1, 13}, {Variant::ProbeD2B, 3, 0},           {Variant::ProbeD2A, 3, 14},
        {Variant::ProbeDummyValid, 3, 13}, {Variant::ProbeD2B, 7, 0},           {Variant::ProbeBothMoved, 1, 13},
        {Variant::ProbeBothMoved, 3, 13},  {Variant::ProbeD2BFace, 3, 0},       {Variant::ProbeD2BThenA, 3, 0},
        {Variant::ProbeFaceMm, 3, 13},     {Variant::ProbeBothMoved, 3 | 8, 0},
    };
    for (const auto& cs : cases) {
        const std::string label = fmt::format("{}+fmt{}", variant_name(cs.v), cs.fmt);
        SCOPED_TRACE(label);
        const auto res =
            run(this->devices_.at(0), RunConfig{.variant = cs.v, .num_in = 2, .reps = 2, .fmt = cs.fmt}, x_tiles);
        ASSERT_EQ(res.tiles.size(), 4u);
        for (uint32_t t = 0; t < 4; ++t) {
            const auto s = probe_stats(x_tiles[t / 2], res.tiles[t]);
            log_info(
                tt::LogTest,
                "TINV_PROBE {:<28} tile {}: exact {:>4}  trunc_tf32 {:>4}  rne_tf32 {:>4}  trunc_bf16 {:>4}  other "
                "{:>4}  (trunc9 {:>4})",
                label,
                t,
                s.exact,
                s.eq_trunc_tf32,
                s.eq_rne_tf32,
                s.eq_trunc_bf16,
                s.other,
                s.eq_trunc9);
            if (t == 0) {
                log_probe_bits(label.c_str(), t, x_tiles[t / 2], res.tiles[t]);
                if (std::getenv("TINV_DUMP") != nullptr) {
                    for (uint32_t r = 0; r < kTileDim; ++r) {
                        std::string line;
                        for (uint32_t c2 = 0; c2 < kTileDim; ++c2) {
                            line += fmt::format("{:>8.3g}", res.tiles[t][r * kTileDim + c2]);
                        }
                        log_info(tt::LogTest, "TINV_DUMP {:<28} r{:>2}:{}", label, r, line);
                    }
                    for (uint32_t r = 0; r < 4; ++r) {
                        std::string line;
                        for (uint32_t c2 = 0; c2 < kTileDim; ++c2) {
                            line += fmt::format("{:>8.3g}", x_tiles[t / 2][r * kTileDim + c2]);
                        }
                        log_info(tt::LogTest, "TINV_DUMP {:<28} x{:>2}:{}", label, r, line);
                    }
                }
            }
            if (cs.expect == 13) {
                EXPECT_EQ(s.eq_trunc_tf32, kTileHW);
            } else if (cs.expect == 14) {
                EXPECT_EQ(s.eq_trunc9, kTileHW);
            }
        }
    }
}

// Fact (b), the discouraged form: a MATH-side SETDVALID instead of the unpacker's dummy-valid.
TEST_F(LLKBlackholeSingleCardFixture, TensixTinvFpuProbeSetDvalid) {
    const auto x_tiles = regime_tiles(Regime::RandomFp32, 1, 201);
    const auto res =
        run(this->devices_.at(0), RunConfig{.variant = Variant::ProbeSetDvalid, .num_in = 1, .reps = 2}, x_tiles);
    ASSERT_EQ(res.tiles.size(), 2u);
    for (uint32_t t = 0; t < 2; ++t) {
        const auto s = probe_stats(x_tiles[0], res.tiles[t]);
        log_info(
            tt::LogTest,
            "TINV_PROBE {:<20} tile {}: exact {:>4}  trunc_tf32 {:>4}  rne_tf32 {:>4}  trunc_bf16 {:>4}  other {:>4}",
            variant_name(Variant::ProbeSetDvalid),
            t,
            s.exact,
            s.eq_trunc_tf32,
            s.eq_rne_tf32,
            s.eq_trunc_bf16,
            s.other);
    }
}

// Accuracy of the four inverses on the three GDN regimes against the fp64 inverse, and the Horner anchor:
// the fused Horner result truncated to tf32 against invert_block.
TEST_F(LLKBlackholeSingleCardFixture, TensixTinvFpuAccuracy) {
    constexpr uint32_t kTiles = 8;
    const Variant variants[] = {
        Variant::InvertBlock, Variant::Sfpu, Variant::FpuHorner, Variant::FpuHornerR, Variant::FpuSquare};
    for (const Regime regime : {Regime::Typical, Regime::Hard, Regime::Adversarial}) {
        const auto negn = regime_tiles(regime, kTiles, 1000 + 100 * static_cast<uint32_t>(regime));
        std::map<Variant, RunResult> results;
        std::map<Variant, Accuracy> worst;
        for (const Variant v : variants) {
            SCOPED_TRACE(std::string(regime_name(regime)) + "/" + variant_name(v));
            results[v] = run(this->devices_.at(0), RunConfig{.variant = v, .num_in = kTiles, .reps = 1}, negn);
            ASSERT_EQ(results[v].tiles.size(), kTiles);
            Accuracy w;
            double sum_mean = 0.0, sum_res_mean = 0.0, sum_rel_mean = 0.0;
            for (uint32_t t = 0; t < kTiles; ++t) {
                const Accuracy a = measure(negn[t], results[v].tiles[t]);
                EXPECT_TRUE(a.finite);
                w.max_abs = std::max(w.max_abs, a.max_abs);
                w.max_ref = std::max(w.max_ref, a.max_ref);
                w.res_max = std::max(w.res_max, a.res_max);
                w.finite = w.finite && a.finite;
                sum_mean += a.mean_abs;
                sum_res_mean += a.res_mean;
                sum_rel_mean += a.rel_mean();
                w.mean_ref += a.mean_ref / kTiles;
            }
            w.mean_abs = sum_mean / kTiles;
            w.res_mean = sum_res_mean / kTiles;
            worst[v] = w;
            log_info(
                tt::LogTest,
                "TINV_ACC {:<12} {:<13} maxabs {:.3e} rel_max {:.3e} mean_abs {:.3e} rel_mean {:.3e} | resid max "
                "{:.3e} mean {:.3e} | max|T| {:.2f}",
                regime_name(regime),
                variant_name(v),
                w.max_abs,
                w.rel_max(),
                w.mean_abs,
                sum_rel_mean / kTiles,
                w.res_max,
                w.res_mean,
                w.max_ref);
        }
        // Anchors. (1) invert_block (its FPU eltwise rounds truncate intermediates below tf32): per-face diff counts.
        // (2) The same dataflow through LLK matmul rounds, face by face: bit-exact expected.
        for (const Variant fused : {Variant::FpuHorner, Variant::FpuHornerR}) {
            uint32_t mismatches = 0, max_ulp = 0, shown = 0;
            uint32_t per_face[4] = {0, 0, 0, 0};
            for (uint32_t t = 0; t < kTiles; ++t) {
                for (uint32_t i = 0; i < kTileHW; ++i) {
                    const uint32_t a = std::bit_cast<uint32_t>(results[Variant::InvertBlock].tiles[t][i]);
                    const uint32_t b = trunc_tf32(std::bit_cast<uint32_t>(results[fused].tiles[t][i]));
                    if (a != b) {
                        mismatches++;
                        const uint32_t r = i / kTileDim, cc = i % kTileDim;
                        per_face[(r / 16) * 2 + cc / 16]++;
                        const uint32_t d = a > b ? a - b : b - a;
                        max_ulp = std::max(max_ulp, d >> 13);
                        if (shown < 3 && t == 0) {
                            shown++;
                            log_info(
                                tt::LogTest,
                                "TINV_ANCHOR_EX {:<12} {:<12} tile 0 r{:>2} c{:>2}: invert_block {:08x} ({: .7e}) "
                                "fused {:08x} ({: .7e})",
                                regime_name(regime),
                                variant_name(fused),
                                r,
                                cc,
                                a,
                                std::bit_cast<float>(a),
                                std::bit_cast<uint32_t>(results[fused].tiles[t][i]),
                                results[fused].tiles[t][i]);
                        }
                    }
                }
            }
            log_info(
                tt::LogTest,
                "TINV_ANCHOR {:<12} {:<12}(tf32-truncated) vs invert_block: {} / {} differ (faces {} {} {} {}), max {} "
                "tf32 ulp",
                regime_name(regime),
                variant_name(fused),
                mismatches,
                kTiles * kTileHW,
                per_face[0],
                per_face[1],
                per_face[2],
                per_face[3],
                max_ulp);
            const Variant llk = fused == Variant::FpuHorner ? Variant::LlkRoundsHorner : Variant::LlkRoundsHornerR;
            const auto llk_res =
                run(this->devices_.at(0), RunConfig{.variant = llk, .num_in = kTiles, .reps = 1}, negn);
            ASSERT_EQ(llk_res.tiles.size(), 3 * kTiles);
            uint32_t exact_mismatch = 0;
            uint32_t face_mm[4] = {0, 0, 0, 0};
            for (uint32_t t = 0; t < kTiles; ++t) {
                for (uint32_t i = 0; i < kTileHW; ++i) {
                    const uint32_t r = i / kTileDim, cc = i % kTileDim;
                    const uint32_t face = (r / 16) * 2 + cc / 16;
                    if (face == 1) {
                        continue;  // zero in both
                    }
                    const uint32_t src = face == 0 ? 0 : face == 3 ? 1 : 2;
                    const uint32_t a = std::bit_cast<uint32_t>(llk_res.tiles[3 * t + src][i]);
                    const uint32_t b = std::bit_cast<uint32_t>(results[fused].tiles[t][i]);
                    if (a != b) {
                        exact_mismatch++;
                        face_mm[face]++;
                        if (exact_mismatch <= 3) {
                            log_info(
                                tt::LogTest,
                                "TINV_ANCHOR_LLK_EX {:<12} {:<12} tile {} r{:>2} c{:>2}: llk_rounds {:08x} fused "
                                "{:08x}",
                                regime_name(regime),
                                variant_name(fused),
                                t,
                                r,
                                cc,
                                a,
                                b);
                        }
                    }
                }
            }
            log_info(
                tt::LogTest,
                "TINV_ANCHOR_LLK {:<12} {:<12} vs {}: {} / {} elements differ (faces {} - {} {})",
                regime_name(regime),
                variant_name(fused),
                variant_name(llk),
                exact_mismatch,
                kTiles * kTileHW * 3 / 4,
                face_mm[0],
                face_mm[2],
                face_mm[3]);
            EXPECT_EQ(exact_mismatch, 0u);
        }
        // Gate: the fused inverses at least as accurate as invert_block (max-abs against fp64, 5% slack).
        EXPECT_LE(worst[Variant::FpuHornerR].max_abs, 1.05 * worst[Variant::InvertBlock].max_abs + 1e-7);
    }
}

// Cycles per inverse over 100 back-to-back inverses of one typical-regime tile, per variant and per fused
// option (stall between stages, negN via DST, phase split).
TEST_F(LLKBlackholeSingleCardFixture, TensixTinvFpuTiming) {
    constexpr uint32_t kReps = 100;
    const auto negn = regime_tiles(Regime::Typical, 1, 1000);
    struct Case {
        std::string label;
        RunConfig cfg;
    };
    const std::vector<Case> cases = {
        {"invert_block(hoist)", {.variant = Variant::InvertBlock, .reps = kReps}},
        {"invert_block(no hoist)", {.variant = Variant::InvertBlock, .reps = kReps, .hoist = false}},
        {"sfpu_solve", {.variant = Variant::Sfpu, .reps = kReps}},
        {"llk_rounds_horner", {.variant = Variant::LlkRoundsHorner, .reps = kReps}},
        {"fpu_horner", {.variant = Variant::FpuHorner, .reps = kReps}},
        {"fpu_hornerR", {.variant = Variant::FpuHornerR, .reps = kReps}},
        {"fpu_square", {.variant = Variant::FpuSquare, .reps = kReps}},
        {"fpu_horner stall", {.variant = Variant::FpuHorner, .reps = kReps, .stall = true}},
        {"fpu_square stall", {.variant = Variant::FpuSquare, .reps = kReps, .stall = true}},
        {"fpu_horner nsrc=dst", {.variant = Variant::FpuHorner, .reps = kReps, .nsrc = 1}},
        {"fpu_square nsrc=dst", {.variant = Variant::FpuSquare, .reps = kReps, .nsrc = 1}},
        {"fpu_hornerR nsrc=dst", {.variant = Variant::FpuHornerR, .reps = kReps, .nsrc = 1}},
        {"fpu_horner split", {.variant = Variant::FpuHorner, .reps = kReps, .split = true}},
        {"fpu_hornerR split", {.variant = Variant::FpuHornerR, .reps = kReps, .split = true}},
        {"fpu_square split", {.variant = Variant::FpuSquare, .reps = kReps, .split = true}},
    };
    std::map<Variant, std::vector<float>> first_tile;  // the fused options of one form must agree bit for bit
    for (const auto& cs : cases) {
        SCOPED_TRACE(cs.label);
        const auto res = run(this->devices_.at(0), cs.cfg, negn);
        const uint32_t per_rep = cs.cfg.variant == Variant::LlkRoundsHorner ? 3 : 1;
        if (per_rep == 1 && static_cast<uint32_t>(cs.cfg.variant) >= 2) {
            auto it = first_tile.find(cs.cfg.variant);
            if (it == first_tile.end()) {
                first_tile[cs.cfg.variant] = res.tiles.at(0);
            } else {
                EXPECT_EQ(std::memcmp(it->second.data(), res.tiles.at(0).data(), kTileBytes), 0)
                    << cs.label << " differs from the first run of its form";
            }
        }
        ASSERT_EQ(res.tiles.size(), kReps * per_rep);
        // every repetition reproduces the same tile(s)
        for (uint32_t r = 1; r < kReps; ++r) {
            EXPECT_EQ(std::memcmp(res.tiles[0].data(), res.tiles[r * per_rep].data(), kTileBytes), 0) << "rep " << r;
        }
        if (per_rep == 1) {
            const Accuracy a = measure(negn[0], res.tiles[0]);
            EXPECT_TRUE(a.finite);
            EXPECT_LT(a.rel_max(), 1e-2);
        }
        const Timing t = summarize(res, cs.cfg.split);
        log_timing(cs.label, t, cs.cfg.split);
    }
}

}  // namespace tt::tt_metal
