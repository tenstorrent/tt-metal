// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The gradient arithmetic of one block pair, against a CPU reference.
//
// The transport is validated separately with a checksum payload; this checks
// the five matmuls and the softmax backward chain in isolation, on one core,
// for one pair. The two are merged afterwards.
//
// The chain is built up in stages so a numerical failure names the step that
// produced it: scores, then P, then dP, then dS, then the three gradients.
// Each stage writes its last intermediate to a probe tile, and the reference
// computes exactly the same quantity, including where the 1/sqrt(d) scale is
// applied -- once to the scores and once to dS, which is what the kernel
// does and what folds the scale into dQ and dK.

#include <gtest/gtest.h>

#include "metal/common/program_utils.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"

namespace {

namespace tt_dist = tt::tt_metal::distributed;

constexpr uint32_t kTile = 32;
constexpr const char* kReaderPath =
    "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/dataflow/cyclic_pair_reader.cpp";
constexpr const char* kComputePath =
    "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/compute/cyclic_pair_compute.cpp";
constexpr const char* kWriterPath =
    "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/dataflow/cyclic_pair_writer.cpp";

// One block pair's operands and the reference results.
struct Problem {
    uint32_t Bt = 1;  // row-tiles per block; B = Bt * 32
    uint32_t B = kTile;
    uint32_t d = 64;
    bool diagonal = false;

    xt::xarray<float> Q, K, V, dO;
    xt::xarray<float> lse_tile, u_tile;  // (1,1,B,32), value in column 0
    xt::xarray<float> S, P, dP, dS;      // (B,B)
    xt::xarray<float> dQ, dK, dV;        // (B,d)
};

// Deterministic pseudo-random values, small enough that bfloat16 inputs do
// not dominate the comparison.
float sample(uint32_t seed) {
    const uint32_t h = seed * 1664525u + 1013904223u;
    return static_cast<float>(static_cast<int32_t>(h % 2001u) - 1000) / 2000.0F;
}

xt::xarray<float> random_matrix(uint32_t rows, uint32_t cols, uint32_t salt) {
    xt::xarray<float> out = xt::zeros<float>({rows, cols});
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            out(r, c) = sample(salt * 7919u + r * 131u + c);
        }
    }
    return out;
}

Problem make_problem(uint32_t d, bool diagonal, uint32_t Bt = 1) {
    Problem p;
    p.Bt = Bt;
    p.B = Bt * kTile;
    p.d = d;
    p.diagonal = diagonal;
    const uint32_t B = p.B;
    const float scale = 1.0F / std::sqrt(static_cast<float>(d));

    p.Q = random_matrix(B, d, 1);
    p.K = random_matrix(B, d, 2);
    p.V = random_matrix(B, d, 3);
    p.dO = random_matrix(B, d, 4);

    // The kernel scales the scores once, then scales dS once. The reference
    // has to fold it in the same two places.
    p.S = xt::linalg::dot(p.Q, xt::transpose(p.K)) * scale;
    if (diagonal) {
        for (uint32_t r = 0; r < B; ++r) {
            for (uint32_t c = r + 1; c < B; ++c) {
                p.S(r, c) = -std::numeric_limits<float>::infinity();
            }
        }
    }

    // A real softmax logsumexp, so P is well conditioned.
    xt::xarray<float> lse = xt::zeros<float>({B});
    for (uint32_t r = 0; r < B; ++r) {
        float m = -std::numeric_limits<float>::infinity();
        for (uint32_t c = 0; c < B; ++c) {
            m = std::max(m, p.S(r, c));
        }
        float sum = 0.0F;
        for (uint32_t c = 0; c < B; ++c) {
            sum += std::exp(p.S(r, c) - m);
        }
        lse(r) = m + std::log(sum);
    }

    p.P = xt::zeros<float>({B, B});
    for (uint32_t r = 0; r < B; ++r) {
        for (uint32_t c = 0; c < B; ++c) {
            p.P(r, c) = std::exp(p.S(r, c) - lse(r));
        }
    }

    p.dP = xt::linalg::dot(p.dO, xt::transpose(p.V));

    // u = rowsum(dO * O) with O = P V, which is what the forward pass leaves.
    const xt::xarray<float> O = xt::linalg::dot(p.P, p.V);
    xt::xarray<float> u = xt::zeros<float>({B});
    for (uint32_t r = 0; r < B; ++r) {
        float sum = 0.0F;
        for (uint32_t c = 0; c < d; ++c) {
            sum += p.dO(r, c) * O(r, c);
        }
        u(r) = sum;
    }

    p.dS = xt::zeros<float>({B, B});
    for (uint32_t r = 0; r < B; ++r) {
        for (uint32_t c = 0; c < B; ++c) {
            p.dS(r, c) = p.P(r, c) * (p.dP(r, c) - u(r)) * scale;
        }
    }

    p.dQ = xt::linalg::dot(p.dS, p.K);
    p.dK = xt::linalg::dot(xt::transpose(p.dS), p.Q);
    p.dV = xt::linalg::dot(xt::transpose(p.P), p.dO);

    // Column 0 carries the per-row scalar; the kernel broadcasts it.
    p.lse_tile = xt::zeros<float>({1u, 1u, B, kTile});
    p.u_tile = xt::zeros<float>({1u, 1u, B, kTile});
    for (uint32_t r = 0; r < B; ++r) {
        p.lse_tile(0, 0, r, 0) = lse(r);
        p.u_tile(0, 0, r, 0) = u(r);
    }
    return p;
}

xt::xarray<float> as_4d(const xt::xarray<float>& m) {
    const auto shape = m.shape();
    xt::xarray<float> out = xt::zeros<float>({1u, 1u, static_cast<uint32_t>(shape[0]),
                                              static_cast<uint32_t>(shape[1])});
    xt::view(out, 0, 0, xt::all(), xt::all()) = m;
    return out;
}

struct Outputs {
    xt::xarray<float> probe;  // (1,1,B,32) -- only the first 32 columns are checked
    xt::xarray<float> dQ, dK, dV;
};

Outputs run_pair(const Problem& p, uint32_t stage) {
    using namespace tt::tt_metal;
    auto* device = &ttml::autograd::ctx().get_device();

    const uint32_t qWt = p.d / kTile;
    const uint32_t vWt = p.d / kTile;
    const uint32_t Bt = p.Bt;
    const uint32_t score_tiles = Bt * Bt;

    const auto query = ttml::core::from_xtensor(as_4d(p.Q), device);
    const auto key = ttml::core::from_xtensor(as_4d(p.K), device);
    const auto value = ttml::core::from_xtensor(as_4d(p.V), device);
    const auto grad_output = ttml::core::from_xtensor(as_4d(p.dO), device);
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(p.lse_tile, device);
    const auto u_scalar = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(p.u_tile, device);

    const xt::xarray<float> zeros_probe = xt::zeros<float>({1u, 1u, p.B, kTile});
    const xt::xarray<float> zeros_grad = xt::zeros<float>({1u, 1u, p.B, p.d});
    const auto probe = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros_probe, device);
    const auto grad_query = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros_grad, device);
    const auto grad_key = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros_grad, device);
    const auto grad_value = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros_grad, device);

    auto program = CreateProgram();
    const auto core = CoreCoord{0, 0};
    const auto region = CoreRange(core, core);

    const uint32_t bf16_tile = 2 * kTile * kTile;
    const uint32_t fp32_tile = 4 * kTile * kTile;
    const auto make_cb = [&](uint32_t index, uint32_t tiles, tt::DataFormat format) {
        const uint32_t page = (format == tt::DataFormat::Float32) ? fp32_tile : bf16_tile;
        CreateCircularBuffer(
            program,
            region,
            CircularBufferConfig(tiles * page, {{index, format}}).set_page_size(index, page));
    };
    make_cb(tt::CBIndex::c_0, Bt * qWt, tt::DataFormat::Float16_b);    // Q
    make_cb(tt::CBIndex::c_1, Bt * qWt, tt::DataFormat::Float16_b);    // K
    make_cb(tt::CBIndex::c_2, Bt * vWt, tt::DataFormat::Float16_b);    // V
    make_cb(tt::CBIndex::c_3, Bt * vWt, tt::DataFormat::Float16_b);    // dO
    make_cb(tt::CBIndex::c_4, Bt, tt::DataFormat::Float32);            // L_i
    make_cb(tt::CBIndex::c_5, Bt, tt::DataFormat::Float32);            // D_i
    make_cb(tt::CBIndex::c_6, 1, tt::DataFormat::Float16_b);           // causal mask
    make_cb(tt::CBIndex::c_10, score_tiles, tt::DataFormat::Float32);  // P
    make_cb(tt::CBIndex::c_11, score_tiles, tt::DataFormat::Float32);  // dP
    make_cb(tt::CBIndex::c_12, score_tiles, tt::DataFormat::Float32);  // dS
    make_cb(tt::CBIndex::c_13, score_tiles, tt::DataFormat::Float32);  // dS^T
    make_cb(tt::CBIndex::c_14, score_tiles, tt::DataFormat::Float32);  // P^T
    make_cb(tt::CBIndex::c_15, 1, tt::DataFormat::Float32);            // probe
    make_cb(tt::CBIndex::c_16, Bt * qWt, tt::DataFormat::Float32);     // dQ
    make_cb(tt::CBIndex::c_17, Bt * qWt, tt::DataFormat::Float32);     // dK
    make_cb(tt::CBIndex::c_18, Bt * vWt, tt::DataFormat::Float32);     // dV

    std::map<std::string, std::string> defines = {{"COMPUTE_STAGE", std::to_string(stage)}};
    if (p.diagonal) {
        defines["DIAGONAL_BLOCK"] = "1";
    }

    std::vector<uint32_t> reader_args = {qWt, vWt, Bt};
    for (const auto* t : {&query, &key, &value, &grad_output, &lse, &u_scalar}) {
        tt::tt_metal::TensorAccessorArgs(*t->buffer()).append_to(reader_args);
    }
    const auto reader = CreateKernel(
        program, kReaderPath, region,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_args,
            .defines = defines});

    std::vector<uint32_t> writer_args = {qWt, vWt, Bt};
    for (const auto* t : {&probe, &grad_query, &grad_key, &grad_value}) {
        tt::tt_metal::TensorAccessorArgs(*t->buffer()).append_to(writer_args);
    }
    const auto writer = CreateKernel(
        program, kWriterPath, region,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_args,
            .defines = defines});

    const uint32_t scaler = std::bit_cast<uint32_t>(1.0F / std::sqrt(static_cast<float>(p.d)));
    const uint32_t minus_one = std::bit_cast<uint32_t>(-1.0F);
    const uint32_t custom_inf = std::bit_cast<uint32_t>(tt::tt_metal::hal::get_inf());
    // P is copied into DST and must stay FP32 through the elementwise dS
    // chain, so its CB unpacks to dest at FP32. The transposed CBs must not:
    // they feed matmul Src registers, where Float32 unpack is unsupported.
    std::vector<UnpackToDestMode> unpack_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_mode[tt::CBIndex::c_10] = UnpackToDestMode::UnpackToDestFp32;

    const auto compute = CreateKernel(
        program, kComputePath, region,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_mode,
            .compile_args = {qWt, vWt, scaler, minus_one, custom_inf, get_block_size(qWt, 4U), Bt},
            .defines = defines});
    (void)compute;

    SetRuntimeArgs(
        program, reader, core,
        {query.buffer()->address(), key.buffer()->address(), value.buffer()->address(),
         grad_output.buffer()->address(), lse.buffer()->address(), u_scalar.buffer()->address(),
         1u, 0u, 0u});  // one pair, row block 0 against column block 0
    SetRuntimeArgs(
        program, writer, core,
        {probe.buffer()->address(), grad_query.buffer()->address(), grad_key.buffer()->address(),
         grad_value.buffer()->address()});

    auto workload = tt_dist::MeshWorkload();
    workload.add_program(tt_dist::MeshCoordinateRange(device->shape()), std::move(program));
    tt_dist::EnqueueMeshWorkload(device->mesh_command_queue(), workload, /*blocking=*/true);

    Outputs out;
    out.probe = ttml::core::to_xtensor(probe);
    out.dQ = ttml::core::to_xtensor(grad_query);
    out.dK = ttml::core::to_xtensor(grad_key);
    out.dV = ttml::core::to_xtensor(grad_value);
    return out;
}

// Relative to the reference's own scale, since bfloat16 operands set the
// floor on what any of this can agree to.
void expect_close(
    const xt::xarray<float>& got_4d,
    const xt::xarray<float>& want,
    float tolerance,
    const std::string& what) {
    const uint32_t rows = static_cast<uint32_t>(want.shape()[0]);
    const uint32_t cols = static_cast<uint32_t>(want.shape()[1]);
    float max_abs = 0.0F;
    float max_diff = 0.0F;
    uint32_t worst_r = 0;
    uint32_t worst_c = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const float ref = want(r, c);
            if (!std::isfinite(ref)) {
                continue;  // masked entries are -inf in the reference
            }
            max_abs = std::max(max_abs, std::abs(ref));
            const float diff = std::abs(got_4d(0, 0, r, c) - ref);
            if (diff > max_diff) {
                max_diff = diff;
                worst_r = r;
                worst_c = c;
            }
        }
    }
    const float relative = max_abs > 0.0F ? max_diff / max_abs : max_diff;
    EXPECT_LT(relative, tolerance)
        << what << ": max |diff| " << max_diff << " against max |ref| " << max_abs
        << " (relative " << relative << "), worst at (" << worst_r << "," << worst_c << ") got "
        << got_4d(0, 0, worst_r, worst_c) << " want " << want(worst_r, worst_c);
}

}  // namespace

TEST(CyclicPairComputeTest, Stage1Scores) {
    const auto p = make_problem(64, /*diagonal=*/false);
    const auto out = run_pair(p, 1);
    expect_close(out.probe, p.S, 0.02F, "S = Q K^T / sqrt(d)");
}

TEST(CyclicPairComputeTest, Stage2AttentionWeights) {
    const auto p = make_problem(64, false);
    const auto out = run_pair(p, 2);
    expect_close(out.probe, p.P, 0.02F, "P = exp(S - L)");
}

TEST(CyclicPairComputeTest, Stage3GradAttentionWeights) {
    const auto p = make_problem(64, false);
    const auto out = run_pair(p, 3);
    expect_close(out.probe, p.dP, 0.02F, "dP = dO V^T");
}

TEST(CyclicPairComputeTest, Stage4GradScores) {
    const auto p = make_problem(64, false);
    const auto out = run_pair(p, 4);
    expect_close(out.probe, p.dS, 0.03F, "dS = P (dP - D) / sqrt(d)");
}

TEST(CyclicPairComputeTest, Stage5AllThreeGradients) {
    const auto p = make_problem(64, false);
    const auto out = run_pair(p, 5);
    expect_close(out.probe, p.dS, 0.03F, "dS");
    expect_close(out.dQ, p.dQ, 0.05F, "dQ = dS K");
    expect_close(out.dK, p.dK, 0.05F, "dK = dS^T Q");
    expect_close(out.dV, p.dV, 0.05F, "dV = P^T dO");
}

// The diagonal block, where the intra-block causal mask applies. The mask
// tile is generated on the core; the reference sets the strictly upper
// triangle of S to -inf, and expect_close skips those entries, so what is
// being compared is that everything below the diagonal is untouched and
// everything above contributes nothing.
TEST(CyclicPairComputeTest, Stage1ScoresOnTheDiagonal) {
    const auto p = make_problem(64, /*diagonal=*/true);
    const auto out = run_pair(p, 1);
    expect_close(out.probe, p.S, 0.02F, "masked S");
}

TEST(CyclicPairComputeTest, Stage2AttentionWeightsOnTheDiagonal) {
    const auto p = make_problem(64, true);
    const auto out = run_pair(p, 2);
    expect_close(out.probe, p.P, 0.02F, "masked P");
    // Above the diagonal P must be exactly zero, not merely small: those
    // entries must contribute nothing to dV and dK.
    for (uint32_t r = 0; r < p.B; ++r) {
        for (uint32_t c = r + 1; c < p.B; ++c) {
            EXPECT_EQ(out.probe(0, 0, r, c), 0.0F)
                << "P(" << r << "," << c << ") above the diagonal is not zero";
        }
    }
}

TEST(CyclicPairComputeTest, Stage5AllThreeGradientsOnTheDiagonal) {
    const auto p = make_problem(64, true);
    const auto out = run_pair(p, 5);
    expect_close(out.dQ, p.dQ, 0.05F, "dQ on the diagonal");
    expect_close(out.dK, p.dK, 0.05F, "dK on the diagonal");
    expect_close(out.dV, p.dV, 0.05F, "dV on the diagonal");
}

// A wider head dimension, so the matmuls run over four inner tiles rather
// than two and the per-row scalars are broadcast across more of them.
// Two row-tiles per block: B = 64. The schedule does not change -- a block
// is still a block -- but every score stage now has four tiles instead of
// one, which is what lets the matmuls of a row issue back to back. The
// reference math was already written in terms of B, so only the kernel's
// tile loops are under test here.
TEST(CyclicPairComputeTest, Stage5TwoRowTilesPerBlock) {
    const auto p = make_problem(64, false, /* Bt */ 2);
    ASSERT_EQ(p.B, 64u);
    const auto out = run_pair(p, 5);
    expect_close(out.dQ, p.dQ, 0.05F, "dQ at Bt = 2");
    expect_close(out.dK, p.dK, 0.05F, "dK at Bt = 2");
    expect_close(out.dV, p.dV, 0.05F, "dV at Bt = 2");
}

// The diagonal block is where the tile loops have to know about the
// triangle: the tile on the diagonal gets the causal mask, the tiles above it
// are wholly masked and are zeroed after the softmax so that every later sum
// can run over all four tiles without knowing anything about it.
TEST(CyclicPairComputeTest, Stage5TwoRowTilesOnTheDiagonal) {
    const auto p = make_problem(64, true, /* Bt */ 2);
    const auto out = run_pair(p, 5);
    expect_close(out.dQ, p.dQ, 0.05F, "dQ at Bt = 2, diagonal");
    expect_close(out.dK, p.dK, 0.05F, "dK at Bt = 2, diagonal");
    expect_close(out.dV, p.dV, 0.05F, "dV at Bt = 2, diagonal");
}

// What the block shape costs, measured rather than argued. Run with the
// device profiler and compare the PAIR zone: Bt = 2 does four times the
// score-stage arithmetic of Bt = 1, so anything under four times the cycles
// is the matmul pipeline latency being amortized.
//
//   TT_METAL_DEVICE_PROFILER=1 ttml_tests \
//     --gtest_filter=CyclicPairComputeTest.DISABLED_ProfileBlockShapes \
//     --gtest_also_run_disabled_tests
TEST(CyclicPairComputeTest, DISABLED_ProfileBlockShapes) {
    for (uint32_t bt : {1u, 2u}) {
        const auto p = make_problem(64, false, bt);
        (void)run_pair(p, 5);
    }
    ttml::autograd::ctx().close_device();
}

TEST(CyclicPairComputeTest, Stage5AllThreeGradientsWithWiderHead) {
    const auto p = make_problem(128, false);
    const auto out = run_pair(p, 5);
    expect_close(out.dQ, p.dQ, 0.05F, "dQ at d = 128");
    expect_close(out.dK, p.dK, 0.05F, "dK at d = 128");
    expect_close(out.dV, p.dV, 0.05F, "dV at d = 128");
}

// ------------------------------------------------ accumulation across pairs
// A streak accumulates dQ into the row packet across the columns it visits;
// a column residency accumulates dK and dV across the rows that pass by.
// Both are L1 accumulation into a buffer that outlives the pair, and this
// checks them without any transport: several pairs in sequence on one core,
// each naming its row and column block.

namespace {

// N blocks of each operand, so pairs can name blocks independently.
struct Blocks {
    uint32_t n = 4;
    uint32_t B = kTile;
    uint32_t d = 64;
    std::vector<xt::xarray<float>> Q, K, V, dO;
    std::vector<float> lse, u;  // one per row block, broadcast down the block
};

Blocks make_blocks(uint32_t n, uint32_t d) {
    Blocks b;
    b.n = n;
    b.d = d;
    for (uint32_t k = 0; k < n; ++k) {
        b.Q.push_back(random_matrix(b.B, d, 11 + k));
        b.K.push_back(random_matrix(b.B, d, 31 + k));
        b.V.push_back(random_matrix(b.B, d, 51 + k));
        b.dO.push_back(random_matrix(b.B, d, 71 + k));
    }
    // Any consistent per-row scalars will do; these keep P and dS in range.
    for (uint32_t k = 0; k < n; ++k) {
        b.lse.push_back(0.25F + 0.1F * static_cast<float>(k));
        b.u.push_back(0.05F * static_cast<float>(k + 1));
    }
    return b;
}

// One pair's contribution, in exactly the order the kernel computes it.
struct Contribution {
    xt::xarray<float> dQ, dK, dV;
};

Contribution pair_contribution(const Blocks& b, uint32_t row, uint32_t col) {
    const float scale = 1.0F / std::sqrt(static_cast<float>(b.d));
    const xt::xarray<float> S = xt::linalg::dot(b.Q[row], xt::transpose(b.K[col])) * scale;
    xt::xarray<float> P = xt::zeros<float>({b.B, b.B});
    xt::xarray<float> dS = xt::zeros<float>({b.B, b.B});
    const xt::xarray<float> dP = xt::linalg::dot(b.dO[row], xt::transpose(b.V[col]));
    for (uint32_t r = 0; r < b.B; ++r) {
        for (uint32_t c = 0; c < b.B; ++c) {
            P(r, c) = std::exp(S(r, c) - b.lse[row]);
            dS(r, c) = P(r, c) * (dP(r, c) - b.u[row]) * scale;
        }
    }
    Contribution out;
    out.dQ = xt::linalg::dot(dS, b.K[col]);
    out.dK = xt::linalg::dot(xt::transpose(dS), b.Q[row]);
    out.dV = xt::linalg::dot(xt::transpose(P), b.dO[row]);
    return out;
}

xt::xarray<float> stack_blocks(const std::vector<xt::xarray<float>>& blocks, uint32_t B, uint32_t w) {
    xt::xarray<float> out = xt::zeros<float>(
        {1u, 1u, static_cast<uint32_t>(blocks.size()) * B, w});
    for (uint32_t k = 0; k < blocks.size(); ++k) {
        for (uint32_t r = 0; r < B; ++r) {
            for (uint32_t c = 0; c < w; ++c) {
                out(0, 0, k * B + r, c) = blocks[k](r, c);
            }
        }
    }
    return out;
}

// seed_dq: a previous dQ the accumulator starts from, as Algorithm 2 loads
// from DRAM every timestep and the relay reloads at a streak start.
Outputs run_pairs(
    const Blocks& b,
    const std::vector<std::pair<uint32_t, uint32_t>>& pairs,
    const xt::xarray<float>* seed_dq = nullptr) {
    using namespace tt::tt_metal;
    auto* device = &ttml::autograd::ctx().get_device();

    const uint32_t qWt = b.d / kTile;
    const uint32_t vWt = b.d / kTile;
    const uint32_t n = b.n;

    const auto query = ttml::core::from_xtensor(stack_blocks(b.Q, b.B, b.d), device);
    const auto key = ttml::core::from_xtensor(stack_blocks(b.K, b.B, b.d), device);
    const auto value = ttml::core::from_xtensor(stack_blocks(b.V, b.B, b.d), device);
    const auto grad_output = ttml::core::from_xtensor(stack_blocks(b.dO, b.B, b.d), device);

    xt::xarray<float> lse_tile = xt::zeros<float>({1u, 1u, n * b.B, kTile});
    xt::xarray<float> u_tile = xt::zeros<float>({1u, 1u, n * b.B, kTile});
    for (uint32_t k = 0; k < n; ++k) {
        for (uint32_t r = 0; r < b.B; ++r) {
            lse_tile(0, 0, k * b.B + r, 0) = b.lse[k];
            u_tile(0, 0, k * b.B + r, 0) = b.u[k];
        }
    }
    const auto lse = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(lse_tile, device);
    const auto u_scalar = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(u_tile, device);

    const xt::xarray<float> zeros_probe = xt::zeros<float>({1u, 1u, b.B, kTile});
    const xt::xarray<float> zeros_grad = xt::zeros<float>({1u, 1u, b.B, b.d});
    const auto probe = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros_probe, device);
    const auto grad_query = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros_grad, device);
    const auto grad_key = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros_grad, device);
    const auto grad_value = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(zeros_grad, device);

    auto program = CreateProgram();
    const auto core = CoreCoord{0, 0};
    const auto region = CoreRange(core, core);

    const uint32_t bf16_tile = 2 * kTile * kTile;
    const uint32_t fp32_tile = 4 * kTile * kTile;
    const auto make_cb = [&](uint32_t index, uint32_t tiles, tt::DataFormat format) {
        const uint32_t page = (format == tt::DataFormat::Float32) ? fp32_tile : bf16_tile;
        CreateCircularBuffer(
            program, region,
            CircularBufferConfig(tiles * page, {{index, format}}).set_page_size(index, page));
    };
    const uint32_t Bt = 1;  // these tests exercise the streak and residency
                            // paths, not the block shape
    make_cb(tt::CBIndex::c_0, qWt, tt::DataFormat::Float16_b);
    make_cb(tt::CBIndex::c_1, qWt, tt::DataFormat::Float16_b);
    make_cb(tt::CBIndex::c_2, vWt, tt::DataFormat::Float16_b);
    make_cb(tt::CBIndex::c_3, vWt, tt::DataFormat::Float16_b);
    make_cb(tt::CBIndex::c_4, 1, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_5, 1, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_6, 1, tt::DataFormat::Float16_b);
    make_cb(tt::CBIndex::c_10, 1, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_11, 1, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_12, 1, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_13, 1, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_14, 1, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_15, 1, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_16, qWt, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_17, qWt, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_18, vWt, tt::DataFormat::Float32);
    make_cb(tt::CBIndex::c_19, qWt, tt::DataFormat::Float32);  // previous dQ

    // A dQ to start the accumulator from, stacked so the first pair's row
    // block names it.
    std::vector<xt::xarray<float>> seed_blocks(b.n, xt::zeros<float>({b.B, b.d}));
    if (seed_dq != nullptr) {
        seed_blocks[pairs.front().first] = *seed_dq;
    }
    const auto seed = ttml::core::from_xtensor<float, ttnn::DataType::FLOAT32>(
        stack_blocks(seed_blocks, b.B, b.d), device);

    std::map<std::string, std::string> defines = {
        {"COMPUTE_STAGE", "5"}, {"NUM_PAIRS", std::to_string(pairs.size())}};
    if (seed_dq != nullptr) {
        defines["SEED_DQ"] = "1";
    }

    std::vector<uint32_t> reader_args = {qWt, vWt, Bt};
    for (const auto* t : {&query, &key, &value, &grad_output, &lse, &u_scalar, &seed}) {
        tt::tt_metal::TensorAccessorArgs(*t->buffer()).append_to(reader_args);
    }
    const auto reader = CreateKernel(
        program, kReaderPath, region,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_args,
            .defines = defines});

    std::vector<uint32_t> writer_args = {qWt, vWt, Bt};
    for (const auto* t : {&probe, &grad_query, &grad_key, &grad_value}) {
        tt::tt_metal::TensorAccessorArgs(*t->buffer()).append_to(writer_args);
    }
    const auto writer = CreateKernel(
        program, kWriterPath, region,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_args,
            .defines = defines});

    const uint32_t scaler = std::bit_cast<uint32_t>(1.0F / std::sqrt(static_cast<float>(b.d)));
    const uint32_t minus_one = std::bit_cast<uint32_t>(-1.0F);
    const uint32_t custom_inf = std::bit_cast<uint32_t>(tt::tt_metal::hal::get_inf());
    std::vector<UnpackToDestMode> unpack_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_mode[tt::CBIndex::c_10] = UnpackToDestMode::UnpackToDestFp32;
    CreateKernel(
        program, kComputePath, region,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_mode,
            .compile_args = {qWt, vWt, scaler, minus_one, custom_inf, get_block_size(qWt, 4U), Bt},
            .defines = defines});

    std::vector<uint32_t> args = {
        query.buffer()->address(), key.buffer()->address(), value.buffer()->address(),
        grad_output.buffer()->address(), lse.buffer()->address(), u_scalar.buffer()->address(),
        static_cast<uint32_t>(pairs.size())};
    for (const auto& [row, col] : pairs) {
        args.push_back(row);
        args.push_back(col);
    }
    args.push_back(seed.buffer()->address());
    SetRuntimeArgs(program, reader, core, args);
    SetRuntimeArgs(
        program, writer, core,
        {probe.buffer()->address(), grad_query.buffer()->address(), grad_key.buffer()->address(),
         grad_value.buffer()->address()});

    auto workload = tt_dist::MeshWorkload();
    workload.add_program(tt_dist::MeshCoordinateRange(device->shape()), std::move(program));
    tt_dist::EnqueueMeshWorkload(device->mesh_command_queue(), workload, /*blocking=*/true);

    Outputs out;
    out.probe = ttml::core::to_xtensor(probe);
    out.dQ = ttml::core::to_xtensor(grad_query);
    out.dK = ttml::core::to_xtensor(grad_key);
    out.dV = ttml::core::to_xtensor(grad_value);
    return out;
}

}  // namespace

// A streak: one row against several columns, so dQ accumulates. dK and dV
// belong to different columns here and are not meaningful, which is exactly
// why the real kernel keeps them in the column buffer and dQ in the packet.
TEST(CyclicPairComputeTest, DqAccumulatesAcrossAStreak) {
    const auto b = make_blocks(4, 64);
    const std::vector<std::pair<uint32_t, uint32_t>> pairs = {{2, 0}, {2, 1}, {2, 2}, {2, 3}};
    const auto out = run_pairs(b, pairs);

    xt::xarray<float> want = xt::zeros<float>({b.B, b.d});
    for (const auto& [row, col] : pairs) {
        want += pair_contribution(b, row, col).dQ;
    }
    expect_close(out.dQ, want, 0.05F, "dQ accumulated over four columns");
}

// A column residency: several rows against one column, so dK and dV
// accumulate.
TEST(CyclicPairComputeTest, ColumnGradientsAccumulateAcrossAResidency) {
    const auto b = make_blocks(4, 64);
    const std::vector<std::pair<uint32_t, uint32_t>> pairs = {{0, 1}, {1, 1}, {2, 1}, {3, 1}};
    const auto out = run_pairs(b, pairs);

    xt::xarray<float> want_dk = xt::zeros<float>({b.B, b.d});
    xt::xarray<float> want_dv = xt::zeros<float>({b.B, b.d});
    for (const auto& [row, col] : pairs) {
        const auto c = pair_contribution(b, row, col);
        want_dk += c.dK;
        want_dv += c.dV;
    }
    expect_close(out.dK, want_dk, 0.05F, "dK accumulated over four rows");
    expect_close(out.dV, want_dv, 0.05F, "dV accumulated over four rows");
}

// The mechanism Algorithm 2 uses at every timestep and the relay uses at
// every streak start: the accumulator does not start at zero, it starts at
// the dQ loaded from DRAM. Isolated here, with one pair and then several, so
// that a failure is in the seeding rather than in the schedule, the barrier,
// or the handoff between three RISCs.
TEST(CyclicPairComputeTest, DqAccumulatesOntoASeedFromMemory) {
    const auto b = make_blocks(4, 64);
    const xt::xarray<float> seed = random_matrix(b.B, b.d, 99) * 0.5F;
    const std::vector<std::pair<uint32_t, uint32_t>> pairs = {{2, 0}};
    const auto out = run_pairs(b, pairs, &seed);

    xt::xarray<float> want = seed;
    want += pair_contribution(b, 2, 0).dQ;
    expect_close(out.dQ, want, 0.05F, "dQ onto a seed, one pair");
}

TEST(CyclicPairComputeTest, DqAccumulatesOntoASeedAcrossAStreak) {
    const auto b = make_blocks(4, 64);
    const xt::xarray<float> seed = random_matrix(b.B, b.d, 98) * 0.5F;
    const std::vector<std::pair<uint32_t, uint32_t>> pairs = {{1, 0}, {1, 1}, {1, 2}};
    const auto out = run_pairs(b, pairs, &seed);

    xt::xarray<float> want = seed;
    for (const auto& [row, col] : pairs) {
        want += pair_contribution(b, row, col).dQ;
    }
    expect_close(out.dQ, want, 0.05F, "dQ onto a seed, three pairs");
}
