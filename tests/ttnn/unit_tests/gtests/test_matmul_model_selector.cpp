// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device-free tests of the opt-in matmul config model selector (matmul_model_selector.hpp): loading, ranking and
// the pick cache.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "ttnn/operations/matmul/device/config/model/matmul_candidates.hpp"
#include "ttnn/operations/matmul/device/config/model/matmul_model_selector.hpp"

namespace ttnn::operations::matmul::model {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr std::uint32_t kBudget = 1461120 - 16 * 1024;  // a free Blackhole p150 L1 bank less the headroom

MatmulDesc desc(std::uint32_t Mt, std::uint32_t Kt, std::uint32_t Nt) {
    MatmulDesc p;
    p.Mt = Mt;
    p.Kt = Kt;
    p.Nt = Nt;
    return p;
}

TEST(MatmulModelSelector, RankIsStableArgminOverFinite) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> pred = {5.0, nan, 3.0, 3.0, std::numeric_limits<double>::infinity(), 4.0};
    EXPECT_EQ(rank_by_prediction(pred), (std::vector<std::size_t>{2, 3, 5, 0}));
    EXPECT_TRUE(rank_by_prediction(std::vector<double>{nan}).empty());
}

TEST(MatmulModelSelector, PickCacheKeepsFirstAndIsThreadSafe) {
    PickCache cache;
    EXPECT_FALSE(cache.find("k").has_value());
    Pick fallback;
    fallback.description = "fallback";
    EXPECT_EQ(cache.insert("k", fallback).description, "fallback");
    Pick other;
    other.candidate = Candidate{};
    other.description = "other";
    EXPECT_EQ(cache.insert("k", other).description, "fallback");  // the first decision stands
    EXPECT_FALSE(cache.find("k")->candidate.has_value());
    std::vector<std::thread> threads;
    threads.reserve(8);
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&cache, t] {
            for (int i = 0; i < 200; ++i) {
                const std::string key = std::to_string(i % 50);
                if (!cache.find(key)) {
                    Pick p;
                    p.cb_bytes = t;
                    cache.insert(key, p);
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    EXPECT_EQ(cache.size(), 51u);
    cache.clear();
    EXPECT_EQ(cache.size(), 0u);
}

TEST(MatmulModelSelector, LoadFailuresAreReportedNotThrown) {
    std::string error;
    EXPECT_EQ(ConfigModel::load_file("", error), nullptr);
    EXPECT_NE(error, "");
    error.clear();
    EXPECT_EQ(ConfigModel::load_file("/nonexistent/matmul_model.blob", error), nullptr);
    EXPECT_NE(error, "");
    const char junk[] = "not a model";
    error.clear();
    EXPECT_EQ(ConfigModel::from_blob(std::as_bytes(std::span<const char>(junk)), error), nullptr);
    EXPECT_NE(error, "");
}

// The real model: TT_MATMUL_MODEL_BLOB, else the blob the build fetched into runtime/ under the runtime root, which is
// TT_METAL_RUNTIME_ROOT (as for ttnn's default matmul_model_path), else TT_METAL_HOME, else the repo root the tests run
// from. Read from the environment, as RunTimeOptions does, since these tests need no device.
std::filesystem::path model_blob_path() {
    if (const char* path = std::getenv("TT_MATMUL_MODEL_BLOB")) {
        return path;
    }
    std::filesystem::path root = ".";
    for (const char* var : {"TT_METAL_RUNTIME_ROOT", "TT_METAL_HOME"}) {
        if (const char* dir = std::getenv(var)) {
            root = dir;
            break;
        }
    }
    return root / "runtime/matmul_model/matmul_model.blob";
}

TEST(MatmulModelSelector, ScoresCandidatesWithTheModel) {
    const std::filesystem::path path = model_blob_path();
    if (!std::filesystem::exists(path)) {
        GTEST_SKIP() << "no matmul config model blob at " << path << " (built with -DTT_MATMUL_MODEL_FETCH=OFF?)";
    }
    std::string error;
    const auto model = ConfigModel::load_file(path.string(), error);
    ASSERT_NE(model, nullptr) << error;
    EXPECT_EQ(model->sha256().size(), 64u);
    MatmulDesc p = desc(32, 128, 128);
    p.in1 = DataType::kBfloat8B;
    const auto cs = enumerate_candidates(p, HardwareDesc::blackhole(11, 10, kBudget));
    MatmulProblem prob;
    prob.M = 1024;
    prob.K = 4096;
    prob.N = 4096;
    prob.b_dtype = DataType::kBfloat8B;
    prob.fidelity = Fidelity::kHiFi2;
    prob.packer_l1_acc = true;
    const auto pred = model->predict_ns(prob, cs);
    ASSERT_EQ(pred.size(), cs.size());
    const auto order = rank_by_prediction(pred);
    ASSERT_FALSE(order.empty());
    for (std::size_t i : order) {
        EXPECT_GE(pred[i], pred[order.front()]);
        EXPECT_GT(pred[i], 0.0);
    }
}

// Regression goldens: for a few matmuls, the enumerator's first candidate and that candidate's base latency, features
// and prediction, as this code computes them for the shipped blob. A change to the enumerator, the analytic equation or
// the device constants moves them, and then the model has to be retrained with the change (README, "The model blob").
// Compared within 1e-12 relative, so FMA contraction differences between compilers (about 1e-16) pass.
struct Golden {
    std::size_t candidates = 0;
    Family family = Family::k2D;
    Blocking blocking;
    std::uint32_t grid_x = 0, grid_y = 0;
    bool fuse_batch = false;
    double base_ns = 0.0, predicted_ns = 0.0;
    std::array<double, kNumFeatures> features{};
};

struct GoldenRow {
    const char* name;
    MatmulDesc desc;
    MatmulProblem problem;
    std::uint32_t grid_x, grid_y;
    Arch arch;
    Golden golden;
};

std::vector<GoldenRow> golden_rows() {
    std::vector<GoldenRow> rows;
    {
        MatmulDesc d;
        d.Mt = 32;
        d.Kt = 128;
        d.Nt = 128;
        d.in1 = DataType::kBfloat8B;
        d.fidelity = Fidelity::kHiFi2;
        d.packer_l1_acc = true;
        MatmulProblem p;
        p.M = 1024;
        p.K = 4096;
        p.N = 4096;
        p.b_dtype = DataType::kBfloat8B;
        p.has_out_dtype = true;
        p.fidelity = Fidelity::kHiFi2;
        p.packer_l1_acc = true;
        rows.push_back({
            "prefill 1024x4096x4096 bf16 x bfp8, HiFi2, packer L1 acc, p150",
            d,
            p,
            11,
            10,
            Arch::kBlackholeP150,
            Golden{
                78,
                Family::k2D,
                {4, 12, 16, 4, 12, 2, 4},
                11,
                10,
                true,
                0x1.6a4f0d2403b58p+17,
                0x1.69981a1e71d52p+17,
                {0x0p+0,
                 0x1p+4,
                 0x1p+3,
                 0x1.193ea7aad030bp+1,
                 0x1p+2,
                 0x1.8p+3,
                 0x1p+1,
                 0x1p+2,
                 0x1.8p+2,
                 0x1p+0,
                 0x1.6p+6,
                 0x1.4p+3,
                 0x1.cp+2,
                 0x1p+3,
                 0x1.6p+3,
                 0x0p+0,
                 0x0p+0,
                 0x0p+0,
                 0x1p+11,
                 0x1.1p+10,
                 0x1p+11,
                 0x1p+1,
                 0x1p+0,
                 0x1p+0,
                 0x0p+0,
                 0x1p+3,
                 0x1p+3,
                 0x1.4cp+8,
                 0x1.a516cc3e5424p-2,
                 0x1.2150695977226p-3,
                 -0x1.5960419cb7104p-1,
                 0x1.4e46b786ce898p+2,
                 0x1p+0,
                 0x1p+5,
                 0x1p+7,
                 0x1p+7,
                 0x1.10a1ef6bdac7ap+3,
                 0x1.f35eeeed11d21p+2,
                 0x1.23f785b1f5c8ep+3,
                 0x1.066173db1022cp+3,
                 0x1.2af277c9819b1p+3,
                 0x1.1dd2f4cd7c8d9p+3,
                 0x1.443df5e895a1ep+3,
                 0x1.4f0158bb669d4p+3,
                 0x1.98253731597c5p-3,
                 0x1.8e6713ace7a7fp-4,
                 0x1.75714f25eb5e8p-2,
                 0x1.283df2e6d80bcp-3,
                 0x1.d07ae8620da45p-2,
                 0x1.3435777fef615p-2,
                 0x1.fffac9c2f43e3p-1,
                 0x0p+0,
                 0x1.2e88a679ed93p-1,
                 -0x1.335808d8c8a27p+0,
                 0x1.5960419cb7108p-1,
                 0x1.8a745b46f8796p+3,
                 0x1.d5f7e2d8c70aep-4}},
        });
    }
    {
        MatmulDesc d;
        d.Mt = 1;
        d.Kt = 64;
        d.Nt = 256;
        d.in0 = d.in1 = d.out = DataType::kBfloat8B;
        d.fidelity = Fidelity::kLoFi;
        d.packer_l1_acc = false;
        MatmulProblem p;
        p.M = 32;
        p.K = 2048;
        p.N = 8192;
        p.a_dtype = p.b_dtype = p.out_dtype = DataType::kBfloat8B;
        p.has_out_dtype = true;
        p.fidelity = Fidelity::kLoFi;
        rows.push_back({
            "decode 32x2048x8192 bfp8 x bfp8 -> bfp8, LoFi, no packer L1 acc, Galaxy",
            d,
            p,
            12,
            10,
            Arch::kBlackholeGalaxy,
            Golden{
                40,
                Family::k1DMcastIn0,
                {1, 3, 4, 1, 3, 1, 3},
                12,
                10,
                true,
                0x1.7cf03e01994cep+15,
                0x1.7592f54b96dabp+15,
                {0x1p+0,
                 0x1p+2,
                 0x1p+4,
                 0x1.6aa6bc1fa7f7ap+1,
                 0x1p+0,
                 0x1.8p+1,
                 0x1p+0,
                 0x1.8p+1,
                 0x1p+0,
                 0x1p+0,
                 0x1.58p+6,
                 0x1.54p+6,
                 0x0p+0,
                 0x1p+0,
                 0x1.58p+6,
                 0x0p+0,
                 0x0p+0,
                 0x0p+0,
                 0x1.1p+10,
                 0x1.1p+10,
                 0x1.1p+10,
                 0x1p+0,
                 0x1.ep+3,
                 0x1p+0,
                 0x0p+0,
                 0x1p+2,
                 0x1p+3,
                 0x1.1p+4,
                 0x1.e74b378386d04p-1,
                 0x1.0155b8305d4b5p-5,
                 0x1.431b35c9ff3dfp+0,
                 0x1.f1865bb62154dp+1,
                 0x1p+0,
                 0x1p+0,
                 0x1p+6,
                 0x1p+8,
                 0x1.9b44a05b89775p+2,
                 0x1.c0b579ed4e03cp+2,
                 0x1.0801c6b622974p+3,
                 0x0p+0,
                 0x1.082175615b00ep+3,
                 0x1.06343ff6f7583p+3,
                 0x1.bf7c1d5036325p+2,
                 0x1.e769b857d9e74p+2,
                 0x1.48a51fe3d46c3p-3,
                 0x1.272c4b3ab570fp-2,
                 0x1.fde3f5443aa9ap-1,
                 0x0p+0,
                 0x1.ffdde55954b03p-1,
                 0x1.e1f36a1791c18p-1,
                 0x1.21937c6421818p-2,
                 0x1.8p+1,
                 0x1.e1f36a1791c18p-1,
                 -0x1.ed356a63a8b09p-5,
                 0x1.faeab3869a052p-9,
                 0x1.267409c9e01bep+3,
                 -0x1.1fb88278cb149p-1}},
        });
    }
    {
        MatmulDesc d;
        d.batch_a = d.batch_b = 8;
        d.rank_a = d.rank_b = 3;
        d.Mt = 4;
        d.Kt = 2;
        d.Nt = 4;
        d.fidelity = Fidelity::kHiFi4;
        d.fp32_dest_acc = true;
        d.packer_l1_acc = true;
        d.a.in_l1 = true;
        MatmulProblem p;
        p.M = 128;
        p.K = 64;
        p.N = 128;
        p.batch = 8;
        p.has_out_dtype = true;
        p.fidelity = Fidelity::kHiFi4;
        p.fp32_dest_acc = true;
        p.packer_l1_acc = true;
        p.a_mem = MemLayout::kL1Interleaved;
        rows.push_back({
            "batched 8x128x64 @ 8x64x128 bf16, HiFi4, fp32 acc, A in L1, p150",
            d,
            p,
            11,
            10,
            Arch::kBlackholeP150,
            Golden{
                26,
                Family::kReuse,
                {1, 4, 2, 1, 4, 1, 4},
                11,
                10,
                true,
                0x1.0b7b210cff44ep+12,
                0x1.09fba2a9c19f6p+12,
                {0x1.8p+1,
                 0x1p+1,
                 0x1p+0,
                 std::numbers::ln2,
                 0x1p+0,
                 0x1p+2,
                 0x1p+0,
                 0x1p+2,
                 0x1p+0,
                 0x1p+0,
                 0x1p+5,
                 0x0p+0,
                 0x0p+0,
                 0x1p+5,
                 0x1p+5,
                 0x1p+0,
                 0x0p+0,
                 0x0p+0,
                 0x1p+11,
                 0x1p+11,
                 0x1p+12,
                 0x1p+2,
                 0x0p+0,
                 0x1p+0,
                 0x0p+0,
                 0x1p+3,
                 0x1p+3,
                 0x1.4p+4,
                 0x1.4e91b291e26bdp-1,
                 0x1.dba77be71f14ep-3,
                 0x1.be6d72fee529fp+0,
                 0x1.74319a4fb0fe4p+0,
                 0x1p+3,
                 0x1p+2,
                 0x1p+1,
                 0x1p+2,
                 0x1.8d19ece3c88ecp+2,
                 0x0p+0,
                 0x1.df4b232e634e1p+2,
                 0x0p+0,
                 0x1.df4b232e634e1p+2,
                 0x1.ffc8e92554134p+2,
                 0x1.902d8c659ac8dp+2,
                 0x1.bd9e88c2e6d9ep+2,
                 0x1.5495f076922cap-3,
                 0x0p+0,
                 0x1.33ff1f52238d6p-1,
                 0x0p+0,
                 0x1.33ff1f52238d6p-1,
                 0x1.ffd3e1d49cdb8p-1,
                 0x1.656477823b0edp-3,
                 0x1p+2,
                 0x1.a92e99a9bce5ep+0,
                 0x1.03ee2fb786297p-1,
                 0x1.03ee2fb786297p-1,
                 0x1.c9dcf13f226d4p+2,
                 0x1.ce77b88463fa9p-1}},
        });
    }
    return rows;
}

bool near(double got, double want) { return std::abs(got - want) <= 1e-12 * std::max(1.0, std::abs(want)); }

TEST(MatmulModelSelector, GoldenCandidatesAndFeatures) {
    for (const auto& r : golden_rows()) {
        const auto cs = enumerate_candidates(r.desc, HardwareDesc::blackhole(r.grid_x, r.grid_y, kBudget));
        ASSERT_EQ(cs.size(), r.golden.candidates) << r.name;
        const Candidate& c = cs.front();
        EXPECT_EQ(c.family, r.golden.family) << r.name;
        EXPECT_EQ(c.blocking, r.golden.blocking) << r.name;
        EXPECT_EQ(c.grid_x, r.golden.grid_x) << r.name;
        EXPECT_EQ(c.grid_y, r.golden.grid_y) << r.name;
        EXPECT_EQ(c.fuse_batch, r.golden.fuse_batch) << r.name;
        const ModelInput in = compute_model_input(r.problem, to_model_candidate(c), device_params(r.arch));
        ASSERT_EQ(in.status, ScoreStatus::kOk) << r.name;
        EXPECT_TRUE(near(in.base_ns, r.golden.base_ns))
            << r.name << ": base_ns " << in.base_ns << ", golden " << r.golden.base_ns;
        for (std::size_t i = 0; i < kNumFeatures; ++i) {
            EXPECT_TRUE(near(in.features[i], r.golden.features[i]))
                << r.name << ": " << kFeatureNames[i] << " " << in.features[i] << ", golden " << r.golden.features[i];
        }
    }
}

TEST(MatmulModelSelector, GoldenPredictions) {
    const std::filesystem::path path = model_blob_path();
    if (!std::filesystem::exists(path)) {
        GTEST_SKIP() << "no matmul config model blob at " << path << " (built with -DTT_MATMUL_MODEL_FETCH=OFF?)";
    }
    std::string error;
    const auto model = ConfigModel::load_file(path.string(), error);
    ASSERT_NE(model, nullptr) << error;
    for (const auto& r : golden_rows()) {
        const auto cs = enumerate_candidates(r.desc, HardwareDesc::blackhole(r.grid_x, r.grid_y, kBudget));
        ASSERT_FALSE(cs.empty()) << r.name;
        const auto predicted = model->predict_ns(r.problem, std::span<const Candidate>(cs.data(), 1), r.arch);
        EXPECT_TRUE(near(predicted[0], r.golden.predicted_ns))
            << r.name << ": " << predicted[0] << " ns, golden " << r.golden.predicted_ns << " ns";
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace
}  // namespace ttnn::operations::matmul::model
