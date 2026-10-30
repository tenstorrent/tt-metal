// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device-free tests of the opt-in matmul config model selector (matmul_model_selector.hpp): loading, ranking and
// the pick cache.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
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
    ASSERT_TRUE(std::filesystem::exists(path)) << "the matmul config model blob is missing: " << path;
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

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace
}  // namespace ttnn::operations::matmul::model
