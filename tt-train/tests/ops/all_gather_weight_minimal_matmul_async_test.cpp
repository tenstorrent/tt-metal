// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "test_utils/random_data.hpp"

namespace {
using ttnn::experimental::prim::MinimalMatmulConfig;

class AllGatherWeightMinimalMatmulAsyncTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

struct Block {
    uint32_t M_block;
    uint32_t K_block;
    uint32_t N_block;
    uint32_t subblock_h;
    uint32_t subblock_w;
};

std::vector<float> reference_matmul(
    const std::vector<float>& x, const std::vector<float>& w_nk, uint32_t M, uint32_t K, uint32_t N) {
    std::vector<float> y(static_cast<size_t>(M) * N);
    const uint32_t num_threads = std::max(1U, std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (uint32_t m = t; m < M; m += num_threads) {
                const float* x_row = &x[static_cast<size_t>(m) * K];
                for (uint32_t n = 0; n < N; ++n) {
                    const float* w_row = &w_nk[static_cast<size_t>(n) * K];
                    std::array<float, 8> acc{};
                    uint32_t k = 0;
                    for (; k + 8 <= K; k += 8) {
                        for (uint32_t j = 0; j < 8; ++j) {
                            acc[j] += x_row[k + j] * w_row[k + j];
                        }
                    }
                    float sum = 0.0F;
                    for (const float a : acc) {
                        sum += a;
                    }
                    for (; k < K; ++k) {
                        sum += x_row[k] * w_row[k];
                    }
                    y[(static_cast<size_t>(m) * N) + n] = sum;
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return y;
}

void expect_close(const std::vector<float>& ref, const std::vector<float>& out) {
    ASSERT_EQ(ref.size(), out.size());
    double mean_r = 0.0;
    double mean_o = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        mean_r += ref[i];
        mean_o += out[i];
    }
    mean_r /= static_cast<double>(ref.size());
    mean_o /= static_cast<double>(ref.size());
    double cov = 0.0;
    double var_r = 0.0;
    double var_o = 0.0;
    double diff = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = ref[i] - mean_r;
        const double o = out[i] - mean_o;
        cov += r * o;
        var_r += r * r;
        var_o += o * o;
        diff += (out[i] - ref[i]) * static_cast<double>(out[i] - ref[i]);
        norm += static_cast<double>(ref[i]) * ref[i];
    }
    EXPECT_GT(cov / std::sqrt(var_r * var_o), 0.9995);
    EXPECT_LT(std::sqrt(diff / norm), 0.02);
}

struct Problem {
    std::vector<float> x;
    std::vector<float> w_nk;
    std::vector<float> w_kn;
    std::vector<float> y;
};

const Problem& problem(uint32_t M, uint32_t K, uint32_t N) {
    static std::map<std::tuple<uint32_t, uint32_t, uint32_t>, Problem> cache;
    auto [it, inserted] = cache.try_emplace({M, K, N});
    if (inserted) {
        Problem& p = it->second;
        p.x = ttml::test_utils::make_uniform_vector<float>(static_cast<size_t>(M) * K, -1.0F, 1.0F, 42);
        p.w_nk = ttml::test_utils::make_uniform_vector<float>(static_cast<size_t>(N) * K, -1.0F, 1.0F, 43);
        p.w_kn.resize(p.w_nk.size());
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t k = 0; k < K; ++k) {
                p.w_kn[(static_cast<size_t>(k) * N) + n] = p.w_nk[(static_cast<size_t>(n) * K) + k];
            }
        }
        p.y = reference_matmul(p.x, p.w_nk, M, K, N);
    }
    return it->second;
}

struct RunArgs {
    uint32_t ring_size = 4;
    uint32_t ring_index = 0;
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring;
    bool fp32 = false;
    bool fp32_acc = true;
    std::optional<Block> block;
    std::optional<tt::tt_metal::CoreCoord> core_grid;
    bool transpose_b = true;
};

void run_striped_linear(uint32_t M, uint32_t K, uint32_t N, const RunArgs& args) {
    auto* device = &ttml::autograd::ctx().get_device();
    const Problem& p = problem(M, K, N);
    const auto& w_data = args.transpose_b ? p.w_nk : p.w_kn;
    const ttnn::Shape w_shape = args.transpose_b ? ttnn::Shape({N, K}) : ttnn::Shape({K, N});
    const auto x = args.fp32 ? ttml::core::from_vector<float, ttnn::DataType::FLOAT32>(p.x, ttnn::Shape({M, K}), device)
                             : ttml::core::from_vector(p.x, ttnn::Shape({M, K}), device);
    const auto w = args.fp32 ? ttml::core::from_vector<float, ttnn::DataType::FLOAT32>(w_data, w_shape, device)
                             : ttml::core::from_vector(w_data, w_shape, device);
    const auto compute_config = ttnn::init_device_compute_kernel_config(
        device->arch(), std::nullopt, tt::tt_metal::MathFidelity::HiFi2, false, args.fp32_acc, true);
    std::optional<MinimalMatmulConfig> config;
    if (args.block.has_value() || args.core_grid.has_value()) {
        const Block b = args.block.value_or(Block{8, 8, 8, 2, 2});
        config = MinimalMatmulConfig{
            .M_block_size = b.M_block,
            .K_block_size = b.K_block,
            .N_block_size = b.N_block,
            .subblock_h = b.subblock_h,
            .subblock_w = b.subblock_w,
            .compute_with_storage_grid_size = args.core_grid.value_or(device->compute_with_storage_grid_size())};
    }
    const auto y = ttml::metal::all_gather_weight_minimal_matmul_async(
        x,
        w,
        args.ring_size,
        args.ring_index,
        args.topology,
        config,
        std::nullopt,
        std::nullopt,
        compute_config,
        args.transpose_b);
    ASSERT_EQ(y.logical_shape(), ttnn::Shape({M, N}));
    expect_close(p.y, ttml::core::to_vector(y));
}
}  // namespace

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, FsdpShapes) {
    const std::array<std::array<uint32_t, 3>, 4> shapes = {
        {{2048, 4096, 11008}, {2048, 11008, 4096}, {2048, 4096, 4096}, {2048, 4096, 8192}}};
    for (const auto& [M, K, N] : shapes) {
        for (uint32_t ring_index = 0; ring_index < 4; ++ring_index) {
            for (const bool transpose_b : {true, false}) {
                SCOPED_TRACE(
                    testing::Message() << M << "x" << K << "x" << N << " ring_index=" << ring_index
                                       << " transpose_b=" << transpose_b);
                run_striped_linear(M, K, N, {.ring_index = ring_index, .transpose_b = transpose_b});
            }
        }
    }
}

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, RingSizes) {
    const std::array<std::pair<uint32_t, uint32_t>, 10> rings = {
        {{1, 0}, {2, 0}, {2, 1}, {3, 1}, {4, 3}, {8, 5}, {16, 0}, {32, 0}, {32, 17}, {32, 31}}};
    for (const auto& [ring_size, ring_index] : rings) {
        for (const auto topology : {ttnn::ccl::Topology::Ring, ttnn::ccl::Topology::Linear}) {
            SCOPED_TRACE(
                testing::Message() << "ring_size=" << ring_size << " ring_index=" << ring_index
                                   << " ring=" << (topology == ttnn::ccl::Topology::Ring));
            run_striped_linear(
                256,
                512,
                ring_size == 3 ? 1056 : 1024,
                {.ring_size = ring_size, .ring_index = ring_index, .topology = topology});
        }
    }
}

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, UnevenPieces) {
    const std::array<std::tuple<uint32_t, uint32_t, Block>, 5> cases = {{
        {11008, 4, {8, 8, 8, 2, 2}},
        {11008, 4, {4, 4, 4, 2, 2}},
        {2688, 4, {4, 4, 16, 2, 2}},
        {96 * 32, 3, {8, 8, 8, 2, 2}},
        {64, 2, {2, 2, 2, 1, 1}},
    }};
    for (const auto& [N, ring_size, block] : cases) {
        for (const bool transpose_b : {true, false}) {
            SCOPED_TRACE(
                testing::Message() << "N=" << N << " ring_size=" << ring_size << " N_block=" << block.N_block
                                   << " transpose_b=" << transpose_b);
            run_striped_linear(
                512,
                512,
                N,
                {.ring_size = ring_size, .ring_index = ring_size - 1, .block = block, .transpose_b = transpose_b});
        }
    }
}

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, CoreGrid) {
    for (const auto grid :
         {tt::tt_metal::CoreCoord{2, 2},
          tt::tt_metal::CoreCoord{2, 4},
          tt::tt_metal::CoreCoord{4, 2},
          tt::tt_metal::CoreCoord{3, 5}}) {
        SCOPED_TRACE(testing::Message() << "grid=" << grid.x << "x" << grid.y);
        run_striped_linear(512, 512, 2048, {.ring_index = 2, .block = Block{2, 2, 2, 1, 1}, .core_grid = grid});
    }
}

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, TransposedGrid) {
    for (const uint32_t ring_index : {0U, 3U}) {
        SCOPED_TRACE(testing::Message() << "ring_index=" << ring_index);
        run_striped_linear(8192, 4096, 2048, {.ring_index = ring_index});
    }
}

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, Dtypes) {
    for (const bool fp32 : {false, true}) {
        for (const bool fp32_acc : {true, false}) {
            SCOPED_TRACE(testing::Message() << "fp32=" << fp32 << " fp32_acc=" << fp32_acc);
            run_striped_linear(
                512, 512, 1024, {.ring_index = 1, .fp32 = fp32, .fp32_acc = fp32_acc, .block = Block{2, 2, 2, 1, 2}});
        }
    }
}

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, ProgramCache) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();
    const Problem& p = problem(256, 256, 1024);
    const auto x = ttml::core::from_vector(p.x, ttnn::Shape({256, 256}), device);
    const auto w = ttml::core::from_vector(p.w_nk, ttnn::Shape({1024, 256}), device);
    const auto entries = device->num_program_cache_entries();
    (void)ttml::metal::all_gather_weight_minimal_matmul_async(x, w, 4, 0);
    EXPECT_EQ(device->num_program_cache_entries(), entries + 1);
    (void)ttml::metal::all_gather_weight_minimal_matmul_async(x, w, 4, 0);
    EXPECT_EQ(device->num_program_cache_entries(), entries + 1);
    (void)ttml::metal::all_gather_weight_minimal_matmul_async(x, w, 4, 2);
    EXPECT_EQ(device->num_program_cache_entries(), entries + 2);
}

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, RejectsUnalignedStripes) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto x = ttml::core::from_vector(std::vector<float>(64 * 64, 1.0F), ttnn::Shape({64, 64}), device);
    const auto w = ttml::core::from_vector(std::vector<float>(96 * 64, 1.0F), ttnn::Shape({96, 64}), device);
    try {
        (void)ttml::metal::all_gather_weight_minimal_matmul_async(x, w, 2, 0);
        FAIL() << "expected a validation error";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("tile-aligned stripe"), std::string::npos) << e.what();
    }
}
