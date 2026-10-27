// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <optional>
#include <thread>
#include <tt-metalium/math.hpp>
#include <tuple>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "test_utils/mesh_utils.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/global_semaphore.hpp"

namespace {
using ttnn::experimental::prim::MinimalMatmulConfig;

constexpr uint32_t kClusterAxis = 1;
constexpr uint32_t kCalls = 4;

class AllGatherWeightMinimalMatmulAsyncTest : public ttml::test_utils::MeshFixture {
protected:
    AllGatherWeightMinimalMatmulAsyncTest() : MeshFixture(tt::tt_metal::distributed::MeshShape(1, 4)) {
    }

    void SetUp() override {
        const char* root = std::getenv("TT_METAL_RUNTIME_ROOT");
        root = root != nullptr ? root : std::getenv("TT_METAL_HOME");
        if (std::getenv("TT_MESH_GRAPH_DESC_PATH") == nullptr && root != nullptr) {
            const auto mgd = std::filesystem::path(root) / "tt-train/configs/mgd/bh_qb_1_4_ring_ring.textproto";
            if (std::filesystem::exists(mgd)) {
                setenv("TT_MESH_GRAPH_DESC_PATH", mgd.c_str(), 1);
            }
        }
        MeshFixture::SetUp();
    }
};

struct Shape {
    uint32_t M;
    uint32_t K;
    uint32_t N;
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

const Problem& problem(const Shape& s) {
    static std::map<std::tuple<uint32_t, uint32_t, uint32_t>, Problem> cache;
    auto [it, inserted] = cache.try_emplace({s.M, s.K, s.N});
    if (inserted) {
        Problem& p = it->second;
        p.x = ttml::test_utils::make_uniform_vector<float>(static_cast<size_t>(s.M) * s.K, -1.0F, 1.0F, 42);
        p.w_nk = ttml::test_utils::make_uniform_vector<float>(static_cast<size_t>(s.N) * s.K, -1.0F, 1.0F, 43);
        p.w_kn.resize(p.w_nk.size());
        for (uint32_t n = 0; n < s.N; ++n) {
            for (uint32_t k = 0; k < s.K; ++k) {
                p.w_kn[(static_cast<size_t>(k) * s.N) + n] = p.w_nk[(static_cast<size_t>(n) * s.K) + k];
            }
        }
        p.y = reference_matmul(p.x, p.w_nk, s.M, s.K, s.N);
    }
    return it->second;
}

void run_fabric_gather(
    const Shape& s,
    uint32_t num_links,
    uint32_t num_workers_per_link,
    bool transpose_b,
    ttnn::ccl::Topology topology,
    const std::optional<Block>& block) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t ring_size = device->shape()[kClusterAxis];
    const auto grid = device->compute_with_storage_grid_size();
    std::optional<MinimalMatmulConfig> config;
    if (block.has_value()) {
        const uint32_t comm_rows = tt::div_up(2 * num_links * (1 + num_workers_per_link), grid.x);
        config = MinimalMatmulConfig{
            .M_block_size = block->M_block,
            .K_block_size = block->K_block,
            .N_block_size = block->N_block,
            .subblock_h = block->subblock_h,
            .subblock_w = block->subblock_w,
            .compute_with_storage_grid_size = {grid.x, grid.y - comm_rows}};
    }

    const Problem& p = problem(s);
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    const ttnn::Shape w_shape = transpose_b ? ttnn::Shape({s.N, s.K}) : ttnn::Shape({s.K, s.N});
    const auto x = ttml::core::from_vector(p.x, ttnn::Shape({s.M, s.K}), device, ttnn::Layout::TILE, replicate.get());
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, transpose_b ? 0 : 1, kClusterAxis);
    const auto w =
        ttml::core::from_vector(transpose_b ? p.w_nk : p.w_kn, w_shape, device, ttnn::Layout::TILE, shard.get());
    const std::vector<float> zeros(p.w_nk.size(), 0.0F);
    const std::array<ttnn::Tensor, 2> buffers = {
        ttml::core::from_vector(zeros, w_shape, device, ttnn::Layout::TILE, replicate.get()),
        ttml::core::from_vector(zeros, w_shape, device, ttnn::Layout::TILE, replicate.get())};
    const tt::tt_metal::CoreRangeSet cores(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
    for (uint32_t i = 0; i < ring_size; ++i) {
        semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(device, cores, 0));
    }

    std::vector<ttnn::Tensor> outputs;
    for (uint32_t i = 0; i < kCalls; ++i) {
        outputs.push_back(ttml::metal::all_gather_weight_minimal_matmul_async(
            x,
            w,
            ring_size,
            0,
            topology,
            config,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            buffers[i % 2],
            semaphores,
            kClusterAxis,
            num_links,
            num_workers_per_link,
            8,
            transpose_b));
    }
    for (uint32_t i = 0; i < kCalls; ++i) {
        for (const auto& device_y : ttnn::distributed::get_device_tensors(outputs[i])) {
            SCOPED_TRACE(testing::Message() << "call " << i);
            expect_close(p.y, ttml::core::to_vector(device_y));
        }
    }
}

void sweep(const Shape& s) {
    const std::array<std::pair<uint32_t, uint32_t>, 4> links_workers = {{{1, 1}, {2, 1}, {1, 2}, {2, 3}}};
    const std::array<std::optional<Block>, 3> blocks = {std::nullopt, Block{8, 8, 12, 2, 2}, Block{8, 8, 6, 1, 3}};
    for (const auto& [num_links, num_workers] : links_workers) {
        for (const bool transpose_b : {true, false}) {
            for (const auto topology : {ttnn::ccl::Topology::Ring, ttnn::ccl::Topology::Linear}) {
                for (const auto& block : blocks) {
                    SCOPED_TRACE(
                        testing::Message()
                        << "links=" << num_links << " workers=" << num_workers << " transpose_b=" << transpose_b
                        << " ring=" << (topology == ttnn::ccl::Topology::Ring) << " block="
                        << (block.has_value()
                                ? std::to_string(block->N_block) + "/" + std::to_string(block->subblock_h) + "x" +
                                      std::to_string(block->subblock_w)
                                : std::string("default")));
                    run_fabric_gather(s, num_links, num_workers, transpose_b, topology, block);
                }
            }
        }
    }
}
}  // namespace

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, Small) {
    sweep({256, 512, 1024});
}

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, W1) {
    sweep({2048, 4096, 11008});
}

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, W2) {
    sweep({2048, 11008, 4096});
}

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, TransposedGrid) {
    sweep({8192, 1024, 2048});
}
