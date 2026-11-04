// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <thread>
#include <tt-metalium/math.hpp>
#include <tt-umd/cluster.hpp>
#include <vector>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/core/xmath.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "test_utils/mesh_utils.hpp"
#include "test_utils/random_data.hpp"
#include "test_utils/relative_l2.hpp"
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
        const auto cluster = tt::umd::Cluster::create_cluster_descriptor();
        if (std::getenv(kMgdEnv) == nullptr && root != nullptr && cluster->get_arch() == tt::ARCH::BLACKHOLE) {
            const auto mgd = std::filesystem::path(root) / "tt-train/configs/mgd/bh_qb_1_4_ring_ring.textproto";
            if (std::filesystem::exists(mgd)) {
                setenv(kMgdEnv, mgd.c_str(), 1);
                m_set_mgd = true;
            }
        }
        MeshFixture::SetUp();
    }

    void TearDown() override {
        MeshFixture::TearDown();
        if (m_set_mgd) {
            unsetenv(kMgdEnv);
        }
    }

    static constexpr const char* kMgdEnv = "TT_MESH_GRAPH_DESC_PATH";
    bool m_set_mgd = false;
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

struct Problem {
    Shape shape;
    std::vector<float> x;
    std::vector<float> w_nk;
    std::vector<float> w_kn;
    xt::xarray<float> y;
};

Problem make_problem(const Shape& s) {
    Problem p{.shape = s};
    p.x = ttml::test_utils::make_uniform_vector<float>(static_cast<size_t>(s.M) * s.K, -1.0F, 1.0F, 42);
    p.w_nk = ttml::test_utils::make_uniform_vector<float>(static_cast<size_t>(s.N) * s.K, -1.0F, 1.0F, 43);
    p.w_kn.resize(p.w_nk.size());
    for (uint32_t n = 0; n < s.N; ++n) {
        for (uint32_t k = 0; k < s.K; ++k) {
            p.w_kn[(static_cast<size_t>(k) * s.N) + n] = p.w_nk[(static_cast<size_t>(n) * s.K) + k];
        }
    }
    p.y = xt::adapt(reference_matmul(p.x, p.w_nk, s.M, s.K, s.N), std::vector<size_t>{s.M, s.N});
    return p;
}

double pcc(const xt::xarray<float>& a, const xt::xarray<float>& b) {
    const xt::xarray<double> da = xt::cast<double>(a) - xt::mean(xt::cast<double>(a))();
    const xt::xarray<double> db = xt::cast<double>(b) - xt::mean(xt::cast<double>(b))();
    return xt::sum(da * db)() / std::sqrt(xt::sum(da * da)() * xt::sum(db * db)());
}

void expect_close(const xt::xarray<float>& ref, const xt::xarray<float>& out) {
    ASSERT_EQ(ref.shape(), out.shape());
    EXPECT_GT(pcc(ref, out), 0.9995);
    EXPECT_LT(ttml::test_utils::relative_l2(out, ref), 0.02);
}

std::vector<tt::tt_metal::GlobalSemaphore> make_semaphores(ttnn::distributed::MeshDevice* device) {
    const auto grid = device->compute_with_storage_grid_size();
    const tt::tt_metal::CoreRangeSet cores(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
    for (uint32_t i = 0; i < device->shape()[kClusterAxis]; ++i) {
        semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(device, cores, 0));
    }
    return semaphores;
}

void run_fabric_gather(
    const Problem& p,
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    uint32_t num_links,
    uint32_t num_workers_per_link,
    bool transpose_b,
    ttnn::ccl::Topology topology,
    const std::optional<Block>& block) {
    auto* device = &ttml::autograd::ctx().get_device();
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

    const auto& s = p.shape;
    const ttnn::Shape w_shape = transpose_b ? ttnn::Shape({s.N, s.K}) : ttnn::Shape({s.K, s.N});
    const std::array<ttnn::Tensor, 2> buffers = {
        ttml::core::zeros(w_shape, device), ttml::core::zeros(w_shape, device)};
    const auto semaphores = make_semaphores(device);

    std::vector<ttnn::Tensor> outputs;
    for (uint32_t i = 0; i < kCalls; ++i) {
        outputs.push_back(ttml::metal::all_gather_weight_minimal_matmul_async(
            x,
            w,
            buffers[i % 2],
            semaphores,
            kClusterAxis,
            topology,
            config,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            num_links,
            num_workers_per_link,
            8,
            transpose_b));
    }
    for (uint32_t i = 0; i < kCalls; ++i) {
        for (const auto& device_y : ttnn::distributed::get_device_tensors(outputs[i])) {
            SCOPED_TRACE(testing::Message() << "call " << i);
            xt::xarray<float> y = ttml::core::to_xtensor(device_y);
            y.reshape({p.shape.M, p.shape.N});
            expect_close(p.y, y);
        }
    }
}

void sweep(const Shape& s) {
    const std::array<std::pair<uint32_t, uint32_t>, 4> links_workers = {{{1, 1}, {2, 1}, {1, 2}, {2, 3}}};
    const std::array<std::optional<Block>, 3> blocks = {std::nullopt, Block{8, 8, 12, 2, 2}, Block{8, 8, 6, 1, 3}};
    auto* device = &ttml::autograd::ctx().get_device();
    const Problem p = make_problem(s);
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    const auto x = ttml::core::from_vector(p.x, ttnn::Shape({s.M, s.K}), device, ttnn::Layout::TILE, replicate.get());
    for (const bool transpose_b : {true, false}) {
        const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, transpose_b ? 0 : 1, kClusterAxis);
        const ttnn::Shape w_shape = transpose_b ? ttnn::Shape({s.N, s.K}) : ttnn::Shape({s.K, s.N});
        const auto w =
            ttml::core::from_vector(transpose_b ? p.w_nk : p.w_kn, w_shape, device, ttnn::Layout::TILE, shard.get());
        for (const auto& [num_links, num_workers] : links_workers) {
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
                    run_fabric_gather(p, x, w, num_links, num_workers, transpose_b, topology, block);
                }
            }
        }
    }
}
}  // namespace

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, SharedSemaphoresAcrossShapes) {
    auto* device = &ttml::autograd::ctx().get_device();
    const std::array<Shape, 3> shapes = {{{256, 512, 1024}, {256, 1024, 512}, {256, 512, 2048}}};
    struct Layer {
        Problem p;
        ttnn::Tensor x;
        ttnn::Tensor w;
        std::array<ttnn::Tensor, 2> buffers;
    };
    std::vector<Layer> layers;
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0, kClusterAxis);
    for (const auto& s : shapes) {
        Problem p = make_problem(s);
        auto x = ttml::core::from_vector(p.x, ttnn::Shape({s.M, s.K}), device, ttnn::Layout::TILE, replicate.get());
        auto w = ttml::core::from_vector(p.w_nk, ttnn::Shape({s.N, s.K}), device, ttnn::Layout::TILE, shard.get());
        const ttnn::Shape full({s.N, s.K});
        layers.push_back(
            {std::move(p),
             std::move(x),
             std::move(w),
             {ttml::core::zeros(full, device), ttml::core::zeros(full, device)}});
    }
    const auto semaphores = make_semaphores(device);

    for (const auto& [num_links, num_workers] : std::array<std::pair<uint32_t, uint32_t>, 2>{{{2, 3}, {1, 2}}}) {
        SCOPED_TRACE(testing::Message() << "links=" << num_links << " workers=" << num_workers);
        constexpr uint32_t kRounds = 3;
        std::vector<std::pair<uint32_t, ttnn::Tensor>> outputs;
        for (uint32_t round = 0; round < kRounds; ++round) {
            for (uint32_t l = 0; l < layers.size(); ++l) {
                outputs.emplace_back(
                    l,
                    ttml::metal::all_gather_weight_minimal_matmul_async(
                        layers[l].x,
                        layers[l].w,
                        layers[l].buffers[round % 2],
                        semaphores,
                        kClusterAxis,
                        ttnn::ccl::Topology::Ring,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        num_links,
                        num_workers));
            }
        }
        for (uint32_t i = 0; i < outputs.size(); ++i) {
            const auto& [l, y] = outputs[i];
            for (const auto& device_y : ttnn::distributed::get_device_tensors(y)) {
                SCOPED_TRACE(testing::Message() << "call " << i << " layer " << l);
                expect_close(layers[l].p.y, ttml::core::to_xtensor(device_y));
            }
        }
    }
}

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, CacheHitWithOtherSemaphores) {
    auto* device = &ttml::autograd::ctx().get_device();
    const Shape s{256, 512, 1024};
    const Problem p = make_problem(s);
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0, kClusterAxis);
    const auto x = ttml::core::from_vector(p.x, ttnn::Shape({s.M, s.K}), device, ttnn::Layout::TILE, replicate.get());
    const auto w = ttml::core::from_vector(p.w_nk, ttnn::Shape({s.N, s.K}), device, ttnn::Layout::TILE, shard.get());
    const std::array<std::vector<tt::tt_metal::GlobalSemaphore>, 2> semaphores = {
        make_semaphores(device), make_semaphores(device)};
    for (uint32_t i = 0; i < 2; ++i) {
        SCOPED_TRACE(testing::Message() << "call " << i);
        const auto y = ttml::metal::all_gather_weight_minimal_matmul_async(
            x,
            w,
            ttml::core::zeros(ttnn::Shape({s.N, s.K}), device),
            semaphores[i],
            kClusterAxis,
            ttnn::ccl::Topology::Ring,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            2,
            3);
        for (const auto& device_y : ttnn::distributed::get_device_tensors(y)) {
            expect_close(p.y, ttml::core::to_xtensor(device_y));
        }
        for (const auto& semaphore : semaphores[0]) {
            semaphore.reset_semaphore_value(1000);
        }
    }
}

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, LargeBatch4D) {
    constexpr uint32_t B = 4;
    constexpr uint32_t S = 2048;
    auto* device = &ttml::autograd::ctx().get_device();
    const Problem p = make_problem({B * S, 4096, 11008});
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    const auto x =
        ttml::core::from_vector(p.x, ttnn::Shape({B, 1, S, p.shape.K}), device, ttnn::Layout::TILE, replicate.get());
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0, kClusterAxis);
    const auto w =
        ttml::core::from_vector(p.w_nk, ttnn::Shape({p.shape.N, p.shape.K}), device, ttnn::Layout::TILE, shard.get());
    for (const auto topology : {ttnn::ccl::Topology::Ring, ttnn::ccl::Topology::Linear}) {
        SCOPED_TRACE(testing::Message() << "ring=" << (topology == ttnn::ccl::Topology::Ring));
        run_fabric_gather(p, x, w, 2, 3, true, topology, std::nullopt);
    }
}

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, NarrowStripes4D) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0, kClusterAxis);
    for (const auto& s : std::array<Shape, 2>{{{256, 512, 256}, {256, 256, 512}}}) {
        const Problem p = make_problem(s);
        const auto x = ttml::core::from_vector(
            p.x, ttnn::Shape({2, 1, s.M / 2, s.K}), device, ttnn::Layout::TILE, replicate.get());
        const auto w =
            ttml::core::from_vector(p.w_nk, ttnn::Shape({s.N, s.K}), device, ttnn::Layout::TILE, shard.get());
        for (const auto& [num_links, num_workers] : std::array<std::pair<uint32_t, uint32_t>, 2>{{{2, 3}, {1, 1}}}) {
            SCOPED_TRACE(
                testing::Message() << "K=" << s.K << " N=" << s.N << " links=" << num_links
                                   << " workers=" << num_workers);
            run_fabric_gather(p, x, w, num_links, num_workers, true, ttnn::ccl::Topology::Ring, std::nullopt);
        }
    }
}

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, TinyShapes) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0, kClusterAxis);
    for (const uint32_t K : {32U, 96U, 256U}) {
        for (const uint32_t N : {128U, 256U, 384U}) {
            const Shape s{128, K, N};
            const Problem p = make_problem(s);
            const auto x =
                ttml::core::from_vector(p.x, ttnn::Shape({s.M, s.K}), device, ttnn::Layout::TILE, replicate.get());
            const auto w =
                ttml::core::from_vector(p.w_nk, ttnn::Shape({s.N, s.K}), device, ttnn::Layout::TILE, shard.get());
            for (const auto& [num_links, num_workers] :
                 std::array<std::pair<uint32_t, uint32_t>, 2>{{{2, 3}, {1, 1}}}) {
                for (const auto topology : {ttnn::ccl::Topology::Ring, ttnn::ccl::Topology::Linear}) {
                    const bool ring = topology == ttnn::ccl::Topology::Ring;
                    SCOPED_TRACE(
                        testing::Message() << "K=" << K << " N=" << N << " links=" << num_links
                                           << " workers=" << num_workers << " ring=" << ring);
                    run_fabric_gather(p, x, w, num_links, num_workers, true, topology, std::nullopt);
                }
            }
        }
    }
}

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
