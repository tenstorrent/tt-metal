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
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-umd-workload/cluster.hpp>
#include <variant>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/views/xview.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "test_utils/mesh_utils.hpp"
#include "test_utils/random_data.hpp"
#include "test_utils/relative_l2.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace {
using ttnn::experimental::prim::MinimalMatmulConfig;

constexpr uint32_t kClusterAxis = 0;
constexpr uint32_t kRingSize = 4;
constexpr uint32_t kCalls = 4;

class AllGatherWeightMinimalMatmulAsyncTest : public ttml::test_utils::MeshFixture {
protected:
    AllGatherWeightMinimalMatmulAsyncTest() : MeshFixture(tt::tt_metal::distributed::MeshShape(kRingSize, 1)) {
    }

    void SetUp() override {
        const char* root = std::getenv("TT_METAL_RUNTIME_ROOT");
        const auto cluster = tt::umd::Cluster::create_cluster_descriptor();
        // Metal reads the descriptor once per process: on a box too small for the ring it would break every later
        // device test instead of letting these skip.
        if (std::getenv(kMgdEnv) == nullptr && root != nullptr && cluster->get_arch() == tt::ARCH::BLACKHOLE &&
            cluster->get_number_of_chips() >= kRingSize) {
            const auto mgd = std::filesystem::path(root) / "tt-train/configs/mgd/bh_4_1_ring_ring.textproto";
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

struct Problem {
    Shape shape;
    std::vector<float> x;
    std::vector<float> w_nk;
    std::vector<float> w_kn;
    std::vector<std::array<uint32_t, 2>> tiles;
    xt::xarray<float> y;
};

std::vector<std::array<uint32_t, 2>> spot_tiles(const Shape& s, uint32_t ring_size) {
    const uint32_t m_tiles = s.M / tt::constants::TILE_HEIGHT;
    const uint32_t stripe_tiles = s.N / tt::constants::TILE_WIDTH / ring_size;
    std::vector<std::array<uint32_t, 2>> tiles;
    const uint32_t columns = std::min<uint32_t>(stripe_tiles, 12);
    tiles.reserve(static_cast<size_t>(ring_size) * columns * 3);
    for (uint32_t stripe = 0; stripe < ring_size; ++stripe) {
        for (uint32_t j = 0; j < columns; ++j) {
            const uint32_t n = (stripe * stripe_tiles) + (((2 * j) + 1) * stripe_tiles / (2 * columns));
            for (const uint32_t m : {0U, m_tiles / 2, m_tiles - 1}) {
                tiles.push_back({m, n});
            }
        }
    }
    return tiles;
}

Problem make_problem(const Shape& s, uint32_t spot_ring_size = 0) {
    Problem p{.shape = s};
    p.x = ttml::test_utils::make_uniform_vector<float>(static_cast<size_t>(s.M) * s.K, -1.0F, 1.0F, 42);
    p.w_nk = ttml::test_utils::make_uniform_vector<float>(static_cast<size_t>(s.N) * s.K, -1.0F, 1.0F, 43);
    // The device sees bf16 inputs, so the reference does too: the bound then measures the op, not input rounding.
    for (float& v : p.x) {
        v = static_cast<float>(bfloat16(v));
    }
    for (float& v : p.w_nk) {
        v = static_cast<float>(bfloat16(v));
    }
    const xt::xarray<float> x = xt::adapt(p.x, std::vector<size_t>{s.M, s.K});
    const xt::xarray<float> w = xt::adapt(p.w_nk, std::vector<size_t>{s.N, s.K});
    if (spot_ring_size == 0) {
        p.w_kn.resize(p.w_nk.size());
        for (uint32_t n = 0; n < s.N; ++n) {
            for (uint32_t k = 0; k < s.K; ++k) {
                p.w_kn[(static_cast<size_t>(k) * s.N) + n] = p.w_nk[(static_cast<size_t>(n) * s.K) + k];
            }
        }
        p.y = xt::linalg::dot(x, xt::transpose(w));
        return p;
    }
    p.tiles = spot_tiles(s, spot_ring_size);
    p.y = xt::zeros<float>({p.tiles.size(), size_t{32}, size_t{32}});
    for (size_t i = 0; i < p.tiles.size(); ++i) {
        const auto [m, n] = p.tiles[i];
        const xt::xarray<float> x_rows = xt::view(x, xt::range(m * 32, (m + 1) * 32), xt::all());
        const xt::xarray<float> w_rows = xt::view(w, xt::range(n * 32, (n + 1) * 32), xt::all());
        xt::view(p.y, i) = xt::linalg::dot(x_rows, xt::transpose(w_rows));
    }
    return p;
}

xt::xarray<float> checked_part(const Problem& p, const xt::xarray<float>& y) {
    if (p.tiles.empty()) {
        return y;
    }
    xt::xarray<float> part = xt::zeros<float>({p.tiles.size(), size_t{32}, size_t{32}});
    for (size_t i = 0; i < p.tiles.size(); ++i) {
        const auto [m, n] = p.tiles[i];
        xt::view(part, i) = xt::view(y, xt::range(m * 32, (m + 1) * 32), xt::range(n * 32, (n + 1) * 32));
    }
    return part;
}

// With bf16 inputs on both sides what's left is the HiFi2 matmul and the bf16 output, about 0.004 relative L2.
void expect_close(const xt::xarray<float>& ref, const xt::xarray<float>& out) {
    ASSERT_EQ(ref.shape(), out.shape());
    EXPECT_LT(ttml::test_utils::relative_l2(out, ref), 1e-2);
}

// Stripe s of a full weight: rows of an [N, K] weight (transpose_b), columns of a [K, N] one.
xt::xarray<float> stripe_of(const xt::xarray<float>& full, size_t s, size_t width, bool transpose_b) {
    if (transpose_b) {
        return xt::view(full, xt::range(s * width, (s + 1) * width), xt::all());
    }
    return xt::view(full, xt::all(), xt::range(s * width, (s + 1) * width));
}

std::vector<tt::tt_metal::GlobalSemaphore> make_semaphores(ttnn::distributed::MeshDevice* device) {
    const auto grid = device->compute_with_storage_grid_size();
    const tt::tt_metal::CoreRangeSet cores(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
    semaphores.reserve(device->shape()[kClusterAxis]);
    for (uint32_t i = 0; i < device->shape()[kClusterAxis]; ++i) {
        semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(device, cores, 0));
    }
    return semaphores;
}

// Calls that reuse a buffer get a different weight: call i multiplies it by kCallScales[i % 4]. Powers of two and -1
// are exact in bf16, so the reference is the same factor times the CPU result, and a call that read a stripe an
// earlier call left in its buffer would not match it.
constexpr std::array<float, 4> kCallScales = {1.0F, -2.0F, 0.5F, -1.0F};

float call_scale(uint32_t call) {
    return kCallScales[call % kCallScales.size()];
}

ttnn::Tensor scaled_weight(const Problem& p, bool transpose_b, float scale) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto& s = p.shape;
    std::vector<float> w = transpose_b ? p.w_nk : p.w_kn;
    for (float& v : w) {
        v *= scale;
    }
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, transpose_b ? 0 : 1, kClusterAxis);
    const ttnn::Shape shape = transpose_b ? ttnn::Shape({s.N, s.K}) : ttnn::Shape({s.K, s.N});
    return ttml::core::from_vector(w, shape, device, ttnn::Layout::TILE, shard.get());
}

void run_fabric_gather(
    const Problem& p,
    const ttnn::Tensor& x,
    const ttnn::Tensor& w,
    uint32_t num_links,
    bool transpose_b,
    ttnn::ccl::Topology topology,
    const std::optional<Block>& block,
    uint32_t num_calls = kCalls) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto grid = device->compute_with_storage_grid_size();
    std::optional<MinimalMatmulConfig> config;
    if (block.has_value()) {
        config = MinimalMatmulConfig{
            .M_block_size = block->M_block,
            .K_block_size = block->K_block,
            .N_block_size = block->N_block,
            .subblock_h = block->subblock_h,
            .subblock_w = block->subblock_w,
            .compute_with_storage_grid_size = grid};
    }

    const auto& s = p.shape;
    const ttnn::Shape w_shape = transpose_b ? ttnn::Shape({s.N, s.K}) : ttnn::Shape({s.K, s.N});
    const std::array<ttnn::Tensor, 2> buffers = {
        ttml::core::zeros(w_shape, device), ttml::core::zeros(w_shape, device)};
    const auto semaphores = make_semaphores(device);
    std::vector<ttnn::Tensor> weights;
    weights.reserve(num_calls);
    for (uint32_t i = 0; i < num_calls; ++i) {
        weights.push_back(call_scale(i) == 1.0F ? w : scaled_weight(p, transpose_b, call_scale(i)));
    }

    std::vector<ttnn::Tensor> outputs;
    outputs.reserve(num_calls);
    for (uint32_t i = 0; i < num_calls; ++i) {
        outputs.push_back(ttml::metal::all_gather_weight_minimal_matmul_async(
            x,
            weights[i],
            buffers[i % 2],
            semaphores,
            kClusterAxis,
            topology,
            config,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            num_links,
            transpose_b));
    }
    for (uint32_t i = 0; i < num_calls; ++i) {
        SCOPED_TRACE(testing::Message() << "call " << i);
        // Every device holds the whole output, so its label must not claim a split.
        for (const auto& placement : outputs[i].tensor_topology().placements()) {
            EXPECT_TRUE(std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Replicate>(placement));
        }
        for (const auto& device_y : ttnn::distributed::get_device_tensors(outputs[i])) {
            xt::xarray<float> y = ttml::core::to_xtensor(device_y);
            y.reshape({p.shape.M, p.shape.N});
            const xt::xarray<float> ref = p.y * call_scale(i);
            expect_close(ref, checked_part(p, y));
        }
    }

    // The gather itself is pure data movement, so each buffer must hold the remote stripes of the last call that used
    // it bit for bit, in every column. The local stripe is read from the shard and never lands in the buffer.
    for (uint32_t b = 0; b < 2 && b < num_calls; ++b) {
        const uint32_t last = (((num_calls - 1 - b) / 2) * 2) + b;
        SCOPED_TRACE(testing::Message() << "buffer " << b << " after call " << last);
        std::vector<xt::xarray<float>> shards;
        for (const auto& shard : ttnn::distributed::get_device_tensors(weights[last])) {
            shards.push_back(ttml::core::to_xtensor(shard));
        }
        const auto gathered = ttnn::distributed::get_device_tensors(buffers[b]);
        for (size_t d = 0; d < gathered.size(); ++d) {
            const xt::xarray<float> buffer = ttml::core::to_xtensor(gathered[d]);
            for (size_t s = 0; s < shards.size(); ++s) {
                if (s == d) {
                    continue;
                }
                const size_t width = shards[s].shape()[transpose_b ? 0 : 1];
                EXPECT_TRUE(xt::all(xt::equal(stripe_of(buffer, s, width, transpose_b), shards[s])))
                    << "device " << d << " stripe " << s;
            }
        }
    }
}

void sweep(const Shape& s) {
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
        for (const uint32_t num_links : {1U, 2U}) {
            for (const auto topology : {ttnn::ccl::Topology::Ring, ttnn::ccl::Topology::Linear}) {
                for (const auto& block : blocks) {
                    SCOPED_TRACE(
                        testing::Message() << "links=" << num_links << " transpose_b=" << transpose_b
                                           << " ring=" << (topology == ttnn::ccl::Topology::Ring) << " block="
                                           << (block.has_value() ? std::to_string(block->N_block) + "/" +
                                                                       std::to_string(block->subblock_h) + "x" +
                                                                       std::to_string(block->subblock_w)
                                                                 : std::string("default")));
                    run_fabric_gather(p, x, w, num_links, transpose_b, topology, block);
                }
            }
        }
    }
}
}  // namespace

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, SharedSemaphoresAcrossShapes) {
    auto* device = &ttml::autograd::ctx().get_device();
    const std::array<Shape, 3> shapes = {{{256, 512, 1024}, {256, 1024, 512}, {256, 512, 2048}}};
    constexpr uint32_t kRounds = 3;
    struct Layer {
        Problem p;
        ttnn::Tensor x;
        std::array<ttnn::Tensor, kRounds> w;
        std::array<ttnn::Tensor, 2> buffers;
    };
    std::vector<Layer> layers;
    layers.reserve(shapes.size());
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    for (const auto& s : shapes) {
        Problem p = make_problem(s);
        auto x = ttml::core::from_vector(p.x, ttnn::Shape({s.M, s.K}), device, ttnn::Layout::TILE, replicate.get());
        std::array<ttnn::Tensor, kRounds> w = {
            scaled_weight(p, true, call_scale(0)),
            scaled_weight(p, true, call_scale(1)),
            scaled_weight(p, true, call_scale(2))};
        const ttnn::Shape full({s.N, s.K});
        layers.push_back(
            {std::move(p),
             std::move(x),
             std::move(w),
             {ttml::core::zeros(full, device), ttml::core::zeros(full, device)}});
    }
    const auto semaphores = make_semaphores(device);

    for (const uint32_t num_links : {2U, 1U}) {
        SCOPED_TRACE(testing::Message() << "links=" << num_links);
        std::vector<std::pair<uint32_t, ttnn::Tensor>> outputs;
        outputs.reserve(kRounds * layers.size());
        for (uint32_t round = 0; round < kRounds; ++round) {
            for (uint32_t l = 0; l < layers.size(); ++l) {
                outputs.emplace_back(
                    l,
                    ttml::metal::all_gather_weight_minimal_matmul_async(
                        layers[l].x,
                        layers[l].w[round],
                        layers[l].buffers[round % 2],
                        semaphores,
                        kClusterAxis,
                        ttnn::ccl::Topology::Ring,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        num_links));
            }
        }
        for (uint32_t i = 0; i < outputs.size(); ++i) {
            const auto& [l, y] = outputs[i];
            const xt::xarray<float> ref = layers[l].p.y * call_scale(static_cast<uint32_t>(i / layers.size()));
            for (const auto& device_y : ttnn::distributed::get_device_tensors(y)) {
                SCOPED_TRACE(testing::Message() << "call " << i << " layer " << l);
                expect_close(ref, ttml::core::to_xtensor(device_y));
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
    // Same shape, different addresses: the hit has to patch the input address too.
    const std::array<ttnn::Tensor, 2> xs = {
        ttml::core::from_vector(p.x, ttnn::Shape({s.M, s.K}), device, ttnn::Layout::TILE, replicate.get()),
        ttml::core::from_vector(p.x, ttnn::Shape({s.M, s.K}), device, ttnn::Layout::TILE, replicate.get())};
    const auto w = ttml::core::from_vector(p.w_nk, ttnn::Shape({s.N, s.K}), device, ttnn::Layout::TILE, shard.get());
    const std::array<ttnn::Tensor, 2> buffers = {
        ttml::core::zeros(ttnn::Shape({s.N, s.K}), device), ttml::core::zeros(ttnn::Shape({s.N, s.K}), device)};
    const std::array<std::vector<tt::tt_metal::GlobalSemaphore>, 2> semaphores = {
        make_semaphores(device), make_semaphores(device)};
    size_t entries = 0;
    for (uint32_t i = 0; i < 3; ++i) {
        // Calls 0 and 1 share a program; call 2 uses one link instead of two, which is a new one.
        const uint32_t num_links = i < 2 ? 2 : 1;
        SCOPED_TRACE(testing::Message() << "call " << i);
        const auto y = ttml::metal::all_gather_weight_minimal_matmul_async(
            xs[i % 2],
            w,
            buffers[i % 2],
            semaphores[i == 0 ? 0 : 1],
            kClusterAxis,
            ttnn::ccl::Topology::Ring,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            num_links);
        if (i == 0) {
            entries = device->num_program_cache_entries();
        } else {
            EXPECT_EQ(device->num_program_cache_entries(), entries + (i - 1));
        }
        for (const auto& device_y : ttnn::distributed::get_device_tensors(y)) {
            expect_close(p.y, ttml::core::to_xtensor(device_y));
        }
        // Poisoning the first set proves later calls run on the second.
        for (const auto& semaphore : semaphores[0]) {
            semaphore.reset_semaphore_value(1000);
        }
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
        for (const uint32_t num_links : {2U, 1U}) {
            SCOPED_TRACE(testing::Message() << "K=" << s.K << " N=" << s.N << " links=" << num_links);
            run_fabric_gather(p, x, w, num_links, true, ttnn::ccl::Topology::Ring, std::nullopt);
        }
    }
}

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, TinyShapes) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0, kClusterAxis);
    // 320 = 8 + 2 tiles: a short last K chunk after a full one.
    for (const uint32_t K : {32U, 96U, 256U, 320U}) {
        for (const uint32_t N : {128U, 256U, 384U}) {
            const Shape s{128, K, N};
            const Problem p = make_problem(s);
            const auto x =
                ttml::core::from_vector(p.x, ttnn::Shape({s.M, s.K}), device, ttnn::Layout::TILE, replicate.get());
            const auto w =
                ttml::core::from_vector(p.w_nk, ttnn::Shape({s.N, s.K}), device, ttnn::Layout::TILE, shard.get());
            for (const uint32_t num_links : {2U, 1U}) {
                for (const auto topology : {ttnn::ccl::Topology::Ring, ttnn::ccl::Topology::Linear}) {
                    const bool ring = topology == ttnn::ccl::Topology::Ring;
                    SCOPED_TRACE(
                        testing::Message() << "K=" << K << " N=" << N << " links=" << num_links << " ring=" << ring);
                    run_fabric_gather(p, x, w, num_links, true, topology, std::nullopt);
                }
            }
        }
    }
}

TEST_F(AllGatherWeightMinimalMatmulAsyncTest, Small) {
    sweep({256, 512, 1024});
}

namespace {
struct LargeCase {
    const char* name;
    uint32_t batch;
    Shape shape;
};

const LargeCase kLargeCases[] = {
    {"W1_2048x4096x11008", 1, {2048, 4096, 11008}},
    {"W2_2048x11008x4096", 1, {2048, 11008, 4096}},
    {"TransposedGrid_8192x1024x2048", 1, {8192, 1024, 2048}},
    {"Batch4_4x2048x4096x11008", 4, {8192, 4096, 11008}},
};

class AllGatherWeightMinimalMatmulAsyncLargeTest : public AllGatherWeightMinimalMatmulAsyncTest,
                                                   public ::testing::WithParamInterface<LargeCase> {};

std::string CaseName(const ::testing::TestParamInfo<LargeCase>& info) {
    return info.param.name;
}
}  // namespace

TEST_P(AllGatherWeightMinimalMatmulAsyncLargeTest, MatchesReference) {
    const LargeCase& c = GetParam();
    const Shape& s = c.shape;
    auto* device = &ttml::autograd::ctx().get_device();
    const Problem p = make_problem(s, device->shape()[kClusterAxis]);
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    const ttnn::Shape x_shape = c.batch == 1 ? ttnn::Shape({s.M, s.K}) : ttnn::Shape({c.batch, 1, s.M / c.batch, s.K});
    const auto x = ttml::core::from_vector(p.x, x_shape, device, ttnn::Layout::TILE, replicate.get());
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0, kClusterAxis);
    const auto w = ttml::core::from_vector(p.w_nk, ttnn::Shape({s.N, s.K}), device, ttnn::Layout::TILE, shard.get());
    const auto grid = device->compute_with_storage_grid_size();
    const size_t available_links = ttnn::operations::ccl::common::get_num_links(*device, kClusterAxis);
    bool ran = false;
    for (const uint32_t num_links : {2U, 4U}) {
        if (num_links > available_links || 2 * num_links > grid.x - 1) {
            continue;
        }
        for (const auto topology : {ttnn::ccl::Topology::Ring, ttnn::ccl::Topology::Linear}) {
            SCOPED_TRACE(
                testing::Message() << "ring=" << (topology == ttnn::ccl::Topology::Ring) << " links=" << num_links);
            run_fabric_gather(p, x, w, num_links, true, topology, std::nullopt, 2);
            ran = true;
        }
    }
    if (!ran) {
        GTEST_SKIP() << "needs at least 2 fabric links on the ring's axis, found " << available_links;
    }
}

INSTANTIATE_TEST_SUITE_P(
    LlamaShapes, AllGatherWeightMinimalMatmulAsyncLargeTest, ::testing::ValuesIn(kLargeCases), CaseName);
