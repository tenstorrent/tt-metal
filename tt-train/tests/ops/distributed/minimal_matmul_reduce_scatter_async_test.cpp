// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <tt-metalium/constants.hpp>
#include <tt-umd-workload/cluster.hpp>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/misc/xmanipulation.hpp>
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

constexpr uint32_t kClusterAxis = 1;
constexpr uint32_t kCalls = 4;
constexpr uint32_t kTile = tt::constants::TILE_HEIGHT;

class MinimalMatmulReduceScatterAsyncTest : public ttml::test_utils::MeshFixture {
protected:
    MinimalMatmulReduceScatterAsyncTest() : MeshFixture(tt::tt_metal::distributed::MeshShape(1, 4)) {
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

struct RunOptions {
    bool transpose_b = false;
    uint32_t num_links = 1;
    bool bidirectional = false;
    std::optional<Block> block;
    uint32_t num_calls = kCalls;
};

float round_to_bf16(float v) {
    const auto bits = std::bit_cast<uint32_t>(v);
    const uint32_t rounded = (bits + 0x7FFFU + ((bits >> 16) & 1U)) & 0xFFFF0000U;
    return std::bit_cast<float>(rounded);
}

std::vector<float> bf16_uniform(size_t n, uint32_t seed) {
    auto v = ttml::test_utils::make_uniform_vector<float>(n, -1.0F, 1.0F, seed);
    std::transform(v.begin(), v.end(), v.begin(), round_to_bf16);
    return v;
}

struct Problem {
    Shape shape;
    uint32_t ring_size = 0;
    bool transpose_a = false;
    std::vector<float> x;
    std::vector<float> w_kn;
    std::vector<float> w_nk;
    std::vector<std::array<uint32_t, 2>> tiles;
    xt::xarray<float> y;
};

std::vector<std::array<uint32_t, 2>> spot_tiles(const Shape& s, uint32_t ring_size) {
    const uint32_t slice_tiles = s.M / kTile / ring_size;
    const uint32_t n_tiles = s.N / kTile;
    const uint32_t columns = std::min<uint32_t>(n_tiles, 8);
    std::vector<std::array<uint32_t, 2>> tiles;
    for (uint32_t slice = 0; slice < ring_size; ++slice) {
        for (const uint32_t r : {0U, slice_tiles / 2, slice_tiles - 1}) {
            for (uint32_t j = 0; j < columns; ++j) {
                tiles.push_back({(slice * slice_tiles) + r, ((2 * j) + 1) * n_tiles / (2 * columns)});
            }
        }
    }
    return tiles;
}

xt::xarray<float> input_rows(const Problem& p, const xt::xarray<float>& x, uint32_t d, uint32_t m0, uint32_t m1) {
    if (p.transpose_a) {
        return xt::transpose(xt::xarray<float>(xt::view(x, d, xt::all(), xt::range(m0, m1))));
    }
    return xt::view(x, d, xt::range(m0, m1), xt::all());
}

Problem make_problem(const Shape& s, uint32_t ring_size, bool spot_check = false, bool transpose_a = false) {
    Problem p{.shape = s, .ring_size = ring_size, .transpose_a = transpose_a};
    p.x = bf16_uniform(static_cast<size_t>(ring_size) * s.M * s.K, 42);
    p.w_kn = bf16_uniform(static_cast<size_t>(ring_size) * s.K * s.N, 43);
    p.w_nk.resize(p.w_kn.size());
    for (uint32_t d = 0; d < ring_size; ++d) {
        const size_t base = static_cast<size_t>(d) * s.K * s.N;
        for (uint32_t k = 0; k < s.K; ++k) {
            for (uint32_t n = 0; n < s.N; ++n) {
                p.w_nk[base + (static_cast<size_t>(n) * s.K) + k] = p.w_kn[base + (static_cast<size_t>(k) * s.N) + n];
            }
        }
    }
    const xt::xarray<float> x = xt::adapt(
        p.x, transpose_a ? std::vector<size_t>{ring_size, s.K, s.M} : std::vector<size_t>{ring_size, s.M, s.K});
    const xt::xarray<float> w = xt::adapt(p.w_kn, std::vector<size_t>{ring_size, s.K, s.N});
    if (!spot_check) {
        p.y = xt::zeros<float>({s.M, s.N});
        for (uint32_t d = 0; d < ring_size; ++d) {
            const xt::xarray<float> xd = input_rows(p, x, d, 0, s.M);
            const xt::xarray<float> wd = xt::view(w, d);
            p.y += xt::linalg::dot(xd, wd);
        }
        return p;
    }
    p.tiles = spot_tiles(s, ring_size);
    p.y = xt::zeros<float>({p.tiles.size(), size_t{kTile}, size_t{kTile}});
    for (size_t i = 0; i < p.tiles.size(); ++i) {
        const auto [m, n] = p.tiles[i];
        for (uint32_t d = 0; d < ring_size; ++d) {
            const xt::xarray<float> x_rows = input_rows(p, x, d, m * kTile, (m + 1) * kTile);
            const xt::xarray<float> w_cols = xt::view(w, d, xt::all(), xt::range(n * kTile, (n + 1) * kTile));
            xt::view(p.y, i) += xt::linalg::dot(x_rows, w_cols);
        }
    }
    return p;
}

xt::xarray<float> checked_part(const Problem& p, const xt::xarray<float>& y) {
    if (p.tiles.empty()) {
        return y;
    }
    xt::xarray<float> part = xt::zeros<float>({p.tiles.size(), size_t{kTile}, size_t{kTile}});
    for (size_t i = 0; i < p.tiles.size(); ++i) {
        const auto [m, n] = p.tiles[i];
        xt::view(part, i) = xt::view(y, xt::range(m * kTile, (m + 1) * kTile), xt::range(n * kTile, (n + 1) * kTile));
    }
    return part;
}

double pcc(const xt::xarray<float>& a, const xt::xarray<float>& b) {
    const xt::xarray<double> da = xt::cast<double>(a) - xt::mean(xt::cast<double>(a))();
    const xt::xarray<double> db = xt::cast<double>(b) - xt::mean(xt::cast<double>(b))();
    return xt::sum(da * db)() / std::sqrt(xt::sum(da * da)() * xt::sum(db * db)());
}

xt::xarray<float> gather_slices(const Problem& p, const ttnn::Tensor& out) {
    const auto& s = p.shape;
    const uint32_t rows = s.M / p.ring_size;
    xt::xarray<float> y = xt::zeros<float>({s.M, s.N});
    const auto device_tensors = ttnn::distributed::get_device_tensors(out);
    EXPECT_EQ(device_tensors.size(), p.ring_size);
    for (uint32_t d = 0; d < device_tensors.size(); ++d) {
        xt::xarray<float> slice = ttml::core::to_xtensor(device_tensors[d]);
        EXPECT_EQ(slice.size(), static_cast<size_t>(rows) * s.N);
        slice.reshape({rows, s.N});
        xt::view(y, xt::range(d * rows, (d + 1) * rows), xt::all()) = slice;
    }
    return y;
}

void expect_close(const Problem& p, const ttnn::Tensor& out) {
    const xt::xarray<float> got = checked_part(p, gather_slices(p, out));
    ASSERT_EQ(p.y.shape(), got.shape());
    const double corr = pcc(p.y, got);
    const double rel = ttml::test_utils::relative_l2(got, p.y);
    std::cout << "[mmrs] M=" << p.shape.M << " K=" << p.shape.K << " N=" << p.shape.N
              << " transpose_a=" << p.transpose_a << " pcc=" << corr << " rel_l2=" << rel << '\n';
    EXPECT_GT(corr, 0.9999);
    EXPECT_LT(rel, 0.01);
}

std::vector<tt::tt_metal::GlobalSemaphore> make_semaphores(ttnn::distributed::MeshDevice* device) {
    const auto grid = device->compute_with_storage_grid_size();
    const tt::tt_metal::CoreRangeSet cores(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    return {
        ttnn::global_semaphore::create_global_semaphore(device, cores, 0),
        ttnn::global_semaphore::create_global_semaphore(device, cores, 0)};
}

std::optional<MinimalMatmulConfig> make_config(const std::optional<Block>& block) {
    if (!block.has_value()) {
        return std::nullopt;
    }
    return MinimalMatmulConfig{
        .M_block_size = block->M_block,
        .K_block_size = block->K_block,
        .N_block_size = block->N_block,
        .subblock_h = block->subblock_h,
        .subblock_w = block->subblock_w,
        .compute_with_storage_grid_size = ttml::autograd::ctx().get_device().compute_with_storage_grid_size()};
}

struct DeviceInputs {
    ttnn::Tensor x;
    ttnn::Tensor w;
    bool transpose_a = false;
};

DeviceInputs upload(const Problem& p, bool transpose_b) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto& s = p.shape;
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0, kClusterAxis);
    const uint32_t r = p.ring_size;
    const ttnn::Shape x_shape = p.transpose_a ? ttnn::Shape({r * s.K, s.M}) : ttnn::Shape({r * s.M, s.K});
    auto x = ttml::core::from_vector(p.x, x_shape, device, ttnn::Layout::TILE, shard.get());
    const ttnn::Shape w_shape = transpose_b ? ttnn::Shape({r * s.N, s.K}) : ttnn::Shape({r * s.K, s.N});
    auto w = ttml::core::from_vector(transpose_b ? p.w_nk : p.w_kn, w_shape, device, ttnn::Layout::TILE, shard.get());
    return {std::move(x), std::move(w), p.transpose_a};
}

ttnn::Tensor make_buffer(const Shape& s) {
    return ttml::core::zeros(ttnn::Shape({s.M, s.N}), &ttml::autograd::ctx().get_device());
}

ttnn::Tensor call_op(
    const DeviceInputs& in,
    const ttnn::Tensor& buffer,
    const std::vector<tt::tt_metal::GlobalSemaphore>& semaphores,
    const RunOptions& opt) {
    return ttml::metal::minimal_matmul_reduce_scatter_async(
        in.x,
        in.w,
        buffer,
        semaphores,
        kClusterAxis,
        opt.num_links,
        make_config(opt.block),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        in.transpose_a,
        opt.transpose_b,
        opt.bidirectional);
}

void run(const Problem& p, const DeviceInputs& in, const RunOptions& opt) {
    auto* device = &ttml::autograd::ctx().get_device();
    const std::array<ttnn::Tensor, 2> buffers = {make_buffer(p.shape), make_buffer(p.shape)};
    const auto semaphores = make_semaphores(device);
    std::vector<ttnn::Tensor> outputs;
    for (uint32_t i = 0; i < opt.num_calls; ++i) {
        outputs.push_back(call_op(in, buffers[i % 2], semaphores, opt));
    }
    for (uint32_t i = 0; i < opt.num_calls; ++i) {
        SCOPED_TRACE(testing::Message() << "call " << i);
        expect_close(p, outputs[i]);
    }
}

std::string describe(const RunOptions& opt) {
    std::string block = "default";
    if (opt.block.has_value()) {
        block = std::to_string(opt.block->M_block) + "x" + std::to_string(opt.block->K_block) + "x" +
                std::to_string(opt.block->N_block) + "/" + std::to_string(opt.block->subblock_h) + "x" +
                std::to_string(opt.block->subblock_w);
    }
    return "links=" + std::to_string(opt.num_links) + " transpose_b=" + std::to_string(opt.transpose_b) +
           " bidirectional=" + std::to_string(opt.bidirectional) + " block=" + block;
}

uint32_t ring_size() {
    return ttml::autograd::ctx().get_device().shape()[kClusterAxis];
}
}  // namespace

TEST_F(MinimalMatmulReduceScatterAsyncTest, Smoke) {
    const Problem p = make_problem({512, 64, 128}, ring_size());
    const DeviceInputs in = upload(p, false);
    run(p, in, RunOptions{.num_links = 1, .num_calls = 1});
}

TEST_F(MinimalMatmulReduceScatterAsyncTest, TinyShapes) {
    for (const uint32_t M : {128U, 512U}) {
        for (const uint32_t K : {32U, 288U}) {
            for (const uint32_t N : {32U, 320U}) {
                const Problem p = make_problem({M, K, N}, ring_size());
                const DeviceInputs in = upload(p, false);
                for (const uint32_t num_links : {1U, 2U}) {
                    for (const bool bidirectional : {false, true}) {
                        const RunOptions opt{.num_links = num_links, .bidirectional = bidirectional};
                        SCOPED_TRACE(
                            testing::Message() << "M=" << M << " K=" << K << " N=" << N << " " << describe(opt));
                        run(p, in, opt);
                    }
                }
            }
        }
    }
}

TEST_F(MinimalMatmulReduceScatterAsyncTest, BothRings) {
    const std::array<std::pair<Shape, Block>, 4> cases = {{
        {{512, 64, 384}, {2, 2, 1, 1, 1}},
        {{512, 64, 704}, {2, 2, 1, 1, 1}},
        {{1024, 256, 2048}, {4, 4, 2, 2, 1}},
        {{2048, 512, 2560}, {4, 4, 1, 4, 1}},
    }};
    for (const auto& [s, block] : cases) {
        const Problem p = make_problem(s, ring_size());
        for (const bool transpose_b : {false, true}) {
            const DeviceInputs in = upload(p, transpose_b);
            for (const uint32_t num_links : {1U, 4U}) {
                const RunOptions opt{
                    .transpose_b = transpose_b, .num_links = num_links, .bidirectional = true, .block = block};
                SCOPED_TRACE(testing::Message() << "M=" << s.M << " K=" << s.K << " N=" << s.N << " " << describe(opt));
                run(p, in, opt);
            }
        }
    }
}

TEST_F(MinimalMatmulReduceScatterAsyncTest, Small) {
    const Problem p = make_problem({1024, 512, 1024}, ring_size());
    const std::array<std::optional<Block>, 4> blocks = {
        std::nullopt, Block{4, 4, 4, 2, 2}, Block{2, 8, 6, 1, 3}, Block{4, 4, 2, 2, 1}};
    for (const bool transpose_b : {false, true}) {
        const DeviceInputs in = upload(p, transpose_b);
        for (const uint32_t num_links : {1U, 4U}) {
            for (const bool bidirectional : {false, true}) {
                for (const auto& block : blocks) {
                    const RunOptions opt{
                        .transpose_b = transpose_b,
                        .num_links = num_links,
                        .bidirectional = bidirectional,
                        .block = block};
                    SCOPED_TRACE(describe(opt));
                    run(p, in, opt);
                }
            }
        }
    }
}

TEST_F(MinimalMatmulReduceScatterAsyncTest, RowsNotMultipleOfBlocks) {
    for (const Shape& s : {Shape{8320, 256, 640}, Shape{1664, 160, 352}}) {
        const Problem p = make_problem(s, ring_size());
        for (const bool transpose_b : {false, true}) {
            const DeviceInputs in = upload(p, transpose_b);
            for (const bool bidirectional : {false, true}) {
                const RunOptions opt{
                    .transpose_b = transpose_b,
                    .num_links = 2,
                    .bidirectional = bidirectional,
                    .block = Block{4, 4, 4, 2, 2},
                    .num_calls = 2};
                SCOPED_TRACE(testing::Message() << "M=" << s.M << " " << describe(opt));
                run(p, in, opt);
            }
        }
    }
}

TEST_F(MinimalMatmulReduceScatterAsyncTest, SharedSemaphoresAcrossShapes) {
    auto* device = &ttml::autograd::ctx().get_device();
    const std::array<Shape, 3> shapes = {{{512, 512, 1024}, {1024, 256, 512}, {512, 1024, 2048}}};
    struct Layer {
        Problem p;
        DeviceInputs in;
        std::array<ttnn::Tensor, 2> buffers;
    };
    std::vector<Layer> layers;
    for (const auto& s : shapes) {
        Problem p = make_problem(s, ring_size());
        DeviceInputs in = upload(p, true);
        layers.push_back({std::move(p), std::move(in), {make_buffer(s), make_buffer(s)}});
    }
    const auto semaphores = make_semaphores(device);
    const RunOptions opt{.transpose_b = true, .num_links = 2, .bidirectional = true};
    constexpr uint32_t kRounds = 3;
    std::vector<std::pair<uint32_t, ttnn::Tensor>> outputs;
    for (uint32_t round = 0; round < kRounds; ++round) {
        for (uint32_t l = 0; l < layers.size(); ++l) {
            outputs.emplace_back(l, call_op(layers[l].in, layers[l].buffers[round % 2], semaphores, opt));
        }
    }
    for (uint32_t i = 0; i < outputs.size(); ++i) {
        const auto& [l, y] = outputs[i];
        SCOPED_TRACE(testing::Message() << "call " << i << " layer " << l);
        expect_close(layers[l].p, y);
    }
}

TEST_F(MinimalMatmulReduceScatterAsyncTest, ProgramCacheHitWithOtherBuffersAndSemaphores) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();
    const Shape s{1024, 512, 1024};
    const Problem p = make_problem(s, ring_size());
    const DeviceInputs in = upload(p, true);
    const std::array<std::vector<tt::tt_metal::GlobalSemaphore>, 2> semaphores = {
        make_semaphores(device), make_semaphores(device)};
    const RunOptions opt{.transpose_b = true, .num_links = 2, .bidirectional = true};
    size_t entries_after_first = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        SCOPED_TRACE(testing::Message() << "call " << i);
        const auto y = call_op(in, make_buffer(s), semaphores[i], opt);
        expect_close(p, y);
        if (i == 0) {
            entries_after_first = device->num_program_cache_entries();
            for (const auto& semaphore : semaphores[0]) {
                semaphore.reset_semaphore_value(1000);
            }
        } else {
            EXPECT_EQ(device->num_program_cache_entries(), entries_after_first);
        }
    }
}

TEST_F(MinimalMatmulReduceScatterAsyncTest, TransposeATinyShapes) {
    for (const uint32_t M : {128U, 512U}) {
        for (const uint32_t K : {32U, 288U}) {
            for (const uint32_t N : {32U, 320U}) {
                const Problem p = make_problem({M, K, N}, ring_size(), false, true);
                for (const bool transpose_b : {false, true}) {
                    const DeviceInputs in = upload(p, transpose_b);
                    for (const bool bidirectional : {false, true}) {
                        const RunOptions opt{
                            .transpose_b = transpose_b, .num_links = 2, .bidirectional = bidirectional, .num_calls = 2};
                        SCOPED_TRACE(
                            testing::Message() << "M=" << M << " K=" << K << " N=" << N << " " << describe(opt));
                        run(p, in, opt);
                    }
                }
            }
        }
    }
}

TEST_F(MinimalMatmulReduceScatterAsyncTest, TransposeASmall) {
    const Problem p = make_problem({1024, 512, 1024}, ring_size(), false, true);
    const std::array<std::optional<Block>, 4> blocks = {
        std::nullopt, Block{4, 4, 4, 2, 2}, Block{2, 8, 6, 1, 3}, Block{4, 4, 2, 2, 1}};
    for (const bool transpose_b : {false, true}) {
        const DeviceInputs in = upload(p, transpose_b);
        for (const uint32_t num_links : {1U, 4U}) {
            for (const bool bidirectional : {false, true}) {
                for (const auto& block : blocks) {
                    const RunOptions opt{
                        .transpose_b = transpose_b,
                        .num_links = num_links,
                        .bidirectional = bidirectional,
                        .block = block};
                    SCOPED_TRACE(describe(opt));
                    run(p, in, opt);
                }
            }
        }
    }
}

TEST_F(MinimalMatmulReduceScatterAsyncTest, TransposeARowsNotMultipleOfBlocks) {
    for (const Shape& s : {Shape{8320, 256, 640}, Shape{1664, 160, 352}}) {
        const Problem p = make_problem(s, ring_size(), false, true);
        const DeviceInputs in = upload(p, false);
        for (const bool bidirectional : {false, true}) {
            const RunOptions opt{
                .num_links = 2, .bidirectional = bidirectional, .block = Block{4, 4, 4, 2, 2}, .num_calls = 2};
            SCOPED_TRACE(testing::Message() << "M=" << s.M << " " << describe(opt));
            run(p, in, opt);
        }
    }
}

TEST_F(MinimalMatmulReduceScatterAsyncTest, TransposeABothRings) {
    const std::array<std::pair<Shape, Block>, 3> cases = {{
        {{512, 64, 704}, {2, 2, 1, 1, 1}},
        {{1024, 256, 2048}, {4, 4, 2, 2, 1}},
        {{2048, 512, 2560}, {4, 4, 1, 4, 1}},
    }};
    for (const auto& [s, block] : cases) {
        const Problem p = make_problem(s, ring_size(), false, true);
        const DeviceInputs in = upload(p, false);
        for (const uint32_t num_links : {1U, 4U}) {
            const RunOptions opt{.num_links = num_links, .bidirectional = true, .block = block};
            SCOPED_TRACE(testing::Message() << "M=" << s.M << " K=" << s.K << " N=" << s.N << " " << describe(opt));
            run(p, in, opt);
        }
    }
}

TEST_F(MinimalMatmulReduceScatterAsyncTest, TransposeATakesRowsFromLastDim) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto x = ttml::core::zeros(ttnn::Shape({512, 96}), device);
    const auto w = ttml::core::zeros(ttnn::Shape({512, 64}), device);
    const auto semaphores = make_semaphores(device);
    const auto buffer = ttml::core::zeros(ttnn::Shape({96, 64}), device);
    try {
        (void)ttml::metal::minimal_matmul_reduce_scatter_async(
            x, w, buffer, semaphores, kClusterAxis, 1U, std::nullopt, std::nullopt, std::nullopt, std::nullopt, true);
        ADD_FAILURE() << "expected a validation error";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("M (96) must be a multiple of ring_size"), std::string::npos) << e.what();
    }
}

TEST_F(MinimalMatmulReduceScatterAsyncTest, TransposeAProgramCache) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->clear_program_cache();
    const Shape s{512, 512, 1024};
    const Problem plain = make_problem(s, ring_size());
    const Problem transposed = make_problem(s, ring_size(), false, true);
    const DeviceInputs plain_in = upload(plain, false);
    const DeviceInputs transposed_in = upload(transposed, false);
    const std::array<std::vector<tt::tt_metal::GlobalSemaphore>, 2> semaphores = {
        make_semaphores(device), make_semaphores(device)};
    const RunOptions opt{.num_links = 2, .bidirectional = true};
    expect_close(plain, call_op(plain_in, make_buffer(s), semaphores[0], opt));
    const size_t entries_plain = device->num_program_cache_entries();
    expect_close(transposed, call_op(transposed_in, make_buffer(s), semaphores[0], opt));
    const size_t entries_both = device->num_program_cache_entries();
    EXPECT_GT(entries_both, entries_plain);
    for (const auto& semaphore : semaphores[0]) {
        semaphore.reset_semaphore_value(1000);
    }
    expect_close(transposed, call_op(transposed_in, make_buffer(s), semaphores[1], opt));
    EXPECT_EQ(device->num_program_cache_entries(), entries_both);
}

namespace {
struct LargeCase {
    const char* name;
    Shape shape;
    bool transpose_b;
    bool transpose_a = false;
};

const LargeCase kLargeCases[] = {
    {"SpRowParallel_8192x2752x4096", {8192, 2752, 4096}, true},
    {"SpColumnParallelDgrad_8192x1024x4096", {8192, 1024, 4096}, false},
    {"FsdpWqDw_4096x8192x4096", {4096, 8192, 4096}, false, true},
    {"FsdpW1Dw_11008x8192x4096", {11008, 8192, 4096}, false, true},
    {"FsdpW2Dw_4096x8192x11008", {4096, 8192, 11008}, false, true},
};

class MinimalMatmulReduceScatterAsyncLargeTest : public MinimalMatmulReduceScatterAsyncTest,
                                                 public ::testing::WithParamInterface<LargeCase> {};

std::string CaseName(const ::testing::TestParamInfo<LargeCase>& info) {
    return info.param.name;
}
}  // namespace

TEST_P(MinimalMatmulReduceScatterAsyncLargeTest, MatchesReference) {
    const LargeCase& c = GetParam();
    auto* device = &ttml::autograd::ctx().get_device();
    const Problem p = make_problem(c.shape, ring_size(), true, c.transpose_a);
    const DeviceInputs in = upload(p, c.transpose_b);
    const auto available_links =
        static_cast<uint32_t>(ttnn::operations::ccl::common::get_num_links(*device, kClusterAxis));
    for (const bool bidirectional : {false, true}) {
        const RunOptions opt{
            .transpose_b = c.transpose_b,
            .num_links = std::min(4U, available_links),
            .bidirectional = bidirectional,
            .num_calls = 2};
        SCOPED_TRACE(describe(opt));
        run(p, in, opt);
    }
}

INSTANTIATE_TEST_SUITE_P(
    LongSequences, MinimalMatmulReduceScatterAsyncLargeTest, ::testing::ValuesIn(kLargeCases), CaseName);
