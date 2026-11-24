// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <tt-metalium/data_formats/bfloat16.hpp>
#include <tt-metalium/data_formats/tile_constants.hpp>
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
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;
using ttnn::experimental::prim::MinimalMatmulConfig;

constexpr uint32_t kClusterAxis = 0;
constexpr uint32_t kRingSize = 4;
constexpr uint32_t kCalls = 4;

class AllGatherMinimalMatmulOuterDimAsyncTest : public ttml::test_utils::MeshFixture {
protected:
    AllGatherMinimalMatmulOuterDimAsyncTest() : MeshFixture(tt::tt_metal::distributed::MeshShape(kRingSize, 1)) {
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
    int32_t dim = -1;  // -1: the second operand is the gathered one, sharded along N; -2: the first one, along M
    // The first operand, [M, K], and the second one stored [N, K], as transpose_b takes it, and stored [K, N].
    std::vector<float> in0;
    std::vector<float> in1_nk;
    std::vector<float> in1_kn;
    std::vector<std::array<uint32_t, 2>> tiles;
    xt::xarray<float> y;
};

// Output tiles {m, n} spread over every stripe of the gathered dimension: N for dim=-1, M for dim=-2.
std::vector<std::array<uint32_t, 2>> spot_tiles(const Shape& s, uint32_t ring_size, int32_t dim) {
    const uint32_t m_tiles = s.M / tt::constants::TILE_HEIGHT;
    const uint32_t n_tiles = s.N / tt::constants::TILE_WIDTH;
    const bool rows = dim == -2;
    const uint32_t stripe_tiles = (rows ? m_tiles : n_tiles) / ring_size;
    const uint32_t other_tiles = rows ? n_tiles : m_tiles;
    std::vector<std::array<uint32_t, 2>> tiles;
    const uint32_t spots = std::min<uint32_t>(stripe_tiles, 12);
    tiles.reserve(static_cast<size_t>(ring_size) * spots * 3);
    for (uint32_t stripe = 0; stripe < ring_size; ++stripe) {
        for (uint32_t j = 0; j < spots; ++j) {
            const uint32_t gathered = (stripe * stripe_tiles) + (((2 * j) + 1) * stripe_tiles / (2 * spots));
            for (const uint32_t other : {0U, other_tiles / 2, other_tiles - 1}) {
                tiles.push_back(
                    rows ? std::array<uint32_t, 2>{gathered, other} : std::array<uint32_t, 2>{other, gathered});
            }
        }
    }
    return tiles;
}

Problem make_problem(const Shape& s, int32_t dim = -1, uint32_t spot_ring_size = 0) {
    Problem p{.shape = s, .dim = dim};
    p.in0 = ttml::test_utils::make_uniform_vector<float>(static_cast<size_t>(s.M) * s.K, -1.0F, 1.0F, 42);
    p.in1_nk = ttml::test_utils::make_uniform_vector<float>(static_cast<size_t>(s.N) * s.K, -1.0F, 1.0F, 43);
    // The device sees bf16 inputs, so the reference does too: the bound then measures the op, not input rounding.
    for (float& v : p.in0) {
        v = static_cast<float>(bfloat16(v));
    }
    for (float& v : p.in1_nk) {
        v = static_cast<float>(bfloat16(v));
    }
    const xt::xarray<float> in0 = xt::adapt(p.in0, std::vector<size_t>{s.M, s.K});
    const xt::xarray<float> in1 = xt::adapt(p.in1_nk, std::vector<size_t>{s.N, s.K});
    if (spot_ring_size == 0) {
        p.in1_kn.resize(p.in1_nk.size());
        for (uint32_t n = 0; n < s.N; ++n) {
            for (uint32_t k = 0; k < s.K; ++k) {
                p.in1_kn[(static_cast<size_t>(k) * s.N) + n] = p.in1_nk[(static_cast<size_t>(n) * s.K) + k];
            }
        }
        p.y = xt::linalg::dot(in0, xt::transpose(in1));
        return p;
    }
    p.tiles = spot_tiles(s, spot_ring_size, dim);
    p.y = xt::zeros<float>({p.tiles.size(), size_t{32}, size_t{32}});
    for (size_t i = 0; i < p.tiles.size(); ++i) {
        const auto [m, n] = p.tiles[i];
        const xt::xarray<float> in0_rows = xt::view(in0, xt::range(m * 32, (m + 1) * 32), xt::all());
        const xt::xarray<float> in1_rows = xt::view(in1, xt::range(n * 32, (n + 1) * 32), xt::all());
        xt::view(p.y, i) = xt::linalg::dot(in0_rows, xt::transpose(in1_rows));
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

// The largest relative L2 error of any one 32x32 tile. `ref` and `out` are [rows, columns] in whole tiles, or a stack
// of tiles [tiles, 32, 32].
float worst_tile_relative_l2(const xt::xarray<float>& ref, const xt::xarray<float>& out) {
    xt::xarray<float> diff2 = xt::square(out - ref);
    xt::xarray<float> ref2 = xt::square(ref);
    std::vector<size_t> tile_axes = {1, 2};
    if (ref.dimension() == 2) {
        const std::array<size_t, 4> tiled = {ref.shape()[0] / 32, 32, ref.shape()[1] / 32, 32};
        diff2.reshape(tiled);
        ref2.reshape(tiled);
        tile_axes = {1, 3};
    }
    const xt::xarray<float> error = xt::sqrt(xt::sum(diff2, tile_axes) / (xt::sum(ref2, tile_axes) + 1e-12F));
    return xt::amax(error)();
}

// With bf16 inputs on both sides what's left is the HiFi2 matmul and the bf16 output: about 0.004 relative L2 for the
// whole output and at most 0.0044 for any one tile, over all the compares of this file. The bound on the worst tile
// catches what the whole output averages out: a tile that missed one K block, lost a row or has one element zeroed.
void expect_close(const xt::xarray<float>& ref, const xt::xarray<float>& out) {
    ASSERT_EQ(ref.shape(), out.shape());
    ASSERT_TRUE(ref.dimension() == 3 || (ref.dimension() == 2 && ref.shape()[0] % 32 == 0 && ref.shape()[1] % 32 == 0));
    EXPECT_LT(ttml::test_utils::relative_l2(out, ref), 1e-2);
    EXPECT_LT(worst_tile_relative_l2(ref, out), 1e-2);
}

// The output's label has to split it along `dim` and along nothing else.
void expect_sharded_along(const ttnn::Tensor& y, int dim) {
    bool sharded = false;
    for (const auto& placement : y.tensor_topology().placements()) {
        if (const auto* split = std::get_if<tt::tt_metal::distributed::MeshMapperConfig::Shard>(&placement)) {
            EXPECT_EQ(split->dim, dim);
            sharded = true;
        }
    }
    EXPECT_TRUE(sharded);
}

// Stripe s of the full gathered tensor: rows of an [N, K] or [M, K] one, columns of a [K, N] one.
xt::xarray<float> stripe_of(const xt::xarray<float>& full, size_t s, size_t width, bool rows) {
    if (rows) {
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

// Calls that reuse a buffer get a different gathered tensor: call i multiplies it by kCallScales[i % 4]. Powers of two
// and -1 are exact in bf16, so the reference is the same factor times the CPU result, and a call that read a stripe an
// earlier call left in its buffer would not match it.
constexpr std::array<float, 4> kCallScales = {1.0F, -2.0F, 0.5F, -1.0F};

float call_scale(uint32_t call) {
    return kCallScales[call % kCallScales.size()];
}

// The second operand's shape: stored [N, K] with transpose_b, else [K, N].
ttnn::Shape in1_shape(const Shape& s, bool transpose_b) {
    return transpose_b ? ttnn::Shape({s.N, s.K}) : ttnn::Shape({s.K, s.N});
}

// The gathered tensor's shape: the second operand for dim=-1, the first one, [M, K], for dim=-2.
ttnn::Shape gathered_shape(const Shape& s, bool transpose_b, int32_t dim) {
    return dim == -2 ? ttnn::Shape({s.M, s.K}) : in1_shape(s, transpose_b);
}

// Whether the gathered tensor's stripes are its rows (else its columns).
bool stripes_are_rows(bool transpose_b, int32_t dim) {
    return dim == -2 || transpose_b;
}

ttnn::Tensor scaled_gathered(const Problem& p, bool transpose_b, float scale) {
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<float> g = p.dim == -2 ? p.in0 : (transpose_b ? p.in1_nk : p.in1_kn);
    for (float& v : g) {
        v *= scale;
    }
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(
        *device, stripes_are_rows(transpose_b, p.dim) ? 0 : 1, kClusterAxis);
    return ttml::core::from_vector(
        g, gathered_shape(p.shape, transpose_b, p.dim), device, ttnn::Layout::TILE, shard.get());
}

// The operands of a call: the gathered tensor sharded over the ring, the other operand replicated as the input.
// dim=-1: the input is the first operand, [M, K] or with its leading dims as [batch, 1, M / batch, K]. dim=-2: it is
// the second operand.
struct Operands {
    ttnn::Tensor input;
    ttnn::Tensor gathered;
};

Operands make_operands(const Problem& p, bool transpose_b, uint32_t batch = 1) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto& s = p.shape;
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    const bool input_is_in1 = p.dim == -2;
    const ttnn::Shape in0_shape = batch == 1 ? ttnn::Shape({s.M, s.K}) : ttnn::Shape({batch, 1, s.M / batch, s.K});
    return {
        .input = ttml::core::from_vector(
            input_is_in1 ? (transpose_b ? p.in1_nk : p.in1_kn) : p.in0,
            input_is_in1 ? in1_shape(s, transpose_b) : in0_shape,
            device,
            ttnn::Layout::TILE,
            replicate.get()),
        .gathered = scaled_gathered(p, transpose_b, 1.0F)};
}

void run_fabric_gather(
    const Problem& p,
    const ttnn::Tensor& x,
    const ttnn::Tensor& g,
    uint32_t num_links,
    bool transpose_b,
    ttnn::ccl::Topology topology,
    const std::optional<Block>& block,
    uint32_t num_calls = kCalls,
    const std::optional<ttnn::CoreCoord>& matmul_grid = std::nullopt) {
    const int32_t dim = p.dim;
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
            .compute_with_storage_grid_size = matmul_grid.value_or(grid)};
    }

    const ttnn::Shape g_shape = gathered_shape(p.shape, transpose_b, dim);
    const std::array<ttnn::Tensor, 2> buffers = {
        ttml::core::zeros(g_shape, device), ttml::core::zeros(g_shape, device)};
    const auto semaphores = make_semaphores(device);
    std::vector<ttnn::Tensor> gathered;
    gathered.reserve(num_calls);
    for (uint32_t i = 0; i < num_calls; ++i) {
        gathered.push_back(call_scale(i) == 1.0F ? g : scaled_gathered(p, transpose_b, call_scale(i)));
    }

    std::vector<ttnn::Tensor> outputs;
    outputs.reserve(num_calls);
    for (uint32_t i = 0; i < num_calls; ++i) {
        outputs.push_back(ttml::metal::all_gather_minimal_matmul_outer_dim_async(
            x,
            gathered[i],
            buffers[i % 2],
            semaphores,
            kClusterAxis,
            topology,
            config,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            num_links,
            transpose_b,
            dim));
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
    // it bit for bit, in every tile. The local stripe is read from the shard and never lands in the buffer.
    const bool rows = stripes_are_rows(transpose_b, dim);
    for (uint32_t b = 0; b < 2 && b < num_calls; ++b) {
        const uint32_t last = (((num_calls - 1 - b) / 2) * 2) + b;
        SCOPED_TRACE(testing::Message() << "buffer " << b << " after call " << last);
        std::vector<xt::xarray<float>> shards;
        for (const auto& shard : ttnn::distributed::get_device_tensors(gathered[last])) {
            shards.push_back(ttml::core::to_xtensor(shard));
        }
        const auto gathered = ttnn::distributed::get_device_tensors(buffers[b]);
        for (size_t d = 0; d < gathered.size(); ++d) {
            const xt::xarray<float> buffer = ttml::core::to_xtensor(gathered[d]);
            for (size_t s = 0; s < shards.size(); ++s) {
                if (s == d) {
                    continue;
                }
                const size_t width = shards[s].shape()[rows ? 0 : 1];
                EXPECT_TRUE(xt::all(xt::equal(stripe_of(buffer, s, width, rows), shards[s])))
                    << "device " << d << " stripe " << s;
            }
        }
    }
}

// Both layouts of the second operand, 1 and 2 links, Ring and Linear, and three block configs that vary the gathered
// dimension's block: N for dim=-1, M for dim=-2.
void sweep(const Shape& s, int32_t dim = -1) {
    const bool rows = dim == -2;
    const std::array<std::optional<Block>, 3> blocks = {
        std::nullopt,
        rows ? Block{12, 8, 8, 2, 2} : Block{8, 8, 12, 2, 2},
        rows ? Block{6, 8, 8, 3, 1} : Block{8, 8, 6, 1, 3}};
    const Problem p = make_problem(s, dim);
    for (const bool transpose_b : {true, false}) {
        const Operands operands = make_operands(p, transpose_b);
        for (const uint32_t num_links : {1U, 2U}) {
            for (const auto topology : {ttnn::ccl::Topology::Ring, ttnn::ccl::Topology::Linear}) {
                for (const auto& block : blocks) {
                    SCOPED_TRACE(
                        testing::Message()
                        << "links=" << num_links << " transpose_b=" << transpose_b
                        << " ring=" << (topology == ttnn::ccl::Topology::Ring) << " block="
                        << (block.has_value()
                                ? std::to_string(rows ? block->M_block : block->N_block) + "/" +
                                      std::to_string(block->subblock_h) + "x" + std::to_string(block->subblock_w)
                                : std::string("default")));
                    run_fabric_gather(p, operands.input, operands.gathered, num_links, transpose_b, topology, block);
                }
            }
        }
    }
}

// The sequence-parallel layers: each device holds its [weight_n / ring, hidden] shard of the weight and its
// [sequence / ring, hidden] shard of the activation.
struct SequenceParallelLayer {
    const char* name;
    uint32_t weight_n;
    uint32_t hidden;
    uint32_t sequence;
};

constexpr std::array<SequenceParallelLayer, 2> kSequenceParallelLayers = {
    {{"wq", 4096, 4096, 2048}, {"w1", 11008, 4096, 2048}}};

// Calls 0 and 1 must share a program although every tensor address and the semaphores differ; call 2 uses one link
// instead of two, which is a new program.
void cache_hit_with_other_semaphores(const Shape& s, int32_t dim) {
    auto* device = &ttml::autograd::ctx().get_device();
    const Problem p = make_problem(s, dim);
    // Same shapes, different addresses: the hit has to patch the operands' addresses too.
    const std::array<Operands, 2> operands = {make_operands(p, true), make_operands(p, true)};
    const ttnn::Shape g_shape = gathered_shape(s, true, dim);
    const std::array<ttnn::Tensor, 2> buffers = {
        ttml::core::zeros(g_shape, device), ttml::core::zeros(g_shape, device)};
    const std::array<std::vector<tt::tt_metal::GlobalSemaphore>, 2> semaphores = {
        make_semaphores(device), make_semaphores(device)};
    size_t entries = 0;
    for (uint32_t i = 0; i < 3; ++i) {
        const uint32_t num_links = i < 2 ? 2 : 1;
        SCOPED_TRACE(testing::Message() << "call " << i);
        const auto y = ttml::metal::all_gather_minimal_matmul_outer_dim_async(
            operands[i % 2].input,
            operands[i % 2].gathered,
            buffers[i % 2],
            semaphores[i == 0 ? 0 : 1],
            kClusterAxis,
            ttnn::ccl::Topology::Ring,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            num_links,
            true,
            dim);
        if (i == 0) {
            entries = device->num_program_cache_entries();
        } else {
            EXPECT_EQ(device->num_program_cache_entries(), entries + (i - 1));
        }
        for (const auto& device_y : ttnn::distributed::get_device_tensors(y)) {
            expect_close(p.y, ttml::core::to_xtensor(device_y));
        }
        // Poisoning both counters of the first set proves later calls run on the second.
        for (const auto& semaphore : semaphores[0]) {
            semaphore.reset_semaphore_value((1000U << 16) | 1000U);
        }
    }
}
}  // namespace

// One semaphore set serves calls of any shape and of either dim, back to back. The injectors that count on it sit on
// row 0 of the grid for the dim=-1 shapes and for the second dim=-2 one, and on column 0 for the first dim=-2 one.
TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, SharedSemaphoresAcrossShapes) {
    auto* device = &ttml::autograd::ctx().get_device();
    struct Case {
        Shape shape;
        int32_t dim;
    };
    const std::array<Case, 5> cases = {
        {{{256, 512, 1024}, -1},
         {{128, 320, 128}, -2},
         {{256, 1024, 512}, -1},
         {{256, 320, 128}, -2},
         {{256, 512, 2048}, -1}}};
    constexpr uint32_t kRounds = 3;
    struct Layer {
        Problem p;
        ttnn::Tensor x;
        std::array<ttnn::Tensor, kRounds> g;
        std::array<ttnn::Tensor, 2> buffers;
    };
    std::vector<Layer> layers;
    layers.reserve(cases.size());
    for (const auto& c : cases) {
        Problem p = make_problem(c.shape, c.dim);
        Operands operands = make_operands(p, true);  // its gathered tensor is round 0's: call_scale(0) is 1
        std::array<ttnn::Tensor, kRounds> g = {
            std::move(operands.gathered),
            scaled_gathered(p, true, call_scale(1)),
            scaled_gathered(p, true, call_scale(2))};
        const ttnn::Shape full = gathered_shape(c.shape, true, c.dim);
        layers.push_back(
            {std::move(p),
             std::move(operands.input),
             std::move(g),
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
                    ttml::metal::all_gather_minimal_matmul_outer_dim_async(
                        layers[l].x,
                        layers[l].g[round],
                        layers[l].buffers[round % 2],
                        semaphores,
                        kClusterAxis,
                        ttnn::ccl::Topology::Ring,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        num_links,
                        true,
                        layers[l].p.dim));
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

TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, CacheHitWithOtherSemaphores) {
    cache_hit_with_other_semaphores({256, 512, 1024}, -1);
}

TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, GatherRowsCacheHitWithOtherSemaphores) {
    cache_hit_with_other_semaphores({1024, 512, 256}, -2);
}

// dim is part of the program, and it names the product's dimension, not a dim of the gathered tensor: the same three
// tensors give input @ gathered^T with dim=-1 and its transpose, gathered @ input^T, with dim=-2.
TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, OtherDimIsAnotherProgram) {
    auto* device = &ttml::autograd::ctx().get_device();
    const Shape s{256, 512, 1024};
    const Problem p = make_problem(s);
    const Operands operands = make_operands(p, true);  // [M, K] replicated and [N, K] sharded along its rows
    const ttnn::Shape g_shape({s.N, s.K});
    const std::array<ttnn::Tensor, 2> buffers = {
        ttml::core::zeros(g_shape, device), ttml::core::zeros(g_shape, device)};
    const auto semaphores = make_semaphores(device);
    const xt::xarray<float> y_transposed = xt::transpose(p.y);
    size_t entries = 0;
    for (const int32_t dim : {-1, -2}) {
        SCOPED_TRACE(testing::Message() << "dim=" << dim);
        const auto y = ttml::metal::all_gather_minimal_matmul_outer_dim_async(
            operands.input,
            operands.gathered,
            buffers[dim == -1 ? 0 : 1],
            semaphores,
            kClusterAxis,
            ttnn::ccl::Topology::Ring,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            2,
            true,
            dim);
        if (dim == -1) {
            entries = device->num_program_cache_entries();
        } else {
            EXPECT_EQ(device->num_program_cache_entries(), entries + 1);
        }
        for (const auto& device_y : ttnn::distributed::get_device_tensors(y)) {
            expect_close(dim == -1 ? p.y : y_transposed, ttml::core::to_xtensor(device_y));
        }
    }
}

// The checks that dim=-2 adds or changes.
TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, RejectsBadDimAndOperandShapes) {
    auto* device = &ttml::autograd::ctx().get_device();
    const Shape s{256, 256, 512};
    const Problem p = make_problem(s, -2);
    const Operands operands = make_operands(p, true);  // input [N, K], gathered [M, K] sharded along its rows
    const auto buffer = ttml::core::zeros(ttnn::Shape({s.M, s.K}), device);
    const auto semaphores = make_semaphores(device);
    auto call = [&](const ttnn::Tensor& input, const ttnn::Tensor& gather_buffer, int32_t dim) {
        ttml::metal::all_gather_minimal_matmul_outer_dim_async(
            input,
            operands.gathered,
            gather_buffer,
            semaphores,
            kClusterAxis,
            ttnn::ccl::Topology::Ring,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            1,
            true,
            dim);
    };
    EXPECT_THAT([&] { call(operands.input, buffer, 0); }, ThrowsMessage<std::runtime_error>(HasSubstr("dim must be")));
    // As the second operand the input has no leading dims to fold into M.
    const auto batched = ttml::core::zeros(ttnn::Shape({2, 1, s.N / 2, s.K}), device);
    EXPECT_THAT(
        [&] { call(batched, buffer, -2); },
        ThrowsMessage<std::runtime_error>(HasSubstr("the input tensor must be a matrix")));
    const auto other_k = ttml::core::zeros(ttnn::Shape({s.N, s.K / 2}), device);
    EXPECT_THAT(
        [&] { call(other_k, buffer, -2); },
        ThrowsMessage<std::runtime_error>(HasSubstr("inner dimensions must match")));
    // The buffer holds the whole gathered tensor, which is [M, K] here.
    const auto short_buffer = ttml::core::zeros(ttnn::Shape({s.M / 2, s.K}), device);
    EXPECT_THAT(
        [&] { call(operands.input, short_buffer, -2); },
        ThrowsMessage<std::runtime_error>(HasSubstr("persistent_gather_buffer shape")));
}

TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, SequenceParallelGathersTheActivation) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0, kClusterAxis);
    for (const SequenceParallelLayer& layer : kSequenceParallelLayers) {
        SCOPED_TRACE(layer.name);
        const Problem p = make_problem({layer.weight_n, layer.hidden, layer.sequence});
        const ttnn::Shape weight_shape({layer.weight_n, layer.hidden});
        const ttnn::Shape activation_shape({layer.sequence, layer.hidden});
        const auto weight_t = ttml::core::from_vector(p.in0, weight_shape, device, ttnn::Layout::TILE, shard.get());
        const auto activation =
            ttml::core::from_vector(p.in1_nk, activation_shape, device, ttnn::Layout::TILE, shard.get());
        const auto buffer = ttml::core::zeros(activation_shape, device);
        const auto y_t = ttml::metal::all_gather_minimal_matmul_outer_dim_async(
            weight_t, activation, buffer, make_semaphores(device), kClusterAxis, ttnn::ccl::Topology::Ring);

        const size_t n_per_device = layer.weight_n / kRingSize;
        const auto device_ys = ttnn::distributed::get_device_tensors(y_t);
        ASSERT_EQ(device_ys.size(), kRingSize);
        for (uint32_t d = 0; d < kRingSize; ++d) {
            SCOPED_TRACE(testing::Message() << "device " << d);
            const xt::xarray<float> ref = xt::view(p.y, xt::range(d * n_per_device, (d + 1) * n_per_device), xt::all());
            expect_close(ref, ttml::core::to_xtensor(device_ys[d]));
        }
    }
}

TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, NarrowStripes4D) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0, kClusterAxis);
    for (const auto& s : std::array<Shape, 2>{{{256, 512, 256}, {256, 256, 512}}}) {
        const Problem p = make_problem(s);
        const auto x = ttml::core::from_vector(
            p.in0, ttnn::Shape({2, 1, s.M / 2, s.K}), device, ttnn::Layout::TILE, replicate.get());
        const auto g =
            ttml::core::from_vector(p.in1_nk, ttnn::Shape({s.N, s.K}), device, ttnn::Layout::TILE, shard.get());
        for (const uint32_t num_links : {2U, 1U}) {
            SCOPED_TRACE(testing::Message() << "K=" << s.K << " N=" << s.N << " links=" << num_links);
            run_fabric_gather(p, x, g, num_links, true, ttnn::ccl::Topology::Ring, std::nullopt);
        }
    }
}

TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, TinyShapes) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0, kClusterAxis);
    // 320 = 8 + 2 tiles: a short last K block.
    for (const uint32_t K : {32U, 96U, 256U, 320U}) {
        for (const uint32_t N : {128U, 256U, 384U}) {
            const Shape s{128, K, N};
            const Problem p = make_problem(s);
            const auto x =
                ttml::core::from_vector(p.in0, ttnn::Shape({s.M, s.K}), device, ttnn::Layout::TILE, replicate.get());
            const auto g =
                ttml::core::from_vector(p.in1_nk, ttnn::Shape({s.N, s.K}), device, ttnn::Layout::TILE, shard.get());
            for (const uint32_t num_links : {2U, 1U}) {
                for (const auto topology : {ttnn::ccl::Topology::Ring, ttnn::ccl::Topology::Linear}) {
                    const bool ring = topology == ttnn::ccl::Topology::Ring;
                    SCOPED_TRACE(
                        testing::Message() << "K=" << K << " N=" << N << " links=" << num_links << " ring=" << ring);
                    run_fabric_gather(p, x, g, num_links, true, topology, std::nullopt);
                }
            }
        }
    }
}

TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, ShortLastKBlockAfterFullChunks) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0, kClusterAxis);
    // K = 20 and 34 tiles: a K chunk of 2 and of 4 whole K blocks, then a chunk that is one short K block.
    for (const auto& s : std::array<Shape, 2>{{{128, 640, 2048}, {128, 1088, 256}}}) {
        const Problem p = make_problem(s);
        const auto x =
            ttml::core::from_vector(p.in0, ttnn::Shape({s.M, s.K}), device, ttnn::Layout::TILE, replicate.get());
        const auto g =
            ttml::core::from_vector(p.in1_nk, ttnn::Shape({s.N, s.K}), device, ttnn::Layout::TILE, shard.get());
        for (const uint32_t num_links : {2U, 1U}) {
            SCOPED_TRACE(testing::Message() << "K=" << s.K << " N=" << s.N << " links=" << num_links);
            run_fabric_gather(p, x, g, num_links, true, ttnn::ccl::Topology::Ring, std::nullopt);
        }
    }
}

// A 4-column matmul grid with 2 links has a fabric sender in every column of the sink row, and this shape turns the
// grid so that the in1 injectors are column 0: one core is both, and its two counters share each semaphore.
TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, SenderSharesACoreWithAnInjector) {
    auto* device = &ttml::autograd::ctx().get_device();
    const Shape s{128, 512, 3840};
    const Problem p = make_problem(s);
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0, kClusterAxis);
    const auto x = ttml::core::from_vector(p.in0, ttnn::Shape({s.M, s.K}), device, ttnn::Layout::TILE, replicate.get());
    const auto g = ttml::core::from_vector(p.in1_nk, ttnn::Shape({s.N, s.K}), device, ttnn::Layout::TILE, shard.get());
    for (const auto topology : {ttnn::ccl::Topology::Ring, ttnn::ccl::Topology::Linear}) {
        SCOPED_TRACE(testing::Message() << "ring=" << (topology == ttnn::ccl::Topology::Ring));
        run_fabric_gather(p, x, g, 2, true, topology, Block{8, 8, 8, 4, 1}, kCalls, ttnn::CoreCoord{4, 10});
    }
}

TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, Small) {
    sweep({256, 512, 1024});
}

// dim=-2: the first operand is the gathered one, sharded along its rows.
TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, GatherRowsSmall) {
    sweep({1024, 512, 256}, -2);
}

// Stripes of one to three tiles, so that most chains of the gathered operand are empty, with and without a short last
// K block (320 = 8 + 2 tiles).
TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, GatherRowsTinyShapes) {
    for (const uint32_t K : {32U, 256U, 320U}) {
        for (const uint32_t M : {128U, 256U, 384U}) {
            const Problem p = make_problem({M, K, 128}, -2);
            const Operands operands = make_operands(p, true);
            for (const uint32_t num_links : {2U, 1U}) {
                for (const auto topology : {ttnn::ccl::Topology::Ring, ttnn::ccl::Topology::Linear}) {
                    const bool ring = topology == ttnn::ccl::Topology::Ring;
                    SCOPED_TRACE(
                        testing::Message() << "M=" << M << " K=" << K << " links=" << num_links << " ring=" << ring);
                    run_fabric_gather(p, operands.input, operands.gathered, num_links, true, topology, std::nullopt);
                }
            }
        }
    }
}

TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, GatherRowsShortLastKBlockAfterFullChunks) {
    // K = 20 and 34 tiles: a K chunk of 2 and of 4 whole K blocks, then a chunk that is one short K block.
    for (const auto& s : std::array<Shape, 2>{{{2048, 640, 128}, {256, 1088, 128}}}) {
        const Problem p = make_problem(s, -2);
        const Operands operands = make_operands(p, true);
        for (const uint32_t num_links : {2U, 1U}) {
            SCOPED_TRACE(testing::Message() << "M=" << s.M << " K=" << s.K << " links=" << num_links);
            run_fabric_gather(
                p, operands.input, operands.gathered, num_links, true, ttnn::ccl::Topology::Ring, std::nullopt);
        }
    }
}

// Blocks smaller than a chain's piece of the gathered rows and than a core's range of N: a hop region is cut into
// several M blocks (equal ones with M block 4, unequal with 2), every core has a full and a partial N block, and the
// subblocks are 2x2 with a short last one. No other dim=-2 test compares a core's second N block.
TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, GatherRowsSeveralBlocksPerCore) {
    const Problem p = make_problem({4096, 512, 960}, -2);
    const Operands operands = make_operands(p, true);
    for (const Block& block : {Block{4, 8, 2, 2, 2}, Block{2, 8, 2, 2, 2}}) {
        for (const auto topology : {ttnn::ccl::Topology::Ring, ttnn::ccl::Topology::Linear}) {
            SCOPED_TRACE(
                testing::Message() << "M_block=" << block.M_block
                                   << " ring=" << (topology == ttnn::ccl::Topology::Ring));
            run_fabric_gather(p, operands.input, operands.gathered, 2, true, topology, block, 2);
        }
    }
}

// As SenderSharesACoreWithAnInjector, for dim=-2: this shape keeps the grid as it is, so that the in0 injectors, which
// now wait for the stripes, are column 0.
TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, GatherRowsSenderSharesACoreWithAnInjector) {
    const Problem p = make_problem({3840, 512, 128}, -2);
    const Operands operands = make_operands(p, true);
    for (const auto topology : {ttnn::ccl::Topology::Ring, ttnn::ccl::Topology::Linear}) {
        SCOPED_TRACE(testing::Message() << "ring=" << (topology == ttnn::ccl::Topology::Ring));
        run_fabric_gather(
            p,
            operands.input,
            operands.gathered,
            2,
            true,
            topology,
            Block{8, 8, 8, 4, 1},
            kCalls,
            ttnn::CoreCoord{4, 10});
    }
}

// The sequence-parallel use without transposes: the activation [sequence, hidden] is gathered along the sequence and
// each device multiplies it by its own weight shard, stored [n, hidden], into its [sequence, n] slice of the output.
TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, SequenceParallelGathersTheActivationRows) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0, kClusterAxis);
    for (const SequenceParallelLayer& layer : kSequenceParallelLayers) {
        SCOPED_TRACE(layer.name);
        const Problem p = make_problem({layer.sequence, layer.hidden, layer.weight_n}, -2);
        const ttnn::Shape weight_shape({layer.weight_n, layer.hidden});
        const ttnn::Shape activation_shape({layer.sequence, layer.hidden});
        const auto weight = ttml::core::from_vector(p.in1_nk, weight_shape, device, ttnn::Layout::TILE, shard.get());
        const auto activation =
            ttml::core::from_vector(p.in0, activation_shape, device, ttnn::Layout::TILE, shard.get());
        const auto buffer = ttml::core::zeros(activation_shape, device);
        const auto y = ttml::metal::all_gather_minimal_matmul_outer_dim_async(
            weight,
            activation,
            buffer,
            make_semaphores(device),
            kClusterAxis,
            ttnn::ccl::Topology::Ring,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            1,
            true,
            -2);

        // The weight's N is dim 0 of its shard and dim 1 of the output, and the label has to follow it.
        expect_sharded_along(y, 1);

        const size_t n_per_device = layer.weight_n / kRingSize;
        const auto device_ys = ttnn::distributed::get_device_tensors(y);
        ASSERT_EQ(device_ys.size(), kRingSize);
        for (uint32_t d = 0; d < kRingSize; ++d) {
            SCOPED_TRACE(testing::Message() << "device " << d);
            const xt::xarray<float> ref = xt::view(p.y, xt::all(), xt::range(d * n_per_device, (d + 1) * n_per_device));
            expect_close(ref, ttml::core::to_xtensor(device_ys[d]));
        }
    }
}

// As above with the second operand stored the other way and with leading dims: wherever its N sits in the input, the
// label has to name the output's last dim, and every device its own slice of N.
TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, GatherRowsLabelFollowsTheShardedInput) {
    auto* device = &ttml::autograd::ctx().get_device();
    const Shape s{256, 256, 512};
    const Problem p = make_problem(s, -2);
    const ttnn::Tensor gathered = scaled_gathered(p, true, 1.0F);
    struct Case {
        const char* name;
        bool transpose_b;
        ttnn::Shape input_shape;
        int shard_dim;
    };
    const std::array<Case, 2> cases = {
        {{"[K, N]", false, ttnn::Shape({s.K, s.N}), 1}, {"[1, 1, N, K]", true, ttnn::Shape({1, 1, s.N, s.K}), 2}}};
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        const auto shard = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, c.shard_dim, kClusterAxis);
        const auto input = ttml::core::from_vector(
            c.transpose_b ? p.in1_nk : p.in1_kn, c.input_shape, device, ttnn::Layout::TILE, shard.get());
        const auto buffer = ttml::core::zeros(ttnn::Shape({s.M, s.K}), device);
        const auto y = ttml::metal::all_gather_minimal_matmul_outer_dim_async(
            input,
            gathered,
            buffer,
            make_semaphores(device),
            kClusterAxis,
            ttnn::ccl::Topology::Ring,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            1,
            c.transpose_b,
            -2);
        expect_sharded_along(y, 1);

        const size_t n_per_device = s.N / kRingSize;
        const auto device_ys = ttnn::distributed::get_device_tensors(y);
        ASSERT_EQ(device_ys.size(), kRingSize);
        for (uint32_t d = 0; d < kRingSize; ++d) {
            SCOPED_TRACE(testing::Message() << "device " << d);
            const xt::xarray<float> ref = xt::view(p.y, xt::all(), xt::range(d * n_per_device, (d + 1) * n_per_device));
            expect_close(ref, ttml::core::to_xtensor(device_ys[d]));
        }
    }
}

// The gathered operand can stay sharded along N on a mesh axis the op does not gather over. Here that is the fixture's
// second axis, of one device: every device holds the same [N, K], the gather joins the four copies, and the output's
// label has to name the output's last dim on that axis, although N is dim 0 of the gathered tensor.
TEST_F(AllGatherMinimalMatmulOuterDimAsyncTest, LabelFollowsAGatheredOperandShardedOnAnotherAxis) {
    auto* device = &ttml::autograd::ctx().get_device();
    const Shape s{256, 512, 256};
    const Problem p = make_problem(s);
    const auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    const auto other_axis = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0, 1 - kClusterAxis);
    const auto x = ttml::core::from_vector(p.in0, ttnn::Shape({s.M, s.K}), device, ttnn::Layout::TILE, replicate.get());
    const auto g =
        ttml::core::from_vector(p.in1_nk, ttnn::Shape({s.N, s.K}), device, ttnn::Layout::TILE, other_axis.get());
    const auto buffer = ttml::core::zeros(ttnn::Shape({kRingSize * s.N, s.K}), device);
    const auto y = ttml::metal::all_gather_minimal_matmul_outer_dim_async(
        x, g, buffer, make_semaphores(device), kClusterAxis, ttnn::ccl::Topology::Ring);
    expect_sharded_along(y, 1);

    // Four copies of the second operand, one under the other, give four copies of the product side by side.
    const xt::xarray<float> ref = xt::concatenate(xt::xtuple(p.y, p.y, p.y, p.y), 1);
    const auto device_ys = ttnn::distributed::get_device_tensors(y);
    ASSERT_EQ(device_ys.size(), kRingSize);
    for (uint32_t d = 0; d < kRingSize; ++d) {
        SCOPED_TRACE(testing::Message() << "device " << d);
        expect_close(ref, ttml::core::to_xtensor(device_ys[d]));
    }
}

namespace {
struct LargeCase {
    const char* name;
    uint32_t batch;
    Shape shape;
    int32_t dim = -1;
};

const LargeCase kLargeCases[] = {
    {"W1_2048x4096x11008", 1, {2048, 4096, 11008}},
    {"W2_2048x11008x4096", 1, {2048, 11008, 4096}},
    {"TransposedGrid_8192x1024x2048", 1, {8192, 1024, 2048}},
    {"Batch4_4x2048x4096x11008", 4, {8192, 4096, 11008}},
    {"SequenceParallelWQ_1024x4096x8192", 1, {1024, 4096, 8192}},
    {"SequenceParallelWKV_2048x4096x16384", 1, {2048, 4096, 16384}},
    {"SequenceParallelW1_2752x4096x8192", 1, {2752, 4096, 8192}},
    {"GatherRowsWQ_8192x4096x1024", 1, {8192, 4096, 1024}, -2},
    {"GatherRowsWKV_16384x4096x2048", 1, {16384, 4096, 2048}, -2},
    {"GatherRowsW1_8192x4096x2752", 1, {8192, 4096, 2752}, -2},
};

class AllGatherMinimalMatmulOuterDimAsyncLargeTest : public AllGatherMinimalMatmulOuterDimAsyncTest,
                                                     public ::testing::WithParamInterface<LargeCase> {};

std::string CaseName(const ::testing::TestParamInfo<LargeCase>& info) {
    return info.param.name;
}
}  // namespace

TEST_P(AllGatherMinimalMatmulOuterDimAsyncLargeTest, MatchesReference) {
    const LargeCase& c = GetParam();
    const Shape& s = c.shape;
    auto* device = &ttml::autograd::ctx().get_device();
    const Problem p = make_problem(s, c.dim, device->shape()[kClusterAxis]);
    const Operands operands = make_operands(p, true, c.batch);
    const auto grid = device->compute_with_storage_grid_size();
    const size_t available_links = ttnn::operations::ccl::common::get_num_links(*device, kClusterAxis);
    bool ran = false;
    for (const uint32_t num_links : {2U, 4U}) {
        if (num_links > available_links || 2 * num_links > grid.x) {
            continue;
        }
        for (const auto topology : {ttnn::ccl::Topology::Ring, ttnn::ccl::Topology::Linear}) {
            SCOPED_TRACE(
                testing::Message() << "ring=" << (topology == ttnn::ccl::Topology::Ring) << " links=" << num_links);
            run_fabric_gather(p, operands.input, operands.gathered, num_links, true, topology, std::nullopt, 2);
            ran = true;
        }
    }
    if (!ran) {
        GTEST_SKIP() << "needs at least 2 fabric links on the ring's axis, found " << available_links;
    }
}

INSTANTIATE_TEST_SUITE_P(
    LlamaShapes, AllGatherMinimalMatmulOuterDimAsyncLargeTest, ::testing::ValuesIn(kLargeCases), CaseName);
