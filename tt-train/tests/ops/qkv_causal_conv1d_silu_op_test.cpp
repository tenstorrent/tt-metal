// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/qkv_causal_conv1d_silu_op.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <random>
#include <tt-metalium/bfloat16.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace {

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;
using ttml::autograd::TensorPtr;

struct Shape {
    uint32_t T, Q, K, V;
    uint32_t C() const {
        return Q + K + V;
    }
};

xt::xarray<float> random_tensor(const std::vector<size_t>& shape, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    xt::xarray<float> out = xt::empty<float>(shape);
    for (auto& value : out) {
        value = static_cast<float>(bfloat16(dist(gen)));  // exact on the device, so references see the same inputs
    }
    return out;
}

struct Problem {
    xt::xarray<float> x, h, grad_y;
    std::array<xt::xarray<float>, 4> taps;
};

Problem random_problem(const Shape& s, uint32_t seed, float x_scale = 1.0F) {
    Problem p;
    p.x = x_scale * random_tensor({1, s.T, s.C()}, seed);
    p.h = x_scale * random_tensor({1, 3, s.C()}, seed + 1);
    p.grad_y = random_tensor({1, s.T, s.C()}, seed + 2);
    for (size_t j = 0; j < 4; ++j) {
        p.taps[j] = random_tensor({1, 1, s.C()}, seed + 3 + j);
    }
    return p;
}

struct Reference {
    xt::xarray<float> y, grad_input;
    std::array<xt::xarray<float>, 4> grad_taps;
};

Reference reference(const Shape& s, const Problem& p) {
    const size_t T = s.T;
    xt::xarray<float> window = xt::concatenate(xt::xtuple(p.h, p.x), 1);
    xt::xarray<float> preact = xt::zeros<float>({size_t{1}, T, size_t{s.C()}});
    for (size_t j = 0; j < 4; ++j) {
        preact += p.taps[j] * xt::view(window, xt::all(), xt::range(j, j + T), xt::all());
    }
    xt::xarray<float> sig = 1.0F / (1.0F + xt::exp(-preact));
    xt::xarray<float> grad_preact = p.grad_y * sig * (1.0F + preact * (1.0F - sig));
    xt::xarray<float> grad_window = xt::zeros_like(window);
    Reference ref;
    ref.y = preact * sig;
    for (size_t j = 0; j < 4; ++j) {
        xt::view(grad_window, xt::all(), xt::range(j, j + T), xt::all()) += p.taps[j] * grad_preact;
        ref.grad_taps[j] =
            xt::sum(grad_preact * xt::view(window, xt::all(), xt::range(j, j + T), xt::all()), {0, 1}, xt::keep_dims);
    }
    ref.grad_input = xt::view(grad_window, xt::all(), xt::range(3, 3 + T), xt::all());
    return ref;
}

// grad_input in double precision, for checks tighter than the float reference can support.
xt::xarray<double> grad_input_f64(const Shape& s, const Problem& p) {
    const size_t T = s.T;
    const auto f64 = [](const xt::xarray<float>& a) { return xt::xarray<double>(xt::cast<double>(a)); };
    xt::xarray<double> window = xt::concatenate(xt::xtuple(f64(p.h), f64(p.x)), 1);
    xt::xarray<double> preact = xt::zeros<double>({size_t{1}, T, size_t{s.C()}});
    for (size_t j = 0; j < 4; ++j) {
        preact += f64(p.taps[j]) * xt::view(window, xt::all(), xt::range(j, j + T), xt::all());
    }
    xt::xarray<double> sig = 1.0 / (1.0 + xt::exp(-preact));
    xt::xarray<double> grad_preact = f64(p.grad_y) * sig * (1.0 + preact * (1.0 - sig));
    xt::xarray<double> grad_window = xt::zeros_like(window);
    for (size_t j = 0; j < 4; ++j) {
        xt::view(grad_window, xt::all(), xt::range(j, j + T), xt::all()) += f64(p.taps[j]) * grad_preact;
    }
    return xt::view(grad_window, xt::all(), xt::range(3, 3 + T), xt::all());
}

// Tolerances relative to the tensor's largest magnitude: inputs and outputs are bf16, and grad_input sums four
// bf16-rounded products, so an element's error scales with the tensor's range rather than with its own value. The
// kernel's measured floor is 0.4% of max per element (the bf16 rounding of the output); 1% per element catches a
// corrupted row or tile, and the mean bound a bias too small to trip the per-element one. The forward rounds the
// preactivation to bf16 before the sigmoid, so its outputs carry about two bf16 ulps of the preactivation: 2%.
constexpr double max_rel_tolerance = 1e-2;
constexpr double forward_rel_tolerance = 2e-2;
constexpr double mean_rel_tolerance = 1e-3;

template <typename T>
void expect_close(
    const xt::xarray<T>& expected,
    const xt::xarray<T>& actual,
    const std::string& name,
    double max_rel = max_rel_tolerance) {
    ASSERT_EQ(expected.shape(), actual.shape()) << name;
    const double scale = xt::amax(xt::abs(expected))();
    const xt::xarray<double> diff = xt::abs(xt::cast<double>(actual) - xt::cast<double>(expected));
    const double max_diff = xt::amax(diff)();
    const double mean_diff = xt::mean(diff)();
    EXPECT_LE(max_diff, max_rel * scale) << name << " max|diff| = " << max_diff << " of max " << scale;
    EXPECT_LE(mean_diff, mean_rel_tolerance * scale) << name << " mean|diff| = " << mean_diff << " of max " << scale;
}

xt::xarray<float> slice_cols(const xt::xarray<float>& t, size_t begin, size_t end) {
    return xt::view(t, xt::all(), xt::all(), xt::range(begin, end));
}

// [1,T,W] -> [1,1,T,W], the rank ttml activations have.
xt::xarray<float> rank4(const xt::xarray<float>& a) {
    return xt::reshape_view(a, {size_t{1}, size_t{1}, a.shape()[1], a.shape()[2]});
}

// The cost model reproduces the choices measured on a 110-core grid (see qkv_causal_conv1d_silu_bw.cpp): the
// measured best, or one within 3% of it.
TEST(QkvCausalConv1dSiluWorkSplit, MatchesMeasuredChoices) {
    constexpr uint32_t cores = 110;
    constexpr uint64_t l1 = 1 << 20;
    const auto pick =
        [&](uint32_t channels, uint32_t rows, bool tap_grads = true, uint32_t chunk = 0, uint32_t splits = 0) {
            const auto w =
                ttml::metal::qkv_causal_conv1d_silu_bw_work_split(channels, rows, cores, l1, chunk, splits, tap_grads);
            return std::pair{w.channel_chunk_size, w.row_splits};
        };
    EXPECT_EQ(pick(2560, 1024), (std::pair{32U, 4U}));  // the Qwen3.8 per-chip shape
    EXPECT_EQ(pick(2560, 1024, false), (std::pair{32U, 4U}));
    EXPECT_EQ(pick(6144, 4096), (std::pair{32U, 4U}));
    EXPECT_EQ(pick(6144, 4096, false), (std::pair{32U, 4U}));
    EXPECT_EQ(pick(12288, 4096), (std::pair{32U, 2U}));  // measured best 64/4, 4% faster
    EXPECT_EQ(pick(12288, 4096, false), (std::pair{32U, 2U}));
    EXPECT_EQ(pick(6144, 512), (std::pair{64U, 1U}));
    EXPECT_EQ(pick(3072, 1024), (std::pair{32U, 1U}));
    // A fixed dimension is kept.
    EXPECT_EQ(pick(6144, 4096, true, 64), (std::pair{64U, 1U}));
    EXPECT_EQ(pick(6144, 4096, true, 0, 9).second, 9U);
    EXPECT_EQ(pick(6144, 4096, true, 128, 3), (std::pair{128U, 3U}));
    // The chunk divides C.
    EXPECT_EQ(192 % pick(192, 256).first, 0U);
    EXPECT_EQ(pick(32, 256).first, 32U);
}

class QkvCausalConv1dSiluTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

struct DeviceInputs {
    ttnn::Tensor input, history, grad_q, grad_k, grad_v;
    std::array<ttnn::Tensor, 4> taps;
};

DeviceInputs make_inputs(const Shape& s, const Problem& p) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    DeviceInputs d{
        .input = core::from_xtensor(p.x, device),
        .history = core::from_xtensor(p.h, device, ttnn::Layout::ROW_MAJOR),
        .grad_q = core::from_xtensor(slice_cols(p.grad_y, 0, s.Q), device),
        .grad_k = core::from_xtensor(slice_cols(p.grad_y, s.Q, s.Q + s.K), device),
        .grad_v = core::from_xtensor(slice_cols(p.grad_y, s.Q + s.K, s.C()), device),
        .taps = {
            core::from_xtensor(p.taps[0], device),
            core::from_xtensor(p.taps[1], device),
            core::from_xtensor(p.taps[2], device),
            core::from_xtensor(p.taps[3], device)}};
    return d;
}

// Runs the fused backward directly: (grad_input, grad_tap0..3).
auto run_metal_bw(
    const Shape& s,
    const DeviceInputs& d,
    uint32_t chunk,
    uint32_t splits,
    bool compute_tap_grads = true,
    const std::optional<ttnn::Tensor>& preallocated_grad_input = std::nullopt) {
    return ttml::metal::qkv_causal_conv1d_silu_bw(
        d.grad_q,
        d.grad_k,
        d.grad_v,
        d.input,
        d.history,
        d.taps[0],
        d.taps[1],
        d.taps[2],
        d.taps[3],
        s.Q,
        s.K,
        s.V,
        chunk,
        splits,
        compute_tap_grads,
        preallocated_grad_input);
}

auto run_metal_bw(const Shape& s, const Problem& p, uint32_t chunk, uint32_t splits) {
    return run_metal_bw(s, make_inputs(s, p), chunk, splits);
}

// The autograd op's inputs: [1,T,.] tensors at rank 3, [1,1,T,.] (the rank ttml activations have) at rank 4.
struct AutogradInputs {
    TensorPtr input;
    ttnn::Tensor history;
    std::array<TensorPtr, 4> taps;
};

AutogradInputs make_autograd_inputs(const Problem& p, bool taps_trainable, uint32_t rank = 3) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto at_rank = [&](const xt::xarray<float>& a) { return rank == 4 ? rank4(a) : a; };
    AutogradInputs a{
        .input = autograd::create_tensor(core::from_xtensor(at_rank(p.x), device), /* requires_grad */ true),
        .history = core::from_xtensor(at_rank(p.h), device, ttnn::Layout::ROW_MAJOR)};
    for (size_t j = 0; j < 4; ++j) {
        a.taps[j] = autograd::create_tensor(core::from_xtensor(at_rank(p.taps[j]), device), taps_trainable);
    }
    return a;
}

auto forward(const AutogradInputs& a, const Shape& s, uint32_t chunk = 0) {
    return ttml::ops::qkv_causal_conv1d_silu(
        a.input, a.history, a.taps[0], a.taps[1], a.taps[2], a.taps[3], s.Q, s.K, s.V, chunk);
}

// Seeds every output's gradient with its value and runs the backward; the reference then takes grad_y = y.
Reference backward_from_outputs(const Shape& s, Problem& p, const AutogradInputs& a) {
    p.grad_y = reference(s, p).y;
    auto [q, k, v] = forward(a, s);
    for (const auto& out : {q, k, v}) {
        out->set_grad(out->get_value());
    }
    q->backward();
    return reference(s, p);
}

// Checks every gradient against the reference: grad_input against the double one.
void check_metal_bw(
    const Shape& s,
    uint32_t chunk,
    uint32_t splits,
    uint32_t seed,
    const std::string& tag,
    bool compute_tap_grads = true,
    float x_scale = 1.0F) {
    using namespace ttml;
    const auto p = random_problem(s, seed, x_scale);
    auto [grad_input, g0, g1, g2, g3] = run_metal_bw(s, make_inputs(s, p), chunk, splits, compute_tap_grads);
    expect_close(
        grad_input_f64(s, p), xt::xarray<double>(xt::cast<double>(core::to_xtensor(grad_input))), tag + " grad_input");
    if (!compute_tap_grads) {
        EXPECT_FALSE(g0.has_value() || g1.has_value() || g2.has_value() || g3.has_value()) << tag;
        return;
    }
    const auto ref = reference(s, p);
    const std::array<ttnn::Tensor, 4> grads = {*g0, *g1, *g2, *g3};
    for (size_t j = 0; j < 4; ++j) {
        expect_close(ref.grad_taps[j], core::to_xtensor(grads[j]), tag + " grad_tap" + std::to_string(j));
    }
}

struct MetalCase {
    Shape shape;
    uint32_t chunk;
    uint32_t splits;
    bool tap_grads = true;  // false compiles the frozen-taps program: no partial, a different DFB set and budget
};

class QkvCausalConv1dSiluMetalTest : public QkvCausalConv1dSiluTest, public ::testing::WithParamInterface<MetalCase> {};

TEST_P(QkvCausalConv1dSiluMetalTest, MatchesReference) {
    const auto& c = GetParam();
    check_metal_bw(c.shape, c.chunk, c.splits, 7, "case", c.tap_grads);
}

INSTANTIATE_TEST_SUITE_P(
    Cases,
    QkvCausalConv1dSiluMetalTest,
    ::testing::Values(
        MetalCase{{32, 32, 32, 32}, 32, 1},
        MetalCase{{64, 96, 64, 32}, 32, 1},
        MetalCase{{64, 96, 64, 32}, 64, 2},
        // Blocks that are not a multiple of the DST batch: a buffer overrun here is silent, run with
        // TT_METAL_LLK_ASSERTS=1 to have the push checked against the buffer end.
        MetalCase{{96, 96, 64, 32}, 192, 2},  // 6 tiles per block: the sigmoid batches are 4 + 2
        MetalCase{{96, 96, 64, 32}, 96, 2},   // 3
        MetalCase{{64, 96, 32, 32}, 160, 2},  // 5
        MetalCase{{64, 96, 64, 64}, 224, 2},  // 7
        MetalCase{{96, 32, 32, 32}, 96, 3},
        MetalCase{{256, 128, 64, 64}, 128, 1},
        MetalCase{{256, 128, 64, 64}, 256, 8},
        MetalCase{{256, 128, 64, 64}, 32, 5},
        MetalCase{{512, 256, 128, 128}, 256, 0},
        MetalCase{{512, 256, 128, 128}, 32, 0},
        MetalCase{{256, 320, 320, 320}, 32, 5},  // 150 items: cores run two, some across a channel block
        MetalCase{{256, 128, 64, 64}, 0, 0},     // auto chunk and splits
        MetalCase{{96, 96, 64, 32}, 0, 2},       // auto chunk at a fixed split
        // Frozen taps, the consumer's mode: two items per core across a block, the widest block with a halo on
        // every split, a short DST batch.
        MetalCase{{256, 320, 320, 320}, 32, 5, false},
        MetalCase{{256, 128, 64, 64}, 256, 8, false},
        MetalCase{{64, 96, 64, 64}, 224, 2, false}));

// A program-cache hit reads the new run's buffers: the first run's inputs stay alive so the second lands elsewhere.
TEST_F(QkvCausalConv1dSiluTest, ProgramCacheHitReadsNewBuffers) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const Shape s{128, 64, 64, 64};
    const auto p_a = random_problem(s, 100);
    const auto p_b = random_problem(s, 101);
    const auto a = make_inputs(s, p_a);
    const auto out_a = run_metal_bw(s, a, 64, 2);
    const auto entries_after_first = device->num_program_cache_entries();
    const auto b = make_inputs(s, p_b);
    ASSERT_NE(a.input.buffer()->address(), b.input.buffer()->address());
    ASSERT_NE(a.grad_q.buffer()->address(), b.grad_q.buffer()->address());
    const auto [grad_input, g0, g1, g2, g3] = run_metal_bw(s, b, 64, 2);
    EXPECT_EQ(device->num_program_cache_entries(), entries_after_first) << "the second run must hit the cache";
    const auto ref = reference(s, p_b);
    expect_close(ref.grad_input, core::to_xtensor(grad_input), "grad_input");
    const std::array<ttnn::Tensor, 4> grads = {*g0, *g1, *g2, *g3};
    for (size_t j = 0; j < 4; ++j) {
        expect_close(ref.grad_taps[j], core::to_xtensor(grads[j]), "grad_tap" + std::to_string(j));
    }
    // Frozen taps compile a different program: a must-miss, so the flag stays in the hash.
    run_metal_bw(s, b, 64, 2, /*compute_tap_grads=*/false);
    EXPECT_EQ(device->num_program_cache_entries(), entries_after_first + 1) << "frozen taps must miss the cache";
}

// Invalid work splits are rejected by validation before anything is launched.
TEST_F(QkvCausalConv1dSiluTest, RejectsInvalidWorkSplit) {
    using namespace ttml;
    const auto run = [&](const Shape& s, uint32_t chunk, uint32_t splits) {
        run_metal_bw(s, random_problem(s, 3), chunk, splits);
    };
    const auto rejects = [](const std::string& message) {
        return ThrowsMessage<std::runtime_error>(HasSubstr(message));
    };
    EXPECT_THAT([&] { run({64, 128, 64, 64}, 96, 1); }, rejects("tile-aligned divisor of Q+K+V"));  // 96 vs C = 256
    EXPECT_THAT([&] { run({64, 128, 64, 64}, 64, 3); }, rejects("row_splits must be in [1, row_count / 32]"));
    EXPECT_THAT([&] { run({64, 512, 256, 256}, 1024, 1); }, rejects("bytes of L1 per core"));  // 32 tiles per block
    EXPECT_THAT([&] { run({48, 128, 64, 64}, 64, 1); }, rejects("row_count must be positive and tile aligned"));
    EXPECT_THAT([&] { run({64, 100, 64, 64}, 32, 1); }, rejects("Q/K/V widths must be tile aligned"));
}

// The forward rejects what the backward would reject, before any work is done.
TEST_F(QkvCausalConv1dSiluTest, ForwardRejectsInvalidShapes) {
    using namespace ttml;
    const auto run = [&](const Shape& s, uint32_t q_width, uint32_t chunk = 0, bool taps_trainable = false) {
        const auto a = make_autograd_inputs(random_problem(s, 3), taps_trainable);
        ops::qkv_causal_conv1d_silu(
            a.input, a.history, a.taps[0], a.taps[1], a.taps[2], a.taps[3], q_width, s.K, s.V, chunk);
    };
    EXPECT_THAT(
        [&] { run({48, 64, 64, 64}, 64); },  // T is not tile aligned: the forward's own check
        ThrowsMessage<std::runtime_error>(HasSubstr("sequence must be positive and tile aligned")));
    EXPECT_THAT(
        [&] { run({64, 64, 64, 64}, 32); },  // Q + K + V is not the channel count
        ThrowsMessage<std::runtime_error>(HasSubstr("Q+K+V is")));
    // A chunk that fits the forward's L1 budget but not the backward's is rejected here, not in backward(). The
    // backward's budget depends on the taps: chunk 512 at C = 2560 fits with frozen taps and not with trainable ones.
    const Shape wide{64, 1024, 1024, 512};
    EXPECT_NO_THROW(run(wide, wide.Q, 512));
    EXPECT_THAT(
        [&] { run(wide, wide.Q, 512, /*taps_trainable=*/true); },
        ThrowsMessage<std::runtime_error>(HasSubstr("fits the backward's L1 budget")));
    EXPECT_THAT(
        [&] { run(wide, wide.Q, 640); }, ThrowsMessage<std::runtime_error>(HasSubstr("fits the backward's L1 budget")));
    EXPECT_THAT(
        [&] { run(wide, wide.Q, 96); },  // 96 does not divide 2560
        ThrowsMessage<std::runtime_error>(HasSubstr("fits the backward's L1 budget")));
}

// Preactivations up to +-16: silu'(x) = s * (1 + x * (1 - s)) is sensitive to the precision of s near 1.
TEST_F(QkvCausalConv1dSiluTest, LargePreactivations) {
    // 1% of max|grad_input| per element against the double reference; a bf16 sigmoid gives several percent.
    for (const auto& [chunk, splits, seed] : {std::tuple{256U, 4U, 7U}, std::tuple{64U, 3U, 8U}}) {
        check_metal_bw(
            {256, 128, 64, 64}, chunk, splits, seed, "chunk " + std::to_string(chunk), true, /*x_scale=*/4.0F);
    }
}

TEST_F(QkvCausalConv1dSiluTest, NIGHTLY_RandomShapes) {
    std::mt19937 gen(42);
    const auto pick = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(gen); };
    for (uint32_t i = 0; i < 12; ++i) {
        const Shape s{32 * pick(1, 8), 32 * pick(1, 4), 32 * pick(1, 4), 32 * pick(1, 4)};
        std::vector<uint32_t> chunks;
        for (uint32_t c = 32; c <= std::min(s.C(), 256U); c += 32) {
            if (s.C() % c == 0) {
                chunks.push_back(c);
            }
        }
        const uint32_t chunk = chunks[pick(0, chunks.size() - 1)];
        const uint32_t splits = pick(1, s.T / 32);
        check_metal_bw(
            s,
            chunk,
            splits,
            1000 + i,
            "random T=" + std::to_string(s.T) + " QKV=" + std::to_string(s.Q) + "," + std::to_string(s.K) + "," +
                std::to_string(s.V) + " chunk=" + std::to_string(chunk) + " splits=" + std::to_string(splits));
    }
}

class QkvCausalConv1dSiluOpTest : public QkvCausalConv1dSiluTest, public ::testing::WithParamInterface<uint32_t> {};

// Autograd wrapper: forward matches, and loss = mean(q) + mean(k) + mean(v) reaches every input through all
// three outputs' nodes.
TEST_P(QkvCausalConv1dSiluOpTest, MatchesReference) {
    using namespace ttml;
    const Shape s{64, 96, 64, 32};
    const uint32_t channel_chunk_size = GetParam();
    auto p = random_problem(s, 1);
    for (const auto& [lo, hi] : {std::pair{0U, s.Q}, std::pair{s.Q, s.Q + s.K}, std::pair{s.Q + s.K, s.C()}}) {
        xt::view(p.grad_y, xt::all(), xt::all(), xt::range(lo, hi)) = 1.0F / static_cast<float>(s.T * (hi - lo));
    }
    const auto ref = reference(s, p);

    const auto a = make_autograd_inputs(p, /*taps_trainable=*/true, /*rank=*/4);
    auto [q, k, v] = forward(a, s, channel_chunk_size);
    expect_close(rank4(slice_cols(ref.y, 0, s.Q)), core::to_xtensor(q->get_value()), "q", forward_rel_tolerance);
    expect_close(
        rank4(slice_cols(ref.y, s.Q, s.Q + s.K)), core::to_xtensor(k->get_value()), "k", forward_rel_tolerance);
    expect_close(
        rank4(slice_cols(ref.y, s.Q + s.K, s.C())), core::to_xtensor(v->get_value()), "v", forward_rel_tolerance);

    auto loss = ops::add(ops::mean(q), ops::add(ops::mean(k), ops::mean(v)));
    loss->backward();

    expect_close(rank4(ref.grad_input), core::to_xtensor(a.input->get_grad()), "grad_input");
    for (size_t j = 0; j < 4; ++j) {
        expect_close(rank4(ref.grad_taps[j]), core::to_xtensor(a.taps[j]->get_grad()), "grad_tap");
    }
}

INSTANTIATE_TEST_SUITE_P(ChannelChunk, QkvCausalConv1dSiluOpTest, ::testing::Values(0U, 32U, 64U, 192U));

// The consumer's per-chip shape (Qwen3.8 27B, TP=4) through the autograd op with frozen taps, auto work split.
TEST_F(QkvCausalConv1dSiluTest, NIGHTLY_QwenShapeFrozenTaps) {
    using namespace ttml;
    const Shape s{1024, 512, 512, 1536};
    auto p = random_problem(s, 9);
    const auto a = make_autograd_inputs(p, /*taps_trainable=*/false);
    const auto ref = backward_from_outputs(s, p, a);
    expect_close(ref.grad_input, core::to_xtensor(a.input->get_grad()), "grad_input");
}

// A preallocated grad_input is written in place.
TEST_F(QkvCausalConv1dSiluTest, PreallocatedGradInput) {
    using namespace ttml;
    const Shape s{64, 96, 64, 32};
    const auto p = random_problem(s, 5);
    const auto ref = reference(s, p);
    const auto d = make_inputs(s, p);
    auto preallocated = ttnn::to_layout(core::zeros_like(d.input), ttnn::Layout::TILE);
    const auto [grad_input, g0, g1, g2, g3] = run_metal_bw(s, d, 0, 0, true, preallocated);
    EXPECT_EQ(grad_input.buffer()->address(), preallocated.buffer()->address());
    expect_close(ref.grad_input, core::to_xtensor(preallocated), "grad_input");
    // input is read after neighbouring rows of grad_input are written, so it cannot be the output.
    EXPECT_THAT(
        [&] { run_metal_bw(s, d, 0, 0, true, d.input); },
        ThrowsMessage<std::runtime_error>(HasSubstr("must not share a buffer with input")));
}

// Frozen taps: no tap gradient is produced, through the op and through the autograd wrapper.
TEST_F(QkvCausalConv1dSiluTest, FrozenTaps) {
    using namespace ttml;
    const Shape s{64, 96, 64, 32};
    auto p = random_problem(s, 4);
    p.grad_y = reference(s, p).y;
    const auto ref = reference(s, p);
    {
        const auto [grad_input, g0, g1, g2, g3] = run_metal_bw(s, make_inputs(s, p), 0, 0, /*compute_tap_grads=*/false);
        expect_close(ref.grad_input, core::to_xtensor(grad_input), "grad_input");
        EXPECT_FALSE(g0.has_value() || g1.has_value() || g2.has_value() || g3.has_value());
    }
    const auto a = make_autograd_inputs(p, /*taps_trainable=*/false);
    backward_from_outputs(s, p, a);
    expect_close(ref.grad_input, core::to_xtensor(a.input->get_grad()), "grad_input");
    for (const auto& tap : a.taps) {
        EXPECT_FALSE(tap->is_grad_initialized());
    }
}

// A loss that uses only q leaves k and v without a gradient; the backward treats them as zero.
TEST_F(QkvCausalConv1dSiluTest, SingleOutputLoss) {
    using namespace ttml;
    const Shape s{64, 96, 64, 32};
    auto p = random_problem(s, 2);
    p.grad_y = reference(s, p).y;
    xt::view(p.grad_y, xt::all(), xt::all(), xt::range(s.Q, s.C())) = 0.0F;
    const auto ref = reference(s, p);

    const auto a = make_autograd_inputs(p, /*taps_trainable=*/true);
    auto [q, k, v] = forward(a, s);
    q->set_grad(q->get_value());
    q->backward();

    expect_close(ref.grad_input, core::to_xtensor(a.input->get_grad()), "grad_input");
    for (size_t j = 0; j < 4; ++j) {
        expect_close(ref.grad_taps[j], core::to_xtensor(a.taps[j]->get_grad()), "grad_tap");
    }
}

}  // namespace
