// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/chunk_gated_delta_rule_bw/chunk_gated_delta_rule_bw.hpp"
#include "ops/chunk_gated_delta_rule_composite.hpp"
#include "test_utils/relative_l2.hpp"

// The fused backward consumes the intermediates (h, v_new, A) the composite forward saves, so every accuracy test
// runs forward -> backward on the device and compares the gradients with the token-by-token definition of the
// recurrence differentiated by hand in double precision (the same oracle as the Python tests).

namespace {

using ttml::metal::ChunkGatedDeltaRuleBwNeeds;

constexpr auto kBF16 = ttnn::DataType::BFLOAT16;
constexpr auto kF32 = ttnn::DataType::FLOAT32;

// Gradients are accumulated in fp32 and rounded once to bf16 on the way out; the Python tests measure a worst
// relative error of 1e-2 (dg) against the same oracle.
constexpr float kMaxRelativeL2 = 2e-2F;

struct Dims {
    uint32_t B, T, HK, HV, K, V, C;
};

// Flat host copies of the problem in the layouts the op takes: token-major [B, T, H*D], gates [B, T, HV],
// states [B, HV, K, V].
struct Problem {
    Dims d;
    std::vector<float> q, k, v, g, beta, d_out, h0, d_final;
};

struct Reference {
    std::vector<double> dq, dk, dv, dg, dbeta, dh0;
};

std::vector<float> random_vector(size_t n, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> x(n);
    std::generate(x.begin(), x.end(), [&] { return dist(rng); });
    return x;
}

// Unit-norm rows of length `dim`: the recurrence diverges for keys that are not L2-normalized.
void l2_normalize_rows(std::vector<float>& x, size_t dim) {
    for (size_t i = 0; i < x.size(); i += dim) {
        double norm = 0.0;
        for (size_t j = 0; j < dim; ++j) {
            norm += static_cast<double>(x[i + j]) * x[i + j];
        }
        const auto inv = static_cast<float>(1.0 / std::sqrt(norm + 1e-12));
        for (size_t j = 0; j < dim; ++j) {
            x[i + j] *= inv;
        }
    }
}

Problem make_problem(const Dims& d, bool with_state, bool with_d_final, uint32_t seed) {
    std::mt19937 rng(seed);
    Problem p{.d = d};
    const size_t tokens = static_cast<size_t>(d.B) * d.T;
    p.q = random_vector(tokens * d.HK * d.K, rng, -1.0F, 1.0F);
    p.k = random_vector(tokens * d.HK * d.K, rng, -1.0F, 1.0F);
    l2_normalize_rows(p.q, d.K);
    l2_normalize_rows(p.k, d.K);
    p.v = random_vector(tokens * d.HV * d.V, rng, -1.0F, 1.0F);
    p.g = random_vector(tokens * d.HV, rng, -0.35F, -0.05F);  // log-space decay
    p.beta = random_vector(tokens * d.HV, rng, 0.2F, 0.9F);
    p.d_out = random_vector(tokens * d.HV * d.V, rng, -1.0F, 1.0F);
    const size_t state = static_cast<size_t>(d.B) * d.HV * d.K * d.V;
    if (with_state) {
        p.h0 = random_vector(state, rng, -0.2F, 0.2F);
    }
    if (with_d_final) {
        p.d_final = random_vector(state, rng, -0.2F, 0.2F);
    }
    return p;
}

// Token-by-token definition, differentiated by hand (reverse accumulation over t):
//   P_t = exp(g_t) S_{t-1};  r_t = P_t^T k_t;  u_t = beta_t (v_t - r_t);  S_t = P_t + k_t u_t^T;  o_t = S_t^T (scale
//   q_t)
// `p` must hold the values the device saw (after bf16 rounding), so the reference does not measure the rounding.
Reference reference_vjp(const Problem& p, float scale) {
    const auto& [B, T, HK, HV, K, V, C] = p.d;
    const uint32_t G = HV / HK;
    Reference ref{
        .dq = std::vector<double>(p.q.size(), 0.0),
        .dk = std::vector<double>(p.k.size(), 0.0),
        .dv = std::vector<double>(p.v.size(), 0.0),
        .dg = std::vector<double>(p.g.size(), 0.0),
        .dbeta = std::vector<double>(p.beta.size(), 0.0),
        .dh0 = std::vector<double>(p.h0.size(), 0.0)};
    const size_t kv = static_cast<size_t>(K) * V;

    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t hv = 0; hv < HV; ++hv) {
            const uint32_t hk = hv / G;
            const auto q_at = [&](uint32_t t) { return (static_cast<size_t>(b) * T + t) * HK * K + hk * K; };
            const auto v_at = [&](uint32_t t) { return (static_cast<size_t>(b) * T + t) * HV * V + hv * V; };
            const auto g_at = [&](uint32_t t) { return (static_cast<size_t>(b) * T + t) * HV + hv; };
            const size_t state_at = (static_cast<size_t>(b) * HV + hv) * kv;

            // Forward, keeping S_0 .. S_T (S[0] is the initial state).
            std::vector<std::vector<double>> S(T + 1U, std::vector<double>(kv, 0.0));
            if (!p.h0.empty()) {
                std::copy(p.h0.begin() + state_at, p.h0.begin() + state_at + kv, S[0].begin());
            }
            std::vector<double> r(static_cast<size_t>(T) * V), u(static_cast<size_t>(T) * V);
            for (uint32_t t = 0; t < T; ++t) {
                const double a = std::exp(static_cast<double>(p.g[g_at(t)]));
                const double beta = p.beta[g_at(t)];
                for (uint32_t j = 0; j < V; ++j) {
                    double acc = 0.0;
                    for (uint32_t i = 0; i < K; ++i) {
                        acc += a * S[t][i * V + j] * p.k[q_at(t) + i];
                    }
                    r[t * V + j] = acc;
                    u[t * V + j] = beta * (p.v[v_at(t) + j] - acc);
                }
                for (uint32_t i = 0; i < K; ++i) {
                    for (uint32_t j = 0; j < V; ++j) {
                        S[t + 1U][i * V + j] = a * S[t][i * V + j] + p.k[q_at(t) + i] * u[t * V + j];
                    }
                }
            }

            // Backward.
            std::vector<double> dS(kv, 0.0);
            if (!p.d_final.empty()) {
                std::copy(p.d_final.begin() + state_at, p.d_final.begin() + state_at + kv, dS.begin());
            }
            std::vector<double> dP(kv), du(V), dr(V);
            for (uint32_t t = T; t-- > 0;) {
                const double a = std::exp(static_cast<double>(p.g[g_at(t)]));
                const double beta = p.beta[g_at(t)];
                const double* d_o = nullptr;
                std::vector<double> d_o_vec(V);
                for (uint32_t j = 0; j < V; ++j) {
                    d_o_vec[j] = p.d_out[v_at(t) + j];
                }
                d_o = d_o_vec.data();

                // o_t = S_t^T (scale q_t)
                for (uint32_t i = 0; i < K; ++i) {
                    double dq = 0.0;
                    for (uint32_t j = 0; j < V; ++j) {
                        dq += S[t + 1U][i * V + j] * d_o[j];
                        dS[i * V + j] += static_cast<double>(scale) * p.q[q_at(t) + i] * d_o[j];
                    }
                    ref.dq[q_at(t) + i] += scale * dq;
                }
                // S_t = P_t + k_t u_t^T
                for (uint32_t j = 0; j < V; ++j) {
                    double acc = 0.0;
                    for (uint32_t i = 0; i < K; ++i) {
                        acc += dS[i * V + j] * p.k[q_at(t) + i];
                    }
                    du[j] = acc;
                }
                for (uint32_t i = 0; i < K; ++i) {
                    double acc = 0.0;
                    for (uint32_t j = 0; j < V; ++j) {
                        acc += dS[i * V + j] * u[t * V + j];
                    }
                    ref.dk[q_at(t) + i] += acc;
                }
                // u_t = beta_t (v_t - r_t)
                double dbeta = 0.0;
                for (uint32_t j = 0; j < V; ++j) {
                    dbeta += du[j] * (p.v[v_at(t) + j] - r[t * V + j]);
                    ref.dv[v_at(t) + j] += beta * du[j];
                    dr[j] = -beta * du[j];
                }
                ref.dbeta[g_at(t)] += dbeta;
                // r_t = P_t^T k_t, then P_t = exp(g_t) S_{t-1}
                double da = 0.0;
                for (uint32_t i = 0; i < K; ++i) {
                    double dk = 0.0;
                    for (uint32_t j = 0; j < V; ++j) {
                        const double p_ij = a * S[t][i * V + j];
                        dk += p_ij * dr[j];
                        dP[i * V + j] = dS[i * V + j] + p.k[q_at(t) + i] * dr[j];
                        da += S[t][i * V + j] * dP[i * V + j];
                    }
                    ref.dk[q_at(t) + i] += dk;
                }
                ref.dg[g_at(t)] += da * a;
                for (size_t x = 0; x < kv; ++x) {
                    dS[x] = a * dP[x];
                }
            }
            if (!p.h0.empty()) {
                std::copy(dS.begin(), dS.end(), ref.dh0.begin() + state_at);
            }
        }
    }
    return ref;
}

std::vector<float> to_host(const ttnn::Tensor& t) {
    return ttml::core::to_vector<float>(t);
}

// Host problem -> device tensors, and back: the problem is overwritten with what the device holds, so bf16 rounding of
// the inputs is part of the problem and not of the error.
struct DeviceProblem {
    ttnn::Tensor q, k, v, g, beta, d_out;
    std::optional<ttnn::Tensor> h0, d_final;
};

DeviceProblem upload(Problem& p, bool head_major_d_out) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto& [B, T, HK, HV, K, V, C] = p.d;
    const auto bf16 = [&](std::vector<float>& x, const ttnn::Shape& shape) {
        auto t = ttml::core::from_vector<float, kBF16>(x, shape, device);
        x = to_host(t);
        return t;
    };
    const auto f32 = [&](std::vector<float>& x, const ttnn::Shape& shape) {
        auto t = ttml::core::from_vector<float, kF32>(x, shape, device);
        x = to_host(t);
        return t;
    };
    auto q = bf16(p.q, ttnn::Shape({B, 1U, T, HK * K}));
    auto k = bf16(p.k, ttnn::Shape({B, 1U, T, HK * K}));
    auto v = bf16(p.v, ttnn::Shape({B, 1U, T, HV * V}));
    auto g = f32(p.g, ttnn::Shape({B, 1U, T, HV}));
    auto beta = f32(p.beta, ttnn::Shape({B, 1U, T, HV}));
    auto d_out = bf16(p.d_out, ttnn::Shape({B, 1U, T, HV * V}));  // also rounds the host copy
    if (head_major_d_out) {
        // [B, T, HV*V] -> [B*HV, 1, T, V]
        std::vector<float> head_major(p.d_out.size());
        for (uint32_t b = 0; b < B; ++b) {
            for (uint32_t t = 0; t < T; ++t) {
                for (uint32_t h = 0; h < HV; ++h) {
                    for (uint32_t j = 0; j < V; ++j) {
                        head_major[((static_cast<size_t>(b) * HV + h) * T + t) * V + j] =
                            p.d_out[((static_cast<size_t>(b) * T + t) * HV + h) * V + j];
                    }
                }
            }
        }
        d_out = ttml::core::from_vector<float, kBF16>(head_major, ttnn::Shape({B * HV, 1U, T, V}), device);
    }
    std::optional<ttnn::Tensor> h0, d_final;
    if (!p.h0.empty()) {
        h0 = f32(p.h0, ttnn::Shape({B, HV, K, V}));
    }
    if (!p.d_final.empty()) {
        d_final = bf16(p.d_final, ttnn::Shape({B, HV, K, V}));
    }
    return DeviceProblem{q, k, v, g, beta, d_out, h0, d_final};
}

// The forward's saved intermediates, as the op consumes them.
ttml::ops::ChunkGatedDeltaRuleCompositeResult run_forward(
    const DeviceProblem& dp, const Dims& d, std::optional<float> scale = std::nullopt) {
    return ttml::ops::chunk_gated_delta_rule_fw_composite(
        dp.q,
        dp.k,
        dp.v,
        dp.g,
        dp.beta,
        d.K,
        dp.h0,
        d.C,
        scale,
        /*output_head_major=*/true,
        kF32,
        /*save_for_backward=*/true);
}

std::vector<std::optional<ttnn::Tensor>> run_backward(
    const DeviceProblem& dp,
    const Dims& d,
    const ttml::ops::ChunkGatedDeltaRuleCompositeResult& fw,
    std::optional<float> scale = std::nullopt,
    const std::optional<ttnn::Tensor>& d_final = std::nullopt,
    const ChunkGatedDeltaRuleBwNeeds& needs = {}) {
    return ttml::metal::chunk_gated_delta_rule_bw(
        dp.q,
        dp.k,
        dp.v,
        dp.g,
        dp.beta,
        dp.d_out,
        *fw.h,
        *fw.v_new,
        *fw.a,
        d.K,
        dp.h0,
        d_final,
        d.C,
        scale,
        /*compute_kernel_config=*/std::nullopt,
        /*memory_config=*/std::nullopt,
        needs);
}

template <class T>
void expect_close(const std::vector<T>& reference, const std::optional<ttnn::Tensor>& got, const std::string& name) {
    ASSERT_TRUE(got.has_value()) << name << " was not returned";
    const auto host = to_host(*got);
    ASSERT_EQ(host.size(), reference.size()) << name;
    xt::xarray<float> expected = xt::zeros<float>({reference.size()});
    xt::xarray<float> actual = xt::zeros<float>({host.size()});
    for (size_t i = 0; i < host.size(); ++i) {
        expected(i) = static_cast<float>(reference[i]);
        actual(i) = host[i];
    }
    const float error = ttml::test_utils::relative_l2(actual, expected);
    EXPECT_TRUE(std::isfinite(error)) << name;
    EXPECT_LE(error, kMaxRelativeL2) << name << ": relative L2 error against the double-precision oracle";
}

void expect_all_gradients_close(
    const Reference& ref, const std::vector<std::optional<ttnn::Tensor>>& grads, bool has_state) {
    ASSERT_EQ(grads.size(), 6U);
    expect_close(ref.dq, grads[0], "dq");
    expect_close(ref.dk, grads[1], "dk");
    expect_close(ref.dv, grads[2], "dv");
    expect_close(ref.dg, grads[3], "dg");
    expect_close(ref.dbeta, grads[4], "dbeta");
    if (has_state) {
        expect_close(ref.dh0, grads[5], "dh0");
    } else {
        EXPECT_FALSE(grads[5].has_value()) << "dh0 must be empty without an initial state";
    }
}

float default_scale(uint32_t K) {
    return 1.0F / std::sqrt(static_cast<float>(K));
}

class ChunkGatedDeltaRuleBwTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

struct AccuracyCase {
    Dims dims;
    bool with_state;
    bool with_d_final;
    bool head_major_d_out;
};

class ChunkGatedDeltaRuleBwAccuracyTest : public ChunkGatedDeltaRuleBwTest,
                                          public ::testing::WithParamInterface<AccuracyCase> {};

TEST_P(ChunkGatedDeltaRuleBwAccuracyTest, MatchesRecurrentOracle) {
    const auto& c = GetParam();
    auto problem = make_problem(c.dims, c.with_state, c.with_d_final, /*seed=*/11U);
    const auto dp = upload(problem, c.head_major_d_out);

    const auto fw = run_forward(dp, c.dims);
    const auto grads = run_backward(dp, c.dims, fw, std::nullopt, dp.d_final);

    const auto ref = reference_vjp(problem, default_scale(c.dims.K));
    expect_all_gradients_close(ref, grads, c.with_state);
}

// {B, T, HK, HV, K, V, C}, with_state, with_d_final, head_major_d_out
INSTANTIATE_TEST_SUITE_P(
    Cases,
    ChunkGatedDeltaRuleBwAccuracyTest,
    ::testing::Values(
        AccuracyCase{{1, 128, 1, 1, 32, 32, 32}, false, false, false},  // one head, no optional inputs
        AccuracyCase{{1, 128, 2, 4, 64, 64, 32}, true, true, false},    // GVA G=2, both states
        AccuracyCase{{2, 128, 1, 3, 64, 64, 64}, true, false, true},    // GVA G=3, batch 2, chunk 64, head-major d_out
        AccuracyCase{{1, 128, 2, 2, 64, 128, 64}, false, true, false}));  // K != V

// Heads per chip of the Qwen3.8 TP=4 shape (4 key heads, 12 value heads, K = V = 128) at a sequence the double
// precision oracle handles in seconds; the full 8192 tokens are covered by the Python test.
TEST_F(ChunkGatedDeltaRuleBwTest, NIGHTLY_QwenTp4HeadsAtSequence2048) {
    const Dims dims{1, 2048, 4, 12, 128, 128, 64};
    auto problem = make_problem(dims, /*with_state=*/false, /*with_d_final=*/false, /*seed=*/5U);
    const auto dp = upload(problem, /*head_major_d_out=*/false);
    const auto fw = run_forward(dp, dims);
    const auto grads = run_backward(dp, dims, fw);
    expect_all_gradients_close(reference_vjp(problem, default_scale(dims.K)), grads, /*has_state=*/false);
}

// `needs` only skips the host work that follows a gradient: the entries it clears come back empty, the rest unchanged.
TEST_F(ChunkGatedDeltaRuleBwTest, NeedsClearsOnlyTheRequestedGradients) {
    const Dims dims{1, 64, 2, 4, 32, 32, 32};
    auto problem = make_problem(dims, /*with_state=*/true, /*with_d_final=*/false, /*seed=*/3U);
    const auto dp = upload(problem, /*head_major_d_out=*/false);
    const auto fw = run_forward(dp, dims);
    const auto grads = run_backward(
        dp,
        dims,
        fw,
        std::nullopt,
        std::nullopt,
        {.dq = false, .dk = true, .dv = false, .dg = true, .dbeta = false, .dh0 = false});
    ASSERT_EQ(grads.size(), 6U);
    EXPECT_FALSE(grads[0].has_value());
    EXPECT_TRUE(grads[1].has_value());
    EXPECT_FALSE(grads[2].has_value());
    EXPECT_TRUE(grads[3].has_value());
    EXPECT_FALSE(grads[4].has_value());
    EXPECT_FALSE(grads[5].has_value());
    const auto ref = reference_vjp(problem, default_scale(dims.K));
    expect_close(ref.dk, grads[1], "dk");
    expect_close(ref.dg, grads[3], "dg");
}

// A program-cache hit must read the new run's buffers and the new scale: the first run's tensors stay alive so the
// second run's land elsewhere, and `scale` is a runtime argument that the override has to refresh.  A different
// attribute that is baked into the program (d_final_state present) must miss.
TEST_F(ChunkGatedDeltaRuleBwTest, ProgramCacheHitReadsNewBuffersAndScale) {
    auto* device = &ttml::autograd::ctx().get_device();
    const Dims dims{1, 128, 1, 1, 32, 32, 32};  // G == 1: the host-side sum over value heads adds no program

    auto problem_a = make_problem(dims, false, false, /*seed=*/21U);
    const auto dp_a = upload(problem_a, false);
    const auto fw_a = run_forward(dp_a, dims);
    const auto grads_a = run_backward(dp_a, dims, fw_a);
    const auto entries_after_first = device->num_program_cache_entries();

    auto problem_b = make_problem(dims, false, false, /*seed=*/22U);
    const auto dp_b = upload(problem_b, false);
    const auto fw_b = run_forward(dp_b, dims);
    ASSERT_NE(dp_a.q.buffer()->address(), dp_b.q.buffer()->address());
    const auto grads_b = run_backward(dp_b, dims, fw_b);
    EXPECT_EQ(device->num_program_cache_entries(), entries_after_first) << "the second run must hit the cache";
    expect_all_gradients_close(reference_vjp(problem_b, default_scale(dims.K)), grads_b, false);

    // Same buffers, new scale: still a hit, and the gradients follow the new scale.
    const float scale = 0.25F;
    const auto grads_scaled = run_backward(dp_b, dims, fw_b, scale);
    EXPECT_EQ(device->num_program_cache_entries(), entries_after_first) << "scale is a runtime argument";
    // h, v_new and A do not depend on the scale (it only multiplies q in the output), so the saved forward is reused.
    expect_all_gradients_close(reference_vjp(problem_b, scale), grads_scaled, false);

    // d_final_state is part of the program: adding it compiles a different one.
    auto problem_c = make_problem(dims, false, true, /*seed=*/23U);
    const auto dp_c = upload(problem_c, false);
    const auto fw_c = run_forward(dp_c, dims);
    run_backward(dp_c, dims, fw_c, std::nullopt, dp_c.d_final);
    EXPECT_GT(device->num_program_cache_entries(), entries_after_first) << "d_final_state must miss the cache";
}

// ---- validation: every case perturbs one thing of otherwise valid (all-zero) inputs ----

struct ZeroInputs {
    ttnn::Tensor q, k, v, g, beta, d_out, h, v_new, a;
};

ZeroInputs zero_inputs(const Dims& d) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t BH = d.B * d.HV;
    const auto zeros = [&](const ttnn::Shape& shape, ttnn::DataType dtype) {
        return ttml::core::zeros(shape, device, dtype);
    };
    return {
        zeros(ttnn::Shape({d.B, 1U, d.T, d.HK * d.K}), kBF16),
        zeros(ttnn::Shape({d.B, 1U, d.T, d.HK * d.K}), kBF16),
        zeros(ttnn::Shape({d.B, 1U, d.T, d.HV * d.V}), kBF16),
        zeros(ttnn::Shape({d.B, 1U, d.T, d.HV}), kF32),
        zeros(ttnn::Shape({d.B, 1U, d.T, d.HV}), kF32),
        zeros(ttnn::Shape({d.B, 1U, d.T, d.HV * d.V}), kBF16),
        zeros(ttnn::Shape({BH, d.T / d.C, d.K, d.V}), kF32),
        zeros(ttnn::Shape({BH, 1U, d.T, d.V}), kF32),
        zeros(ttnn::Shape({BH, 1U, d.T, d.C}), kF32)};
}

void expect_rejected(const std::function<void()>& call, const std::string& message) {
    try {
        call();
        FAIL() << "expected an error containing \"" << message << "\"";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find(message), std::string::npos)
            << "error does not contain \"" << message << "\": " << e.what();
    }
}

void call_backward(const ZeroInputs& in, uint32_t key_head_dim, uint32_t chunk_size) {
    ttml::metal::chunk_gated_delta_rule_bw(
        in.q,
        in.k,
        in.v,
        in.g,
        in.beta,
        in.d_out,
        in.h,
        in.v_new,
        in.a,
        key_head_dim,
        std::nullopt,
        std::nullopt,
        chunk_size);
}

TEST_F(ChunkGatedDeltaRuleBwTest, ValidInputsAreAccepted) {
    const Dims d{1, 64, 1, 1, 32, 32, 32};
    EXPECT_NO_THROW(call_backward(zero_inputs(d), d.K, d.C));
}

TEST_F(ChunkGatedDeltaRuleBwTest, RejectsUnsupportedChunkSize) {
    const Dims d{1, 64, 1, 1, 32, 32, 32};
    expect_rejected([&] { call_backward(zero_inputs(d), d.K, 16U); }, "chunk_size must be 32 or 64");
}

TEST_F(ChunkGatedDeltaRuleBwTest, RejectsSequenceThatIsNotAMultipleOfChunk) {
    const Dims d{1, 96, 1, 1, 32, 32, 64};  // 96 = 1.5 chunks
    expect_rejected([&] { call_backward(zero_inputs(d), d.K, d.C); }, "must be a multiple of chunk_size");
}

TEST_F(ChunkGatedDeltaRuleBwTest, RejectsValueHeadsNotDivisibleByKeyHeads) {
    const Dims d{1, 64, 2, 3, 32, 32, 32};  // 2 key heads, 3 value heads
    expect_rejected([&] { call_backward(zero_inputs(d), d.K, d.C); }, "must be a multiple of key heads");
}

TEST_F(ChunkGatedDeltaRuleBwTest, RejectsHeadDimOffTheTileGrid) {
    const Dims d{1, 64, 1, 1, 48, 32, 32};
    expect_rejected([&] { call_backward(zero_inputs(d), d.K, d.C); }, "must be multiples of 32");
}

TEST_F(ChunkGatedDeltaRuleBwTest, RejectsHostTensor) {
    const Dims d{1, 64, 1, 1, 32, 32, 32};
    auto in = zero_inputs(d);
    in.q = in.q.cpu();
    expect_rejected([&] { call_backward(in, d.K, d.C); }, "q must be on Device");
}

TEST_F(ChunkGatedDeltaRuleBwTest, RejectsSavedTensorsInTheWrongDtype) {
    const Dims d{1, 64, 1, 1, 32, 32, 32};
    auto in = zero_inputs(d);
    in.v_new = ttnn::typecast(in.v_new, kBF16);
    expect_rejected([&] { call_backward(in, d.K, d.C); }, "v_new has unsupported dtype");
}

TEST_F(ChunkGatedDeltaRuleBwTest, RejectsSavedTensorsFromAnotherChunkSize) {
    const Dims d{1, 64, 1, 1, 32, 32, 32};
    auto in = zero_inputs(d);
    in.a = zero_inputs(Dims{1, 64, 1, 1, 32, 32, 64}).a;  // [1, 1, 64, 64]: saved with chunk 64, called with 32
    expect_rejected([&] { call_backward(in, d.K, d.C); }, "a must be");
}

TEST_F(ChunkGatedDeltaRuleBwTest, RejectsMismatchedGateDtypes) {
    const Dims d{1, 64, 1, 1, 32, 32, 32};
    auto in = zero_inputs(d);
    in.beta = ttnn::typecast(in.beta, kBF16);
    expect_rejected([&] { call_backward(in, d.K, d.C); }, "beta has unsupported dtype");
}

}  // namespace
