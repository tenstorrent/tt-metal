// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <tt-metalium/bfloat16.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/chunk_gated_delta_rule_composite.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/relative_l2.hpp"

namespace {

using ttnn::DataType;

struct Case {
    uint32_t B, T, HK, HV, K, V, chunk;
    DataType gate_dtype;
    bool with_h0;
    std::string name;
};

struct Inputs {
    // q, k [B, T, HK, K]; v [B, T, HV, V]; g, beta [B, T, HV]; h0 [B, HV, K, V], each rounded to its device dtype.
    std::vector<float> q, k, v, g, beta, h0;
};

float round_bf16(float x) {
    return static_cast<float>(bfloat16(x));
}

Inputs make_inputs(const Case& c, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
    const auto gate = [&](float x) { return c.gate_dtype == DataType::BFLOAT16 ? round_bf16(x) : x; };
    Inputs in;
    // L2-normalized along K, the caller's contract.
    for (auto* x : {&in.q, &in.k}) {
        x->resize(size_t{c.B} * c.T * c.HK * c.K);
        for (size_t row = 0; row < x->size() / c.K; ++row) {
            float norm = 0.0F;
            for (uint32_t i = 0; i < c.K; ++i) {
                (*x)[row * c.K + i] = normal(gen);
                norm += (*x)[row * c.K + i] * (*x)[row * c.K + i];
            }
            for (uint32_t i = 0; i < c.K; ++i) {
                (*x)[row * c.K + i] = round_bf16((*x)[row * c.K + i] / std::sqrt(norm));
            }
        }
    }
    in.v.resize(size_t{c.B} * c.T * c.HV * c.V);
    for (auto& x : in.v) {
        x = round_bf16(normal(gen));
    }
    in.g.resize(size_t{c.B} * c.T * c.HV);
    in.beta.resize(in.g.size());
    for (size_t i = 0; i < in.g.size(); ++i) {
        in.g[i] = gate(-0.02F * std::log1p(std::exp(-normal(gen))));  // 0.02 * logsigmoid: a weak decay
        in.beta[i] = gate(uniform(gen));
    }
    in.h0.assign(c.with_h0 ? size_t{c.B} * c.HV * c.K * c.V : 0U, 0.0F);
    for (auto& x : in.h0) {
        x = 0.1F * normal(gen);
    }
    return in;
}

struct Reference {
    std::vector<double> o;            // [B*HV, T, V]
    std::vector<double> final_state;  // [B, HV, K, V]
};

// The token-by-token rule in double: S = exp(g) S; S += k (beta (v - S^T k))^T; o = scale S^T q.
Reference recurrent_reference(const Case& c, const Inputs& in) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(c.K));
    Reference ref{
        std::vector<double>(size_t{c.B} * c.HV * c.T * c.V), std::vector<double>(size_t{c.B} * c.HV * c.K * c.V)};
    std::vector<double> read(c.V);
    for (uint32_t b = 0; b < c.B; ++b) {
        for (uint32_t h = 0; h < c.HV; ++h) {
            const uint32_t hk = h / (c.HV / c.HK);
            double* s = &ref.final_state[(size_t{b} * c.HV + h) * c.K * c.V];
            for (size_t i = 0; i < size_t{c.K} * c.V; ++i) {
                s[i] = c.with_h0 ? in.h0[(size_t{b} * c.HV + h) * c.K * c.V + i] : 0.0;
            }
            for (uint32_t t = 0; t < c.T; ++t) {
                const size_t gate_at = (size_t{b} * c.T + t) * c.HV + h;
                const float* q = &in.q[((size_t{b} * c.T + t) * c.HK + hk) * c.K];
                const float* k = &in.k[((size_t{b} * c.T + t) * c.HK + hk) * c.K];
                const float* v = &in.v[((size_t{b} * c.T + t) * c.HV + h) * c.V];
                const double decay = std::exp(static_cast<double>(in.g[gate_at]));
                std::fill(read.begin(), read.end(), 0.0);
                for (uint32_t i = 0; i < c.K; ++i) {
                    for (uint32_t j = 0; j < c.V; ++j) {
                        s[i * c.V + j] *= decay;
                        read[j] += s[i * c.V + j] * k[i];
                    }
                }
                for (uint32_t j = 0; j < c.V; ++j) {
                    read[j] = (v[j] - read[j]) * in.beta[gate_at];
                }
                double* o = &ref.o[((size_t{b} * c.HV + h) * c.T + t) * c.V];
                for (uint32_t i = 0; i < c.K; ++i) {
                    for (uint32_t j = 0; j < c.V; ++j) {
                        s[i * c.V + j] += k[i] * read[j];
                        o[j] += scale * s[i * c.V + j] * q[i];
                    }
                }
            }
        }
    }
    return ref;
}

xt::xarray<float> to_float(const std::vector<double>& values, const xt::xarray<float>::shape_type& shape) {
    xt::xarray<float> out = xt::xarray<float>::from_shape(shape);
    std::copy(values.begin(), values.end(), out.begin());
    return out;
}

// The bound of the Python tests on o and final_state, 2-3x the error measured on Blackhole.
constexpr float kRelativeL2Max = 8e-3F;

class ChunkGatedDeltaRuleFwCompositeTest : public ttml::test_utils::DeviceSuiteTest,
                                           public ::testing::WithParamInterface<Case> {};

TEST_P(ChunkGatedDeltaRuleFwCompositeTest, MatchesRecurrentReference) {
    using namespace ttml;
    const auto& c = GetParam();
    auto* device = &autograd::ctx().get_device();
    const auto in = make_inputs(c, 7);
    const auto upload = [&](const std::vector<float>& x, ttnn::Shape shape, DataType dtype) {
        return dtype == DataType::FLOAT32 ? core::from_vector<float, DataType::FLOAT32>(x, shape, device)
                                          : core::from_vector<float, DataType::BFLOAT16>(x, shape, device);
    };
    const auto q = upload(in.q, ttnn::Shape{c.B, 1, c.T, c.HK * c.K}, DataType::BFLOAT16);
    const auto k = upload(in.k, ttnn::Shape{c.B, 1, c.T, c.HK * c.K}, DataType::BFLOAT16);
    const auto v = upload(in.v, ttnn::Shape{c.B, 1, c.T, c.HV * c.V}, DataType::BFLOAT16);
    const auto g = upload(in.g, ttnn::Shape{c.B, 1, c.T, c.HV}, c.gate_dtype);
    const auto beta = upload(in.beta, ttnn::Shape{c.B, 1, c.T, c.HV}, c.gate_dtype);
    std::optional<ttnn::Tensor> h0;
    if (c.with_h0) {
        h0 = upload(in.h0, ttnn::Shape{c.B, c.HV, c.K, c.V}, DataType::FLOAT32);
    }

    const auto result = ops::chunk_gated_delta_rule_fw_composite(q, k, v, g, beta, c.K, h0, c.chunk);
    const auto ref = recurrent_reference(c, in);
    const auto o = core::to_xtensor(result.output);
    const auto final_state = core::to_xtensor(result.final_state);
    EXPECT_LE(test_utils::relative_l2(o, to_float(ref.o, o.shape())), kRelativeL2Max) << "o";
    EXPECT_LE(test_utils::relative_l2(final_state, to_float(ref.final_state, final_state.shape())), kRelativeL2Max)
        << "final_state";
}

INSTANTIATE_TEST_SUITE_P(
    Cases,
    ChunkGatedDeltaRuleFwCompositeTest,
    ::testing::Values(
        Case{1, 128, 1, 2, 64, 64, 32, DataType::FLOAT32, true, "Chunk32F32GatesInitialState"},
        Case{2, 128, 2, 4, 32, 64, 64, DataType::BFLOAT16, false, "Chunk64Bf16GatesBatch2"}),
    [](const auto& info) { return info.param.name; });

}  // namespace
