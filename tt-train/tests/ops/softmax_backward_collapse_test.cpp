// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Softmax collapse in softmax backward and in attention backward (https://arxiv.org/abs/2610.31642, Section 2).
//
// For p = softmax(x) and upstream dy, dx = p * (dy - s) with s = sum_j p_j dy_j. When one entry k dominates its row
// (the others trail by a gap, tail T = sum_{j != k} p_j / p_k small), p_k rounds to 1 and s to dy_k, so
// dx_k = p_k * (dy_k - s) is a difference of two nearly equal values: its rounding error is ~eps * |dy_k| while the
// true value is p_k * sum_{j != k} p_j (dy_k - dy_j) ~ T * |dy|. The other entries stay accurate. Eq. 7 of the paper
// computes the dominant entry as -sum_{j != k} dx_j instead, which needs neither p_k nor the cancellation.
//
// The same expression is the attention-score gradient: dS = P * (dP - u), u = rowsum(dO * O), so a sharply peaked
// attention row gets a noisy gradient on its dominant score, and dQ (and dK of the dominant key) inherit it.
//
// Every check compares against a float64 oracle computed from the logits (not from the rounded probabilities the
// kernels receive), with the dominant entry evaluated in the stable Eq. 7 form, and the tolerance of each element
// scaled to the magnitudes that form sums. A correct Eq. 7 implementation meets these bounds from BF16 inputs; a
// collapsed dominant entry does not. Rows are swept over gaps so the output shows where the error begins.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ttnn/operations/moreh/moreh_softmax_backward/moreh_softmax_backward.hpp"

namespace ttml::ops::tests {

namespace {

// Relative to the magnitudes each value sums (see the file comment). BF16 inputs and outputs. The dominant entry is
// the collapse check; the other entries only guard against gross errors, since the kernels keep BF16 intermediates.
constexpr double kSoftmaxRtol = 2e-2;
constexpr double kSoftmaxOthersRtol = 1e-1;
constexpr double kAttentionRtol = 5e-2;
constexpr double kDenormalAtol = 1e-30;
constexpr size_t kMaxReportedMismatches = 8;

// Gap between the dominant logit and the largest of the others. 0 marks a control row of random logits.
constexpr std::array<double, 9> kGaps = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 24.0};

// Largest |device - reference| / tolerance per gap, over the dominant entries only.
using GapReport = std::map<double, double>;

std::string format_report(const GapReport& report) {
    std::ostringstream out;
    out << "gap: worst dominant-entry error in units of its tolerance\n";
    for (const auto& [gap, ratio] : report) {
        out << "  " << std::setw(4) << gap << ": " << std::setprecision(3) << ratio << '\n';
    }
    return out.str();
}

struct Mismatches {
    size_t count = 0;
    std::ostringstream detail;

    void add(const std::string& where, double got, double want, double tolerance) {
        if (count++ < kMaxReportedMismatches) {
            detail << "  " << where << ": got " << std::setprecision(6) << got << ", want " << want << " (tol "
                   << tolerance << ")\n";
        }
    }
};

ttnn::Tensor to_device(const std::vector<float>& values, const std::array<uint32_t, 4>& shape) {
    auto* device = &autograd::ctx().get_device();
    return core::from_vector(values, ttnn::Shape({shape[0], shape[1], shape[2], shape[3]}), device);
}

std::vector<double> read_back(const ttnn::Tensor& tensor) {
    const auto values = core::to_vector<float>(tensor);
    return {values.begin(), values.end()};
}

// ============================================================================
// Softmax backward on a [1, 1, R, W] block of rows.
// ============================================================================

struct SoftmaxRows {
    uint32_t rows = 0;
    uint32_t width = 0;
    std::vector<double> logits;      // exact
    std::vector<double> gap;         // per row
    std::vector<uint32_t> dominant;  // per row
};

SoftmaxRows make_softmax_rows(uint32_t rows, uint32_t width, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> spread(0.0, 3.0);
    std::uniform_real_distribution<double> random_logit(-2.0, 2.0);
    const std::array<uint32_t, 6> positions = {0U, width - 1U, width / 2U, 31U, 32U, 7U};
    SoftmaxRows result{.rows = rows, .width = width};
    result.logits.resize(static_cast<size_t>(rows) * width);
    for (uint32_t r = 0; r < rows; ++r) {
        const double gap = kGaps[r % kGaps.size()];
        auto* row = &result.logits[static_cast<size_t>(r) * width];
        uint32_t k = positions[(r / kGaps.size()) % positions.size()] % width;
        if (gap == 0.0) {
            for (uint32_t j = 0; j < width; ++j) {
                row[j] = random_logit(rng);
            }
            k = static_cast<uint32_t>(std::max_element(row, row + width) - row);
        } else {
            for (uint32_t j = 0; j < width; ++j) {
                row[j] = -gap - spread(rng);
            }
            row[k] = 0.0;
        }
        result.gap.push_back(gap);
        result.dominant.push_back(k);
    }
    return result;
}

std::vector<double> softmax64(const double* x, uint32_t width) {
    const double m = *std::max_element(x, x + width);
    std::vector<double> p(width);
    double sum = 0.0;
    for (uint32_t j = 0; j < width; ++j) {
        p[j] = std::exp(x[j] - m);
        sum += p[j];
    }
    for (auto& value : p) {
        value /= sum;
    }
    return p;
}

// dx and the per-element tolerance scale of one softmax row (see the file comment).
void softmax_backward_reference(
    const std::vector<double>& p,
    const double* dy,
    uint32_t k,
    std::vector<double>& dx,
    std::vector<double>& magnitude) {
    const auto width = static_cast<uint32_t>(p.size());
    double s = 0.0;
    for (uint32_t j = 0; j < width; ++j) {
        s += p[j] * dy[j];
    }
    dx.assign(width, 0.0);
    magnitude.assign(width, 0.0);
    double dominant = 0.0;
    double dominant_magnitude = 0.0;
    for (uint32_t j = 0; j < width; ++j) {
        if (j == k) {
            continue;
        }
        dx[j] = p[j] * (dy[j] - s);
        magnitude[j] = p[j] * (std::abs(dy[j]) + std::abs(s));
        dominant += p[j] * (dy[k] - dy[j]);
        dominant_magnitude += p[j] * (std::abs(dy[k]) + std::abs(dy[j]));
    }
    dx[k] = p[k] * dominant;
    magnitude[k] = p[k] * dominant_magnitude;
}

using SoftmaxBackward = ttnn::Tensor (*)(const ttnn::Tensor& y, const ttnn::Tensor& dy);

ttnn::Tensor metal_softmax_backward(const ttnn::Tensor& y, const ttnn::Tensor& dy) {
    return metal::softmax_backward(y, dy, /* dim */ 3);
}

ttnn::Tensor moreh_softmax_backward(const ttnn::Tensor& y, const ttnn::Tensor& dy) {
    return ttnn::moreh_softmax_backward(
        y,
        dy,
        /* axis */ 3,
        /* output */ std::nullopt,
        ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp::SOFTMAX,
        ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
        /* output_mem_config */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::precise());
}

void check_softmax_backward(SoftmaxBackward op, uint32_t width, uint32_t seed) {
    constexpr uint32_t kRows = 64U;
    const auto rows = make_softmax_rows(kRows, width, seed);
    std::vector<float> probabilities(rows.logits.size());
    for (uint32_t r = 0; r < kRows; ++r) {
        const auto p = softmax64(&rows.logits[static_cast<size_t>(r) * width], width);
        std::copy(p.begin(), p.end(), probabilities.begin() + static_cast<ptrdiff_t>(r) * width);
    }
    std::mt19937 rng(seed + 1U);
    std::uniform_real_distribution<float> upstream(-1.0F, 1.0F);
    std::vector<float> dy_values(rows.logits.size());
    for (auto& value : dy_values) {
        value = upstream(rng);
    }
    const std::array<uint32_t, 4> shape = {1U, 1U, kRows, width};
    const auto y = to_device(probabilities, shape);
    const auto dy = to_device(dy_values, shape);
    const auto dx = read_back(op(y, dy));
    const auto dy_device = read_back(dy);

    Mismatches dominant_mismatches;
    Mismatches other_mismatches;
    GapReport report;
    for (uint32_t r = 0; r < kRows; ++r) {
        const size_t offset = static_cast<size_t>(r) * width;
        const auto p = softmax64(&rows.logits[offset], width);
        std::vector<double> want;
        std::vector<double> magnitude;
        softmax_backward_reference(p, &dy_device[offset], rows.dominant[r], want, magnitude);
        for (uint32_t j = 0; j < width; ++j) {
            const bool is_dominant = j == rows.dominant[r];
            const double tolerance =
                (is_dominant ? kSoftmaxRtol : kSoftmaxOthersRtol) * magnitude[j] + kDenormalAtol;
            const double error = std::abs(dx[offset + j] - want[j]);
            if (is_dominant) {
                report[rows.gap[r]] = std::max(report[rows.gap[r]], error / tolerance);
            }
            if (!(error <= tolerance)) {
                (is_dominant ? dominant_mismatches : other_mismatches)
                    .add(
                        "row " + std::to_string(r) + " (gap " + std::to_string(rows.gap[r]) + ") col " +
                            std::to_string(j),
                        dx[offset + j],
                        want[j],
                        tolerance);
            }
        }
    }
    EXPECT_EQ(dominant_mismatches.count, 0U)
        << "width " << width << ": " << dominant_mismatches.count << " of " << kRows << " dominant entries wrong\n"
        << dominant_mismatches.detail.str() << format_report(report);
    EXPECT_EQ(other_mismatches.count, 0U) << "width " << width << ": " << other_mismatches.count
                                          << " non-dominant entries wrong\n"
                                          << other_mismatches.detail.str();
}

// ============================================================================
// Attention backward: one batch, one head, S queries and keys of dimension D, causal.
// ============================================================================

constexpr uint32_t kSeq = 64U;
constexpr uint32_t kHeadDim = 64U;

struct AttentionCase {
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> d_out;
    std::vector<double> gap;  // per query; 0 = random query
};

// Key j carries 1 in coordinate j % D and small noise elsewhere. A peaked query is a multiple of one coordinate,
// so its score against the matching key leads the others by about the chosen gap.
AttentionCase make_attention_case(uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
    std::uniform_real_distribution<float> noise(-0.02F, 0.02F);
    const float sqrt_d = std::sqrt(static_cast<float>(kHeadDim));
    AttentionCase c;
    c.q.resize(static_cast<size_t>(kSeq) * kHeadDim);
    c.k.resize(c.q.size());
    c.v.resize(c.q.size());
    c.d_out.resize(c.q.size());
    for (uint32_t j = 0; j < kSeq; ++j) {
        for (uint32_t d = 0; d < kHeadDim; ++d) {
            c.k[j * kHeadDim + d] = d == j % kHeadDim ? 1.0F : noise(rng);
        }
    }
    for (auto& value : c.v) {
        value = unit(rng);
    }
    for (auto& value : c.d_out) {
        value = unit(rng);
    }
    for (uint32_t q = 0; q < kSeq; ++q) {
        // Queries 0-7 attend to few keys under the causal mask; keep them random. Query 0 sees one key, so its
        // softmax is exactly 1 and its true dQ is exactly 0: the extreme case of a dominant score.
        const double gap = q < 8U ? 0.0 : kGaps[q % kGaps.size()];
        const uint32_t target = q - (q % 3U);  // a key at or before the query
        for (uint32_t d = 0; d < kHeadDim; ++d) {
            const float peaked = d == target % kHeadDim ? static_cast<float>(gap) * sqrt_d : 0.0F;
            c.q[q * kHeadDim + d] = gap == 0.0 ? unit(rng) : peaked;
        }
        c.gap.push_back(gap);
    }
    return c;
}

struct AttentionReference {
    std::vector<double> dq;
    std::vector<double> dk;
    std::vector<double> dq_magnitude;
    std::vector<double> dk_magnitude;
    std::vector<uint32_t> dominant;  // per query
};

// Float64 causal attention backward on the device's BF16 inputs, with the dominant score of every query row in the
// Eq. 7 form and per-element tolerance scales.
AttentionReference attention_reference(
    const std::vector<double>& q, const std::vector<double>& k, const std::vector<double>& v,
    const std::vector<double>& d_out) {
    const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    AttentionReference ref;
    ref.dq.assign(q.size(), 0.0);
    ref.dk.assign(k.size(), 0.0);
    ref.dq_magnitude.assign(q.size(), 0.0);
    ref.dk_magnitude.assign(k.size(), 0.0);
    for (uint32_t i = 0; i < kSeq; ++i) {
        const uint32_t keys = i + 1U;
        std::vector<double> scores(keys);
        for (uint32_t j = 0; j < keys; ++j) {
            double dot = 0.0;
            for (uint32_t d = 0; d < kHeadDim; ++d) {
                dot += q[i * kHeadDim + d] * k[j * kHeadDim + d];
            }
            scores[j] = dot * scale;
        }
        const auto p = softmax64(scores.data(), keys);
        const auto top = static_cast<uint32_t>(std::max_element(p.begin(), p.end()) - p.begin());
        ref.dominant.push_back(top);
        std::vector<double> dp(keys);
        for (uint32_t j = 0; j < keys; ++j) {
            double dot = 0.0;
            for (uint32_t d = 0; d < kHeadDim; ++d) {
                dot += d_out[i * kHeadDim + d] * v[j * kHeadDim + d];
            }
            dp[j] = dot;
        }
        std::vector<double> ds;
        std::vector<double> ds_magnitude;
        softmax_backward_reference(p, dp.data(), top, ds, ds_magnitude);
        for (uint32_t j = 0; j < keys; ++j) {
            for (uint32_t d = 0; d < kHeadDim; ++d) {
                ref.dq[i * kHeadDim + d] += scale * ds[j] * k[j * kHeadDim + d];
                ref.dq_magnitude[i * kHeadDim + d] += scale * ds_magnitude[j] * std::abs(k[j * kHeadDim + d]);
                ref.dk[j * kHeadDim + d] += scale * ds[j] * q[i * kHeadDim + d];
                ref.dk_magnitude[j * kHeadDim + d] += scale * ds_magnitude[j] * std::abs(q[i * kHeadDim + d]);
            }
        }
    }
    return ref;
}

using Attention = autograd::TensorPtr (*)(
    const autograd::TensorPtr&, const autograd::TensorPtr&, const autograd::TensorPtr&, const autograd::TensorPtr&);

autograd::TensorPtr fused_attention(
    const autograd::TensorPtr& q, const autograd::TensorPtr& k, const autograd::TensorPtr& v,
    const autograd::TensorPtr&) {
    return scaled_dot_product_attention(q, k, v);  // causal when no mask is given
}

autograd::TensorPtr composite_attention(
    const autograd::TensorPtr& q, const autograd::TensorPtr& k, const autograd::TensorPtr& v,
    const autograd::TensorPtr& causal_mask) {
    return scaled_dot_product_attention_composite(q, k, v, causal_mask);
}

void check_attention_backward(Attention attention, uint32_t seed) {
    const auto c = make_attention_case(seed);
    const std::array<uint32_t, 4> shape = {1U, 1U, kSeq, kHeadDim};
    auto q = autograd::create_tensor(to_device(c.q, shape), /* requires_grad */ true);
    auto k = autograd::create_tensor(to_device(c.k, shape), /* requires_grad */ true);
    auto v = autograd::create_tensor(to_device(c.v, shape), /* requires_grad */ true);
    std::vector<float> causal(static_cast<size_t>(kSeq) * kSeq, 0.0F);
    for (uint32_t i = 0; i < kSeq; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            causal[i * kSeq + j] = 1.0F;
        }
    }
    auto mask = autograd::create_tensor(to_device(causal, {1U, 1U, kSeq, kSeq}), /* requires_grad */ false);
    const auto d_out = to_device(c.d_out, shape);

    auto out = attention(q, k, v, mask);
    out->set_grad(d_out);
    out->backward();
    const auto dq = read_back(q->get_grad());
    const auto dk = read_back(k->get_grad());
    const auto ref =
        attention_reference(read_back(q->get_value()), read_back(k->get_value()), read_back(v->get_value()),
                            read_back(d_out));
    autograd::ctx().reset_graph();

    Mismatches mismatches;
    GapReport report;
    for (uint32_t i = 0; i < kSeq; ++i) {
        double worst = 0.0;
        for (uint32_t d = 0; d < kHeadDim; ++d) {
            const size_t index = i * kHeadDim + d;
            const double tolerance = kAttentionRtol * ref.dq_magnitude[index] + kDenormalAtol;
            const double error = std::abs(dq[index] - ref.dq[index]);
            worst = std::max(worst, error / tolerance);
            if (!(error <= tolerance)) {
                mismatches.add(
                    "dQ query " + std::to_string(i) + " (gap " + std::to_string(c.gap[i]) + ") dim " +
                        std::to_string(d),
                    dq[index],
                    ref.dq[index],
                    tolerance);
            }
        }
        report[c.gap[i]] = std::max(report[c.gap[i]], worst);
    }
    for (size_t index = 0; index < dk.size(); ++index) {
        const double tolerance = kAttentionRtol * ref.dk_magnitude[index] + kDenormalAtol;
        if (!(std::abs(dk[index] - ref.dk[index]) <= tolerance)) {
            mismatches.add(
                "dK key " + std::to_string(index / kHeadDim) + " dim " + std::to_string(index % kHeadDim),
                dk[index],
                ref.dk[index],
                tolerance);
        }
    }
    EXPECT_EQ(mismatches.count, 0U) << mismatches.count << " mismatches\n"
                                    << mismatches.detail.str() << "dQ rows by query gap, " << format_report(report);
}

}  // namespace

class SoftmaxBackwardCollapseTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }
    void TearDown() override {
        autograd::ctx().close_device();
    }
};

TEST_F(SoftmaxBackwardCollapseTest, MetalSoftmaxBackwardNarrow) {
    check_softmax_backward(metal_softmax_backward, /* width */ 64U, /* seed */ 11U);
}

TEST_F(SoftmaxBackwardCollapseTest, MetalSoftmaxBackwardWide) {
    check_softmax_backward(metal_softmax_backward, /* width */ 2048U, /* seed */ 12U);
}

TEST_F(SoftmaxBackwardCollapseTest, MorehSoftmaxBackwardNarrow) {
    check_softmax_backward(moreh_softmax_backward, /* width */ 64U, /* seed */ 13U);
}

TEST_F(SoftmaxBackwardCollapseTest, MorehSoftmaxBackwardWide) {
    check_softmax_backward(moreh_softmax_backward, /* width */ 2048U, /* seed */ 14U);
}

TEST_F(SoftmaxBackwardCollapseTest, FusedAttentionBackwardPeakedRows) {
    check_attention_backward(fused_attention, /* seed */ 21U);
}

TEST_F(SoftmaxBackwardCollapseTest, CompositeAttentionBackwardPeakedRows) {
    check_attention_backward(composite_attention, /* seed */ 22U);
}

}  // namespace ttml::ops::tests
