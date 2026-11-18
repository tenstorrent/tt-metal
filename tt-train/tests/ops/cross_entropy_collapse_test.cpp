// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Softmax collapse in the fused cross-entropy (https://arxiv.org/abs/2610.31642, Section 2).
//
// When the target is the row's maximum and the other logits trail it by a large gap, the loss is
// log1p(tail) ~ tail and the target's gradient is -s * tail / (1 + tail), with
// tail = sum_{j != k} exp(x[j] - x[k]). Computing log(1 + tail) or softmax(x)[t] - 1 after 1 + tail has
// been rounded loses both entirely once tail falls below the rounding unit of the format holding the
// sum: they come out exactly zero, and training stops pushing on confident tokens.
//
// Every check compares against a float64 oracle on the BF16 logits the device actually holds, with
// relative tolerances and no absolute slack beyond flushed denormals, so a collapsed value fails.
// The corrected fused log_softmax is a second, on-device oracle: CE(x, t) = -log_softmax(x)[t] and
// d CE / dx = log_softmax backward with upstream -s * one_hot(t).

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tt-metalium/experimental/tensor_layout_apis_with_custom_alignment.hpp>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "metal/ops/cross_entropy_bw/device/cross_entropy_bw_device_operation.hpp"
#include "metal/ops/cross_entropy_fw/device/cross_entropy_fw_device_operation.hpp"
#include "metal/ops/log_softmax/log_softmax.hpp"
#include "ops/losses.hpp"

namespace ttml::ops::tests {

namespace {

// Same relative tolerances as the log_softmax tests. Absolute slack only covers values that
// flush to zero as FP32/BF16 denormals.
constexpr double kBf16Rtol = 1e-2;
constexpr double kFp32Rtol = 3e-5;
constexpr double kFp32UnitRoundoff = 1.0 / 16777216.0;  // 2^-24
constexpr double kDenormalAtol = 1e-30;
constexpr uint32_t kNoTarget = 0xFFFFFFFFU;
constexpr size_t kMaxReportedMismatches = 8;

std::string shape_name(const ttnn::Shape& shape) {
    std::string name = "[";
    for (size_t d = 0; d < shape.rank(); ++d) {
        name += (d == 0 ? "" : ",") + std::to_string(shape[static_cast<int>(d)]);
    }
    return name + "]";
}

class MismatchLog {
public:
    explicit MismatchLog(std::string label) : label_(std::move(label)) {
    }
    MismatchLog(const MismatchLog&) = delete;
    MismatchLog& operator=(const MismatchLog&) = delete;
    ~MismatchLog() {
        EXPECT_EQ(failures_, 0U) << label_ << ": total mismatches (the first " << kMaxReportedMismatches
                                 << " are listed above)";
    }

    void near(std::string_view what, size_t row, size_t col, double actual, double expected, double tolerance) {
        if (std::abs(actual - expected) <= tolerance) {  // NaN compares false and is reported.
            return;
        }
        std::ostringstream text;
        text << std::setprecision(10) << what << " at row " << row << ", column " << col << ": actual " << actual
             << ", expected " << expected << ", tolerance " << tolerance;
        fail(text.str());
    }

    void expect(bool condition, std::string_view what, size_t row, size_t col, double actual) {
        if (condition) {
            return;
        }
        std::ostringstream text;
        text << std::setprecision(10) << "expected " << what << " at row " << row << ", column " << col << ", actual "
             << actual;
        fail(text.str());
    }

private:
    void fail(const std::string& message) {
        if (failures_ < kMaxReportedMismatches) {
            ADD_FAILURE() << label_ << ": " << message;
        }
        ++failures_;
    }

    std::string label_;
    size_t failures_ = 0;
};

// Float64 reference of one row: first maximum k, m = x[k], tail = sum_{j != k} exp(x[j] - m).
struct RowReference {
    size_t pivot = 0;
    double max = 0.0;
    double tail = 0.0;
    double log_normalizer = 0.0;  // log1p(tail)
};

RowReference reference_row(std::span<const float> row) {
    RowReference ref;
    for (size_t k = 1; k < row.size(); ++k) {
        if (row[k] > row[ref.pivot]) {  // Strict: the first of tied maxima.
            ref.pivot = k;
        }
    }
    ref.max = row[ref.pivot];
    for (size_t k = 0; k < row.size(); ++k) {
        if (k != ref.pivot) {
            ref.tail += std::exp(static_cast<double>(row[k]) - ref.max);
        }
    }
    ref.log_normalizer = std::log1p(ref.tail);
    return ref;
}

// Loss convention of the fused op: an out-of-range target contributes a zero logit.
double reference_loss(std::span<const float> row, const RowReference& ref, uint32_t target) {
    const double target_logit = target < row.size() ? static_cast<double>(row[target]) : 0.0;
    return (ref.max - target_logit) + ref.log_normalizer;
}

// s * (softmax(x)[j] - [j == target]). At j == target == k it is -s * tail / (1 + tail), which the float64
// subtraction would only approximate for tiny tails.
double reference_grad(std::span<const float> row, const RowReference& ref, uint32_t target, size_t j, double s) {
    if (j == target && j == ref.pivot) {
        return -s * ref.tail / (1.0 + ref.tail);
    }
    const double probability = std::exp(static_cast<double>(row[j]) - ref.max - ref.log_normalizer);
    return s * (probability - (j == target ? 1.0 : 0.0));
}

struct Case {
    ttnn::Shape shape;  // [N, 1, H, W]
    std::vector<float> logits;
    std::vector<uint32_t> targets;  // N * H
    std::vector<bool> confident;    // Rows whose target is the unique first maximum with a gap <= 24.

    explicit Case(const ttnn::Shape& s) :
        shape(s),
        logits(s.volume()),
        targets(static_cast<size_t>(s[0]) * s[2], 0U),
        confident(static_cast<size_t>(s[0]) * s[2], false) {
    }
    [[nodiscard]] size_t width() const {
        return shape[-1];
    }
    [[nodiscard]] size_t rows() const {
        return targets.size();
    }
    std::span<float> row(size_t r) {
        return std::span(logits).subspan(r * width(), width());
    }
};

struct DeviceCase {
    ttnn::DataType dtype = ttnn::DataType::BFLOAT16;
    ttnn::Tensor input;
    ttnn::Tensor target;
    std::vector<float> x;  // Quantized logits, as the device holds them.
    std::vector<RowReference> refs;
};

// FP32 outputs also allow the sequential FP32 accumulation of about width / 32 + 32 terms behind the tail,
// as in the log_softmax tests; a collapse loses the whole tail and still fails.
double rtol_for(const DeviceCase& d, size_t width) {
    return d.dtype == ttnn::DataType::FLOAT32 ? kFp32Rtol + static_cast<double>(width / 32 + 32) * kFp32UnitRoundoff
                                              : kBf16Rtol;
}

DeviceCase upload(const Case& c, ttnn::DataType dtype = ttnn::DataType::BFLOAT16) {
    auto* device = &autograd::ctx().get_device();
    DeviceCase d;
    d.dtype = dtype;
    d.input = dtype == ttnn::DataType::FLOAT32
                  ? core::from_vector<float, ttnn::DataType::FLOAT32>(c.logits, c.shape, device)
                  : core::from_vector(c.logits, c.shape, device);
    d.target = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        c.targets, ttnn::Shape({c.shape[0], c.shape[2]}), device, ttnn::Layout::ROW_MAJOR);
    d.x = core::to_vector(d.input);
    for (size_t r = 0; r < c.rows(); ++r) {
        d.refs.push_back(reference_row(std::span<const float>(d.x).subspan(r * c.width(), c.width())));
    }
    return d;
}

void check_loss(const Case& c, const DeviceCase& d, std::span<const float> loss, const std::string& label) {
    MismatchLog mismatches(label);
    ASSERT_EQ(loss.size(), c.rows());
    for (size_t r = 0; r < c.rows(); ++r) {
        const auto row = std::span<const float>(d.x).subspan(r * c.width(), c.width());
        const double expected = reference_loss(row, d.refs[r], c.targets[r]);
        mismatches.near("loss", r, c.targets[r], loss[r], expected, rtol_for(d, c.width()) * std::abs(expected));
        if (c.confident[r]) {
            mismatches.expect(loss[r] > 0.F, "loss > 0 (confident target, collapse)", r, c.targets[r], loss[r]);
        }
    }
}

// `row_scale[r]` is the scale s of row r (the scaler times the row's upstream gradient).
void check_grad(
    const Case& c,
    const DeviceCase& d,
    std::span<const float> dx,
    std::span<const double> row_scale,
    const std::string& label) {
    MismatchLog mismatches(label);
    ASSERT_EQ(dx.size(), c.logits.size());
    ASSERT_EQ(row_scale.size(), c.rows());
    for (size_t r = 0; r < c.rows(); ++r) {
        const auto row = std::span<const float>(d.x).subspan(r * c.width(), c.width());
        const size_t base = r * c.width();
        const double s = row_scale[r];
        for (size_t j = 0; j < c.width(); ++j) {
            const double expected = reference_grad(row, d.refs[r], c.targets[r], j, s);
            mismatches.near(
                "dx", r, j, dx[base + j], expected, rtol_for(d, c.width()) * std::abs(expected) + kDenormalAtol);
        }
        if (c.confident[r] && s != 0.0) {
            const size_t t = c.targets[r];
            mismatches.expect(
                dx[base + t] * s < 0.0,
                "dx[target] nonzero, sign of -s (confident target, collapse)",
                r,
                t,
                dx[base + t]);
        }
    }
}

void check_grad(const Case& c, const DeviceCase& d, std::span<const float> dx, double s, const std::string& label) {
    const std::vector<double> row_scale(c.rows(), s);
    check_grad(c, d, dx, row_scale, label);
}

std::vector<float> run_loss(const DeviceCase& d) {
    return core::to_vector(metal::cross_entropy_fw(d.input, d.target));
}

std::vector<float> run_grad(const DeviceCase& d, float scaler) {
    auto* device = &autograd::ctx().get_device();
    const auto grad = core::ones(ttnn::Shape({1, 1, 1, 1}), device);
    return core::to_vector(metal::cross_entropy_bw(d.input, d.target, grad, scaler));
}

// Rows whose target is the unique first maximum. Competitors trail by `gap`:
//   one competitor in the target's SFPU lane (a tile further along), one in another lane, or all of them.
// The rest sit 100 below the maximum, where exp underflows to an FP32 denormal (flushed on device).
enum class Competitors : uint8_t { SameLane, OtherLane, All };

void fill_confident_row(std::span<float> row, size_t target, float offset, float gap, Competitors competitors) {
    const size_t width = row.size();
    std::fill(row.begin(), row.end(), competitors == Competitors::All ? offset - gap : offset - 100.F);
    if (competitors != Competitors::All && width > 1) {
        const size_t step = competitors == Competitors::SameLane && width > 32 ? 32 : 1;
        row[(target + step) % width] = offset - gap;
    }
    row[target] = offset;
}

// Gaps on both sides of the BF16 (about 2^-8) and FP32 (2^-24) rounding cliffs of 1 + tail.
constexpr std::array<float, 6> kGaps{8.F, 9.F, 12.F, 17.F, 20.F, 24.F};
constexpr std::array<float, 3> kOffsets{0.F, -32.F, 32.F};

Case confident_case(const ttnn::Shape& shape) {
    Case c(shape);
    const size_t width = c.width();
    for (size_t r = 0; r < c.rows(); ++r) {
        const std::array<size_t, 4> positions{0, width - 1, width / 2, (r * 131 + 7) % width};
        const size_t target = positions[r % positions.size()];
        const auto competitors = static_cast<Competitors>((r / kGaps.size()) % 3);
        fill_confident_row(c.row(r), target, kOffsets[r % kOffsets.size()], kGaps[r % kGaps.size()], competitors);
        c.targets[r] = static_cast<uint32_t>(target);
        c.confident[r] = true;
    }
    return c;
}

}  // namespace

class CrossEntropyCollapseTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        autograd::ctx().close_device();
    }

protected:
    void TearDown() override {
        autograd::ctx().reset_graph();
    }
};

// The target is the maximum and every other logit trails by 8-24: the loss and the target's gradient must
// stay strictly positive / negative and within BF16 relative tolerance of float64.
TEST_F(CrossEntropyCollapseTest, ConfidentTargetKeepsLossAndGradient) {
    for (const auto& shape :
         {ttnn::Shape({1, 1, 32, 32000}),
          ttnn::Shape({2, 1, 35, 1000}),
          ttnn::Shape({1, 1, 64, 35}),
          ttnn::Shape({4, 1, 64, 4096})}) {
        SCOPED_TRACE(shape_name(shape));
        const auto c = confident_case(shape);
        const auto d = upload(c);
        check_loss(c, d, run_loss(d), shape_name(shape) + " forward");
        const float scaler = 1.F / static_cast<float>(c.rows());
        check_grad(c, d, run_grad(d, scaler), scaler, shape_name(shape) + " backward");
    }
}

// FP32 logits (supported by the fused program): FP32 loss and gradient at FP32 tolerance, so the gaps
// beyond the BF16 cliff (17-24, below the FP32 one) are resolved too.
TEST_F(CrossEntropyCollapseTest, Fp32ConfidentTargetKeepsLossAndGradient) {
    for (const auto& shape : {ttnn::Shape({1, 1, 32, 32000}), ttnn::Shape({2, 1, 35, 1000})}) {
        SCOPED_TRACE(shape_name(shape));
        const auto c = confident_case(shape);
        const auto d = upload(c, ttnn::DataType::FLOAT32);
        const auto loss = metal::cross_entropy_fw(d.input, d.target);
        EXPECT_EQ(loss.dtype(), ttnn::DataType::FLOAT32);
        check_loss(c, d, core::to_vector(loss), shape_name(shape) + " fp32 forward");
        const float scaler = 1.F / static_cast<float>(c.rows());
        check_grad(c, d, run_grad(d, scaler), scaler, shape_name(shape) + " fp32 backward");
    }
}

// The loss's upstream gradient is folded into the kernel's row scale: one value (MEAN reduction) or one
// per row (NONE, e.g. GRPO's per-token weights), BF16 or FP32, including negative and zero entries.
TEST_F(CrossEntropyCollapseTest, UpstreamGradientScalesRows) {
    auto* device = &autograd::ctx().get_device();
    const auto shape = ttnn::Shape({2, 1, 35, 1000});
    const auto c = confident_case(shape);
    const auto d = upload(c);
    const float scaler = 0.25F;
    std::mt19937 rng(77U);
    std::uniform_real_distribution<float> value(-2.F, 2.F);
    for (const auto dtype : {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32}) {
        for (const bool per_row : {false, true}) {
            const std::string label = std::string(per_row ? "per-row" : "scalar") + " upstream " +
                                      (dtype == ttnn::DataType::FLOAT32 ? "fp32" : "bf16");
            SCOPED_TRACE(label);
            std::vector<float> g(per_row ? c.rows() : 1);
            for (auto& v : g) v = value(rng);
            g[0] = per_row ? 0.F : g[0];
            const auto upstream_shape = per_row ? ttnn::Shape({shape[0], 1, shape[2], 1}) : ttnn::Shape({1, 1, 1, 1});
            const auto upstream = dtype == ttnn::DataType::FLOAT32
                                      ? core::from_vector<float, ttnn::DataType::FLOAT32>(g, upstream_shape, device)
                                      : core::from_vector(g, upstream_shape, device);
            const auto quantized = core::to_vector(upstream);
            std::vector<double> row_scale(c.rows());
            for (size_t r = 0; r < c.rows(); ++r) {
                row_scale[r] = static_cast<double>(scaler) * quantized[per_row ? r : 0];
            }
            const auto dx = core::to_vector(metal::cross_entropy_bw(d.input, d.target, upstream, scaler));
            check_grad(c, d, dx, row_scale, label);
        }
    }
}

// Random logits at the NanoLlama3 vocabulary, mixed with confident rows, tied maxima whose target is the
// later copy (not the selected first maximum) and out-of-range targets (zero target logit, no -s term).
TEST_F(CrossEntropyCollapseTest, MixedRowsMatchFloat64) {
    for (const auto& shape : {ttnn::Shape({2, 1, 256, 32000}), ttnn::Shape({1, 1, 33, 1025})}) {
        SCOPED_TRACE(shape_name(shape));
        Case c(shape);
        const size_t width = c.width();
        std::mt19937 rng(2026U);
        std::uniform_real_distribution<float> logit(-10.F, 10.F);
        std::uniform_int_distribution<uint32_t> column(0U, static_cast<uint32_t>(width - 1));
        for (size_t r = 0; r < c.rows(); ++r) {
            auto row = c.row(r);
            const uint32_t target = column(rng);
            c.targets[r] = target;
            switch (r % 8) {
                case 0:
                    fill_confident_row(
                        row, target, kOffsets[r % 3], kGaps[(r / 8) % kGaps.size()], Competitors::OtherLane);
                    c.confident[r] = true;
                    break;
                case 1: {
                    // Tied maxima: the target is the later copy, so it is not the selected first maximum.
                    for (auto& v : row) v = logit(rng);
                    const size_t first = target / 2;
                    if (first != target) {
                        row[first] = 12.F;
                    }
                    row[target] = 12.F;
                    break;
                }
                case 2:
                    for (auto& v : row) v = logit(rng);
                    c.targets[r] = r % 16 == 2 ? static_cast<uint32_t>(width) : kNoTarget;
                    break;
                default:
                    for (auto& v : row) v = logit(rng);
            }
        }
        const auto d = upload(c);
        check_loss(c, d, run_loss(d), shape_name(shape) + " forward");
        const float scaler = 0.37F;
        check_grad(c, d, run_grad(d, scaler), scaler, shape_name(shape) + " backward");
    }
}

// The corrected fused log_softmax as an on-device oracle: CE(x, t) = -log_softmax(x)[t], and the CE
// gradient equals log_softmax backward with upstream -s * one_hot(t) (FP32 upstream, so it is not rounded).
TEST_F(CrossEntropyCollapseTest, MatchesCorrectedLogSoftmax) {
    auto* device = &autograd::ctx().get_device();
    for (const auto& shape : {ttnn::Shape({1, 1, 32, 32000}), ttnn::Shape({2, 1, 35, 1000})}) {
        SCOPED_TRACE(shape_name(shape));
        const auto c = confident_case(shape);
        const auto d = upload(c);
        const float scaler = 1.F / static_cast<float>(c.rows());
        const auto log_softmax = metal::log_softmax_fw(d.input);
        ASSERT_EQ(log_softmax.size(), 2U);
        const auto y = core::to_vector(log_softmax[0]);
        std::vector<float> upstream(c.logits.size(), 0.F);
        for (size_t r = 0; r < c.rows(); ++r) {
            upstream[r * c.width() + c.targets[r]] = -scaler;
        }
        const auto dy = core::from_vector<float, ttnn::DataType::FLOAT32>(upstream, c.shape, device);
        const auto oracle_dx = core::to_vector(metal::log_softmax_bw(d.input, dy, log_softmax[1]));

        const auto loss = run_loss(d);
        const auto dx = run_grad(d, scaler);
        MismatchLog mismatches(shape_name(shape) + " CE vs corrected log_softmax");
        for (size_t r = 0; r < c.rows(); ++r) {
            const double expected = -static_cast<double>(y[r * c.width() + c.targets[r]]);
            mismatches.near("loss", r, c.targets[r], loss[r], expected, kBf16Rtol * std::abs(expected));
            for (size_t j = 0; j < c.width(); ++j) {
                const size_t i = r * c.width() + j;
                mismatches.near(
                    "dx", r, j, dx[i], oracle_dx[i], kBf16Rtol * std::abs(static_cast<double>(oracle_dx[i])) + 1e-30);
            }
        }
    }
}

// The training path: ops::cross_entropy_loss with MEAN reduction (fused forward, moreh_mean, fused backward
// times the upstream gradient) on a batch where every target is confidently predicted.
TEST_F(CrossEntropyCollapseTest, MeanLossAutogradKeepsConfidentSignal) {
    const auto shape = ttnn::Shape({2, 1, 32, 1024});
    Case c(shape);
    for (size_t r = 0; r < c.rows(); ++r) {
        const size_t target = (r * 37 + 3) % c.width();
        fill_confident_row(c.row(r), target, 0.F, 17.F + static_cast<float>(r % 8), Competitors::All);
        c.targets[r] = static_cast<uint32_t>(target);
        c.confident[r] = true;
    }
    const auto d = upload(c);
    auto prediction = autograd::create_tensor(d.input, /* requires_grad */ true);
    auto target = autograd::create_tensor(d.target, /* requires_grad */ false);
    auto loss = ops::cross_entropy_loss(prediction, target, ops::ReduceType::MEAN);
    loss->backward();

    double expected_loss = 0.0;
    for (size_t r = 0; r < c.rows(); ++r) {
        const auto row = std::span<const float>(d.x).subspan(r * c.width(), c.width());
        expected_loss += reference_loss(row, d.refs[r], c.targets[r]);
    }
    expected_loss /= static_cast<double>(c.rows());
    const auto value = core::to_vector(loss->get_value());
    ASSERT_EQ(value.size(), 1U);
    EXPECT_GT(value[0], 0.F) << "mean loss collapsed to zero";
    EXPECT_NEAR(value[0], expected_loss, kBf16Rtol * expected_loss);

    const float scaler = 1.F / static_cast<float>(c.rows());
    check_grad(c, d, core::to_vector(prediction->get_grad()), scaler, "autograd MEAN backward");
}

// Masked logits (-inf) covering whole cores' columns of a row (one tile row, so its columns are split across
// cores), as with a vocabulary mask: the empty partitions contribute nothing (no NaN from -inf - (-inf)) and the
// masked columns get a zero gradient. Finite columns: one tile at either end or in the middle, or one half.
TEST_F(CrossEntropyCollapseTest, MaskedPartitionsContributeNothing) {
    const auto shape = ttnn::Shape({1, 1, 32, 32000});
    Case c(shape);
    const size_t width = c.width();
    const std::array<std::pair<size_t, size_t>, 5> windows{
        {{width - 32, width}, {0, 32}, {width / 2, width / 2 + 32}, {width / 2, width}, {0, width / 2}}};
    std::mt19937 rng(91U);
    std::uniform_real_distribution<float> logit(-10.F, 10.F);
    for (size_t r = 0; r < c.rows(); ++r) {
        const auto [begin, end] = windows[r % windows.size()];
        const size_t target = begin + (r * 7) % (end - begin);
        auto row = c.row(r);
        std::fill(row.begin(), row.end(), -std::numeric_limits<float>::infinity());
        auto finite = row.subspan(begin, end - begin);
        if (r % 2 == 0) {
            fill_confident_row(
                finite, target - begin, kOffsets[r % 3], kGaps[(r / 2) % kGaps.size()], Competitors::All);
            c.confident[r] = true;
        } else {
            for (auto& v : finite) v = logit(rng);
        }
        c.targets[r] = static_cast<uint32_t>(target);
    }
    for (const auto dtype : {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32}) {
        const std::string label = dtype == ttnn::DataType::FLOAT32 ? "fp32" : "bf16";
        SCOPED_TRACE(label);
        const auto d = upload(c, dtype);
        check_loss(c, d, run_loss(d), label + " masked forward");
        const float scaler = 1.F / static_cast<float>(c.rows());
        check_grad(c, d, run_grad(d, scaler), scaler, label + " masked backward");
    }
}

// The program writes its output tiles in the input's format, so a preallocated output must have the spec of the
// output the op would allocate (an FP32 loss into a BF16 buffer would write past its end). Outputs in the input's
// dtype, BF16 or FP32, are written in place; another dtype or shape is rejected.
TEST_F(CrossEntropyCollapseTest, PreallocatedOutputMatchesInput) {
    auto* device = &autograd::ctx().get_device();
    const auto shape = ttnn::Shape({2, 1, 35, 1000});
    const auto loss_shape = ttnn::Shape({shape[0], 1, shape[2], 1});
    const auto c = confident_case(shape);
    const float scaler = 1.F / static_cast<float>(c.rows());
    // Starts at -1, which no loss or gradient here can be.
    const auto filled = [&](const ttnn::Shape& s, ttnn::DataType dtype) {
        const std::vector<float> values(s.volume(), -1.F);
        return dtype == ttnn::DataType::FLOAT32 ? core::from_vector<float, ttnn::DataType::FLOAT32>(values, s, device)
                                                : core::from_vector(values, s, device);
    };
    for (const auto dtype : {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32}) {
        const std::string label = dtype == ttnn::DataType::FLOAT32 ? "fp32" : "bf16";
        SCOPED_TRACE(label);
        const auto other = dtype == ttnn::DataType::FLOAT32 ? ttnn::DataType::BFLOAT16 : ttnn::DataType::FLOAT32;
        const auto d = upload(c, dtype);

        const auto loss = filled(loss_shape, dtype);
        const auto loss_result = ttnn::prim::ttml_cross_entropy_fw(d.input, d.target, loss);
        EXPECT_EQ(loss_result.buffer()->address(), loss.buffer()->address());
        check_loss(c, d, core::to_vector(loss), label + " forward into a preallocated output");
        const auto dx = filled(shape, dtype);
        const auto dx_result = ttnn::prim::ttml_cross_entropy_bw(d.input, d.target, scaler, dx, std::nullopt);
        EXPECT_EQ(dx_result.buffer()->address(), dx.buffer()->address());
        check_grad(c, d, core::to_vector(dx), scaler, label + " backward into a preallocated output");

        EXPECT_ANY_THROW(ttnn::prim::ttml_cross_entropy_fw(d.input, d.target, filled(loss_shape, other)));
        EXPECT_ANY_THROW(
            ttnn::prim::ttml_cross_entropy_bw(d.input, d.target, scaler, filled(shape, other), std::nullopt));
        EXPECT_ANY_THROW(
            ttnn::prim::ttml_cross_entropy_fw(d.input, d.target, filled(ttnn::Shape({shape[0], 1, 34, 1}), dtype)));
        EXPECT_ANY_THROW(ttnn::prim::ttml_cross_entropy_bw(
            d.input, d.target, scaler, filled(ttnn::Shape({shape[0], 1, shape[2], 999}), dtype), std::nullopt));
    }
}

// Logits padded past whole tiles: logical [1, 1, 32, 32] padded to [1, 1, 64, 32] would run a second tile row the
// one-tile-row loss and gradient outputs do not hold, and [1, 1, 32, 64] would read a padding tile as logits. Both
// must be refused before any kernel runs.
TEST_F(CrossEntropyCollapseTest, RejectsPaddingBeyondTiles) {
    auto* device = &autograd::ctx().get_device();
    const ttnn::Shape logical({1, 1, 32, 32});
    const auto target = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        std::vector<uint32_t>(32, 0U), ttnn::Shape({1, 32}), device, ttnn::Layout::ROW_MAJOR);
    for (const auto& padded : {ttnn::Shape({1, 1, 64, 32}), ttnn::Shape({1, 1, 32, 64})}) {
        SCOPED_TRACE(padded[-2] == 64 ? "padded rows" : "padded columns");
        const tt::tt_metal::TensorSpec spec(
            logical,
            tt::tt_metal::experimental::tensor_layout_from_padded_shape(
                ttnn::DataType::BFLOAT16,
                tt::tt_metal::PageConfig(ttnn::Layout::TILE),
                ttnn::DRAM_MEMORY_CONFIG,
                logical,
                padded));
        const auto input = ttnn::create_device_tensor(spec, device);
        ASSERT_EQ(input.padded_shape(), padded);
        EXPECT_ANY_THROW((void)ttnn::prim::ttml_cross_entropy_fw(input, target));
        EXPECT_ANY_THROW((void)ttnn::prim::ttml_cross_entropy_bw(input, target, 1.F, std::nullopt, std::nullopt));
    }
}

}  // namespace ttml::ops::tests
