// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation-agnostic coverage for the collapse-resistant log_softmax
// (https://arxiv.org/abs/2610.31642, Section 2). For each row, k is the FIRST
// maximum, m = x[k], t = sum_{j != k} exp(x[j] - m), l = log1p(t),
// y[j] = (x[j] - m) - l. Backward: S = sum(dy), dx[j] = dy[j] - exp(y[j]) * S
// for j != k and dx[k] = -sum_{j != k} dx[j].
//
// Every check compares against a float64 oracle evaluated on the quantized
// logits and upstream gradients actually seen by the device. The only fused
// internals relied on are the row-state contract: FP32 TILE tensor of shape
// [1, 1, physical_rows, 96] whose columns 0, 32 and 64 hold the maximum, the
// first-maximum index and log1p(tail) of each physical row.

#include "metal/ops/log_softmax/log_softmax.hpp"

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
#include <string>
#include <string_view>
#include <tt-metalium/experimental/tensor_layout_apis_with_custom_alignment.hpp>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/log_softmax/device/log_softmax_device_operation.hpp"
#include "ops/unary_ops.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops::tests {

namespace {

constexpr double kFp32Rtol = 3e-5;
constexpr double kBf16Rtol = 1e-2;
constexpr double kGradAtol = 1e-7;
constexpr double kFp32UnitRoundoff = 1.0 / 16777216.0;  // 2^-24
constexpr size_t kTile = 32;
constexpr uint32_t kStateWidth = 96;
constexpr size_t kStateMax = 0;
constexpr size_t kStateIndex = 32;
constexpr size_t kStateLogNormalizer = 64;
constexpr size_t kMaxReportedMismatches = 8;
constexpr std::array<float, 3> kRowOffsets{0.F, -32.F, 32.F};

double rtol_for(ttnn::DataType dtype) {
    return dtype == ttnn::DataType::FLOAT32 ? kFp32Rtol : kBf16Rtol;
}

std::string dtype_name(ttnn::DataType dtype) {
    return dtype == ttnn::DataType::FLOAT32 ? "fp32" : "bf16";
}

std::string shape_name(const ttnn::Shape& shape) {
    std::string name = "[";
    for (size_t d = 0; d < shape.rank(); ++d) {
        name += (d == 0 ? "" : ",") + std::to_string(shape[static_cast<int>(d)]);
    }
    return name + "]";
}

ttnn::Tensor make_device_tensor(const std::vector<float>& data, const ttnn::Shape& shape, ttnn::DataType dtype) {
    auto* device = &autograd::ctx().get_device();
    return dtype == ttnn::DataType::FLOAT32 ? core::from_vector<float, ttnn::DataType::FLOAT32>(data, shape, device)
                                            : core::from_vector(data, shape, device);
}

// Rows are the last axis. State rows follow the tile-padded physical rows of the input.
struct RowLayout {
    size_t width = 1;
    size_t rows = 0;
    size_t height = 1;
    size_t padded_height = kTile;

    explicit RowLayout(const ttnn::Shape& shape) :
        width(shape[-1]),
        rows(static_cast<size_t>(shape.volume()) / shape[-1]),
        height(shape[-2]),
        padded_height((static_cast<size_t>(shape[-2]) + kTile - 1) / kTile * kTile) {
    }

    [[nodiscard]] size_t physical_row(size_t row) const {
        return row / height * padded_height + row % height;
    }

    [[nodiscard]] size_t physical_rows() const {
        return rows / height * padded_height;
    }
};

struct RowReference {
    size_t pivot = 0;
    double max = 0.0;
    double log_normalizer = 0.0;
};

std::vector<RowReference> reference_rows(std::span<const float> x, size_t width) {
    std::vector<RowReference> refs(x.size() / width);
    for (size_t r = 0; r < refs.size(); ++r) {
        const auto row = x.subspan(r * width, width);
        size_t pivot = 0;
        for (size_t k = 1; k < width; ++k) {
            if (row[k] > row[pivot]) {  // Strict: the first of tied maxima is selected.
                pivot = k;
            }
        }
        const double maximum = row[pivot];
        double tail = 0.0;
        for (size_t k = 0; k < width; ++k) {
            if (k != pivot) {
                tail += std::exp(static_cast<double>(row[k]) - maximum);
            }
        }
        refs[r] = RowReference{pivot, maximum, std::log1p(tail)};
    }
    return refs;
}

// Normal: the existing tolerances (atol 1e-7 on gradients).
// Confident: the existing collapse-case tolerances (atol 0) and strictly negative pivot output/gradient.
// Underflow: exp(x - m) of the tail may flush to zero; values may lie anywhere between exact and fully flushed.
enum class RowKind : uint8_t { Normal, Confident, Underflow };

RowKind row_kind(std::span<const RowKind> kinds, size_t row) {
    return kinds.empty() ? RowKind::Normal : kinds[row];
}

// dx[k] is reconstructed as -sum(dx[j], j != k). When those terms cancel, FP32 accumulation can lose
// up to half an ulp per added term, possibly with a consistent bias (e.g. a constant tail added to -1).
// Any tile implementation accumulates at most about width / 32 + 32 terms sequentially, so allow that
// many unit roundoffs of the cancelled mass sum(|dx[j]|) - |sum(dx[j])|. The allowance is zero for
// same-sign sums, so the confident (collapse) cases keep the existing relative tolerances.
double cancellation_allowance(size_t width, double abs_sum, double sum) {
    const double sequential_terms = static_cast<double>(width / kTile + kTile);
    return sequential_terms * kFp32UnitRoundoff * (abs_sum - std::abs(sum));
}

// Worst-case relative error of an FP32 sum accumulated sequentially in tile lanes: about width / 32
// terms per lane plus the 32-lane combine, each rounding by up to one unit roundoff. Applied relative to
// the summed magnitudes (the tail behind log1p, S = sum(dy), the other gradients), so a collapse, which
// loses the whole quantity, is still caught.
double accumulation_allowance(size_t width) {
    return static_cast<double>(width / kTile + kTile) * kFp32UnitRoundoff;
}

double underflow_floor(size_t width) {
    return 2.0 * static_cast<double>(width) * static_cast<double>(std::numeric_limits<float>::min());
}

// Counts mismatches and reports only the first few, so a broken wide tensor fails quickly and readably.
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
        // NaN compares false and is reported. An infinite expectation (a masked logit's output) must match exactly.
        if (std::isinf(expected) ? actual == expected : std::abs(actual - expected) <= tolerance) {
            return;
        }
        if (failures_ < kMaxReportedMismatches) {
            ADD_FAILURE() << std::setprecision(10) << label_ << ": " << what << " at row " << row << ", column " << col
                          << ": actual " << actual << ", expected " << expected << ", tolerance " << tolerance;
        }
        ++failures_;
    }

    void expect(bool condition, std::string_view what, size_t row, size_t col, double actual) {
        if (condition) {
            return;
        }
        if (failures_ < kMaxReportedMismatches) {
            ADD_FAILURE() << std::setprecision(10) << label_ << ": expected " << what << " at row " << row
                          << ", column " << col << ", actual " << actual;
        }
        ++failures_;
    }

private:
    std::string label_;
    size_t failures_ = 0;
};

void check_forward(
    const RowLayout& layout,
    std::span<const RowReference> refs,
    std::span<const float> x,
    std::span<const float> y,
    std::span<const float> state,
    ttnn::DataType dtype,
    std::span<const RowKind> kinds,
    MismatchLog& mismatches) {
    ASSERT_EQ(y.size(), x.size());
    const double rtol = rtol_for(dtype);
    for (size_t r = 0; r < layout.rows; ++r) {
        const auto& ref = refs[r];
        const RowKind kind = row_kind(kinds, r);
        // In underflow rows the entire tail may legitimately vanish.
        const double slack = (kind == RowKind::Underflow ? ref.log_normalizer + underflow_floor(layout.width) : 0.0) +
                             ref.log_normalizer * accumulation_allowance(layout.width);
        const size_t base = r * layout.width;
        for (size_t k = 0; k < layout.width; ++k) {
            const double expected = static_cast<double>(x[base + k]) - ref.max - ref.log_normalizer;
            mismatches.near("y", r, k, y[base + k], expected, std::abs(expected) * rtol + slack);
        }
        const double y_pivot = y[base + ref.pivot];
        if (kind == RowKind::Confident) {
            mismatches.expect(y_pivot < 0.0, "y[first max] < 0 (confident row)", r, ref.pivot, y_pivot);
        } else {
            mismatches.expect(y_pivot <= 0.0, "y[first max] <= 0", r, ref.pivot, y_pivot);
        }
        if (!state.empty()) {
            const size_t s = layout.physical_row(r) * kStateWidth;
            mismatches.near("state max", r, ref.pivot, state[s + kStateMax], ref.max, 0.0);
            mismatches.near(
                "state first-max index", r, ref.pivot, state[s + kStateIndex], static_cast<double>(ref.pivot), 0.0);
            mismatches.near(
                "state log1p(tail)",
                r,
                ref.pivot,
                state[s + kStateLogNormalizer],
                ref.log_normalizer,
                ref.log_normalizer * kFp32Rtol + slack);
        }
    }
}

void check_backward(
    const RowLayout& layout,
    std::span<const RowReference> refs,
    std::span<const float> x,
    std::span<const float> dy,
    std::span<const float> dx,
    ttnn::DataType dtype,
    std::span<const RowKind> kinds,
    bool pivot_one_hot,
    MismatchLog& mismatches) {
    ASSERT_EQ(dy.size(), x.size());
    ASSERT_EQ(dx.size(), x.size());
    const double rtol = rtol_for(dtype);
    for (size_t r = 0; r < layout.rows; ++r) {
        const auto& ref = refs[r];
        const RowKind kind = row_kind(kinds, r);
        const size_t base = r * layout.width;
        double sum_dy = 0.0;
        double abs_dy = 0.0;
        for (size_t k = 0; k < layout.width; ++k) {
            sum_dy += dy[base + k];
            abs_dy += std::abs(dy[base + k]);
        }
        const double accumulation = accumulation_allowance(layout.width);
        const double atol = kind == RowKind::Confident ? 0.0 : kGradAtol;
        const bool underflow = kind == RowKind::Underflow;
        const double flush_floor = underflow ? underflow_floor(layout.width) * std::max(1.0, std::abs(sum_dy)) : 0.0;
        double sum_others = 0.0;
        double abs_others = 0.0;
        double flushable = 0.0;
        for (size_t k = 0; k < layout.width; ++k) {
            if (k == ref.pivot) {
                continue;
            }
            const double probability = std::exp(static_cast<double>(x[base + k]) - ref.max - ref.log_normalizer);
            const double expected = dy[base + k] - probability * sum_dy;
            const double slack = underflow ? probability * std::abs(sum_dy) + flush_floor : 0.0;
            // Relative to both terms of dy - p * S (their difference may cancel), plus the error of S itself.
            // With a one-hot upstream at the pivot dy is zero here, so collapse rows keep rtol * |dx|.
            const double scale = std::abs(static_cast<double>(dy[base + k])) + probability * std::abs(sum_dy);
            mismatches.near(
                "dx", r, k, dx[base + k], expected, scale * rtol + probability * abs_dy * accumulation + atol + slack);
            sum_others += expected;
            abs_others += std::abs(expected);
            flushable += probability * std::abs(sum_dy);
        }
        double actual_sum = 0.0;
        double actual_abs = 0.0;
        for (size_t k = 0; k < layout.width; ++k) {
            actual_sum += dx[base + k];
            actual_abs += std::abs(dx[base + k]);
        }
        const double cancelled =
            cancellation_allowance(layout.width, abs_others, sum_others) + abs_others * accumulation;
        const double pivot_slack = underflow ? flushable + flush_floor : 0.0;
        const double dx_pivot = dx[base + ref.pivot];
        mismatches.near(
            "dx[first max] = -sum(other dx)",
            r,
            ref.pivot,
            dx_pivot,
            -sum_others,
            std::abs(sum_others) * rtol + cancelled + atol + pivot_slack);
        mismatches.near("sum(dx)", r, ref.pivot, actual_sum, 0.0, actual_abs * rtol + cancelled + atol + pivot_slack);
        if (kind == RowKind::Confident && pivot_one_hot) {
            mismatches.expect(dx_pivot < 0.0, "dx[first max] < 0 (confident row)", r, ref.pivot, dx_pivot);
        }
    }
}

enum class Upstream : uint8_t { PivotOneHot, OtherOneHot, CrossEntropyGrad, Random, Zero };

std::string upstream_name(Upstream kind) {
    switch (kind) {
        case Upstream::PivotOneHot: return "one-hot at first max";
        case Upstream::OtherOneHot: return "one-hot off first max";
        case Upstream::CrossEntropyGrad: return "softmax - one-hot";
        case Upstream::Random: return "random";
        case Upstream::Zero: return "zero";
    }
    return "unknown";
}

size_t upstream_target(Upstream kind, std::span<const size_t> targets, size_t row, size_t pivot, size_t width) {
    if (!targets.empty()) {
        return targets[row];
    }
    const size_t other = (pivot + std::max<size_t>(1, width / 2)) % width;
    // Cross-entropy alternates between the correct class (the first max) and a wrong one.
    return kind == Upstream::CrossEntropyGrad && row % 2 == 0 ? pivot : other;
}

std::vector<float> make_upstream(
    Upstream kind,
    std::span<const float> x,
    size_t width,
    std::span<const RowReference> refs,
    std::span<const size_t> targets,
    uint32_t seed) {
    std::vector<float> dy(x.size(), 0.F);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-1.F, 1.F);
    for (size_t r = 0; r < refs.size(); ++r) {
        const size_t base = r * width;
        const auto& ref = refs[r];
        const size_t target = upstream_target(kind, targets, r, ref.pivot, width);
        switch (kind) {
            case Upstream::PivotOneHot: dy[base + ref.pivot] = -1.F; break;
            case Upstream::OtherOneHot: dy[base + target] = -1.F; break;
            case Upstream::CrossEntropyGrad:
                for (size_t k = 0; k < width; ++k) {
                    const double probability =
                        std::exp(static_cast<double>(x[base + k]) - ref.max - ref.log_normalizer);
                    dy[base + k] = static_cast<float>(probability - (k == target ? 1.0 : 0.0));
                }
                break;
            case Upstream::Random:
                for (size_t k = 0; k < width; ++k) {
                    dy[base + k] = uniform(rng);
                }
                break;
            case Upstream::Zero: break;
        }
    }
    return dy;
}

struct BackwardSpec {
    Upstream kind = Upstream::PivotOneHot;
    std::optional<ttnn::DataType> dtype;  // Upstream dtype; the input dtype when unset.
};

struct CaseOptions {
    std::vector<BackwardSpec> backward;
    std::vector<size_t> expected_pivots;  // Guards the test construction: the oracle must select these.
    std::vector<size_t> targets;          // Optional per-row class for OtherOneHot / CrossEntropyGrad.
    std::vector<RowKind> kinds;           // Optional per-row expectations; Normal when empty.
    uint32_t seed = 0;
};

struct FusedForward {
    ttnn::Tensor input;
    ttnn::Tensor output;
    ttnn::Tensor state;
    std::vector<float> x;
    std::vector<RowReference> refs;
};

FusedForward run_fused_forward(
    const ttnn::Shape& shape, ttnn::DataType dtype, const std::vector<float>& logits, const CaseOptions& options) {
    const RowLayout layout(shape);
    MismatchLog mismatches(shape_name(shape) + " " + dtype_name(dtype) + " forward");
    FusedForward result;
    result.input = make_device_tensor(logits, shape, dtype);
    // The oracle uses the quantized logits, not the pre-conversion host data.
    result.x = core::to_vector(result.input);
    result.refs = reference_rows(result.x, layout.width);
    for (size_t r = 0; r < options.expected_pivots.size(); ++r) {
        mismatches.near(
            "test construction: oracle first max",
            r,
            options.expected_pivots[r],
            static_cast<double>(result.refs[r].pivot),
            static_cast<double>(options.expected_pivots[r]),
            0.0);
    }
    auto values = metal::log_softmax_fw(result.input);
    EXPECT_EQ(values.size(), 2U);
    if (values.size() != 2U) {
        return result;
    }
    result.output = values[0];
    result.state = values[1];
    EXPECT_EQ(result.output.dtype(), dtype);
    EXPECT_EQ(result.output.logical_shape(), shape);
    EXPECT_EQ(result.state.dtype(), ttnn::DataType::FLOAT32);
    EXPECT_EQ(result.state.layout(), ttnn::Layout::TILE);
    EXPECT_EQ(
        result.state.logical_shape(),
        ttnn::Shape({1U, 1U, static_cast<uint32_t>(layout.physical_rows()), kStateWidth}));
    check_forward(
        layout,
        result.refs,
        result.x,
        core::to_vector(result.output),
        core::to_vector(result.state),
        dtype,
        options.kinds,
        mismatches);
    return result;
}

void run_fused_backward(
    const FusedForward& forward,
    const ttnn::Shape& shape,
    ttnn::DataType dtype,
    const BackwardSpec& spec,
    const CaseOptions& options) {
    const RowLayout layout(shape);
    const auto upstream_dtype = spec.dtype.value_or(dtype);
    MismatchLog mismatches(
        shape_name(shape) + " " + dtype_name(dtype) + " backward, dy " + upstream_name(spec.kind) + " " +
        dtype_name(upstream_dtype));
    const auto upstream = make_device_tensor(
        make_upstream(spec.kind, forward.x, layout.width, forward.refs, options.targets, options.seed + 1U),
        shape,
        upstream_dtype);
    const auto grad = metal::log_softmax_bw(forward.input, upstream, forward.state);
    EXPECT_EQ(grad.dtype(), dtype);
    EXPECT_EQ(grad.logical_shape(), shape);
    check_backward(
        layout,
        forward.refs,
        forward.x,
        core::to_vector(upstream),
        core::to_vector(grad),
        dtype,
        options.kinds,
        spec.kind == Upstream::PivotOneHot,
        mismatches);
}

FusedForward run_fused_case(
    const ttnn::Shape& shape, ttnn::DataType dtype, const std::vector<float>& logits, const CaseOptions& options) {
    auto forward = run_fused_forward(shape, dtype, logits, options);
    for (const auto& spec : options.backward) {
        run_fused_backward(forward, shape, dtype, spec, options);
    }
    return forward;
}

struct AutogradResult {
    std::vector<float> y;
    std::vector<float> dx;
};

AutogradResult run_autograd(const ttnn::Tensor& value, const ttnn::Tensor& upstream, int dim, bool composite) {
    auto input = autograd::create_tensor(value, /* requires_grad */ true);
    auto output = composite ? ops::log_softmax_composite(input, dim) : ops::log_softmax(input, dim);
    const auto& output_value = output->get_value(autograd::PreferredPrecision::NATIVE);
    EXPECT_EQ(output_value.dtype(), value.dtype());
    AutogradResult result;
    result.y = core::to_vector(output_value);
    output->set_grad(upstream);
    output->backward();
    EXPECT_EQ(input->get_grad().dtype(), value.dtype());
    result.dx = core::to_vector(input->get_grad());
    autograd::ctx().reset_graph();
    return result;
}

// Swaps `dim` of a rank-4 host tensor with the last axis, the same permutation the op applies.
std::pair<std::vector<float>, ttnn::Shape> swap_with_last_axis(
    std::span<const float> data, const ttnn::Shape& shape, int dim) {
    const std::array<size_t, 4> dims{shape[0], shape[1], shape[2], shape[3]};
    std::array<size_t, 4> strides{dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
    std::array<size_t, 4> out_dims = dims;
    std::swap(out_dims[dim], out_dims[3]);
    std::swap(strides[dim], strides[3]);
    std::vector<float> out;
    out.reserve(data.size());
    for (size_t a = 0; a < out_dims[0]; ++a) {
        for (size_t b = 0; b < out_dims[1]; ++b) {
            for (size_t c = 0; c < out_dims[2]; ++c) {
                for (size_t d = 0; d < out_dims[3]; ++d) {
                    out.push_back(data[a * strides[0] + b * strides[1] + c * strides[2] + d * strides[3]]);
                }
            }
        }
    }
    return {
        std::move(out),
        ttnn::Shape(
            {static_cast<uint32_t>(out_dims[0]),
             static_cast<uint32_t>(out_dims[1]),
             static_cast<uint32_t>(out_dims[2]),
             static_cast<uint32_t>(out_dims[3])})};
}

// Columns where a split across lanes, tiles or cores plausibly starts or ends, most important first.
std::vector<size_t> boundary_columns(size_t width) {
    const size_t tiles = (width + kTile - 1) / kTile;
    const size_t full_tiles_end = width / kTile * kTile;
    std::vector<size_t> candidates{
        0,
        width - 1,
        full_tiles_end - 1,  // Last column of the last full tile (before tile padding).
        full_tiles_end,      // First column of a partial tile.
        width / 2,
        width / 2 - 1,
        width / 3,
        width / 8,
        width * 7 / 8,
        width - 2,
        width - kTile,
        1,
        31,
        32,
        33};
    for (const size_t parts : {2U, 3U, 4U, 5U, 7U, 8U, 10U, 13U, 16U}) {
        for (size_t i = 1; i < parts; ++i) {
            const size_t ceil_split = i * ((tiles + parts - 1) / parts) * kTile;
            const size_t floor_split = i * (tiles / parts) * kTile;
            candidates.insert(
                candidates.end(), {ceil_split - 1, ceil_split, floor_split - 1, floor_split, i * width / parts});
        }
    }
    for (const size_t k : {1U, 2U, 3U, 5U, 8U, 13U, 16U, 21U, 31U, 64U, 100U, 125U}) {
        candidates.insert(candidates.end(), {k * 1024 - 1, k * 1024});
    }
    for (const size_t k : {1U, 3U, 7U, 11U, 61U}) {
        candidates.insert(candidates.end(), {k * 256 - 1, k * 256});
    }
    std::vector<size_t> columns;
    for (const size_t c : candidates) {  // Unsigned wrap-around from width < offset is filtered here.
        if (c < width && std::ranges::find(columns, c) == columns.end()) {
            columns.push_back(c);
        }
    }
    return columns;
}

// Sorted, de-duplicated sets of >= 2 columns holding the same maximum; the first entry is the expected pivot.
std::vector<std::vector<size_t>> tie_patterns(size_t width) {
    const size_t tiles = (width + kTile - 1) / kTile;
    const auto col = [](size_t tile, size_t lane) { return tile * kTile + lane; };
    const std::vector<std::vector<size_t>> raw{
        {col(1, 8), col(2, 1)},  // The later copy sits in an earlier lane of a later tile.
        {col(0, 31), col(1, 0)},
        {col(0, 5), col(tiles / 2, 5), col(tiles - 1, 5)},  // Same lane, far-apart tiles.
        {0, width - 1},
        {col(tiles / 8, 20), col(tiles * 7 / 8, 3)},  // Different would-be chunks, later copy in an earlier lane.
        {col(tiles / 2, 30), col(tiles - 1, 0)},
        {width / 3, width / 2, width - 1},
        {width - 2, width - 1},
        {col(tiles / 2, 0) - 1, col(tiles / 2, 0)},  // Straddles a half-way split.
        {col(tiles / 3, 17), col(2 * tiles / 3, 2), col(tiles - 1, 1)},
    };
    std::vector<std::vector<size_t>> patterns;
    for (const auto& pattern : raw) {
        std::vector<size_t> valid;
        for (const size_t c : pattern) {
            if (c < width) {
                valid.push_back(c);
            }
        }
        std::ranges::sort(valid);
        valid.erase(std::unique(valid.begin(), valid.end()), valid.end());
        if (valid.size() >= 2) {
            patterns.push_back(std::move(valid));
        }
    }
    return patterns;
}

// Fills a row with values strictly below `offset` (random when gap == 0, else offset - gap) and the
// unique maximum `offset` at `pivot`. All constant values used with this helper are BF16-exact.
void fill_row(std::span<float> row, size_t pivot, float offset, float gap, std::mt19937& rng) {
    std::uniform_real_distribution<float> noise(0.5F, 6.F);
    for (auto& value : row) {
        value = offset - (gap > 0.F ? gap : noise(rng));
    }
    row[pivot] = offset;
}

}  // namespace

class FusedLogSoftmaxTest : public ::testing::Test {
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

class FusedLogSoftmaxDtypeTest : public FusedLogSoftmaxTest, public ::testing::WithParamInterface<ttnn::DataType> {};

// Few rows, wide vocabularies (the shapes a width split targets), with the unique maximum placed at
// column 0, the last column, the last column before tile padding and plausible lane/tile/core chunk
// boundaries. Rows alternate between random tails and confident gaps across the BF16/FP32 cliffs.
TEST_P(FusedLogSoftmaxDtypeTest, WideVocabPivotPlacement) {
    const auto dtype = GetParam();
    constexpr std::array<float, 4> kGaps{0.F, 9.F, 18.F, 28.F};  // 0: random tail.
    for (const auto& shape :
         {ttnn::Shape({1, 1, 1, 32000}),
          ttnn::Shape({1, 1, 32, 32000}),
          ttnn::Shape({1, 1, 64, 32003}),
          ttnn::Shape({1, 1, 32, 128256})}) {
        const RowLayout layout(shape);
        const auto columns = boundary_columns(layout.width);
        const size_t passes = layout.rows == 1 ? 6 : (layout.width > 100000 ? 1 : 2);
        for (size_t pass = 0; pass < passes; ++pass) {
            SCOPED_TRACE(shape_name(shape) + " pass " + std::to_string(pass));
            std::mt19937 rng(1000U + static_cast<uint32_t>(pass));
            std::vector<float> logits(shape.volume());
            CaseOptions options;
            options.seed = 17U + static_cast<uint32_t>(pass);
            options.backward = {
                {Upstream::PivotOneHot, std::nullopt},
                {Upstream::CrossEntropyGrad, ttnn::DataType::FLOAT32},
                {Upstream::Random, std::nullopt}};
            for (size_t r = 0; r < layout.rows; ++r) {
                const size_t pivot = columns[(pass * layout.rows + r) % columns.size()];
                const float gap = kGaps[r % kGaps.size()];
                fill_row(
                    std::span(logits).subspan(r * layout.width, layout.width),
                    pivot,
                    kRowOffsets[r % kRowOffsets.size()],
                    gap,
                    rng);
                options.expected_pivots.push_back(pivot);
                options.kinds.push_back(gap > 0.F ? RowKind::Confident : RowKind::Normal);
            }
            run_fused_case(shape, dtype, logits, options);
        }
    }
}

// Tied maxima far apart: across tiles, lanes and would-be core chunks, including a later copy in an
// earlier lane. The state index must be the first occurrence, and backward must put the reconstructed
// gradient only there (upstream one-hot at the first copy, at the later copy, and CE toward the later copy).
TEST_P(FusedLogSoftmaxDtypeTest, TiedMaximaSelectFirstOccurrence) {
    const auto dtype = GetParam();
    for (const auto& shape :
         {ttnn::Shape({1, 1, 1, 32000}), ttnn::Shape({1, 1, 32, 32000}), ttnn::Shape({1, 1, 33, 1025})}) {
        const RowLayout layout(shape);
        const auto patterns = tie_patterns(layout.width);
        const size_t passes = layout.rows == 1 ? patterns.size() : 1;
        for (size_t pass = 0; pass < passes; ++pass) {
            SCOPED_TRACE(shape_name(shape) + " pass " + std::to_string(pass));
            std::mt19937 rng(2000U + static_cast<uint32_t>(pass));
            std::vector<float> logits(shape.volume());
            CaseOptions options;
            options.seed = 29U + static_cast<uint32_t>(pass);
            options.backward = {
                {Upstream::PivotOneHot, std::nullopt},
                {Upstream::OtherOneHot, std::nullopt},
                {Upstream::CrossEntropyGrad, ttnn::DataType::FLOAT32}};
            for (size_t r = 0; r < layout.rows; ++r) {
                const auto& pattern = patterns[(pass + r) % patterns.size()];
                const float offset = kRowOffsets[r % kRowOffsets.size()];
                auto row = std::span(logits).subspan(r * layout.width, layout.width);
                fill_row(row, pattern.front(), offset, r % 2 == 0 ? 0.F : 9.F, rng);
                for (const size_t c : pattern) {
                    row[c] = offset;
                }
                options.expected_pivots.push_back(pattern.front());
                options.targets.push_back(pattern[1]);  // Upstream classes land on the first later copy.
            }
            run_fused_case(shape, dtype, logits, options);
        }
    }
}

// Online-softmax stress: the running maximum rises at nearly every tile, falls (max at column 0),
// alternates between lanes, rows are constant, offsets are huge (+-1e4 FP32, +-9984 BF16-exact),
// magnitudes are mixed (finite huge negatives before and after the maximum) and the maximum jumps late.
TEST_P(FusedLogSoftmaxDtypeTest, OnlineRescalingStress) {
    const auto dtype = GetParam();
    const bool fp32 = dtype == ttnn::DataType::FLOAT32;
    const float big = fp32 ? 1e4F : 9984.F;  // 9984 = 156 * 64 is exact in BF16.
    constexpr size_t kStyles = 11;
    for (const auto& shape : {ttnn::Shape({1, 1, 33, 1025}), ttnn::Shape({1, 1, 32, 32000})}) {
        SCOPED_TRACE(shape_name(shape));
        const RowLayout layout(shape);
        const size_t width = layout.width;
        const size_t tiles = (width + kTile - 1) / kTile;
        const double last = static_cast<double>(width - 1);
        std::mt19937 rng(3000U);
        std::uniform_real_distribution<float> noise(-1.F, 1.F);
        std::vector<float> logits(shape.volume());
        CaseOptions options;
        options.seed = 41U;
        options.backward = {
            {Upstream::PivotOneHot, std::nullopt},
            {Upstream::CrossEntropyGrad, std::nullopt},
            {Upstream::Zero, std::nullopt}};
        for (size_t r = 0; r < layout.rows; ++r) {
            auto row = std::span(logits).subspan(r * width, width);
            RowKind kind = RowKind::Normal;
            switch (r % kStyles) {
                case 0:  // Monotonically increasing: the running maximum changes at every step.
                    for (size_t k = 0; k < width; ++k) {
                        row[k] = static_cast<float>(-16.0 + 16.0 * static_cast<double>(k) / last);
                    }
                    break;
                case 1:  // Monotonically decreasing: the maximum is column 0.
                    for (size_t k = 0; k < width; ++k) {
                        row[k] = static_cast<float>(-16.0 * static_cast<double>(k) / last);
                    }
                    break;
                case 2:  // Staircase rising per tile, falling across lanes inside a tile.
                    for (size_t k = 0; k < width; ++k) {
                        const double tile = static_cast<double>(k / kTile);
                        row[k] = static_cast<float>(
                            -16.0 * (static_cast<double>(tiles) - 1.0 - tile) / static_cast<double>(tiles) -
                            static_cast<double>(k % kTile) / 64.0);
                    }
                    break;
                case 3:  // Alternating: even lanes rise, odd lanes fall.
                    for (size_t k = 0; k < width; ++k) {
                        const double ramp = 4.0 * static_cast<double>(k) / static_cast<double>(width);
                        row[k] = static_cast<float>(k % 2 == 0 ? ramp : -ramp);
                    }
                    break;
                case 4: {  // Constant row: every column ties, the first is selected, l = log(width).
                    const std::array<float, 4> values{-3.F, 5.5F, -big, 0.F};
                    std::ranges::fill(row, values[(r / kStyles) % values.size()]);
                    break;
                }
                case 5:    // Huge positive offset with a confident gap.
                case 6: {  // Huge negative offset with a confident gap.
                    const float offset = r % kStyles == 5 ? big : -big;
                    const std::array<float, 3> fp32_gaps{8.F, 18.F, 24.F};
                    const float gap = fp32 ? fp32_gaps[(r / kStyles) % fp32_gaps.size()] : 64.F;
                    std::ranges::fill(row, offset - gap);
                    row[(r * 977) % width] = offset;
                    kind = RowKind::Confident;
                    break;
                }
                case 7: {  // Huge negatives first, then O(1) noise and a late maximum.
                    for (size_t k = 0; k < width; ++k) {
                        row[k] = (k < width / 2 || k % 7 == 0) ? -big : noise(rng);
                    }
                    row[width / 2 + (r * 131) % (width - width / 2)] = 2.F;
                    break;
                }
                case 8: {  // Early maximum, then O(1) noise, then huge negatives.
                    for (size_t k = 0; k < width; ++k) {
                        row[k] = k > 2 * width / 3 ? -big : noise(rng);
                    }
                    row[3] = 2.F;
                    break;
                }
                case 9:  // Late jump: the running maximum moves at the very end.
                    std::ranges::fill(row, -40.F);
                    row[0] = -1.F;
                    row[width - 1] = 0.F;
                    break;
                case 10:  // Sawtooth: repeated near-ties that quantization may turn into exact ties.
                    for (size_t k = 0; k < width; ++k) {
                        row[k] = static_cast<float>(-8.0 + 8.0 * static_cast<double>(k % 97) / 97.0);
                    }
                    break;
                default: break;
            }
            options.kinds.push_back(kind);
        }
        run_fused_case(shape, dtype, logits, options);
    }
}

// Masked logits (-inf) covering whole partitions of a row: a core's columns of the row, as with a padding or
// vocabulary mask on a width split. An empty partition must contribute nothing to the row statistics (its
// maximum minus the row's is -inf - (-inf) when taken naively, which is NaN). Masked outputs are exactly -inf
// and their gradients dy.
TEST_P(FusedLogSoftmaxDtypeTest, MaskedPartitionsContributeNothing) {
    const auto dtype = GetParam();
    constexpr float kMasked = -std::numeric_limits<float>::infinity();
    constexpr size_t kPatterns = 6;
    // Whether column `col` of a row with masking pattern `pattern` holds a finite logit; each pattern leaves whole
    // chunks of any width split masked.
    const auto finite = [](size_t pattern, size_t col, size_t width) {
        switch (pattern) {
            case 0: return col >= width - kTile;                         // Only the last tile.
            case 1: return col < kTile;                                  // Only the first tile.
            case 2: return col >= width / 2 && col < width / 2 + kTile;  // One tile in the middle.
            case 3: return col >= width / 2;                             // The second half.
            case 4: return col < width / 2;                              // The first half.
            default: return (col / 1024) % 2 == 0;                       // Alternating 1024-column blocks.
        }
    };
    for (const auto& shape : {ttnn::Shape({1, 1, 32, 32000}), ttnn::Shape({1, 1, 64, 8192})}) {
        SCOPED_TRACE(shape_name(shape));
        const RowLayout layout(shape);
        std::mt19937 rng(4000U);
        std::vector<float> logits(shape.volume());
        CaseOptions options;
        options.seed = 41U;
        options.backward = {
            {Upstream::PivotOneHot, std::nullopt},
            {Upstream::CrossEntropyGrad, ttnn::DataType::FLOAT32},
            {Upstream::Random, std::nullopt}};
        for (size_t r = 0; r < layout.rows; ++r) {
            const size_t pattern = r % kPatterns;
            size_t first = layout.width;
            size_t last = 0;
            for (size_t col = 0; col < layout.width; ++col) {
                if (finite(pattern, col, layout.width)) {
                    first = std::min(first, col);
                    last = col;
                }
            }
            const size_t middle = (first + last) / 2;
            const size_t pivot = finite(pattern, middle, layout.width) ? middle : last;
            const float gap = r % 2 == 0 ? 0.F : 9.F;
            auto row = std::span(logits).subspan(r * layout.width, layout.width);
            fill_row(row, pivot, kRowOffsets[r % kRowOffsets.size()], gap, rng);
            for (size_t col = 0; col < layout.width; ++col) {
                if (!finite(pattern, col, layout.width)) {
                    row[col] = kMasked;
                }
            }
            options.expected_pivots.push_back(pivot);
            options.kinds.push_back(gap > 0.F ? RowKind::Confident : RowKind::Normal);
        }
        run_fused_case(shape, dtype, logits, options);
    }
}

// Gaps around the BF16 (8-9, 17-18) and FP32 (24-28) collapse cliffs and up to FP32 exp underflow (88+).
// Gaps <= 60 must keep a strictly negative dominant log-probability and gradient; larger gaps may flush
// the tail but must stay finite and between the exact and the fully flushed result.
TEST_P(FusedLogSoftmaxDtypeTest, CollapseCliffGaps) {
    const auto dtype = GetParam();
    constexpr std::array<float, 16> kGaps{
        8.F, 9.F, 17.F, 18.F, 24.F, 27.F, 28.F, 40.F, 60.F, 80.F, 87.F, 88.F, 89.F, 100.F, 104.F, 150.F};
    for (const auto& shape :
         {ttnn::Shape({1, 1, 33, 2}),
          ttnn::Shape({1, 1, 33, 33}),
          ttnn::Shape({1, 1, 33, 1025}),
          ttnn::Shape({1, 1, 32, 32000})}) {
        SCOPED_TRACE(shape_name(shape));
        const RowLayout layout(shape);
        const size_t width = layout.width;
        std::mt19937 rng(4000U);
        std::vector<float> logits(shape.volume());
        CaseOptions options;
        options.seed = 53U;
        options.backward = {
            {Upstream::PivotOneHot, std::nullopt},
            {Upstream::PivotOneHot, ttnn::DataType::FLOAT32},
            {Upstream::OtherOneHot, std::nullopt}};
        for (size_t r = 0; r < layout.rows; ++r) {
            const float gap = kGaps[r % kGaps.size()];
            const float offset = (r / kGaps.size()) % 2 == 0 ? -32.F : 32.F;
            const std::array<size_t, 4> pivots{0, width - 1, width / 2, (r * 131 + 7) % width};
            const size_t pivot = pivots[r % pivots.size()];
            fill_row(std::span(logits).subspan(r * width, width), pivot, offset, gap, rng);
            options.expected_pivots.push_back(pivot);
            options.kinds.push_back(gap <= 60.F ? RowKind::Confident : RowKind::Underflow);
        }
        run_fused_case(shape, dtype, logits, options);
    }
}

// Widths 1, 2, 31, 32, 33, 1023, 1024, 1025 and row counts 1, 31, 33, plus N*C > 1 with H padding
// (state rows must follow the tile-padded physical rows).
TEST_P(FusedLogSoftmaxDtypeTest, WidthAndRowCountSweep) {
    const auto dtype = GetParam();
    std::vector<ttnn::Shape> shapes;
    for (const uint32_t width : {1U, 2U, 31U, 32U, 33U, 1023U, 1024U, 1025U}) {
        shapes.push_back(ttnn::Shape({1U, 1U, 33U, width}));
    }
    for (const uint32_t height : {1U, 31U}) {
        shapes.push_back(ttnn::Shape({1U, 1U, height, 33U}));
    }
    shapes.push_back(ttnn::Shape({2U, 3U, 37U, 40U}));
    for (const auto& shape : shapes) {
        SCOPED_TRACE(shape_name(shape));
        const RowLayout layout(shape);
        std::mt19937 rng(5000U);
        std::vector<float> logits(shape.volume());
        CaseOptions options;
        options.seed = 67U;
        options.backward = {
            {Upstream::Random, std::nullopt},
            {Upstream::CrossEntropyGrad, ttnn::DataType::FLOAT32},
            {Upstream::PivotOneHot, std::nullopt}};
        for (size_t r = 0; r < layout.rows; ++r) {
            const size_t pivot = (r * 7 + layout.width - 1) % layout.width;
            fill_row(
                std::span(logits).subspan(r * layout.width, layout.width),
                pivot,
                kRowOffsets[r % kRowOffsets.size()],
                0.F,
                rng);
            options.expected_pivots.push_back(pivot);
        }
        run_fused_case(shape, dtype, logits, options);
    }
}

// More physical tile rows than the compute grid has cores, with a count that the core count does not
// divide and a tile-padded last row group; some rows contain ties.
TEST_P(FusedLogSoftmaxDtypeTest, MoreTileRowsThanCores) {
    const auto dtype = GetParam();
    const auto grid = autograd::ctx().get_device().compute_with_storage_grid_size();
    const size_t cores = grid.x * grid.y;
    const size_t tile_rows = 2 * cores + 1;
    const ttnn::Shape shape({1U, 1U, static_cast<uint32_t>(tile_rows * kTile - 7), 33U});
    SCOPED_TRACE(shape_name(shape));
    const RowLayout layout(shape);
    std::mt19937 rng(6000U);
    std::vector<float> logits(shape.volume());
    CaseOptions options;
    options.seed = 79U;
    options.backward = {{Upstream::PivotOneHot, std::nullopt}, {Upstream::Random, std::nullopt}};
    for (size_t r = 0; r < layout.rows; ++r) {
        size_t pivot = (r * 5) % layout.width;
        const float offset = kRowOffsets[r % kRowOffsets.size()];
        auto row = std::span(logits).subspan(r * layout.width, layout.width);
        fill_row(row, pivot, offset, r % 3 == 0 ? 18.F : 0.F, rng);
        if (r % 11 == 0) {
            const size_t copy = (pivot + 16) % layout.width;
            row[copy] = offset;
            pivot = std::min(pivot, copy);
        }
        options.expected_pivots.push_back(pivot);
    }
    run_fused_case(shape, dtype, logits, options);
}

// Every upstream pattern (one-hot at / off the first maximum, the true CE gradient softmax - one-hot,
// random, zero), with native and FP32 upstream dtypes, plus the autograd entry point.
TEST_P(FusedLogSoftmaxDtypeTest, UpstreamPatterns) {
    const auto dtype = GetParam();
    const ttnn::Shape shape({1, 1, 33, 1025});
    const RowLayout layout(shape);
    std::mt19937 rng(7000U);
    std::vector<float> logits(shape.volume());
    CaseOptions options;
    options.seed = 83U;
    for (const auto kind :
         {Upstream::PivotOneHot, Upstream::OtherOneHot, Upstream::CrossEntropyGrad, Upstream::Random, Upstream::Zero}) {
        options.backward.push_back({kind, std::nullopt});
        if (dtype != ttnn::DataType::FLOAT32) {
            options.backward.push_back({kind, ttnn::DataType::FLOAT32});
        }
    }
    for (size_t r = 0; r < layout.rows; ++r) {
        const size_t pivot = (r * 97 + 13) % layout.width;
        const float gap = r % 2 == 0 ? 0.F : 18.F;
        fill_row(
            std::span(logits).subspan(r * layout.width, layout.width),
            pivot,
            kRowOffsets[r % kRowOffsets.size()],
            gap,
            rng);
        options.expected_pivots.push_back(pivot);
        options.kinds.push_back(gap > 0.F ? RowKind::Confident : RowKind::Normal);
    }
    const auto forward = run_fused_case(shape, dtype, logits, options);

    // ops::log_softmax with autograd must produce the same values as the oracle.
    const auto upstream = make_device_tensor(
        make_upstream(Upstream::CrossEntropyGrad, forward.x, layout.width, forward.refs, {}, 5U),
        shape,
        ttnn::DataType::FLOAT32);
    const auto result = run_autograd(forward.input, upstream, -1, /* composite */ false);
    MismatchLog mismatches("ops::log_softmax autograd " + dtype_name(dtype));
    check_forward(layout, forward.refs, forward.x, result.y, {}, dtype, options.kinds, mismatches);
    check_backward(
        layout,
        forward.refs,
        forward.x,
        core::to_vector(upstream),
        result.dx,
        dtype,
        options.kinds,
        /* pivot_one_hot */ false,
        mismatches);
}

// ops::log_softmax (fused route) and ops::log_softmax_composite agree with each other and with the oracle
// on random logits, including a non-last axis.
TEST_P(FusedLogSoftmaxDtypeTest, MatchesCompositeOnRandomLogits) {
    const auto dtype = GetParam();
    const double rtol = rtol_for(dtype);
    const std::vector<std::pair<ttnn::Shape, int>> cases{
        {ttnn::Shape({1, 1, 32, 1000}), -1}, {ttnn::Shape({2, 1, 35, 4099}), 3}, {ttnn::Shape({1, 2, 1030, 40}), 2}};
    for (const auto& test_case : cases) {
        const ttnn::Shape& shape = test_case.first;
        const int dim = test_case.second;
        SCOPED_TRACE(shape_name(shape) + " dim " + std::to_string(dim));
        std::mt19937 rng(8000U);
        // Moderate spread keeps every exp(y) * sum(dy) well below 1, so FP32 exp rounding cannot dominate a
        // nearly cancelled gradient element in either implementation.
        std::normal_distribution<float> normal(0.F, 1.F);
        std::uniform_real_distribution<float> uniform(-1.F, 1.F);
        std::vector<float> logits(shape.volume());
        std::vector<float> upstream_values(shape.volume());
        std::ranges::generate(logits, [&] { return normal(rng); });
        std::ranges::generate(upstream_values, [&] { return uniform(rng); });
        const auto input = make_device_tensor(logits, shape, dtype);
        const auto upstream = make_device_tensor(upstream_values, shape, dtype);
        const auto fused = run_autograd(input, upstream, dim, /* composite */ false);
        const auto composite = run_autograd(input, upstream, dim, /* composite */ true);

        const int axis = dim < 0 ? dim + 4 : dim;
        const auto to_last = [&](const std::vector<float>& data) {
            return axis == 3 ? data : swap_with_last_axis(data, shape, axis).first;
        };
        const ttnn::Shape last_shape = axis == 3 ? shape : swap_with_last_axis(logits, shape, axis).second;
        const RowLayout layout(last_shape);
        const auto x = to_last(core::to_vector(input));
        const auto dy = to_last(core::to_vector(upstream));
        const auto refs = reference_rows(x, layout.width);
        for (const auto& [name, result] :
             {std::pair<std::string, const AutogradResult*>{"fused", &fused},
              std::pair<std::string, const AutogradResult*>{"composite", &composite}}) {
            MismatchLog mismatches(name + " vs oracle " + dtype_name(dtype));
            check_forward(layout, refs, x, to_last(result->y), {}, dtype, {}, mismatches);
            check_backward(layout, refs, x, dy, to_last(result->dx), dtype, {}, false, mismatches);
        }
        // Direct agreement; the reconstructed pivot gradient is compared through the oracle above.
        MismatchLog mismatches("fused vs composite " + dtype_name(dtype));
        const auto fused_y = to_last(fused.y);
        const auto fused_dx = to_last(fused.dx);
        const auto composite_y = to_last(composite.y);
        const auto composite_dx = to_last(composite.dx);
        for (size_t r = 0; r < layout.rows; ++r) {
            for (size_t k = 0; k < layout.width; ++k) {
                if (k == refs[r].pivot) {
                    continue;
                }
                const size_t i = r * layout.width + k;
                const double y_scale = std::max(std::abs(fused_y[i]), std::abs(composite_y[i]));
                const double dx_scale = std::max(std::abs(fused_dx[i]), std::abs(composite_dx[i]));
                mismatches.near("y", r, k, fused_y[i], composite_y[i], y_scale * rtol);
                mismatches.near("dx", r, k, fused_dx[i], composite_dx[i], dx_scale * rtol + kGradAtol);
            }
        }
    }
}

// Inputs the fused path does not support (rank 3) still take the corrected composite path.
TEST_P(FusedLogSoftmaxDtypeTest, CompositeFallbackForRank3) {
    const auto dtype = GetParam();
    const ttnn::Shape shape({2, 7, 45});
    const RowLayout layout(shape);
    std::mt19937 rng(9000U);
    std::uniform_real_distribution<float> uniform(-1.F, 1.F);
    std::vector<float> logits(shape.volume());
    std::vector<float> upstream_values(shape.volume());
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = i % 5 == 0 ? 2.F : -static_cast<float>(i % 3);  // Tied maxima in every row.
        upstream_values[i] = uniform(rng);
    }
    const auto input = make_device_tensor(logits, shape, dtype);
    const auto upstream = make_device_tensor(upstream_values, shape, dtype);
    const auto result = run_autograd(input, upstream, -1, /* composite */ false);
    const auto x = core::to_vector(input);
    const auto refs = reference_rows(x, layout.width);
    MismatchLog mismatches("rank-3 fallback " + dtype_name(dtype));
    check_forward(layout, refs, x, result.y, {}, dtype, {}, mismatches);
    check_backward(layout, refs, x, core::to_vector(upstream), result.dx, dtype, {}, false, mismatches);
}

INSTANTIATE_TEST_SUITE_P(
    Dtypes,
    FusedLogSoftmaxDtypeTest,
    ::testing::Values(ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32),
    [](const ::testing::TestParamInfo<ttnn::DataType>& info) {
        return std::string(info.param == ttnn::DataType::FLOAT32 ? "FP32" : "BF16");
    });

// With the program cache enabled, alternate shapes, dtypes and upstream dtypes back to back, checking
// every result. Each saved state stays alive across later calls and is used for backward only after
// another forward has run; the first state is reused once more at the end.
TEST_F(FusedLogSoftmaxTest, ProgramCacheShapeAndDtypeAlternation) {
    autograd::ctx().get_device().enable_program_cache();
    struct Step {
        ttnn::Shape shape;
        ttnn::DataType dtype;
        BackwardSpec backward;
    };
    const auto bf16 = ttnn::DataType::BFLOAT16;
    const auto fp32 = ttnn::DataType::FLOAT32;
    const std::vector<Step> steps{
        {ttnn::Shape({1, 1, 32, 32000}), bf16, {Upstream::PivotOneHot, std::nullopt}},
        {ttnn::Shape({1, 1, 256, 32000}), bf16, {Upstream::CrossEntropyGrad, std::nullopt}},
        {ttnn::Shape({1, 1, 32, 32000}), fp32, {Upstream::PivotOneHot, std::nullopt}},
        {ttnn::Shape({1, 1, 32, 32000}), bf16, {Upstream::CrossEntropyGrad, fp32}},
        {ttnn::Shape({1, 1, 256, 32000}), fp32, {Upstream::Random, std::nullopt}},
        {ttnn::Shape({1, 1, 1, 32000}), fp32, {Upstream::PivotOneHot, std::nullopt}},
        {ttnn::Shape({1, 1, 32, 32000}), bf16, {Upstream::OtherOneHot, std::nullopt}},
    };
    std::vector<CaseOptions> options(steps.size());
    std::vector<FusedForward> forwards;
    forwards.reserve(steps.size());
    const auto backward = [&](size_t i) {
        SCOPED_TRACE("backward of step " + std::to_string(i));
        run_fused_backward(forwards[i], steps[i].shape, steps[i].dtype, steps[i].backward, options[i]);
    };
    for (size_t i = 0; i < steps.size(); ++i) {
        SCOPED_TRACE("forward of step " + std::to_string(i));
        const auto& step = steps[i];
        const RowLayout layout(step.shape);
        const auto columns = boundary_columns(layout.width);
        std::mt19937 rng(10000U + static_cast<uint32_t>(i));
        std::vector<float> logits(step.shape.volume());
        options[i].seed = 97U + static_cast<uint32_t>(i);
        for (size_t r = 0; r < layout.rows; ++r) {
            const size_t pivot = columns[(i * 7 + r) % columns.size()];
            const float gap = (r + i) % 3 == 0 ? 18.F : 0.F;
            fill_row(
                std::span(logits).subspan(r * layout.width, layout.width),
                pivot,
                kRowOffsets[(r + i) % kRowOffsets.size()],
                gap,
                rng);
            options[i].expected_pivots.push_back(pivot);
            options[i].kinds.push_back(gap > 0.F ? RowKind::Confident : RowKind::Normal);
        }
        forwards.push_back(run_fused_forward(step.shape, step.dtype, logits, options[i]));
        if (i > 0) {
            backward(i - 1);
        }
    }
    backward(steps.size() - 1);
    backward(0);
}

// set_grad accepts a gradient in L1. With the program cache on, a same-shaped DRAM gradient runs first, then the L1
// one, through the last axis and through a permuted one (the permuted gradient stays in L1): both give the DRAM
// result. The device op itself still rejects an L1 gradient, cached program or not.
TEST_F(FusedLogSoftmaxTest, L1UpstreamAfterDramUpstream) {
    autograd::ctx().get_device().enable_program_cache();
    const auto dtype = ttnn::DataType::BFLOAT16;
    const ttnn::Shape shape({1, 1, 64, 4000});
    const RowLayout layout(shape);
    std::mt19937 rng(11000U);
    std::uniform_real_distribution<float> uniform(-4.F, 4.F);
    std::vector<float> logits(shape.volume());
    std::vector<float> upstream_values(shape.volume());
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = uniform(rng);
        upstream_values[i] = uniform(rng) / 4.F;
    }
    const auto input = make_device_tensor(logits, shape, dtype);
    const auto dram = make_device_tensor(upstream_values, shape, dtype);
    const auto l1 = ttnn_fixed::to_l1_interleaved(dram);
    ASSERT_EQ(l1.buffer()->buffer_type(), ttnn::BufferType::L1);
    for (const int dim : {3, 2}) {
        SCOPED_TRACE("dim " + std::to_string(dim));
        const auto from_dram = run_autograd(input, dram, dim, /* composite */ false);
        const auto from_l1 = run_autograd(input, l1, dim, /* composite */ false);
        EXPECT_EQ(from_l1.y, from_dram.y);
        EXPECT_EQ(from_l1.dx, from_dram.dx);
        if (dim == 3) {
            const auto x = core::to_vector(input);
            const auto refs = reference_rows(x, layout.width);
            MismatchLog mismatches("L1 upstream");
            check_backward(layout, refs, x, core::to_vector(dram), from_l1.dx, dtype, {}, false, mismatches);
        }
    }
    const auto forward = metal::log_softmax_fw(input);
    (void)ttnn::prim::ttml_log_softmax_bw(input, dram, forward[1]);
    EXPECT_ANY_THROW((void)ttnn::prim::ttml_log_softmax_bw(input, l1, forward[1]));
}

// Logical [1, 1, 32, 32] padded to [1, 1, 32, 64]: the second tile column is padding past whole tiles, which the
// kernels would read as logits. The fused path must refuse it, so ops::log_softmax falls back to the composite.
TEST_F(FusedLogSoftmaxTest, RejectsPaddingBeyondTiles) {
    const ttnn::Shape logical({1, 1, 32, 32});
    const tt::tt_metal::TensorSpec spec(
        logical,
        tt::tt_metal::experimental::tensor_layout_from_padded_shape(
            ttnn::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(ttnn::Layout::TILE),
            ttnn::DRAM_MEMORY_CONFIG,
            logical,
            ttnn::Shape({1, 1, 32, 64})));
    const auto input = ttnn::create_device_tensor(spec, &autograd::ctx().get_device());
    ASSERT_EQ(input.padded_shape(), ttnn::Shape({1, 1, 32, 64}));
    EXPECT_FALSE(metal::log_softmax_supported(input));
    EXPECT_ANY_THROW((void)ttnn::prim::ttml_log_softmax_fw(input));
}
}  // namespace ttml::ops::tests
