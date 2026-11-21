// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Softmax collapse in the vocabulary-parallel cross-entropy (https://arxiv.org/abs/2610.31642, Section 2).
//
// The vocabulary is sharded across the TP devices of a [1, tp] mesh. When the target is the row's maximum
// and every other logit trails it by a large gap, the loss is log1p(tail) ~ tail and the target's gradient is
// -s * tail / (1 + tail), with tail = sum_{j != k} exp(x[j] - x[k]) over the whole (all-shard) row. Forming
// log(1 + tail) after 1 + tail has been rounded, adding it to the maximum before subtracting the target
// logit, or forming softmax(x)[t] - 1 after softmax(x)[t] has been rounded to BF16 loses both entirely: they
// come out exactly zero while tail is still representable.
//
// Every check compares against a float64 oracle on the BF16 logits the devices actually hold, with relative
// tolerances and no absolute slack beyond flushed denormals, so a collapsed value fails. The selected maximum
// is placed on every shard, at shard boundaries, with its competitors on the same or another shard, and tied
// across shards (the first maximum in vocabulary order is the selected one).

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
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
#include "metal/ops/vocab_parallel_cross_entropy/device/vocab_parallel_cross_entropy_device_operation.hpp"
#include "ops/distributed/losses.hpp"
#include "ops/losses.hpp"
#include "test_utils/mesh_utils.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops::distributed::tests {

namespace {

// Same relative tolerance as the log_softmax and fused cross-entropy collapse tests (BF16 outputs).
// Absolute slack only covers values that flush to zero as FP32/BF16 denormals.
constexpr double kBf16Rtol = 1e-2;
constexpr double kDenormalAtol = 1e-30;
constexpr size_t kMaxReportedMismatches = 8;

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

    [[nodiscard]] size_t failures() const {
        return failures_;
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

// Float64 reference of one full (all-shard) row: first maximum k, m = x[k], tail = sum_{j != k} exp(x[j] - m).
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

double reference_loss(std::span<const float> row, const RowReference& ref, uint32_t target) {
    return (ref.max - static_cast<double>(row[target])) + ref.log_normalizer;
}

// s * (softmax(x)[j] - [j == target]). At j == target == k it is -s * tail / (1 + tail).
double reference_grad(std::span<const float> row, const RowReference& ref, uint32_t target, size_t j, double s) {
    if (j == target && j == ref.pivot) {
        return -s * ref.tail / (1.0 + ref.tail);
    }
    const double probability = std::exp(static_cast<double>(row[j]) - ref.max - ref.log_normalizer);
    return s * (probability - (j == target ? 1.0 : 0.0));
}

std::string shape_name(uint32_t batch, uint32_t seq, uint32_t tp, uint32_t local_v) {
    return "[" + std::to_string(batch) + ",1," + std::to_string(seq) + "," + std::to_string(tp) + "x" +
           std::to_string(local_v) + "]";
}

// Logits [B, 1, S, tp * local_V]; shard s owns columns [s * local_V, (s + 1) * local_V).
struct Case {
    uint32_t batch;
    uint32_t seq;
    uint32_t tp;
    uint32_t local_v;
    std::vector<float> logits;
    std::vector<uint32_t> targets;  // B * S
    std::vector<bool> confident;    // Rows whose target is the unique first maximum with a gap <= 24.

    Case(uint32_t b, uint32_t s, uint32_t shards, uint32_t lv) :
        batch(b),
        seq(s),
        tp(shards),
        local_v(lv),
        logits(static_cast<size_t>(b) * s * shards * lv),
        targets(static_cast<size_t>(b) * s, 0U),
        confident(static_cast<size_t>(b) * s, false) {
    }
    [[nodiscard]] size_t width() const {
        return static_cast<size_t>(tp) * local_v;
    }
    [[nodiscard]] size_t rows() const {
        return targets.size();
    }
    std::span<float> row(size_t r) {
        return std::span(logits).subspan(r * width(), width());
    }
    [[nodiscard]] std::string name() const {
        return shape_name(batch, seq, tp, local_v);
    }
};

struct DeviceCase {
    autograd::TensorPtr logits;
    autograd::TensorPtr targets;
    std::vector<float> x;  // Quantized full logits, as the devices hold them.
    std::vector<RowReference> refs;
};

// Production TP layout (nano_gpt, cluster_axis = 1): logits sharded on the vocabulary along mesh axis 1,
// targets replicated.
DeviceCase upload(const Case& c) {
    auto* device = &autograd::ctx().get_device();
    using Placement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
    using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    auto logits_mapper = ttnn::distributed::TensorToMesh::create(
        *device, {.placements = ttsl::SmallVector<Placement>{Replicate{}, Shard{3}}});
    auto targets_mapper = ttnn::distributed::TensorToMesh::create(
        *device, {.placements = ttsl::SmallVector<Placement>{Replicate{}, Replicate{}}});

    const size_t width = c.width();
    xt::xarray<float> logits_xt = xt::adapt(c.logits, std::vector<size_t>{c.batch, 1U, c.seq, width});
    xt::xarray<uint32_t> targets_xt = xt::adapt(c.targets, std::vector<size_t>{c.batch, c.seq});
    auto logits_dev =
        core::from_xtensor<float, ttnn::DataType::BFLOAT16>(logits_xt, device, ttnn::Layout::TILE, &logits_mapper);
    auto targets_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        targets_xt, device, ttnn::Layout::ROW_MAJOR, &targets_mapper);

    DeviceCase d;
    d.logits = autograd::create_tensor(logits_dev, /* requires_grad */ true);
    d.targets = autograd::create_tensor(targets_dev, /* requires_grad */ false);
    const auto shards = core::to_xtensor<float>(logits_dev, core::IdentityComposer{});
    EXPECT_EQ(shards.size(), c.tp);
    d.x.resize(c.logits.size());
    for (size_t r = 0; r < c.rows(); ++r) {
        for (uint32_t s = 0; s < c.tp; ++s) {
            const auto& shard = shards[s];
            const size_t b = r / c.seq, q = r % c.seq;
            for (uint32_t j = 0; j < c.local_v; ++j) {
                d.x[r * width + static_cast<size_t>(s) * c.local_v + j] = shard(b, 0U, q, j);
            }
        }
        d.refs.push_back(reference_row(std::span<const float>(d.x).subspan(r * width, width)));
    }
    return d;
}

autograd::TensorPtr run_loss(const DeviceCase& d, ReduceType reduce) {
    return vocab_parallel_cross_entropy_loss(d.logits, d.targets, /* cluster_axis */ 1U, reduce);
}

// Per-row losses (ReduceType::NONE). Every TP device must hold the same, correct values.
void check_loss(const Case& c, const DeviceCase& d, const ttnn::Tensor& loss, const std::string& label) {
    const auto copies = core::to_xtensor<float>(loss, core::IdentityComposer{});
    ASSERT_EQ(copies.size(), c.tp);
    for (uint32_t s = 0; s < c.tp; ++s) {
        MismatchLog mismatches(label + " (device " + std::to_string(s) + ")");
        const auto& value = copies[s];
        ASSERT_EQ(value.size(), c.rows());
        for (size_t r = 0; r < c.rows(); ++r) {
            const auto row = std::span<const float>(d.x).subspan(r * c.width(), c.width());
            const double expected = reference_loss(row, d.refs[r], c.targets[r]);
            const double actual = value.flat(r);
            mismatches.near("loss", r, c.targets[r], actual, expected, kBf16Rtol * std::abs(expected));
            if (c.confident[r]) {
                mismatches.expect(actual > 0.0, "loss > 0 (confident target, collapse)", r, c.targets[r], actual);
            }
        }
    }
}

// Logit gradient shards; `row_scale[r]` is the scale s of row r (1/N for MEAN, times the upstream gradient).
void check_grad(const Case& c, const DeviceCase& d, std::span<const double> row_scale, const std::string& label) {
    ASSERT_TRUE(core::is_tensor_initialized(d.logits->get_grad()));
    const auto shards = core::to_xtensor<float>(d.logits->get_grad(), core::IdentityComposer{});
    ASSERT_EQ(shards.size(), c.tp);
    MismatchLog mismatches(label);
    for (size_t r = 0; r < c.rows(); ++r) {
        const auto row = std::span<const float>(d.x).subspan(r * c.width(), c.width());
        const size_t b = r / c.seq, q = r % c.seq;
        const double s = row_scale[r];
        for (uint32_t shard = 0; shard < c.tp; ++shard) {
            for (uint32_t j = 0; j < c.local_v; ++j) {
                const size_t col = static_cast<size_t>(shard) * c.local_v + j;
                const double expected = reference_grad(row, d.refs[r], c.targets[r], col, s);
                const double actual = shards[shard](b, 0U, q, j);
                mismatches.near("dx", r, col, actual, expected, kBf16Rtol * std::abs(expected) + kDenormalAtol);
            }
        }
        if (c.confident[r] && s != 0.0) {
            const size_t t = c.targets[r];
            const double actual = shards[t / c.local_v](b, 0U, q, t % c.local_v);
            mismatches.expect(
                actual * s < 0.0, "dx[target] nonzero, sign of -s (confident target, collapse)", r, t, actual);
        }
    }
}

void backward_with(const autograd::TensorPtr& loss, const std::vector<float>& upstream, const ttnn::Shape& shape) {
    auto* device = &autograd::ctx().get_device();
    auto replicate = ttnn::distributed::replicate_tensor_to_mesh_mapper(*device);
    xt::xarray<float> g = xt::adapt(upstream, std::vector<size_t>(shape.cbegin(), shape.cend()));
    loss->set_grad(core::from_xtensor<float, ttnn::DataType::BFLOAT16>(g, device, ttnn::Layout::TILE, replicate.get()));
    loss->backward();
}

// Rows whose target is the unique first maximum, on shard (r % tp) at its first column, last column, middle,
// or a pseudo-random column. Competitors trail by `gap`: one on the same shard, one on the next shard (across
// the boundary when the target sits at a shard edge), or all of them. The rest sit 100 below the maximum,
// where exp underflows to an FP32 denormal (flushed on device).
enum class Competitors : uint8_t { SameShard, OtherShard, All };

constexpr std::array<float, 6> kGaps{8.F, 9.F, 12.F, 17.F, 20.F, 24.F};
constexpr std::array<float, 3> kOffsets{0.F, -32.F, 32.F};

void fill_confident_row(Case& c, size_t r, size_t target, float offset, float gap, Competitors competitors) {
    auto row = c.row(r);
    const size_t width = row.size();
    std::fill(row.begin(), row.end(), competitors == Competitors::All ? offset - gap : offset - 100.F);
    if (competitors == Competitors::SameShard && c.local_v > 1) {
        const size_t shard_begin = (target / c.local_v) * c.local_v;
        row[shard_begin + (target - shard_begin + 1) % c.local_v] = offset - gap;
    } else if (competitors == Competitors::OtherShard) {
        row[(target + c.local_v) % width] = offset - gap;
        if (target % c.local_v == c.local_v - 1) {
            row[(target + 1) % width] = offset - gap;  // First column of the next shard.
        }
    }
    row[target] = offset;
    c.targets[r] = static_cast<uint32_t>(target);
    c.confident[r] = true;
}

Case confident_case(uint32_t batch, uint32_t seq, uint32_t tp, uint32_t local_v) {
    Case c(batch, seq, tp, local_v);
    for (size_t r = 0; r < c.rows(); ++r) {
        const size_t shard = r % tp;
        const std::array<size_t, 4> positions{0, local_v - 1U, local_v / 2U, (r * 131 + 7) % local_v};
        const size_t target = shard * local_v + positions[(r / tp) % positions.size()];
        const auto competitors = static_cast<Competitors>((r / kGaps.size()) % 3);
        fill_confident_row(c, r, target, kOffsets[r % kOffsets.size()], kGaps[r % kGaps.size()], competitors);
    }
    return c;
}

uint32_t mesh_tp() {
    const size_t devices = test_utils::system_mesh_size();
    return devices >= 8 ? 8U : (devices >= 4 ? 4U : 2U);
}

}  // namespace

// A [1, tp] mesh (tp = 8, 4 or 2, the largest the system holds), opened once for the suite.
class VocabParallelCollapseTest : public ::testing::Test {
protected:
    static tt::tt_metal::distributed::MeshShape mesh_shape() {
        return tt::tt_metal::distributed::MeshShape(1, mesh_tp());
    }

    static void SetUpTestSuite() {
        if (test_utils::system_supports_mesh(tt::tt_metal::distributed::MeshShape(1, 2))) {
            ttnn_fixed::distributed::enable_fabric(static_cast<uint32_t>(mesh_shape().mesh_size()));
            autograd::ctx().open_device(mesh_shape());
        }
    }

    static void TearDownTestSuite() {
        if (test_utils::system_supports_mesh(tt::tt_metal::distributed::MeshShape(1, 2))) {
            autograd::ctx().close_device();
        }
    }

    void SetUp() override {
        SKIP_UNLESS_MESH_SUPPORTED(tt::tt_metal::distributed::MeshShape(1, 2));
    }

    void TearDown() override {
        autograd::ctx().reset_graph();
    }

    static uint32_t tp() {
        return mesh_tp();
    }
};

// Validation that needs one device only, so it also runs without a mesh.
class VocabParallelValidationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        autograd::ctx().close_device();
    }
};

// Logits padded past whole tiles: logical [1, 1, 32, 32] padded to [1, 1, 64, 32] would run a second tile row the
// one-tile-row stats output does not hold, and [1, 1, 32, 64] would read a padding tile as logits. Both must be
// refused before any kernel runs.
TEST_F(VocabParallelValidationTest, RejectsPaddingBeyondTiles) {
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
        const auto logits = ttnn::create_device_tensor(spec, device);
        ASSERT_EQ(logits.padded_shape(), padded);
        EXPECT_ANY_THROW((void)ttnn::prim::ttml_vocab_parallel_ce_stats(logits, target, std::nullopt));
    }
}

// The target is the global maximum on each shard in turn and every other logit trails by 8-24: per-row
// loss (NONE) and the target's gradient must stay strictly positive / negative and within BF16 relative
// tolerance of float64. Local widths: NanoLlama3's 32000 / tp, a tile-aligned 96 and an unaligned 100.
TEST_F(VocabParallelCollapseTest, ConfidentTargetKeepsLossAndGradient) {
    for (const auto& [batch, seq, local_v] :
         {std::array<uint32_t, 3>{1U, 64U, 32000U / tp()},
          std::array<uint32_t, 3>{2U, 35U, 96U},
          std::array<uint32_t, 3>{1U, 64U, 100U}}) {
        const auto c = confident_case(batch, seq, tp(), local_v);
        SCOPED_TRACE(c.name());
        const auto d = upload(c);
        auto loss = run_loss(d, ReduceType::NONE);
        check_loss(c, d, loss->get_value(), c.name() + " forward");
        std::vector<float> upstream(c.rows(), 0.25F);
        backward_with(loss, upstream, ttnn::Shape({batch, 1, seq, 1}));
        const std::vector<double> scale(c.rows(), 0.25);
        check_grad(c, d, scale, c.name() + " backward");
        autograd::ctx().reset_graph();
    }
}

// The training path: MEAN reduction (nano_gpt) on a batch where every target is confidently predicted with
// a full tail (gaps 17-24, below the FP32 cliff of 1 + tail): the mean loss must not collapse to zero and
// every target gradient must stay negative.
TEST_F(VocabParallelCollapseTest, MeanLossKeepsConfidentSignal) {
    const uint32_t batch = 2U, seq = 32U, local_v = 1024U;
    Case c(batch, seq, tp(), local_v);
    for (size_t r = 0; r < c.rows(); ++r) {
        const size_t target = (r % tp()) * local_v + (r * 37 + 3) % local_v;
        fill_confident_row(c, r, target, 0.F, 17.F + static_cast<float>(r % 8), Competitors::All);
    }
    const auto d = upload(c);
    auto loss = run_loss(d, ReduceType::MEAN);
    double expected_loss = 0.0;
    for (size_t r = 0; r < c.rows(); ++r) {
        const auto row = std::span<const float>(d.x).subspan(r * c.width(), c.width());
        expected_loss += reference_loss(row, d.refs[r], c.targets[r]);
    }
    expected_loss /= static_cast<double>(c.rows());
    const auto copies = core::to_xtensor<float>(loss->get_value(), core::IdentityComposer{});
    for (const auto& value : copies) {
        ASSERT_EQ(value.size(), 1U);
        EXPECT_GT(value.flat(0), 0.F) << "mean loss collapsed to zero";
        EXPECT_NEAR(value.flat(0), expected_loss, kBf16Rtol * expected_loss);
    }
    backward_with(loss, {1.F}, ttnn::Shape({1, 1, 1, 1}));
    const std::vector<double> scale(c.rows(), 1.0 / static_cast<double>(c.rows()));
    check_grad(c, d, scale, c.name() + " MEAN backward");
}

// Maxima tied across shards: the first copy in vocabulary order is the selected maximum, and every other
// copy belongs to the tail (each contributes exactly 1). Targets: the first copy, a later copy, or neither.
// With the rest 20 below, the loss is log(copies) to BF16 precision and not a collapse case, but the
// selection must be consistent across devices.
TEST_F(VocabParallelCollapseTest, MaximaTiedAcrossShards) {
    const uint32_t batch = 1U, seq = 64U, local_v = 96U;
    Case c(batch, seq, tp(), local_v);
    for (size_t r = 0; r < c.rows(); ++r) {
        auto row = c.row(r);
        std::fill(row.begin(), row.end(), -20.F);
        // Copies at the same local column on shards a < b (a shard boundary when the column is 0 or local_v-1).
        const size_t a = r % tp(), b = (a + 1 + (r / tp()) % (tp() - 1)) % tp();
        const size_t first = std::min(a, b), second = std::max(a, b);
        const size_t column = std::array<size_t, 3>{0, local_v - 1U, 41U}[r % 3];
        row[first * local_v + column] = 5.F;
        row[second * local_v + column] = 5.F;
        if (r % 4 == 3) {
            row[(first * local_v + column + 1) % c.width()] = 5.F;  // A third copy right after the first.
        }
        const size_t pick = (r / 3) % 3;
        c.targets[r] = static_cast<uint32_t>(
            pick == 0 ? first * local_v + column : (pick == 1 ? second * local_v + column : (r * 53) % c.width()));
    }
    const auto d = upload(c);
    auto loss = run_loss(d, ReduceType::NONE);
    check_loss(c, d, loss->get_value(), c.name() + " tied forward");
    std::vector<float> upstream(c.rows(), 1.F);
    backward_with(loss, upstream, ttnn::Shape({batch, 1, seq, 1}));
    const std::vector<double> scale(c.rows(), 1.0);
    check_grad(c, d, scale, c.name() + " tied backward");
}

// Random logits at the NanoLlama3 vocabulary, mixed with confident rows (target and competitors on any
// shard) and tied maxima whose target is the later copy.
TEST_F(VocabParallelCollapseTest, MixedRowsMatchFloat64) {
    const uint32_t batch = 2U, seq = 64U, local_v = 32000U / tp();
    Case c(batch, seq, tp(), local_v);
    std::mt19937 rng(2026U);
    std::uniform_real_distribution<float> logit(-10.F, 10.F);
    std::uniform_int_distribution<uint32_t> column(0U, static_cast<uint32_t>(c.width() - 1));
    for (size_t r = 0; r < c.rows(); ++r) {
        auto row = c.row(r);
        const uint32_t target = column(rng);
        c.targets[r] = target;
        switch (r % 4) {
            case 0:
                fill_confident_row(
                    c, r, target, kOffsets[r % 3], kGaps[(r / 4) % kGaps.size()], Competitors::OtherShard);
                break;
            case 1: {
                for (auto& v : row) v = logit(rng);
                const size_t first = target / 2;
                if (first != target) {
                    row[first] = 12.F;
                }
                row[target] = 12.F;
                break;
            }
            default:
                for (auto& v : row) v = logit(rng);
        }
    }
    const auto d = upload(c);
    auto loss = run_loss(d, ReduceType::NONE);
    check_loss(c, d, loss->get_value(), c.name() + " mixed forward");
    std::vector<float> upstream(c.rows());
    // Per-row weights exactly representable in BF16 (the upstream gradient is uploaded as BF16), some zero.
    for (size_t r = 0; r < c.rows(); ++r) upstream[r] = r % 7 == 0 ? 0.F : 0.5F + static_cast<float>(r % 50) / 64.F;
    backward_with(loss, upstream, ttnn::Shape({batch, 1, seq, 1}));
    std::vector<double> scale(c.rows());
    for (size_t r = 0; r < c.rows(); ++r) scale[r] = static_cast<double>(upstream[r]);
    check_grad(c, d, scale, c.name() + " mixed backward");
}

// Shards whose columns of a row are all masked (-inf), as with a vocabulary mask: an empty shard contributes
// nothing to its own row statistics or to the combine (no NaN from -inf - (-inf)) and gets a zero gradient. Rows:
// random logits on the target's shard only, random logits with the next shard masked, zeros on the target's shard
// only (loss log(local_V); with tp = 2, zeros next to a fully masked shard), and a confident target on its shard
// only. Local widths: one tile, three tiles, and NanoLlama3's 32000 / tp (split across cores within a shard).
TEST_F(VocabParallelCollapseTest, MaskedShardsContributeNothing) {
    for (const uint32_t local_v : {32U, 96U, 32000U / tp()}) {
        Case c(1U, 64U, tp(), local_v);
        std::mt19937 rng(local_v);
        std::uniform_real_distribution<float> logit(-10.F, 10.F);
        for (size_t r = 0; r < c.rows(); ++r) {
            const size_t shard = r % tp();
            const size_t target = shard * local_v + (r * 13) % local_v;
            const size_t pattern = (r / tp()) % 4;
            auto row = c.row(r);
            if (pattern == 3) {
                fill_confident_row(c, r, target, kOffsets[r % 3], kGaps[r % kGaps.size()], Competitors::SameShard);
            } else {
                for (auto& v : row) v = pattern == 2 ? 0.F : logit(rng);
                c.targets[r] = static_cast<uint32_t>(target);
            }
            for (size_t s = 0; s < tp(); ++s) {
                const bool masked = pattern == 1 ? s == (shard + 1) % tp() : s != shard;
                if (masked) {
                    std::fill_n(
                        row.begin() + static_cast<std::ptrdiff_t>(s * local_v),
                        local_v,
                        -std::numeric_limits<float>::infinity());
                }
            }
        }
        SCOPED_TRACE(c.name());
        const auto d = upload(c);
        auto loss = run_loss(d, ReduceType::NONE);
        check_loss(c, d, loss->get_value(), c.name() + " masked forward");
        backward_with(loss, std::vector<float>(c.rows(), 1.F), ttnn::Shape({1, 1, 64, 1}));
        check_grad(c, d, std::vector<double>(c.rows(), 1.0), c.name() + " masked backward");
        autograd::ctx().reset_graph();
    }
}

// The target may live in DRAM or L1, which the readers' tensor accessors compile in. A launch with L1 targets after
// one with same-shaped DRAM targets must build its own programs: reusing the DRAM ones would read the L1 address in
// DRAM. The second launch has other targets, so stale reads cannot pass by matching.
TEST_F(VocabParallelCollapseTest, L1TargetsAfterDramTargets) {
    const uint32_t seq = 64U, local_v = 96U;
    const auto dram_case = confident_case(1U, seq, tp(), local_v);
    Case l1_case(1U, seq, tp(), local_v);
    std::mt19937 rng(7U);
    std::uniform_real_distribution<float> logit(-10.F, 10.F);
    for (auto& v : l1_case.logits) v = logit(rng);
    for (size_t r = 0; r < l1_case.rows(); ++r) {
        l1_case.targets[r] = static_cast<uint32_t>((dram_case.targets[r] + 1 + r * 29) % l1_case.width());
    }
    for (const bool l1 : {false, true}) {
        const auto& c = l1 ? l1_case : dram_case;
        const std::string label = c.name() + (l1 ? " L1 targets" : " DRAM targets");
        SCOPED_TRACE(label);
        auto d = upload(c);
        if (l1) {
            d.targets = autograd::create_tensor(
                ttnn_fixed::to_l1_interleaved(d.targets->get_value()), /* requires_grad */ false);
            ASSERT_EQ(d.targets->get_value().memory_config().buffer_type(), ttnn::BufferType::L1);
        }
        auto loss = run_loss(d, ReduceType::NONE);
        check_loss(c, d, loss->get_value(), label + " forward");
        backward_with(loss, std::vector<float>(c.rows(), 1.F), ttnn::Shape({1, 1, seq, 1}));
        check_grad(c, d, std::vector<double>(c.rows(), 1.0), label + " backward");
        autograd::ctx().reset_graph();
    }
}

}  // namespace ttml::ops::distributed::tests
