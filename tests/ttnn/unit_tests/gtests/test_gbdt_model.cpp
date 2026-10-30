// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>

#include "ttnn/operations/matmul/device/config/model/gbdt_model.hpp"

namespace ttnn::operations::matmul::model {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

constexpr std::uint8_t missing(std::uint8_t type, bool default_left) {
    return static_cast<std::uint8_t>((type << kGbdtMissingShift) | (default_left ? kGbdtDefaultLeft : 0));
}

// Three trees over 3 features, one per missing-value mode, plus a single-leaf tree:
//  tree 0: x0 <= 1 (missing NaN, default left) ? 1 : (x1 <= 0 (missing none) ? 10 : 100)
//  tree 1: 0.5
//  tree 2: x2 <= -1 (missing zero, default right) ? 1000 : 2000
//  tree 3: x2 <= +inf (missing NaN, default right) ? 0 : 10000
constexpr std::uint32_t kNodeBegin[] = {0, 2, 2, 3, 4};
constexpr std::uint32_t kLeafBegin[] = {0, 3, 4, 6, 8};
constexpr std::uint8_t kFeature[] = {0, 1, 2, 2};
constexpr std::uint8_t kDecision[] = {
    missing(kGbdtMissingNaN, true),
    missing(kGbdtMissingNone, true),
    missing(kGbdtMissingZero, false),
    missing(kGbdtMissingNaN, false)};
constexpr double kThresholds[] = {-1.0, 0.0, 1.0, kInf};
constexpr std::uint16_t kThresholdIndex[] = {2, 1, 0, 3};
constexpr std::int16_t kLeft[] = {~0, ~1, ~0, ~0};
constexpr std::int16_t kRight[] = {1, ~2, ~1, ~1};
constexpr double kLeaves[] = {1.0, 10.0, 100.0, 0.5, 1000.0, 2000.0, 0.0, 10000.0};

constexpr GbdtModel kToy{
    .num_features = 3,
    .node_begin = kNodeBegin,
    .leaf_begin = kLeafBegin,
    .split_feature = kFeature,
    .decision = kDecision,
    .threshold_index = kThresholdIndex,
    .left_child = kLeft,
    .right_child = kRight,
    .thresholds = kThresholds,
    .leaf_value = kLeaves,
};

double toy(double x0, double x1, double x2) {
    const double x[] = {x0, x1, x2};
    return kToy.predict(x);
}

TEST(GbdtModel, ToyIsValid) { EXPECT_EQ(kToy.structure_error(), ""); }

TEST(GbdtModel, SplitRuleIsLessOrEqual) {
    EXPECT_EQ(toy(1.0, 0.0, -1.0), 1.0 + 0.5 + 1000.0);
    EXPECT_EQ(toy(std::nextafter(1.0, 2.0), 0.0, -1.0), 10.0 + 0.5 + 1000.0);
    EXPECT_EQ(toy(2.0, 1e-30, std::nextafter(-1.0, 0.0)), 100.0 + 0.5 + 2000.0);
}

TEST(GbdtModel, MissingNaNTakesDefaultDirection) {
    EXPECT_EQ(toy(kNaN, 5.0, 5.0), 1.0 + 0.5 + 2000.0);  // tree 0: default left
    // tree 2 (missing zero) reads NaN as 0 and takes its default; tree 3 (missing NaN) takes its default
    EXPECT_EQ(toy(5.0, 5.0, kNaN), 100.0 + 0.5 + 2000.0 + 10000.0);
}

TEST(GbdtModel, MissingNoneTreatsNaNAsZero) {
    EXPECT_EQ(toy(2.0, kNaN, 5.0), 10.0 + 0.5 + 2000.0);  // NaN -> 0 <= 0
}

TEST(GbdtModel, MissingZeroTakesDefaultForZeroAndNaN) {
    EXPECT_EQ(toy(0.0, 0.0, 0.0), 1.0 + 0.5 + 2000.0);
    EXPECT_EQ(toy(0.0, 0.0, -0.0), 1.0 + 0.5 + 2000.0);
    EXPECT_EQ(toy(0.0, 0.0, -5.0), 1.0 + 0.5 + 1000.0);
}

TEST(GbdtModel, InputsBelowZeroThresholdReadAsZero) {
    EXPECT_EQ(toy(2.0, 1e-40, 5.0), 10.0 + 0.5 + 2000.0);
    EXPECT_EQ(toy(2.0, 1e-30, 5.0), 100.0 + 0.5 + 2000.0);
}

TEST(GbdtModel, InfinityThreshold) {
    EXPECT_EQ(toy(2.0, kInf, kInf), 100.0 + 0.5 + 2000.0);  // inf <= inf: left in tree 3
    EXPECT_EQ(toy(2.0, -kInf, -kInf), 10.0 + 0.5 + 1000.0);
}

TEST(GbdtModel, BatchMatchesSingleRows) {
    const double rows[] = {1.0, 0.0, -1.0, kNaN, 5.0, 5.0, 2.0, kNaN, 5.0, 2.0, -kInf, -0.0};
    double out[5];
    kToy.predict_batch(rows, out);
    for (int r = 0; r < 4; ++r) {
        EXPECT_EQ(out[r], kToy.predict(std::span<const double>(rows + 3 * r, 3))) << r;
    }
    EXPECT_TRUE(std::isnan(out[4]));  // no row for this slot
}

TEST(GbdtModel, ShortRowIsNaN) {
    const double x[] = {1.0, 2.0};
    EXPECT_TRUE(std::isnan(kToy.predict(x)));
}

TEST(GbdtModel, ValidateRejectsBackEdges) {
    constexpr std::int16_t bad_right[] = {0, ~2, ~1, ~1};  // node 0 -> node 0
    GbdtModel m = kToy;
    m.right_child = bad_right;
    EXPECT_NE(m.structure_error(), "");
}

TEST(GbdtModel, ValidateRejectsTreeOffsetsPastTheArrays) {
    // Tree 0 claims 5 of the 4 nodes; its end offsets pass the per-tree checks, so the walk must be bounded first
    constexpr std::uint32_t node_begin[] = {0, 5, 2, 3, 4};
    constexpr std::uint32_t leaf_begin[] = {0, 6, 4, 6, 8};
    GbdtModel m = kToy;
    m.node_begin = node_begin;
    m.leaf_begin = leaf_begin;
    EXPECT_EQ(m.structure_error(), "tree offsets not monotone or out of range");
}

TEST(GbdtModel, BlobRejectsGarbage) {
    const char junk[] = "not a model";
    std::string error;
    EXPECT_FALSE(GbdtModelStorage::from_blob(std::as_bytes(std::span<const char>(junk)), error).has_value());
    EXPECT_NE(error, "");
}

std::vector<char> slurp(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}

// Parity against LightGBM on feature rows, when TT_GBDT_PARITY_DIR names a directory holding model.blob,
// rows.bin (row-major float64, num_features per row) and ref.bin (LightGBM's Booster.predict(rows, raw_score=True)).
TEST(GbdtModel, ParityWithLightGBM) {
    const char* dir = std::getenv("TT_GBDT_PARITY_DIR");
    if (dir == nullptr) {
        GTEST_SKIP() << "TT_GBDT_PARITY_DIR not set";
    }
    const std::filesystem::path d(dir);
    const auto blob = slurp(d / "model.blob");
    std::string error;
    const auto storage =
        GbdtModelStorage::from_blob(std::as_bytes(std::span<const char>(blob.data(), blob.size())), error);
    ASSERT_TRUE(storage.has_value()) << error;
    const GbdtModel& m = storage->model();
    const auto xb = slurp(d / "rows.bin");
    const auto rb = slurp(d / "ref.bin");
    ASSERT_EQ(xb.size(), rb.size() * m.num_features);
    std::vector<double> x(xb.size() / sizeof(double));
    std::vector<double> ref(rb.size() / sizeof(double));
    std::memcpy(x.data(), xb.data(), xb.size());
    std::memcpy(ref.data(), rb.data(), rb.size());
    std::vector<double> batch(ref.size());
    m.predict_batch(x, batch);
    double max_abs = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double p = m.predict(std::span<const double>(x.data() + i * m.num_features, m.num_features));
        max_abs = std::max({max_abs, std::fabs(p - ref[i]), std::fabs(batch[i] - ref[i])});
    }
    EXPECT_LT(max_abs, 1e-9) << "over " << ref.size() << " rows";
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace
}  // namespace ttnn::operations::matmul::model
