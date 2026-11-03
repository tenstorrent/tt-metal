// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tt-metalium/constants.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/core/xoperation.hpp"

// Helpers for the program-cache tests, which launch an op again on fresh tensors while the earlier launches' tensors
// stay alive: a cache hit must read and write its own buffers on every core.

namespace ttml::test_utils {

// The device with its program cache enabled and empty, so that the calling test's first launch is a miss.
inline ttnn::distributed::MeshDevice& device_with_empty_program_cache() {
    auto& device = ttml::autograd::ctx().get_device();
    device.enable_program_cache();
    device.clear_program_cache();
    return device;
}

// The cores of the device's compute grid, which the program factories split their work over.
inline uint32_t num_worker_cores() {
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    return static_cast<uint32_t>(grid.x * grid.y);
}

// Two work units per core and `extra` more. With 0 < extra < cores, split_work_to_cores gives `extra` cores three
// units and the others two: two core groups, so both of a factory's per-group kernels run, on any grid.
inline uint32_t two_core_group_units(uint32_t extra) {
    return 2U * num_worker_cores() + extra;
}

// [1, 1, two_core_group_units(3) * 32, width], for an op that splits tile rows (or, at width 32, tiles) over the
// cores: three cores take three and the others two, so both core groups run on any grid.
inline std::vector<uint32_t> two_core_group_rows_shape(uint32_t width) {
    return {1U, 1U, two_core_group_units(3U) * tt::constants::TILE_HEIGHT, width};
}

// The smallest count of `unit`-tile blocks whose tiles are more than twice the core count and not a multiple of it,
// so that their split has two core groups.
inline uint32_t blocks_for_two_core_groups(uint32_t unit) {
    const uint32_t cores = num_worker_cores();
    uint32_t blocks = 2U * cores / unit + 1U;
    if ((blocks * unit) % cores == 0U) {
        ++blocks;  // one step is enough: unit < cores, so the next total is not a multiple
    }
    return blocks;
}

// A sequence length S for an op that splits N * ceil(S / 32) tile rows over the cores. With N = 2 that is
// 2 * cores + 4 tile rows: four cores take three and the others two, so both core groups run on any grid. The last
// tile row of each batch entry holds 32 - `padding` rows.
inline uint32_t two_core_group_seq_len(uint32_t padding) {
    return (num_worker_cores() + 2U) * tt::constants::TILE_HEIGHT - padding;
}

// With N = 2, a sequence length that gives grid.y + 1 or grid.y + 2 tile rows, one per core: a factory that walks the
// grid column by column fills its first column and one or two cores of the second. A hit that walks the cores in
// another order misses some that got work, and one that walks further reaches idle cores, which have no runtime args.
// The last tile row of each batch entry holds 32 - `padding` rows.
inline uint32_t partial_grid_seq_len(uint32_t padding) {
    const auto grid = ttml::autograd::ctx().get_device().compute_with_storage_grid_size();
    return (static_cast<uint32_t>(grid.y) / 2U + 1U) * tt::constants::TILE_HEIGHT - padding;
}

// A bf16 TILE tensor of `value`, e.g. a preallocated output filled with a value the op cannot write. ttnn fills a large
// one on the device, with a program that adds a cache entry of its own: make it before reading the entry count that a
// launch's cache check compares with.
inline ttnn::Tensor filled_on_device(const std::vector<uint32_t>& shape, float value) {
    return ttml::core::full(ttnn::Shape(shape), value, &ttml::autograd::ctx().get_device());
}

// A bf16 TILE tensor of zeros, e.g. for a preallocated output. Make it before reading the entry count, as for
// filled_on_device().
inline ttnn::Tensor zeros_on_device(const std::vector<uint32_t>& shape) {
    return ttml::core::zeros(ttnn::Shape(shape), &ttml::autograd::ctx().get_device());
}

// A later launch's tensor must not share a buffer with an earlier one's, or a hit that kept the old address would pass.
inline void expect_new_buffer(const ttnn::Tensor& earlier, const ttnn::Tensor& later, const std::string& label) {
    EXPECT_NE(later.buffer()->address(), earlier.buffer()->address()) << label;
}

// expect_new_buffer for each pair: later[i] against earlier[i].
inline void expect_new_buffers(
    const std::vector<ttnn::Tensor>& earlier, const std::vector<ttnn::Tensor>& later, const std::string& label) {
    ASSERT_EQ(earlier.size(), later.size()) << label << ": the lists must pair up";
    for (std::size_t i = 0; i < earlier.size(); ++i) {
        expect_new_buffer(earlier[i], later[i], label + ": tensor " + std::to_string(i));
    }
}

// What a launch must do to the program cache: compile a program, reuse a cached one, or either.
enum class CacheExpectation { Miss, Hit, Unchecked };

// A non-fatal check of what the launch run since `entries_before` was read did to the program cache. Read
// `entries_before` after making every tensor of the launch: a bf16 upload tilizes on the device and a large fill runs a
// fill program, and either adds an entry of its own the first time.
inline void expect_cache(std::size_t entries_before, CacheExpectation expectation, const std::string& label) {
    const auto entries = ttml::autograd::ctx().get_device().num_program_cache_entries();
    if (expectation == CacheExpectation::Miss) {
        EXPECT_GT(entries, entries_before) << label << " must compile a program";
    } else if (expectation == CacheExpectation::Hit) {
        EXPECT_EQ(entries, entries_before) << label << " must reuse a cached program";
    }
}

// A later launch must not write into an earlier launch's tensors: `tensor` still holds `snapshot`.
inline void expect_unchanged(const ttnn::Tensor& tensor, const xt::xarray<float>& snapshot, const std::string& label) {
    EXPECT_TRUE(xt::all(xt::equal(ttml::core::to_xtensor(tensor), snapshot)))
        << label << " changed after a later launch";
}

// Equal bits: tells -0 from 0, and a NaN equals the same NaN.
inline bool same_bits(float a, float b) {
    return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b);
}

// The elements whose bits differ; all of them if the shapes differ.
inline std::size_t count_bit_mismatches(const xt::xarray<float>& expected, const xt::xarray<float>& actual) {
    if (expected.shape() != actual.shape()) {
        return std::max(expected.size(), actual.size());
    }
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        mismatches += static_cast<std::size_t>(!same_bits(expected.flat(i), actual.flat(i)));
    }
    return mismatches;
}

// A suite that opens the device once for all of its tests.
class DeviceSuiteTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// A suite that opens the device once and starts every test from an empty program cache.
class ProgramCacheSuiteTest : public DeviceSuiteTest {
protected:
    void SetUp() override {
        device_with_empty_program_cache();
    }
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

}  // namespace ttml::test_utils
