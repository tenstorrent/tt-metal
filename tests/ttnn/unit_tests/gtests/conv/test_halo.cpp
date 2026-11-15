// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// halo's optional inputs, each against the default path on the same shapes:
// - halo_boundary: halo of an image whose border comes from the boundary tensor must equal, bit for bit, halo with
//   zero padding of the image with that border written in (same output sticks, data movement only).
// - prologue_scale / prologue_shift: silu(x * scale + shift) in the untilize pass, against a host fp32 golden of the
//   default output; padding sticks stay pad_val.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include "ttnn/device.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/sliding_window/halo/device/halo_device_operation_types.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/unit_tests/gtests/ttnn_test_fixtures.hpp"

namespace ttnn::operations::sliding_window::test {

using tt::tt_metal::distributed::MeshDevice;

class HaloTest : public TTNNFixtureWithSuiteDevice<HaloTest> {
public:
    // halo keeps its kernel config tensors in L1_SMALL, which the default fixture does not reserve.
    static void SetUpTestSuite() {
        device_holder_ = ttnn::open_mesh_device(/*device_id=*/0, /*l1_small_size=*/16384, DEFAULT_TRACE_REGION_SIZE);
        device_ = device_holder_.get();
    }
};

namespace {

// NHWC image, one stick of `c` channels per pixel.
struct Image {
    uint32_t n, h, w, c;
    std::vector<bfloat16> data;
    bfloat16* stick(uint32_t b, uint32_t r, uint32_t col) { return &data[(((b * h) + r) * w + col) * c]; }
};

Image random_image(uint32_t n, uint32_t h, uint32_t w, uint32_t c, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    Image img{n, h, w, c, std::vector<bfloat16>(static_cast<size_t>(n) * h * w * c)};
    for (auto& v : img.data) {
        v = bfloat16(dist(rng));
    }
    return img;
}

// Columns [c0, c1) of every row.
Image crop_columns(Image& img, uint32_t c0, uint32_t c1) {
    Image out{img.n, img.h, c1 - c0, img.c, {}};
    out.data.reserve(static_cast<size_t>(out.n) * out.h * out.w * out.c);
    for (uint32_t b = 0; b < img.n; ++b) {
        for (uint32_t r = 0; r < img.h; ++r) {
            out.data.insert(out.data.end(), img.stick(b, r, c0), img.stick(b, r, c1 - 1) + img.c);
        }
    }
    return out;
}

// Interior of a bordered image: rows and columns 1 .. size - 2.
Image interior(Image& framed) {
    Image cropped = crop_columns(framed, 1, framed.w - 1);
    Image out{framed.n, framed.h - 2, cropped.w, framed.c, {}};
    for (uint32_t b = 0; b < framed.n; ++b) {
        out.data.insert(out.data.end(), cropped.stick(b, 1, 0), cropped.stick(b, framed.h - 1, 0));
    }
    return out;
}

// The one-stick border of a bordered image, in the ExternalHaloSpec order: top row, bottom row, left column of the
// interior rows, right column of the interior rows.
std::vector<bfloat16> border_sticks(Image& framed) {
    std::vector<bfloat16> out;
    const auto append = [&](uint32_t b, uint32_t r, uint32_t c0, uint32_t c1) {
        out.insert(out.end(), framed.stick(b, r, c0), framed.stick(b, r, c1 - 1) + framed.c);
    };
    for (uint32_t b = 0; b < framed.n; ++b) {
        append(b, 0, 0, framed.w);
        append(b, framed.h - 1, 0, framed.w);
        for (uint32_t r = 1; r + 1 < framed.h; ++r) {
            append(b, r, 0, 1);
        }
        for (uint32_t r = 1; r + 1 < framed.h; ++r) {
            append(b, r, framed.w - 1, framed.w);
        }
    }
    return out;
}

// bfloat8_b goes through from_vector<float> (encode, then convert).
Tensor from_bf16(const std::vector<bfloat16>& data, const TensorSpec& spec) {
    if (spec.data_type() == DataType::BFLOAT16) {
        return Tensor::from_vector(data, spec);
    }
    return Tensor::from_vector(std::vector<float>(data.begin(), data.end()), spec);
}

Tensor height_sharded(
    MeshDevice& device, const Image& img, Layout layout, uint32_t num_cores, DataType dtype = DataType::BFLOAT16) {
    const uint32_t sticks = img.n * img.h * img.w;
    TT_FATAL(sticks % num_cores == 0, "test shapes split evenly");
    const uint32_t shard_h = sticks / num_cores;
    TT_FATAL(layout == Layout::ROW_MAJOR || shard_h % tt::constants::TILE_HEIGHT == 0, "tile-aligned shards");
    const CoreRangeSet grid(CoreRange({0, 0}, {num_cores - 1, 0}));
    const MemoryConfig mem(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, ShardSpec(grid, {shard_h, img.c}));
    const TensorSpec spec(ttnn::Shape({1, 1, sticks, img.c}), TensorLayout(dtype, PageConfig(layout), mem));
    return from_bf16(img.data, spec).to_device(&device, mem);
}

Tensor dram(
    MeshDevice& device,
    const std::vector<bfloat16>& data,
    const ttnn::Shape& shape,
    Layout layout,
    DataType dtype = DataType::BFLOAT16) {
    const MemoryConfig mem(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    return from_bf16(data, TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), mem))).to_device(&device, mem);
}

SlidingWindowConfig conv3x3(const Image& img, const std::array<uint32_t, 4>& padding, uint32_t num_cores) {
    return SlidingWindowConfig{
        .batch_size = img.n,
        .input_hw = {img.h, img.w},
        .window_hw = {3, 3},
        .stride_hw = {1, 1},
        .padding = padding,
        .num_cores_nhw = num_cores,
        .core_range_set = CoreRangeSet(CoreRange({0, 0}, {num_cores - 1, 0})),
        .snap_to_tile = true};
}

DeviceComputeKernelConfig compute_config(MeshDevice& device) {
    return init_device_compute_kernel_config(device.arch(), std::nullopt, MathFidelity::HiFi4, false, true);
}

Tensor halo(
    MeshDevice& device,
    const Tensor& input,
    const SlidingWindowConfig& config,
    const prim::HaloOptionalInputs& optional_inputs = {},
    uint32_t pad_val = 0) {
    return ttnn::halo(
        input,
        config,
        compute_config(device),
        pad_val,
        /*remote_read=*/false,
        /*transpose_mcast=*/false,
        /*is_out_tiled=*/true,
        /*config_tensors_in_dram=*/false,
        optional_inputs);
}

std::vector<float> run_halo(
    MeshDevice& device,
    const Tensor& input,
    const SlidingWindowConfig& config,
    const prim::HaloOptionalInputs& optional_inputs = {},
    uint32_t pad_val = 0) {
    const Tensor out = halo(device, input, config, optional_inputs, pad_val);
    // Each core's output shard has max_out_nsticks_per_core slots; only the first (its input range) are written.
    const auto boundaries = generate_shard_boundaries(config);
    const uint32_t slots = generate_max_out_nsticks_per_core(boundaries);
    const uint32_t c = input.logical_shape()[-1];
    const auto v = out.to_vector<bfloat16>();
    std::vector<float> written;
    for (uint32_t core = 0; core < boundaries.size(); ++core) {
        const auto& range = boundaries[core].input_range;
        const auto first = v.begin() + static_cast<size_t>(core) * slots * c;
        written.insert(written.end(), first, first + static_cast<size_t>(range.end - range.start + 1) * c);
    }
    return written;
}

}  // namespace

// The whole border of each image comes from the boundary (one device's tile of a spatially sharded image).
TEST_F(HaloTest, ExternalHaloEqualsTheBorderedImage) {
    auto& device = *device_;
    for (uint32_t c : {64u, 256u}) {
        constexpr uint32_t n = 2, h = 16, w = 32, num_cores = 4;
        Image framed = random_image(n, h + 2, w + 2, c, c);
        Image x = interior(framed);
        const auto boundary = border_sticks(framed);

        const auto expected = run_halo(
            device, height_sharded(device, framed, Layout::ROW_MAJOR, num_cores), conv3x3(framed, {}, num_cores));
        const auto actual = run_halo(
            device,
            height_sharded(device, x, Layout::TILE, num_cores),
            conv3x3(x, {1, 1, 1, 1}, num_cores),
            {.halo_boundary = dram(device, boundary, ttnn::Shape({1, 1, boundary.size() / c, c}), Layout::ROW_MAJOR)});
        EXPECT_EQ(actual, expected) << "c " << c;
    }
}

// Width slices of one device image, the way conv2d's DRAM width slicing cuts it: an edge slice takes the image's side
// border, an inner slice only the rows above and below its columns.
TEST_F(HaloTest, ExternalHaloWidthSlicesEqualTheBorderedImage) {
    auto& device = *device_;
    constexpr uint32_t h = 16, w = 32, c = 64, num_cores = 2;
    Image framed = random_image(1, h + 2, w + 2, c, 7);
    Image x = interior(framed);
    const auto border = border_sticks(framed);
    const Tensor boundary = dram(device, border, ttnn::Shape({1, 1, border.size() / c, c}), Layout::ROW_MAJOR);

    for (auto [c0, c1] : {std::array<uint32_t, 2>{0, 8}, {8, 24}, {24, 32}}) {
        const uint32_t pad_left = c0 == 0 ? 1 : 0;
        const uint32_t pad_right = c1 == w ? 1 : 0;
        Image expected_input = crop_columns(framed, c0 + 1 - pad_left, c1 + 1 + pad_right);
        Image slice = crop_columns(x, c0, c1);

        const auto expected = run_halo(
            device,
            height_sharded(device, expected_input, Layout::ROW_MAJOR, num_cores),
            conv3x3(expected_input, {}, num_cores));
        const auto actual = run_halo(
            device,
            height_sharded(device, slice, Layout::TILE, num_cores),
            conv3x3(slice, {1, 1, pad_left, pad_right}, num_cores),
            {.halo_boundary = boundary,
             .external_window = {.col_offset = c0, .boundary_w = w, .dev_pad_left = 1, .dev_pad_right = 1}});
        EXPECT_EQ(actual, expected) << "columns [" << c0 << ", " << c1 << ")";
    }
}

// 512 channels = 16 tiles per row, wider than one pack_untilize block.
TEST_F(HaloTest, PrologueAppliesAffineSiluToTheInputOnly) {
    auto& device = *device_;
    for (uint32_t c : {64u, 512u}) {
        constexpr uint32_t n = 1, h = 16, w = 32, num_cores = 4;
        Image x = random_image(n, h, w, c, c + 1);
        Image scale = random_image(1, 1, 1, c, c + 2);
        Image shift = random_image(1, 1, 1, c, c + 3);
        const Tensor input = height_sharded(device, x, Layout::TILE, num_cores);
        const auto config = conv3x3(x, {1, 1, 1, 1}, num_cores);

        // NaN padding marks which output sticks are padding.
        const auto plain = run_halo(device, input, config, {}, /*pad_val=*/0x7FC0);
        const auto actual = run_halo(
            device,
            input,
            config,
            {.prologue_scale = dram(device, scale.data, ttnn::Shape({1, 1, 1, c}), Layout::TILE),
             .prologue_shift = dram(device, shift.data, ttnn::Shape({1, 1, 1, c}), Layout::TILE)});
        ASSERT_EQ(actual.size(), plain.size());

        size_t failures = 0;
        for (size_t i = 0; i < plain.size(); ++i) {
            if (std::isnan(plain[i])) {
                failures += actual[i] != 0.0f;
                continue;
            }
            const float z = plain[i] * static_cast<float>(scale.data[i % c]) + static_cast<float>(shift.data[i % c]);
            const float golden = z / (1.0f + std::exp(-z));
            failures += std::abs(actual[i] - golden) > 0.02f + 0.02f * std::abs(golden);
        }
        EXPECT_EQ(failures, 0u) << "c " << c;
    }
}

// Both together: the prologue output inside, the boundary's sticks (already activated by the caller) in the border.
TEST_F(HaloTest, PrologueWithExternalHalo) {
    auto& device = *device_;
    constexpr uint32_t n = 1, h = 16, w = 32, c = 128, num_cores = 4;
    Image framed = random_image(n, h + 2, w + 2, c, 11);
    Image x = interior(framed);
    Image scale = random_image(1, 1, 1, c, 12);
    Image shift = random_image(1, 1, 1, c, 13);
    const auto border = border_sticks(framed);
    const Tensor input = height_sharded(device, x, Layout::TILE, num_cores);
    const auto config = conv3x3(x, {1, 1, 1, 1}, num_cores);
    const Tensor boundary = dram(device, border, ttnn::Shape({1, 1, border.size() / c, c}), Layout::ROW_MAJOR);
    const Tensor scale_t = dram(device, scale.data, ttnn::Shape({1, 1, 1, c}), Layout::TILE);
    const Tensor shift_t = dram(device, shift.data, ttnn::Shape({1, 1, 1, c}), Layout::TILE);

    const auto plain = run_halo(device, input, config);
    const auto external = run_halo(device, input, config, {.halo_boundary = boundary});
    const auto prologue = run_halo(device, input, config, {.prologue_scale = scale_t, .prologue_shift = shift_t});
    const auto both = run_halo(
        device, input, config, {.halo_boundary = boundary, .prologue_scale = scale_t, .prologue_shift = shift_t});
    ASSERT_EQ(both.size(), plain.size());
    // external - plain is the border alone (zero inside); prologue is zero on the border.
    std::vector<float> expected(plain.size());
    for (size_t i = 0; i < plain.size(); ++i) {
        expected[i] = prologue[i] + (external[i] - plain[i]);
    }
    EXPECT_EQ(both, expected);
}

// The prologue's scale / shift circular buffers use the bf16 output format, so a bfloat8_b input must be rejected, as
// must a shift without a scale.
TEST_F(HaloTest, PrologueRejectsUnsupportedInputs) {
    auto& device = *device_;
    constexpr uint32_t c = 64, num_cores = 4;
    Image x = random_image(1, 16, 32, c, 21);
    Image scale = random_image(1, 1, 1, c, 22);
    const auto config = conv3x3(x, {1, 1, 1, 1}, num_cores);
    const Tensor bf8_input = height_sharded(device, x, Layout::TILE, num_cores, DataType::BFLOAT8_B);
    const Tensor bf8_scale = dram(device, scale.data, ttnn::Shape({1, 1, 1, c}), Layout::TILE, DataType::BFLOAT8_B);
    EXPECT_ANY_THROW(halo(device, bf8_input, config, {.prologue_scale = bf8_scale, .prologue_shift = bf8_scale}));

    const Tensor input = height_sharded(device, x, Layout::TILE, num_cores);
    const Tensor scale_t = dram(device, scale.data, ttnn::Shape({1, 1, 1, c}), Layout::TILE);
    EXPECT_ANY_THROW(halo(device, input, config, {.prologue_shift = scale_t}));
}

}  // namespace ttnn::operations::sliding_window::test
