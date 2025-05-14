// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <fmt/base.h>
#include <stddef.h>
#include <tt-metalium/shape2d.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt_stl/span.hpp>
#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "common_tensor_test_utils.hpp"
#include <tt-metalium/core_coord.hpp>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <tt-metalium/math.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/tile.hpp>
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace {

void pretty_print_data_as_shards(
    const std::vector<float>& data, const Shape2D& shape, const Shape2D& shard_shape, const size_t char_count = 3) {
    TT_FATAL(
        data.size() == shape.height() * shape.width(),
        "Data size {} should be same as shape size {}",
        data.size(),
        shape.height() * shape.width());

    const auto [num_shards_height, last_shard_height, num_shards_width, last_shard_width] =
        tt::tt_metal::compute_shard_division_spec(shape, shard_shape);

    std::cout << "2D shape: " << shape << std::endl;
    for (size_t shard_height_idx = 0; shard_height_idx < num_shards_height; shard_height_idx++) {
        const auto num_shard_rows =
            shard_height_idx == num_shards_height - 1 ? last_shard_height : shard_shape.height();
        for (size_t shard_row_idx = 0; shard_row_idx < num_shard_rows; shard_row_idx++) {
            for (size_t shard_width_idx = 0; shard_width_idx < num_shards_width; shard_width_idx++) {
                const auto num_shard_cols =
                    shard_width_idx == num_shards_width - 1 ? last_shard_width : shard_shape.width();
                for (size_t shard_col_idx = 0; shard_col_idx < num_shard_cols; shard_col_idx++) {
                    const auto data_idx = (shard_height_idx * shard_shape.height() + shard_row_idx) * shape.width() +
                                          shard_width_idx * shard_shape.width() + shard_col_idx;
                    std::cout << fmt::format("{:>{}}", data[data_idx], char_count);
                    if (shard_col_idx < num_shard_cols - 1) {
                        std::cout << ", ";
                    } else if (shard_width_idx < num_shards_width - 1) {
                        std::cout << fmt::format("{:>{}}", "|", char_count);
                    }
                }
                std::cout << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

}  // namespace

namespace {
struct ShardWithAlignmentInputs {
    Shape shape;
    Shape2D logical_shard_shape;
    std::optional<Shape2D> physical_shard_shape;
    TensorMemoryLayout memory_layout;
    PageConfig page_config;
    std::vector<float> logical_data;
};

struct ShardWithAlignmentExpected {
    Shape2D physical_shard_shape;
    Shape2D physical_shape;
    std::vector<float> physical_data;
};

struct ShardWithAlignmentParams {
    ShardWithAlignmentInputs inputs;
    ShardWithAlignmentExpected expected;
};
}  // namespace
// namespace

class ShardWithAlignmentTests : public ::testing::TestWithParam<ShardWithAlignmentParams> {};

TEST_P(ShardWithAlignmentTests, LogicalToPhysical) {
    const auto& params = GetParam();

    // Only shard shapes and shard mode matters for this test
    auto shard_spec = params.inputs.physical_shard_shape.has_value()
                          ? ShardSpec(
                                CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{6, 6})),
                                params.inputs.logical_shard_shape,
                                params.inputs.physical_shard_shape.value(),
                                ShardOrientation::ROW_MAJOR)
                          : ShardSpec(
                                CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{6, 6})),
                                params.inputs.logical_shard_shape,
                                ShardOrientation::ROW_MAJOR,
                                ShardMode::LOGICAL);
    MemoryConfig memory_config{params.inputs.memory_layout, BufferType::L1, shard_spec};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, params.inputs.page_config, memory_config);
    auto tensor_spec = TensorSpec(params.inputs.shape, tensor_layout);

    auto logical_shard_shape = tensor_layout.get_logical_shard_shape();
    ASSERT_EQ(logical_shard_shape, params.inputs.logical_shard_shape);

    auto physical_shard_shape = tensor_layout.get_physical_shard_shape();
    ASSERT_EQ(physical_shard_shape, params.expected.physical_shard_shape);

    auto physical_shape = tensor_spec.physical_shape();
    ASSERT_EQ(physical_shape, params.expected.physical_shape);

    auto logical_data = params.inputs.logical_data;
    const auto& expected_physical_data = params.expected.physical_data;

    // Convert output physical data to row major (if necessary) for testing
    auto physical_data = tensor_impl::encode_tensor_data(std::move(logical_data), tensor_spec);
    if (tensor_spec.layout() == Layout::TILE) {
        // TODO: Fix convert_layout_tile_to_row_major to take in vector instead of buffer?
        physical_data = tensor_impl::convert_layout_tile_to_row_major(
            physical_shape, tensor_spec.tile(), tt::stl::make_const_span(physical_data));
    }

    // auto shape_2d = tensor_spec.logical_2d_shape();
    // pretty_print_data_as_shards(params.inputs.logical_data, shape_2d, logical_shard_shape);
    // pretty_print_data_as_shards(physical_data, physical_shape, physical_shard_shape);

    ASSERT_EQ(physical_data.size(), expected_physical_data.size());
    for (size_t i = 0; i < physical_data.size(); i++) {
        ASSERT_EQ(physical_data[i], expected_physical_data[i]);
    }
}

TEST_P(ShardWithAlignmentTests, PhysicalToLogical) {
    const auto& params = GetParam();

    // Only shard shapes and shard mode matters for this test
    // For grid size, set it to a 7x7 grid (fine as long as > num sharded)
    auto shard_spec = params.inputs.physical_shard_shape.has_value()
                          ? ShardSpec(
                                CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{6, 6})),
                                params.inputs.logical_shard_shape,
                                params.inputs.physical_shard_shape.value(),
                                ShardOrientation::ROW_MAJOR)
                          : ShardSpec(
                                CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{6, 6})),
                                params.inputs.logical_shard_shape,
                                ShardOrientation::ROW_MAJOR,
                                ShardMode::LOGICAL);
    MemoryConfig memory_config{params.inputs.memory_layout, BufferType::L1, shard_spec};
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, params.inputs.page_config, memory_config);
    auto tensor_spec = TensorSpec(params.inputs.shape, tensor_layout);

    auto logical_shard_shape = tensor_layout.get_logical_shard_shape();
    ASSERT_EQ(logical_shard_shape, params.inputs.logical_shard_shape);

    auto physical_shard_shape = tensor_layout.get_physical_shard_shape();
    ASSERT_EQ(physical_shard_shape, params.expected.physical_shard_shape);

    auto physical_shape = tensor_spec.physical_shape();
    ASSERT_EQ(physical_shape, params.expected.physical_shape);

    // Use expected value as input physical data
    auto physical_data = params.expected.physical_data;
    const auto& expected_data = params.inputs.logical_data;

    // Convert input physical data to TILE layout (if necessary) for testing
    if (tensor_spec.layout() == Layout::TILE) {
        // TODO: Fix convert_layout_row_major_to_tile to take in vector instead of buffer?
        physical_data = tensor_impl::convert_layout_row_major_to_tile(
            physical_shape, tensor_spec.tile(), tt::stl::make_const_span(physical_data));
    }
    auto logical_data = tensor_impl::decode_tensor_data(std::move(physical_data), tensor_spec);

    // auto shape_2d = tensor_spec.logical_2d_shape();
    // pretty_print_data_as_shards(params.expected.physical_data, physical_shape, physical_shard_shape);
    // pretty_print_data_as_shards(logical_data, shape_2d, logical_shard_shape);

    ASSERT_EQ(logical_data.size(), expected_data.size());
    for (size_t i = 0; i < logical_data.size(); i++) {
        ASSERT_EQ(logical_data[i], expected_data[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    TensorShardingTests,
    ShardWithAlignmentTests,
    // clang-format off
    ::testing::Values(
        // TILE interleaved is equivalent to setting logical shard size to full height and width
        // NOTE: This can also be interpreted as height sharded where we don't break apart height
        ShardWithAlignmentParams{
            ShardWithAlignmentInputs{
                .shape = Shape{1, 2, 15, 20},
                .logical_shard_shape = Shape2D{15, 20},
                .physical_shard_shape = std::nullopt,
                .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
                .page_config = PageConfig(Layout::TILE, Tile({16, 16})),
                .logical_data = {  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
                                  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                                  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
                                  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                                  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
                                 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                                 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
                                 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
                                 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
                                 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
                                 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
                                 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
                                 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259,
                                 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279,
                                 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299,

                                 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319,
                                 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339,
                                 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359,
                                 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379,
                                 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399,
                                 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419,
                                 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439,
                                 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459,
                                 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
                                 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499,
                                 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519,
                                 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539,
                                 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,
                                 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579,
                                 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599}
            },
            ShardWithAlignmentExpected{
                .physical_shard_shape = Shape2D{16, 32},
                .physical_shape = Shape2D{32, 32},
                .physical_data = {  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                   20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                   40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                   60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                   80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,

                                  300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                  580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0}
            }
        },
        // TILE height sharded is equivalent to setting logical shard width to full width
        // NOTE: This also supports logical shard height that breaks the height logically
        ShardWithAlignmentParams{
            ShardWithAlignmentInputs{
                .shape = Shape{1, 1, 15, 15},
                .logical_shard_shape = Shape2D{5, 15},
                .physical_shard_shape = std::nullopt,
                .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
                .page_config = PageConfig(Layout::TILE, Tile({16, 16})),
                .logical_data = {  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
                                  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
                                  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
                                  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
                                  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,

                                  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
                                  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104,
                                 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                                 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
                                 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,

                                 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
                                 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
                                 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
                                 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
                                 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224}
            },
            ShardWithAlignmentExpected{
                .physical_shard_shape = Shape2D{16, 16},
                .physical_shape = Shape2D{48, 16},
                .physical_data = {  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,   0,
                                   15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,   0,
                                   30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,   0,
                                   45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,   0,
                                   60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,

                                   75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,   0,
                                   90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104,   0,
                                  105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,   0,
                                  120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,   0,
                                  135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,

                                  150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,   0,
                                  165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,   0,
                                  180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,   0,
                                  195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,   0,
                                  210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0}
            }
        },
        // TILE width sharded is equivalent to setting logical shard height to full flattened tensor height
        ShardWithAlignmentParams{
            ShardWithAlignmentInputs{
                .shape = Shape{1, 2, 5, 20},
                .logical_shard_shape = Shape2D{10, 10},
                .physical_shard_shape = std::nullopt,
                .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
                .page_config = PageConfig(Layout::TILE, Tile({16, 16})),
                .logical_data = { 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  /**/  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
                                 20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  /**/  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                                 40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  /**/  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
                                 60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  /**/  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                                 80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  /**/  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
                                100, 101, 102, 103, 104, 105, 106, 107, 108, 109,  /**/ 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                                120, 121, 122, 123, 124, 125, 126, 127, 128, 129,  /**/ 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
                                140, 141, 142, 143, 144, 145, 146, 147, 148, 149,  /**/ 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
                                160, 161, 162, 163, 164, 165, 166, 167, 168, 169,  /**/ 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
                                180, 181, 182, 183, 184, 185, 186, 187, 188, 189,  /**/ 190, 191, 192, 193, 194, 195, 196, 197, 198, 199}
            },
            ShardWithAlignmentExpected{
                .physical_shard_shape = Shape2D{16, 16},
                .physical_shape = Shape2D{16, 32},
                .physical_data = {  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   0,   0,   0,   0,   0,   0,  /**/  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,   0,   0,   0,   0,   0,   0,
                                   20,  21,  22,  23,  24,  25,  26,  27,  28,  29,   0,   0,   0,   0,   0,   0,  /**/  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,   0,   0,   0,   0,   0,   0,
                                   40,  41,  42,  43,  44,  45,  46,  47,  48,  49,   0,   0,   0,   0,   0,   0,  /**/  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,   0,   0,   0,   0,   0,   0,
                                   60,  61,  62,  63,  64,  65,  66,  67,  68,  69,   0,   0,   0,   0,   0,   0,  /**/  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,   0,   0,   0,   0,   0,   0,
                                   80,  81,  82,  83,  84,  85,  86,  87,  88,  89,   0,   0,   0,   0,   0,   0,  /**/  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,   0,   0,   0,   0,   0,   0,
                                  100, 101, 102, 103, 104, 105, 106, 107, 108, 109,   0,   0,   0,   0,   0,   0,  /**/ 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,   0,   0,   0,   0,   0,   0,
                                  120, 121, 122, 123, 124, 125, 126, 127, 128, 129,   0,   0,   0,   0,   0,   0,  /**/ 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,   0,   0,   0,   0,   0,   0,
                                  140, 141, 142, 143, 144, 145, 146, 147, 148, 149,   0,   0,   0,   0,   0,   0,  /**/ 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,   0,   0,   0,   0,   0,   0,
                                  160, 161, 162, 163, 164, 165, 166, 167, 168, 169,   0,   0,   0,   0,   0,   0,  /**/ 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,   0,   0,   0,   0,   0,   0,
                                  180, 181, 182, 183, 184, 185, 186, 187, 188, 189,   0,   0,   0,   0,   0,   0,  /**/ 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0}
            }
        },
        // TILE block sharded with alignment up to nearest page instead of physical shard shape in last row/col
        ShardWithAlignmentParams{
            ShardWithAlignmentInputs{
                .shape = Shape{1, 1, 30, 30},
                .logical_shard_shape = Shape2D{18, 20},
                .physical_shard_shape = std::nullopt,
                .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
                .page_config = PageConfig(Layout::TILE, Tile({16, 16})),
                .logical_data = {  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  /**/  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
                                  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  /**/  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
                                  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  /**/  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
                                  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,  /**/ 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                                 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,  /**/ 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
                                 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,  /**/ 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
                                 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,  /**/ 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
                                 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229,  /**/ 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
                                 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259,  /**/ 260, 261, 262, 263, 264, 265, 266, 267, 268, 269,
                                 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289,  /**/ 290, 291, 292, 293, 294, 295, 296, 297, 298, 299,
                                 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319,  /**/ 320, 321, 322, 323, 324, 325, 326, 327, 328, 329,
                                 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,  /**/ 350, 351, 352, 353, 354, 355, 356, 357, 358, 359,
                                 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379,  /**/ 380, 381, 382, 383, 384, 385, 386, 387, 388, 389,
                                 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409,  /**/ 410, 411, 412, 413, 414, 415, 416, 417, 418, 419,
                                 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439,  /**/ 440, 441, 442, 443, 444, 445, 446, 447, 448, 449,
                                 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469,  /**/ 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
                                 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499,  /**/ 500, 501, 502, 503, 504, 505, 506, 507, 508, 509,
                                 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529,  /**/ 530, 531, 532, 533, 534, 535, 536, 537, 538, 539,

                                 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,  /**/ 560, 561, 562, 563, 564, 565, 566, 567, 568, 569,
                                 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589,  /**/ 590, 591, 592, 593, 594, 595, 596, 597, 598, 599,
                                 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619,  /**/ 620, 621, 622, 623, 624, 625, 626, 627, 628, 629,
                                 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649,  /**/ 650, 651, 652, 653, 654, 655, 656, 657, 658, 659,
                                 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679,  /**/ 680, 681, 682, 683, 684, 685, 686, 687, 688, 689,
                                 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709,  /**/ 710, 711, 712, 713, 714, 715, 716, 717, 718, 719,
                                 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739,  /**/ 740, 741, 742, 743, 744, 745, 746, 747, 748, 749,
                                 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769,  /**/ 770, 771, 772, 773, 774, 775, 776, 777, 778, 779,
                                 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799,  /**/ 800, 801, 802, 803, 804, 805, 806, 807, 808, 809,
                                 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829,  /**/ 830, 831, 832, 833, 834, 835, 836, 837, 838, 839,
                                 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859,  /**/ 860, 861, 862, 863, 864, 865, 866, 867, 868, 869,
                                 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889,  /**/ 890, 891, 892, 893, 894, 895, 896, 897, 898, 899}
            },
            ShardWithAlignmentExpected{
                .physical_shard_shape = Shape2D{32, 32},
                .physical_shape = Shape2D{48, 48},
                .physical_data = {  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,   0,   0,   0,   0,   0,   0,
                                   30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,   0,   0,   0,   0,   0,   0,
                                   60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,   0,   0,   0,   0,   0,   0,
                                   90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,   0,   0,   0,   0,   0,   0,
                                  120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,   0,   0,   0,   0,   0,   0,
                                  150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,   0,   0,   0,   0,   0,   0,
                                  180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,   0,   0,   0,   0,   0,   0,
                                  210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,   0,   0,   0,   0,   0,   0,
                                  240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 260, 261, 262, 263, 264, 265, 266, 267, 268, 269,   0,   0,   0,   0,   0,   0,
                                  270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 290, 291, 292, 293, 294, 295, 296, 297, 298, 299,   0,   0,   0,   0,   0,   0,
                                  300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 320, 321, 322, 323, 324, 325, 326, 327, 328, 329,   0,   0,   0,   0,   0,   0,
                                  330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 350, 351, 352, 353, 354, 355, 356, 357, 358, 359,   0,   0,   0,   0,   0,   0,
                                  360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 380, 381, 382, 383, 384, 385, 386, 387, 388, 389,   0,   0,   0,   0,   0,   0,
                                  390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 410, 411, 412, 413, 414, 415, 416, 417, 418, 419,   0,   0,   0,   0,   0,   0,
                                  420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 440, 441, 442, 443, 444, 445, 446, 447, 448, 449,   0,   0,   0,   0,   0,   0,
                                  450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,   0,   0,   0,   0,   0,   0,
                                  480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 500, 501, 502, 503, 504, 505, 506, 507, 508, 509,   0,   0,   0,   0,   0,   0,
                                  510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 530, 531, 532, 533, 534, 535, 536, 537, 538, 539,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,

                                  540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 560, 561, 562, 563, 564, 565, 566, 567, 568, 569,   0,   0,   0,   0,   0,   0,
                                  570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 590, 591, 592, 593, 594, 595, 596, 597, 598, 599,   0,   0,   0,   0,   0,   0,
                                  600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 620, 621, 622, 623, 624, 625, 626, 627, 628, 629,   0,   0,   0,   0,   0,   0,
                                  630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 650, 651, 652, 653, 654, 655, 656, 657, 658, 659,   0,   0,   0,   0,   0,   0,
                                  660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 680, 681, 682, 683, 684, 685, 686, 687, 688, 689,   0,   0,   0,   0,   0,   0,
                                  690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 710, 711, 712, 713, 714, 715, 716, 717, 718, 719,   0,   0,   0,   0,   0,   0,
                                  720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 740, 741, 742, 743, 744, 745, 746, 747, 748, 749,   0,   0,   0,   0,   0,   0,
                                  750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 770, 771, 772, 773, 774, 775, 776, 777, 778, 779,   0,   0,   0,   0,   0,   0,
                                  780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 800, 801, 802, 803, 804, 805, 806, 807, 808, 809,   0,   0,   0,   0,   0,   0,
                                  810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 830, 831, 832, 833, 834, 835, 836, 837, 838, 839,   0,   0,   0,   0,   0,   0,
                                  840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 860, 861, 862, 863, 864, 865, 866, 867, 868, 869,   0,   0,   0,   0,   0,   0,
                                  870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/ 890, 891, 892, 893, 894, 895, 896, 897, 898, 899,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0}
            }
        },
        // RM interleaved is equivalent to setting logical shard size to 1 by 1
        ShardWithAlignmentParams{
            ShardWithAlignmentInputs{
                .shape = Shape{1, 2, 5, 1},
                .logical_shard_shape = Shape2D{1, 1},
                .physical_shard_shape = std::nullopt,
                .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
                .page_config = PageConfig(Layout::ROW_MAJOR),
                .logical_data = {  0,

                                   1,

                                   2,

                                   3,

                                   4,

                                   5,

                                   6,

                                   7,

                                   8,

                                   9}
            },
            ShardWithAlignmentExpected{
                .physical_shard_shape = Shape2D{1, 1},
                .physical_shape = Shape2D{10, 1},
                .physical_data = {  0,

                                    1,

                                    2,

                                    3,

                                    4,

                                    5,

                                    6,

                                    7,

                                    8,

                                    9}
            }
        },
        // RM height sharded with padding along width to arbitrary shard width
        ShardWithAlignmentParams{
            ShardWithAlignmentInputs{
                .shape = Shape{1, 2, 5, 1},
                .logical_shard_shape = Shape2D{3, 1},
                .physical_shard_shape = Shape2D{3, 4},
                .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
                .page_config = PageConfig(Layout::ROW_MAJOR),
                .logical_data = {  0,
                                   1,
                                   2,

                                   3,
                                   4,
                                   5,

                                   6,
                                   7,
                                   8,

                                   9}
            },
            ShardWithAlignmentExpected{
                .physical_shard_shape = Shape2D{3, 4},
                .physical_shape = Shape2D{12, 4},
                .physical_data = {  0,   0,   0,   0,
                                    1,   0,   0,   0,
                                    2,   0,   0,   0,

                                    3,   0,   0,   0,
                                    4,   0,   0,   0,
                                    5,   0,   0,   0,

                                    6,   0,   0,   0,
                                    7,   0,   0,   0,
                                    8,   0,   0,   0,

                                    9,   0,   0,   0,
                                    0,   0,   0,   0,
                                    0,   0,   0,   0}
            }
        },
        // RM width sharded with alignment up to nearest page in last shard (in RM sharded, it is set to physical shard width)
        ShardWithAlignmentParams{
            ShardWithAlignmentInputs{
                .shape = Shape{1, 2, 5, 10},
                .logical_shard_shape = Shape2D{10, 3},
                .physical_shard_shape = std::nullopt,
                .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
                .page_config = PageConfig(Layout::ROW_MAJOR),
                .logical_data = {  0,   1,   2,  /**/   3,   4,   5,  /**/   6,   7,   8,  /**/   9,
                                  10,  11,  12,  /**/  13,  14,  15,  /**/  16,  17,  18,  /**/  19,
                                  20,  21,  22,  /**/  23,  24,  25,  /**/  26,  27,  28,  /**/  29,
                                  30,  31,  32,  /**/  33,  34,  35,  /**/  36,  37,  38,  /**/  39,
                                  40,  41,  42,  /**/  43,  44,  45,  /**/  46,  47,  48,  /**/  49,
                                  50,  51,  52,  /**/  53,  54,  55,  /**/  56,  57,  58,  /**/  59,
                                  60,  61,  62,  /**/  63,  64,  65,  /**/  66,  67,  68,  /**/  69,
                                  70,  71,  72,  /**/  73,  74,  75,  /**/  76,  77,  78,  /**/  79,
                                  80,  81,  82,  /**/  83,  84,  85,  /**/  86,  87,  88,  /**/  89,
                                  90,  91,  92,  /**/  93,  94,  95,  /**/  96,  97,  98,  /**/  99}
            },
            ShardWithAlignmentExpected{
                .physical_shard_shape = Shape2D{10, 3},
                .physical_shape = Shape2D{10, 12},
                .physical_data = {  0,   1,   2,  /**/   3,   4,   5,  /**/   6,   7,   8,  /**/   9,   0,   0,
                                   10,  11,  12,  /**/  13,  14,  15,  /**/  16,  17,  18,  /**/  19,   0,   0,
                                   20,  21,  22,  /**/  23,  24,  25,  /**/  26,  27,  28,  /**/  29,   0,   0,
                                   30,  31,  32,  /**/  33,  34,  35,  /**/  36,  37,  38,  /**/  39,   0,   0,
                                   40,  41,  42,  /**/  43,  44,  45,  /**/  46,  47,  48,  /**/  49,   0,   0,
                                   50,  51,  52,  /**/  53,  54,  55,  /**/  56,  57,  58,  /**/  59,   0,   0,
                                   60,  61,  62,  /**/  63,  64,  65,  /**/  66,  67,  68,  /**/  69,   0,   0,
                                   70,  71,  72,  /**/  73,  74,  75,  /**/  76,  77,  78,  /**/  79,   0,   0,
                                   80,  81,  82,  /**/  83,  84,  85,  /**/  86,  87,  88,  /**/  89,   0,   0,
                                   90,  91,  92,  /**/  93,  94,  95,  /**/  96,  97,  98,  /**/  99,   0,   0}
            }
        },
        // RM width sharded with padding along width to arbitrary shard width
        ShardWithAlignmentParams{
            ShardWithAlignmentInputs{
                .shape = Shape{1, 2, 5, 10},
                .logical_shard_shape = Shape2D{10, 3},
                .physical_shard_shape = Shape2D{10, 4},
                .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
                .page_config = PageConfig(Layout::ROW_MAJOR),
                .logical_data = {  0,   1,   2,  /**/   3,   4,   5,  /**/   6,   7,   8,  /**/   9,
                                  10,  11,  12,  /**/  13,  14,  15,  /**/  16,  17,  18,  /**/  19,
                                  20,  21,  22,  /**/  23,  24,  25,  /**/  26,  27,  28,  /**/  29,
                                  30,  31,  32,  /**/  33,  34,  35,  /**/  36,  37,  38,  /**/  39,
                                  40,  41,  42,  /**/  43,  44,  45,  /**/  46,  47,  48,  /**/  49,
                                  50,  51,  52,  /**/  53,  54,  55,  /**/  56,  57,  58,  /**/  59,
                                  60,  61,  62,  /**/  63,  64,  65,  /**/  66,  67,  68,  /**/  69,
                                  70,  71,  72,  /**/  73,  74,  75,  /**/  76,  77,  78,  /**/  79,
                                  80,  81,  82,  /**/  83,  84,  85,  /**/  86,  87,  88,  /**/  89,
                                  90,  91,  92,  /**/  93,  94,  95,  /**/  96,  97,  98,  /**/  99}
            },
            ShardWithAlignmentExpected{
                .physical_shard_shape = Shape2D{10, 4},
                .physical_shape = Shape2D{10, 16},
                .physical_data = {  0,   1,   2,   0,  /**/   3,   4,   5,   0,  /**/   6,   7,   8,   0,  /**/   9,   0,   0,   0,
                                   10,  11,  12,   0,  /**/  13,  14,  15,   0,  /**/  16,  17,  18,   0,  /**/  19,   0,   0,   0,
                                   20,  21,  22,   0,  /**/  23,  24,  25,   0,  /**/  26,  27,  28,   0,  /**/  29,   0,   0,   0,
                                   30,  31,  32,   0,  /**/  33,  34,  35,   0,  /**/  36,  37,  38,   0,  /**/  39,   0,   0,   0,
                                   40,  41,  42,   0,  /**/  43,  44,  45,   0,  /**/  46,  47,  48,   0,  /**/  49,   0,   0,   0,
                                   50,  51,  52,   0,  /**/  53,  54,  55,   0,  /**/  56,  57,  58,   0,  /**/  59,   0,   0,   0,
                                   60,  61,  62,   0,  /**/  63,  64,  65,   0,  /**/  66,  67,  68,   0,  /**/  69,   0,   0,   0,
                                   70,  71,  72,   0,  /**/  73,  74,  75,   0,  /**/  76,  77,  78,   0,  /**/  79,   0,   0,   0,
                                   80,  81,  82,   0,  /**/  83,  84,  85,   0,  /**/  86,  87,  88,   0,  /**/  89,   0,   0,   0,
                                   90,  91,  92,   0,  /**/  93,  94,  95,   0,  /**/  96,  97,  98,   0,  /**/  99,   0,   0,   0}
            }
        },
        // Arbitrary logical shard shape and alignment to stress test edges with padding
        ShardWithAlignmentParams{
            ShardWithAlignmentInputs{
                .shape = Shape{1, 2, 10, 10},
                .logical_shard_shape = Shape2D{3, 4},
                .physical_shard_shape = Shape2D{5, 7},
                .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
                .page_config = PageConfig(Layout::ROW_MAJOR),
                .logical_data = {  0,   1,   2,   3,  /**/   4,   5,   6,   7,  /**/   8,   9,
                                  10,  11,  12,  13,  /**/  14,  15,  16,  17,  /**/  18,  19,
                                  20,  21,  22,  23,  /**/  24,  25,  26,  27,  /**/  28,  29,

                                  30,  31,  32,  33,  /**/  34,  35,  36,  37,  /**/  38,  39,
                                  40,  41,  42,  43,  /**/  44,  45,  46,  47,  /**/  48,  49,
                                  50,  51,  52,  53,  /**/  54,  55,  56,  57,  /**/  58,  59,

                                  60,  61,  62,  63,  /**/  64,  65,  66,  67,  /**/  68,  69,
                                  70,  71,  72,  73,  /**/  74,  75,  76,  77,  /**/  78,  79,
                                  80,  81,  82,  83,  /**/  84,  85,  86,  87,  /**/  88,  89,

                                  90,  91,  92,  93,  /**/  94,  95,  96,  97,  /**/  98,  99,
                                 100, 101, 102, 103,  /**/ 104, 105, 106, 107,  /**/ 108, 109,
                                 110, 111, 112, 113,  /**/ 114, 115, 116, 117,  /**/ 118, 119,

                                 120, 121, 122, 123,  /**/ 124, 125, 126, 127,  /**/ 128, 129,
                                 130, 131, 132, 133,  /**/ 134, 135, 136, 137,  /**/ 138, 139,
                                 140, 141, 142, 143,  /**/ 144, 145, 146, 147,  /**/ 148, 149,

                                 150, 151, 152, 153,  /**/ 154, 155, 156, 157,  /**/ 158, 159,
                                 160, 161, 162, 163,  /**/ 164, 165, 166, 167,  /**/ 168, 169,
                                 170, 171, 172, 173,  /**/ 174, 175, 176, 177,  /**/ 178, 179,

                                 180, 181, 182, 183,  /**/ 184, 185, 186, 187,  /**/ 188, 189,
                                 190, 191, 192, 193,  /**/ 194, 195, 196, 197,  /**/ 198, 199}
            },
            ShardWithAlignmentExpected{
                .physical_shard_shape = Shape2D{5, 7},
                .physical_shape = Shape2D{35, 21},
                .physical_data = { 0,   1,   2,   3,   0,   0,   0,  /**/   4,   5,   6,   7,   0,   0,   0,  /**/   8,   9,   0,   0,   0,   0,   0,
                                  10,  11,  12,  13,   0,   0,   0,  /**/  14,  15,  16,  17,   0,   0,   0,  /**/  18,  19,   0,   0,   0,   0,   0,
                                  20,  21,  22,  23,   0,   0,   0,  /**/  24,  25,  26,  27,   0,   0,   0,  /**/  28,  29,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,

                                  30,  31,  32,  33,   0,   0,   0,  /**/  34,  35,  36,  37,   0,   0,   0,  /**/  38,  39,   0,   0,   0,   0,   0,
                                  40,  41,  42,  43,   0,   0,   0,  /**/  44,  45,  46,  47,   0,   0,   0,  /**/  48,  49,   0,   0,   0,   0,   0,
                                  50,  51,  52,  53,   0,   0,   0,  /**/  54,  55,  56,  57,   0,   0,   0,  /**/  58,  59,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,

                                  60,  61,  62,  63,   0,   0,   0,  /**/  64,  65,  66,  67,   0,   0,   0,  /**/  68,  69,   0,   0,   0,   0,   0,
                                  70,  71,  72,  73,   0,   0,   0,  /**/  74,  75,  76,  77,   0,   0,   0,  /**/  78,  79,   0,   0,   0,   0,   0,
                                  80,  81,  82,  83,   0,   0,   0,  /**/  84,  85,  86,  87,   0,   0,   0,  /**/  88,  89,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,

                                  90,  91,  92,  93,   0,   0,   0,  /**/  94,  95,  96,  97,   0,   0,   0,  /**/  98,  99,   0,   0,   0,   0,   0,
                                 100, 101, 102, 103,   0,   0,   0,  /**/ 104, 105, 106, 107,   0,   0,   0,  /**/ 108, 109,   0,   0,   0,   0,   0,
                                 110, 111, 112, 113,   0,   0,   0,  /**/ 114, 115, 116, 117,   0,   0,   0,  /**/ 118, 119,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,

                                 120, 121, 122, 123,   0,   0,   0,  /**/ 124, 125, 126, 127,   0,   0,   0,  /**/ 128, 129,   0,   0,   0,   0,   0,
                                 130, 131, 132, 133,   0,   0,   0,  /**/ 134, 135, 136, 137,   0,   0,   0,  /**/ 138, 139,   0,   0,   0,   0,   0,
                                 140, 141, 142, 143,   0,   0,   0,  /**/ 144, 145, 146, 147,   0,   0,   0,  /**/ 148, 149,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,

                                 150, 151, 152, 153,   0,   0,   0,  /**/ 154, 155, 156, 157,   0,   0,   0,  /**/ 158, 159,   0,   0,   0,   0,   0,
                                 160, 161, 162, 163,   0,   0,   0,  /**/ 164, 165, 166, 167,   0,   0,   0,  /**/ 168, 169,   0,   0,   0,   0,   0,
                                 170, 171, 172, 173,   0,   0,   0,  /**/ 174, 175, 176, 177,   0,   0,   0,  /**/ 178, 179,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,

                                 180, 181, 182, 183,   0,   0,   0,  /**/ 184, 185, 186, 187,   0,   0,   0,  /**/ 188, 189,   0,   0,   0,   0,   0,
                                 190, 191, 192, 193,   0,   0,   0,  /**/ 194, 195, 196, 197,   0,   0,   0,  /**/ 198, 199,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0,  /**/   0,   0,   0,   0,   0,   0,   0}
            }
        }
    ) // Values
    // clang-format on
);

namespace {
const CoreCoord grid_size{8, 7};

struct CreateShardedTensorWithAlignmentInputs {
    Shape shape;
    DataType data_type;
    PageConfig page_config;
    MemoryConfig memory_config;
};

struct CreateShardedTensorWithAlignmentExpected {
    Shape2D physical_shape;
};

struct CreateShardedTensorWithAlignmentParams {
    CreateShardedTensorWithAlignmentInputs inputs;
    CreateShardedTensorWithAlignmentExpected expected;
};
}  // namespace

class CreateShardedTensorWithAlignmentTests
    : public ttnn::TTNNFixtureWithDevice,
      public ::testing::WithParamInterface<CreateShardedTensorWithAlignmentParams> {};

TEST_P(CreateShardedTensorWithAlignmentTests, AllocateTensor) {
    const auto& params = GetParam();
    const auto& input_shape = params.inputs.shape;

    TensorLayout layout(params.inputs.data_type, params.inputs.page_config, params.inputs.memory_config);

    test_utils::test_tensor_on_device(input_shape, layout, device_);

    EXPECT_EQ(layout.compute_physical_shape(input_shape), params.expected.physical_shape);
}

INSTANTIATE_TEST_SUITE_P(
    TensorShardingTests,
    CreateShardedTensorWithAlignmentTests,
    // clang-format off
    ::testing::Values(
        //////////////////////////////////////////////////////////////////////////////////////////
        // EXAMPLE 1: TILE tensor with different representation for height sharded / interleaved
        //////////////////////////////////////////////////////////////////////////////////////////
        // Example 1a: Logical shard shape + alignment after
        // - Along height: 48 * 56 / 48 is 56 shards; 56 * 64 = 3584
        CreateShardedTensorWithAlignmentParams{
            CreateShardedTensorWithAlignmentInputs{
                .shape = Shape{1, 48, 56, 32},
                .data_type = DataType::BFLOAT16,
                .page_config = PageConfig(Layout::TILE),
                .memory_config =
                    MemoryConfig{
                        TensorMemoryLayout::HEIGHT_SHARDED,
                        BufferType::L1,
                        ShardSpec{
                            num_cores_to_corerangeset(tt::div_up(48 * 56, 48), grid_size, /*row_wise=*/true),
                            {48, 32},
                            ShardOrientation::ROW_MAJOR,
                            ShardMode::LOGICAL
                        }
                    }
            },
            CreateShardedTensorWithAlignmentExpected{.physical_shape = Shape2D{3584, 32}}
        },
        // Example 1b: Logical shard shape that is already aligned
        // NOTE: If ShardMode::PHYSICAL, it expects height 56 to be padded up to 64
        // - Along height: 48 * 56 / 64 is 42 shards; 42 * 64 = 2688
        CreateShardedTensorWithAlignmentParams{
            CreateShardedTensorWithAlignmentInputs{
                .shape = Shape{1, 48, 56, 32},
                .data_type = DataType::BFLOAT16,
                .page_config = PageConfig(Layout::TILE),
                .memory_config =
                    MemoryConfig{
                        TensorMemoryLayout::HEIGHT_SHARDED,
                        BufferType::L1,
                        ShardSpec{
                            num_cores_to_corerangeset(tt::div_up(48 * 56, 64), grid_size, /*row_wise=*/true),
                            {64, 32},
                            ShardOrientation::ROW_MAJOR,
                            ShardMode::LOGICAL
                        }
                    }
            },
            CreateShardedTensorWithAlignmentExpected{.physical_shape = Shape2D{2688, 32}}
        },
        // Example 1c: For interleaved, we treat entire height/width as "logical shard shape" for calculations
        // 48 "shards" with 56 aligned to 32 for tile alignment; 48 * 64 = 3072
        CreateShardedTensorWithAlignmentParams{
            CreateShardedTensorWithAlignmentInputs{
                .shape = Shape{1, 48, 56, 32},
                .data_type = DataType::BFLOAT16,
                .page_config = PageConfig(Layout::TILE),
                .memory_config =
                    MemoryConfig{
                        TensorMemoryLayout::INTERLEAVED,
                        BufferType::DRAM,
                        std::nullopt
                    }
            },
            CreateShardedTensorWithAlignmentExpected{.physical_shape = Shape2D{3072, 32}}
        },
        /////////////////////////////////////////////////////////////////////////////////////////////////////
        // EXAMPLE 2: ROW_MAJOR tensor with different representation for width sharded / interleaved
        // - In this example, (shard) width alignment is 1 because it's row major
        /////////////////////////////////////////////////////////////////////////////////////////////////////
        // Example 2a: Logical shard shape that is already aligned
        // NOTE: ShardMode::PHYSICAL is equivalent in this case
        // - Along width: 5 / 1 is 5 shards; 5 * 1 = 5
        CreateShardedTensorWithAlignmentParams{
            CreateShardedTensorWithAlignmentInputs{
                .shape = Shape{1, 2, 10, 5},
                .data_type = DataType::UINT8,
                .page_config = PageConfig(Layout::ROW_MAJOR),
                .memory_config =
                    MemoryConfig{
                        TensorMemoryLayout::WIDTH_SHARDED,
                        BufferType::L1,
                        ShardSpec{
                                num_cores_to_corerangeset(tt::div_up(5, 1), grid_size, /*row_wise=*/true),
                                {20, 1},
                                ShardOrientation::ROW_MAJOR,
                                ShardMode::LOGICAL
                            }
                    }
            },
            CreateShardedTensorWithAlignmentExpected{.physical_shape = Shape2D{20, 5}}
        },
        // Example 2b: Logical shard shape that is already aligned
        // NOTE: ShardMode::PHYSICAL is equivalent in this case
        // - Along width: 8 / 4 is 2 shards; 2 * 4 = 8
        CreateShardedTensorWithAlignmentParams{
            CreateShardedTensorWithAlignmentInputs{
                .shape = Shape{1, 2, 10, 8},
                .data_type = DataType::UINT8,
                .page_config = PageConfig(Layout::ROW_MAJOR),
                .memory_config =
                    MemoryConfig{
                        TensorMemoryLayout::WIDTH_SHARDED,
                        BufferType::L1,
                        ShardSpec{
                            num_cores_to_corerangeset(tt::div_up(5, 4), grid_size, /*row_wise=*/true),
                            {20, 4},
                            ShardOrientation::ROW_MAJOR,
                            ShardMode::LOGICAL
                        }
                    }
            },
            CreateShardedTensorWithAlignmentExpected{.physical_shape = Shape2D{20, 8}}
        },
        // Example 2c: For interleaved, we treat entire height/width as "logical shard shape" for calculations
        // 20 "shards" with 5 aligned to 1
        CreateShardedTensorWithAlignmentParams{
            CreateShardedTensorWithAlignmentInputs{
                .shape = Shape{1, 2, 10, 5},
                .data_type = DataType::UINT8,
                .page_config = PageConfig(Layout::ROW_MAJOR),
                .memory_config =
                    MemoryConfig{
                        TensorMemoryLayout::INTERLEAVED,
                        BufferType::L1,
                        std::nullopt
                    }
            },
            CreateShardedTensorWithAlignmentExpected{.physical_shape = Shape2D{20, 5}}
        },
        ////////////////////////////////////////////////////////////////////
        // EXAMPLE 3: Interesting cases with custom (legal) shard alignment
        ////////////////////////////////////////////////////////////////////
        // Example 3a: TILE block sharded tensor with shard alignment of 3 * 16 along the width
        // - Along height: 8 * 36 / 48 is 6 shards; 6 * 64 = 384
        // - Along width: 32 / 10 is 4 shards; 4 * custom alignment 48 = 192 (48 % 16 == 0, so it is legal)
        CreateShardedTensorWithAlignmentParams{
            CreateShardedTensorWithAlignmentInputs{
                .shape = Shape{1, 8, 36, 32},
                .data_type = DataType::BFLOAT8_B,
                .page_config = PageConfig(Layout::TILE, Tile({32, 16})),
                .memory_config =
                    MemoryConfig{
                        TensorMemoryLayout::BLOCK_SHARDED,
                        BufferType::L1,
                        ShardSpec{
                                CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{3, 5})),
                                {48, 10},
                                {64, 48},
                                ShardOrientation::ROW_MAJOR
                            }
                    }
            },
            CreateShardedTensorWithAlignmentExpected{.physical_shape = Shape2D{384, 192}}
        },
        // Example 3b: ROW_MAJOR block sharded tensor with 2 and 1 extra rows and col per shard, respectively
        // - Along height: 2 * 10 / 5 is 4 shards; 4 * custom alignment 7 = 28 (no restriction on height alignment for ROW_MAJOR)
        // - Along width: 5 / 2 is 3 shards; 3 * custom alignment 3 = 9 (alignment on width can be arbitrary because UINT32 is already 4-byte aligned)
        CreateShardedTensorWithAlignmentParams{
            CreateShardedTensorWithAlignmentInputs{
                .shape = Shape{1, 2, 10, 5},
                .data_type = DataType::UINT32,
                .page_config = PageConfig(Layout::ROW_MAJOR),
                .memory_config =
                    MemoryConfig{
                        TensorMemoryLayout::BLOCK_SHARDED,
                        BufferType::L1,
                        ShardSpec{
                                CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{2, 3})),
                                {5, 2},
                                {7, 3},
                                ShardOrientation::ROW_MAJOR
                            }
                    }
            },
            CreateShardedTensorWithAlignmentExpected{.physical_shape = Shape2D{28, 9}}
        }
    )  // Values
    // clang-format on
);

namespace {
struct IllegalShardSpecParams {
    Shape shape;
    PageConfig page_config;
    MemoryConfig memory_config;
    std::string expected_err_msg;
};
}  // namespace

class IllegalTensorLayoutCreationTests : public ::testing::TestWithParam<IllegalShardSpecParams> {};

TEST_P(IllegalTensorLayoutCreationTests, ExpectFailAndCheckErrMsg) {
    const auto& params = GetParam();

    EXPECT_THAT(
        std::function<void()>([&params]() {
            auto tensor_layout = TensorLayout(DataType::BFLOAT16, params.page_config, params.memory_config);
        }),
        ThrowsMessage<std::runtime_error>(::testing::HasSubstr(params.expected_err_msg)));
}

INSTANTIATE_TEST_SUITE_P(
    TensorShardingTests,
    IllegalTensorLayoutCreationTests,
    // clang-format off
    ::testing::Values(
        // Physical shard shape is not tile sized
        IllegalShardSpecParams{
            .shape = Shape{100, 20},
            .page_config = PageConfig(Layout::TILE, Tile({32, 16})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::HEIGHT_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            num_cores_to_corerangeset(10, grid_size, /*row_wise=*/true),
                            {10, 20},
                            ShardOrientation::ROW_MAJOR,
                        }
                },
            .expected_err_msg = "Physical shard shape (10, 20) must be tile {32, 16} sized!"
        },
        // Custom physical shard shape for logical sharding is not tile sized (check along shard width)
        IllegalShardSpecParams{
            .shape = Shape{100, 20},
            .page_config = PageConfig(Layout::TILE, Tile({32, 32})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::HEIGHT_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            num_cores_to_corerangeset(10, grid_size, /*row_wise=*/true),
                            {10, 20},
                            {40, 20},
                            ShardOrientation::ROW_MAJOR,
                        }
                },
            .expected_err_msg = "Wrong custom Tensor Layout alignment Alignment([40, 20]). For Tile layout innermost dimension should be multiple of tile width 32."
        },
        // Custom physical shard shape for logical sharding is not tile sized (check along shard height)
        IllegalShardSpecParams{
            .shape = Shape{100, 20},
            .page_config = PageConfig(Layout::TILE, Tile({32, 32})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::HEIGHT_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            num_cores_to_corerangeset(10, grid_size, /*row_wise=*/true),
                            {10, 20},
                            {40, 32},
                            ShardOrientation::ROW_MAJOR,
                        }
                },
            .expected_err_msg = "Wrong custom Tensor Layout alignment Alignment([40, 32]). For Tile layout second innermost dimension should be multiple of tile height 32."
        }
    )  // Values
    // clang-format on
);

class IllegalTensorSpecCreationTests : public ::testing::TestWithParam<IllegalShardSpecParams> {};

TEST_P(IllegalTensorSpecCreationTests, ExpectFailAndCheckErrMsg) {
    const auto& params = GetParam();

    auto tensor_layout = TensorLayout(DataType::BFLOAT16, params.page_config, params.memory_config);
    EXPECT_THAT(
        std::function<void()>(
            [&params, &tensor_layout]() { auto tensor_spec = TensorSpec(params.shape, tensor_layout); }),
        ThrowsMessage<std::runtime_error>(::testing::HasSubstr(params.expected_err_msg)));
}

INSTANTIATE_TEST_SUITE_P(
    TensorShardingTests,
    IllegalTensorSpecCreationTests,
    // clang-format off
    ::testing::Values(
        // HEIGHT sharded: Not enough cores
        IllegalShardSpecParams{
            .shape = Shape{100, 16},
            .page_config = PageConfig(Layout::TILE, Tile({32, 16})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::HEIGHT_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            num_cores_to_corerangeset(3, grid_size, /*row_wise=*/true),
                            {32, 16},
                            ShardOrientation::ROW_MAJOR,
                        }
                },
            .expected_err_msg = "Number of shards along height 4 must not exceed number of cores 3"
        },
        // HEIGHT sharded: Not enough cores
        IllegalShardSpecParams{
            .shape = Shape{100, 20},
            .page_config = PageConfig(Layout::TILE, Tile({32, 16})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::HEIGHT_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            num_cores_to_corerangeset(5, grid_size, /*row_wise=*/true),
                            {10, 20},
                            ShardOrientation::ROW_MAJOR,
                            ShardMode::LOGICAL,
                        }
                },
            .expected_err_msg = "Number of shards along height 10 must not exceed number of cores 5"
        },
        // HEIGHT sharded: Shard width does not match
        IllegalShardSpecParams{
            .shape = Shape{100, 20},
            .page_config = PageConfig(Layout::TILE, Tile({32, 16})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::HEIGHT_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            num_cores_to_corerangeset(10, grid_size, /*row_wise=*/true),
                            {32, 16},
                            ShardOrientation::ROW_MAJOR,
                        }
                },
            .expected_err_msg = "Shard width 16 must match physical width 32 for height sharded"
        },
        // HEIGHT sharded: Shard width does not match
        IllegalShardSpecParams{
            .shape = Shape{100, 20},
            .page_config = PageConfig(Layout::TILE, Tile({32, 16})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::HEIGHT_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            num_cores_to_corerangeset(10, grid_size, /*row_wise=*/true),
                            {32, 10},
                            ShardOrientation::ROW_MAJOR,
                            ShardMode::LOGICAL,
                        }
                },
            .expected_err_msg = "Shard width 16 must match physical width 32 for height sharded"
        },
        // WIDTH sharded: Not enough cores
        IllegalShardSpecParams{
            .shape = Shape{16, 100},
            .page_config = PageConfig(Layout::TILE, Tile({16, 32})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::WIDTH_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            num_cores_to_corerangeset(3, grid_size, /*row_wise=*/true),
                            {16, 32},
                            ShardOrientation::ROW_MAJOR,
                        }
                },
            .expected_err_msg = "Number of shards along width 4 must not exceed number of cores 3"
        },
        // WIDTH sharded: Not enough cores
        IllegalShardSpecParams{
            .shape = Shape{20, 100},
            .page_config = PageConfig(Layout::TILE, Tile({16, 32})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::WIDTH_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            num_cores_to_corerangeset(5, grid_size, /*row_wise=*/true),
                            {20, 10},
                            ShardOrientation::ROW_MAJOR,
                            ShardMode::LOGICAL,
                        }
                },
            .expected_err_msg = "Number of shards along width 10 must not exceed number of cores 5"
        },
        // WIDTH sharded: Shard height does not match
        IllegalShardSpecParams{
            .shape = Shape{20, 100},
            .page_config = PageConfig(Layout::TILE, Tile({16, 32})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::WIDTH_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            num_cores_to_corerangeset(10, grid_size, /*row_wise=*/true),
                            {16, 32},
                            ShardOrientation::ROW_MAJOR,
                        }
                },
            .expected_err_msg = "Shard height 16 must match physical height 32 for width sharded"
        },
        // WIDTH sharded: Shard height does not match
        IllegalShardSpecParams{
            .shape = Shape{20, 100},
            .page_config = PageConfig(Layout::TILE, Tile({16, 32})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::WIDTH_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            num_cores_to_corerangeset(10, grid_size, /*row_wise=*/true),
                            {10, 32},
                            ShardOrientation::ROW_MAJOR,
                            ShardMode::LOGICAL,
                        }
                },
            .expected_err_msg = "Shard height 16 must match physical height 32 for width sharded"
        },
        // BLOCK sharded: Grid is not rectangular
        IllegalShardSpecParams{
            .shape = Shape{100, 100},
            .page_config = PageConfig(Layout::TILE, Tile({16, 32})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::BLOCK_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            CoreRangeSet((std::set<CoreRange>){CoreRange(CoreCoord{0, 0}, CoreCoord{6, 0}), CoreRange(CoreCoord{0, 1}, CoreCoord{1, 1})}),
                            {10, 32},
                            ShardOrientation::ROW_MAJOR,
                            ShardMode::LOGICAL,
                        }
                },
            .expected_err_msg = "Shard grid must be one full rectangular grid for block sharded!"
        },
        // BLOCK sharded: Shards must stay within row/col
        IllegalShardSpecParams{
            .shape = Shape{100, 100},
            .page_config = PageConfig(Layout::TILE, Tile({32, 32})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::BLOCK_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{6, 6})),
                            {10, 20},
                            ShardOrientation::ROW_MAJOR,
                            ShardMode::LOGICAL,
                        }
                },
            .expected_err_msg = "Number of shards along height 10 must not exceed number of rows 7 for row major orientation!"
        },
        // BLOCK sharded: Shards must stay within row/col
        IllegalShardSpecParams{
            .shape = Shape{100, 100},
            .page_config = PageConfig(Layout::TILE, Tile({32, 32})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::BLOCK_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{6, 6})),
                            {20, 10},
                            ShardOrientation::ROW_MAJOR,
                            ShardMode::LOGICAL,
                        }
                },
            .expected_err_msg = "Number of shards along width 10 must not exceed number of columns 7 for row major orientation!"
        },
        // BLOCK sharded: Shards must stay within row/col
        IllegalShardSpecParams{
            .shape = Shape{100, 100},
            .page_config = PageConfig(Layout::TILE, Tile({32, 32})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::BLOCK_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{6, 6})),
                            {10, 20},
                            ShardOrientation::COL_MAJOR,
                            ShardMode::LOGICAL,
                        }
                },
            .expected_err_msg = "Number of shards along height 10 must not exceed number of columns 7 for column major orientation!"
        },
        // BLOCK sharded: Shards must stay within row/col
        IllegalShardSpecParams{
            .shape = Shape{100, 100},
            .page_config = PageConfig(Layout::TILE, Tile({32, 32})),
            .memory_config =
                MemoryConfig{
                    TensorMemoryLayout::BLOCK_SHARDED,
                    BufferType::L1,
                    ShardSpec{
                            CoreRangeSet(CoreRange(CoreCoord{0, 0}, CoreCoord{6, 6})),
                            {20, 10},
                            ShardOrientation::COL_MAJOR,
                            ShardMode::LOGICAL,
                        }
                },
            .expected_err_msg = "Number of shards along width 10 must not exceed number of rows 7 for column major orientation!"
        }
    )  // Values
    // clang-format on
);
