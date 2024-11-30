// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "common_tensor_test_utils.hpp"
#include "gtest/gtest.h"
#include "host_api.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/async_runtime.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace {
// Helpers that potentially need to be moved into TensorLayout
Size flatten_to_2D(const ttnn::SimpleShape& shape) {
    const int rank = static_cast<int>(shape.rank());

    size_t width = 1;
    size_t height = 1;

    // Iterate dims in reverse order
    // Even tensor of rank 0 or 1
    for (int i = -1; i >= -rank; --i) {
        auto& dim = i == -1 ? width : height;
        dim *= shape[i];
    }

    Size size{height, width};

    return size;
}

std::array<size_t, 4> compute_shard_spec(const Size& shape, const Size& shard_shape) {
    const auto num_shards_height = tt::div_up(shape.height(), shard_shape.height());
    const auto last_shard_height =
        shape.height() % shard_shape.height() > 0 ? shape.height() % shard_shape.height() : shard_shape.height();
    const auto num_shards_width = tt::div_up(shape.width(), shard_shape.width());
    const auto last_shard_width =
        shape.width() % shard_shape.width() > 0 ? shape.width() % shard_shape.width() : shard_shape.width();

    return {num_shards_height, last_shard_height, num_shards_width, last_shard_width};
};

void pretty_print_data_as_shards(
    const std::vector<float>& data, const Size& shape, const Size& shard_shape, const size_t char_count = 3) {
    TT_FATAL(
        data.size() == shape.height() * shape.width(),
        "Data size {} should be same as shape size {}",
        data.size(),
        shape.height() * shape.width());

    const auto [num_shards_height, last_shard_height, num_shards_width, last_shard_width] =
        compute_shard_spec(shape, shard_shape);

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

Size get_physical_shape(
    const ttnn::SimpleShape& shape, const Size& logical_shard_shape, const Size& physical_shard_shape) {
    const int rank = static_cast<int>(shape.rank());

    size_t width = 1;
    size_t height = 1;

    // Iterate dims in reverse order
    // Even tensor of rank 0 or 1
    for (int i = -1; i >= -rank; --i) {
        auto& dim = i == -1 ? width : height;
        dim *= shape[i];
    }

    auto get_physical_size = [](auto original_size, auto logical_shard_size, auto physical_shard_size) {
        auto num_shards = tt::div_up(original_size, logical_shard_size);

        return physical_shard_size * num_shards;
    };

    auto physical_height = get_physical_size(height, logical_shard_shape.height(), physical_shard_shape.height());
    auto physical_width = get_physical_size(width, logical_shard_shape.width(), physical_shard_shape.width());
    Size size{physical_height, physical_width};

    return size;
}

using LogicalPhysicalIdxPairs = std::vector<std::array<size_t, 2>>;
using LogicalPhysicalMapping = std::pair<LogicalPhysicalIdxPairs, size_t>;
std::vector<LogicalPhysicalMapping> compute_logical_to_physical_shards_mapping(
    const Size& logical_2D_shape,
    const Size& logical_shard_shape,
    const Size& physical_shard_shape,
    const size_t physical_stride) {
    const auto logical_stride = logical_2D_shape.width();

    const auto [num_shards_height, last_shard_height, num_shards_width, last_shard_width] =
        compute_shard_spec(logical_2D_shape, logical_shard_shape);

    std::vector<LogicalPhysicalMapping> logical_physical_mapping(num_shards_height * num_shards_width);

    for (size_t shard_height_idx = 0; shard_height_idx < num_shards_height; shard_height_idx++) {
        for (size_t shard_width_idx = 0; shard_width_idx < num_shards_width; shard_width_idx++) {
            const auto num_shard_rows =
                shard_height_idx == num_shards_height - 1 ? last_shard_height : logical_shard_shape.height();
            const auto num_shard_cols =
                shard_width_idx == num_shards_width - 1 ? last_shard_width : logical_shard_shape.width();

            auto indices = LogicalPhysicalIdxPairs(num_shard_rows);
            const auto logical_start_idx = shard_height_idx * logical_shard_shape.height() * logical_stride +
                                           shard_width_idx * logical_shard_shape.width();
            const auto physical_start_idx = shard_height_idx * physical_shard_shape.height() * physical_stride +
                                            shard_width_idx * physical_shard_shape.width();
            for (size_t i = 0; i < num_shard_rows; i++) {
                indices[i] = {i * logical_stride + logical_start_idx, i * physical_stride + physical_start_idx};
            }

            logical_physical_mapping.push_back((LogicalPhysicalMapping){indices, num_shard_cols});
        }
    }
    return logical_physical_mapping;
};

std::vector<float> convert_fp32_logical_data_to_physical_data(
    const std::vector<float>& data,
    const ttnn::SimpleShape& shape,
    const Size& logical_shard_shape,
    const Size& physical_shard_shape) {
    TT_FATAL(
        data.size() == shape.volume(),
        "Data size {} should be same as volume indicated by shape {}",
        data.size(),
        shape);
    auto physical_size = get_physical_shape(shape, logical_shard_shape, physical_shard_shape);

    std::vector<float> physical_data(physical_size.height() * physical_size.width(), 0);

    auto logical_2D_shape = flatten_to_2D(shape);
    size_t physical_stride = physical_size.width();

    const auto logical_physical_mapping = compute_logical_to_physical_shards_mapping(
        logical_2D_shape, logical_shard_shape, physical_shard_shape, physical_stride);

    for (const auto& [indices, cols] : logical_physical_mapping) {
        for (const auto& idx_pair : indices) {
            auto logical_idx_start = idx_pair[0];
            auto physical_idx_start = idx_pair[1];

            for (size_t col = 0; col < cols; col++) {
                physical_data[physical_idx_start + col] = data[logical_idx_start + col];
            }
        }
    }

    TT_FATAL(
        physical_data.size() == physical_size.height() * physical_size.width(),
        "Physical data size {} should be same as calculated physical size {}",
        physical_data.size(),
        physical_size);

    return physical_data;
};

std::vector<float> convert_fp32_physical_data_to_logical_data(
    const std::vector<float>& physical_data,
    const ttnn::SimpleShape& shape,
    const Size& logical_shard_shape,
    const Size& physical_shard_shape) {
    auto physical_size = get_physical_shape(shape, logical_shard_shape, physical_shard_shape);
    TT_FATAL(
        physical_data.size() == physical_size.height() * physical_size.width(),
        "Physical data size {} should be same as calculated physical size {}",
        physical_data.size(),
        physical_size);

    auto logical_2D_shape = flatten_to_2D(shape);
    std::vector<float> data(logical_2D_shape.height() * logical_2D_shape.width(), 0);

    size_t physical_stride = physical_size.width();

    const auto logical_physical_mapping = compute_logical_to_physical_shards_mapping(
        logical_2D_shape, logical_shard_shape, physical_shard_shape, physical_stride);

    for (const auto& [indices, cols] : logical_physical_mapping) {
        for (const auto& idx_pair : indices) {
            auto logical_idx_start = idx_pair[0];
            auto physical_idx_start = idx_pair[1];

            for (size_t col = 0; col < cols; col++) {
                data[logical_idx_start + col] = physical_data[physical_idx_start + col];
            }
        }
    }

    TT_FATAL(
        data.size() == shape.volume(),
        "Data size {} should be same as volume indicated by shape {}",
        data.size(),
        shape);

    return data;
};

}  // namespace

namespace {
struct ShardWithAlignmentInputs {
    SimpleShape shape;
    Size shard_shape;
    Alignment shard_alignment;
    std::vector<float> data;
};

struct ShardWithAlignmentExpected {
    Size physical_shard_shape;
    Size physical_size;
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

    auto physical_shard_height = tt::round_up(params.inputs.shard_shape.height(), params.inputs.shard_alignment[0]);
    auto physical_shard_width = tt::round_up(params.inputs.shard_shape.width(), params.inputs.shard_alignment[1]);
    Size physical_shard_shape{physical_shard_height, physical_shard_width};
    ASSERT_EQ(physical_shard_shape, params.expected.physical_shard_shape);

    auto physical_size = get_physical_shape(params.inputs.shape, params.inputs.shard_shape, physical_shard_shape);
    ASSERT_EQ(physical_size, params.expected.physical_size);

    const auto& data = params.inputs.data;
    const auto& expected_physical_data = params.expected.physical_data;

    auto physical_data = convert_fp32_logical_data_to_physical_data(
        data, params.inputs.shape, params.inputs.shard_shape, physical_shard_shape);

    // auto shape_2D = flatten_to_2D(params.inputs.shape);
    // pretty_print_data_as_shards(data, shape_2D, params.inputs.shard_shape);
    // pretty_print_data_as_shards(physical_data, physical_size, physical_shard_shape);

    ASSERT_EQ(physical_data.size(), expected_physical_data.size());
    for (size_t i = 0; i < physical_data.size(); i++) {
        EXPECT_EQ(physical_data[i], expected_physical_data[i]);
    }
}

TEST_P(ShardWithAlignmentTests, PhysicalToLogical) {
    const auto& params = GetParam();

    auto physical_shard_height = tt::round_up(params.inputs.shard_shape.height(), params.inputs.shard_alignment[0]);
    auto physical_shard_width = tt::round_up(params.inputs.shard_shape.width(), params.inputs.shard_alignment[1]);
    Size physical_shard_shape{physical_shard_height, physical_shard_width};
    ASSERT_EQ(physical_shard_shape, params.expected.physical_shard_shape);

    auto physical_size = get_physical_shape(params.inputs.shape, params.inputs.shard_shape, physical_shard_shape);
    ASSERT_EQ(physical_size, params.expected.physical_size);

    // Use expected value as input physical data
    const auto& physical_data = params.expected.physical_data;
    const auto& expected_data = params.inputs.data;

    auto data = convert_fp32_physical_data_to_logical_data(
        physical_data, params.inputs.shape, params.inputs.shard_shape, physical_shard_shape);

    // auto shape_2D = flatten_to_2D(params.inputs.shape);
    // pretty_print_data_as_shards(physical_data, physical_size, physical_shard_shape);
    // pretty_print_data_as_shards(data, shape_2D, params.inputs.shard_shape);

    ASSERT_EQ(data.size(), expected_data.size());
    for (size_t i = 0; i < data.size(); i++) {
        EXPECT_EQ(data[i], expected_data[i]);
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
                .shape = SimpleShape{1, 2, 15, 20},
                .shard_shape = {15, 20},
                .shard_alignment = Alignment({16, 16}),
                .data = {  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
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
                .physical_shard_shape = {16, 32},
                .physical_size = {32, 32},
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
                .shape = SimpleShape{1, 1, 15, 15},
                .shard_shape = {5, 15},
                .shard_alignment = Alignment({16, 16}),
                .data = {  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
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
                .physical_shard_shape = {16, 16},
                .physical_size = {48, 16},
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
                .shape = SimpleShape{1, 2, 5, 20},
                .shard_shape = {10, 10},
                .shard_alignment = Alignment({16, 16}),
                .data = { 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  /**/  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
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
                .physical_shard_shape = {16, 16},
                .physical_size = {16, 32},
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
        // RM interleaved is equivalent to setting logical shard size to 1 by 1
        ShardWithAlignmentParams{
            ShardWithAlignmentInputs{
                .shape = SimpleShape{1, 2, 5, 1},
                .shard_shape = {1, 1},
                .shard_alignment = Alignment({1, 4}),
                .data = {  0,

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
                .physical_shard_shape = {1, 4},
                .physical_size = {10, 4},
                .physical_data = {  0,   0,   0,   0,

                                    1,   0,   0,   0,

                                    2,   0,   0,   0,

                                    3,   0,   0,   0,

                                    4,   0,   0,   0,

                                    5,   0,   0,   0,

                                    6,   0,   0,   0,

                                    7,   0,   0,   0,

                                    8,   0,   0,   0,

                                    9,   0,   0,   0}
            }
        },
        // RM height sharded with padding along width to align shards
        ShardWithAlignmentParams{
            ShardWithAlignmentInputs{
                .shape = SimpleShape{1, 2, 5, 1},
                .shard_shape = {3, 1},
                .shard_alignment = Alignment({1, 4}),
                .data = {  0,
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
                .physical_shard_shape = {3, 4},
                .physical_size = {12, 4},
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
        // RM width sharded with padding along width to align shards
        ShardWithAlignmentParams{
            ShardWithAlignmentInputs{
                .shape = SimpleShape{1, 2, 5, 10},
                .shard_shape = {10, 3},
                .shard_alignment = Alignment({1, 4}),
                .data = {  0,   1,   2,  /**/   3,   4,   5,  /**/   6,   7,   8,  /**/   9,
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
                .physical_shard_shape = {10, 4},
                .physical_size = {10, 16},
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
                .shape = SimpleShape{1, 2, 10, 10},
                .shard_shape = {3, 4},
                .shard_alignment = Alignment({5, 7}),
                .data = {  0,   1,   2,   3,  /**/   4,   5,   6,   7,  /**/   8,   9,
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
                .physical_shard_shape = {5, 7},
                .physical_size = {35, 21},
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
    SimpleShape shape;
    DataType data_type;
    PageConfig page_config;
    MemoryConfig memory_config;
};

struct CreateShardedTensorWithAlignmentExpected {
    Size physical_size;
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

    EXPECT_EQ(layout.compute_physical_shape(input_shape), params.expected.physical_size);
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
                .shape = SimpleShape{1, 48, 56, 32},
                .data_type = DataType::BFLOAT16,
                .page_config = PageConfig(Layout::TILE),
                .memory_config =
                    MemoryConfig{
                        .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
                        .buffer_type = BufferType::L1,
                        .shard_spec = ShardSpec{
                            num_cores_to_corerangeset(tt::div_up(48 * 56, 48), grid_size, /*row_wise=*/true),
                            {48, 32},
                            ShardOrientation::ROW_MAJOR,
                            false,
                            ShardMode::LOGICAL}
                    }
            },
            CreateShardedTensorWithAlignmentExpected{
                .physical_size = Size{3584, 32}
            }
        },
        // Example 1b: Logical shard shape that is already aligned
        // NOTE: If ShardMode::PHYSICAL, it expects height 56 to be padded up to 64
        // - Along height: 48 * 56 / 64 is 42 shards; 42 * 64 = 2688
        CreateShardedTensorWithAlignmentParams{
            CreateShardedTensorWithAlignmentInputs{
                .shape = SimpleShape{1, 48, 56, 32},
                .data_type = DataType::BFLOAT16,
                .page_config = PageConfig(Layout::TILE),
                .memory_config =
                    MemoryConfig{
                        .memory_layout = TensorMemoryLayout::HEIGHT_SHARDED,
                        .buffer_type = BufferType::L1,
                        .shard_spec = ShardSpec{
                            num_cores_to_corerangeset(tt::div_up(48 * 56, 64), grid_size, /*row_wise=*/true),
                            {64, 32},
                            ShardOrientation::ROW_MAJOR,
                            false,
                            ShardMode::LOGICAL}
                    }
            },
            CreateShardedTensorWithAlignmentExpected{
                .physical_size = Size{2688, 32}
            }
        },
        // Example 1c: For interleaved, we treat entire height/width as "logical shard shape" for calculations
        // 48 "shards" with 56 aligned to 32 for tile alignment; 48 * 64 = 3072
        CreateShardedTensorWithAlignmentParams{
            CreateShardedTensorWithAlignmentInputs{
                .shape = SimpleShape{1, 48, 56, 32},
                .data_type = DataType::BFLOAT16,
                .page_config = PageConfig(Layout::TILE),
                .memory_config =
                    MemoryConfig{
                        .memory_layout = TensorMemoryLayout::INTERLEAVED,
                        .buffer_type = BufferType::DRAM,
                        .shard_spec = std::nullopt
                    }
            },
            CreateShardedTensorWithAlignmentExpected{
                .physical_size = Size{3072, 32}
            }
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
                .shape = SimpleShape{1, 2, 10, 5},
                .data_type = DataType::UINT8,
                .page_config = PageConfig(Layout::ROW_MAJOR),
                .memory_config =
                    MemoryConfig{
                        .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
                        .buffer_type = BufferType::L1,
                        .shard_spec = ShardSpec{
                            num_cores_to_corerangeset(tt::div_up(5, 1), grid_size, /*row_wise=*/true),
                            {20, 1},
                            ShardOrientation::ROW_MAJOR,
                            false,
                            ShardMode::LOGICAL}
                    }
            },
            CreateShardedTensorWithAlignmentExpected{
                .physical_size = Size{20, 5}
            }
        },
        // Example 2b: Logical shard shape that is already aligned
        // NOTE: ShardMode::PHYSICAL is equivalent in this case
        // - Along width: 8 / 4 is 2 shards; 2 * 4 = 8
        CreateShardedTensorWithAlignmentParams{
            CreateShardedTensorWithAlignmentInputs{
                .shape = SimpleShape{1, 2, 10, 8},
                .data_type = DataType::UINT8,
                .page_config = PageConfig(Layout::ROW_MAJOR),
                .memory_config =
                    MemoryConfig{
                        .memory_layout = TensorMemoryLayout::WIDTH_SHARDED,
                        .buffer_type = BufferType::L1,
                        .shard_spec = ShardSpec{
                            num_cores_to_corerangeset(tt::div_up(5, 4), grid_size, /*row_wise=*/true),
                            {20, 4},
                            ShardOrientation::ROW_MAJOR,
                            false,
                            ShardMode::LOGICAL}
                    }
            },
            CreateShardedTensorWithAlignmentExpected{
                .physical_size = Size{20, 8}
            }
        },
        // Example 2c: For interleaved, we treat entire height/width as "logical shard shape" for calculations
        // 20 "shards" with 5 aligned to 1
        CreateShardedTensorWithAlignmentParams{
            CreateShardedTensorWithAlignmentInputs{
                .shape = SimpleShape{1, 2, 10, 5},
                .data_type = DataType::UINT8,
                .page_config = PageConfig(Layout::ROW_MAJOR),
                .memory_config =
                    MemoryConfig{
                        .memory_layout = TensorMemoryLayout::INTERLEAVED,
                        .buffer_type = BufferType::L1,
                        .shard_spec = std::nullopt
                    }
            },
            CreateShardedTensorWithAlignmentExpected{
                .physical_size = Size{20, 5}
            }
        },
        ////////////////////////////////////////////////////////////////////
        // EXAMPLE 3: Interesting cases with custom (legal) shard alignment
        ////////////////////////////////////////////////////////////////////
        // Example 3a: TILE block sharded tensor with shard alignment of 3 * 16 along the width
        // - Along height: 8 * 36 / 48 is 6 shards; 6 * 64 = 384
        // - Along width: 32 / 10 is 4 shards; 4 * custom alignment 48 = 192 (48 % 16 == 0, so it is legal)
        CreateShardedTensorWithAlignmentParams{
            CreateShardedTensorWithAlignmentInputs{
                .shape = SimpleShape{1, 8, 36, 32},
                .data_type = DataType::BFLOAT8_B,
                .page_config = PageConfig(Layout::TILE, Tile({32, 16})),
                .memory_config =
                    MemoryConfig{
                        .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
                        .buffer_type = BufferType::L1,
                        .shard_spec = ShardSpec{
                            num_cores_to_corerangeset(tt::div_up(8 * 36, 48) * tt::div_up(32, 10), grid_size, /*row_wise=*/true),
                            {48, 10},
                            {64, 48},
                            ShardOrientation::ROW_MAJOR,
                            false}
                    }
            },
            CreateShardedTensorWithAlignmentExpected{
                .physical_size = Size{384, 192}
            }
        },
        // Example 3b: ROW_MAJOR block sharded tensor with 2 and 1 extra rows and col per shard, respectively
        // - Along height: 2 * 10 / 5 is 4 shards; 4 * custom alignment 7 = 28 (no restriction on height alignment for ROW_MAJOR)
        // - Along width: 5 / 2 is 3 shards; 3 * custom alignment 3 = 9 (alignment on width can be arbitrary because UINT32 is already 4-byte aligned)
        CreateShardedTensorWithAlignmentParams{
            CreateShardedTensorWithAlignmentInputs{
                .shape = SimpleShape{1, 2, 10, 5},
                .data_type = DataType::UINT32,
                .page_config = PageConfig(Layout::ROW_MAJOR),
                .memory_config =
                    MemoryConfig{
                        .memory_layout = TensorMemoryLayout::BLOCK_SHARDED,
                        .buffer_type = BufferType::L1,
                        .shard_spec = ShardSpec{
                            num_cores_to_corerangeset(tt::div_up(2 * 10, 5) * tt::div_up(5, 2), grid_size, /*row_wise=*/true),
                            {5, 2},
                            {7, 3},
                            ShardOrientation::ROW_MAJOR,
                            false}
                    }
            },
            CreateShardedTensorWithAlignmentExpected{
                .physical_size = Size{28, 9}
            }
        }
    )  // Values
    // clang-format on
);
