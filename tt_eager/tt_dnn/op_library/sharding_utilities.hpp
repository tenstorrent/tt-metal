/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Contains utility functions for partitioning shards work between multiple cores.
//

#pragma once

#include "tt_metal/common/math.hpp"
#include "tt_metal/common/core_coord.h"

#include "tt_metal/host_api.hpp"


namespace tt {
namespace tt_metal {

struct ShardingConfig {
    uint32_t first_partial_right_aligned_row_width;
    uint32_t first_partial_image_num_rows;
    uint32_t num_full_images;
    uint32_t last_partial_image_num_rows;
    uint32_t last_partial_left_aligned_row_width;
    uint32_t skip_after_partial_right_aligned_row;
    uint32_t skip_after_first_partial_image_row;
    uint32_t skip_after_full_image;
};

inline ShardingConfig get_specs_for_sharding_partition(uint32_t start_stick, uint32_t end_stick, uint32_t in_h, uint32_t in_w, uint32_t window_w, uint32_t pad_h, uint32_t pad_w) {
    /* Logic to compute:
     * NOTE: This logic is wrong if stride !=1
     *   - Input:
     *     start_stick
     *     end_stick
     *     in_h
     *     in_w
     *     window_w
     *     pad_h
     *     pad_w
     *
     *   - Output:
     *     first_partial_right_aligned_row_width
     *     first_partial_image_num_rows
     *     num_full_images
     *     last_partial_image_num_rows
     *     last_partial_left_aligned_row_width
     *     skip_after_partial_right_aligned_row
     *     skip_after_first_partial_image_row
     *     skip_after_full_image
     */

    uint32_t nsticks_per_core = end_stick - start_stick;

    // First partial right-aligned row
    uint32_t image_row_start_left_width = start_stick % in_w;
    uint32_t first_partial_right_aligned_row_width = image_row_start_left_width > 0 ? in_w - image_row_start_left_width : 0;

    // Last partial left-aligned row
    uint32_t sticks_after_first_partial_row = nsticks_per_core - first_partial_right_aligned_row_width;
    uint32_t last_partial_left_aligned_row_width = sticks_after_first_partial_row % in_w;

    // Figure out how to allocate full image rows to first partial image, full images, or last partial image
    // This also affects skip after first_partial_right_aligned_row
    uint32_t image_row_start_idx = start_stick / in_w;
    uint32_t image_row_start_idx_after_partial_right_aligned_row = (start_stick + first_partial_right_aligned_row_width) / in_w;
    uint32_t image_row_end_idx = end_stick / in_w;
    uint32_t image_start_idx = image_row_start_idx / in_h;
    uint32_t image_start_idx_after_partial_right_aligned_row = image_row_start_idx_after_partial_right_aligned_row / in_h;
    uint32_t image_start_height_after_partial_right_aligned_row = image_row_start_idx_after_partial_right_aligned_row % in_h;
    uint32_t image_end_idx = image_row_end_idx / in_h;

    // Default case: We don't have a partial right aligned row; so we start with either full images, last partial image, or partial left aligned row
    uint32_t skip_after_partial_right_aligned_row = 0;
    // Case: partial_right_aligned_row > 0 and completes an image
    if (first_partial_right_aligned_row_width > 0 and image_start_height_after_partial_right_aligned_row == 0) {
        skip_after_partial_right_aligned_row = window_w - 1 + pad_h * (in_w + 2 * pad_w);
    // Case: partial_right_aligned_row > 0 and doesn't complete an image
    } else if (first_partial_right_aligned_row_width > 0) {
        skip_after_partial_right_aligned_row = window_w - 1;
    }

    uint32_t first_partial_image_num_rows = 0;
    uint32_t skip_after_first_partial_image_row = 0;
    // Only case where we have first_partial_image_rows: We have at at least 1 completed image and the starting image row is in the middle of an image
    if (image_end_idx - image_start_idx_after_partial_right_aligned_row > 0 and image_start_height_after_partial_right_aligned_row > 0) {
        first_partial_image_num_rows = in_h - image_start_height_after_partial_right_aligned_row;
        skip_after_first_partial_image_row = pad_h * (in_w + 2 * pad_w);
    }

    // Full images
    uint32_t image_rows_after_first_partial_image = sticks_after_first_partial_row / in_w - first_partial_image_num_rows;
    uint32_t num_full_images = image_rows_after_first_partial_image / in_h;
    uint32_t skip_after_full_image = num_full_images > 0 ? pad_h * (in_w + 2 * pad_w) : 0;

    // Last partial image rows
    uint32_t last_partial_image_num_rows = image_rows_after_first_partial_image % in_h;

    return ShardingConfig{
               .first_partial_right_aligned_row_width = first_partial_right_aligned_row_width,
               .first_partial_image_num_rows = first_partial_image_num_rows,
               .num_full_images = num_full_images,
               .last_partial_image_num_rows = last_partial_image_num_rows,
               .last_partial_left_aligned_row_width = last_partial_left_aligned_row_width,
               .skip_after_partial_right_aligned_row = skip_after_partial_right_aligned_row,
               .skip_after_first_partial_image_row = skip_after_first_partial_image_row,
               .skip_after_full_image = skip_after_full_image
           };
}

} } // namespace tt::tt_metal
