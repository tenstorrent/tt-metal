// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/shape.hpp>

namespace tt::tt_metal {

// ======================================================================================
//                         Host pad / unpad
// ======================================================================================
//
// Host-side pad/unpad outside layout/dtype transforms with limited supports.

/**
 * Pad **tensor** into a host tensor with padded shape **output_padded_shape**.
 *
 * Return: a new HostTensor whose padded shape is **output_padded_shape**, with
 * **pad_value** written into the padded regions.
 *
 * pre-conditions:
 * - **tensor** layout must be ROW_MAJOR.
 * - **tensor** must not be sharded.
 * - **tensor** dtype is not FP8_E4M3.
 *
 * post-conditions:
 * - Result padded shape equals **output_padded_shape**.
 */
HostTensor pad(
    const HostTensor& tensor, const Shape& output_padded_shape, const Shape& input_tensor_start, float pad_value);

/**
 * Unpad **tensor** to the inclusive range [**output_tensor_start**, **output_tensor_end**).
 *
 * Return: a new HostTensor containing only the selected sub-volume.
 *
 * pre-conditions:
 * - **tensor** layout must be ROW_MAJOR.
 * - **tensor** must not be sharded.
 * - **tensor** dtype is not FP8_E4M3.
 *
 * post-conditions:
 * - Result MemoryConfig is default-constructed (DRAM Interleaved).
 */
HostTensor unpad(const HostTensor& tensor, const Shape& output_tensor_start, const Shape& output_tensor_end);

/**
 * Pad **input_tensor** so the last two dims are multiples of the default tile
 * (TILE_HEIGHT / TILE_WIDTH).
 *
 * Return: `pad(...)` with those rounded dims and zero starts on leading dims.
 *
 * pre-conditions:
 * - **input_tensor** layout must be ROW_MAJOR.
 * - **input_tensor** must not be sharded.
 * - **input_tensor** dtype is not FP8_E4M3.
 *
 * post-conditions:
 * - Result padded shape has last two dims rounded up to TILE_HEIGHT / TILE_WIDTH;
 *   leading dims match **input_tensor** padded shape.
 * - Result MemoryConfig is default-constructed (DRAM Interleaved); input memory
 *   config is not preserved.
 */
HostTensor pad_to_tile(const HostTensor& input_tensor, float pad_value);

/**
 * Inverse of `pad_to_tile`: unpad last two dims back to **output_tensor_shape**.
 *
 * Return: `unpad(...)` over starts-at-zero through **output_tensor_shape**.
 *
 * pre-conditions:
 * - **tensor** layout must be ROW_MAJOR.
 * - **tensor** must not be sharded.
 * - **tensor** dtype is not FP8_E4M3.
 * - Leading dims (all but last two) of logical shape must match **output_tensor_shape**.
 * - Input last two padded dims must be multiples of the default tile.
 * - **output_tensor_shape** last two dims must lie within one tile of the input padded dims.
 *
 * post-conditions:
 * - Result padded shape equals **output_tensor_shape**.
 * - Result MemoryConfig is default-constructed (DRAM Interleaved); input memory
 *   config is not preserved.
 */
HostTensor unpad_from_tile(const HostTensor& input_tensor, const Shape& output_tensor_shape);

}  // namespace tt::tt_metal
