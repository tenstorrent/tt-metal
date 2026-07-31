// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/shape.hpp>

#include <tt-metalium/experimental/tensor/spec/layout/alignment.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>

namespace tt::tt_metal {

// ======================================================================================
//                    TensorLayout APIs with a caller-supplied Alignment
// ======================================================================================
//
// Outside the Runtime Tensor graduation surface. A TensorLayout derives its Alignment
// from its PageConfig, DataType and MemoryConfig; that derived Alignment is the only one
// Runtime intends to support. These APIs let a caller override it with an arbitrary
// Alignment, which is then merged with the derived one by rounding each dimension up.
//
// These functions are in the experimental stage because a caller-supplied Alignment is a
// way to spell a padded shape without saying so: it decouples the physical footprint of
// a tensor from its logical shape for reasons the Tensor itself cannot describe. New code
// should let TensorLayout derive its own Alignment.

/**
 * Same as constructing a TensorLayout, but the derived Alignment is merged with
 * **alignment** — each dimension of **alignment** is rounded up to a multiple of the
 * corresponding derived dimension, right-aligned when the ranks differ.
 *
 * pre-conditions:
 * - **alignment** has rank <= 2, or **memory_config** is interleaved.
 * - **alignment** is compatible with **page_config** / **dtype** / **memory_config**
 *   (same validation the derived Alignment is subject to).
 */
TensorLayout tensor_layout_with_custom_alignment(
    DataType dtype, const PageConfig& page_config, const MemoryConfig& memory_config, const Alignment& alignment);

/**
 * Construct a TensorLayout whose padded shape for **logical_shape** reproduces
 * **padded_shape**, by reverse-engineering an Alignment from the two shapes.
 *
 * This is the bridge for call sites that still carry a legacy padded shape around
 * instead of a logical shape plus a layout.
 *
 * pre-conditions:
 * - If **memory_config** is sharded, **logical_shape** and **padded_shape** may only
 *   differ in their last two dimensions.
 */
[[deprecated("Use of Padded Shape is deprecated")]] TensorLayout tensor_layout_from_padded_shape(
    DataType dtype,
    const PageConfig& page_config,
    const MemoryConfig& memory_config,
    const tt::tt_metal::Shape& logical_shape,
    const tt::tt_metal::Shape& padded_shape);

}  // namespace tt::tt_metal
