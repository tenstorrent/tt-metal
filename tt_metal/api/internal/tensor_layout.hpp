// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>

// Internal TensorLayout helpers. Not part of the public API; no stability guarantee.
namespace tt::tt_metal::internal {

// Builds a TensorLayout whose alignment reproduces `padded_shape` for `logical_shape`.
// Padded shape is a legacy concept; prefer the TensorLayout constructor taking an explicit Alignment.
TensorLayout tensor_layout_from_padded_shape(
    DataType dtype,
    const PageConfig& page_config,
    const MemoryConfig& memory_config,
    const tt::tt_metal::Shape& logical_shape,
    const tt::tt_metal::Shape& padded_shape);

}  // namespace tt::tt_metal::internal
