// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <optional>
#include <span>

#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>

namespace tt::tt_metal::experimental {

/**
 * Try to create an owning BFP host tensor directly from matrix rows.
 *
 * The logical tensor is flattened to a matrix using spec.logical_2d_shape().
 * Values within a row must be contiguous. row_stride is the distance between
 * row starts in source elements; zero means the logical matrix width. buffer
 * must cover the last logical element, including gaps between rows.
 *
 * Supports float/bfloat16 sources, BFP4_B/BFP8_B destinations, unpadded 32x32
 * tiles, standard 16x16 faces and no tile transpose. Returns nullopt without
 * modifying the input if the format, layout or source extent is unsupported.
 * The returned tensor owns its packed data and retains spec's metadata. Its
 * bytes match HostTensor::from_span applied to contiguous logical rows.
 * Explicit instantiations: float, bfloat16.
 */
template <typename T>
std::optional<HostTensor> try_create_bfp_host_tensor(
    std::span<const T> buffer, const TensorSpec& spec, size_t row_stride = 0);

}  // namespace tt::tt_metal::experimental
