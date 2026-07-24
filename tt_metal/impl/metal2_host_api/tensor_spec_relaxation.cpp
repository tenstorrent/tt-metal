// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/metal2_host_api/tensor_spec_relaxation.hpp>

#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt_stl/reflection.hpp>

namespace tt::tt_metal::experimental {

// Return type spelled std::uint64_t to match the public header (== ttsl::hash::hash_t); the body
// works in ttsl::hash and its combiners, which is why reflection.hpp is included here, not there.
std::uint64_t hash_tensorspec_with_relaxation(
    const tt::tt_metal::TensorSpec& spec, const TensorSpecRelaxation& relaxation) {
    // Mirror ValidateTensorArgs's match lattice (program_run_args.cpp): hash exactly the fields
    // the match treats as load-bearing, so matches-under-relaxation <=> equal returned hash.
    // Precedence follows the validator: dynamic_tensor_shape wins over match_padded_shape_only.
    if (relaxation.dynamic_tensor_shape) {
        // Match requires tensor_layout to be equal and the logical_shape rank to be equal; the
        // per-dim logical (and padded) shape values may vary. Hash exactly that pair.
        return ttsl::hash::hash_objects_with_default_seed(spec.tensor_layout(), spec.logical_shape().rank());
    }
    if (relaxation.match_padded_shape_only) {
        // Match requires tensor_layout and padded_shape to be equal; logical_shape may differ.
        return ttsl::hash::hash_objects_with_default_seed(spec.tensor_layout(), spec.padded_shape());
    }
    // Strict: full TensorSpec. (logical_shape + tensor_layout are exactly TensorSpec's own
    // reflected attributes, so this is equivalent to hashing the whole spec.)
    return ttsl::hash::hash_objects_with_default_seed(spec.logical_shape(), spec.tensor_layout());
}

}  // namespace tt::tt_metal::experimental
