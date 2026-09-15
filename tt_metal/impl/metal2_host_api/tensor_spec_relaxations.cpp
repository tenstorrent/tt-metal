// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp>

#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt_stl/reflection.hpp>

namespace tt::tt_metal::experimental {

namespace {

// The fields a relaxation treats as load-bearing. Selecting the mode -- including the precedence
// rule that dynamic_tensor_shape subsumes match_padded_shape_only -- is the one piece of logic
// hash_tensorspec_with_relaxation and tensorspecs_match_with_relaxation must agree on, so it lives
// here once and both consult it. Their notions of equivalence therefore cannot drift.
enum class RelaxationMode {
    Strict,           // full TensorSpec
    PaddedShapeOnly,  // tensor_layout + padded_shape (logical_shape may differ)
    DynamicRank,      // tensor_layout + logical_shape rank (per-dim shape values may differ)
};

RelaxationMode relaxation_mode(const TensorSpecRelaxations& relaxation) {
    if (relaxation.dynamic_tensor_shape) {
        return RelaxationMode::DynamicRank;
    }
    if (relaxation.match_padded_shape_only) {
        return RelaxationMode::PaddedShapeOnly;
    }
    return RelaxationMode::Strict;
}

}  // namespace

// Return type spelled std::uint64_t to match the public header (== ttsl::hash::hash_t); the body
// works in ttsl::hash and its combiners, which is why reflection.hpp is included here, not there.
std::uint64_t hash_tensorspec_with_relaxation(
    const tt::tt_metal::TensorSpec& spec, const TensorSpecRelaxations& relaxation) {
    // Hash exactly the load-bearing fields for the mode, so two specs that match under the
    // relaxation hash equally. (logical_shape + tensor_layout are TensorSpec's own reflected
    // attributes, so the Strict case is equivalent to hashing the whole spec.)
    switch (relaxation_mode(relaxation)) {
        case RelaxationMode::DynamicRank:
            return ttsl::hash::hash_objects_with_default_seed(spec.tensor_layout(), spec.logical_shape().rank());
        case RelaxationMode::PaddedShapeOnly:
            return ttsl::hash::hash_objects_with_default_seed(spec.tensor_layout(), spec.padded_shape());
        case RelaxationMode::Strict: break;
    }
    return ttsl::hash::hash_objects_with_default_seed(spec.logical_shape(), spec.tensor_layout());
}

bool tensorspecs_match_with_relaxation(
    const tt::tt_metal::TensorSpec& a, const tt::tt_metal::TensorSpec& b, const TensorSpecRelaxations& relaxation) {
    // Compare exactly the load-bearing fields for the mode. This must compare fields, never hash
    // equality: a hash collision would otherwise report a false match -- the very failure the
    // exact-comparison key machinery exists to prevent.
    //
    // NOTE: ValidateTensorArgs (program_run_args.cpp) delegates its run-time accept/reject to this,
    // so validation and the program-cache hash share one definition of equivalence.
    switch (relaxation_mode(relaxation)) {
        case RelaxationMode::DynamicRank:
            return a.tensor_layout() == b.tensor_layout() && a.logical_shape().rank() == b.logical_shape().rank();
        case RelaxationMode::PaddedShapeOnly:
            return a.tensor_layout() == b.tensor_layout() && a.padded_shape() == b.padded_shape();
        case RelaxationMode::Strict: break;
    }
    return a == b;
}

}  // namespace tt::tt_metal::experimental
