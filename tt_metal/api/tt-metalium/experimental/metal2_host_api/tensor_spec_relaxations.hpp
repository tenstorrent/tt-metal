// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {
class TensorSpec;
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {

// ============================================================================
//  TensorSpecRelaxations
// ============================================================================
//
// A Program may declare a TensorParameter with a particular TensorSpec.
// At execution time (runtime), a MeshTensor argument is supplied to the Program.
// By default, the MeshTensor argument provided at execution time must EXACTLY
// match the TensorParameter's declared TensorSpec.
//
// A TensorSpecRelaxations can "relax" this requirement: it declares the way(s) in
// which the MeshTensor argument's TensorSpec is permitted to deviate from the
// TensorParameter's declared TensorSpec.
//
// A default-constructed TensorSpecRelaxations requires an exact match.
//
// CAUTION: These options are UNSAFE if set. Most kernels will NOT function
// correctly if the tensor argument's spec deviates from the declared spec! You
// must guarantee that your kernel logic outside of the TensorAccessor itself
// tolerates any relaxations that you declare.
//
// NOTE: The TensorSpecRelaxations structure is under active development and will
// change. We are starting with a crude "bag of bools" approach, introducing new
// relaxations as they are needed. This will be replaced with a more structured
// construct after the set of required relaxations is better understood.
//
// ============================================================================
struct TensorSpecRelaxations {
    // Permit tensor arguments whose logical_shape differs from the declared shape.
    // The MeshTensor argument's padded_shape must still match exactly.
    //
    // Effects:
    //  - Validation checks are relaxed.
    //  - TensorAccessor configuration is completely unchanged.
    //
    bool match_padded_shape_only = false;

    // Permit tensor arguments with dynamic logical shape.
    // The argument's logical_shape AND padded_shape may differ from the declared
    // shape.
    //
    // Notes:
    //  - This setting takes precedence over match_padded_shape_only -- if both are
    //    set, dynamic_tensor_shape wins.
    //  - dynamic_tensor_shape requires that the logical rank remain constant. To
    //    further relax the logical rank, additionally set relax_logical_rank.
    //  - dynamic_tensor_shape assumes that an interleaved row-major tensor's page
    //    size may vary with shape. To tighten, additionally set match_page_size.
    //
    // Effects:
    //  - Validation checks are relaxed.
    //  - For a sharded tensor:
    //    The TensorAccessor configuration DYNAMICALLY reflects the tensor argument's actual shape.
    //    Shape, expressed in pages-per-dim, becomes implicit common runtime arguments.
    //    NOTE: only the shape values are dynamic. The rest of the distribution geometry -- the
    //    shard shape in pages, the number of banks, and the bank coordinates -- is fixed when the
    //    ProgramSpec is built, so tensor arguments must agree on it. That is NOT implied by
    //    agreeing on the shard spec: the tensor and shard shapes are jointly squeezed to minimize
    //    rank, and the squeeze depends on the shape VALUES, so two shapes sharing a shard spec can
    //    still resolve to different geometry. Such an argument is REJECTED rather than silently
    //    mis-addressed; if you need it accepted, the shape you vary must leave the squeeze alone.
    //  - For an interleaved TILED tensor:
    //    TensorAccessor configuration is unchanged
    //    (The page size is fixed by dtype/tile dims, so it cannot vary with shape).
    //  - For an interleaved ROW-MAJOR tensor:
    //    The TensorAccessor configuration DYNAMICALLY reflects the tensor argument's page size.
    //    NOTE: page_size = last_dim_width * element_size is part of the varying shape!
    //    The aligned_page_size becomes an implicit common runtime argument.
    //    (Your kernel can access this value via TensorAccessor::get_aligned_page_size().)
    bool dynamic_tensor_shape = false;

    // Permit tensor arguments with a different logical rank than the declared shape.
    // This is intended to be used in conjunction with dynamic_tensor_shape.
    // (The flag is inert if dynamic_tensor_shape is not set.)
    //
    // Effects:
    //  - Validation checks are relaxed.
    //  - TensorAccessor configuration is completely unchanged.
    //    (The "rank" carried by the TensorAccessor is not the logical rank, but the physical
    //     "squeezed" rank of the memory layout.)
    bool relax_logical_rank = false;

    // Require that the tensor argument's page size matches that of the TensorParameter.
    // This is intended to be used in conjunction with dynamic_tensor_shape.
    // (The flag is inert if dynamic_tensor_shape is not set.)
    //
    // Effects:
    //  - Validation checks are tightened (relative to the dynamic_tensor_shape default behavior).
    //    The (unaligned) page size must match exactly.
    //  - For an interleaved ROW-MAJOR tensor:
    //    The TensorAccessor's aligned_page_size is made a static compile-time constant, rather
    //    than a dynamic common runtime argument (the default dynamic_tensor_shape behavior).
    //  - For any other tensor layout, it has no effect.
    bool match_page_size = false;
};

// Do two TensorSpecs "match" under a TensorSpecRelaxations?
//
// A relaxation defines an equivalence relation on TensorSpecs:
// Two TensorSpecs match under relaxation when they agree on every field the relaxation defines
// as pertinent. A given relaxation implies an equivalence relationship as defined in the
// TensorSpecRelaxations documentation above.
//
// NOTE: This check is used by SetProgramRunArgs, UpdateProgramRunArgs, and UpdateTensorArgs when
// validating a supplied MeshTensor argument against its TensorParameter's declared TensorSpec.
//
bool tensorspecs_match_with_relaxation(
    const tt::tt_metal::TensorSpec& a, const tt::tt_metal::TensorSpec& b, const TensorSpecRelaxations& relaxation);

// Hash a TensorSpec's pertinent fields under a TensorSpecRelaxations.
// If two TensorSpecs match under a given relaxation, they will return the same hash.
//
// NOTE: Return type is std::uint64_t, not ttsl::hash::hash_t.
// This is done so this public header need not include <tt_stl/reflection.hpp>.
//
std::uint64_t hash_tensorspec_with_relaxation(
    const tt::tt_metal::TensorSpec& spec, const TensorSpecRelaxations& relaxation);

}  // namespace tt::tt_metal::experimental
