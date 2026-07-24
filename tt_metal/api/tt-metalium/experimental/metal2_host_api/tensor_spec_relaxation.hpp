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
//  TensorSpecRelaxation
// ============================================================================
//
// Declares the ways in which a runtime tensor's TensorSpec is permitted to
// deviate from the TensorSpec a TensorParameter declares. Default-constructed
// (all flags false) means an EXACT TensorSpec match is required.
//
// The same relaxation governs two things, which MUST stay consistent:
//   1. Run-time validation -- how loosely SetProgramRunArgs / UpdateTensorArgs
//      match the supplied MeshTensor against the declared spec (see
//      ValidateTensorArgs in program_run_args.cpp).
//   2. Program-cache keying -- an op author writing a custom program hash must
//      key on exactly the spec fields the relaxation treats as load-bearing.
//      Key on too much and the cache misses when it should hit; key on too
//      little and it returns a Program built for a differently-shaped tensor.
//
// hash_tensorspec_with_relaxation() below is the tool for (2): it hashes the
// same field projection the validator compares, so the hash and the match
// cannot drift apart.
//
// CAUTION: These options are UNSAFE if set. Most kernels will NOT function
// correctly if the tensor argument's spec deviates from the declared spec. You
// must ensure your kernel logic outside of the TensorAccessor itself tolerates
// the deviation you permit.
// ============================================================================
struct TensorSpecRelaxation {
    // Permit tensor arguments whose logical_shape differs from the declared shape.
    // The argument's padded_shape must still match exactly.
    //
    // Effects:
    //  - Validation checks are relaxed.
    //  - TensorAccessor configuration is completely unchanged.
    bool match_padded_shape_only = false;

    // Permit tensor arguments with dynamic logical shape.
    // The argument's logical_shape AND padded_shape may differ from the declared
    // shape (the rank must remain constant). Strictly subsumes
    // match_padded_shape_only -- when both are set, dynamic_tensor_shape wins.
    //
    // Effects:
    //  - Validation checks are relaxed.
    //  - For a sharded tensor:
    //    The TensorAccessor configuration DYNAMICALLY reflects the tensor argument's actual shape.
    //    Shape, expressed in pages-per-dim, becomes implicit common runtime arguments.
    //  - For an interleaved TILED tensor:
    //    TensorAccessor configuration is unchanged
    //    (The page size is fixed by dtype/tile dims, so it cannot vary with shape).
    //  - For an interleaved ROW-MAJOR tensor:
    //    The TensorAccessor configuration DYNAMICALLY reflects the tensor argument's page size.
    //    NOTE: page_size = last_dim_width * element_size is part of the varying shape!
    //    The aligned_page_size becomes an implicit common runtime argument.
    //    (Your kernel can access this value via TensorAccessor::get_aligned_page_size().)
    bool dynamic_tensor_shape = false;
};

// Hash a TensorSpec's *load-bearing* fields under a relaxation: the exact
// projection that ValidateTensorArgs (program_run_args.cpp) compares for that
// relaxation. This guarantees
//
//     matches-under-relaxation  <=>  equal returned hash
//
// (modulo ordinary 64-bit collision resolution), so an op author's custom
// compute_program_hash stays consistent with the relaxation it declared on the
// TensorParameter -- keying the program cache on precisely what the fast-path
// tensor update is able to tolerate.
//
// A default-constructed (strict) relaxation projects the full TensorSpec, so it
// is exactly as discriminating as TensorSpec equality.
//
// Fold the result into the op's running hash the usual way, e.g.
//     ttsl::hash::hash_combine(seed, hash_tensorspec_with_relaxation(spec, rel));
//
// Return type is ttsl::hash::hash_t (== std::uint64_t), spelled std::uint64_t so this public
// header need not include <tt_stl/reflection.hpp> (banned in public API: it pulls in <reflect>
// and <nlohmann/json.hpp>). The .cpp uses ttsl::hash and its combiners directly.
std::uint64_t hash_tensorspec_with_relaxation(
    const tt::tt_metal::TensorSpec& spec, const TensorSpecRelaxation& relaxation);

}  // namespace tt::tt_metal::experimental
