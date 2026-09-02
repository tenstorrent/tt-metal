// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp>

#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt_stl/reflection.hpp>

namespace tt::tt_metal::experimental {

namespace {

// The TensorSpec fields a relaxation treats as load-bearing. Deriving that set -- including the
// precedence rule that dynamic_tensor_shape subsumes match_padded_shape_only -- is the one piece
// of logic hash_tensorspec_with_relaxation and tensorspecs_match_with_relaxation must agree on, so
// it lives here once and both consult it. Their notions of equivalence therefore cannot drift.
//
// The relaxations are modifier flags rather than a mutually exclusive chain, so the load-bearing
// set is a product of the flags rather than one of a fixed list of modes. Naming the fields, not
// the modes, is what lets a new modifier be one line here plus one line in each consumer.
//
// tensor_layout is load-bearing in every relaxed set, so it is implicit rather than a member.
struct PertinentFields {
    // The unrelaxed default is special: the whole TensorSpec is load-bearing, and is compared via
    // TensorSpec's own operator== rather than field-by-field. Kept distinct so the strict path
    // keeps exactly the behavior it has always had.
    bool whole_spec = false;

    bool padded_shape = false;
    bool logical_rank = false;
};

// Both functions below must consume every member; if one gains a term the other does not, they
// disagree silently and the whole point of deriving the set once is lost. Nothing in the language
// enforces that, so this fires on the edit that always accompanies a new term. It is a prompt, not
// a proof -- also extend the flag-combination sweep in test_tensor_spec_relaxations.cpp, which is
// what actually detects a disagreement.
static_assert(
    sizeof(PertinentFields) == 3,
    "A PertinentFields member was added or removed. Wire the new term into BOTH "
    "hash_tensorspec_with_relaxation and tensorspecs_match_with_relaxation, then update this count.");

PertinentFields pertinent_fields(const TensorSpecRelaxations& relaxation) {
    if (relaxation.dynamic_tensor_shape) {
        // Per-dim shape values are free. The rank stays load-bearing unless it too is relaxed.
        return PertinentFields{.logical_rank = !relaxation.relax_logical_rank};
    }
    if (relaxation.match_padded_shape_only) {
        return PertinentFields{.padded_shape = true};
    }
    // relax_logical_rank is inert on its own, as its doc states: neither the strict comparison nor
    // padded-shape matching carries a separable rank term that dropping it could loosen. Note what
    // padded-shape matching does to the rank, which is not quite "pins it": a padded_shape's rank is
    // max(logical rank, alignment rank), so it pins every rank at or above the alignment rank -- the
    // ordinary case -- while already tolerating, with nobody opting in, the rank changes the padding
    // absorbs below it (with TILE's rank-2 default alignment, rank 1 against rank 2).
    return PertinentFields{.whole_spec = true};
}

}  // namespace

// Return type spelled std::uint64_t to match the public header (== ttsl::hash::hash_t); the body
// works in ttsl::hash and its combiners, which is why reflection.hpp is included here, not there.
std::uint64_t hash_tensorspec_with_relaxation(
    const tt::tt_metal::TensorSpec& spec, const TensorSpecRelaxations& relaxation) {
    // Hash exactly the load-bearing fields, so two specs that match under the relaxation hash
    // equally. Mirror tensorspecs_match_with_relaxation below term for term.
    const PertinentFields fields = pertinent_fields(relaxation);
    if (fields.whole_spec) {
        // logical_shape + tensor_layout are TensorSpec's own reflected attributes, so this is
        // equivalent to hashing the whole spec.
        return ttsl::hash::hash_objects_with_default_seed(spec.logical_shape(), spec.tensor_layout());
    }
    ttsl::hash::hash_t hash = ttsl::hash::hash_objects_with_default_seed(spec.tensor_layout());
    if (fields.padded_shape) {
        hash = ttsl::hash::hash_objects(hash, spec.padded_shape());
    }
    if (fields.logical_rank) {
        hash = ttsl::hash::hash_objects(hash, spec.logical_shape().rank());
    }
    return hash;
}

bool tensorspecs_match_with_relaxation(
    const tt::tt_metal::TensorSpec& a, const tt::tt_metal::TensorSpec& b, const TensorSpecRelaxations& relaxation) {
    // Compare exactly the load-bearing fields. This must compare fields, never hash equality: a
    // hash collision would otherwise report a false match -- the very failure the exact-comparison
    // key machinery exists to prevent.
    //
    // NOTE: ValidateTensorArgs (program_run_args.cpp) delegates its run-time accept/reject to this,
    // so validation and the program-cache hash share one definition of equivalence.
    const PertinentFields fields = pertinent_fields(relaxation);
    if (fields.whole_spec) {
        return a == b;
    }
    if (!(a.tensor_layout() == b.tensor_layout())) {
        return false;
    }
    if (fields.padded_shape && !(a.padded_shape() == b.padded_shape())) {
        return false;
    }
    if (fields.logical_rank && a.logical_shape().rank() != b.logical_shape().rank()) {
        return false;
    }
    return true;
}

}  // namespace tt::tt_metal::experimental
