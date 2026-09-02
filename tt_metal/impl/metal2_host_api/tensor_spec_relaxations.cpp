// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp>

#include <cstdint>
#include <optional>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt_stl/reflection.hpp>

namespace tt::tt_metal::experimental {

// File-local helpers. A NAMED namespace rather than an anonymous one: these translation units are
// built as a unity build, where two anonymous namespaces merge into one and same-named entities
// collide (or, worse, silently resolve to the wrong one).
namespace relaxation_fields {

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

    // The sharded distribution geometry: the squeezed shard shape and the bank (core) list.
    //
    // Only pertinent where the shape is otherwise free. When the logical or padded shape is pinned
    // the geometry is pinned with it, since it is a pure function of that shape plus the shard spec
    // -- and tensor_layout, load-bearing everywhere, pins the shard spec.
    bool shard_distribution = false;
};

// Both functions below must consume every member; if one gains a term the other does not, they
// disagree silently and the whole point of deriving the set once is lost. Nothing in the language
// enforces that, so this fires on the edit that always accompanies a new term. It is a prompt, not
// a proof -- also extend the flag-combination sweep in test_tensor_spec_relaxations.cpp, which is
// what actually detects a disagreement.
static_assert(
    sizeof(PertinentFields) == 4,
    "A PertinentFields member was added or removed. Wire the new term into BOTH "
    "hash_tensorspec_with_relaxation and tensorspecs_match_with_relaxation, then update this count.");

PertinentFields pertinent_fields(const TensorSpecRelaxations& relaxation) {
    if (relaxation.dynamic_tensor_shape) {
        // Per-dim shape values are free. The rank stays load-bearing unless it too is relaxed, and
        // the sharded distribution geometry is load-bearing either way -- see shard_distribution.
        return PertinentFields{
            .logical_rank = !relaxation.relax_logical_rank,
            .shard_distribution = true,
        };
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

// The sharded distribution geometry a TensorSpec resolves to, or nullopt when it is interleaved.
//
// Why this has to be part of the equivalence class. For a sharded TensorParameter the CTA payload
// bakes the WHOLE geometry from the declared spec -- squeezed rank, num_banks, squeezed shard shape,
// packed bank coordinates -- and only tensor_shape_in_pages is re-emitted per dispatch as CRTA words
// (ResolveTensorParameterStaticCTAs / EmitBindingCrtaValues). But squeeze_shape_ranks decides its
// merges from the tensor AND shard shape VALUES, so a shape change dynamic_tensor_shape permits can
// move the geometry underneath those CTAs. The device then divides and mods the page coordinate by a
// stale shard shape (tensor_accessor.h) and silently addresses the wrong memory. Two specs are
// interchangeable only if the geometry they resolve to agrees.
//
// shard_shape_in_pages and cores together cover everything baked: squeeze_shape_ranks pushes to the
// tensor and shard shapes in lockstep so their ranks are always equal (hence pinning the shard shape
// pins the squeezed rank), and num_banks is cores().size(). The bank list needs pinning in its own
// right -- under GRID_2D it is computed from the UNsqueezed shapes, so it can go stale even when the
// squeeze output is identical.
std::optional<BufferDistributionSpec> shard_distribution_of(const tt::tt_metal::TensorSpec& spec) {
    if (!spec.memory_config().is_sharded()) {
        return std::nullopt;
    }
    return spec.compute_buffer_sharding_args().buffer_distribution_spec();
}

}  // namespace relaxation_fields

// Return type spelled std::uint64_t to match the public header (== ttsl::hash::hash_t); the body
// works in ttsl::hash and its combiners, which is why reflection.hpp is included here, not there.
std::uint64_t hash_tensorspec_with_relaxation(
    const tt::tt_metal::TensorSpec& spec, const TensorSpecRelaxations& relaxation) {
    // Hash exactly the load-bearing fields, so two specs that match under the relaxation hash
    // equally. Mirror tensorspecs_match_with_relaxation below term for term.
    const relaxation_fields::PertinentFields fields = relaxation_fields::pertinent_fields(relaxation);
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
    if (fields.shard_distribution) {
        // CoreCoord is a bare tt_xy_pair with no reflected hash, so fold the coordinates directly
        // rather than rely on hashing the vector.
        if (const std::optional<BufferDistributionSpec> dist = relaxation_fields::shard_distribution_of(spec);
            dist.has_value()) {
            hash = ttsl::hash::hash_objects(hash, dist->shard_shape_in_pages());
            for (const CoreCoord& core : dist->cores()) {
                hash = ttsl::hash::hash_objects(
                    hash, static_cast<std::uint32_t>(core.x), static_cast<std::uint32_t>(core.y));
            }
        }
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
    const relaxation_fields::PertinentFields fields = relaxation_fields::pertinent_fields(relaxation);
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
    if (fields.shard_distribution) {
        const std::optional<BufferDistributionSpec> a_dist = relaxation_fields::shard_distribution_of(a);
        const std::optional<BufferDistributionSpec> b_dist = relaxation_fields::shard_distribution_of(b);
        if (a_dist.has_value() != b_dist.has_value()) {
            return false;
        }
        if (a_dist.has_value()) {
            if (!(a_dist->shard_shape_in_pages() == b_dist->shard_shape_in_pages())) {
                return false;
            }
            if (a_dist->cores() != b_dist->cores()) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace tt::tt_metal::experimental
