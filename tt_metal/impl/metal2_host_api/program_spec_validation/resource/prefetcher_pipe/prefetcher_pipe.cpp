// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <unordered_set>

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec_validation/resource/resource.hpp"

namespace tt::tt_metal::experimental {

// Everything about a PrefetcherPipeParameter is decidable from the spec alone (geometry and kernel
// placement); the pipe object arrives later via ProgramRunArgs and is reconciled against this
// geometry then.
//
// Rule 1. Geometry: non-empty receivers, ring_size > 0, entry_size > 0, L1-aligned and <= ring_size.
// (Rules 2-6 are structural: roles.cpp and lanes_and_relays.cpp.)
void ValidatePrefetcherPipeParameter(const PrefetcherPipeParameter& pipe, uint32_t l1_alignment) {
    const NodeRangeSet receivers = to_node_range_set(pipe.receivers);
    TT_FATAL(receivers.num_cores() > 0, "PrefetcherPipeParameter '{}' has no receiver nodes", pipe.unique_id);
    TT_FATAL(pipe.ring_size > 0, "PrefetcherPipeParameter '{}' has ring_size = 0", pipe.unique_id);
    TT_FATAL(pipe.entry_size > 0, "PrefetcherPipeParameter '{}' has entry_size = 0", pipe.unique_id);
    TT_FATAL(
        pipe.entry_size % l1_alignment == 0,
        "PrefetcherPipeParameter '{}' entry_size {} must be a multiple of the L1 alignment ({})",
        pipe.unique_id,
        pipe.entry_size,
        l1_alignment);
    TT_FATAL(
        pipe.entry_size <= pipe.ring_size,
        "PrefetcherPipeParameter '{}' entry_size {} exceeds ring_size {}",
        pipe.unique_id,
        pipe.entry_size,
        pipe.ring_size);
}

// PrefetcherPipe bindings: accessor names, non-empty pipe lists, each pipe bound once
void ValidatePrefetcherPipeBindings(const KernelSpec& kernel) {
    std::unordered_set<std::string> accessor_names;
    // A kernel binds a given pipe at most once, within and across accessors: a second binding
    // would be a second device object over the same credit counters (two names for one pipe is a
    // handle alias, not a binding).
    std::unordered_set<PrefetcherPipeParamName> bound_pipes;
    for (const auto& binding : kernel.advanced_options.prefetcher_pipe_bindings) {
        auto [it, inserted] = accessor_names.insert(binding.accessor_name);
        TT_FATAL(
            inserted,
            "Kernel '{}' has duplicate PrefetcherPipe accessor_name '{}'",
            kernel.unique_id,
            binding.accessor_name);
        TT_FATAL(
            IsValidCppIdentifier(binding.accessor_name),
            "Kernel '{}' PrefetcherPipe accessor_name '{}' must be a valid C++ identifier",
            kernel.unique_id,
            binding.accessor_name);
        ValidateAccessorNameLength(kernel.unique_id, "PrefetcherPipe", binding.accessor_name);
        TT_FATAL(
            !binding.pipe_parameter_names.empty(),
            "Kernel '{}' PrefetcherPipe accessor '{}' names no PrefetcherPipeParameter",
            kernel.unique_id,
            binding.accessor_name);
        for (const auto& pipe_name : binding.pipe_parameter_names) {
            auto [pit, pinserted] = bound_pipes.insert(pipe_name);
            TT_FATAL(
                pinserted,
                "Kernel '{}' binds PrefetcherPipeParameter '{}' more than once (latest under accessor_name '{}'). "
                "A kernel may bind a given pipe at most once.",
                kernel.unique_id,
                pipe_name,
                binding.accessor_name);
        }
    }
}

void ValidatePrefetcherPipesUsed(const ValidationContext& ctx) {
    const ProgramSpec& spec = ctx.spec;
    const CollectedSpecData& collected = ctx.collected;

    // Every declared PrefetcherPipeParameter must be used by a kernel binding or a relay DFB.
    // (An unused pipe parameter would demand a run arg nothing reads.)
    std::unordered_set<PrefetcherPipeParamName> used_pipes;
    for (const auto& kernel : spec.kernels) {
        for (const auto& binding : kernel.advanced_options.prefetcher_pipe_bindings) {
            used_pipes.insert(binding.pipe_parameter_names.begin(), binding.pipe_parameter_names.end());
        }
    }
    for (const auto& [pipe_name, relays] : collected.prefetcher_pipe_relays) {
        used_pipes.insert(pipe_name);
    }
    for (const auto& pipe_parameter : spec.advanced_options.prefetcher_pipe_parameters) {
        TT_FATAL(
            used_pipes.contains(pipe_parameter.unique_id),
            "PrefetcherPipeParameter '{}' is defined but not bound by any kernel or relay DFB",
            pipe_parameter.unique_id);
    }
}

}  // namespace tt::tt_metal::experimental
