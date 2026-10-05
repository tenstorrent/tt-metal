// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

namespace tt::tt_metal::experimental {

// Data structure built up from ProgramSpec to enable fast lookups
struct CollectedSpecData {
    // Name -> spec lookups.
    // dfb_by_name covers BOTH local and cross-node DFBs.
    // For cross-node DFBs, the pointee is the inner dfb_spec.
    // To check if a DFB is cross-node, check the cross_node_dfb_by_name map.
    std::unordered_map<KernelSpecName, const KernelSpec*> kernel_by_name;
    std::unordered_map<DFBSpecName, const DataflowBufferSpec*> dfb_by_name;
    std::unordered_map<DFBSpecName, const CrossNodeDataflowBufferSpec*> cross_node_dfb_by_name;
    std::unordered_map<SemaphoreSpecName, const SemaphoreSpec*> semaphore_by_name;
    std::unordered_map<ScratchpadSpecName, const ScratchpadSpec*> scratchpad_by_name;
    std::unordered_map<TensorParamName, const TensorParameter*> tensor_parameter_by_name;
    std::unordered_map<PrefetcherPipeParamName, const PrefetcherPipeParameter*> prefetcher_pipe_by_name;

    // Tensor parameter usage (derived from kernel tensor bindings).
    // Tracks which kernels bind a given tensor parameter.
    std::unordered_map<TensorParamName, std::vector<const KernelSpec*>> tensor_parameter_users;

    // PrefetcherPipe parameter usage. A pipe parameter is used either by a kernel that binds it
    // (KernelAdvancedOptions::prefetcher_pipe_bindings; the kernel is a sender or receiver of the pipe) or by a
    // relay DFB that aliases its ring (DFBAdvancedOptions::prefetcher_pipe_relays). A kernel binds a
    // given pipe at most once, through one accessor (enforced during collection); the binder record
    // keeps that accessor so role checks can reason about the whole pipe group it names.
    struct PrefetcherPipeBinderRecord {
        const KernelSpec* kernel;
        const KernelAdvancedOptions::PrefetcherPipeBinding* binding;
    };
    struct PrefetcherPipeUsers {
        std::vector<PrefetcherPipeBinderRecord> binders;
        std::vector<const DataflowBufferSpec*> relays;
    };
    std::unordered_map<PrefetcherPipeParamName, PrefetcherPipeUsers> prefetcher_pipe_users;

    // Scratchpad binders (derived from kernel scratchpad bindings).
    // Tracks which kernels bind a given ScratchpadSpec. More than one may, provided their node sets
    // are disjoint; the per-node placement census in ValidateProgramSpec enforces that (it needs the
    // kernel node sets, derived below). A kernel binds a given scratchpad at most once (enforced
    // during collection), so each kernel appears at most once in a spec's binder list.
    std::unordered_map<ScratchpadSpecName, std::vector<const KernelSpec*>> scratchpad_binders;

    // DFB endpoint info (derived from kernel bindings).
    // Populated for both local and cross-node DFBs.
    //
    // Multiple PRODUCER KernelSpecs (and multiple CONSUMER KernelSpecs) may bind the same DFB,
    // provided they have non-overlapping node coverage and matching binding-site parameters
    // (access_pattern, num_threads). This permits the canonical Metal 2.0 expression of the
    // legacy "two KernelDescriptors per work split, sharing CBs" pattern. The physical
    // invariant is local: at each node, exactly one producer kernel instance and one
    // consumer kernel instance.
    struct DFBEndpointInfo {
        struct EndpointRecord {
            const KernelSpec* kernel = nullptr;
            const DFBBinding* binding = nullptr;
        };
        std::vector<EndpointRecord> producers;
        std::vector<EndpointRecord> consumers;
    };
    std::unordered_map<DFBSpecName, DFBEndpointInfo> dfb_endpoints;

    // WorkUnit membership: a kernel may belong to multiple WorkUnitSpecs.
    std::unordered_map<KernelSpecName, std::vector<const WorkUnitSpec*>> kernel_work_units;

    // Derived node sets:
    //  - kernel_node_set: union of containing WorkUnitSpec target_nodes
    //  - dfb_node_set: union of binding-kernels' node sets (local DFBs only).
    std::unordered_map<KernelSpecName, NodeRangeSet> kernel_node_set;
    std::unordered_map<DFBSpecName, NodeRangeSet> dfb_node_set;
};

// ----------------------------------------------------------------------------
// CollectSpecData: Build derived data structures from a ProgramSpec
// ----------------------------------------------------------------------------
//
// Indexes the ProgramSpec into lookup tables for efficient access.
// Function enforces STRUCTURAL invariants only:
//   - No duplicate names (would corrupt map lookups)
//   - No dangling references (would cause .at() failures later)
//   - Complete endpoint info (DFBs have both producer and consumer)
//
// If this function returns, the CollectedSpecData is internally consistent,
// and the ProgramSpec is structurally well-formed.
// Semantic validation (thread limits, architecture rules, etc.) is separate.
CollectedSpecData CollectSpecData(const ProgramSpec& spec);

}  // namespace tt::tt_metal::experimental
