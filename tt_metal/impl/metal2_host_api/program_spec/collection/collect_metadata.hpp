// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

namespace tt::tt_metal::experimental {

// Data structure built up from ProgramSpec to enable fast lookups
struct CollectedSpecData {
    // Note that invariants listed within CollectedSpecData is a subset of the invariant
    // of ProgramSpec it's trying to represent.
    //
    // Pointers stored here point into the ProgramSpec this was built from.

    // ------------------------------------------------------------------------
    // Name -> spec lookups.
    //
    // dfb_by_name covers BOTH local and cross-node DFBs.
    // For cross-node DFBs, the pointee is the inner dfb_spec.
    // To check if a DFB is cross-node, check the cross_node_dfb_by_name map.
    //
    // Invariant within this section:
    // - No values of *_by_name are nullptr.
    // - Each key of *_by_name is the unique_id (or equivalent) of the spec its value points at.
    // - Every key k of cross_node_dfb_by_name is also a key of dfb_by_name, and
    //   dfb_by_name[k] == &cross_node_dfb_by_name[k]->dfb_spec.
    std::unordered_map<KernelSpecName, const KernelSpec*> kernel_by_name;
    std::unordered_map<DFBSpecName, const DataflowBufferSpec*> dfb_by_name;
    std::unordered_map<DFBSpecName, const CrossNodeDataflowBufferSpec*> cross_node_dfb_by_name;
    std::unordered_map<SemaphoreSpecName, const SemaphoreSpec*> semaphore_by_name;
    std::unordered_map<ScratchpadSpecName, const ScratchpadSpec*> scratchpad_by_name;
    std::unordered_map<TensorParamName, const TensorParameter*> tensor_parameter_by_name;
    std::unordered_map<PrefetcherPipeParamName, const PrefetcherPipeParameter*> prefetcher_pipe_by_name;

    // ------------------------------------------------------------------------
    // Resource -> Users lookups.

    // Relay DFBs per PrefetcherPipeParameter. A local DFB relays a pipe when it lists it in
    // DFBAdvancedOptions::prefetcher_pipe_relays (its ring aliases the pipe's ring).
    //
    // Invariants:
    // - Every key is a key in prefetcher_pipe_by_name.
    // - Every vector is non-empty.
    // - Every pointer is non-nullptr, and the pointers in each vector are unique within the vector.
    // - Every pointer p is the registered local DFB: dfb_by_name[p->unique_id] == p and
    //   p->unique_id is not in cross_node_dfb_by_name.
    // - Every element lists this pipe in its advanced_options.prefetcher_pipe_relays.
    std::unordered_map<PrefetcherPipeParamName, std::vector<const DataflowBufferSpec*>> prefetcher_pipe_relays;

    // Scratchpad binders (derived from kernel scratchpad bindings).
    // Tracks which kernels bind a given ScratchpadSpec. More than one may, provided their node sets
    // are disjoint; the per-node placement census in ValidateProgramSpec enforces that (it needs the
    // kernel node sets, derived below). A kernel binds a given scratchpad at most once (enforced
    // during collection), so each kernel appears at most once in a spec's binder list.
    //
    // Invariants:
    // - Every key is a key in scratchpad_by_name.
    // - Every vector is non-empty.
    // - Every pointer is non-nullptr, and the pointers in each vector are unique within the vector.
    // - Every pointer k is the registered kernel: kernel_by_name[k->unique_id] == k.
    // - Every kernel in a vector has a scratchpad binding naming this key
    //   (some k->scratchpad_bindings entry has scratchpad_spec_name == key).
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
        // Invariants:
        // - Neither producers nor consumers is empty.
        // - Every producers' binding->endpoint_type == EndpointType::PRODUCER.
        // - Every consumers' binding->endpoint_type == EndpointType::CONSUMER.
        // - All bindings (across producers and consumers) have the same dfb_spec_name.
        // - A binding appears at most once across producers and consumers (a binding has one role).

        struct EndpointRecord {
            // Invariants:
            // - Neither pointer is nullptr.
            // - kernel is the registered kernel: kernel_by_name[kernel->unique_id] == kernel.
            // - binding is one of kernel->dfb_bindings.
            const KernelSpec* kernel = nullptr;
            const DFBBinding* binding = nullptr;
        };
        std::vector<EndpointRecord> producers;
        std::vector<EndpointRecord> consumers;
    };
    // Invariants:
    // - Every key is a key in dfb_by_name, and equals the dfb_spec_name of every binding in its entry.
    // - Every local DFB (a key of dfb_by_name that is not in cross_node_dfb_by_name) has an entry.
    //   (A cross-node DFB has an entry only if some kernel binds it.)
    std::unordered_map<DFBSpecName, DFBEndpointInfo> dfb_endpoints;

    // ------------------------------------------------------------------------
    // Derived node sets

    // kernel_node_set: union of containing WorkUnitSpec target_nodes.
    //  - For each KernelSpecName, the node set could be empty.
    //  - Every kernel in kernel_by_name have an entry.
    std::unordered_map<KernelSpecName, NodeRangeSet> kernel_node_set;

    // dfb_node_set: union of binding-kernels' node sets (local DFBs only).
    //   - For each DFBSpecName, the node set could be empty.
    //   - Exactly the local DFBs (keys of dfb_by_name that are not in cross_node_dfb_by_name) have an entry.
    //   - The node set of a DFB d is the union of kernel_node_set[rec.kernel->unique_id] over every
    //     record in dfb_endpoints[d].producers and dfb_endpoints[d].consumers.
    std::unordered_map<DFBSpecName, NodeRangeSet> dfb_node_set;

    // Semaphore binder census (derived from kernel semaphore bindings and kernel_node_set): which
    // kernel instances bind each semaphore, and their placement. A kernel binds a given semaphore at
    // most once (enforced during collection). A declared but unbound semaphore has no entry.
    struct SemaphoreBinderInfo {
        struct BinderRecord {
            // Invariants:
            // - Neither pointer is nullptr.
            // - kernel is the registered kernel: kernel_by_name[kernel->unique_id] == kernel.
            // - binding is one of kernel->semaphore_bindings.
            const KernelSpec* kernel = nullptr;
            const SemaphoreBinding* binding = nullptr;
        };
        // Invariants:
        // - Non-empty.
        // - No kernel appears more than once.
        // - Every binding->semaphore_spec_name equals the key of this entry in semaphore_binders.
        std::vector<BinderRecord> binders;
        // Invariant: union of kernel_node_set[rec.kernel->unique_id] over binders.
        // May contain no nodes if every binder kernel is placed on no nodes (binders itself is never empty).
        NodeRangeSet binder_node_set;
        // Invariant: sum over binders of
        //   kernel_node_set[rec.kernel->unique_id].num_cores() * rec.kernel->num_threads.
        // Counts kernel instances, not distinct nodes: kernels overlapping on a node each count there.
        uint32_t binder_instance_count = 0;
    };
    using SemaphoreBinderCensus = std::unordered_map<SemaphoreSpecName, SemaphoreBinderInfo>;
    // Invariant: every key is a key in semaphore_by_name.
    SemaphoreBinderCensus semaphore_binders;
};

// ----------------------------------------------------------------------------
// CollectSpecData: Build derived data structures from a ProgramSpec
// ----------------------------------------------------------------------------
//
// Indexes the ProgramSpec into lookup tables for efficient access.
//
// Some structural validations are to the ProgramSpec, but it is by no means exhaustive.
//
// post-condition:
//   - The CollectedSpecData follows it's invariants and reflects the ProgramSpec correctly.
//
CollectedSpecData CollectSpecData(const ProgramSpec& spec);

}  // namespace tt::tt_metal::experimental
