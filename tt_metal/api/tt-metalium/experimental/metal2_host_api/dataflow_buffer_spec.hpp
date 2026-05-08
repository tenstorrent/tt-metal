// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>  // tt::DataFormat

namespace tt::tt_metal::experimental::metal2_host_api {

// A name identifying a DataflowBufferSpec within a ProgramSpec.
//
// CONVENTION: define names as `constexpr const char*` constants, e.g.:
//   constexpr const char* INPUT_DFB = "input_dfb";
//   DataflowBufferSpec{.unique_id = INPUT_DFB, ...};
// Reusing a single constant helps catch typos and errors at compile time.
using DFBSpecName = std::string;

// DataflowBuffer endpoint access patterns:
//  - STRIDED: a kernel thread accesses every N-th entry (where N = num_threads)
//  - ALL: each kernel thread accesses every DFB entry
//  - BLOCKED: a kernel thread accesses blocks of N entries, in strides of N blocks
enum class DFBAccessPattern { STRIDED, ALL, BLOCKED };

// A DataflowBufferSpec is a descriptor for a Dataflow Buffer (DFB):
// A software FIFO for sharing data between a producer kernel and a consumer kernel.
//
// A DFB has the following properties:
//  - Entry size
//  - Number of entries
//  - (Optional) entry format metadata (data format, tile format)
//  - (Additional advanced options)
//
// A DFB's endpoint configuration is specified at the DFB binding site in KernelSpec, not here.
// (producer/consumer kernel identity, threads, and access patterns)
//
// Invariant: A local DFB has exactly one producer kernel and one consumer kernel.
// Both must share identical WorkUnitSpec membership.
// (For cross-node communication, use RemoteDataflowBufferSpec.)
//
// Instancing: Like KernelSpec, a DataflowBufferSpec is a *per-node template*. One
// independent DFB instance is allocated per node where its endpoint kernels run, in
// that node's local SRAM. That instance serves the same-node producer and consumer
// kernel instances.
//
// Placement: Derived — the DFB's effective node set is the union of its bound
// kernels' WorkUnitSpec target_nodes.
//
struct DataflowBufferSpec {
    // DFB identifier: used to reference this DFB within the ProgramSpec
    DFBSpecName unique_id;

    // Backing memory
    uint32_t entry_size = 0;  // in bytes
    uint32_t num_entries = 0;
    // Note: It is possible to override these per-Program execution (via ProgramRunParams).

    ////////////////////////////////////
    // Entry format metadata
    ////////////////////////////////////

    // Required for DFBs bound to compute kernels; optional for DM-only DFBs
    std::optional<tt::DataFormat> data_format_metadata = std::nullopt;

    // Optional; used to pass tile type info from host to kernel
    std::optional<tt::tt_metal::Tile> tile_format_metadata = std::nullopt;

    //////////////////////////////
    // Advanced options
    //////////////////////////////

    // Build DFB on borrowed memory
    // Instead of program-scope memory allocation, the DFB is built on user-allocated memory (buffer or tensor)
    // If uses_borrowed_memory is true, you must pass a BufferView or MeshTensorView as a program execution parameter
    // (The user is responsible for ensuring the memory is valid for the duration of the Program execution.)
    bool uses_borrowed_memory = false;
    // Note: The borrowed memory itself is specified as a program execution parameter.

    // Alias two or more DFBs
    // Aliased DFBs are logically distinct, but physically share the same backing memory.
    // All aliased DFBs must have the same total size (num_entries * entry_size), must be bound to the same kernels,
    // and must mutually declare each other as aliases.
    // (Aliased DFBs offer NO guarantees against data clobbering; the kernel author must ensure safety.)
    using DFBIdentifiers = std::vector<DFBSpecName>;
    DFBIdentifiers alias_with;  // empty vector means no aliasing

    // Disable implicit sync
    // Implicit sync is handled via ISR (available on Gen2 only)
    // Disabling may be useful in niche cases for fine tuning performance or performance debug.
    bool disable_implicit_sync = false;
};

// NOTE: Remote DataflowBuffer is not yet supported!
//       A sketch is included in the experimental Metal 2.0 APIs for visibility.
//
// RemoteDataflowBufferSpec is the descriptor for a "remote" DFB:
// A DFB whose producer and consumer kernels run on different nodes, with data
// flowing over the NoC. Its semantics should be as close as possible to that of
// a local DFB.
//
// A RemoteDataflowBufferSpec has all of the properties of a DataflowBufferSpec,
// but must specify additional remote-DFB specific properties, such as the
// producer-consumer node mapping.
//
// TBD: Much about remote DFBs is still TBD! Everything below this line is expected
// to change with the implementation.
//
// Invariant: Every remote DFB instance has exactly one producer kernel instance and
// one consumer kernel instance. The instances must not be on the same node.
//
// Instancing: At runtime, one remote DFB instance is allocated per entry in the
// producer_consumer_map. The runtime infrastructure allocates SRAM ("L1") at both
// endpoints.
//
// Placement: Specified directly via producer_consumer_map (rather than derived as
// for local DFBs).
//
struct RemoteDataflowBufferSpec {
    // A remote DFB has all of the same properties as a local DFB
    DataflowBufferSpec dfb_spec;

    // Plus, some remote-DFB-specific properties.
    // (These are TBD...)

    // Producer-consumer node mapping: each entry pairs a producer node with the
    // consumer node it feeds.
    using ProducerConsumerMap = std::vector<std::pair<NodeCoord, NodeCoord>>;
    ProducerConsumerMap producer_consumer_map;
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
