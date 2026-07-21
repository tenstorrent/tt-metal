// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/advanced_options.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/table.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/face_geometry.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>  // tt::DataFormat
#include <tt_stl/strong_type.hpp>

// ============================================================================
//  DataflowBufferSpec API
// ============================================================================
//
// A DataflowBufferSpec is a descriptor for a Dataflow Buffer (DFB).
// A dataflow buffer is a software FIFO for sharing data between a producer kernel
// and a consumer kernel. (The "DFB" abbreviation is used throughout the API.)
//
// A DFB has the following properties:
//  - Entry size
//  - Number of entries
//  - (Optional) entry format metadata (data format, tile format)
//  - (Additional advanced options)
//
// A DFB's endpoint configuration is specified at the DFB binding site in
// KernelSpec, not here. (The endpoint configuration encodes producer/consumer
// kernel identity, number of kernel threads, and multi-threaded access patterns.)
//
// INVARIANT: At the node level, a DFB instance has exactly one producer kernel
//   instance and exactly one consumer kernel instance. You must respect this
//   invariant across the DataflowBufferSpec's endpoint bindings.
//   This is a per-node rule, not a per-spec one: you MAY bind more than one
//   KernelSpec to a producer (or consumer) endpoint, because their non-overlapping
//   node sets each still contribute exactly one instance per node. Such multiple
//   bindings on one endpoint are legal provided they have:
//     - non-overlapping node coverage, AND
//     - the same kernel kind (compute or data movement), AND
//     - identical binding-site parameters (access_pattern, num_threads)
//
// INSTANCING: Like KernelSpec, a DataflowBufferSpec is a *per-node template*.
//   One independent DFB instance is allocated per node where its endpoint
//   kernels run, in that node's local SRAM. That instance serves the same-node
//   producer and consumer kernel instances.
//
// PLACEMENT: Derived — the DFB's effective node set is the union of its bound
//   kernels' WorkUnitSpec target_nodes.
//
// HW RESOURCES: Gen1 architectures (Wormhole and Blackhole) support a fixed
//   number of DFBs per node. On Gen2, the DFB-per-node-limit depends on the
//   resident DFBs' endpoint configurations, as the DFB hardware resource footprint
//   varies with endpoint configuration.
//   The DFB's backing storage is allocated in SRAM ("L1"). The allocation
//   lifetime is Program-scope, and is allocated anew with every execution of
//   the Program.
//
// ============================================================================

namespace tt::tt_metal::experimental {

// Name identifying a DataflowBufferSpec within a ProgramSpec.
using DFBSpecName = ttsl::StrongType<std::string, struct DFBSpecNameTag>;

//------------------------------------------------
// DataflowBufferSpec
//------------------------------------------------

struct DataflowBufferSpec {
    // DFB identifier: used to reference this DFB within the ProgramSpec
    DFBSpecName unique_id;

    // Backing memory
    uint32_t entry_size = 0;  // in bytes
    uint32_t num_entries = 0;
    // Note: It is possible to override these per-Program execution (via ProgramRunArgs).

    ////////////////////////////////////
    // Entry format metadata
    ////////////////////////////////////

    // The fields in this section are used to convey DFB entry format metadata to the
    // Low-Level Kernel (LLK) device APIs (compute primitives).
    // (These only need to be considered for DFBs that are bound to a compute kernel.)

    // The data format is required for any DFB bound to a compute kernel
    std::optional<tt::DataFormat> data_format_metadata = std::nullopt;

    // Optional; if unspecified, the default tile format (32x32) is assumed
    std::optional<tt::tt_metal::Tile> tile_format_metadata = std::nullopt;

    // Optional override for this DFB's tile face layout.
    //
    // A tile is physically stored as a grid of fixed-size sub-blocks called "faces". The compute
    // engine normally infers how many faces a tile has, and how many rows each face holds, from
    // `tile_format_metadata`. Set this field only when an entry does not occupy a full tile, so it
    // holds fewer faces and/or shorter faces than the default; the compute engine then reads exactly
    // that much data instead of a whole tile. `FaceGeometry` carries those two values (rows-per-face
    // and number of faces).
    std::optional<FaceGeometry> unpack_face_geometry_metadata = std::nullopt;

    //////////////////////////////
    // Backing memory
    //////////////////////////////

    // Build DFB on borrowed memory.
    //
    // Instead of allocating Program execution-lifetime DFB storage in L1/SRAM (default),
    // build the DFB on top of a user-managed memory object. The DFB gets a non-owning view
    // of the memory for the duration of Program execution.
    //
    // The user-managed device memory object is declared at ProgramSpec scope and bound here.
    // (Currently, only TensorParameter is supported.) The actual memory address is supplied
    // at runtime via ProgramRunArgs.
    //
    // The bound memory object must have L1-based storage and be large enough to hold the DFB's
    // total size (entry_size * num_entries).
    //
    // (TODO: this should become std::variant<TensorParamName, BufferParameterName>.)
    std::optional<TensorParamName> borrowed_from = std::nullopt;

    //////////////////////////////
    // Advanced options (see advanced_options.hpp)
    //////////////////////////////
    DFBAdvancedOptions advanced_options;
};

//------------------------------------------------
// CrossNodeDataflowBufferSpec
//------------------------------------------------

// NOTE: Cross-Node DataflowBuffer is not yet supported!
//       A sketch is included in the experimental Metal 2.0 APIs for visibility.
//       See also Global DataflowBuffer (which has a user-managed lifetime).
//
// CrossNodeDataflowBufferSpec is the descriptor for a "cross-node" DFB:
// A DFB whose producer and consumer kernels run on different nodes, with data
// flowing over the NoC. Its semantics should be as close as possible to that of
// a local DFB.
//
// A CrossNodeDataflowBufferSpec has all of the properties of a DataflowBufferSpec,
// but must specify additional cross-node DFB specific properties, such as the
// producer-consumer node mapping.
//
// TBD: Much about cross-node DFBs is still TBD! Everything below this line is expected
//   to change with the implementation.
//
// Invariant: Every cross-node DFB instance has exactly one producer kernel instance and
//   one consumer kernel instance. The instances must not be on the same node.
//
// Instancing: At runtime, one cross-node DFB instance is allocated per entry in the
//   producer_consumer_map. The runtime infrastructure allocates SRAM ("L1") at both
//   endpoints.
//
// Placement: Specified directly via producer_consumer_map (rather than derived as
//   for local DFBs).
//
struct CrossNodeDataflowBufferSpec {
    // A cross-node DFB has all of the same properties as a local DFB
    DataflowBufferSpec dfb_spec;

    // Plus, some cross-node DFB-specific properties.
    // (These are TBD...)

    // Producer-consumer node mapping: each entry pairs a producer node with the
    // consumer node it feeds.
    // (What about multi-casting? TBD.)
    using ProducerNode = NodeCoord;
    using ConsumerNode = NodeCoord;
    using ProducerConsumerMap = Table<ProducerNode, ConsumerNode>;
    ProducerConsumerMap producer_consumer_map;
};

}  // namespace tt::tt_metal::experimental
