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
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/face_geometry.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>  // tt::DataFormat

namespace tt::tt_metal::experimental::metal2_host_api {

// A name identifying a DataflowBufferSpec within a ProgramSpec.
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
    // at runtime via ProgramRunParams.
    //
    // The bound memory object must have L1-based storage and be large enough to hold the DFB's
    // total size (entry_size * num_entries).
    //
    // (TODO: this should become std::variant<TensorParameterName, BufferParameterName>.)
    std::optional<TensorParameterName> borrowed_from = std::nullopt;

    //////////////////////////////
    // Advanced options (see advanced_options.hpp)
    //////////////////////////////
    DFBAdvancedOptions advanced_options;
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
