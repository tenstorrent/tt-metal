// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp> 

namespace tt::tt_metal::experimental::metal2_host_api {

typedef DFBSpecID = uint_32;
typedef DFBSpecName = std::string;
enum class DFBAccessPattern {STRIDED, BLOCKED, CONTIGUOUS};
struct TileDescriptor;


struct DataflowBufferSpec { 

    // DFB identifier
    // A handle used to reference this DFB within the ProgramSpec
    std::variant<DFBSpecID, DFBSpecName> unique_id;  
    // (I intend to remove either the string or uint32_t option. Having both is annoying. Thoughts?)

    // Target nodes
    using Nodes = std::variant<NodeCoord, NodeRange, NodeRangeSet>
    Nodes target_nodes;

    // Backing memory
    uint32_t entry_size;  // in bytes
    uint32_t num_entries;  
    // Note: It is possible to override these per-Program execution (via ProgramRunParams).

    // Endpoint info 
    // Configuring a DFB requires endpoint-specific info.
    // This is specified when the DFB is bound to a kernel:
    //   - Producer and consumer kernel identity
    //   - Number of producer threads, number of consumer threads
    //   - Producer and consumer access patterns

    // Entry format (optional)
    // Optional metadata; used to pass entry type info from host to kernel
    std::optional<TileDescriptor> tile_format_metadata;
    std::optional<DataFormat> data_format_metadata;

    // Remote DFB
    // A DFB is "local" by default. Its producer and consumer kernels are on the same node, 
    // sharing common L1 memory.
    // A "remote DFB" has its producer and consumer kernels on different nodes.
    // For a remote DFB, you must specify the producer-consumer map.
    bool is_remote_dfb = false;
    using ProducerConsumerMap = std::vector<std::pair<NodeCoord, NodeCoord>>;
    std::optional<ProducerConsumerMap> producer_consumer_map = std::nullopt;


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
    // All aliased DFBs must have size and target nodes, and must mutually declare each other as aliases.
    // (Aliased DFBs offer NO guarantees against data clobbering; the kernel author must ensure safety.)
    using DFBIdentifiers = std::vector<std::variant<DFBid, std::string>>;
    std::optional<DFBIdentifiers> alias_with = std::nullopt; 
       
    
    // Disable implicit sync
    // Implicit sync is handled via ISR (available on Gen2 only) 
    // Disabling may be useful in niche cases for fine tuning performance or performance debug.
    bool disable_implicit_sync = false;  

};


// Stolen from program_descriptors.hpp
struct TileDescriptor {
    TileDescriptor() = default;
    TileDescriptor(const Tile& tile);
    TileDescriptor(uint32_t height, uint32_t width, bool transpose) :
        height(height), width(width), transpose(transpose) {}

    uint32_t height = constants::TILE_HEIGHT;
    uint32_t width = constants::TILE_WIDTH;
    bool transpose = false;

    bool operator==(const TileDescriptor& other) const {
        return height == other.height && width == other.width && transpose == other.transpose;
    }
};

}  // namespace tt::tt_metal::experimental::metal2_host_api