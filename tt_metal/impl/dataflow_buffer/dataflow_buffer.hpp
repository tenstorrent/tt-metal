// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>

#include <tt_stl/assert.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/face_geometry.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/kernel_types.hpp>

#include "tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"

namespace tt {
enum class DataFormat : uint8_t;
namespace tt_metal {
class Program;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal::experimental::dfb {

using AccessPattern = ::dfb::AccessPattern;

// This enum is used to distinguish DFBs within one Neo vs DFBs across Neos
// Both cases run the same compute kernel on all Neos
enum class TensixScope : uint8_t {
    INTRA,
    INTER,
};

struct DataflowBufferConfig {
    uint32_t entry_size = 0;
    uint32_t num_entries = 0;
    uint16_t producer_risc_mask = 0x0;  // bits 0-7 = DM riscs, bits 8-15 = Tensix riscs
    uint8_t num_producers = 1;
    AccessPattern pap = AccessPattern::STRIDED;
    uint16_t consumer_risc_mask = 0x0;  // bits 0-7 = DM riscs, bits 8-15 = Tensix riscs
    uint8_t num_consumers = 1;
    AccessPattern cap = AccessPattern::STRIDED;

    // Implicit sync — per-side opt-in to the streamlined ISR-driven credit posting.
    // (Only applies to DM riscs; Tensix riscs always require explicit sync.)
    // Setting the two sides asymmetrically is a niche debug knob for isolating sync bugs;
    // typical usage sets both to the same value.
    bool enable_producer_implicit_sync = false;
    bool enable_consumer_implicit_sync = false;

    // Data format and tile formats for LLKs
    DataFormat data_format = tt::DataFormat::Float16_b;
    std::optional<Tile> tile = std::nullopt;
    /**
     * Optional override for how the compute engine interprets this DFB's tile faces. When set, it overrides the
     * face layout otherwise derived from @ref tile. Use it when an operand's data is laid out with a non-default
     * number of faces or rows-per-face.
     */
    std::optional<FaceGeometry> unpack_face_geometry = std::nullopt;
    // Set only when both producer and consumer are the same compute kernel
    std::optional<TensixScope> tensix_scope = std::nullopt;
    // When true, the DFB borrows L1 memory from an externally managed buffer
    // instead of allocating its own L1 region. The actual base address must be
    // supplied before launch via DataflowBufferImpl::set_borrowed_memory_base_addr.
    bool borrows_memory = false;
};

namespace detail {

inline uint32_t checked_total_size(uint32_t entry_size, uint32_t num_entries, std::string_view context) {
    const uint64_t total_size = static_cast<uint64_t>(entry_size) * num_entries;
    TT_FATAL(
        total_size <= std::numeric_limits<uint32_t>::max(),
        "{}: DFB size overflow: entry_size {} * num_entries {} = {} bytes exceeds UINT32_MAX ({} bytes).",
        context,
        entry_size,
        num_entries,
        total_size,
        std::numeric_limits<uint32_t>::max());
    return static_cast<uint32_t>(total_size);
}

}  // namespace detail

// Note: This API and the DataflowBufferConfig are placeholder only, the final DataflowBuffer APIs will conform with
// host API redesign. Returns logical DFB id
uint32_t CreateDataflowBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const DataflowBufferConfig& config);

// Need to know which riscs the producer and consumer kernel run on to do tile counter and remapper allocation before a
// program is launched This information may not be available to user at time of DFB creation so binding can be done
// explicitly after kernels are created
void BindDataflowBufferToProducerConsumerKernels(
    Program& program, uint32_t dfb_id, KernelHandle producer_kernel_handle, KernelHandle consumer_kernel_handle);

}  // namespace tt::tt_metal::experimental::dfb
