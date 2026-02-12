// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <variant>

#include <tt-metalium/core_coord.hpp>

#include "tt_metal/hw/inc/internal/dataflow_buffer_interface.h"

namespace tt::tt_metal {
class Program;
}

namespace tt::tt_metal::experimental::dfb {

struct DataflowBufferConfig {
    uint32_t entry_size = 0;
    uint32_t num_entries = 0;
    uint16_t producer_risc_mask = 0x0;  // bits 0-7 = DM riscs, bits 8-15 = Tensix riscs
    uint8_t num_producers = 1;
    ::experimental::AccessPattern pap = ::experimental::AccessPattern::STRIDED;
    uint16_t consumer_risc_mask = 0x0;  // bits 0-7 = DM riscs, bits 8-15 = Tensix riscs
    uint8_t num_consumers = 1;
    ::experimental::AccessPattern cap = ::experimental::AccessPattern::STRIDED;
    bool enable_implicit_sync = false;
};

// Note: This API and the DataflowBufferConfig are placeholder only, the final DataflowBuffer APIs will conform with
// host API redesign. Returns logical DFB id
uint32_t CreateDataflowBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const DataflowBufferConfig& config);

}  // namespace tt::tt_metal::experimental::dfb
