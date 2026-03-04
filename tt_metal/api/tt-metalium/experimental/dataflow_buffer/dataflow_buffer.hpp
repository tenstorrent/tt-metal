// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <variant>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/kernel_types.hpp>

#include "tt_metal/hw/inc/internal/dataflow_buffer_interface.h"

namespace tt {
enum class DataFormat : uint8_t;
namespace tt_metal {
class Program;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal::experimental::dfb {

// Alias to avoid ambiguous 'experimental' when used inside tt::tt_metal::experimental
using AccessPattern = ::experimental::AccessPattern;

struct DataflowBufferConfig {
    uint32_t entry_size = 0;
    uint32_t num_entries = 0;
    uint16_t producer_risc_mask = 0x0;  // bits 0-7 = DM riscs, bits 8-15 = Tensix riscs
    uint8_t num_producers = 1;
    AccessPattern pap = AccessPattern::STRIDED;
    uint16_t consumer_risc_mask = 0x0;  // bits 0-7 = DM riscs, bits 8-15 = Tensix riscs
    uint8_t num_consumers = 1;
    AccessPattern cap = AccessPattern::STRIDED;
    bool enable_implicit_sync = false;
    DataFormat data_format = tt::DataFormat::Float16_b;
    std::optional<Tile> tile = std::nullopt;
};

// Note: This API and the DataflowBufferConfig are placeholder only, the final DataflowBuffer APIs will conform with
// host API redesign. Returns logical DFB id
uint32_t CreateDataflowBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const DataflowBufferConfig& config);

// Need to know which riscs the producer and consumer kernel run on to do tile counter and remapper allocation before a program is launched
// This information may not be available to user at time of DFB creation so binding can be done explicitly after kernels are created
void BindDataflowBufferToProducerConsumerKernels(Program& program, uint32_t dfb_id, KernelHandle producer_kernel_handle, KernelHandle consumer_kernel_handle);

}  // namespace tt::tt_metal::experimental::dfb
