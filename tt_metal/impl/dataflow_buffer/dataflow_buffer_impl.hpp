// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>

#include "tt_metal/hw/inc/internal/dataflow_buffer_interface.h"

namespace tt::tt_metal {
struct KernelGroup;
}

namespace tt::tt_metal::experimental::dfb::detail {

// Per-risc config matching dfb_initializer_per_risc_t (40 bytes)
struct LocalDFBInterfaceHost {
    std::array<uint32_t, 4> base_addr = {0};
    std::array<uint32_t, 4> limit = {0};
    std::array<::experimental::PackedTileCounter, 4> packed_tile_counter = {0};
    uint8_t num_tcs_to_rr = 1;
    uint8_t remapper_pair_index = 0;
    bool should_init_tc = false;
    uint32_t consumer_tcs = 0;
};

struct DFBRiscConfig {
    uint8_t risc_id = 0xFF;
    bool is_producer{};
    LocalDFBInterfaceHost config;
};

struct DataflowBufferImpl {
    uint32_t id{};
    CoreRangeSet core_ranges;
    DataflowBufferConfig config;

    uint16_t risc_mask = 0;  // bits 0-7 = DM riscs, bits 8-15 = Tensix riscs
    uint16_t capacity = 0;
    std::vector<DFBRiscConfig> risc_configs;

    // Shared config fields (written to dfb_initializer_t)
    uint32_t entry_size = 0;
    uint32_t stride_size = 0;
    std::array<uint8_t, 4> txn_ids = {0};
    uint8_t num_entries_per_txn_id = 0;
    uint8_t num_entries_per_txn_id_per_tc = 0;
    uint8_t remapper_consumer_mask = 0;
    uint8_t num_txn_ids = 0;

    std::optional<uint32_t> allocated_address;

    uint32_t total_size() const { return config.entry_size * config.num_entries; }
    uint32_t serialized_size() const;
    std::vector<uint8_t> serialize() const;  // returns config to write to device
};

class TileCounterAllocator {
public:
    ::experimental::PackedTileCounter allocate(uint8_t tensix_id);
    void reset() { next_tc_id_.fill(0); }

private:
    std::array<uint8_t, 4> next_tc_id_ = {0};
};

class RemapperIndexAllocator {
public:
    uint8_t allocate(const CoreCoord& core_coord);
    void reset();

private:
    std::unordered_map<CoreCoord, uint8_t> next_index_;
};

uint32_t finalize_dfbs(
    uint32_t programmable_core_type_index,
    std::vector<std::shared_ptr<tt::tt_metal::KernelGroup>>& kernel_groups,
    const std::vector<std::shared_ptr<DataflowBufferImpl>>& dataflow_buffers,
    uint32_t base_offset,
    uint32_t& dfb_offset,
    uint32_t& dfb_size);

}  // namespace tt::tt_metal::experimental::dfb::detail
