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

#include "tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"

namespace tt::tt_metal {
struct KernelGroup;
}

namespace tt::tt_metal::experimental::dfb::detail {

// Per-risc config matching dfb_initializer_per_risc_t
struct LocalDFBInterfaceHost {
    std::array<uint32_t, 4> base_addr = {0};
    std::array<uint32_t, 4> limit = {0};
    std::array<::dfb::PackedTileCounter, 4> packed_tile_counter = {0};
    uint8_t num_tcs_to_rr = 1;
    bool broadcast_tc = false;  // DM-DM BLOCKED producer: post to all TCs instead of round-robin
    uint8_t remapper_pair_index = 0;
    uint32_t consumer_tcs = 0;
    uint8_t remapper_consumer_ids_mask = 0;
    uint8_t producer_client_type = 0;
};

struct DFBRiscConfig {
    uint8_t risc_id = 0xFF;
    bool is_producer{};
    LocalDFBInterfaceHost config;
};

// One group of cores that share the same HW config (TC/remapper assignments).
// l1_by_core stores the allocated L1 base address per core (same formula fills base_addr/limit at serialize time).
struct DfbGroup {
    CoreRangeSet core_ranges;
    std::vector<DFBRiscConfig> hw_risc_configs;            // base_addr/limit intentionally zeroed
    std::vector<std::pair<CoreCoord, uint32_t>> l1_by_core; // (core, alloc_addr)
};

struct DataflowBufferImpl {
    uint32_t id{};
    CoreRangeSet core_ranges;
    DataflowBufferConfig config;

    uint16_t risc_mask = 0;  // bits 0-7 = DM riscs, bits 8-15 = Tensix riscs
    uint8_t tensix_trisc_mask = 0;  // bits 0-3: which TRISC(s) use DFB (producer=bit2, consumer=bit0 or bit3)
    uint16_t capacity = 0;

    // Per-CoreRangeSet groups of cores sharing identical HW config (TC/remapper).
    std::vector<DfbGroup> groups;

    // Maps each CoreCoord to {group_index, alloc_addr}
    std::unordered_map<CoreCoord, std::pair<size_t, uint32_t>> core_lookup_;

    // Shared config fields (written to dfb_initializer_t, same for all cores)
    uint32_t entry_size = 0;
    uint32_t stride_in_entries = 0;
    dfb_txn_id_descriptor_t producer_txn_descriptor = {};
    dfb_txn_id_descriptor_t consumer_txn_descriptor = {};

    // Flag to track if TC/remapper allocation has been finalized
    bool configs_finalized = false;
    // Flag to track if this DFB uses remapper (set during finalization)
    bool use_remapper = false;

    uint32_t total_size() const { return config.entry_size * config.num_entries; }
    uint32_t serialized_size() const;
    std::vector<uint8_t> serialize_for_core(const CoreCoord& core) const;
};

class TileCounterAllocator {
public:
    // Allocate a tile counter for (core, tensix_id). Each core has an independent counter sequence
    ::dfb::PackedTileCounter allocate(const CoreCoord& core, uint8_t tensix_id);
    void reset() { next_tc_id_.clear(); }

private:
    std::unordered_map<CoreCoord, std::array<uint8_t, 4>> next_tc_id_;
};

class RemapperIndexAllocator {
public:
    uint8_t allocate(const CoreCoord& core_coord);
    void reset();

private:
    std::unordered_map<CoreCoord, uint8_t> next_index_;
};

// Allocates hardware transaction IDs. Valid range: [0, 31]
class TxnIdAllocator {
public:
    std::vector<uint8_t> allocate(uint8_t count);
    void reset() { next_id_ = 0; }

private:
    uint8_t next_id_ = 0;
};

// Allocates Remapper clientTypes for BLOCKED consumer mode.
//
// Hardware access rules:
//   - DM RISCs (risc_id 0-7):    clientR must be in [0, 3] (DM TC groups 0-3)
//   - Tensix RISCs (risc_id 8-11): clientR must be in [4, 7] (NEO_0 to NEO_3)
//   - clientL (producer) must not equal clientR (consumer)
//
// Allocation:
//   - Tensix consumer: clientR = 4 + (risc_id - 8), and unique per Tensix neo.
//   - DM consumer: cycle through [0, 3] \ {producer_client_type} in round-robin order.
//     DM clientR IDs may repeat across consumers; uniqueness is provided by the tc_id
//     (cnt_sel) within each clientR group, which is assigned by TileCounterAllocator.
class ClientTypeAllocator {
public:
    // producer_client_type: the clientL already assigned to the producer.
    // consumer_risc_id: RISC ID of the consumer being allocated.
    // Returns clientType in [0, 3] for DM consumers, [4, 7] for Tensix consumers.
    uint8_t allocate_for_consumer(uint8_t producer_client_type, uint8_t consumer_risc_id);

    static uint8_t get_tensix_id(uint8_t client_type) { return client_type % 4; }

    void reset() {
        tensix_used_mask_ = 0;
        dm_alloc_count_ = 0;
    }

private:
    uint8_t tensix_used_mask_ = 0;  // Bitmask of used Tensix clientTypes (bits 0-3 → ids 4-7)
    uint8_t dm_alloc_count_ = 0;    // Counts DM consumers allocated; used for round-robin cycling
};

uint32_t finalize_dfbs(
    uint32_t programmable_core_type_index,
    std::vector<std::shared_ptr<tt::tt_metal::KernelGroup>>& kernel_groups,
    const std::vector<std::shared_ptr<DataflowBufferImpl>>& dataflow_buffers,
    uint32_t base_offset,
    uint32_t& dfb_offset,
    uint32_t& dfb_size);

}  // namespace tt::tt_metal::experimental::dfb::detail
