// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>

#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "tt_metal/impl/program/program_impl.hpp"

namespace tt::tt_metal::experimental::dfb {

uint32_t CreateDataflowBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const DataflowBufferConfig& config) {
    auto core_range_set = std::visit(
        ttsl::overloaded{
            [](const CoreCoord& core_spec) { return CoreRangeSet(CoreRange(core_spec, core_spec)); },
            [](const CoreRange& core_spec) { return CoreRangeSet(core_spec); },
            [](const CoreRangeSet& core_spec) { return core_spec; },
        },
        core_spec);

    return program.impl().add_dataflow_buffer(core_range_set, config);
}

namespace detail {

::experimental::PackedTileCounter TileCounterAllocator::allocate(uint8_t tensix_id) {
    TT_FATAL(tensix_id < ::experimental::NUM_TENSIX, "Invalid tensix_id: {}", tensix_id);
    TT_FATAL(
        next_tc_id_[tensix_id] < ::experimental::NUM_TENSIX_TILE_COUNTERS_FOR_DM,
        "Out of tile counters for tensix {}",
        tensix_id);
    uint8_t tc_id = next_tc_id_[tensix_id]++;
    return static_cast<::experimental::PackedTileCounter>(
        (tensix_id << ::experimental::PACKED_TC_COUNTER_ID_BITS) | tc_id);
}

uint8_t RemapperIndexAllocator::allocate(const CoreCoord& core_coord) {
    uint8_t idx = next_index_[core_coord]++;
    TT_FATAL(
        idx < ::experimental::NUM_REMAPPER_PAIRINGS,
        "Out of remapper pairs for core ({}, {})",
        core_coord.x,
        core_coord.y);
    return idx;
}

void RemapperIndexAllocator::reset() { next_index_.clear(); }

std::array<uint8_t, 2> TransactionIdAllocator::allocate() {
    TT_FATAL(
        next_txn_id_ + TXN_IDS_PER_ALLOCATION <= ::experimental::MAX_TOTAL_TXN_IDS,
        "Transaction ID pool exhausted (max {})",
        ::experimental::MAX_TOTAL_TXN_IDS);
    std::array<uint8_t, 2> ids = {next_txn_id_, static_cast<uint8_t>(next_txn_id_ + 1)};
    next_txn_id_ += TXN_IDS_PER_ALLOCATION;
    return ids;
}

uint8_t calculate_num_tile_counters(const DataflowBufferConfig& config, bool is_producer) {
    if (config.cap == ::experimental::AccessPattern::BLOCKED) {
        return is_producer ? 1 : config.num_producers;
    }
    // Strided mode:
    // Producer: num_consumers / num_producers (number of consumers each producer is paired with)
    // Consumer: num_producers / num_consumers (number of producers each consumer is paired with)
    // When the ratio is < 1, use 1 (each pairs with exactly one of the other)
    if (is_producer) {
        if (config.num_consumers >= config.num_producers) {
            TT_FATAL(
                config.num_consumers % config.num_producers == 0,
                "num_consumers {} must be divisible by num_producers {} for strided producer",
                config.num_consumers,
                config.num_producers);
            return config.num_consumers / config.num_producers;
        }
        // More producers than consumers: each producer pairs with 1 consumer
        return 1;
    }

    if (config.num_producers >= config.num_consumers) {
        TT_FATAL(
            config.num_producers % config.num_consumers == 0,
            "num_producers {} must be divisible by num_consumers {} for strided consumer",
            config.num_producers,
            config.num_consumers);
        return config.num_producers / config.num_consumers;
    }
    // More consumers than producers: each consumer pairs with 1 producer
    return 1;
}

// Extract tensix IDs from risc_mask (bits 8-11)
// Returns vector of tensix_ids (0-3) that are being used
std::vector<uint8_t> extract_tensix_ids(uint16_t risc_mask) {
    std::vector<uint8_t> tensix_ids;
    uint16_t tensix_mask = (risc_mask >> 8) & 0x0F;  // bits 8-11
    for (uint8_t i = 0; i < ::experimental::NUM_TENSIX; i++) {
        if (tensix_mask & (1 << i)) {
            tensix_ids.push_back(i);
        }
    }
    return tensix_ids;
}

// Get tensix_id for allocation when only DM RISCs are used
// Round-robins through 0-3 based on pair index
uint8_t get_dm_tensix_id_for_pair(uint8_t pair_index) { return pair_index % 4; }

// Holds tile counters allocated together for a producer-consumer group
struct TileCounterGroup {
    ::experimental::PackedTileCounter producer_tc{};
    std::vector<::experimental::PackedTileCounter> consumer_tcs;
};

uint32_t DataflowBufferImpl::serialized_size() const {
    // One dfb_initializer_t + one dfb_initializer_per_risc_t per risc
    return sizeof(::experimental::dfb_initializer_t) +
           (risc_configs.size() * sizeof(::experimental::dfb_initializer_per_risc_t));
}

std::vector<uint8_t> DataflowBufferImpl::serialize() const {
    std::vector<uint8_t> data;
    data.reserve(serialized_size());

    ::experimental::dfb_initializer_t init = {};
    init.logical_id = this->id;
    init.entry_size = this->entry_size;
    init.stride_size = this->stride_size;
    init.capacity = this->capacity;
    init.risc_mask_bits.dm_mask = this->risc_mask & 0xFF;
    init.risc_mask_bits.tensix_mask = (this->risc_mask >> 8) & 0x0F;
    init.risc_mask_bits.reserved = 0;
    init.risc_mask_bits.tc_initialized = 0;  // set by device after init
    init.num_entries_to_process_threshold_producer = this->num_entries_to_process_threshold_producer;
    init.num_entries_to_process_threshold_consumer = this->num_entries_to_process_threshold_consumer;
    init.remapper_consumer_mask = this->remapper_consumer_mask;
    init.padding = 0;

    log_info(
        tt::LogMetal,
        "Serializing DFB {} with {} producers and {} consumers. risc_mask: 0x{:x}",
        this->id,
        this->config.num_producers,
        this->config.num_consumers,
        this->risc_mask);

    log_info(tt::LogMetal, "Entry size: {}", this->entry_size);
    log_info(tt::LogMetal, "Stride size: {}", this->stride_size);
    log_info(tt::LogMetal, "Capacity: {}", this->capacity);
    log_info(tt::LogMetal, "Risc mask: 0x{:x}", this->risc_mask);
    log_info(
        tt::LogMetal,
        "Threshold producer: {}, Threshold consumer: {}",
        this->num_entries_to_process_threshold_producer,
        this->num_entries_to_process_threshold_consumer);
    log_info(tt::LogMetal, "Remapper consumer mask: 0x{:x}", this->remapper_consumer_mask);

    const auto* init_bytes = reinterpret_cast<const uint8_t*>(&init);
    data.insert(data.end(), init_bytes, init_bytes + sizeof(init));

    // Write one dfb_initializer_per_risc_t per risc
    for (const auto& rc : risc_configs) {
        log_info(tt::LogMetal, "New risc config");
        ::experimental::dfb_initializer_per_risc_t per_risc = {};

        // Copy per-risc arrays
        for (int i = 0; i < ::experimental::MAX_NUM_TILE_COUNTERS_TO_RR; i++) {
            per_risc.base_addr[i] = rc.config.base_addr[i];
            per_risc.limit[i] = rc.config.limit[i];
            per_risc.packed_tile_counter[i] = rc.config.packed_tile_counter[i];
            log_info(tt::LogMetal, "Base addr {}: {}", i, static_cast<uint32_t>(per_risc.base_addr[i]));
            log_info(tt::LogMetal, "Limit {}: {}", i, static_cast<uint32_t>(per_risc.limit[i]));
            log_info(tt::LogMetal, "Packed tile counter {}: {}", i, (uint32_t)per_risc.packed_tile_counter[i]);
        }
        per_risc.num_tcs_to_rr = rc.config.num_tcs_to_rr;
        log_info(tt::LogMetal, "Num tcs to rr: {}", per_risc.num_tcs_to_rr);
        per_risc.flags.remapper_pair_index = static_cast<uint8_t>(rc.config.remapper_pair_index) & 0x3F;
        per_risc.flags.remapper_en =
            this->config.cap ==
            ::experimental::AccessPattern::BLOCKED;  // TODO: update this when there is 1 consumer to not use remapper
                                                     // and en when there are multiple dfbs on a core where any use
                                                     // remapper
        per_risc.flags.should_init_tc = rc.config.should_init_tc;
        per_risc.consumer_tcs = rc.config.consumer_tcs;
        log_info(tt::LogMetal, "Should init tc: {}", rc.config.should_init_tc);

        // Per-risc transaction ID fields
        per_risc.num_txn_ids = rc.config.num_txn_ids;
        for (int i = 0; i < ::experimental::NUM_TXN_IDS; i++) {
            per_risc.txn_ids[i] = rc.config.txn_ids[i];
        }
        per_risc.num_entries_per_txn_id = rc.config.num_entries_per_txn_id;
        per_risc.num_entries_per_txn_id_per_tc = rc.config.num_entries_per_txn_id_per_tc;
        per_risc.init_txn_id_descriptor = (uint8_t)rc.config.init_txn_id_descriptor;
        log_info(
            tt::LogMetal,
            "Per-risc txn: num_txn_ids={}, entries_per_txn_id={}, entries_per_txn_id_per_tc={}, "
            "init_txn_id_descriptor={}",
            rc.config.num_txn_ids,
            rc.config.num_entries_per_txn_id,
            rc.config.num_entries_per_txn_id_per_tc,
            rc.config.init_txn_id_descriptor);

        const auto* cfg_bytes = reinterpret_cast<const uint8_t*>(&per_risc);
        data.insert(data.end(), cfg_bytes, cfg_bytes + sizeof(per_risc));
    }

    log_info(tt::LogMetal, "Serialized DFB {} size: {}", this->id, data.size());

    return data;
}

uint32_t finalize_dfbs(
    uint32_t /*programmable_core_type_index*/,
    std::vector<std::shared_ptr<tt::tt_metal::KernelGroup>>& kernel_groups,
    const std::vector<std::shared_ptr<DataflowBufferImpl>>& dataflow_buffers,
    uint32_t base_offset,
    uint32_t& dfb_offset,
    uint32_t& dfb_size) {
    if (dataflow_buffers.empty()) {
        return base_offset;
    }

    const auto& hal = MetalContext::instance().hal();

    dfb_offset = base_offset;
    dfb_size = 0;

    for (auto& kg : kernel_groups) {
        auto kernel_config = kg->launch_msg.view().kernel_config();
        kernel_config.local_cb_offset() = base_offset;

        // Calculate total DFB size for this kernel group
        uint32_t kg_dfb_size = 0;
        for (const auto& dfb : dataflow_buffers) {
            // Check if this DFB overlaps with any core in the kernel group
            bool dfb_on_kg = false;
            for (const CoreRange& kg_range : kg->core_ranges.ranges()) {
                if (dfb->core_ranges.intersects(kg_range)) {
                    dfb_on_kg = true;
                    break;
                }
            }
            if (dfb_on_kg) {
                kg_dfb_size += dfb->serialized_size();
            }
        }

        // Track max across all kernel groups
        dfb_size = std::max(dfb_size, kg_dfb_size);
    }

    dfb_size = tt::align(
        dfb_size, 64);  // workaround where non-64 byte aligned writes on sim seem to get zero padded to 64 bytes

    log_info(
        tt::LogMetal,
        "Finalize dfb: dfb_offset == base_offset: {}, dfb size: {}, return value: {}",
        base_offset,
        dfb_size,
        tt::align(base_offset + dfb_size, hal.get_alignment(HalMemType::L1)));

    return tt::align(base_offset + dfb_size, hal.get_alignment(HalMemType::L1));
}

}  // namespace detail

}  // namespace tt::tt_metal::experimental::dfb

namespace tt::tt_metal::detail {

using namespace tt::tt_metal::experimental::dfb;
using namespace tt::tt_metal::experimental::dfb::detail;

uint32_t ProgramImpl::add_dataflow_buffer(const CoreRangeSet& core_range_set, const DataflowBufferConfig& config) {
    TT_FATAL(this->compiled_.empty(), "Cannot add dataflow buffer to an already compiled program {}", this->id);

    TT_FATAL(this->circular_buffers_.empty(), "Cannot add dataflow buffer to a program with circular buffers");

    TT_FATAL(config.entry_size > 0, "Entry size must be > 0");
    TT_FATAL(config.num_entries > 0, "Num entries must be > 0");
    TT_FATAL(config.producer_risc_mask != 0, "producer_risc_mask must have at least one bit set");
    TT_FATAL(config.consumer_risc_mask != 0, "consumer_risc_mask must have at least one bit set");
    TT_FATAL((config.producer_risc_mask & 0xFF00) == 0, "producer cannot be a Tensix yet");
    TT_FATAL((config.consumer_risc_mask & 0xFF00) == 0, "consumer cannot be a Tensix yet");
    TT_FATAL(
        (config.producer_risc_mask & config.consumer_risc_mask) == 0,
        "producer_risc_mask and consumer_risc_mask must not overlap");
    TT_FATAL(config.pap != ::experimental::AccessPattern::BLOCKED, "Blocked producer pattern not supported");
    // Implicit sync is now supported for DM RISCs
    TT_FATAL(
        core_range_set.num_cores() == 1,
        "DFB only supports single core, but CoreRangeSet contains {} cores: {}",
        core_range_set.num_cores(),
        core_range_set.str());
    TT_FATAL(
        config.cap != ::experimental::AccessPattern::BLOCKED || config.num_consumers <= 4,
        "Blocked consumer pattern supports at most 4 consumers, but {} were specified",
        config.num_consumers);

    auto dfb = std::make_shared<DataflowBufferImpl>();

    dfb->id = static_cast<uint32_t>(this->dataflow_buffers_.size());
    dfb->core_ranges = core_range_set.merge_ranges();
    dfb->config = config;

    dfb->risc_mask = config.producer_risc_mask | config.consumer_risc_mask;

    dfb->entry_size = config.entry_size;

    log_info(
        tt::LogMetal,
        "Creating DFB {} with {} producers (mask 0x{:x}) and {} consumers (mask 0x{:x})",
        dfb->id,
        config.num_producers,
        config.producer_risc_mask,
        config.num_consumers,
        config.consumer_risc_mask);

    uint32_t capacity;
    switch (config.cap) {
        case ::experimental::AccessPattern::STRIDED:
            TT_FATAL(
                config.num_entries % std::max(config.num_producers, config.num_consumers) == 0,
                "Num entries in DFB {} must be divisible by max of num producers and consumers {}",
                config.num_entries,
                std::max(config.num_producers, config.num_consumers));
            capacity = config.num_entries / std::max(config.num_producers, config.num_consumers);
            dfb->stride_size = config.entry_size * std::max(config.num_producers, config.num_consumers);
            break;
        case ::experimental::AccessPattern::BLOCKED:
            TT_FATAL(
                config.num_entries % config.num_producers == 0,
                "Num entries in DFB {} must be divisible by num producers {}",
                config.num_entries,
                config.num_producers);
            capacity = config.num_entries / config.num_producers;
            dfb->stride_size = config.entry_size;
            break;
        default: TT_FATAL(false, "Invalid access pattern", (uint32_t)config.cap);
    }
    dfb->capacity = capacity;
    log_info(tt::LogMetal, "Capacity: {}", capacity);

    if (config.cap == ::experimental::AccessPattern::BLOCKED) {
        dfb->remapper_consumer_mask = config.consumer_risc_mask & 0xFF;
    }

    uint8_t num_producer_tcs = calculate_num_tile_counters(config, true);
    uint8_t num_consumer_tcs = calculate_num_tile_counters(config, false);

    std::vector<uint8_t> producer_risc_ids;
    std::vector<uint8_t> consumer_risc_ids;
    for (uint8_t risc_id = 0; risc_id < 16; risc_id++) {
        if (config.producer_risc_mask & (1 << risc_id)) {
            producer_risc_ids.push_back(risc_id);
        }
        if (config.consumer_risc_mask & (1 << risc_id)) {
            consumer_risc_ids.push_back(risc_id);
        }
    }

    // Extract tensix IDs from risc masks
    std::vector<uint8_t> producer_tensix_ids = extract_tensix_ids(config.producer_risc_mask);
    std::vector<uint8_t> consumer_tensix_ids = extract_tensix_ids(config.consumer_risc_mask);
    bool has_tensix_riscs = !producer_tensix_ids.empty() || !consumer_tensix_ids.empty();

    // Determine tensix_id to use for allocation
    // If tensix RISCs are used, use those specific tensix_ids
    // If only DM RISCs, round-robin through 0-3 per producer-consumer pair
    auto get_tensix_id_for_pair = [&](uint8_t pair_index) -> uint8_t {
        if (has_tensix_riscs) {
            // Use tensix_ids from masks, round-robin if multiple
            std::vector<uint8_t> all_tensix_ids = producer_tensix_ids;
            all_tensix_ids.insert(all_tensix_ids.end(), consumer_tensix_ids.begin(), consumer_tensix_ids.end());
            if (all_tensix_ids.empty()) {
                return get_dm_tensix_id_for_pair(pair_index);
            }
            return all_tensix_ids[pair_index % all_tensix_ids.size()];
        }
        return get_dm_tensix_id_for_pair(pair_index);
    };

    std::vector<std::vector<TileCounterGroup>> tc_groups;  // [producer_idx][tc_slot]
    tc_groups.resize(producer_risc_ids.size());

    // For each producer, allocate TC groups (one per TC slot)
    for (size_t producer_idx = 0; producer_idx < producer_risc_ids.size(); producer_idx++) {
        tc_groups[producer_idx].resize(num_producer_tcs);

        for (uint8_t tc_slot = 0; tc_slot < num_producer_tcs; tc_slot++) {
            TileCounterGroup& group = tc_groups[producer_idx][tc_slot];

            if (config.cap == ::experimental::AccessPattern::STRIDED) {
                // Determine which consumer(s) this producer TC slot pairs with
                // Strided pairing: producer N pairs with consumers N, N+num_producers, N+2*num_producers, etc.
                uint8_t consumer_idx = (producer_idx + tc_slot * producer_risc_ids.size()) % consumer_risc_ids.size();

                // Determine tensix_id based on consumer (all producers pairing with same consumer use same tensix_id)
                uint8_t tensix_id = get_tensix_id_for_pair(consumer_idx);

                group.producer_tc = tile_counter_allocator_.allocate(tensix_id);
                group.consumer_tcs.push_back(group.producer_tc);  // Shared TC for strided

                log_info(
                    tt::LogMetal,
                    "Strided: Producer[{}] TC[{}] (tensix_id={}) pairs with Consumer[{}]",
                    producer_idx,
                    tc_slot,
                    tensix_id,
                    consumer_idx);
            } else if (config.cap == ::experimental::AccessPattern::BLOCKED) {
                // Determine tensix_id for this producer TC slot
                uint8_t tensix_id = get_tensix_id_for_pair((producer_idx * num_producer_tcs) + tc_slot);

                group.producer_tc = tile_counter_allocator_.allocate(tensix_id);

                // Allocate separate consumer TCs for Remapper 1-to-many mapping
                for (size_t consumer_idx = 0; consumer_idx < consumer_risc_ids.size(); consumer_idx++) {
                    uint8_t consumer_tensix_id = get_tensix_id_for_pair((producer_idx * num_producer_tcs) + tc_slot);
                    ::experimental::PackedTileCounter consumer_tc =
                        tile_counter_allocator_.allocate(consumer_tensix_id);
                    group.consumer_tcs.push_back(consumer_tc);
                }

                log_info(
                    tt::LogMetal,
                    "Blocked: Producer[{}] TC[{}] (tensix_id={}) maps to {} consumer TCs via Remapper",
                    producer_idx,
                    tc_slot,
                    tensix_id,
                    group.consumer_tcs.size());
            } else {
                TT_FATAL(false, "Unsupported consumer access pattern");
            }
        }
    }

    // Create producer risc_configs and assign TCs from groups
    CoreCoord core = dfb->core_ranges.ranges()[0].start_coord;
    for (size_t producer_idx = 0; producer_idx < producer_risc_ids.size(); producer_idx++) {
        uint8_t risc_id = producer_risc_ids[producer_idx];

        DFBRiscConfig risc_config;
        risc_config.risc_id = risc_id;
        risc_config.is_producer = true;
        risc_config.config.should_init_tc = true;  // producer is responsible for initializing tile counters
        risc_config.config.init_txn_id_descriptor = config.enable_implicit_sync && producer_idx == 0;

        log_info(tt::LogMetal, "Producer risc {} uses {} TCs", risc_id, num_producer_tcs);

        for (uint8_t tc = 0; tc < num_producer_tcs; tc++) {
            risc_config.config.packed_tile_counter[tc] = tc_groups[producer_idx][tc].producer_tc;
            log_info(
                tt::LogMetal,
                "\tAssigned TC[{}]: (0x{:x}, 0x{:x})",
                tc,
                (uint32_t)::experimental::get_tensix_id(risc_config.config.packed_tile_counter[tc]),
                (uint32_t)::experimental::get_counter_id(risc_config.config.packed_tile_counter[tc]));
        }
        risc_config.config.num_tcs_to_rr = num_producer_tcs;

        if (config.cap == ::experimental::AccessPattern::BLOCKED) {
            risc_config.config.remapper_pair_index = remapper_index_allocator_.allocate(core);
            const TileCounterGroup& group = tc_groups[producer_idx][0];
            uint32_t packed = 0;
            for (size_t i = 0; i < group.consumer_tcs.size() && i < ::experimental::MAX_NUM_TILE_COUNTERS_TO_RR; i++) {
                packed |= (::experimental::get_counter_id(group.consumer_tcs[i]) & 0x1F) << (i * 5);
            }
            risc_config.config.consumer_tcs = packed;
        }

        dfb->risc_configs.push_back(risc_config);
    }

    // Create consumer risc_configs and assign TCs from groups
    for (size_t consumer_idx = 0; consumer_idx < consumer_risc_ids.size(); consumer_idx++) {
        uint8_t risc_id = consumer_risc_ids[consumer_idx];

        DFBRiscConfig risc_config;
        risc_config.risc_id = risc_id;
        risc_config.is_producer = false;
        risc_config.config.init_txn_id_descriptor = config.enable_implicit_sync && consumer_idx == 0;

        log_info(tt::LogMetal, "Consumer risc {} uses {} TCs", risc_id, num_consumer_tcs);

        for (uint8_t tc = 0; tc < num_consumer_tcs; tc++) {
            if (config.cap == ::experimental::AccessPattern::STRIDED) {
                // Strided pairing inverse: find which producer pairs with this consumer
                // Producer pairing: consumer_idx = (producer_idx + tc_slot * num_producers) % num_consumers
                uint8_t producer_idx;
                uint8_t producer_tc_slot;

                if (producer_risc_ids.size() > consumer_risc_ids.size()) {
                    // More producers than consumers: each consumer pairs with multiple producers
                    // The t-th producer pairing with consumer C is: C + t * num_consumers
                    producer_idx = consumer_idx + tc * consumer_risc_ids.size();
                    producer_tc_slot = 0;
                } else if (consumer_risc_ids.size() > producer_risc_ids.size()) {
                    // More consumers than producers: each producer pairs with multiple consumers
                    producer_idx = consumer_idx % producer_risc_ids.size();
                    producer_tc_slot = consumer_idx / producer_risc_ids.size();
                } else {
                    // Equal: 1:1 mapping
                    producer_idx = consumer_idx;
                    producer_tc_slot = tc;
                }

                risc_config.config.packed_tile_counter[tc] = tc_groups[producer_idx][producer_tc_slot].producer_tc;
            } else if (config.cap == ::experimental::AccessPattern::BLOCKED) {
                // For blocked mode, each consumer has num_producers TCs, one per producer
                // The tc-th TC on this consumer pairs with the tc-th producer
                // Consumer uses its own allocated TC from consumer_tcs, not the producer's TC
                uint8_t producer_idx = tc;
                uint8_t producer_tc_slot = 0;
                risc_config.config.packed_tile_counter[tc] =
                    tc_groups[producer_idx][producer_tc_slot].consumer_tcs[consumer_idx];
            } else {
                TT_FATAL(false, "Unsupported consumer access pattern");
            }
            log_info(
                tt::LogMetal,
                "\tAssigned TC[{}]: (0x{:x}, 0x{:x})",
                tc,
                (uint32_t)::experimental::get_tensix_id(risc_config.config.packed_tile_counter[tc]),
                (uint32_t)::experimental::get_counter_id(risc_config.config.packed_tile_counter[tc]));
        }
        risc_config.config.num_tcs_to_rr = num_consumer_tcs;

        dfb->risc_configs.push_back(risc_config);
    }

    log_info(tt::LogMetal, "DFB {} risc_mask: 0x{:x}", dfb->id, dfb->risc_mask);

    // Transaction ID assignment for implicit sync
    if (config.enable_implicit_sync) {
        bool producer_is_dm = (config.producer_risc_mask & 0xFF) != 0;
        bool consumer_is_dm = (config.consumer_risc_mask & 0xFF) != 0;

        std::array<uint8_t, 2> producer_txn_ids = {0xFF, 0xFF};
        std::array<uint8_t, 2> consumer_txn_ids = {0xFF, 0xFF};
        uint8_t num_producer_txn_ids = 0;
        uint8_t num_consumer_txn_ids = 0;

        if (producer_is_dm) {
            producer_txn_ids = txn_id_allocator_.allocate();
            num_producer_txn_ids = 2;
            log_info(tt::LogMetal, "Allocated producer txn IDs: {}, {}", producer_txn_ids[0], producer_txn_ids[1]);
        }
        if (consumer_is_dm) {
            consumer_txn_ids = txn_id_allocator_.allocate();
            num_consumer_txn_ids = 2;
            log_info(tt::LogMetal, "Allocated consumer txn IDs: {}, {}", consumer_txn_ids[0], consumer_txn_ids[1]);
        }

        // Calculate entry thresholds (stored in dfb_initializer_t)
        uint8_t threshold_producer = producer_is_dm ? (capacity / num_producer_txn_ids) : 0;
        uint8_t threshold_consumer = consumer_is_dm ? (capacity / num_consumer_txn_ids) : 0;
        dfb->num_entries_to_process_threshold_producer = threshold_producer;
        dfb->num_entries_to_process_threshold_consumer = threshold_consumer;

        // Calculate per-risc values
        uint8_t entries_to_post_per_txn_id = producer_is_dm ? (threshold_producer / config.num_producers) : 0;
        uint8_t entries_to_ack_per_txn_id = consumer_is_dm ? (threshold_consumer / config.num_consumers) : 0;

        // Assign to risc configs - only DM RISCs get valid txn IDs
        for (auto& risc_config : dfb->risc_configs) {
            bool is_dm_risc = (risc_config.risc_id < 8);  // DM RISCs are 0-7
            if (is_dm_risc) {
                if (risc_config.is_producer) {
                    risc_config.config.txn_ids[0] = producer_txn_ids[0];
                    risc_config.config.txn_ids[1] = producer_txn_ids[1];
                    risc_config.config.num_txn_ids = num_producer_txn_ids;
                    risc_config.config.num_entries_per_txn_id = entries_to_post_per_txn_id;
                    risc_config.config.num_entries_per_txn_id_per_tc =
                        entries_to_post_per_txn_id / risc_config.config.num_tcs_to_rr;
                    log_info(
                        tt::LogMetal,
                        "Producer RISC {} txn_ids=[{},{}] entries_per_txn_id={} entries_per_txn_id_per_tc={}",
                        risc_config.risc_id,
                        producer_txn_ids[0],
                        producer_txn_ids[1],
                        entries_to_post_per_txn_id,
                        risc_config.config.num_entries_per_txn_id_per_tc);
                } else {
                    risc_config.config.txn_ids[0] = consumer_txn_ids[0];
                    risc_config.config.txn_ids[1] = consumer_txn_ids[1];
                    risc_config.config.num_txn_ids = num_consumer_txn_ids;
                    risc_config.config.num_entries_per_txn_id = entries_to_ack_per_txn_id;
                    risc_config.config.num_entries_per_txn_id_per_tc =
                        entries_to_ack_per_txn_id / risc_config.config.num_tcs_to_rr;
                    log_info(
                        tt::LogMetal,
                        "Consumer RISC {} txn_ids=[{},{}] entries_per_txn_id={} entries_per_txn_id_per_tc={}",
                        risc_config.risc_id,
                        consumer_txn_ids[0],
                        consumer_txn_ids[1],
                        entries_to_ack_per_txn_id,
                        risc_config.config.num_entries_per_txn_id_per_tc);
                }
            }
        }
    }

    this->dataflow_buffers_.push_back(dfb);
    this->dataflow_buffer_by_id_.insert({dfb->id, dfb});

    for (const CoreRange& core_range : dfb->core_ranges.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                per_core_num_dfbs_[logical_core]++;
            }
        }
    }

    this->local_dataflow_buffer_allocation_needed_ = true;

    return dfb->id;
}

void ProgramImpl::invalidate_dataflow_buffer_allocation() {
    if (this->local_dataflow_buffer_allocation_needed_) {
        return;
    }
    for (CircularBufferAllocator& dfb_allocator : this->dfb_allocators_) {
        dfb_allocator.reset_available_addresses();
    }
    this->local_dataflow_buffer_allocation_needed_ = true;
}

// Same as allocate_circular_buffers
void ProgramImpl::allocate_dataflow_buffers(const IDevice* device) {
    if (not this->local_dataflow_buffer_allocation_needed_) {
        return;
    }

    uint64_t base_dfb_address = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    for (auto& dfb : this->dataflow_buffers_) {
        uint64_t computed_addr = base_dfb_address;
        for (const CoreRange& core_range : dfb->core_ranges.ranges()) {
            // Need the max available address across all cores dataflow buffer is placed on
            for (const CircularBufferAllocator& dfb_allocator : this->dfb_allocators_) {
                if (dfb_allocator.core_range == core_range) {
                    computed_addr = std::max(computed_addr, dfb_allocator.get_cb_region_end());
                    break;
                }
            }
        }
        computed_addr = align(computed_addr, device->allocator()->get_alignment(BufferType::DRAM));
        for (const CoreRange& core_range : dfb->core_ranges.ranges()) {
            for (CircularBufferAllocator& dfb_allocator : this->dfb_allocators_) {
                if (dfb_allocator.core_range.intersects(core_range)) {
                    if (dfb_allocator.core_range != core_range and computed_addr < dfb_allocator.get_cb_region_end()) {
                        // Intersecting core range has already been marked to have allocation at this address. This
                        // could have been marked by a dataflow buffer on a core range disjoint from current
                        // `core_range` but also intersecting `dfb_allocator.core_range`
                        continue;
                    }
                    dfb_allocator.mark_address(computed_addr, dfb->total_size(), base_dfb_address);
                }
            }
        }
        dfb->allocated_address = computed_addr;

        // Populate base_addr[] and limit[] arrays for each risc config.
        // Layout is column-major by (tile_counter, producer/consumer): all producers' tc0 first, then all tc1, etc.
        uint32_t entry_size = dfb->config.entry_size;
        uint32_t max_prod_cons = std::max(dfb->config.num_producers, dfb->config.num_consumers);

        uint8_t num_producer_tcs = 0;
        for (const auto& rc : dfb->risc_configs) {
            if (rc.is_producer) {
                num_producer_tcs = rc.config.num_tcs_to_rr;
                break;
            }
        }
        uint32_t base_addr = static_cast<uint32_t>(computed_addr);
        for (uint8_t tc = 0; tc < num_producer_tcs; tc++) {
            for (auto& rc : dfb->risc_configs) {
                if (rc.is_producer && tc < rc.config.num_tcs_to_rr) {
                    rc.config.base_addr[tc] = base_addr;
                    rc.config.limit[tc] =
                        rc.config.base_addr[tc] + ((entry_size * max_prod_cons) * (dfb->capacity - 1)) + entry_size;
                    base_addr += entry_size;
                }
            }
        }

        uint8_t num_consumer_tcs = 0;
        for (const auto& rc : dfb->risc_configs) {
            if (!rc.is_producer) {
                num_consumer_tcs = rc.config.num_tcs_to_rr;
                break;
            }
        }
        base_addr = static_cast<uint32_t>(computed_addr);
        for (uint8_t tc = 0; tc < num_consumer_tcs; tc++) {
            for (auto& rc : dfb->risc_configs) {
                if (rc.is_producer) {
                    continue;
                }
                rc.config.base_addr[tc] = base_addr;
                rc.config.limit[tc] =
                    rc.config.base_addr[tc] + ((entry_size * max_prod_cons) * (dfb->capacity - 1)) + entry_size;
                if (dfb->config.cap == ::experimental::AccessPattern::STRIDED && tc < rc.config.num_tcs_to_rr) {
                    base_addr += entry_size;
                }
            }
        }
    }
    this->local_dataflow_buffer_allocation_needed_ = false;
}

// Same as validate_circular_buffer_region
void ProgramImpl::validate_dataflow_buffer_region(const IDevice* device) {
    std::optional<DeviceAddr> lowest_address =
        device->lowest_occupied_compute_l1_address(this->determine_sub_device_ids(device));
    uint32_t max_l1_size = device->l1_size_per_core();

    for (const CircularBufferAllocator& dfb_allocator : this->dfb_allocators_) {
        if (dfb_allocator.l1_regions.empty()) {
            continue;
        }
        uint64_t dfb_region_end = dfb_allocator.l1_regions.back().second;
        if (dfb_region_end > max_l1_size) {
            TT_THROW(
                "Statically allocated dataflow buffers on core range {} grow to {} B which is beyond max L1 size of {} "
                "B",
                dfb_allocator.core_range.str(),
                dfb_region_end,
                max_l1_size);
        }
        if (lowest_address.has_value() and lowest_address.value() < dfb_region_end) {
            TT_THROW(
                "Statically allocated dataflow buffers in program {} clash with L1 buffers on core range {}. L1 buffer "
                "allocated at {} and static dataflow buffer region ends at {}",
                this->id,
                dfb_allocator.core_range.str(),
                lowest_address.value(),
                dfb_region_end);
        }
    }
}

std::vector<std::shared_ptr<tt::tt_metal::experimental::dfb::detail::DataflowBufferImpl>>
ProgramImpl::dataflow_buffers_on_core(const CoreCoord& core) const {
    std::vector<std::shared_ptr<tt::tt_metal::experimental::dfb::detail::DataflowBufferImpl>> dfbs_on_core;
    for (const auto& dfb : dataflow_buffers_) {
        if (dfb->core_ranges.intersects(core)) {
            dfbs_on_core.push_back(dfb);
        }
    }
    return dfbs_on_core;
}

}  // namespace tt::tt_metal::detail
