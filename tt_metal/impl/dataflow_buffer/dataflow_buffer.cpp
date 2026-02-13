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

uint8_t ClientTypeAllocator::allocate_for_consumer(uint8_t producer_tensix_id, uint8_t consumer_risc_id) {
    uint8_t client_type;

    if (consumer_risc_id >= 8) {
        // Tensix RISC: risc_id 8-11 -> clientType 4-7 (NEO_0 to NEO_3)
        // Derive id_R directly from consumer's RISC ID
        client_type = 4 + (consumer_risc_id - 8);

        // Validate: Tensix consumer's tensix_id must not conflict with producer's tensix_id
        TT_FATAL(
            (client_type % 4) != producer_tensix_id,
            "Tensix consumer risc_id {} (tensix_id={}) conflicts with producer tensix_id {}",
            consumer_risc_id,
            client_type % 4,
            producer_tensix_id);
    } else {
        // DM RISC: find first available clientType whose tensix_id != producer_tensix_id
        client_type = 0xFF;  // Invalid until found
        for (uint8_t ct = 0; ct < 8; ct++) {
            if (!(used_mask_ & (1u << ct)) && (ct % 4) != producer_tensix_id) {
                client_type = ct;
                break;
            }
        }
        TT_FATAL(client_type != 0xFF, "Out of client types for BLOCKED DM consumer allocation");
    }

    // Mark this clientType as used
    TT_FATAL(!(used_mask_ & (1u << client_type)), "ClientType {} already used", client_type);
    used_mask_ |= (1u << client_type);

    return client_type;
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

bool has_dm_risc(uint16_t risc_mask) { return (risc_mask & 0xFF) != 0; }

bool has_tensix_risc(uint16_t risc_mask) { return (risc_mask & 0x0F00) != 0; }

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
    TT_FATAL(this->configs_finalized, "DFB {} configs not finalized before serialization", this->id);

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
    init.num_txn_ids = this->num_txn_ids;
    for (int i = 0; i < 4; i++) {
        init.txn_ids[i] = this->txn_ids[i];
    }
    init.num_entries_per_txn_id = this->num_entries_per_txn_id;
    init.num_entries_per_txn_id_per_tc = this->num_entries_per_txn_id_per_tc;
    init.padding = 0;

    log_info(
        tt::LogMetal,
        "Serializing DFB {} with {} producers and {} consumers. risc_mask: 0x{:x} use_remapper: {}",
        this->id,
        this->config.num_producers,
        this->config.num_consumers,
        this->risc_mask,
        this->use_remapper);

    log_info(tt::LogMetal, "Entry size: {}", this->entry_size);
    log_info(tt::LogMetal, "Stride size: {}", this->stride_size);
    log_info(tt::LogMetal, "Capacity: {}", this->capacity);
    log_info(tt::LogMetal, "Risc mask: 0x{:x}", this->risc_mask);
    log_info(tt::LogMetal, "Num txn ids: {}", this->num_txn_ids);
    for (int i = 0; i < ::experimental::NUM_TXN_IDS; i++) {
        log_info(tt::LogMetal, "Txn id {}: {}", i, this->txn_ids[i]);
    }
    log_info(tt::LogMetal, "Num entries per txn id: {}", this->num_entries_per_txn_id);
    log_info(tt::LogMetal, "Num entries per txn id per tc: {}", this->num_entries_per_txn_id_per_tc);

    const auto* init_bytes = reinterpret_cast<const uint8_t*>(&init);
    data.insert(data.end(), init_bytes, init_bytes + sizeof(init));

    // Write one dfb_initializer_per_risc_t per risc
    for (const auto& rc : risc_configs) {
        log_info(tt::LogMetal, "New risc config (risc_id={}, is_producer={})", rc.risc_id, rc.is_producer);
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
        per_risc.flags.remapper_en = this->use_remapper;
        per_risc.flags.should_init_tc = rc.config.should_init_tc;
        per_risc.consumer_tcs = rc.config.consumer_tcs;
        // Per-producer remapper fields
        per_risc.remapper_consumer_ids_mask = rc.config.remapper_consumer_ids_mask;
        per_risc.producer_client_type = rc.config.producer_client_type;
        log_info(tt::LogMetal, "Should init tc: {}", rc.config.should_init_tc);
        log_info(tt::LogMetal, "Remapper en: {}", this->use_remapper);
        if (this->use_remapper && rc.is_producer) {
            log_info(
                tt::LogMetal,
                "Producer remapper: pair_idx={}, clientL={}, consumer_ids_mask=0x{:02x}",
                rc.config.remapper_pair_index,
                rc.config.producer_client_type,
                rc.config.remapper_consumer_ids_mask);
        }
        log_info(tt::LogMetal, "Should init tc: {}", rc.config.should_init_tc);

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
    TT_FATAL(
        (config.producer_risc_mask & config.consumer_risc_mask) == 0,
        "producer_risc_mask and consumer_risc_mask must not overlap");

    bool producer_has_dm = has_dm_risc(config.producer_risc_mask);
    bool consumer_has_dm = has_dm_risc(config.consumer_risc_mask);
    bool producer_is_tensix_only = !producer_has_dm && has_tensix_risc(config.producer_risc_mask);
    bool consumer_is_tensix_only = !consumer_has_dm && has_tensix_risc(config.consumer_risc_mask);
    TT_FATAL(
        !(producer_is_tensix_only && consumer_is_tensix_only),
        "Both producer and consumer cannot be Tensix-only RISCs - at least one DM RISC is required to initialize tile "
        "counters");
    TT_FATAL(
        !(producer_is_tensix_only && config.cap == ::experimental::AccessPattern::BLOCKED),
        "Tensix producer with BLOCKED consumer pattern is not supported");
    TT_FATAL(config.pap != ::experimental::AccessPattern::BLOCKED, "Blocked producer pattern not supported");
    TT_FATAL(!config.enable_implicit_sync, "Implicit sync not supported yet");
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

    dfb->configs_finalized = false;

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

// Finalize DFB configurations - allocate TCs and remapper indices
// This must be called after all add_dataflow_buffer calls are complete to determine whether remapper needs to be
// enabled on the core
void ProgramImpl::finalize_dataflow_buffer_configs() {
    if (this->dataflow_buffers_.empty()) {
        return;
    }

    // Group DFBs by core
    std::unordered_map<CoreCoord, std::vector<std::shared_ptr<DataflowBufferImpl>>> dfbs_by_core;
    for (auto& dfb : this->dataflow_buffers_) {
        if (dfb->configs_finalized) {
            continue;
        }
        CoreCoord core = dfb->core_ranges.ranges()[0].start_coord;
        dfbs_by_core[core].push_back(dfb);
    }

    // Process each core's DFBs together
    for (auto& [core, core_dfbs] : dfbs_by_core) {
        bool core_needs_remapper = false;
        for (const auto& dfb : core_dfbs) {
            if (dfb->config.cap == ::experimental::AccessPattern::BLOCKED) {
                core_needs_remapper = true;
                break;
            }
        }

        log_info(
            tt::LogMetal,
            "Finalizing {} DFBs on core ({}, {}), core_needs_remapper={}",
            core_dfbs.size(),
            core.x,
            core.y,
            core_needs_remapper);

        // Allocate TCs and remapper configs for all DFBs on this core
        for (auto& dfb : core_dfbs) {
            finalize_single_dfb_config(dfb, core, core_needs_remapper);
        }
    }
}

void ProgramImpl::finalize_single_dfb_config(
    std::shared_ptr<DataflowBufferImpl>& dfb, const CoreCoord& core, bool use_remapper) {
    const auto& config = dfb->config;

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

    // Determine tensix_id based on which RISC in the pair is Tensix
    // Without remapper,Tensix RISCs can only access TCs from their own tensix_id
    // DM RISCs can access any of the 64 TC available to them
    auto get_tensix_id_for_pair =
        [&](uint8_t producer_risc_id, uint8_t consumer_risc_id, uint8_t pair_counter) -> uint8_t {
        if (producer_risc_id >= 8) {
            // Producer is Tensix: must use producer's tensix_id
            return (producer_risc_id - 8) % 4;
        } else if (consumer_risc_id >= 8) {
            // Consumer is Tensix: must use consumer's tensix_id
            return (consumer_risc_id - 8) % 4;
        } else {
            // Both DM: round-robin across tensix_ids
            return get_dm_tensix_id_for_pair(pair_counter);
        }
    };

    std::vector<std::vector<TileCounterGroup>> tc_groups;  // [producer_idx][tc_slot]
    tc_groups.resize(producer_risc_ids.size());

    // For remapper mode, pre-allocate clientTypes for each consumer
    // Also allocate producer clientTypes (clientL)
    std::vector<uint8_t> consumer_client_types;
    std::vector<uint8_t> producer_client_types;
    ClientTypeAllocator client_type_allocator;

    if (use_remapper) {
        // First allocate producer clientTypes (clientL)
        for (size_t producer_idx = 0; producer_idx < producer_risc_ids.size(); producer_idx++) {
            uint8_t producer_risc_id = producer_risc_ids[producer_idx];
            // Producer clientType: based on producer's risc_id
            uint8_t producer_client_type;
            if (producer_risc_id >= 8) {
                // Tensix producer: use NEO clientType (4-7)
                producer_client_type = 4 + (producer_risc_id - 8);
            } else {
                // DM producer: use DM clientType (0-3), based on risc_id % 4
                producer_client_type = producer_risc_id % 4;
            }
            producer_client_types.push_back(producer_client_type);
            log_info(
                tt::LogMetal,
                "Remapper: Producer[{}] (risc_id={}) assigned clientL={}",
                producer_idx,
                producer_risc_id,
                producer_client_type);
        }

        // Then allocate consumer clientTypes (clientR) - must differ from producer's clientL
        uint8_t producer_tensix_id = producer_risc_ids[0] % 4;
        for (size_t consumer_idx = 0; consumer_idx < consumer_risc_ids.size(); consumer_idx++) {
            uint8_t consumer_risc_id = consumer_risc_ids[consumer_idx];
            uint8_t client_type = client_type_allocator.allocate_for_consumer(producer_tensix_id, consumer_risc_id);
            consumer_client_types.push_back(client_type);

            log_info(
                tt::LogMetal,
                "Remapper: Consumer[{}] (risc_id={}) assigned clientR={} (tensix_id={})",
                consumer_idx,
                consumer_risc_id,
                client_type,
                ClientTypeAllocator::get_tensix_id(client_type));
        }
    }

    // For each producer, allocate TC groups (one per TC slot)
    uint8_t pair_counter = 0;  // For round-robin tensix_id when both producer and consumer are DM
    for (size_t producer_idx = 0; producer_idx < producer_risc_ids.size(); producer_idx++) {
        tc_groups[producer_idx].resize(num_producer_tcs);

        for (uint8_t tc_slot = 0; tc_slot < num_producer_tcs; tc_slot++) {
            TileCounterGroup& group = tc_groups[producer_idx][tc_slot];

            if (config.cap == ::experimental::AccessPattern::STRIDED) {
                // Determine which consumer(s) this producer TC slot pairs with
                uint8_t consumer_idx = (producer_idx + tc_slot * producer_risc_ids.size()) % consumer_risc_ids.size();

                uint8_t producer_risc_id = producer_risc_ids[producer_idx];
                uint8_t consumer_risc_id = consumer_risc_ids[consumer_idx];
                uint8_t tensix_id = get_tensix_id_for_pair(producer_risc_id, consumer_risc_id, pair_counter++);

                group.producer_tc = tile_counter_allocator_.allocate(tensix_id);

                if (use_remapper) {
                    // With remapper: allocate separate consumer TC
                    uint8_t consumer_tensix_id =
                        ClientTypeAllocator::get_tensix_id(consumer_client_types[consumer_idx]);
                    ::experimental::PackedTileCounter consumer_tc =
                        tile_counter_allocator_.allocate(consumer_tensix_id);
                    group.consumer_tcs.push_back(consumer_tc);
                } else {
                    // Without remapper: shared TC
                    group.consumer_tcs.push_back(group.producer_tc);
                }

                log_info(
                    tt::LogMetal,
                    "Strided: Producer[{}] (risc_id={}) TC[{}] (tensix_id={}) pairs with Consumer[{}] (risc_id={}) "
                    "use_remapper={}",
                    producer_idx,
                    producer_risc_id,
                    tc_slot,
                    tensix_id,
                    consumer_idx,
                    consumer_risc_id,
                    use_remapper);
            } else if (config.cap == ::experimental::AccessPattern::BLOCKED) {
                // Producer TC: use producer's risc_id % 4 as tensix_id
                uint8_t producer_tensix_id = producer_risc_ids[producer_idx] % 4;

                group.producer_tc = tile_counter_allocator_.allocate(producer_tensix_id);

                // Allocate separate consumer TCs for Remapper 1-to-many mapping
                for (size_t consumer_idx = 0; consumer_idx < consumer_risc_ids.size(); consumer_idx++) {
                    uint8_t consumer_tensix_id =
                        ClientTypeAllocator::get_tensix_id(consumer_client_types[consumer_idx]);
                    ::experimental::PackedTileCounter consumer_tc =
                        tile_counter_allocator_.allocate(consumer_tensix_id);
                    group.consumer_tcs.push_back(consumer_tc);
                }

                log_info(
                    tt::LogMetal,
                    "Blocked: Producer[{}] TC[{}] (tensix_id={}) maps to {} consumer TCs via Remapper",
                    producer_idx,
                    tc_slot,
                    producer_tensix_id,
                    group.consumer_tcs.size());
            } else {
                TT_FATAL(false, "Unsupported consumer access pattern");
            }
        }
    }

    // Create producer risc_configs and assign TCs from groups
    for (size_t producer_idx = 0; producer_idx < producer_risc_ids.size(); producer_idx++) {
        uint8_t risc_id = producer_risc_ids[producer_idx];

        DFBRiscConfig risc_config;
        risc_config.risc_id = risc_id;
        risc_config.is_producer = true;
        // Producer inits TCs only if it's a DM RISC (Tensix RISCs can't run TC init code)
        bool producer_is_dm = risc_id < 8;
        risc_config.config.should_init_tc = producer_is_dm;

        log_info(
            tt::LogMetal,
            "Producer risc {} uses {} TCs (should_init_tc={})",
            risc_id,
            num_producer_tcs,
            risc_config.config.should_init_tc);

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

        if (use_remapper) {
            risc_config.config.remapper_pair_index = remapper_index_allocator_.allocate(core);
            risc_config.config.producer_client_type = producer_client_types[producer_idx];

            // Build consumer_tcs packed and consumer_ids_mask for this producer
            const TileCounterGroup& group = tc_groups[producer_idx][0];
            uint32_t packed = 0;
            uint8_t consumer_ids_mask = 0;

            if (config.cap == ::experimental::AccessPattern::BLOCKED) {
                // BLOCKED: 1-to-many, all consumers
                for (size_t i = 0; i < group.consumer_tcs.size() && i < ::experimental::MAX_NUM_TILE_COUNTERS_TO_RR;
                     i++) {
                    packed |= (::experimental::get_counter_id(group.consumer_tcs[i]) & 0x1F) << (i * 5);
                    consumer_ids_mask |= (1u << consumer_client_types[i]);
                }
            } else {
                // STRIDED via remapper: 1-to-1 per tc_slot
                for (uint8_t tc = 0; tc < num_producer_tcs; tc++) {
                    uint8_t consumer_idx = (producer_idx + tc * producer_risc_ids.size()) % consumer_risc_ids.size();
                    packed |= (::experimental::get_counter_id(tc_groups[producer_idx][tc].consumer_tcs[0]) & 0x1F)
                              << (tc * 5);
                    consumer_ids_mask |= (1u << consumer_client_types[consumer_idx]);
                }
            }
            risc_config.config.consumer_tcs = packed;
            risc_config.config.remapper_consumer_ids_mask = consumer_ids_mask;

            log_info(
                tt::LogMetal,
                "Producer[{}] remapper: pair_idx={}, clientL={}, consumer_ids_mask=0x{:02x}",
                producer_idx,
                risc_config.config.remapper_pair_index,
                risc_config.config.producer_client_type,
                risc_config.config.remapper_consumer_ids_mask);
        }

        dfb->risc_configs.push_back(risc_config);
    }

    // Create consumer risc_configs and assign TCs from groups
    // If producer is Tensix, the first DM consumer should init TCs
    bool need_consumer_dm_init = !producer_risc_ids.empty() && producer_risc_ids[0] >= 8;
    bool first_dm_consumer_assigned = false;

    for (size_t consumer_idx = 0; consumer_idx < consumer_risc_ids.size(); consumer_idx++) {
        uint8_t risc_id = consumer_risc_ids[consumer_idx];

        DFBRiscConfig risc_config;
        risc_config.risc_id = risc_id;
        risc_config.is_producer = false;

        // Determine if this consumer should init TCs
        bool is_dm_risc = risc_id < 8;
        if (need_consumer_dm_init && is_dm_risc && !first_dm_consumer_assigned) {
            risc_config.config.should_init_tc = true;
            first_dm_consumer_assigned = true;
        } else {
            risc_config.config.should_init_tc = false;
        }

        log_info(
            tt::LogMetal,
            "Consumer risc {} uses {} TCs (should_init_tc={})",
            risc_id,
            num_consumer_tcs,
            risc_config.config.should_init_tc);

        for (uint8_t tc = 0; tc < num_consumer_tcs; tc++) {
            if (config.cap == ::experimental::AccessPattern::STRIDED) {
                uint8_t producer_idx;
                uint8_t producer_tc_slot;

                if (producer_risc_ids.size() > consumer_risc_ids.size()) {
                    producer_idx = consumer_idx + tc * consumer_risc_ids.size();
                    producer_tc_slot = 0;
                } else if (consumer_risc_ids.size() > producer_risc_ids.size()) {
                    producer_idx = consumer_idx % producer_risc_ids.size();
                    producer_tc_slot = consumer_idx / producer_risc_ids.size();
                } else {
                    producer_idx = consumer_idx;
                    producer_tc_slot = tc;
                }

                if (use_remapper) {
                    // With remapper: consumer uses its own TC
                    risc_config.config.packed_tile_counter[tc] =
                        tc_groups[producer_idx][producer_tc_slot].consumer_tcs[0];
                } else {
                    // Without remapper: shared TC with producer
                    risc_config.config.packed_tile_counter[tc] = tc_groups[producer_idx][producer_tc_slot].producer_tc;
                }
            } else if (config.cap == ::experimental::AccessPattern::BLOCKED) {
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

    dfb->configs_finalized = true;
    dfb->use_remapper = use_remapper;
    log_info(
        tt::LogMetal, "DFB {} finalized risc_mask: 0x{:x} use_remapper: {}", dfb->id, dfb->risc_mask, use_remapper);
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
