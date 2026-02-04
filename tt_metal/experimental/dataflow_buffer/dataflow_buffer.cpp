// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/experimental/dataflow_buffer/dataflow_buffer.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
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
    // 16 exposed to overlay
    // TODO: Update for remapper
    TT_FATAL(next_tc_id_ < 16, "Out of tile counters for tensix {}", (uint32_t)tensix_id);
    uint8_t tc_id = next_tc_id_++;
    return static_cast<::experimental::PackedTileCounter>((tensix_id << 5) | tc_id);
}

uint8_t calculate_num_tile_counters(const DataflowBufferConfig& config, bool is_producer) {
    if (config.cap == ::experimental::AccessPattern::BLOCKED) {
        return is_producer ? config.num_consumers : 1;
    }
    return (config.num_consumers + config.num_producers - 1) / config.num_producers;
}

::experimental::PackedTileCounter get_shared_tc_for_consumer(
    const DataflowBufferImpl* dfb, uint8_t consumer_idx, uint8_t tc_idx) {
    // In strided mode, consumers share TCs with producers (unless remapper is used and we have diff 1:1 remappings)
    // TODO: this needs to be updated when remapper is added
    uint8_t producer_idx = (consumer_idx * dfb->config.num_producers) / dfb->config.num_consumers;
    return dfb->risc_configs[producer_idx].config.packed_tile_counter[tc_idx];
}

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
    init.remapper_pair_index = this->remapper_pair_index;
    init.num_txn_ids = this->num_txn_ids;
    for (int i = 0; i < 4; i++) {
        init.txn_ids[i] = this->txn_ids[i];
    }
    init.num_entries_per_txn_id = this->num_entries_per_txn_id;
    init.num_entries_per_txn_id_per_tc = this->num_entries_per_txn_id_per_tc;

    log_info(
        tt::LogMetal,
        "Serializing DFB {} with {} producers and {} consumers. risc_mask: 0x{:x}",
        this->id,
        this->config.num_producers,
        this->config.num_consumers,
        this->risc_mask);

    const auto* init_bytes = reinterpret_cast<const uint8_t*>(&init);
    data.insert(data.end(), init_bytes, init_bytes + sizeof(init));

    // Write one dfb_initializer_per_risc_t per risc
    for (const auto& rc : risc_configs) {
        ::experimental::dfb_initializer_per_risc_t per_risc = {};

        // Copy per-risc arrays
        for (int i = 0; i < 4; i++) {
            per_risc.base_addr[i] = rc.config.base_addr[i];
            per_risc.limit[i] = rc.config.limit[i];
            per_risc.packed_tile_counter[i] = rc.config.packed_tile_counter[i];
        }
        per_risc.num_tcs_to_rr = rc.config.num_tcs_to_rr;
        per_risc.should_init_tc = rc.config.should_init_tc ? 1 : 0;

        log_info(
            tt::LogMetal,
            "\tRisc {}: base_addr[0]={}, limit[0]={}, num_tcs_to_rr={}",
            rc.risc_id,
            rc.config.base_addr[0],
            rc.config.limit[0],
            rc.config.num_tcs_to_rr);

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

    const auto& hal = get_hal();

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
    TT_FATAL(config.num_producers == 1, "DFB only supports one producer for now");
    TT_FATAL(config.num_consumers == 1, "DFB only supports one consumer for now");
    TT_FATAL(config.pap != ::experimental::AccessPattern::BLOCKED, "Blocked producer pattern not supported");
    TT_FATAL(config.cap != ::experimental::AccessPattern::BLOCKED, "Blocked consumer pattern not supported yet");
    TT_FATAL(!config.enable_implicit_sync, "Implicit sync not supported yet");
    TT_FATAL(
        core_range_set.num_cores() == 1,
        "DFB only supports single core, but CoreRangeSet contains {} cores: {}",
        core_range_set.num_cores(),
        core_range_set.str());

    auto dfb = std::make_shared<DataflowBufferImpl>();

    // Assign logical ID (0, 1, 2, ...)
    dfb->id = static_cast<uint32_t>(this->dataflow_buffers_.size());
    dfb->core_ranges = core_range_set.merge_ranges();
    dfb->config = config;

    // Use risc_mask from config
    dfb->risc_mask = config.producer_risc_mask | config.consumer_risc_mask;

    // Set shared config fields
    dfb->entry_size = config.entry_size;
    dfb->stride_size = config.entry_size * std::max(config.num_producers, config.num_consumers);

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
            break;
        case ::experimental::AccessPattern::BLOCKED:
            TT_FATAL(
                config.num_entries % config.num_producers == 0,
                "Num entries in DFB {} must be divisible by num producers {}",
                config.num_entries,
                config.num_producers);
            capacity = config.num_entries / config.num_producers;
            break;
        default: TT_FATAL(false, "Invalid access pattern", (uint32_t)config.cap);
    }
    dfb->capacity = capacity;

    uint8_t num_producer_tcs = calculate_num_tile_counters(config, true);
    uint8_t num_consumer_tcs = calculate_num_tile_counters(config, false);

    // Iterate over producer_risc_mask to create producer risc_configs
    for (uint8_t risc_id = 0; risc_id < 16; risc_id++) {
        if (!(config.producer_risc_mask & (1 << risc_id))) {
            continue;
        }

        DFBRiscConfig risc_config;
        risc_config.risc_id = risc_id;
        risc_config.is_producer = true;
        risc_config.config.should_init_tc = true;  // producer is responsible for initializing tile counters

        log_info(tt::LogMetal, "Producer risc {} uses {} TCs", risc_id, num_producer_tcs);

        // Fill arrays for round-robin TCs
        for (uint8_t tc = 0; tc < num_producer_tcs; tc++) {
            risc_config.config.packed_tile_counter[tc] = tile_counter_allocator_.allocate(risc_id);
            log_info(tt::LogMetal, "\tAssigned TC[{}]: {}", tc, (uint32_t)risc_config.config.packed_tile_counter[tc]);
        }
        risc_config.config.num_tcs_to_rr = num_producer_tcs;

        dfb->risc_configs.push_back(risc_config);
    }

    // Iterate over consumer_risc_mask to create consumer risc_configs
    uint8_t consumer_count = 0;
    for (uint8_t risc_id = 0; risc_id < 16; risc_id++) {
        if (!(config.consumer_risc_mask & (1 << risc_id))) {
            continue;
        }

        DFBRiscConfig risc_config;
        risc_config.risc_id = risc_id;
        risc_config.is_producer = false;

        log_info(tt::LogMetal, "Consumer risc {} uses {} TCs", risc_id, num_consumer_tcs);

        // Fill arrays for round-robin TCs
        for (uint8_t tc = 0; tc < num_consumer_tcs; tc++) {
            if (config.cap == ::experimental::AccessPattern::STRIDED) {
                risc_config.config.packed_tile_counter[tc] = get_shared_tc_for_consumer(dfb.get(), consumer_count, tc);
                log_info(
                    tt::LogMetal, "\tAssigned TC[{}]: {}", tc, (uint32_t)risc_config.config.packed_tile_counter[tc]);
            } else {
                TT_FATAL(false, "Need to implement blocked consumer access pattern");
            }
        }
        risc_config.config.num_tcs_to_rr = num_consumer_tcs;
        consumer_count++;

        dfb->risc_configs.push_back(risc_config);
    }

    log_info(tt::LogMetal, "DFB {} risc_mask: 0x{:x}", dfb->id, dfb->risc_mask);

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

        // Populate base_addr[] and limit[] arrays for each risc config
        uint32_t entry_size = dfb->config.entry_size;
        uint32_t max_prod_cons = std::max(dfb->config.num_producers, dfb->config.num_consumers);

        for (auto& rc : dfb->risc_configs) {
            for (uint8_t tc = 0; tc < rc.config.num_tcs_to_rr; tc++) {
                rc.config.base_addr[tc] = static_cast<uint32_t>(computed_addr) + (tc * entry_size);
                rc.config.limit[tc] =
                    rc.config.base_addr[tc] + ((entry_size * max_prod_cons) * (dfb->capacity - 1)) + entry_size;
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
