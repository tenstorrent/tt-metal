// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>

#include <algorithm>

#include "impl/context/metal_context.hpp"
#include "jit_build/jit_build_options.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "tt_metal/impl/program/program_impl.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"

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

void BindDataflowBufferToProducerConsumerKernels(Program& program, uint32_t dfb_id, KernelHandle producer_kernel_handle, KernelHandle consumer_kernel_handle) {
    auto dfb = program.impl().get_dataflow_buffer(dfb_id);

    TT_FATAL(!dfb->configs_finalized, "Cannot bind kernels to DFB {} after configuration has been finalized", dfb_id);

    // Not great but temporary until we have the updated host APIs
    std::shared_ptr<Kernel> producer_kernel = program.impl().get_kernel(producer_kernel_handle);
    std::shared_ptr<Kernel> consumer_kernel = program.impl().get_kernel(consumer_kernel_handle);

    TT_FATAL(producer_kernel != nullptr, "Producer kernel not found");
    TT_FATAL(consumer_kernel != nullptr, "Consumer kernel not found");

    // --- producer ---
    if (auto compute_producer = std::dynamic_pointer_cast<experimental::quasar::QuasarComputeKernel>(producer_kernel)) {
        TT_FATAL(
            dfb->config.num_producers >= 1 && dfb->config.num_producers <= 4,
            "Tensix producer count must be between 1 and 4, got {}",
            dfb->config.num_producers);
        dfb->config.producer_risc_mask = static_cast<uint16_t>(((1u << dfb->config.num_producers) - 1u) << ::dfb::TENSIX_RISC_OFFSET);
        dfb->tensix_trisc_mask |= (1u << 2);  // Tensix producer uses trisc2
    } else if (auto dm_producer = std::dynamic_pointer_cast<experimental::quasar::QuasarDataMovementKernel>(producer_kernel)) {
        TT_FATAL(
            dfb->config.num_producers >= 1 && dfb->config.num_producers <= 8,
            "DM producer count must be between 1 and 8, got {}",
            dfb->config.num_producers);
        const auto& producer_dm_riscvs = dm_producer->get_dm_processors();
        for (DataMovementProcessor dm : producer_dm_riscvs) {
            dfb->config.producer_risc_mask |= (1u << static_cast<std::underlying_type_t<DataMovementProcessor>>(dm));
        }
    } else if (auto gen1_dm_producer = std::dynamic_pointer_cast<DataMovementKernel>(producer_kernel)) {
        // WH/BH DataMovementKernel: RISCV_0 = BRISC (bit 0), RISCV_1 = NCRISC (bit 1)
        const DataMovementConfig dm_config = std::get<DataMovementConfig>(gen1_dm_producer->config());
        dfb->config.producer_risc_mask |= (1u << static_cast<std::underlying_type_t<DataMovementProcessor>>(dm_config.processor));
    } else if (std::dynamic_pointer_cast<ComputeKernel>(producer_kernel)) {
        // WH/BH ComputeKernel: bit 2 = Tensix
        dfb->config.producer_risc_mask |= (1u << 2);
    } else {
        TT_FATAL(false, "Unsupported kernel type for DFB producer");
    }

    // --- consumer ---
    if (auto compute_consumer = std::dynamic_pointer_cast<experimental::quasar::QuasarComputeKernel>(consumer_kernel)) {
        TT_FATAL(
            dfb->config.num_consumers >= 1 && dfb->config.num_consumers <= 4,
            "Tensix consumer count must be between 1 and 4, got {}",
            dfb->config.num_consumers);
        dfb->config.consumer_risc_mask = static_cast<uint16_t>(((1u << dfb->config.num_consumers) - 1u) << ::dfb::TENSIX_RISC_OFFSET);
        dfb->tensix_trisc_mask |= (1u << 0);  // Default: Tensix consumer uses trisc0; use (1u << 3) for trisc3
    } else if (auto dm_consumer = std::dynamic_pointer_cast<experimental::quasar::QuasarDataMovementKernel>(consumer_kernel)) {
        TT_FATAL(
            dfb->config.num_consumers >= 1 && dfb->config.num_consumers <= 8,
            "DM consumer count must be between 1 and 8, got {}",
            dfb->config.num_consumers);
        const auto& consumer_dm_riscvs = dm_consumer->get_dm_processors();
        for (DataMovementProcessor dm : consumer_dm_riscvs) {
            dfb->config.consumer_risc_mask |= (1u << static_cast<std::underlying_type_t<DataMovementProcessor>>(dm));
        }
    } else if (auto gen1_dm_consumer = std::dynamic_pointer_cast<DataMovementKernel>(consumer_kernel)) {
        // WH/BH DataMovementKernel: RISCV_0 = BRISC (bit 0), RISCV_1 = NCRISC (bit 1)
        const DataMovementConfig dm_config = std::get<DataMovementConfig>(gen1_dm_consumer->config());
        dfb->config.consumer_risc_mask |= (1u << static_cast<std::underlying_type_t<DataMovementProcessor>>(dm_config.processor));
    } else if (std::dynamic_pointer_cast<ComputeKernel>(consumer_kernel)) {
        // WH/BH ComputeKernel: bit 2 = Tensix
        dfb->config.consumer_risc_mask |= (1u << 2);
    } else {
        TT_FATAL(false, "Unsupported kernel type for DFB consumer");
    }
}

namespace detail {

::dfb::PackedTileCounter TileCounterAllocator::allocate(
    const CoreCoord& core, uint8_t tensix_id, bool use_t6_only) {
    TT_FATAL(tensix_id < ::dfb::NUM_TENSIX, "Invalid tensix_id: {}", tensix_id);
    auto& counters = next_tc_id_[core];

    uint8_t tc_id;
    if (use_t6_only) {
        TT_FATAL(
            ::dfb::TC_TENSIX_POOL_START + counters.t6_only_next[tensix_id] < ::dfb::NUM_TILE_COUNTERS_PER_TENSIX,
            "Out of Tensix-only tile counters for tensix {} on core ({}, {})",
            tensix_id, core.x, core.y);
        tc_id = ::dfb::TC_TENSIX_POOL_START + counters.t6_only_next[tensix_id]++;
    } else {
        TT_FATAL(
            counters.dm_next[tensix_id] < ::dfb::NUM_TENSIX_TILE_COUNTERS_FOR_DM,
            "Out of DM-visible tile counters for tensix {} on core ({}, {})",
            tensix_id, core.x, core.y);
        tc_id = counters.dm_next[tensix_id]++;
    }
    return static_cast<::dfb::PackedTileCounter>(
        (tensix_id << ::dfb::PACKED_TC_COUNTER_ID_BITS) | tc_id);
}

uint8_t RemapperIndexAllocator::allocate(const CoreCoord& core_coord) {
    uint8_t idx = next_index_[core_coord]++;
    TT_FATAL(
        idx < ::dfb::NUM_REMAPPER_PAIRINGS,
        "Out of remapper pairs for core ({}, {})",
        core_coord.x,
        core_coord.y);
    return idx;
}

void RemapperIndexAllocator::reset() { next_index_.clear(); }

std::vector<uint8_t> TxnIdAllocator::allocate(uint8_t count) {
    TT_FATAL(
        next_id_ + count <= 32,
        "TxnIdAllocator exhausted: requested {} IDs at next_id_={}, but only 32 are available",
        count,
        next_id_);
    std::vector<uint8_t> ids;
    ids.reserve(count);
    for (uint8_t i = 0; i < count; i++) {
        ids.push_back(next_id_++);
    }
    return ids;
}

uint8_t ClientTypeAllocator::allocate_for_consumer(uint8_t producer_client_type, uint8_t consumer_risc_id) {
    uint8_t client_type;

    if (consumer_risc_id >= 8) {
        // Tensix RISC: risc_id 8-11 -> clientType 4-7 (NEO_0 to NEO_3).
        // Each Tensix RISC maps to a unique id, so duplicates indicate a bug.
        client_type = 4 + (consumer_risc_id - 8);
        uint8_t bit = client_type - 4;
        TT_FATAL(!(tensix_used_mask_ & (1u << bit)), "Tensix clientType {} already used", client_type);
        tensix_used_mask_ |= (1u << bit);
    } else {
        // DM consumer: clientType must be in [0, 3] (DM TC groups only).
        // clientL != clientR but DM clientR IDs may repeat across consumers
        uint8_t available[4];
        uint8_t count = 0;
        for (uint8_t ct = 0; ct < 4; ct++) {
            if (ct != producer_client_type) {
                available[count++] = ct;
            }
        }
        // count is always > 0: producer_client_type is in [4,7] (Tensix) or one of [0,3] (DM),
        // leaving at least 3 eligible DM groups.
        client_type = available[dm_alloc_count_ % count];
        dm_alloc_count_++;
    }

    return client_type;
}

// Computes dfb_txn_id_descriptor_t for either the producer or consumer side of a DFB.
static dfb_txn_id_descriptor_t compute_txn_descriptor(
    uint16_t capacity,
    uint8_t num_producers,
    uint8_t num_consumers,
    bool is_producer,
    const std::vector<uint8_t>& txn_ids,
    uint8_t num_tcs_per_risc) {
    uint8_t num_prods_or_cons = is_producer ? num_producers : num_consumers;
    uint8_t num_txn_ids = static_cast<uint8_t>(txn_ids.size());

    // threshold is the number of transactions that each txn ID needs to process before posting/acking
    // for reads the transaction needs to be committed to dst, for writes the transaction needs to be sent out
    uint8_t threshold;
    if (num_producers == 1 && num_consumers == 1) {
        TT_FATAL(
            capacity % num_txn_ids == 0,
            "DFB capacity {} must be divisible by num_txn_ids {} for implicit sync",
            capacity,
            num_txn_ids);
        threshold = static_cast<uint8_t>(capacity / num_txn_ids);
    } else {
        threshold = num_prods_or_cons * num_tcs_per_risc;
    }

    TT_FATAL(
        threshold % num_prods_or_cons == 0,
        "num_entries_to_process_threshold {} must be divisible by num_prods_or_cons {}",
        threshold,
        num_prods_or_cons);
    uint8_t per_txn = threshold / num_prods_or_cons;

    TT_FATAL(
        per_txn % num_tcs_per_risc == 0,
        "num_entries_per_txn_id {} must be divisible by num_tcs_per_risc {}",
        per_txn,
        num_tcs_per_risc);
    uint8_t per_txn_per_tc = per_txn / num_tcs_per_risc;

    dfb_txn_id_descriptor_t desc = {};
    desc.num_txn_ids = num_txn_ids;
    desc.num_entries_to_process_threshold = threshold;
    desc.num_entries_per_txn_id = per_txn; // number of transactions each DM producer/consumer contributes
    desc.num_entries_per_txn_id_per_tc = per_txn_per_tc; // number of transactions each TC contributes
    for (uint8_t i = 0; i < num_txn_ids; i++) {
        desc.txn_ids[i] = txn_ids[i];
    }
    return desc;
}

bool has_dm_risc(uint16_t risc_mask) { return (risc_mask & 0xFF) != 0; }

bool has_tensix_risc(uint16_t risc_mask) { return (risc_mask & 0x0F00) != 0; }

uint8_t calculate_num_tile_counters(const DataflowBufferConfig& config, bool is_producer) {
    if (config.cap == ::dfb::AccessPattern::ALL) {
        bool producer_has_dm = has_dm_risc(config.producer_risc_mask);
        bool consumer_has_dm = has_dm_risc(config.consumer_risc_mask);
        bool producer_is_tensix_only = !producer_has_dm && has_tensix_risc(config.producer_risc_mask);
        bool consumer_is_tensix_only = !consumer_has_dm && has_tensix_risc(config.consumer_risc_mask);
        bool dm_dm_all = !producer_is_tensix_only && !consumer_is_tensix_only;
        if (is_producer) {
            if (dm_dm_all) {
                return config.num_consumers;
            }
            return 1;
        }
        return config.num_producers;
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
    for (uint8_t i = 0; i < ::dfb::NUM_TENSIX; i++) {
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
    ::dfb::PackedTileCounter producer_tc{};
    std::vector<::dfb::PackedTileCounter> consumer_tcs;
};

uint32_t DataflowBufferImpl::serialized_size() const {
    // On WH/BH: one 4-word CB-format config entry per DFB (identical to a circular buffer config)
    if (!MetalContext::instance().hal().has_tile_counter_registers()) {
        return 4 * sizeof(uint32_t);
    }
    // On Quasar: one dfb_initializer_t + one dfb_initializer_per_risc_t per risc.
    // All groups have the same number of RISC configs
    TT_FATAL(!groups.empty(), "DFB {} has no groups (configs not finalized?)", id);
    return sizeof(dfb_initializer_t) +
           (groups[0].hw_risc_configs.size() * sizeof(dfb_initializer_per_risc_t));
}

std::vector<uint8_t> DataflowBufferImpl::serialize_for_core(const CoreCoord& core) const {
    TT_FATAL(this->configs_finalized, "DFB {} configs not finalized before serialization", this->id);

    // On WH/BH: emit the same 4-word format used for circular buffers so the existing
    // setup_local_cb_read_write_interfaces firmware path can initialise the DFB slot.
    // Layout: [base_addr, total_size_bytes, num_pages, page_size_bytes]
    if (!MetalContext::instance().hal().has_tile_counter_registers()) {
        auto it = this->core_lookup_.find(core);
        TT_FATAL(
            it != this->core_lookup_.end(), "DFB {} has no config for core ({}, {})", this->id, core.x, core.y);
        const uint32_t alloc_addr = it->second.second;

        std::vector<uint8_t> data;
        data.reserve(4 * sizeof(uint32_t));
        const uint32_t words[4] = {
            alloc_addr,                                     // fifo_addr (base)
            this->config.entry_size * this->config.num_entries,  // fifo_size
            this->config.num_entries,                       // fifo_num_pages
            this->config.entry_size,                        // fifo_page_size
        };
        const auto* bytes = reinterpret_cast<const uint8_t*>(words);
        data.insert(data.end(), bytes, bytes + sizeof(words));
        return data;
    }

    auto it = this->core_lookup_.find(core);
    TT_FATAL(it != this->core_lookup_.end(), "DFB {} has no config for core ({}, {})", this->id, core.x, core.y);
    const auto& [group_idx, alloc_addr] = it->second;
    const DfbGroup* core_group = &this->groups[group_idx];

    const auto& hw_risc_configs = core_group->hw_risc_configs;

    std::vector<uint8_t> data;
    data.reserve(serialized_size());

    dfb_initializer_t init = {};
    init.logical_id = this->id;
    init.entry_size = this->entry_size;
    init.stride_in_entries = this->stride_in_entries;
    init.capacity = this->capacity;
    init.risc_mask_bits.dm_mask = this->risc_mask & 0xFF;
    init.risc_mask_bits.tensix_mask = (this->risc_mask >> 8) & 0x0F;
    init.risc_mask_bits.tensix_trisc_mask = this->tensix_trisc_mask & 0x0F;
    init.num_producers = this->config.num_producers;
    init.producer_txn_descriptor = this->producer_txn_descriptor;
    init.consumer_txn_descriptor = this->consumer_txn_descriptor;

    log_debug(
        tt::LogMetal,
        "Serializing DFB {} for core ({},{}) with {} producers and {} consumers. risc_mask: 0x{:x} use_remapper: {}",
        this->id,
        core.x,
        core.y,
        this->config.num_producers,
        this->config.num_consumers,
        this->risc_mask,
        this->use_remapper);

    log_debug(tt::LogMetal, "Entry size: {}", this->entry_size);
    log_debug(tt::LogMetal, "Stride in entries: {}", this->stride_in_entries);
    log_debug(tt::LogMetal, "Capacity: {}", this->capacity);
    log_debug(tt::LogMetal, "Risc mask: 0x{:x}", this->risc_mask);
    log_debug(tt::LogMetal, "Producer txn descriptor: num_txn_ids={} threshold={} per_txn={} per_tc={}",
        this->producer_txn_descriptor.num_txn_ids,
        this->producer_txn_descriptor.num_entries_to_process_threshold,
        this->producer_txn_descriptor.num_entries_per_txn_id,
        this->producer_txn_descriptor.num_entries_per_txn_id_per_tc);
    log_debug(tt::LogMetal, "Consumer txn descriptor: num_txn_ids={} threshold={} per_txn={} per_tc={}",
        this->consumer_txn_descriptor.num_txn_ids,
        this->consumer_txn_descriptor.num_entries_to_process_threshold,
        this->consumer_txn_descriptor.num_entries_per_txn_id,
        this->consumer_txn_descriptor.num_entries_per_txn_id_per_tc);

    const auto* init_bytes = reinterpret_cast<const uint8_t*>(&init);
    data.insert(data.end(), init_bytes, init_bytes + sizeof(init));

    const uint32_t entry_size = this->config.entry_size;
    // const uint32_t max_prod_cons = std::max(this->config.num_producers, this->config.num_consumers);

    // Find num_producer_tcs and num_consumer_tcs from the HW config.
    uint8_t num_producer_tcs = 0;
    uint8_t num_consumer_tcs = 0;
    for (const auto& rc : hw_risc_configs) {
        if (rc.is_producer) {
            num_producer_tcs = std::max(num_producer_tcs, rc.config.num_tcs_to_rr);
        } else {
            num_consumer_tcs = std::max(num_consumer_tcs, rc.config.num_tcs_to_rr);
        }
    }

    // Address arithmetic for L1 base/limit/step:
    //   - STRIDED (stride_in_entries = max_prod_cons): interleaved layout.
    //     Each producer occupies 1 slot per round → base step = entry_size.
    //   - ALL (stride_in_entries = 1): contiguous block per producer/TC.
    //     Each producer occupies `capacity` consecutive slots → base step = capacity * entry_size.
    //     This applies to both DM-DM ALL (broadcast_tc) and Tensix-involved ALL (remapper).
    //     The hardware derives the per-pop stride from (limit - base - entry_size) / (capacity - 1),
    //     which equals entry_size for ALL and max_prod_cons * entry_size for STRIDED.
    const uint32_t effective_stride = this->stride_in_entries;
    const uint32_t base_step = (effective_stride > 1) ? entry_size : (this->capacity * entry_size);

    std::vector<DFBRiscConfig> per_core_rc = hw_risc_configs;
    uint32_t base = alloc_addr;
    for (uint8_t tc = 0; tc < num_producer_tcs; tc++) {
        for (auto& rc : per_core_rc) {
            if (rc.is_producer && tc < rc.config.num_tcs_to_rr) {
                rc.config.base_addr[tc] = base;
                rc.config.limit[tc] =
                    rc.config.base_addr[tc] + ((entry_size * effective_stride) * (this->capacity - 1)) + entry_size;
                // Always advance base so each producer/TC gets its own L1 slot range.
                // broadcast_tc only governs device-side credit posting, not L1 addressing.
                base += base_step;
            }
        }
    }
    base = alloc_addr;
    for (uint8_t tc = 0; tc < num_consumer_tcs; tc++) {
        for (auto& rc : per_core_rc) {
            if (rc.is_producer) {
                continue;
            }
            rc.config.base_addr[tc] = base;
            rc.config.limit[tc] =
                rc.config.base_addr[tc] + ((entry_size * effective_stride) * (this->capacity - 1)) + entry_size;
            // In strided case each consumer maps to a different producer region, so advance base per consumer.
            if (this->config.cap == dfb::AccessPattern::STRIDED && tc < rc.config.num_tcs_to_rr) {
                base += base_step;
            }
        }
        // In ALL case all consumers share the producer address regions as they see every producer's data.
        if (this->config.cap == dfb::AccessPattern::ALL && this->config.num_producers > 1 &&
            tc < num_consumer_tcs) {
            base += base_step;
        }
    }

    // Write one dfb_initializer_per_risc_t per risc, in risc_mask order
    for (int bit = 0; bit < 16; bit++) {
        if (!(this->risc_mask & (1 << bit))) {
            continue;
        }
        const DFBRiscConfig* rc = nullptr;
        for (const auto& c : per_core_rc) {
            if (c.risc_id == static_cast<uint8_t>(bit)) {
                rc = &c;
                break;
            }
        }
        TT_FATAL(rc != nullptr, "DFB {}: no risc_config for risc_id {} on core ({},{})", this->id, bit, core.x, core.y);

        log_debug(tt::LogMetal, "New risc config (risc_id={}, is_producer={})", rc->risc_id, rc->is_producer);
        dfb_initializer_per_risc_t per_risc = {};

        per_risc.num_tcs_and_init.num_tcs_to_rr = rc->config.num_tcs_to_rr;
        per_risc.num_tcs_and_init.tc_init_done = 0;  // set by device when this producer finishes TC init
        per_risc.num_tcs_and_init.broadcast_tc = rc->config.broadcast_tc;
        log_debug(tt::LogMetal, "Num tcs to rr: {}", rc->config.num_tcs_to_rr);
        // Copy per-risc arrays
        for (int i = 0; i < rc->config.num_tcs_to_rr; i++) {
            per_risc.base_addr[i] = rc->config.base_addr[i];
            per_risc.limit[i] = rc->config.limit[i];
            per_risc.packed_tile_counter[i] = rc->config.packed_tile_counter[i];
            log_trace(tt::LogMetal, "Base addr {}: {}", i, static_cast<uint32_t>(per_risc.base_addr[i]));
            log_trace(tt::LogMetal, "Limit {}: {}", i, static_cast<uint32_t>(per_risc.limit[i]));
            log_trace(tt::LogMetal, "Packed tile counter {}: {}", i, (uint32_t)per_risc.packed_tile_counter[i]);
        }
        per_risc.flags.remapper_pair_index = static_cast<uint8_t>(rc->config.remapper_pair_index) & 0x3F;
        per_risc.flags.remapper_en = this->use_remapper;
        per_risc.flags.is_producer = rc->is_producer;
        per_risc.consumer_tcs = rc->config.consumer_tcs;
        // Per-producer remapper fields
        per_risc.remapper_consumer_ids_mask = rc->config.remapper_consumer_ids_mask;
        per_risc.producer_client_type = rc->config.producer_client_type;
        log_debug(tt::LogMetal, "Is producer: {}", rc->is_producer);
        log_debug(tt::LogMetal, "Remapper en: {}", this->use_remapper);
        if (this->use_remapper && rc->is_producer) {
            log_debug(
                tt::LogMetal,
                "Producer remapper: pair_idx={}, clientL={}, consumer_ids_mask=0x{:02x}",
                rc->config.remapper_pair_index,
                rc->config.producer_client_type,
                rc->config.remapper_consumer_ids_mask);
        }

        const auto* cfg_bytes = reinterpret_cast<const uint8_t*>(&per_risc);
        data.insert(data.end(), cfg_bytes, cfg_bytes + sizeof(per_risc));
    }

    log_debug(tt::LogMetal, "Serialized DFB {} for core ({},{}) size: {}", this->id, core.x, core.y, data.size());

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

        uint32_t kg_dfb_size = 0;
        for (const auto& dfb : dataflow_buffers) {
            TT_ASSERT(dfb->configs_finalized, "DFB {} configs not finalized before serialization", dfb->id);
            for (const CoreRange& kg_range : kg->core_ranges.ranges()) {
                if (dfb->core_ranges.intersects(kg_range)) {
                    kg_dfb_size += dfb->serialized_size();
                    break;
                }
            }
        }

        dfb_size = std::max(dfb_size, kg_dfb_size);
    }

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

    TT_FATAL(config.pap != dfb::AccessPattern::ALL, "ALL producer pattern not supported");

    TT_FATAL(
        config.cap != dfb::AccessPattern::ALL || config.num_consumers <= 4,
        "ALL consumer pattern supports at most 4 consumers, but {} were specified",
        config.num_consumers);

    if (config.tensix_scope.has_value()) {
        TT_FATAL(
            *config.tensix_scope != TensixScope::INTER,
            "Inter-tensix DFBs are not yet supported. Use TensixScope::INTRA for same-Neo packer→unpacker DFBs.");
        // INTRA: each Neo has an independent packer (TRISC2) → unpacker (TRISC0) credit flow, one Tensix-only TC per Neo.
        // Always STRIDED for both producer and consumer — blocked access and remapper are never used.
        // num_producers == num_consumers == number of Neos (one packer-unpacker pair per Neo).
        TT_FATAL(
            config.num_producers == config.num_consumers,
            "Intra-tensix DFBs require equal producers (packers) and consumers (unpackers), got {} and {}",
            config.num_producers, config.num_consumers);
        TT_FATAL(
            config.pap == dfb::AccessPattern::STRIDED && config.cap == dfb::AccessPattern::STRIDED,
            "Intra-tensix DFBs require STRIDED access for both producer (packer) and consumer (unpacker)");
        TT_FATAL(
            !config.enable_implicit_sync,
            "Intra-tensix DFBs do not support implicit sync (ISR-based credits)");
    }

    auto dfb = std::make_shared<DataflowBufferImpl>();

    dfb->id = static_cast<uint32_t>(this->dataflow_buffers_.size());

    // DFB IDs are auto-assigned contiguously from 0, so enforce the limit here.
    if (!MetalContext::instance().hal().has_tile_counter_registers()) {
        uint32_t max_dfb_id = MetalContext::instance().hal().get_arch_num_circular_buffers();
        TT_FATAL(
            dfb->id < max_dfb_id,
            "Cannot create DFB {}: WH/BH supports at most {} dataflow buffers",
            dfb->id,
            max_dfb_id);
    }

    dfb->core_ranges = core_range_set.merge_ranges();
    dfb->config = config;

    dfb->entry_size = config.entry_size;

    log_debug(
        tt::LogMetal,
        "Creating DFB {} with {} producers and {} consumers",
        dfb->id,
        config.num_producers,
        config.num_consumers);

    uint32_t capacity;
    switch (config.cap) {
        case dfb::AccessPattern::STRIDED:
            TT_FATAL(
                config.num_entries % std::max(config.num_producers, config.num_consumers) == 0,
                "Num entries in DFB {} must be divisible by max of num producers and consumers {}",
                config.num_entries,
                std::max(config.num_producers, config.num_consumers));
            capacity = config.num_entries / std::max(config.num_producers, config.num_consumers);
            dfb->stride_in_entries = std::max(config.num_producers, config.num_consumers);
            break;
        case dfb::AccessPattern::ALL:
            TT_FATAL(
                config.num_entries % config.num_producers == 0,
                "Num entries in DFB {} must be divisible by num producers {}",
                config.num_entries,
                config.num_producers);
            capacity = config.num_entries / config.num_producers;
            dfb->stride_in_entries = 1;
            break;
        default: TT_FATAL(false, "Invalid access pattern", (uint32_t)config.cap);
    }
    dfb->capacity = capacity;
    log_debug(tt::LogMetal, "Capacity: {}", capacity);

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

        // There is one DataflowBufferAllocator per unique core range, create one if it does not already exist for
        // current core range
        auto val = std::find_if(
            dfb_allocators_.begin(), dfb_allocators_.end(), [&core_range](const CircularBufferAllocator& dfb_allocator) {
                return dfb_allocator.core_range == core_range;
            });
        if (val == dfb_allocators_.end()) {
            this->dfb_allocators_.emplace_back(core_range);
        }
    }

    this->local_dataflow_buffer_allocation_needed_ = true;

    return dfb->id;
}

// Allocates TCs and remapper indices, per (dfb, core).
void ProgramImpl::finalize_dataflow_buffer_configs() {
    if (this->dataflow_buffers_.empty()) {
        return;
    }

    // On WH/BH there are no tile counters or remapper hardware.
    // Mark configs finalized and create a single dummy group per DFB so allocate_dataflow_buffers() can fill in the L1
    // address.
    if (!MetalContext::instance().hal().has_tile_counter_registers()) {
        for (auto& dfb : this->dataflow_buffers_) {
            if (dfb->configs_finalized) {
                continue;
            }
            dfb->risc_mask = dfb->config.producer_risc_mask | dfb->config.consumer_risc_mask;

            DfbGroup group;
            for (const CoreRange& cr : dfb->core_ranges.ranges()) {
                for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
                    for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
                        group.l1_by_core.emplace_back(CoreCoord(x, y), 0u);
                    }
                }
            }
            group.core_ranges = dfb->core_ranges;
            dfb->groups.push_back(std::move(group));
            dfb->configs_finalized = true;
        }
        return;
    }

    // Collect all (dfb, core) pairs that need finalization, grouped by core so that
    // remapper need can be determined per logical core
    std::unordered_map<CoreCoord, std::vector<std::shared_ptr<DataflowBufferImpl>>> dfbs_by_core;
    for (auto& dfb : this->dataflow_buffers_) {
        if (dfb->configs_finalized) {
            continue;
        }
        for (const CoreRange& core_range : dfb->core_ranges.ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    dfbs_by_core[CoreCoord(x, y)].push_back(dfb);
                }
            }
        }
    }

    // Process each core's DFBs together
    for (auto& [core, core_dfbs] : dfbs_by_core) {
        bool core_needs_remapper = false;
        for (const auto& dfb : core_dfbs) {
            if (dfb->config.cap == dfb::AccessPattern::ALL) {
                bool dm_dm_all = !has_tensix_risc(dfb->config.producer_risc_mask) &&
                                     !has_tensix_risc(dfb->config.consumer_risc_mask);
                if (!dm_dm_all) {
                    core_needs_remapper = true;
                    break;
                }
            }
        }

        log_debug(
            tt::LogMetal,
            "Finalizing {} DFBs on core ({}, {}), core_needs_remapper={}",
            core_dfbs.size(),
            core.x,
            core.y,
            core_needs_remapper);

        for (auto& dfb : core_dfbs) {
            finalize_single_dfb_config(dfb, core, core_needs_remapper);
        }
    }

    for (auto& dfb : this->dataflow_buffers_) {
        if (!dfb->configs_finalized && !dfb->groups.empty()) {
            dfb->configs_finalized = true;
        }
    }
}

void ProgramImpl::finalize_single_dfb_config(
    std::shared_ptr<DataflowBufferImpl>& dfb, const CoreCoord& core, bool core_has_remapper) {
    const auto& config = dfb->config;
    std::vector<DFBRiscConfig> new_hw_risc_configs;

    // Finds the DfbGroup whose hw_risc_configs match new_hw_risc_configs (creating one
    // if none exists), extends its core_ranges to include `core`, and appends an
    // l1_by_core entry.  base_addr/limit are not part of the equality check because
    // they are derived per-core in serialize_for_core() from each core's alloc_addr.
    auto bin_into_group = [&]() {
        auto hw_risc_configs_equal = [](const std::vector<DFBRiscConfig>& a, const std::vector<DFBRiscConfig>& b) {
            if (a.size() != b.size()) {
                return false;
            }
            for (size_t i = 0; i < a.size(); i++) {
                if (a[i].risc_id != b[i].risc_id || a[i].is_producer != b[i].is_producer) {
                    return false;
                }
                const auto& ca = a[i].config;
                const auto& cb = b[i].config;
                if (ca.num_tcs_to_rr != cb.num_tcs_to_rr ||
                    ca.broadcast_tc != cb.broadcast_tc ||
                    ca.remapper_pair_index != cb.remapper_pair_index ||
                    ca.consumer_tcs != cb.consumer_tcs ||
                    ca.remapper_consumer_ids_mask != cb.remapper_consumer_ids_mask ||
                    ca.producer_client_type != cb.producer_client_type) {
                    return false;
                }
                for (int j = 0; j < ca.num_tcs_to_rr; j++) {
                    if (ca.packed_tile_counter[j] != cb.packed_tile_counter[j]) {
                        return false;
                    }
                }
            }
            return true;
        };

        DfbGroup* matching_group = nullptr;
        for (auto& grp : dfb->groups) {
            if (hw_risc_configs_equal(grp.hw_risc_configs, new_hw_risc_configs)) {
                matching_group = &grp;
                break;
            }
        }
        if (matching_group == nullptr) {
            DfbGroup new_group;
            new_group.hw_risc_configs = new_hw_risc_configs;
            dfb->groups.push_back(std::move(new_group));
            matching_group = &dfb->groups.back();
        }
        CoreRange core_as_range(core, core);
        if (matching_group->core_ranges.ranges().empty()) {
            matching_group->core_ranges = CoreRangeSet(core_as_range);
        } else {
            matching_group->core_ranges = matching_group->core_ranges.merge(CoreRangeSet(core_as_range));
        }
        matching_group->l1_by_core.emplace_back(core, 0u);
    };

    TT_FATAL(config.producer_risc_mask != 0, "producer_risc_mask must be set before program launch. Either set it in DataflowBufferConfig or call BindDataflowBufferToProducerConsumerKernels after creating kernels");
    TT_FATAL(config.consumer_risc_mask != 0, "consumer_risc_mask must be set before program launch. Either set it in DataflowBufferConfig or call BindDataflowBufferToProducerConsumerKernels after creating kernels");

    const bool is_intra_tensix =
        config.tensix_scope.has_value() && *config.tensix_scope == TensixScope::INTRA;

    // Intra-tensix: producer and consumer share the same Neo bit — overlap is intentional.
    if (!is_intra_tensix) {
        TT_FATAL(
            (config.producer_risc_mask & config.consumer_risc_mask) == 0,
            "producer_risc_mask and consumer_risc_mask must not overlap");
    }

    bool producer_has_dm = has_dm_risc(config.producer_risc_mask);
    bool consumer_has_dm = has_dm_risc(config.consumer_risc_mask);
    bool producer_is_tensix_only = !producer_has_dm && has_tensix_risc(config.producer_risc_mask);
    bool consumer_is_tensix_only = !consumer_has_dm && has_tensix_risc(config.consumer_risc_mask);

    if (producer_is_tensix_only && consumer_is_tensix_only) {
        TT_FATAL(
            config.tensix_scope.has_value(),
            "Both producer and consumer are Tensix-only RISCs. Set tensix_scope to INTRA (same Neo) or INTER "
            "(different Neos). Un-scoped Tensix-to-Tensix DFBs are not allowed.");
    }

    // TRISC pack/unpack store ring extent in uint16_t L1-aligned units; host must reject oversized rings.
    if (MetalContext::instance().hal().has_tile_counter_registers()) {
        const bool tensix_on_dfb =
            has_tensix_risc(config.producer_risc_mask) || has_tensix_risc(config.consumer_risc_mask);
        if (tensix_on_dfb && dfb->capacity > 0) {
            const uint64_t ring_bytes =
                dfb->entry_size * (dfb->stride_in_entries * (dfb->capacity - 1U) + 1U);
            const uint32_t l1_align = MetalContext::instance().hal().get_alignment(HalMemType::L1);
            TT_FATAL(
                ring_bytes % l1_align == 0,
                "DFB {}: ring size in bytes ({}) must be a multiple of L1 alignment ({})",
                dfb->id,
                ring_bytes,
                l1_align);
            const uint64_t ring_trisc_units = ring_bytes / l1_align;
            TT_FATAL(
                ring_trisc_units > 0U,
                "DFB {}: TRISC ring extent is zero L1 units (ring_bytes={}, align={})",
                dfb->id,
                ring_bytes,
                l1_align);
            TT_FATAL(
                ring_trisc_units < 65536U,
                "DFB {}: TRISC ring extent ({} L1 units of {} bytes) exceeds uint16_t; reduce capacity, stride, or "
                "entry_size",
                dfb->id,
                ring_trisc_units,
                l1_align);
        }
    }

    dfb->risc_mask = config.producer_risc_mask | config.consumer_risc_mask;

    // ---------------------------------------------------------------------------
    // Intra-tensix: packer TRISC2 (producer) → unpacker TRISC0 (consumer) within the same Neo.
    // No DM RISC, no remapper, no strided/blocked access pattern — each Neo is a fully
    // independent packer→unpacker credit flow backed by one Tensix-only TC on that Neo.
    // For N Neos (num_producers == num_consumers == N): N Tensix-only TCs, one per Neo.
    // ---------------------------------------------------------------------------
    if (is_intra_tensix) {
        // Iterate over every Neo bit in producer_risc_mask and allocate a separate Tensix-only
        // TC for each, giving each Neo its own independent credit counter.
        dfb->use_remapper = false;

        for (uint8_t risc_id = ::dfb::TENSIX_RISC_OFFSET;
             risc_id < ::dfb::TENSIX_RISC_OFFSET + ::dfb::NUM_TENSIX;
             risc_id++) {
            if (!(config.producer_risc_mask & (1u << risc_id))) {
                continue;
            }
            uint8_t tensix_id = risc_id - ::dfb::TENSIX_RISC_OFFSET;

            ::dfb::PackedTileCounter t6_only_tc =
                tile_counter_allocator_.allocate(core, tensix_id, /*use_t6_only=*/true);

            log_info(
                tt::LogMetal,
                "Intra-tensix DFB {}: Neo{} Tensix-only TC (tensix_id={}, tc_id={})",
                dfb->id,
                tensix_id,
                (uint32_t)::dfb::get_tensix_id(t6_only_tc),
                (uint32_t)::dfb::get_counter_id(t6_only_tc));

            DFBRiscConfig risc_config;
            risc_config.risc_id = risc_id;
            risc_config.is_producer = true;
            risc_config.config.packed_tile_counter[0] = t6_only_tc;
            risc_config.config.num_tcs_to_rr = 1;
            risc_config.config.broadcast_tc = false;
            new_hw_risc_configs.push_back(risc_config);
        }

        bin_into_group();
        return;
    }

    // DM-DM ALL: producer broadcasts to N TCs (one per consumer) instead of using remapper.
    // No Tensix involved on either side.
    bool dm_dm_all = (config.cap == dfb::AccessPattern::ALL) &&
                         !producer_is_tensix_only && !consumer_is_tensix_only;

    // Remapper is needed only for ALL 1-to-many with Tensix
    // Adding a TC to a remapper config entry removes it from the default Tensix<->DM mirror group, even with
    // remapper enabled the default mirroring holds for STRIDED cases
    bool use_remapper = core_has_remapper &&
                        (config.cap == dfb::AccessPattern::ALL) &&
                        !dm_dm_all;

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
        }
        if (consumer_risc_id >= 8) {
            // Consumer is Tensix: must use consumer's tensix_id
            return (consumer_risc_id - 8) % 4;
        }
        // Both DM: round-robin across tensix_ids
        return get_dm_tensix_id_for_pair(pair_counter);
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
            log_debug(
                tt::LogMetal,
                "Remapper: Producer[{}] (risc_id={}) assigned clientL={}",
                producer_idx,
                producer_risc_id,
                producer_client_type);
        }

        // Then allocate consumer clientTypes (clientR)
        // Pass the producer's actual clientL so DM consumers can avoid it (hardware: clientL != clientR).
        for (size_t consumer_idx = 0; consumer_idx < consumer_risc_ids.size(); consumer_idx++) {
            uint8_t consumer_risc_id = consumer_risc_ids[consumer_idx];
            uint8_t client_type = client_type_allocator.allocate_for_consumer(producer_client_types[0], consumer_risc_id);
            consumer_client_types.push_back(client_type);

            log_debug(
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

            if (config.cap == dfb::AccessPattern::STRIDED) {
                // Determine which consumer(s) this producer TC slot pairs with
                uint8_t consumer_idx = (producer_idx + tc_slot * producer_risc_ids.size()) % consumer_risc_ids.size();

                uint8_t producer_risc_id = producer_risc_ids[producer_idx];
                uint8_t consumer_risc_id = consumer_risc_ids[consumer_idx];
                uint8_t tensix_id = get_tensix_id_for_pair(producer_risc_id, consumer_risc_id, pair_counter++);

                group.producer_tc = tile_counter_allocator_.allocate(core, tensix_id);

                if (use_remapper) {
                    // With remapper: allocate separate consumer TC
                    uint8_t consumer_tensix_id =
                        ClientTypeAllocator::get_tensix_id(consumer_client_types[consumer_idx]);
                    dfb::PackedTileCounter consumer_tc =
                        tile_counter_allocator_.allocate(core, consumer_tensix_id);
                    group.consumer_tcs.push_back(consumer_tc);
                } else {
                    // Without remapper: shared TC
                    group.consumer_tcs.push_back(group.producer_tc);
                }

                log_trace(
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
            } else if (config.cap == dfb::AccessPattern::ALL) {
                if (dm_dm_all) {
                    // DM-DM ALL: allocate one TC per consumer (tc_slot == consumer_idx).
                    // The TC is shared between producer and consumer i -- no remapper needed.
                    uint8_t tensix_id = get_dm_tensix_id_for_pair(pair_counter++);
                    group.producer_tc = tile_counter_allocator_.allocate(core, tensix_id);
                    group.consumer_tcs.push_back(group.producer_tc);  // shared

                    log_trace(
                        tt::LogMetal,
                        "ALL DM-DM: Producer[{}] TC[{}] (tensix_id={}) shared with Consumer[{}]",
                        producer_idx,
                        tc_slot,
                        tensix_id,
                        tc_slot);
                } else {
                    // Tensix-involved ALL: use remapper for 1-to-many
                    uint8_t producer_tensix_id = producer_risc_ids[producer_idx] % 4;
                    group.producer_tc = tile_counter_allocator_.allocate(core, producer_tensix_id);

                    for (size_t consumer_idx = 0; consumer_idx < consumer_risc_ids.size(); consumer_idx++) {
                        uint8_t consumer_tensix_id =
                            ClientTypeAllocator::get_tensix_id(consumer_client_types[consumer_idx]);
                        dfb::PackedTileCounter consumer_tc =
                            tile_counter_allocator_.allocate(core, consumer_tensix_id);
                        group.consumer_tcs.push_back(consumer_tc);
                    }

                    log_trace(
                        tt::LogMetal,
                        "ALL: Producer[{}] TC[{}] (tensix_id={}) maps to {} consumer TCs via Remapper",
                        producer_idx,
                        tc_slot,
                        producer_tensix_id,
                        group.consumer_tcs.size());
                }
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

        log_debug(
            tt::LogMetal,
            "Producer risc {} uses {} TCs",
            risc_id,
            num_producer_tcs);

        for (uint8_t tc = 0; tc < num_producer_tcs; tc++) {
            risc_config.config.packed_tile_counter[tc] = tc_groups[producer_idx][tc].producer_tc;
            log_trace(
                tt::LogMetal,
                "\tAssigned TC[{}]: (0x{:x}, 0x{:x})",
                tc,
                (uint32_t)dfb::get_tensix_id(risc_config.config.packed_tile_counter[tc]),
                (uint32_t)dfb::get_counter_id(risc_config.config.packed_tile_counter[tc]));
        }
        risc_config.config.num_tcs_to_rr = num_producer_tcs;
        risc_config.config.broadcast_tc = dm_dm_all;

        if (use_remapper) {
            risc_config.config.remapper_pair_index = remapper_index_allocator_.allocate(core);
            risc_config.config.producer_client_type = producer_client_types[producer_idx];

            // Build consumer_tcs packed and consumer_ids_mask for this producer
            const TileCounterGroup& group = tc_groups[producer_idx][0];
            uint32_t packed = 0;
            uint8_t consumer_ids_mask = 0;

            if (config.cap == dfb::AccessPattern::ALL) {
                // ALL: 1-to-many, all consumers
                for (size_t i = 0; i < group.consumer_tcs.size() && i < ::dfb::MAX_NUM_TILE_COUNTERS_TO_RR;
                     i++) {
                    packed |= (dfb::get_counter_id(group.consumer_tcs[i]) & 0x1F) << (i * 5);
                    consumer_ids_mask |= (1u << consumer_client_types[i]);
                }
            } else {
                // STRIDED via remapper: 1-to-1 per tc_slot
                for (uint8_t tc = 0; tc < num_producer_tcs; tc++) {
                    uint8_t consumer_idx = (producer_idx + tc * producer_risc_ids.size()) % consumer_risc_ids.size();
                    packed |= (dfb::get_counter_id(tc_groups[producer_idx][tc].consumer_tcs[0]) & 0x1F)
                              << (tc * 5);
                    consumer_ids_mask |= (1u << consumer_client_types[consumer_idx]);
                }
            }
            risc_config.config.consumer_tcs = packed;
            risc_config.config.remapper_consumer_ids_mask = consumer_ids_mask;

            log_debug(
                tt::LogMetal,
                "Producer[{}] remapper: pair_idx={}, clientL={}, consumer_ids_mask=0x{:02x}",
                producer_idx,
                risc_config.config.remapper_pair_index,
                risc_config.config.producer_client_type,
                risc_config.config.remapper_consumer_ids_mask);
        }

        new_hw_risc_configs.push_back(risc_config);
    }

    // Create consumer risc_configs and assign TCs from groups
    for (size_t consumer_idx = 0; consumer_idx < consumer_risc_ids.size(); consumer_idx++) {
        uint8_t risc_id = consumer_risc_ids[consumer_idx];

        DFBRiscConfig risc_config;
        risc_config.risc_id = risc_id;
        risc_config.is_producer = false;

        log_debug(
            tt::LogMetal,
            "Consumer risc {} uses {} TCs",
            risc_id,
            num_consumer_tcs);

        for (uint8_t tc = 0; tc < num_consumer_tcs; tc++) {
            if (config.cap == dfb::AccessPattern::STRIDED) {
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
            } else if (config.cap == dfb::AccessPattern::ALL) {
                if (dm_dm_all) {
                    // DM-DM ALL: consumer[consumer_idx] TC[tc] = shared TC from producer[tc][consumer_idx].
                    // tc iterates over num_consumer_tcs = num_producers.
                    uint8_t producer_idx = tc;
                    risc_config.config.packed_tile_counter[tc] =
                        tc_groups[producer_idx][consumer_idx].consumer_tcs[0];
                } else {
                    // Tensix-involved ALL: consumer gets its remapper-translated TC
                    uint8_t producer_idx = tc;
                    uint8_t producer_tc_slot = 0;
                    risc_config.config.packed_tile_counter[tc] =
                        tc_groups[producer_idx][producer_tc_slot].consumer_tcs[consumer_idx];
                }
            } else {
                TT_FATAL(false, "Unsupported consumer access pattern");
            }
            log_trace(
                tt::LogMetal,
                "\tAssigned TC[{}]: (0x{:x}, 0x{:x})",
                tc,
                (uint32_t)dfb::get_tensix_id(risc_config.config.packed_tile_counter[tc]),
                (uint32_t)dfb::get_counter_id(risc_config.config.packed_tile_counter[tc]));
        }
        risc_config.config.num_tcs_to_rr = num_consumer_tcs;

        new_hw_risc_configs.push_back(risc_config);
    }

    // Allocate transaction IDs and compute ISR descriptor fields when implicit sync is enabled.
    // Only done on the first core processed for this DFB because txn IDs are core-invariant
    // Two txn IDs per side for double buffering.
    if (config.enable_implicit_sync && dfb->groups.empty()) {
        constexpr uint8_t TXN_IDS_PER_SIDE = 2;

        if (!producer_is_tensix_only) {
            auto producer_txn_ids = txn_id_allocator_.allocate(TXN_IDS_PER_SIDE);
            dfb->producer_txn_descriptor = compute_txn_descriptor(
                dfb->capacity,
                config.num_producers,
                config.num_consumers,
                /*is_producer=*/true,
                producer_txn_ids,
                num_producer_tcs);
            log_debug(
                tt::LogMetal,
                "DFB {} implicit sync: producer txn_ids=[{},{}] threshold={} per_txn={} per_tc={}",
                dfb->id,
                dfb->producer_txn_descriptor.txn_ids[0],
                dfb->producer_txn_descriptor.txn_ids[1],
                dfb->producer_txn_descriptor.num_entries_to_process_threshold,
                dfb->producer_txn_descriptor.num_entries_per_txn_id,
                dfb->producer_txn_descriptor.num_entries_per_txn_id_per_tc);
        }

        if (!consumer_is_tensix_only) {
            auto consumer_txn_ids = txn_id_allocator_.allocate(TXN_IDS_PER_SIDE);
            dfb->consumer_txn_descriptor = compute_txn_descriptor(
                dfb->capacity,
                config.num_producers,
                config.num_consumers,
                /*is_producer=*/false,
                consumer_txn_ids,
                num_consumer_tcs);
            log_debug(
                tt::LogMetal,
                "DFB {} implicit sync: "
                "consumer txn_ids=[{},{}] threshold={} per_txn={} per_tc={}",
                dfb->id,
                dfb->consumer_txn_descriptor.txn_ids[0],
                dfb->consumer_txn_descriptor.txn_ids[1],
                dfb->consumer_txn_descriptor.num_entries_to_process_threshold,
                dfb->consumer_txn_descriptor.num_entries_per_txn_id,
                dfb->consumer_txn_descriptor.num_entries_per_txn_id_per_tc);
        }
    }

    dfb->use_remapper = use_remapper;
    log_debug(
        tt::LogMetal, "DFB {} finalized risc_mask: 0x{:x} use_remapper: {}", dfb->id, dfb->risc_mask, use_remapper);

    bin_into_group();
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
        // Fill alloc_addr per core in each group.  All cores of a DFB get the same computed_addr so the L1 buffer is at a uniform
        // absolute address on every physical core.
        uint32_t alloc_addr = static_cast<uint32_t>(computed_addr);
        dfb->core_lookup_.clear();
        for (size_t gi = 0; gi < dfb->groups.size(); gi++) {
            for (auto& [core, addr] : dfb->groups[gi].l1_by_core) {
                addr = alloc_addr;
                dfb->core_lookup_.emplace(core, std::make_pair(gi, alloc_addr));
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

const std::vector<std::shared_ptr<tt::tt_metal::experimental::dfb::detail::DataflowBufferImpl>>& ProgramImpl::dataflow_buffers() const {
    return dataflow_buffers_;
}

std::shared_ptr<tt::tt_metal::experimental::dfb::detail::DataflowBufferImpl> ProgramImpl::get_dataflow_buffer(uint32_t dfb_id) const {
    if (!this->dataflow_buffer_by_id_.contains(dfb_id)) {
        TT_THROW("No dataflow buffer with id {} exists in Program {}", dfb_id, this->id);
    }
    return dataflow_buffer_by_id_.at(dfb_id);
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

std::vector<std::shared_ptr<tt::tt_metal::experimental::dfb::detail::DataflowBufferImpl>> ProgramImpl::dataflow_buffers_on_corerange(const CoreRange& cr) const {
    std::vector<std::shared_ptr<tt::tt_metal::experimental::dfb::detail::DataflowBufferImpl>> dfbs_on_core;
    for (const auto& dfb : dataflow_buffers_) {
        if (dfb->core_ranges.intersects(cr)) {
            dfbs_on_core.push_back(dfb);
        }
    }
    return dfbs_on_core;
}

std::vector<CoreRange> ProgramImpl::dataflow_buffers_unique_coreranges() const {
    std::vector<CoreRange> core_ranges;
    for (const auto& dfb : dataflow_buffers_) {
        for (const CoreRange& core_range : dfb->core_ranges.ranges()) {
            if (std::find(core_ranges.begin(), core_ranges.end(), core_range) == core_ranges.end()) {
                core_ranges.push_back(core_range);
            }
        }
    }

    // Fast path: if no ranges overlap, return as-is.
    bool has_overlap = false;
    for (size_t i = 0; i < core_ranges.size() && !has_overlap; ++i) {
        for (size_t j = i + 1; j < core_ranges.size(); ++j) {
            if (core_ranges[i].intersects(core_ranges[j])) {
                has_overlap = true;
                break;
            }
        }
    }
    if (!has_overlap) {
        return core_ranges;
    }

    // Make ranges non-overlapping so each core is targeted by exactly one multicast
    // during DFB config dispatch.  Same A\B / A∩B / B\A splitting as CBs.
    std::vector<CoreRange> result = std::move(core_ranges);
    size_t i = 0;
    while (i < result.size()) {
        size_t j = i + 1;
        for (; j < result.size(); ++j) {
            if (result[i].intersects(result[j])) {
                break;
            }
        }
        if (j == result.size()) {
            ++i;
            continue;
        }
        CoreRangeSet a_set(result[i]), b_set(result[j]);
        result.erase(result.begin() + j);
        result.erase(result.begin() + i);
        auto a_only = a_set.subtract(b_set);
        auto b_only = b_set.subtract(a_set);
        auto common = a_set.intersection(b_set);
        result.insert(result.end(), a_only.ranges().begin(), a_only.ranges().end());
        result.insert(result.end(), b_only.ranges().begin(), b_only.ranges().end());
        result.insert(result.end(), common.ranges().begin(), common.ranges().end());
        i = 0;
    }
    return result;
}

void ProgramImpl::set_dfb_data_fmt_and_tile(const std::vector<CoreRange>& crs, JitBuildOptions& build_options) const {
    // ZoneScoped;
    for (const auto& logical_cr : crs) {
        const auto& dfbs_on_core = this->dataflow_buffers_on_corerange(logical_cr);
        for (const auto& dfb : dfbs_on_core) {
            build_options.set_cb_data_fmt_and_tile(
                static_cast<CBIndex>(dfb->id), dfb->config.data_format, dfb->config.tile);
        }
    }
}

}  // namespace tt::tt_metal::detail
