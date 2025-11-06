// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <enchantum/enchantum.hpp>

#include <stdint.h>
#include <tt_stl/assert.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/device.hpp>
#include "erisc_datamover_builder.hpp"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/code_profiling_types.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <optional>
#include <unordered_set>
#include <variant>
#include <vector>

#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_remote_channels_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_router_recipe.hpp"
#include "tt_metal/fabric/builder/channel_to_pool_mapping.hpp"
#include "tt_metal/fabric/builder/multi_pool_channel_allocator.hpp"

#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "core_coord.hpp"
#include "fabric_edm_types.hpp"
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt {
namespace tt_metal {
class Program;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_fabric {

// The channel structure is as follows:
//              &header->  |----------------| channel_base_address
//                         |    header      |
//             &payload->  |----------------|
//                         |                |
//                         |    payload     |
//                         |                |
//        &channel_sync->  |----------------|
//                         |  channel_sync  |
//                         ------------------
//

/**
 * Configure RISC settings based on architecture and RISC ID.
 * This function centralizes the per-ERISC configuration logic for easy modification.
 *
 * @param risc_id The RISC ID (0 or 1)
 * @param arch The architecture (WORMHOLE_B0 or BLACKHOLE)
 * @param enable_handshake Output parameter for handshake enable
 * @param enable_context_switch Output parameter for context switch enable
 * @param enable_interrupts Output parameter for interrupts enable
 * @param is_sender_channel_serviced Output parameter for sender channel service flags
 * @param is_receiver_channel_serviced Output parameter for receiver channel service flags
 */
namespace {
void configure_risc_settings(
    size_t num_riscv_cores,
    size_t risc_id,
    tt::ARCH arch,
    bool& enable_handshake,
    bool& enable_context_switch,
    bool& enable_interrupts,
    std::array<bool, builder_config::num_sender_channels>& is_sender_channel_serviced,
    std::array<bool, builder_config::num_receiver_channels>& is_receiver_channel_serviced) {
    if (arch == tt::ARCH::WORMHOLE_B0) {
        // Wormhole: All RISC cores handle both sender and receiver channels
        enable_handshake = true;
        enable_context_switch = true;
        enable_interrupts = false;
        is_sender_channel_serviced.fill(true);
        is_receiver_channel_serviced.fill(true);
    } else if (arch == tt::ARCH::BLACKHOLE) {
        if (num_riscv_cores == 1) {
            enable_handshake = true;
            enable_context_switch = tt::tt_metal::MetalContext::instance().rtoptions().get_enable_2_erisc_mode();
            enable_interrupts = false;
            is_sender_channel_serviced.fill(true);
            is_receiver_channel_serviced.fill(true);
        } else {
            // Blackhole: Distribute sender/receiver across two RISC cores
            if (risc_id == 0) {
                // ERISC0: Handle sender channels only
                enable_handshake = true;
                enable_context_switch = true;
                enable_interrupts = false;
                is_sender_channel_serviced.fill(true);
                is_receiver_channel_serviced.fill(false);
            } else if (risc_id == 1) {
                // ERISC1: Handle receiver channels only
                enable_handshake = false;
                enable_context_switch = false;
                enable_interrupts = false;
                is_sender_channel_serviced.fill(false);
                is_receiver_channel_serviced.fill(true);
            } else {
                TT_THROW("Invalid RISC ID {} for BLACKHOLE architecture", risc_id);
            }
        }
    } else {
        TT_THROW("Unsupported architecture for RISC configuration: {}", enchantum::to_string(arch));
    }
}

// for fabric with tensix extension, for linear/mesh topology, only one sender channel is used, and all
// other sender channels are marked as skipped. For ring/torus topology, one extra vc1 sender channel will
// also be used.
void update_sender_channel_servicing(
    tt::tt_fabric::FabricTensixConfig fabric_tensix_config,
    std::vector<FabricRiscConfig>& risc_configs,
    eth_chan_directions direction,
    Topology topology) {
    switch (fabric_tensix_config) {
        case tt::tt_fabric::FabricTensixConfig::MUX: break;
        default: TT_FATAL(false, "Error, invalid fabric_tensix_config: {}", static_cast<int>(fabric_tensix_config));
    }

    // Determine which channel corresponds to the current direction
    uint32_t target_channel = get_worker_connected_sender_channel(direction, topology);

    // For ring/torus topologies, determine VC1 channel (last channel) and service it
    uint32_t vc1_target_channel = get_worker_or_vc1_connected_sender_channel(direction, topology);

    auto arch = tt::tt_metal::MetalContext::instance().hal().get_arch();
    if (arch == tt::ARCH::WORMHOLE_B0) {
        for (auto& risc_config : risc_configs) {
            risc_config.reset_sender_channel_serviced();
            // Set the channel corresponding to the current direction and VC1 channel to true
            for (size_t i = 0; i < builder_config::num_sender_channels; i++) {
                risc_config.set_sender_channel_serviced(i, i == target_channel || i == vc1_target_channel);
            }
        }
    } else if (arch == tt::ARCH::BLACKHOLE) {
        risc_configs[0].reset_sender_channel_serviced();
        // Set the channel corresponding to the current direction and VC1 channel to true
        for (size_t i = 0; i < builder_config::num_sender_channels; i++) {
            risc_configs[0].set_sender_channel_serviced(i, i == target_channel || i == vc1_target_channel);
        }
    }
}

size_t get_num_riscv_cores() {
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_is_fabric_2_erisc_mode_enabled()) {
        size_t nriscs = tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
        if (nriscs > 1) {
            log_warning(tt::LogFabric, "Launching fabric in experimental 2-erisc mode.");
        }
        return nriscs;
    } else {
        return 1;
    }
}


}  // anonymous namespace

FabricRiscConfig::FabricRiscConfig(uint32_t risc_id) :
    noc_(risc_id == 0 ? tt::tt_metal::NOC::NOC_0 : tt::tt_metal::NOC::NOC_1),
    iterations_between_ctx_switch_and_teardown_checks_(
        FabricEriscDatamoverConfig::default_iterations_between_ctx_switch_and_teardown_checks),
    enable_handshake_(true),
    enable_context_switch_(true),
    enable_interrupts_(true) {
    auto arch = tt::tt_metal::MetalContext::instance().hal().get_arch();

    configure_risc_settings(
        get_num_riscv_cores(),
        risc_id,
        arch,
        this->enable_handshake_,
        this->enable_context_switch_,
        this->enable_interrupts_,
        this->is_sender_channel_serviced_,
        this->is_receiver_channel_serviced_);
}

namespace {
bool requires_forced_assignment_to_noc1() {
    // When creating a kernel on erisc0 and 2 erisc mode is disabled, the physical processor is erisc1 while erisc0 is
    // running base firmware. As base firmware may occasionally use noc0 force fabric on "erisc0" to use noc1
    //
    // When 2 erisc mode is enabled on the runtime, erisc index == noc index is enforced hence the condition
    // !get_enable_2_erisc_mode() below.
    //
    return tt::tt_metal::MetalContext::instance().hal().get_arch() == tt::ARCH::BLACKHOLE &&
           get_num_riscv_cores() == 1 && !tt::tt_metal::MetalContext::instance().rtoptions().get_enable_2_erisc_mode();
}
}  // anonymous namespace

FabricEriscDatamoverConfig::FabricEriscDatamoverConfig(Topology topology) : topology(topology) {
    const bool is_2D_routing = FabricContext::is_2D_topology(topology);
    uint32_t num_sender_channels = builder_config::get_sender_channel_count(is_2D_routing);
    uint32_t num_downstream_edms = builder_config::get_downstream_edm_count(is_2D_routing);
    // Global
    size_t next_l1_addr = tt::tt_metal::hal::get_erisc_l1_unreserved_base();

    // https://github.com/tenstorrent/tt-metal/issues/26354 to track fix for this hack where we always set aside the
    // memory for the telemetry buffer in Blackhole
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_enable_fabric_telemetry() ||
        tt::tt_metal::MetalContext::instance().hal().get_arch() == tt::ARCH::BLACKHOLE) {
        // Avoid a bug on BH, always allocate the space for the telemetry buffer
        this->perf_telemetry_buffer_address = next_l1_addr;
        next_l1_addr += 32;
    }

    // Allocate code profiling buffer (conditionally enabled)
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    if (rtoptions.get_enable_fabric_code_profiling_rx_ch_fwd()) {
        // Buffer size: max timer types * 16 bytes per result
        constexpr size_t code_profiling_buffer_size = get_max_code_profiling_timer_types() * sizeof(CodeProfilingTimerResult);
        this->code_profiling_buffer_address = next_l1_addr;
        next_l1_addr += code_profiling_buffer_size;
    } else {
        this->code_profiling_buffer_address = 0; // Not allocated
    }

    this->handshake_addr = next_l1_addr;
    next_l1_addr += eth_channel_sync_size;

    // issue: https://github.com/tenstorrent/tt-metal/issues/29073. TODO: Re-enable after hang is resolved.
    // Ethernet txq IDs on WH are 0,1 and on BH are 0,1,2.
    if (tt::tt_metal::MetalContext::instance().hal().get_arch() == tt::ARCH::BLACKHOLE && tt::tt_metal::MetalContext::instance().rtoptions().get_is_fabric_2_erisc_mode_enabled()) {
        this->receiver_txq_id = 1;
    }
    this->num_riscv_cores = get_num_riscv_cores();
    for (uint32_t risc_id = 0; risc_id < this->num_riscv_cores; risc_id++) {
        this->risc_configs.emplace_back(risc_id);
    }
    // temporary work-around for BH until 2-erisc is enabled. We are required to switch entirely to noc1 for
    // routers when only using a single erisc because erisc0 will be used by base FW and may periodically use
    // noc0 for link health related functionality.
    if (requires_forced_assignment_to_noc1()) {
        for (uint32_t risc_id = 0; risc_id < this->num_riscv_cores; risc_id++) {
            this->risc_configs[risc_id].set_configured_noc(tt::tt_metal::NOC::NOC_1);
        }
    }

    if (this->sender_txq_id != this->receiver_txq_id) {
        // counters are packed contiguously in memory, This can lead to resends of values but
        // this is safe for free running counters, which are enabled in this mode.
        size_t num_words_consumed_per_counter = tt::align(sizeof(uint32_t) * num_sender_channels, field_size);

        next_l1_addr = tt::align(next_l1_addr, field_size);

        this->to_sender_channel_remote_ack_counters_base_addr = next_l1_addr;
        next_l1_addr += num_words_consumed_per_counter;

        this->to_sender_channel_remote_completion_counters_base_addr = next_l1_addr;
        next_l1_addr += num_words_consumed_per_counter;

        this->receiver_channel_remote_ack_counters_base_addr = next_l1_addr;
        next_l1_addr += num_words_consumed_per_counter;

        this->receiver_channel_remote_completion_counters_base_addr = next_l1_addr;
        next_l1_addr += num_words_consumed_per_counter;
    }

    this->edm_channel_ack_addr = next_l1_addr;
    this->termination_signal_address =
        edm_channel_ack_addr +
        (4 * eth_channel_sync_size);  // pad extra bytes to match old EDM so handshake logic will still work
    this->edm_local_sync_address = termination_signal_address + field_size;
    this->edm_status_address = edm_local_sync_address + field_size;

    uint32_t buffer_address = edm_status_address + field_size;

    for (uint32_t i = 0; i < builder_config::num_receiver_channels; i++) {
        this->receiver_channels_counters_address[i] = buffer_address;
        buffer_address += receiver_channel_counters_size_bytes;
    }
    for (uint32_t i = 0; i < num_sender_channels; i++) {
        this->sender_channels_counters_address[i] = buffer_address;
        buffer_address += sender_channel_counters_size_bytes;
    }

    // Packet header history buffer(s)
    for (uint32_t i = 0; i < builder_config::num_receiver_channels; i++) {
        this->receivers_completed_packet_header_cb_address[i] = buffer_address;
        buffer_address += receiver_completed_packet_header_cb_size_bytes;
    }
    for (uint32_t i = 0; i < num_sender_channels; i++) {
        this->senders_completed_packet_header_cb_address[i] = buffer_address;
        buffer_address += sender_completed_packet_header_cb_size_bytes;
    }

    // ----------- Sender Channels
    for (uint32_t i = 0; i < num_sender_channels; i++) {
        this->sender_channels_buffer_index_address[i] = buffer_address;
        buffer_address += field_size;
        // Connection info layout:
        // 0: buffer_index_rdptr -> Tells EDM the address in worker L1 to update EDM's copy of channel rdptr
        // 1: worker_teardown_semaphore_address -> Tells EDM where to signal connection teardown completion in
        // worker's L1 2: WorkerXY (as uint32_t) 3: Hold's EDM's rdptr for the buffer index in the channel
        this->sender_channels_worker_conn_info_base_address[i] = buffer_address;
        buffer_address += sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo);
        this->sender_channels_local_flow_control_semaphore_address[i] = buffer_address;
        buffer_address += field_size;
        this->sender_channels_producer_terminate_connection_address[i] = buffer_address;
        buffer_address += field_size;
        // persistent mode field
        this->sender_channels_connection_semaphore_address[i] = buffer_address;
        buffer_address += field_size;
        // persistent mode field
        this->sender_channels_buffer_index_semaphore_address[i] = buffer_address;
        buffer_address += field_size;
    }
    // ----------- Receiver Channels
    for (uint32_t i = 0; i < num_downstream_edms; i++) {
        // temporarily padded to have exact parity with addresses pre-refactor
        // because receiver_channels_local_buffer_index_address was removed (as dead code) and is no longer
        // needed. We still waste the L1 to minimize the incremental changes in builder refactor
        buffer_address += field_size;

        // persistent mode field
        this->receiver_channels_downstream_flow_control_semaphore_address[i] = buffer_address;
        buffer_address += field_size;
        this->receiver_channels_downstream_teardown_semaphore_address[i] = buffer_address;
        buffer_address += field_size;
    }

    // Issue: https://github.com/tenstorrent/tt-metal/issues/29249. Move it back to after edm_local_sync_address once
    // the hang is root caused for multiprocess test.
    this->edm_local_tensix_sync_address = buffer_address;
    buffer_address += field_size;

    // location for temporarily store the src address when performing inline writes to L1 with spoof
    if (tt::tt_metal::MetalContext::instance().hal().get_arch() == tt::ARCH::BLACKHOLE) {
        this->notify_worker_of_read_counter_update_src_address = buffer_address;
        buffer_address += field_size;
    }

    // Channel Allocations
    this->max_l1_loading_size =
        tt::tt_metal::hal::get_erisc_l1_unreserved_size() + tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    auto buffer_region_start = (buffer_address + buffer_alignment) & ~(buffer_alignment - 1);  // Align
    auto available_channel_buffering_space = max_l1_loading_size - buffer_region_start;
    this->available_buffer_memory_regions.emplace_back(buffer_region_start, available_channel_buffering_space);
}

void FabricEriscDatamoverConfig::configure_skip_connection_flags(Topology topology, FabricEriscDatamoverOptions const& options) {
    if (topology == Topology::Ring) {
        auto buffer_config = options.edm_buffer_config;
        switch (options.edm_type) {
            case FabricEriscDatamoverType::Dateline: break;
            case FabricEriscDatamoverType::DatelineUpstream:
                if (buffer_config.enable_dateline_upstream_sender_extra_buffer_slots) {
                    this->skip_sender_channel_1_connection = true;
                }
                // set num_receiver_buffer_slots
                if (buffer_config.enable_dateline_upstream_receiver_extra_buffer_slots) {
                    this->skip_receiver_channel_1_connection = true;
                }
                break;
            case FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice:
                if (buffer_config.enable_dateline_upstream_adjacent_sender_extra_buffer_slots) {
                    this->skip_sender_vc1_channel_connection = true;
                }
                break;
            default: break;
        }
    }
}

FabricEriscDatamoverConfig::FabricEriscDatamoverConfig(
    std::size_t channel_buffer_size_bytes, Topology topology, FabricEriscDatamoverOptions options) :
    FabricEriscDatamoverConfig(topology) {
    // Update sender channel servicing based on fabric tensix configuration
    if (options.fabric_tensix_config != tt::tt_fabric::FabricTensixConfig::DISABLED) {
        // Use default direction (EAST) for the constructor case since direction isn't available here
        update_sender_channel_servicing(options.fabric_tensix_config, this->risc_configs, options.direction, topology);
    }

    const bool is_2D_routing = FabricContext::is_2D_topology(topology);

    this->channel_buffer_size_bytes = channel_buffer_size_bytes;
    this->num_used_sender_channels = builder_config::get_sender_channel_count(is_2D_routing);
    this->num_used_receiver_channels = builder_config::num_receiver_channels;

    if (is_2D_routing) {
        // For 2D there is no forwarding to self but we are still initialize the settings for it.
        // Routers ignore the settings at self index.
        this->num_fwd_paths = this->num_used_sender_channels;
    } else {
        this->num_fwd_paths = this->num_used_sender_channels - 1;
    }

    // Ring/Torus have extra channels
    if (topology == Topology::Linear || topology == Topology::Mesh) {
        this->num_used_sender_channels -= 1;
        this->num_used_receiver_channels -= 1;
        this->num_fwd_paths -= 1;
    }

    for (uint32_t i = 0; i < this->num_used_receiver_channels; i++) {
        TT_FATAL(
            (receivers_completed_packet_header_cb_address[i] % eth_word_l1_alignment == 0),
            "receivers_completed_packet_header_cb_address[{}] {} must be aligned to {} bytes",
            i,
            receivers_completed_packet_header_cb_address[i],
            eth_word_l1_alignment);
    }
    for (uint32_t i = 0; i < this->num_used_sender_channels; i++) {
        TT_FATAL(
            (senders_completed_packet_header_cb_address[i] % eth_word_l1_alignment == 0),
            "senders_completed_packet_header_cb_address[{}] {} must be aligned to {} bytes",
            i,
            senders_completed_packet_header_cb_address[i],
            eth_word_l1_alignment);
        TT_FATAL(
            (sender_channels_buffer_index_address[i] % eth_word_l1_alignment == 0),
            "sender_channels_buffer_index_address[{}] {} must be aligned to {} bytes",
            i,
            sender_channels_buffer_index_address[i],
            eth_word_l1_alignment);
        TT_FATAL(
            (sender_channels_worker_conn_info_base_address[i] % eth_word_l1_alignment == 0),
            "sender_channels_worker_conn_info_base_address[{}] {} must be aligned to {} bytes",
            i,
            sender_channels_worker_conn_info_base_address[i],
            eth_word_l1_alignment);
        TT_FATAL(
            (sender_channels_local_flow_control_semaphore_address[i] % eth_word_l1_alignment == 0),
            "sender_channels_local_flow_control_semaphore_address[{}] {} must be aligned to {} bytes",
            i,
            sender_channels_local_flow_control_semaphore_address[i],
            eth_word_l1_alignment);
        TT_FATAL(
            (sender_channels_producer_terminate_connection_address[i] % eth_word_l1_alignment == 0),
            "sender_channels_producer_terminate_connection_address[{}] {} must be aligned to {} bytes",
            i,
            sender_channels_producer_terminate_connection_address[i],
            eth_word_l1_alignment);
    }
    TT_FATAL(
        std::unordered_set<size_t>(
            sender_channels_buffer_index_address.begin(),
            sender_channels_buffer_index_address.begin() + this->num_used_sender_channels)
                .size() == this->num_used_sender_channels,
        "FabricEriscDatamoverConfig was constructed with illegal buffer index address");

    const size_t min_buffer_size = sizeof(tt::tt_fabric::PacketHeader);
    TT_FATAL(
        channel_buffer_size_bytes >= min_buffer_size,
        "FabricEriscDatamoverConfig was constructed with `channel_buffer_size_bytes` argument set smaller than "
        "minimum "
        "size of {}",
        min_buffer_size);
    this->channel_buffer_size_bytes = channel_buffer_size_bytes;

    // Compute available channel buffering space from memory regions
    size_t available_channel_buffering_space = std::accumulate(
        this->available_buffer_memory_regions.begin(),
        this->available_buffer_memory_regions.end(),
        size_t{0},
        [](size_t sum, const MemoryRegion& region) { return sum + region.get_size(); });


    configure_skip_connection_flags(topology, options);

    // Create a default recipe with a single static pool for backward compatibility
    // All channels map to pool 0 (the single static pool)
    auto recipe = tt::tt_fabric::FabricRouterRecipe::create_default_single_static_pool_recipe(
        this->num_used_sender_channels, this->num_used_receiver_channels);
    auto remote_channels_recipe = tt::tt_fabric::FabricRouterRecipe::create_default_single_static_pool_recipe(
        0, this->num_used_receiver_channels);

    // Create the single static pool allocator
    auto static_allocator = std::make_shared<tt::tt_fabric::FabricStaticSizedChannelsAllocator>(
        topology,
        options,
        this->num_used_sender_channels,
        this->num_used_receiver_channels,
        this->channel_buffer_size_bytes,
        available_channel_buffering_space,
        this->available_buffer_memory_regions);

    // Assign static allocator directly to channel_allocator (composition, not wrapped)
    this->channel_allocator = static_allocator;

    // Create remote channels allocator from the static allocator
    this->remote_channels_allocator = std::make_shared<tt::tt_fabric::FabricRemoteChannelsAllocator>(*static_allocator);

    // Create multi-pool coordinator that manages the pool allocators
    std::vector<std::shared_ptr<tt::tt_fabric::FabricChannelAllocator>> pool_allocators;
    pool_allocators.push_back(static_allocator);

    std::vector<tt::tt_fabric::FabricChannelPoolType> pool_types;
    pool_types.push_back(tt::tt_fabric::FabricChannelPoolType::STATIC);

    this->multi_pool_allocator =
        std::make_shared<tt::tt_fabric::MultiPoolChannelAllocator>(std::move(pool_allocators), std::move(pool_types));

    // Create the channel-to-pool mapping
    this->channel_to_pool_mapping = std::make_shared<tt::tt_fabric::ChannelToPoolMapping>(recipe);
    this->remote_channel_to_pool_mapping =
        std::make_shared<tt::tt_fabric::ChannelToPoolMapping>(remote_channels_recipe);

    // set default noc and cmd bufs (current setup in TG 4U)
    for (uint32_t i = 0; i < builder_config::num_receiver_channels; i++) {
        this->receiver_channel_forwarding_noc_ids[i] = FabricEriscDatamoverConfig::DEFAULT_RECEIVER_FORWARDING_NOC;
        this->receiver_channel_forwarding_data_cmd_buf_ids[i] = FabricEriscDatamoverConfig::WR_REG_CMD_BUF;
        this->receiver_channel_forwarding_sync_cmd_buf_ids[i] = FabricEriscDatamoverConfig::RD_CMD_BUF;
        this->receiver_channel_local_write_noc_ids[i] = FabricEriscDatamoverConfig::DEFAULT_RECEIVER_LOCAL_WRITE_NOC;
        this->receiver_channel_local_write_cmd_buf_ids[i] = FabricEriscDatamoverConfig::WR_CMD_BUF;

        if (requires_forced_assignment_to_noc1()) {
            this->receiver_channel_forwarding_noc_ids[i] = FabricEriscDatamoverConfig::BLACKHOLE_SINGLE_ERISC_MODE_RECEIVER_FORWARDING_NOC;
            this->receiver_channel_local_write_noc_ids[i] =
                FabricEriscDatamoverConfig::BLACKHOLE_SINGLE_ERISC_MODE_RECEIVER_LOCAL_WRITE_NOC;
            this->receiver_channel_forwarding_data_cmd_buf_ids[i] = FabricEriscDatamoverConfig::WR_CMD_BUF;
        }
    }
    for (uint32_t i = 0; i < builder_config::num_sender_channels; i++) {
        this->sender_channel_ack_noc_ids[i] = FabricEriscDatamoverConfig::DEFAULT_SENDER_ACK_NOC;
        this->sender_channel_ack_cmd_buf_ids[i] = FabricEriscDatamoverConfig::AT_CMD_BUF;

        if (requires_forced_assignment_to_noc1()) {
            this->sender_channel_ack_noc_ids[i] =
                FabricEriscDatamoverConfig::BLACKHOLE_SINGLE_ERISC_MODE_SENDER_ACK_NOC;
        }
    }
    this->edm_noc_vc = FabricEriscDatamoverConfig::DEFAULT_NOC_VC;
}

void get_runtime_args_for_edm_termination_infos(
    const std::vector<edm_termination_info_t>& edm_termination_infos, std::vector<uint32_t>& args_out) {
    args_out.reserve(args_out.size() + (edm_termination_infos.size() * 4) + 1);
    args_out.push_back(edm_termination_infos.size());
    for (const auto& info : edm_termination_infos) {
        args_out.push_back(info.edm_noc_x);
        args_out.push_back(info.edm_noc_y);
        args_out.push_back(info.distance);
        args_out.push_back(info.termination_addr);
        log_trace(
            tt::LogTest,
            "EDM termination info: x={}, y={}, distance={}, termination_addr={}",
            info.edm_noc_x,
            info.edm_noc_y,
            info.distance,
            info.termination_addr);
    }
}

// TODO: will be deprecated. currently for ethernet dispatch case
//       ethernet core need to have same memory mapping as worker
void append_worker_to_fabric_edm_sender_rt_args(
    const SenderWorkerAdapterSpec& connection,
    size_t sender_worker_flow_control_semaphore_id,
    size_t sender_worker_terminate_semaphore_id,
    size_t sender_worker_buffer_index_semaphore_id,
    std::vector<uint32_t>& args_out) {
    auto edm_noc_xy = tt::tt_fabric::WorkerXY(connection.edm_noc_x, connection.edm_noc_y);

    TT_FATAL(
        (sender_worker_flow_control_semaphore_id & 0xFFFF) == sender_worker_flow_control_semaphore_id,
        "sender_worker_flow_control_semaphore_id is not being interpreted as a semaphore ID for worker connection");

    const std::vector<uint32_t> values = {
        connection.edm_direction,
        edm_noc_xy.to_uint32(),
        connection.edm_buffer_base_addr,
        connection.num_buffers_per_channel,
        connection.edm_l1_sem_addr,
        connection.edm_connection_handshake_addr,
        connection.edm_worker_location_info_addr,
        connection.buffer_size_bytes,
        connection.buffer_index_semaphore_id,
        sender_worker_flow_control_semaphore_id,
        sender_worker_terminate_semaphore_id,
        sender_worker_buffer_index_semaphore_id};
    args_out.reserve(args_out.size() + (values.size() / sizeof(size_t)));
    std::ranges::copy(values, std::back_inserter(args_out));
}

void append_worker_to_fabric_edm_sender_rt_args(
    chan_id_t eth_channel,
    size_t sender_worker_terminate_semaphore_id,
    size_t sender_worker_buffer_index_semaphore_id,
    std::vector<uint32_t>& args_out) {
    const std::vector<uint32_t> values = {
        eth_channel, sender_worker_terminate_semaphore_id, sender_worker_buffer_index_semaphore_id};
    args_out.reserve(args_out.size() + (values.size() / sizeof(size_t)));
    std::ranges::copy(values, std::back_inserter(args_out));
}

// TODO: will be deprecated. non device init fabric case
void append_worker_to_fabric_edm_sender_rt_args(
    const SenderWorkerAdapterSpec& connection,
    ChipId chip_id,
    const CoreRangeSet& worker_cores,
    size_t sender_worker_terminate_semaphore_id,
    size_t sender_worker_buffer_index_semaphore_id,
    std::vector<uint32_t>& args_out) {
    chan_id_t eth_channel =
        tt::tt_metal::MetalContext::instance()
            .get_cluster()
            .get_logical_ethernet_core_from_virtual(chip_id, CoreCoord(connection.edm_noc_x, connection.edm_noc_y))
            .y;

    // copy "only" connections[eth_channel] to L1, not the whole tensix_fabric_connections_l1_info_t
    // because this function is called several times for same device which overwrites info written by previous calls
    tt::tt_fabric::tensix_fabric_connections_l1_info_t fabric_connections = {};
    auto& connection_info = fabric_connections.read_only[eth_channel];
    connection_info.edm_direction = connection.edm_direction;
    connection_info.edm_noc_x = connection.edm_noc_x;
    connection_info.edm_noc_y = connection.edm_noc_y;
    connection_info.edm_buffer_base_addr = connection.edm_buffer_base_addr;
    connection_info.num_buffers_per_channel = connection.num_buffers_per_channel;
    connection_info.edm_connection_handshake_addr = connection.edm_connection_handshake_addr;
    connection_info.edm_worker_location_info_addr = connection.edm_worker_location_info_addr;
    connection_info.buffer_size_bytes = connection.buffer_size_bytes;
    connection_info.buffer_index_semaphore_id = connection.buffer_index_semaphore_id;
    // NOTE: valid_connections_mask is not copied to L1 from performance reason
    //       because this callstack will be deprecated and not used in WorkerToFabricEdmSenderImpl yet
    //       we want to reduce the number of write_core calls
    fabric_connections.valid_connections_mask |= (1u << eth_channel);

    size_t connection_offset = offsetof(tt::tt_fabric::tensix_fabric_connections_l1_info_t, read_only) +
                               (eth_channel * sizeof(tt::tt_fabric::fabric_connection_info_t));
    // Write to Tensix cores
    std::vector<CoreCoord> worker_core_coords = corerange_to_cores(worker_cores, std::nullopt, true);
    for (const auto& logical_core : worker_core_coords) {
        CoreCoord tensix_core =
            tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
                chip_id, logical_core, CoreType::WORKER);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            &connection_info,
            sizeof(tt::tt_fabric::fabric_connection_info_t),
            tt_cxy_pair(chip_id, tensix_core),
            tt_metal::MetalContext::instance().hal().get_dev_addr(
                tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS) +
                connection_offset);
    }

    const std::vector<uint32_t> values = {
        eth_channel, sender_worker_terminate_semaphore_id, sender_worker_buffer_index_semaphore_id};
    args_out.reserve(args_out.size() + (values.size() / sizeof(size_t)));
    std::ranges::copy(values, std::back_inserter(args_out));
}

size_t log_worker_to_fabric_edm_sender_rt_args(const std::vector<uint32_t>& args, size_t starting_arg_idx) {
    log_trace(tt::LogFabric, "Worker to fabric EDM Sender has {} RT Args: {}", args.size(), args);
    log_trace(tt::LogFabric, "arg[{}]: edm_noc_xy {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogFabric, "arg[{}]: edm_buffer_base_addr {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogFabric, "arg[{}]: num_buffers_per_channel {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogFabric, "arg[{}]: edm_l1_sem_addr {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogFabric, "arg[{}]: edm_connection_handshake_addr {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogFabric, "arg[{}]: edm_worker_location_info_addr {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogFabric, "arg[{}]: buffer_size_bytes {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogFabric, "arg[{}]: buffer_index_semaphore_id {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(
        tt::LogFabric, "arg[{}]: sender_worker_flow_control_semaphore_id {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(
        tt::LogFabric, "arg[{}]: sender_worker_buffer_index_semaphore_id {}", starting_arg_idx, args[starting_arg_idx++]);
    return starting_arg_idx + 10;
}

FabricEriscDatamoverBuilder::FabricEriscDatamoverBuilder(
    const CoreCoord& my_eth_core_logical,
    size_t my_noc_x,
    size_t my_noc_y,
    const FabricNodeId& local_fabric_node_id,
    const FabricNodeId& peer_fabric_node_id,

    const std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>&
        receiver_channels_downstream_flow_control_semaphore_id,
    const std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>&
        receiver_channels_downstream_teardown_semaphore_id,
    const std::array<size_t, builder_config::num_sender_channels>& sender_channels_flow_control_semaphore_id,
    const std::array<size_t, builder_config::num_sender_channels>& sender_channels_connection_semaphore_id,
    const std::array<size_t, builder_config::num_sender_channels>& sender_channels_buffer_index_semaphore_id,

    const FabricEriscDatamoverConfig& config,
    eth_chan_directions direction,
    bool build_in_worker_connection_mode,
    FabricEriscDatamoverType fabric_edm_type,
    bool has_tensix_extension) :
    my_eth_core_logical(my_eth_core_logical),
    my_eth_channel(my_eth_core_logical.y),
    my_noc_x(my_noc_x),
    my_noc_y(my_noc_y),
    config(config),
    local_fabric_node_id(local_fabric_node_id),
    peer_fabric_node_id(peer_fabric_node_id),
    handshake_address(tt::round_up(
        tt::tt_metal::hal::get_erisc_l1_unreserved_base(), FabricEriscDatamoverConfig::eth_channel_sync_size)),
    channel_buffer_size(config.channel_buffer_size_bytes),
    local_sender_channels_connection_info_addr(config.sender_channels_worker_conn_info_base_address),
    termination_signal_ptr(config.termination_signal_address),
    edm_local_sync_ptr(config.edm_local_sync_address),
    edm_local_tensix_sync_ptr(config.edm_local_tensix_sync_address),
    edm_status_ptr(config.edm_status_address),
    direction(direction),

    // this is the receiver channel's local sem for flow controlling with downstream fabric sender
    receiver_channels_downstream_flow_control_semaphore_id(receiver_channels_downstream_flow_control_semaphore_id),
    receiver_channels_downstream_teardown_semaphore_id(receiver_channels_downstream_teardown_semaphore_id),
    sender_channels_flow_control_semaphore_id(sender_channels_flow_control_semaphore_id),
    sender_channels_connection_semaphore_id(sender_channels_connection_semaphore_id),
    sender_channels_buffer_index_semaphore_id(sender_channels_buffer_index_semaphore_id),
    downstream_vcs_sender_channel_buffer_index_semaphore_id(sender_channels_buffer_index_semaphore_id),
    build_in_worker_connection_mode(build_in_worker_connection_mode),
    fabric_edm_type(fabric_edm_type),
    dateline_connection(fabric_edm_type == tt::tt_fabric::FabricEriscDatamoverType::Dateline),
    has_tensix_extension(has_tensix_extension) {
    std::fill(
        sender_channel_connection_liveness_check_disable_array.begin(),
        sender_channel_connection_liveness_check_disable_array.end(),
        false);

    TT_FATAL(config.channel_allocator.get() != nullptr, "Channel allocator is not set. Failed to build TT-Fabric router. Internal error.");
    auto static_allocator =
        dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(config.channel_allocator.get());
    TT_FATAL(
        static_allocator != nullptr,
        "Channel allocator must be a FabricStaticSizedChannelsAllocator. Failed to build TT-Fabric router. Internal "
        "error.");
    this->receiver_channel_to_downstream_adapter =
        std::make_shared<tt::tt_fabric::StaticSizedChannelConnectionWriterAdapter>(*static_allocator, config.topology);

    // Add this log right at the beginning of the constructor body
    log_debug(
        tt::LogFabric,
        "FabricEriscDatamoverBuilder config for device (local chip_id: {}, peer chip_id: {}): "
        "buffer_size={}, topology={}, num_sender_ch={}, num_receiver_ch={}, direction={}",
        local_fabric_node_id.chip_id,
        peer_fabric_node_id.chip_id,
        config.channel_buffer_size_bytes,
        static_cast<int>(config.topology),
        config.num_used_sender_channels,
        config.num_used_receiver_channels,
        static_cast<int>(direction));
}

void FabricEriscDatamoverBuilder::get_telemetry_compile_time_args(std::vector<uint32_t>& ct_args) const {
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    uint32_t telemetry_mode = static_cast<uint32_t>(rtoptions.get_enable_fabric_telemetry() ? 1 : 0);
    ct_args.push_back(telemetry_mode);

    // Add telemetry buffer address (16B aligned)
    ct_args.push_back(static_cast<uint32_t>(config.perf_telemetry_buffer_address));

    // Add code profiling arguments (conditionally enabled)
    if (rtoptions.get_enable_fabric_code_profiling_rx_ch_fwd()) {
        // Enable RECEIVER_CHANNEL_FORWARD timer (bit 0)
        uint32_t code_profiling_enabled_timers = static_cast<uint32_t>(CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD);
        ct_args.push_back(code_profiling_enabled_timers);

        // Add code profiling buffer address (16B aligned)
        ct_args.push_back(static_cast<uint32_t>(config.code_profiling_buffer_address));
    } else {
        // Code profiling disabled - add zeros
        ct_args.push_back(0); // No timers enabled
        ct_args.push_back(0); // No buffer address
    }
}

/*
 * Channel ct args schema:
 * 1) First arg: defines the number of "pools": `num_channel_pools`
 * 2) The next `num_channel_pools` of compile time args is an array of `FabricChannelPoolType`: `channel_pool_types`
 * 3) The next indeterminate number of args are the args for the individual pools. For each entry in
 * `channel_pool_types`, you invoke the corresponding `Pool` class by passing the next CT arg index. After each `Pool`,
 * you increment the compile time arg index by <type>::NUM_ARGS_USED. 4) All of the pools from step 3 should be placed
 * into a constexprable tuple.
 *
 * 5) a sender channel to pool index mapping is passed as compile time args
 * 6) a sender channel to pool type mapping is passed as compile time args
 * 7) a receiver channel to pool index mapping is passed as compile time args
 * 8) a receiver channel to pool type mapping is passed as compile time args
 */

std::vector<uint32_t> FabricEriscDatamoverBuilder::get_compile_time_args(uint32_t risc_id) const {
    TT_ASSERT(this->local_fabric_node_id != this->peer_fabric_node_id);

    // Tie break policy for selecting the handshake master:
    // 1. If both nodes are on the same mesh, compare chip_ids
    // 2. If nodes are on different meshes, compare mesh_ids (since chip_ids can alias across meshes)
    auto peer_tie_break_id = (local_fabric_node_id.mesh_id == peer_fabric_node_id.mesh_id)
                                 ? peer_fabric_node_id.chip_id
                                 : *(peer_fabric_node_id.mesh_id);
    auto local_tie_break_id = (local_fabric_node_id.mesh_id == peer_fabric_node_id.mesh_id)
                                  ? local_fabric_node_id.chip_id
                                  : *(local_fabric_node_id.mesh_id);
    bool is_handshake_master = local_tie_break_id < peer_tie_break_id;

    // TODO print allocations

    // TODO: promote to user-configurable parameter (user could be just control plane based on arch in this case)
    // specifies if we do spin waits on eth_txq_busy in send_next_data
    const bool eth_txq_spin_wait_send_next_data = false;
    const bool eth_txq_spin_wait_receiver_send_completion_ack = false;

    // TODO: allow specification per eth txq
    const size_t default_num_eth_txq_data_packet_accept_ahead = 32;
    // By default have the ERISC cores context switch to base routing FW every 4K cycles during the peer handshake.
    // This allows host to write Fabric kernels to remote chips over ethernet, when ERISC cores already running fabric
    // are waiting for the handshake to complete.
    const size_t default_handshake_context_switch_timeout = 4096;
    size_t num_sender_channels = config.num_used_sender_channels;
    size_t num_receiver_channels = config.num_used_receiver_channels;
    auto dispatch_core_type =
        tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config().get_core_type();
    uint32_t my_eth_channel_ = [&]() -> uint32_t {
        if (dispatch_core_type == CoreType::WORKER) {
            return this->my_eth_channel;
        } else if (dispatch_core_type == CoreType::ETH) {
            return tt::tt_fabric::USE_DYNAMIC_CREDIT_ADDR;
        } else {
            TT_THROW("Fabric Mux does not support core type {}", enchantum::to_string(dispatch_core_type));
        }
    }();

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto local_physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(this->local_fabric_node_id);
    auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(local_physical_chip_id);

    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();

    auto sender_channel_to_check = get_worker_connected_sender_channel(direction, topology);

    const auto remote_routing_direction =
        control_plane.get_forwarding_direction(this->peer_fabric_node_id, this->local_fabric_node_id);

    uint32_t remote_worker_sender_channel = 0;
    uint32_t remote_vc1_sender_channel = 0;
    bool skip_src_ch_id_update = false;
    // when there is no remote router paired with current router, don't care about skip_src_ch_id_update and set it to
    // false
    if (remote_routing_direction.has_value()) {
        skip_src_ch_id_update = this->has_tensix_extension;
        auto remote_eth_direction = control_plane.routing_direction_to_eth_direction(remote_routing_direction.value());
        remote_worker_sender_channel = get_worker_connected_sender_channel(remote_eth_direction, topology);
        remote_vc1_sender_channel =
            this->dateline_connection ? remote_worker_sender_channel : get_vc1_connected_sender_channel(topology);
    }

    bool update_pkt_hdr_on_rx_ch = true;
    bool fabric_tensix_extension_enabled = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config() !=
                                           tt::tt_fabric::FabricTensixConfig::DISABLED;
    bool support_pkt_hdr_update_on_sender_channel =
        !fabric_tensix_extension_enabled &&
        (topology == tt::tt_fabric::Topology::Torus || topology == tt::tt_fabric::Topology::Mesh);
    if (support_pkt_hdr_update_on_sender_channel) {
        // The only mode that currently supports packet header update on sender channel is
        // a mesh when we aren't also implementing a mux on a tensix core.
        // The other topologies haven't been updated to support this new mode yet.
        // For mux case, information about direction isn't tied to the channels yet.
        // Information about turning and how to update the packet header route isn't exposed
        // to the mux yet to implement sender side updates.
        update_pkt_hdr_on_rx_ch = false;
    }

    // TODO: this validation should be done in the allocator with the channel IDs passed in
    auto channel_allocator = config.channel_allocator.get();
    const auto static_channel_allocator =
        dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(channel_allocator);
    TT_FATAL(static_channel_allocator != nullptr, "Channel allocator must be a FabricStaticSizedChannelsAllocator.");
    size_t receiver_channel_num_buffers = this->dateline_connection
                                              ? static_channel_allocator->get_receiver_channel_number_of_slots(1)
                                              : static_channel_allocator->get_receiver_channel_number_of_slots(0);
    TT_FATAL(
        static_channel_allocator->get_sender_channel_number_of_slots(sender_channel_to_check) > 0,
        "Sender channel on direction {} num buffers must be greater than 0",
        sender_channel_to_check);
    TT_FATAL(receiver_channel_num_buffers > 0, "Receiver channel num buffers must be greater than 0");

    const auto& stream_ids = StreamRegAssignments::get_all_stream_ids();
    auto ct_args = std::vector<uint32_t>(stream_ids.begin(), stream_ids.end());
    ct_args.push_back(0xFFEE0001);

    // add the downstream tensix connection arg here, num_downstream_tensix_connections
    ct_args.push_back(this->num_downstream_tensix_connections);

    // when have dateline vc (vc1) and with tensix exntension enabled, need to send vc1 to downstream fabric router
    // instead of downstream tensix exntension.
    bool vc1_has_different_downstream_dest =
        fabric_context.need_deadlock_avoidance_support(this->direction) && this->has_tensix_extension;

    // Compute edge-facing flags for this ethernet core/channel
    // Use the builder's logical channel id (my_eth_channel) for facing classification
    bool is_intermesh_router_on_edge = false;
    bool is_intramesh_router_on_edge = false;
    const auto& intermesh_chans = control_plane.get_intermesh_facing_eth_chans(this->local_fabric_node_id);
    const auto& intramesh_chans = control_plane.get_intramesh_facing_eth_chans(this->local_fabric_node_id);
    bool is_edge_chip = !intermesh_chans.empty();
    if (is_edge_chip) {
        is_intermesh_router_on_edge =
            std::find(intermesh_chans.begin(), intermesh_chans.end(), this->my_eth_channel) != intermesh_chans.end();
        is_intramesh_router_on_edge =
            std::find(intramesh_chans.begin(), intramesh_chans.end(), this->my_eth_channel) != intramesh_chans.end();
    }

    const std::vector<uint32_t> main_args_part1 = {
        num_sender_channels,
        num_receiver_channels,
        config.num_fwd_paths,
        this->wait_for_host_signal ? 1 : 0,

        this->firmware_context_switch_interval,
        this->fuse_receiver_flush_and_completion_ptr,
        fabric_context.need_deadlock_avoidance_support(this->direction),
        this->dateline_connection,
        control_plane.is_cross_host_eth_link(local_physical_chip_id, this->my_eth_channel),
        is_handshake_master,
        this->handshake_address,
        this->channel_buffer_size,
        vc1_has_different_downstream_dest,
        this->has_tensix_extension};

    const std::vector<uint32_t> main_args_part2 = {
        config.skip_receiver_channel_1_connection,
        config.skip_sender_channel_1_connection,
        config.skip_sender_vc1_channel_connection,

        config.sender_channels_worker_conn_info_base_address[0],
        config.sender_channels_worker_conn_info_base_address[1],
        config.sender_channels_worker_conn_info_base_address[2],
        config.sender_channels_worker_conn_info_base_address[3],
        config.sender_channels_worker_conn_info_base_address[4],

        this->termination_signal_ptr,
        this->edm_local_sync_ptr,
        this->edm_local_tensix_sync_ptr,
        this->edm_status_ptr,

        config.notify_worker_of_read_counter_update_src_address,
        0x7a9b3c4d,  // special tag marker to catch incorrect ct args

        // fabric counters
        FabricEriscDatamoverConfig::enable_fabric_counters,
        config.receiver_channels_counters_address[0],
        config.receiver_channels_counters_address[1],
        config.sender_channels_counters_address[0],
        config.sender_channels_counters_address[1],
        config.sender_channels_counters_address[2],
        config.sender_channels_counters_address[3],
        config.sender_channels_counters_address[4],

        // fabric pkt header recording
        FabricEriscDatamoverConfig::enable_fabric_pkt_header_recording,

        config.receivers_completed_packet_header_cb_address[0],
        FabricEriscDatamoverConfig::receiver_completed_packet_header_cb_size_headers,
        config.receivers_completed_packet_header_cb_address[1],
        FabricEriscDatamoverConfig::receiver_completed_packet_header_cb_size_headers,
        config.senders_completed_packet_header_cb_address[0],
        FabricEriscDatamoverConfig::sender_completed_packet_header_cb_size_headers,
        config.senders_completed_packet_header_cb_address[1],
        FabricEriscDatamoverConfig::sender_completed_packet_header_cb_size_headers,
        config.senders_completed_packet_header_cb_address[2],
        FabricEriscDatamoverConfig::sender_completed_packet_header_cb_size_headers,
        config.senders_completed_packet_header_cb_address[3],
        FabricEriscDatamoverConfig::sender_completed_packet_header_cb_size_headers,
        config.senders_completed_packet_header_cb_address[4],
        FabricEriscDatamoverConfig::sender_completed_packet_header_cb_size_headers,
        config.risc_configs[risc_id].is_sender_channel_serviced(0),
        config.risc_configs[risc_id].is_sender_channel_serviced(1),
        config.risc_configs[risc_id].is_sender_channel_serviced(2),
        config.risc_configs[risc_id].is_sender_channel_serviced(3),
        config.risc_configs[risc_id].is_sender_channel_serviced(4),
        config.risc_configs[risc_id].is_receiver_channel_serviced(0),
        config.risc_configs[risc_id].is_receiver_channel_serviced(1),
        config.risc_configs[risc_id].enable_handshake(),
        config.risc_configs[risc_id].enable_context_switch(),
        config.risc_configs[risc_id].enable_interrupts(),
        config.sender_txq_id,
        config.receiver_txq_id,
        config.risc_configs[risc_id].iterations_between_ctx_switch_and_teardown_checks(),
        fabric_context.is_2D_routing_enabled(),
        this->direction,
        soc_desc.get_num_eth_channels(),

        eth_txq_spin_wait_send_next_data,
        eth_txq_spin_wait_receiver_send_completion_ack,
        default_num_eth_txq_data_packet_accept_ahead,

        default_handshake_context_switch_timeout,
        static_cast<uint32_t>(
            this->firmware_context_switch_type == FabricEriscDatamoverContextSwitchType::WAIT_FOR_IDLE),
        my_eth_channel_,

        risc_id,
        this->get_configured_risc_count(),

        update_pkt_hdr_on_rx_ch,

        requires_forced_assignment_to_noc1(),
        is_intermesh_router_on_edge,
        is_intramesh_router_on_edge,
        // Special marker to help with identifying misalignment bugs
        0x00c0ffee};

    // Add first part of main arguments to ct_args
    ct_args.insert(ct_args.end(), main_args_part1.begin(), main_args_part1.end());

    // Conditionally add remote channel info when skip_src_ch_id_update is true
    // (these values are used to initialize src channel IDs once, rather than updating them dynamically)
    if (skip_src_ch_id_update) {
        ct_args.push_back(remote_vc1_sender_channel);
        ct_args.push_back(remote_worker_sender_channel);
    }

    // special tag
    ct_args.push_back(0xabcd9876);

    // Emit pool data via multi-pool coordinator (steps 1-4 of schema: special tag, num_pools, pool_types, individual
    // pool CT args)
    config.multi_pool_allocator->emit_ct_args(
        ct_args, config.num_fwd_paths, num_sender_channels, num_receiver_channels);

    // Emit channel-to-pool mappings (steps 5-8 of schema)
    ct_args.push_back(0xabaddad8);
    config.channel_to_pool_mapping->emit_ct_args(ct_args);

    // Emit remote channel pool data (for remote_receiver_channels initialization)
    // Create a multi-pool for remote channels, following the same pattern as local channels
    ct_args.push_back(0xabaddad6);

    // Create remote multi-pool allocator with the remote channels allocator
    TT_FATAL(config.remote_channels_allocator != nullptr, "Remote channels allocator must be non-null");
    MultiPoolChannelAllocator remote_multi_pool_allocator(
        {config.remote_channels_allocator}, {FabricChannelPoolType::STATIC});

    // Emit remote channel pool data via multi-pool coordinator
    remote_multi_pool_allocator.emit_ct_args(ct_args, config.num_fwd_paths, 0, num_receiver_channels);

    config.remote_channel_to_pool_mapping->emit_ct_args(ct_args);

    ct_args.push_back(0xabaddad7);
    receiver_channel_to_downstream_adapter->emit_ct_args(ct_args, config.num_fwd_paths);
    ct_args.push_back(0xabaddad9);

    // Add second part of main arguments to ct_args
    ct_args.insert(ct_args.end(), main_args_part2.begin(), main_args_part2.end());

    for (size_t i = 0; i < num_sender_channels; i++) {
        ct_args.push_back(this->sender_channel_connection_liveness_check_disable_array[i]);
    }

    // Sender channel args
    for (size_t i = 0; i < num_sender_channels; i++) {
        ct_args.push_back(config.sender_channel_ack_noc_ids[i]);
    }

    // Populate the sender ack cmd buf ids for each datapath
    for (size_t i = 0; i < num_sender_channels; i++) {
        ct_args.push_back(config.sender_channel_ack_cmd_buf_ids[i]);
    }

    for (size_t i = 0; i < num_receiver_channels; i++) {
        ct_args.push_back(config.receiver_channel_forwarding_noc_ids[i]);
    }
    for (size_t i = 0; i < num_receiver_channels; i++) {
        ct_args.push_back(
            config.receiver_channel_forwarding_data_cmd_buf_ids[i]);  // maps to
                                                                      // receiver_channel_forwarding_data_cmd_buf_ids
    }
    for (size_t i = 0; i < num_receiver_channels; i++) {
        ct_args.push_back(
            config.receiver_channel_forwarding_sync_cmd_buf_ids[i]);  // maps to
                                                                      // receiver_channel_forwarding_sync_cmd_buf_ids
    }
    for (size_t i = 0; i < num_receiver_channels; i++) {
        // TODO: pass this to the tranmission file
        ct_args.push_back(
            config.receiver_channel_local_write_noc_ids[i]);  // maps to receiver_channel_local_write_noc_ids
    }
    for (size_t i = 0; i < num_receiver_channels; i++) {
        ct_args.push_back(
            config.receiver_channel_local_write_cmd_buf_ids[i]);  // maps to receiver_channel_local_write_cmd_buf_ids
    }
    ct_args.push_back(config.edm_noc_vc);

    // Special marker to help with identifying misalignment bugs
    ct_args.push_back(0x10c0ffee);

    get_telemetry_compile_time_args(ct_args);

    // Special marker 2
    ct_args.push_back(0x20c0ffee);

    bool multi_txq_enabled = config.sender_txq_id != config.receiver_txq_id;
    if (multi_txq_enabled) {
        ct_args.push_back(config.to_sender_channel_remote_ack_counters_base_addr);
        ct_args.push_back(config.to_sender_channel_remote_completion_counters_base_addr);
        ct_args.push_back(config.receiver_channel_remote_ack_counters_base_addr);
        ct_args.push_back(config.receiver_channel_remote_completion_counters_base_addr);
    }

    ct_args.push_back(0x30c0ffee);
    return ct_args;
}

std::vector<uint32_t> FabricEriscDatamoverBuilder::get_runtime_args() const {
    // auto &downstream_connection_1 = receiver_channel_to_downstream_adapter.at(0);
    // auto &downstream_connection_2 = receiver_channel_to_downstream_adapter.at(1);

    std::vector<uint32_t> rt_args = {
        this->sender_channels_connection_semaphore_id[0],
        this->sender_channels_connection_semaphore_id[1],
        this->sender_channels_connection_semaphore_id[2],
        this->sender_channels_connection_semaphore_id[3],
        this->sender_channels_connection_semaphore_id[4],

        this->downstream_vcs_sender_channel_buffer_index_semaphore_id[0],
        this->downstream_vcs_sender_channel_buffer_index_semaphore_id[1],
        this->downstream_vcs_sender_channel_buffer_index_semaphore_id[2],
        this->downstream_vcs_sender_channel_buffer_index_semaphore_id[3],
        this->downstream_vcs_sender_channel_buffer_index_semaphore_id[4]};

    receiver_channel_to_downstream_adapter->pack_inbound_channel_rt_args(0, rt_args);
    receiver_channel_to_downstream_adapter->pack_inbound_channel_rt_args(1, rt_args);
    // downstream_connection_1->pack_inbound_channel_rt_args(0, rt_args);
    // downstream_connection_2->pack_inbound_channel_rt_args(1, rt_args);

    auto args_pt2 = std::vector<uint32_t>{
        this->receiver_channels_downstream_teardown_semaphore_id[0].value_or(-1),
        this->receiver_channels_downstream_teardown_semaphore_id[1].value_or(-1),
        this->receiver_channels_downstream_teardown_semaphore_id[2].value_or(-1),
        this->receiver_channels_downstream_teardown_semaphore_id[3].value_or(-1),
        this->receiver_channels_downstream_teardown_semaphore_id[4].value_or(-1),
        this->sender_channels_flow_control_semaphore_id[0],
        this->sender_channels_flow_control_semaphore_id[1],
        this->sender_channels_flow_control_semaphore_id[2],
        this->sender_channels_flow_control_semaphore_id[3],
        this->sender_channels_flow_control_semaphore_id[4]};

    rt_args.reserve(rt_args.size() + args_pt2.size());
    std::ranges::copy(args_pt2, std::back_inserter(rt_args));
    return rt_args;
}

FabricEriscDatamoverBuilder FabricEriscDatamoverBuilder::build(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    const CoreCoord& ethernet_core,
    ChipId local_physical_chip_id,
    ChipId peer_physical_chip_id,
    const FabricEriscDatamoverConfig& config,
    bool build_in_worker_connection_mode,
    FabricEriscDatamoverType fabric_edm_type,
    eth_chan_directions direction,
    bool has_tensix_extension) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    log_debug(
        tt::LogFabric,
        "Building FabricEriscDatamover for device {}:  "
        "channel_buffer_size={}, topology={}, num_sender_channels={}, num_receiver_channels={}",
        device->id(),
        config.channel_buffer_size_bytes,
        (int)config.topology,
        config.num_used_sender_channels,
        config.num_used_receiver_channels);
    return FabricEriscDatamoverBuilder::build(
        device,
        program,
        ethernet_core,
        control_plane.get_fabric_node_id_from_physical_chip_id(local_physical_chip_id),
        control_plane.get_fabric_node_id_from_physical_chip_id(peer_physical_chip_id),
        config,
        build_in_worker_connection_mode,
        fabric_edm_type,
        direction,
        has_tensix_extension);
}

FabricEriscDatamoverBuilder FabricEriscDatamoverBuilder::build(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    const CoreCoord& ethernet_core,
    const FabricNodeId& local_fabric_node_id,
    const FabricNodeId& peer_fabric_node_id,
    const FabricEriscDatamoverConfig& config,
    bool build_in_worker_connection_mode,
    FabricEriscDatamoverType fabric_edm_type,
    eth_chan_directions direction,
    bool has_tensix_extension) {
    std::array<size_t, builder_config::num_sender_channels> sender_channels_buffer_index_semaphore_id{};
    std::array<size_t, builder_config::num_sender_channels> sender_channels_flow_control_semaphore_id{};
    std::array<size_t, builder_config::num_sender_channels> sender_channels_connection_semaphore_id{};
    std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>
        receiver_channels_downstream_flow_control_semaphore_id;
    std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>
        receiver_channels_downstream_teardown_semaphore_id;

    auto remote_pool_allocators =
        std::vector<std::shared_ptr<tt::tt_fabric::FabricChannelAllocator>>{config.remote_channels_allocator};
    auto remote_pool_types =
        std::vector<tt::tt_fabric::FabricChannelPoolType>{tt::tt_fabric::FabricChannelPoolType::STATIC};
    auto remote_multi_pool_allocator = std::make_shared<tt::tt_fabric::MultiPoolChannelAllocator>(
        std::move(remote_pool_allocators), std::move(remote_pool_types));

    log_debug(
        tt::LogFabric,
        "FABRIC NODE ID: M={},D={} eth=(x={},y={})\n"
        "\tnum_sender_channels={}, num_receiver_channels={}\n"
        "\tchannel_allocator={}\n"
        "\tremote_channel_allocator={}\n",

        local_fabric_node_id.mesh_id,
        local_fabric_node_id.chip_id,
        ethernet_core.x,
        ethernet_core.y,
        config.num_used_sender_channels,
        config.num_used_receiver_channels,
        *config.channel_allocator,
        *remote_multi_pool_allocator->get_pool(0));

    if (build_in_worker_connection_mode) {
        for (uint32_t i = 0; i < builder_config::num_receiver_channels; i++) {
            receiver_channels_downstream_flow_control_semaphore_id[i] = 0;
            receiver_channels_downstream_teardown_semaphore_id[i] = 0;
        }
        // Sender channel 0 uses addresses instead of ids in persistent mode
        sender_channels_buffer_index_semaphore_id[0] = config.sender_channels_buffer_index_semaphore_address[0];
        sender_channels_flow_control_semaphore_id[0] = config.sender_channels_local_flow_control_semaphore_address[0];
        sender_channels_connection_semaphore_id[0] = config.sender_channels_connection_semaphore_address[0];
        for (uint32_t i = 1; i < builder_config::num_sender_channels; i++) {
            sender_channels_flow_control_semaphore_id[i] = 0;
            sender_channels_connection_semaphore_id[i] = 0;
            sender_channels_buffer_index_semaphore_id[i] = 0;
        }
    } else {
        const bool is_2D_routing =
            tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context().is_2D_routing_enabled();
        uint32_t num_vc0_downstream_edms = builder_config::get_vc0_downstream_edm_count(is_2D_routing);

        // Setup VC0 downstrteam edm semaphore settings.
        // 1D has 1 downstream edm. 2D has 3 downstream EDMs
        // 2D uses the reserved addresses in L1 from FabricEriscDatamoverConfig
        for (uint32_t i = 0; i < num_vc0_downstream_edms; i++) {
            receiver_channels_downstream_flow_control_semaphore_id[i] =
                config.receiver_channels_downstream_flow_control_semaphore_address[i];
            receiver_channels_downstream_teardown_semaphore_id[i] =
                config.receiver_channels_downstream_teardown_semaphore_address[i];
        }
        // Setup VC1 downstream edm
        // 1D and 2D have 1 downstream edm for VC1 in the diretion of respective axis
        receiver_channels_downstream_flow_control_semaphore_id[num_vc0_downstream_edms] =
            config.receiver_channels_downstream_flow_control_semaphore_address[num_vc0_downstream_edms];
        receiver_channels_downstream_teardown_semaphore_id[num_vc0_downstream_edms] =
            config.receiver_channels_downstream_teardown_semaphore_address[num_vc0_downstream_edms];
        uint32_t num_sender_channels = builder_config::get_sender_channel_count(is_2D_routing);
        for (uint32_t i = 0; i < num_sender_channels; i++) {
            sender_channels_buffer_index_semaphore_id[i] = config.sender_channels_buffer_index_semaphore_address[i];
            sender_channels_flow_control_semaphore_id[i] =
                config.sender_channels_local_flow_control_semaphore_address[i];
            sender_channels_connection_semaphore_id[i] = config.sender_channels_connection_semaphore_address[i];
        }
    }
    return FabricEriscDatamoverBuilder(
        ethernet_core,
        device->ethernet_core_from_logical_core(ethernet_core).x,
        device->ethernet_core_from_logical_core(ethernet_core).y,
        local_fabric_node_id,
        peer_fabric_node_id,

        receiver_channels_downstream_flow_control_semaphore_id,
        receiver_channels_downstream_teardown_semaphore_id,
        sender_channels_flow_control_semaphore_id,
        sender_channels_connection_semaphore_id,
        sender_channels_buffer_index_semaphore_id,

        config,
        direction,
        build_in_worker_connection_mode,
        fabric_edm_type,
        has_tensix_extension);
}

// SenderWorkerAdapterSpec FabricEriscDatamoverBuilder::build_connection_to_worker_channel() const {
//     log_trace(tt::LogFabric, "Building connection to persistent fabric");
//     static constexpr uint32_t worker_chan = 0;
//     TT_FATAL(
//         sender_channels_buffer_index_semaphore_id[worker_chan] !=
//             sender_channels_flow_control_semaphore_id[worker_chan],
//         "Internal error - sender_channel_buffer_index_semaphore_id and sender_channel_flow_control_semaphore_id "
//         "aliased eachother");

//     auto channel_allocator = config.channel_allocator.get();
//     TT_FATAL(dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(channel_allocator) != nullptr, "Only
//     FabricStaticSizedChannelsAllocator is supported currently."); const auto static_channel_allocator =
//     dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(channel_allocator); return
//     SenderWorkerAdapterSpec{
//         this->my_noc_x,
//         this->my_noc_y,
//         static_channel_allocator->get_sender_channel_base_address(worker_chan),
//         static_channel_allocator->get_sender_channel_number_of_slots(worker_chan),
//         this->sender_channels_flow_control_semaphore_id[worker_chan],
//         this->sender_channels_connection_semaphore_id[worker_chan],
//         this->config.sender_channels_worker_conn_info_base_address[worker_chan],
//         this->config.channel_buffer_size_bytes,
//         this->sender_channels_buffer_index_semaphore_id[worker_chan],
//         this->direction};
// }

SenderWorkerAdapterSpec FabricEriscDatamoverBuilder::build_connection_to_fabric_channel(uint32_t ds_edm) {
    const bool is_2D_routing =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context().is_2D_routing_enabled();
    auto max_ds_edm_count = builder_config::get_sender_channel_count(is_2D_routing);
    if (ds_edm >= max_ds_edm_count) {
        TT_THROW("Invalid VC");
    }

    auto channel_allocator = config.channel_allocator.get();
    const auto static_channel_allocator =
        dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(channel_allocator);
    TT_FATAL(static_channel_allocator != nullptr, "Channel allocator must be a FabricStaticSizedChannelsAllocator.");
    size_t sender_channels_num_buffer = 0;
    if (this->has_tensix_extension) {
        // for edm builders with has_tensix_extension set to true (non-dispatch links and enabled fabric tensix config),
        // the vc1 sender channel should be on fabric erisc router, and we use the last sender channel for vc1
        sender_channels_num_buffer = static_channel_allocator->get_sender_channel_number_of_slots(ds_edm);
    } else {
        // for all edm types except for dateline upstream will have non zero buffer slots for channel 1,
        // for dateline upstream channel 1 is removed and we need to use channel 2.
        static constexpr std::size_t none_zero_buffer_slot_idx = 1;
        static constexpr std::size_t dateline_upstream_none_zero_idx = 2;

        switch (this->fabric_edm_type) {
            case FabricEriscDatamoverType::DatelineUpstream:
                sender_channels_num_buffer =
                    static_channel_allocator->get_sender_channel_number_of_slots(dateline_upstream_none_zero_idx);
                break;
            default:
                sender_channels_num_buffer =
                    static_channel_allocator->get_sender_channel_number_of_slots(none_zero_buffer_slot_idx);
                break;
        }
    }

    TT_FATAL(sender_channels_num_buffer != 0, "sender_channels_num_buffer should not be 0!");

    this->sender_channel_connection_liveness_check_disable_array[ds_edm] = true;
    return SenderWorkerAdapterSpec{
        this->my_noc_x,
        this->my_noc_y,
        static_channel_allocator->get_sender_channel_base_address(ds_edm),
        sender_channels_num_buffer,
        this->sender_channels_flow_control_semaphore_id[ds_edm],
        this->sender_channels_connection_semaphore_id[ds_edm],
        this->config.sender_channels_worker_conn_info_base_address[ds_edm],
        this->config.channel_buffer_size_bytes,
        this->sender_channels_buffer_index_semaphore_id[ds_edm],
        eth_chan_directions::EAST};
}

// Internal implementation for connect_to_downstream_edm
void FabricEriscDatamoverBuilder::connect_to_downstream_edm_impl(
    FabricDatamoverBuilder downstream_builder, FabricDatamoverBuilder vc1_edm_builder) {
    TT_FATAL(
        !this->build_in_worker_connection_mode, "Tried to connect EDM to downstream builder in worker connection mode");

    std::visit(
        [this, &vc1_edm_builder](auto&& builder_ref) {
            auto& builder = builder_ref.get();

            [[maybe_unused]] const auto ds_noc_x = builder.get_noc_x();
            [[maybe_unused]] const auto ds_noc_y = builder.get_noc_y();
            eth_chan_directions ds_dir = builder.get_direction();

            log_debug(
                tt::LogTest,
                "EDM at x={}, y={}, Direction={}, FabricNodeId={} :: Connecting to downstream EDM at x={}, y={}, "
                "Direction={}",
                my_noc_x,
                my_noc_y,
                direction,
                local_fabric_node_id,
                ds_noc_x,
                ds_noc_y,
                ds_dir);

            const auto& fabric_context =
                tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
            const bool is_2D_routing = fabric_context.is_2D_routing_enabled();

            // Setup VC0 connection
            constexpr uint32_t ds_vc0_index = 0;
            auto ds_vc0_send_chan = get_downstream_edm_sender_channel(is_2D_routing, this->direction);
            setup_downstream_vc_connection(builder, ds_vc0_index, ds_vc0_send_chan, false);

            if (!fabric_context.need_deadlock_avoidance_support(this->direction)) {
                return;
            }

            // for vc1, need to connect using the edm builder not tensix builder.
            if (is_2D_routing) {
                // for 2D routing we can only connect VC1 if the downstream is on the same axis
                bool connect_vc1 =
                    (this->direction == eth_chan_directions::EAST && ds_dir == eth_chan_directions::WEST) ||
                    (this->direction == eth_chan_directions::WEST && ds_dir == eth_chan_directions::EAST) ||
                    (this->direction == eth_chan_directions::NORTH && ds_dir == eth_chan_directions::SOUTH) ||
                    (this->direction == eth_chan_directions::SOUTH && ds_dir == eth_chan_directions::NORTH);
                if (!connect_vc1) {
                    return;
                }
            }

            // Setup VC1 connection if needed
            constexpr uint32_t ds_index = 1;
            auto vc1_send_chan = builder_config::get_sender_channel_count(is_2D_routing) - 1;
            std::visit(
                [this, ds_index, vc1_send_chan](auto&& vc1_builder_ref) {
                    auto& vc1_builder = vc1_builder_ref.get();
                    this->setup_downstream_vc_connection(vc1_builder, ds_index, vc1_send_chan, true);
                },
                vc1_edm_builder);
        },
        downstream_builder);
}

void FabricEriscDatamoverBuilder::connect_to_downstream_edm(FabricDatamoverBuilder downstream_builder) {
    connect_to_downstream_edm_impl(downstream_builder, downstream_builder);
}

void FabricEriscDatamoverBuilder::connect_to_downstream_edm(
    FabricDatamoverBuilder downstream_builder, FabricDatamoverBuilder vc1_edm_builder) {
    connect_to_downstream_edm_impl(downstream_builder, vc1_edm_builder);
}

// TODO: take the downstream sender channel, based on the VC index, and use it to construct our
// `to_sender_channel_adapter` The `to_sender_channel_adapter` type is resolved based on the type of the downstream
// sender channel it is connecting to.
//   downstream == static? => instantiate static_sender_channel_adapter
//   downstream == elastic? => instantiate elastic_sender_channel_adapter
template <typename BuilderType>
void FabricEriscDatamoverBuilder::setup_downstream_vc_connection(
    BuilderType& downstream_builder, uint32_t vc_idx, uint32_t channel_id, bool is_vc1) {
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const bool is_2D_routing = fabric_context.is_2D_routing_enabled();
    const auto ds_noc_x = downstream_builder.get_noc_x();
    const auto ds_noc_y = downstream_builder.get_noc_y();
    eth_chan_directions ds_dir = downstream_builder.get_direction();

    auto adapter_spec = downstream_builder.build_connection_to_fabric_channel(channel_id);

    if constexpr (std::is_same_v<BuilderType, FabricTensixDatamoverBuilder>) {
        this->num_downstream_tensix_connections++;
        downstream_builder.append_upstream_routers_noc_xy(this->my_noc_x, this->my_noc_y);
    }

    auto channel_allocator = config.channel_allocator.get();
    const auto static_channel_allocator =
        dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(channel_allocator);
    TT_FATAL(static_channel_allocator != nullptr, "Channel allocator must be a FabricStaticSizedChannelsAllocator.");
    auto adapter_ptr = receiver_channel_to_downstream_adapter.get();//receiver_channel_to_downstream_adapter.at(vc_idx);
    TT_FATAL(adapter_ptr != nullptr, "Adapter is not set. Failed to build TT-Fabric router. Internal error.");
    adapter_ptr->add_downstream_connection(adapter_spec, vc_idx, ds_dir, CoreCoord(ds_noc_x, ds_noc_y), is_2D_routing, is_vc1);
}

eth_chan_directions FabricEriscDatamoverBuilder::get_direction() const { return this->direction; }

size_t FabricEriscDatamoverBuilder::get_configured_risc_count() const { return this->config.risc_configs.size(); }

size_t FabricEriscDatamoverBuilder::get_noc_x() const { return this->my_noc_x; }

size_t FabricEriscDatamoverBuilder::get_noc_y() const { return this->my_noc_y; }

void FabricEriscDatamoverBuilder::teardown_from_host(
    tt::tt_metal::IDevice* d, tt::tt_fabric::TerminationSignal termination_signal) const {
    std::vector<uint32_t> val(1, termination_signal);
    tt::tt_metal::detail::WriteToDeviceL1(
        d,
        d->logical_core_from_ethernet_core(CoreCoord(this->my_noc_x, this->my_noc_y)),
        config.termination_signal_address,
        val,
        CoreType::ETH);
}

void FabricEriscDatamoverBuilder::set_firmware_context_switch_interval(size_t interval) {
    this->firmware_context_switch_interval = interval;
}

void FabricEriscDatamoverBuilder::set_wait_for_host_signal(bool wait_for_host_signal) {
    this->wait_for_host_signal = wait_for_host_signal;
}

void FabricEriscDatamoverBuilder::set_firmware_context_switch_type(FabricEriscDatamoverContextSwitchType type) {
    this->firmware_context_switch_type = type;
}

}  // namespace tt::tt_fabric
