// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <enchantum/enchantum.hpp>

#include <cstdint>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
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
#include <vector>

#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_remote_channels_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_router_recipe.hpp"
#include "tt_metal/fabric/builder/channel_to_pool_mapping.hpp"
#include "tt_metal/fabric/builder/multi_pool_channel_allocator.hpp"
#include "tt_metal/fabric/builder/connection_writer_adapter.hpp"

#include "impl/context/metal_context.hpp"
#include "core_coord.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_common.hpp"

namespace tt::tt_metal {
class Program;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

size_t FabricEriscDatamoverBuilder::get_max_packet_payload_size_for_arch(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::WORMHOLE_B0: return max_packet_payload_size_bytes_wormhole;
        case tt::ARCH::BLACKHOLE: return max_packet_payload_size_bytes_blackhole;
        default:
            TT_FATAL(false, "Custom packet sizes not supported for architecture: {}", tt::arch_to_str(arch));
            return 0;  // unreachable
    }
}

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
// Returns true when fabric 2-ERISC should be considered enabled for builders.
// This is disabled for Wormhole and enabled for Blackhole when 2-eriscs can be dispatched to
// and if we are not building with the tensix mux extension (due to stack size issue).
bool is_fabric_two_erisc_enabled() {
    auto& mc = tt::tt_metal::MetalContext::instance();
    // Force-disable if the override is present
    bool force_disable_2_erisc = mc.rtoptions().get_disable_fabric_2_erisc_mode();
    if (force_disable_2_erisc) {
        log_debug(tt::LogFabric, "Disabling fabric 2-ERISC mode due to force disable");
        return false;
    }

    const auto& hal = mc.hal();
    // by default, enable only single erisc mode for future architectures as well to simplify bringup
    bool arch_bh = hal.get_arch() == tt::ARCH::BLACKHOLE;

    // out of stack size issue on the erisc, to be investigated
    bool tensix_extensions_enabled = mc.get_fabric_tensix_config() != tt::tt_fabric::FabricTensixConfig::DISABLED;

    bool single_erisc_dispatch = hal.get_num_risc_processors(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH) < 2;

    // 2d dynamic fabric doesn't properly support 2-erisc yet but is being deprecated anyways so we
    // simply disable 2-erisc on it for now.
    // Issue [#32419](https://github.com/tenstorrent/tt-metal/issues/32419)
    return arch_bh && !tensix_extensions_enabled && !single_erisc_dispatch;
}

void configure_risc_settings(
    size_t num_riscv_cores,
    size_t risc_id,
    tt::ARCH arch,
    bool& enable_handshake,
    bool& enable_context_switch,
    bool& enable_interrupts) {
    if (arch == tt::ARCH::WORMHOLE_B0) {
        // Wormhole: All RISC cores handle both sender and receiver channels
        enable_handshake = true;
        enable_context_switch = true;
        enable_interrupts = false;
    } else if (arch == tt::ARCH::BLACKHOLE) {
        if (num_riscv_cores == 1) {
            enable_handshake = true;
            enable_context_switch = tt::tt_metal::MetalContext::instance().rtoptions().get_enable_2_erisc_mode();
            enable_interrupts = false;
        } else {
            // Blackhole: Distribute sender/receiver across two RISC cores
            if (risc_id == 0) {
                // ERISC0: Handle sender channels only
                enable_handshake = true;
                enable_context_switch = true;
                enable_interrupts = false;
            } else if (risc_id == 1) {
                // ERISC1: Handle receiver channels only
                enable_handshake = false;
                enable_context_switch = false;
                enable_interrupts = false;
            } else {
                TT_THROW("Invalid RISC ID {} for BLACKHOLE architecture", risc_id);
            }
        }
    } else {
        TT_THROW("Unsupported architecture for RISC configuration: {}", enchantum::to_string(arch));
    }
}

size_t get_num_riscv_cores() {
    if (is_fabric_two_erisc_enabled()) {
        size_t nriscs = tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
        return nriscs;
    }
    return 1;
}

// Helper to determine if a RISC core should service sender channels
// On Blackhole with 2 ERISCs: ERISC0 services senders, ERISC1 does not
// On Wormhole or Blackhole with 1 ERISC: all RISCs service both
bool should_risc_service_sender_channels(size_t risc_id) {
    auto arch = tt::tt_metal::MetalContext::instance().hal().get_arch();
    size_t num_riscv_cores = get_num_riscv_cores();

    if (arch == tt::ARCH::BLACKHOLE && num_riscv_cores == 2) {
        return risc_id == 0;  // Only ERISC0 services senders
    }
    return true;  // Wormhole or single-ERISC mode: all service senders
}

// Helper to determine if a RISC core should service receiver channels
// On Blackhole with 2 ERISCs: ERISC1 services receivers, ERISC0 does not
// On Wormhole or Blackhole with 1 ERISC: all RISCs service both
bool should_risc_service_receiver_channels(size_t risc_id) {
    auto arch = tt::tt_metal::MetalContext::instance().hal().get_arch();
    size_t num_riscv_cores = get_num_riscv_cores();

    if (arch == tt::ARCH::BLACKHOLE && num_riscv_cores == 2) {
        return risc_id == 1;  // Only ERISC1 services receivers
    }
    return true;  // Wormhole or single-ERISC mode: all service receivers
}

}  // anonymous namespace

static std::pair<bool, bool> compute_edge_facing_flags(
    const ControlPlane& control_plane, const FabricNodeId& local_fabric_node_id, uint32_t my_eth_channel) {
    const auto& intermesh_chans = control_plane.get_intermesh_facing_eth_chans(local_fabric_node_id);
    const auto& intramesh_chans = control_plane.get_intramesh_facing_eth_chans(local_fabric_node_id);
    if (intermesh_chans.empty()) {
        return {false, false};
    }
    const bool is_intermesh =
        std::find(intermesh_chans.begin(), intermesh_chans.end(), my_eth_channel) != intermesh_chans.end();
    const bool is_intramesh =
        std::find(intramesh_chans.begin(), intramesh_chans.end(), my_eth_channel) != intramesh_chans.end();
    return {is_intermesh, is_intramesh};
}

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
        this->enable_interrupts_);
}

namespace {
bool requires_forced_assignment_to_noc1() {
    // When creating a kernel on erisc0 and 2 erisc mode is disabled, the physical processor is erisc1 while erisc0 is
    // running base firmware. As base firmware may occasionally use noc0 force fabric on "erisc0" to use noc1
    //
    // When 2 erisc mode is enabled on the runtime, erisc index == noc index is enforced in tt_metal.cpp
    //
    return tt::tt_metal::MetalContext::instance().hal().get_arch() == tt::ARCH::BLACKHOLE && get_num_riscv_cores() == 1;
}
}  // anonymous namespace

FabricEriscDatamoverConfig::FabricEriscDatamoverConfig(Topology topology) : topology(topology) {
    const bool is_2D_routing = is_2D_topology(topology);
    // Allocate L1 addresses for MAX sender channels (9) to support all router types
    // Even though most routers use fewer channels, we need addresses for all possible channels
    uint32_t num_sender_channels = is_2D_routing
                                       ? builder_config::num_max_sender_channels
                                       : builder_config::get_sender_channel_count(false);  // Use MAX (9) instead of 8
    uint32_t num_downstream_edms = builder_config::get_downstream_edm_count(is_2D_routing);
    // Global
    size_t next_l1_addr = tt::tt_metal::hal::get_erisc_l1_unreserved_base();

    // https://github.com/tenstorrent/tt-metal/issues/26354 to track fix for this hack where we always set aside the
    // memory for the telemetry buffer in Blackhole
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_enable_fabric_bw_telemetry() ||
        tt::tt_metal::MetalContext::instance().hal().get_arch() == tt::ARCH::BLACKHOLE) {
        // Avoid a bug on BH, always allocate the space for the telemetry buffer
        this->perf_telemetry_buffer_address = next_l1_addr;
        next_l1_addr += 32;
    }

    // Allocate code profiling buffer (conditionally enabled)
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    if (rtoptions.get_enable_fabric_code_profiling_rx_ch_fwd()) {
        // Buffer size: max timer types * 16 bytes per result
        constexpr size_t code_profiling_buffer_size =
            get_max_code_profiling_timer_types() * sizeof(CodeProfilingTimerResult);
        this->code_profiling_buffer_address = next_l1_addr;
        next_l1_addr += code_profiling_buffer_size;
    } else {
        this->code_profiling_buffer_address = 0;  // Not allocated
    }

    this->handshake_addr = next_l1_addr;
    next_l1_addr += eth_channel_sync_size;

    // issue: https://github.com/tenstorrent/tt-metal/issues/29073. TODO: Re-enable after hang is resolved.
    // Ethernet txq IDs on WH are 0,1 and on BH are 0,1,2.
    if (is_fabric_two_erisc_enabled()) {
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
        this->router_buffer_clear_size_words = num_words_consumed_per_counter;

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

    // ----------- Local Tensix Relay Connection (UDM mode only)
    // Dedicated connection buffer index for the local tensix relay interface
    this->tensix_relay_connection_buffer_index_id = buffer_address;
    buffer_address += field_size;

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

FabricEriscDatamoverConfig::FabricEriscDatamoverConfig(
    std::size_t channel_buffer_size_bytes,
    Topology topology,
    FabricEriscDatamoverOptions options,
    const std::array<std::size_t, builder_config::MAX_NUM_VCS>& sender_channels_per_vc,
    const std::array<std::size_t, builder_config::MAX_NUM_VCS>& receiver_channels_per_vc) :
    FabricEriscDatamoverConfig(topology) {
    this->channel_buffer_size_bytes = channel_buffer_size_bytes;

    // Set channel counts from parameters
    for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        this->num_used_sender_channels_per_vc[vc] = sender_channels_per_vc[vc];
        this->num_used_receiver_channels_per_vc[vc] = receiver_channels_per_vc[vc];
        log_debug(
            tt::LogFabric,
            "  VC{}: {} senders, {} receivers",
            vc,
            this->num_used_sender_channels_per_vc[vc],
            this->num_used_receiver_channels_per_vc[vc]);
    }

    // Set total counts for backward compatibility
    this->num_used_sender_channels = std::accumulate(
        this->num_used_sender_channels_per_vc.begin(), this->num_used_sender_channels_per_vc.end(), size_t{0});
    this->num_used_receiver_channels = std::accumulate(
        this->num_used_receiver_channels_per_vc.begin(), this->num_used_receiver_channels_per_vc.end(), size_t{0});

    // num_fwd_paths = total sender channels - 1 (worker channel)
    this->num_fwd_paths = this->num_used_sender_channels - 1;

    for (uint32_t i = 0; i < this->num_used_sender_channels; i++) {
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

    // Create a default recipe with a single static pool for backward compatibility
    // All channels map to pool 0 (the single static pool)
    auto recipe = tt::tt_fabric::FabricRouterRecipe::create_default_single_static_pool_recipe(
        this->num_used_sender_channels, this->num_used_receiver_channels);
    auto remote_channels_recipe = tt::tt_fabric::FabricRouterRecipe::create_default_single_static_pool_recipe(
        0, this->num_used_receiver_channels);

    // Create the single static pool allocator with per-VC channel distribution
    auto static_allocator = std::make_shared<tt::tt_fabric::FabricStaticSizedChannelsAllocator>(
        topology,
        options,
        this->num_used_sender_channels_per_vc,
        this->num_used_receiver_channels_per_vc,
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
    for (uint32_t i = 0; i < builder_config::num_max_receiver_channels; i++) {
        this->receiver_channel_forwarding_noc_ids[i] = FabricEriscDatamoverConfig::DEFAULT_RECEIVER_FORWARDING_NOC;
        this->receiver_channel_forwarding_data_cmd_buf_ids[i] = FabricEriscDatamoverConfig::WR_REG_CMD_BUF;
        this->receiver_channel_forwarding_sync_cmd_buf_ids[i] = FabricEriscDatamoverConfig::RD_CMD_BUF;
        this->receiver_channel_local_write_noc_ids[i] = FabricEriscDatamoverConfig::DEFAULT_RECEIVER_LOCAL_WRITE_NOC;
        this->receiver_channel_local_write_cmd_buf_ids[i] = FabricEriscDatamoverConfig::WR_CMD_BUF;

        if (requires_forced_assignment_to_noc1()) {
            this->receiver_channel_forwarding_noc_ids[i] =
                FabricEriscDatamoverConfig::BLACKHOLE_SINGLE_ERISC_MODE_RECEIVER_FORWARDING_NOC;
            this->receiver_channel_local_write_noc_ids[i] =
                FabricEriscDatamoverConfig::BLACKHOLE_SINGLE_ERISC_MODE_RECEIVER_LOCAL_WRITE_NOC;
            this->receiver_channel_forwarding_data_cmd_buf_ids[i] = FabricEriscDatamoverConfig::WR_CMD_BUF;
        }
    }
    for (uint32_t i = 0; i < builder_config::num_max_sender_channels; i++) {
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
        static_cast<uint32_t>(connection.edm_buffer_base_addr),
        static_cast<uint32_t>(connection.num_buffers_per_channel),
        static_cast<uint32_t>(connection.edm_l1_sem_addr),
        static_cast<uint32_t>(connection.edm_connection_handshake_addr),
        static_cast<uint32_t>(connection.edm_worker_location_info_addr),
        static_cast<uint32_t>(connection.buffer_size_bytes),
        static_cast<uint32_t>(connection.buffer_index_semaphore_id),
        static_cast<uint32_t>(sender_worker_flow_control_semaphore_id),
        static_cast<uint32_t>(sender_worker_terminate_semaphore_id),
        static_cast<uint32_t>(sender_worker_buffer_index_semaphore_id)};
    args_out.reserve(args_out.size() + (values.size() / sizeof(size_t)));
    std::ranges::copy(values, std::back_inserter(args_out));
}

void append_worker_to_fabric_edm_sender_rt_args(
    chan_id_t eth_channel,
    size_t sender_worker_terminate_semaphore_id,
    size_t sender_worker_buffer_index_semaphore_id,
    std::vector<uint32_t>& args_out) {
    const std::vector<uint32_t> values = {
        eth_channel,
        static_cast<uint32_t>(sender_worker_terminate_semaphore_id),
        static_cast<uint32_t>(sender_worker_buffer_index_semaphore_id)};
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
        eth_channel,
        static_cast<uint32_t>(sender_worker_terminate_semaphore_id),
        static_cast<uint32_t>(sender_worker_buffer_index_semaphore_id)};
    args_out.reserve(args_out.size() + (values.size() / sizeof(size_t)));
    std::ranges::copy(values, std::back_inserter(args_out));
}

size_t log_worker_to_fabric_edm_sender_rt_args(
    const std::vector<uint32_t>& args [[maybe_unused]], size_t starting_arg_idx) {
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
        tt::LogFabric,
        "arg[{}]: sender_worker_flow_control_semaphore_id {}",
        starting_arg_idx,
        args[starting_arg_idx++]);
    log_trace(
        tt::LogFabric,
        "arg[{}]: sender_worker_buffer_index_semaphore_id {}",
        starting_arg_idx,
        args[starting_arg_idx++]);
    return starting_arg_idx + 10;
}

FabricEriscDatamoverBuilder::FabricEriscDatamoverBuilder(
    const CoreCoord& my_eth_core_logical,
    size_t my_noc_x,
    size_t my_noc_y,
    const FabricNodeId& local_fabric_node_id,
    const FabricNodeId& peer_fabric_node_id,

    const std::array<std::optional<size_t>, builder_config::max_downstream_edms>&
        receiver_channels_downstream_flow_control_semaphore_id,
    const std::array<std::optional<size_t>, builder_config::max_downstream_edms>&
        receiver_channels_downstream_teardown_semaphore_id,
    const std::array<size_t, builder_config::num_max_sender_channels>& sender_channels_flow_control_semaphore_id,
    const std::array<size_t, builder_config::num_max_sender_channels>& sender_channels_connection_semaphore_id,
    const std::array<size_t, builder_config::num_max_sender_channels>& sender_channels_buffer_index_semaphore_id,

    const FabricEriscDatamoverConfig& config,
    eth_chan_directions direction,
    std::vector<bool>&& sender_channel_injection_flags,
    bool build_in_worker_connection_mode,
    bool has_tensix_extension,
    std::optional<std::array<std::size_t, builder_config::MAX_NUM_VCS>> actual_sender_channels_per_vc,
    std::optional<std::array<std::size_t, builder_config::MAX_NUM_VCS>> actual_receiver_channels_per_vc) :
    FabricDatamoverBuilderBase(my_noc_x, my_noc_y, direction),
    my_eth_core_logical(my_eth_core_logical),
    my_eth_channel(my_eth_core_logical.y),
    config(config),
    local_fabric_node_id(local_fabric_node_id),
    peer_fabric_node_id(peer_fabric_node_id),
    is_inter_mesh(local_fabric_node_id.mesh_id != peer_fabric_node_id.mesh_id),
    handshake_address(tt::round_up(
        tt::tt_metal::hal::get_erisc_l1_unreserved_base(), FabricEriscDatamoverConfig::eth_channel_sync_size)),
    channel_buffer_size(config.channel_buffer_size_bytes),
    local_sender_channels_connection_info_addr(config.sender_channels_worker_conn_info_base_address),
    termination_signal_ptr(config.termination_signal_address),
    edm_local_sync_ptr(config.edm_local_sync_address),
    edm_local_tensix_sync_ptr(config.edm_local_tensix_sync_address),
    edm_status_ptr(config.edm_status_address),

    // this is the receiver channel's local sem for flow controlling with downstream fabric sender
    receiver_channels_downstream_flow_control_semaphore_id(receiver_channels_downstream_flow_control_semaphore_id),
    receiver_channels_downstream_teardown_semaphore_id(receiver_channels_downstream_teardown_semaphore_id),
    sender_channels_flow_control_semaphore_id(sender_channels_flow_control_semaphore_id),
    sender_channels_connection_semaphore_id(sender_channels_connection_semaphore_id),
    sender_channels_buffer_index_semaphore_id(sender_channels_buffer_index_semaphore_id),
    sender_channel_is_traffic_injection_channel_array(std::move(sender_channel_injection_flags)),
    actual_sender_channels_per_vc_(actual_sender_channels_per_vc),
    actual_receiver_channels_per_vc_(actual_receiver_channels_per_vc),
    build_in_worker_connection_mode(build_in_worker_connection_mode),
    has_tensix_extension(has_tensix_extension),
    // First level ack is enabled to support bubble flow control
    enable_first_level_ack(
        config.topology == tt::tt_fabric::Topology::Ring || config.topology == tt::tt_fabric::Topology::Torus) {
    // NOTE: actual_sender_channels_per_vc and actual_receiver_channels_per_vc are:
    // 1. Stored as members for later use in compile-time args
    // 2. Used for connection validation
    // We intentionally DON'T override config values because:
    // 1. config.multi_pool_allocator was created with MAX channel counts
    // 2. Changing config counts would cause emit_ct_args to emit wrong number of args
    // The config stays at MAX, but we pass actual counts to device via compile-time args
    // Validate injection flags vector size matches the number of sender channels
    TT_FATAL(
        sender_channel_is_traffic_injection_channel_array.size() == config.num_used_sender_channels,
        "Internal error: injection_flags vector size {} does not match num_used_sender_channels {}",
        sender_channel_is_traffic_injection_channel_array.size(),
        config.num_used_sender_channels);

    // Initialize per-RISC channel servicing flags
    const auto& sender_counts = actual_sender_channels_per_vc.value_or(this->config.num_used_sender_channels_per_vc);
    const auto& receiver_counts =
        actual_receiver_channels_per_vc.value_or(this->config.num_used_receiver_channels_per_vc);

    bool is_mux_mode = has_tensix_extension && (tt::tt_metal::MetalContext::instance().get_fabric_tensix_config() ==
                                                tt::tt_fabric::FabricTensixConfig::MUX);

    size_t num_riscv_cores = this->config.risc_configs.size();
    for (size_t risc_id = 0; risc_id < num_riscv_cores; ++risc_id) {
        this->is_sender_channel_serviced_[risc_id].fill(false);
        this->is_receiver_channel_serviced_[risc_id].fill(false);

        if (is_mux_mode) {
            // MUX mode: Only worker channel serviced for senders
            uint32_t worker_channel = get_worker_connected_sender_channel();
            this->is_sender_channel_serviced_[risc_id][worker_channel] = true;

            // All receiver channels serviced in MUX mode
            size_t receiver_offset = 0;
            for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
                size_t num_channels = receiver_counts[vc];
                for (size_t i = 0; i < num_channels; ++i) {
                    this->is_receiver_channel_serviced_[risc_id][receiver_offset + i] = true;
                }
                receiver_offset += num_channels;
            }
        } else {
            // Normal mode: Enable channels based on per-VC counts AND this RISC's responsibility
            bool services_senders = should_risc_service_sender_channels(risc_id);
            bool services_receivers = should_risc_service_receiver_channels(risc_id);

            if (services_senders) {
                size_t sender_offset = 0;
                for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
                    size_t num_channels = sender_counts[vc];
                    for (size_t i = 0; i < num_channels; ++i) {
                        this->is_sender_channel_serviced_[risc_id][sender_offset + i] = true;
                    }
                    sender_offset += num_channels;
                }
            }

            if (services_receivers) {
                size_t receiver_offset = 0;
                for (size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
                    size_t num_channels = receiver_counts[vc];
                    for (size_t i = 0; i < num_channels; ++i) {
                        this->is_receiver_channel_serviced_[risc_id][receiver_offset + i] = true;
                    }
                    receiver_offset += num_channels;
                }
            }
        }
    }

    std::fill(
        sender_channel_connection_liveness_check_disable_array.begin(),
        sender_channel_connection_liveness_check_disable_array.end(),
        false);

    TT_FATAL(
        config.channel_allocator.get() != nullptr,
        "Channel allocator is not set. Failed to build TT-Fabric router. Internal error.");
    auto* static_allocator =
        dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(config.channel_allocator.get());
    TT_FATAL(
        static_allocator != nullptr,
        "Channel allocator must be a FabricStaticSizedChannelsAllocator. Failed to build TT-Fabric router. Internal "
        "error.");
    this->receiver_channel_to_downstream_adapter =
        std::make_shared<tt::tt_fabric::StaticSizedChannelConnectionWriterAdapter>(
            *static_allocator, config.topology, direction);
    // worker is always index 0.
    // rest of the downstream buffer index addresses will be populated when building connections to downstream edm
    // channels.
    downstream_vcs_sender_channel_buffer_index_semaphore_id[0] = sender_channels_buffer_index_semaphore_id[0];

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

    configure_telemetry_settings();
}

void FabricEriscDatamoverBuilder::configure_telemetry_settings() {
    auto& telemetry_rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const auto& telemetry_settings = telemetry_rtoptions.get_fabric_telemetry_settings();
    const bool telemetry_globally_enabled = telemetry_rtoptions.get_enable_fabric_telemetry() &&
                                            telemetry_settings.enabled && telemetry_settings.stats_mask != 0;
    const auto local_physical_chip_id =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_physical_chip_id_from_fabric_node_id(
            this->local_fabric_node_id);
    for (uint32_t risc_id = 0; risc_id < this->config.num_riscv_cores; risc_id++) {
        bool telemetry_enabled_on_erisc =
            telemetry_globally_enabled &&
            telemetry_settings.is_telemetry_enabled(
                static_cast<uint32_t>(local_physical_chip_id), static_cast<uint32_t>(this->my_eth_channel), risc_id);
        this->config.risc_configs[risc_id].set_telemetry_enabled(telemetry_enabled_on_erisc);
        this->config.risc_configs[risc_id].set_telemetry_stats_mask(telemetry_settings.stats_mask);
    }
}

void FabricEriscDatamoverBuilder::get_telemetry_compile_time_args(
    uint32_t risc_id, std::vector<uint32_t>& ct_args) const {
    const auto& risc_config = config.risc_configs[risc_id];
    const bool telemetry_enabled = risc_config.telemetry_enabled();

    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    ct_args.push_back(static_cast<uint32_t>(telemetry_enabled));

    // Add telemetry statistic mask (per ERISC)
    const uint8_t stats_mask = telemetry_enabled ? risc_config.telemetry_stats_mask() : 0;
    ct_args.push_back(static_cast<uint32_t>(stats_mask));

    uint32_t bw_telemetry_mode = static_cast<uint32_t>(rtoptions.get_enable_fabric_bw_telemetry() ? 1 : 0);
    ct_args.push_back(bw_telemetry_mode);

    // Add telemetry buffer address (16B aligned)
    ct_args.push_back(static_cast<uint32_t>(config.perf_telemetry_buffer_address));

    // Add code profiling arguments (conditionally enabled)
    if (rtoptions.get_enable_fabric_code_profiling_rx_ch_fwd()) {
        // Enable RECEIVER_CHANNEL_FORWARD timer (bit 0)
        uint32_t code_profiling_enabled_timers =
            static_cast<uint32_t>(CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD);
        ct_args.push_back(code_profiling_enabled_timers);

        // Add code profiling buffer address (16B aligned)
        ct_args.push_back(static_cast<uint32_t>(config.code_profiling_buffer_address));
    } else {
        // Code profiling disabled - add zeros
        ct_args.push_back(0);  // No timers enabled
        ct_args.push_back(0);  // No buffer address
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

    auto dispatch_core_type = get_core_type_from_config(
        tt::tt_metal::MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_config());
    uint32_t my_eth_channel_ = [&]() -> uint32_t {
        if (dispatch_core_type == CoreType::WORKER) {
            return this->my_eth_channel;
        }
        if (dispatch_core_type == CoreType::ETH) {
            return tt::tt_fabric::USE_DYNAMIC_CREDIT_ADDR;
        }
        TT_THROW("Fabric Mux does not support core type {}", enchantum::to_string(dispatch_core_type));
    }();

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto local_physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(this->local_fabric_node_id);
    const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(local_physical_chip_id);

    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();

    auto sender_channel_to_check = get_worker_connected_sender_channel();

    const auto remote_routing_direction =
        control_plane.get_forwarding_direction(this->peer_fabric_node_id, this->local_fabric_node_id);

    uint32_t remote_worker_sender_channel = 0;
    bool skip_src_ch_id_update = false;
    // when there is no remote router paired with current router, don't care about skip_src_ch_id_update and set it to
    // false
    if (remote_routing_direction.has_value()) {
        skip_src_ch_id_update = this->has_tensix_extension;
        remote_worker_sender_channel = get_worker_connected_sender_channel();
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
    auto* channel_allocator = config.channel_allocator.get();
    auto* const static_channel_allocator =
        dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(channel_allocator);
    TT_FATAL(static_channel_allocator != nullptr, "Channel allocator must be a FabricStaticSizedChannelsAllocator.");
    size_t receiver_channel_num_buffers = static_channel_allocator->get_receiver_channel_number_of_slots(0);
    TT_FATAL(
        static_channel_allocator->get_sender_channel_number_of_slots(sender_channel_to_check) > 0,
        "Sender channel on direction {} num buffers must be greater than 0",
        sender_channel_to_check);
    TT_FATAL(receiver_channel_num_buffers > 0, "Receiver channel num buffers must be greater than 0");

    const auto& stream_ids = StreamRegAssignments::get_all_stream_ids();
    constexpr bool enable_risc_cpu_data_cache = false;
    auto ct_args = std::vector<uint32_t>(stream_ids.begin(), stream_ids.end());
    ct_args.push_back(0xFFEE0001);

    // Maximum channel counts from builder_config
    ct_args.push_back(builder_config::num_max_sender_channels);
    ct_args.push_back(builder_config::num_max_receiver_channels);

    // add the downstream tensix connection arg here, num_downstream_tensix_connections
    ct_args.push_back(this->num_downstream_tensix_connections);

    // Compute edge-facing flags for this ethernet core/channel
    const auto [is_intermesh_router_on_edge, is_intramesh_router_on_edge] =
        compute_edge_facing_flags(control_plane, this->local_fabric_node_id, this->my_eth_channel);

    // Get actual downstream EDM counts from the adapter (actual connections made)
    uint32_t num_vc0_downstream_edms = this->receiver_channel_to_downstream_adapter->get_downstream_edm_count_for_vc(0);

    bool needs_vc1 = config.num_used_receiver_channels_per_vc[1] > 0;
    // Get actual VC1 downstream EDM count from adapter (only relevant for multi-mesh 2D routing)
    uint32_t num_vc1_downstream_edms =
        needs_vc1 ? this->receiver_channel_to_downstream_adapter->get_downstream_edm_count_for_vc(1) : 0;
    // unsure which one we should prefer at the moment
    // bool z_routers_enabled = fabric_context.get_builder_context().get_intermesh_vc_config().router_type ==
    // IntermeshRouterType::Z_INTERMESH;
    bool z_router_enabled = fabric_context.has_z_router_on_device(local_fabric_node_id);

    log_debug(
        LogFabric,
        "z_router_enabled: {}, needs_vc1: {}, num_vc0_downstream_edms: {}, num_vc1_downstream_edms: {}",
        z_router_enabled,
        needs_vc1,
        num_vc0_downstream_edms,
        num_vc1_downstream_edms);
    // Calculate array sizes for downstream EDMs based on masks
    // Size = position of highest set bit + 1 (e.g., mask 0x5 = size 3, mask 0x3 = size 2)
    auto calc_array_size_from_mask = [](uint32_t mask) -> uint32_t {
        if (mask == 0) {
            return 0;
        }
        // Find position of most significant bit + 1
        return 32 - __builtin_clz(mask);
    };

    uint32_t vc0_downstream_edm_size =
        calc_array_size_from_mask(this->receiver_channel_to_downstream_adapter->get_downstream_edm_mask_for_vc(0));
    uint32_t vc1_downstream_edm_size =
        needs_vc1
            ? calc_array_size_from_mask(this->receiver_channel_to_downstream_adapter->get_downstream_edm_mask_for_vc(1))
            : 0;

    // Ensure minimum size of 1 to avoid zero-sized arrays (causes undefined behavior when accessed)
    // Even if mask is 0 (no downstream connections), we need at least 1 element for the array template
    if (vc0_downstream_edm_size == 0) {
        vc0_downstream_edm_size = 1;
    }
    if (needs_vc1 && vc1_downstream_edm_size == 0) {
        vc1_downstream_edm_size = 1;
    }

    auto actual_sender_channels_vc0 = actual_sender_channels_per_vc_.has_value()
                                          ? actual_sender_channels_per_vc_.value()[0]
                                          : config.num_used_sender_channels_per_vc[0];
    auto actual_sender_channels_vc1 = actual_sender_channels_per_vc_.has_value()
                                          ? actual_sender_channels_per_vc_.value()[1]
                                          : config.num_used_sender_channels_per_vc[1];
    const std::vector<uint32_t> main_args_part1 = {
        static_cast<uint32_t>(num_sender_channels),
        static_cast<uint32_t>(num_receiver_channels),
        static_cast<uint32_t>(config.num_fwd_paths),
        num_vc0_downstream_edms,
        num_vc1_downstream_edms,
        static_cast<uint32_t>(this->wait_for_host_signal ? 1 : 0),

        static_cast<uint32_t>(this->firmware_context_switch_interval),
        this->fuse_receiver_flush_and_completion_ptr,
        fabric_context.need_deadlock_avoidance_support(this->direction_),
        this->is_inter_mesh,  // Whether this router connects different meshes (determines VC crossover)
        is_handshake_master,
        static_cast<uint32_t>(this->handshake_address),
        static_cast<uint32_t>(this->channel_buffer_size),
        this->has_tensix_extension,
        this->enable_first_level_ack,  // VC0 first level ack (for Ring/Torus)
        false,                         // VC1 first level ack (always false - VC1 doesn't use bubble flow control)
        enable_risc_cpu_data_cache,
        z_router_enabled,
        vc0_downstream_edm_size,  // VC0_DOWNSTREAM_EDM_SIZE: array size for VC0 downstream EDMs
        vc1_downstream_edm_size,  // VC1_DOWNSTREAM_EDM_SIZE: array size for VC1 downstream EDMs
        // Actual sender channel counts per VC for this router (may differ from MAX)
        static_cast<uint32_t>(actual_sender_channels_vc0),   // ACTUAL_VC0_SENDER_CHANNELS
        static_cast<uint32_t>(actual_sender_channels_vc1)};  // ACTUAL_VC1_SENDER_CHANNELS

    const std::vector<uint32_t> main_args_part2 = {
        static_cast<uint32_t>(config.sender_channels_worker_conn_info_base_address[0]),
        static_cast<uint32_t>(config.sender_channels_worker_conn_info_base_address[1]),
        static_cast<uint32_t>(config.sender_channels_worker_conn_info_base_address[2]),
        static_cast<uint32_t>(config.sender_channels_worker_conn_info_base_address[3]),
        static_cast<uint32_t>(config.sender_channels_worker_conn_info_base_address[4]),
        static_cast<uint32_t>(config.sender_channels_worker_conn_info_base_address[5]),
        static_cast<uint32_t>(config.sender_channels_worker_conn_info_base_address[6]),
        static_cast<uint32_t>(config.sender_channels_worker_conn_info_base_address[7]),
        static_cast<uint32_t>(config.sender_channels_worker_conn_info_base_address[8]),  // 9th channel for Z routers

        static_cast<uint32_t>(this->termination_signal_ptr),
        static_cast<uint32_t>(this->edm_local_sync_ptr),
        static_cast<uint32_t>(this->edm_local_tensix_sync_ptr),
        static_cast<uint32_t>(this->edm_status_ptr),

        static_cast<uint32_t>(config.notify_worker_of_read_counter_update_src_address),
        0x7a9b3c4d,  // DELETEME Issue #33360 special tag marker to catch incorrect ct args

        this->is_sender_channel_serviced_[risc_id][0],
        this->is_sender_channel_serviced_[risc_id][1],
        this->is_sender_channel_serviced_[risc_id][2],
        this->is_sender_channel_serviced_[risc_id][3],
        this->is_sender_channel_serviced_[risc_id][4],
        this->is_sender_channel_serviced_[risc_id][5],
        this->is_sender_channel_serviced_[risc_id][6],
        this->is_sender_channel_serviced_[risc_id][7],
        this->is_sender_channel_serviced_[risc_id][8],  // 9th sender channel for Z routers
        this->is_receiver_channel_serviced_[risc_id][0],
        this->is_receiver_channel_serviced_[risc_id][1],
        config.risc_configs[risc_id].enable_handshake(),
        config.risc_configs[risc_id].enable_context_switch(),
        config.risc_configs[risc_id].enable_interrupts(),
        static_cast<uint32_t>(config.sender_txq_id),
        static_cast<uint32_t>(config.receiver_txq_id),
        static_cast<uint32_t>(config.risc_configs[risc_id].iterations_between_ctx_switch_and_teardown_checks()),
        fabric_context.is_2D_routing_enabled(),
        this->direction_,
        soc_desc.get_num_eth_channels(),

        eth_txq_spin_wait_send_next_data,
        eth_txq_spin_wait_receiver_send_completion_ack,
        default_num_eth_txq_data_packet_accept_ahead,

        default_handshake_context_switch_timeout,
        static_cast<uint32_t>(
            this->firmware_context_switch_type == FabricEriscDatamoverContextSwitchType::WAIT_FOR_IDLE),
        my_eth_channel_,

        risc_id,
        static_cast<uint32_t>(this->get_configured_risc_count()),

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
        ct_args.push_back(remote_worker_sender_channel);
    }

    // Add UDM mode flag and relay buffer count
    ct_args.push_back(this->udm_mode ? 1 : 0);
    if (this->udm_mode) {
        ct_args.push_back(this->local_tensix_relay_num_buffers);
    }

    // special tag
    ct_args.push_back(0xabcd9876);

    // Emit pool data via multi-pool coordinator (steps 1-4 of schema: special tag, num_pools, pool_types, individual
    // pool CT args)
    config.multi_pool_allocator->emit_ct_args(
        ct_args, actual_sender_channels_vc0, actual_sender_channels_vc1, num_receiver_channels);

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
    remote_multi_pool_allocator.emit_ct_args(ct_args, 0, 0, num_receiver_channels);

    config.remote_channel_to_pool_mapping->emit_ct_args(ct_args);

    ct_args.push_back(0xabaddad7);
    receiver_channel_to_downstream_adapter->emit_ct_args(ct_args, config.num_fwd_paths);
    ct_args.push_back(0xabaddad9);

    // Add second part of main arguments to ct_args
    ct_args.insert(ct_args.end(), main_args_part2.begin(), main_args_part2.end());

    for (size_t i = 0; i < num_sender_channels; i++) {
        ct_args.push_back(this->sender_channel_connection_liveness_check_disable_array[i]);
    }

    for (size_t i = 0; i < num_sender_channels; i++) {
        ct_args.push_back(this->sender_channel_is_traffic_injection_channel_array.at(i));
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

    get_telemetry_compile_time_args(risc_id, ct_args);

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
    std::vector<uint32_t> rt_args = {
        static_cast<uint32_t>(this->sender_channels_connection_semaphore_id[0]),
        static_cast<uint32_t>(this->sender_channels_connection_semaphore_id[1]),
        static_cast<uint32_t>(this->sender_channels_connection_semaphore_id[2]),
        static_cast<uint32_t>(this->sender_channels_connection_semaphore_id[3]),
        static_cast<uint32_t>(this->sender_channels_connection_semaphore_id[4]),
        static_cast<uint32_t>(this->sender_channels_connection_semaphore_id[5]),
        static_cast<uint32_t>(this->sender_channels_connection_semaphore_id[6]),
        static_cast<uint32_t>(this->sender_channels_connection_semaphore_id[7]),
        static_cast<uint32_t>(this->sender_channels_connection_semaphore_id[8]),  // 9th channel for Z routers

        static_cast<uint32_t>(this->downstream_vcs_sender_channel_buffer_index_semaphore_id[0]),
        static_cast<uint32_t>(this->downstream_vcs_sender_channel_buffer_index_semaphore_id[1]),
        static_cast<uint32_t>(this->downstream_vcs_sender_channel_buffer_index_semaphore_id[2]),
        static_cast<uint32_t>(this->downstream_vcs_sender_channel_buffer_index_semaphore_id[3]),
        static_cast<uint32_t>(this->downstream_vcs_sender_channel_buffer_index_semaphore_id[4]),
        static_cast<uint32_t>(this->downstream_vcs_sender_channel_buffer_index_semaphore_id[5]),
        static_cast<uint32_t>(this->downstream_vcs_sender_channel_buffer_index_semaphore_id[6]),
        static_cast<uint32_t>(this->downstream_vcs_sender_channel_buffer_index_semaphore_id[7]),
        static_cast<uint32_t>(
            this->downstream_vcs_sender_channel_buffer_index_semaphore_id[8]),  // 9th channel for Z routers
    };

    // Pack VC0 runtime args
    receiver_channel_to_downstream_adapter->pack_inbound_channel_rt_args(0, rt_args);

    // Pack VC1 runtime args if VC1 is configured
    // Both inter-mesh and intra-mesh routers have VC1 in multi-mesh topologies
    bool needs_vc1 = config.num_used_receiver_channels_per_vc[1] > 0;
    if (needs_vc1) {
        receiver_channel_to_downstream_adapter->pack_inbound_channel_rt_args(1, rt_args);
    }

    // Pack runtime args - device side reads fixed MAX_NUM_SENDER_CHANNELS values
    // Only the first NUM_DOWNSTREAM_CHANNELS values are used based on topology
    auto args_pt2 = std::vector<uint32_t>{};

    // Pack downstream teardown semaphores (always send MAX_NUM_SENDER_CHANNELS for compatibility)
    for (uint32_t i = 0; i < builder_config::num_max_sender_channels; i++) {
        args_pt2.push_back(this->receiver_channels_downstream_teardown_semaphore_id[i].value_or(-1));
    }

    rt_args.reserve(rt_args.size() + args_pt2.size());
    std::ranges::copy(args_pt2, std::back_inserter(rt_args));

    // Pack relay connection args at the end (UDM mode only)
    receiver_channel_to_downstream_adapter->pack_adaptor_to_relay_rt_args(rt_args);

    return rt_args;
}

FabricEriscDatamoverBuilder FabricEriscDatamoverBuilder::build(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    const CoreCoord& ethernet_core,
    ChipId local_physical_chip_id,
    ChipId peer_physical_chip_id,
    const FabricEriscDatamoverConfig& config,
    std::vector<bool>&& sender_channel_injection_flags,
    bool build_in_worker_connection_mode,
    eth_chan_directions direction,
    bool has_tensix_extension,
    std::optional<std::array<std::size_t, builder_config::MAX_NUM_VCS>> actual_sender_channels_per_vc,
    std::optional<std::array<std::size_t, builder_config::MAX_NUM_VCS>> actual_receiver_channels_per_vc) {
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
        std::move(sender_channel_injection_flags),
        build_in_worker_connection_mode,
        direction,
        has_tensix_extension,
        actual_sender_channels_per_vc,
        actual_receiver_channels_per_vc);
}

FabricEriscDatamoverBuilder FabricEriscDatamoverBuilder::build(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& /*program*/,
    const CoreCoord& ethernet_core,
    const FabricNodeId& local_fabric_node_id,
    const FabricNodeId& peer_fabric_node_id,
    const FabricEriscDatamoverConfig& config,
    std::vector<bool>&& sender_channel_injection_flags,
    bool build_in_worker_connection_mode,
    eth_chan_directions direction,
    bool has_tensix_extension,
    std::optional<std::array<std::size_t, builder_config::MAX_NUM_VCS>> actual_sender_channels_per_vc,
    std::optional<std::array<std::size_t, builder_config::MAX_NUM_VCS>> actual_receiver_channels_per_vc) {
    std::array<size_t, builder_config::num_max_sender_channels> sender_channels_buffer_index_semaphore_id{};
    std::array<size_t, builder_config::num_max_sender_channels> sender_channels_flow_control_semaphore_id{};
    std::array<size_t, builder_config::num_max_sender_channels> sender_channels_connection_semaphore_id{};
    std::array<std::optional<size_t>, builder_config::max_downstream_edms>
        receiver_channels_downstream_flow_control_semaphore_id;
    std::array<std::optional<size_t>, builder_config::max_downstream_edms>
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
        for (uint32_t i = 0; i < builder_config::num_max_receiver_channels; i++) {
            receiver_channels_downstream_flow_control_semaphore_id[i] = 0;
            receiver_channels_downstream_teardown_semaphore_id[i] = 0;
        }
        // Sender channel 0 uses addresses instead of ids in persistent mode
        sender_channels_buffer_index_semaphore_id[0] = config.sender_channels_buffer_index_semaphore_address[0];
        sender_channels_flow_control_semaphore_id[0] = config.sender_channels_local_flow_control_semaphore_address[0];
        sender_channels_connection_semaphore_id[0] = config.sender_channels_connection_semaphore_address[0];
        for (uint32_t i = 1; i < builder_config::num_max_sender_channels; i++) {
            sender_channels_flow_control_semaphore_id[i] = 0;
            sender_channels_connection_semaphore_id[i] = 0;
            sender_channels_buffer_index_semaphore_id[i] = 0;
        }
    } else {
        const bool is_2D_routing =
            tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context().is_2D_routing_enabled();
        uint32_t num_vc0_downstream_edms = builder_config::get_vc0_downstream_edm_count(is_2D_routing);

        // Setup VC0 downstream edm semaphore settings.
        // 1D has 1 downstream edm. 2D has 3 downstream EDMs (excluding router's own direction)
        // 2D uses the reserved addresses in L1 from FabricEriscDatamoverConfig
        for (uint32_t i = 0; i < num_vc0_downstream_edms; i++) {
            receiver_channels_downstream_flow_control_semaphore_id[i] =
                config.receiver_channels_downstream_flow_control_semaphore_address[i];
            receiver_channels_downstream_teardown_semaphore_id[i] =
                config.receiver_channels_downstream_teardown_semaphore_address[i];
        }

        // Setup VC1 downstream edm semaphore settings (only for 2D routing)
        if (is_2D_routing) {
            uint32_t num_vc1_downstream_edms = builder_config::get_vc1_downstream_edm_count(is_2D_routing);
            for (uint32_t i = 0; i < num_vc1_downstream_edms; i++) {
                receiver_channels_downstream_flow_control_semaphore_id[num_vc0_downstream_edms + i] =
                    config.receiver_channels_downstream_flow_control_semaphore_address[num_vc0_downstream_edms + i];
                receiver_channels_downstream_teardown_semaphore_id[num_vc0_downstream_edms + i] =
                    config.receiver_channels_downstream_teardown_semaphore_address[num_vc0_downstream_edms + i];
            }
        }

        // Initialize ALL max sender channels (up to 9 for Z routers) with their addresses
        // This ensures that if is_sender_channel_serviced[i] is true for any channel,
        // the address is valid (not 0)
        for (uint32_t i = 0; i < builder_config::num_max_sender_channels; i++) {
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
        std::move(sender_channel_injection_flags),
        build_in_worker_connection_mode,
        has_tensix_extension,
        actual_sender_channels_per_vc,
        actual_receiver_channels_per_vc);
}

SenderWorkerAdapterSpec FabricEriscDatamoverBuilder::build_connection_to_fabric_channel(
    uint32_t vc, uint32_t absolute_channel_id, uint32_t vc_relative_channel_id) const {
    TT_FATAL(
        vc < builder_config::MAX_NUM_VCS,
        "VC index {} exceeds maximum supported VCs ({}). Got vc: {}",
        vc,
        builder_config::MAX_NUM_VCS,
        vc);

    // Both absolute and VC-relative channel IDs are passed from upstream caller
    // - absolute_channel_id: used for flat arrays (sender_channel_connection_liveness_check_disable_array)
    // - vc_relative_channel_id: used for VC-aware allocator calls
    // Both are already validated by ComputeMeshRouterBuilder::establish_connections_to_router

    auto* channel_allocator = config.channel_allocator.get();
    auto* const static_channel_allocator =
        dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(channel_allocator);
    TT_FATAL(static_channel_allocator != nullptr, "Channel allocator must be a FabricStaticSizedChannelsAllocator.");

    // Use VC-aware allocator getters with VC-relative channel ID
    size_t sender_channels_num_buffer =
        static_channel_allocator->get_sender_channel_number_of_slots(vc, vc_relative_channel_id);

    TT_FATAL(sender_channels_num_buffer != 0, "sender_channels_num_buffer should not be 0!");

    // Use absolute index for flat arrays in FabricEriscDatamoverBuilder
    this->sender_channel_connection_liveness_check_disable_array[absolute_channel_id] = true;
    return SenderWorkerAdapterSpec{
        this->noc_x_,
        this->noc_y_,
        static_channel_allocator->get_sender_channel_base_address(vc, vc_relative_channel_id),  // Use VC-relative ID
        sender_channels_num_buffer,
        this->sender_channels_flow_control_semaphore_id[absolute_channel_id],
        this->sender_channels_connection_semaphore_id[absolute_channel_id],
        this->config.sender_channels_worker_conn_info_base_address[absolute_channel_id],
        this->config.channel_buffer_size_bytes,
        this->sender_channels_buffer_index_semaphore_id[absolute_channel_id],
        eth_chan_directions::EAST};
}

// Base class override for backward compatibility (treats channel_id as both absolute and VC0-relative)
SenderWorkerAdapterSpec FabricEriscDatamoverBuilder::build_connection_to_fabric_channel(uint32_t channel_id) const {
    // For VC0, absolute == VC-relative
    return build_connection_to_fabric_channel(0, channel_id, channel_id);  // Default to VC0
}

// TODO: take the downstream sender channel, based on the VC index, and use it to construct our
// `to_sender_channel_adapter` The `to_sender_channel_adapter` type is resolved based on the type of the downstream
// sender channel it is connecting to.
//   downstream == static? => instantiate static_sender_channel_adapter
//   downstream == elastic? => instantiate elastic_sender_channel_adapter
void FabricEriscDatamoverBuilder::setup_downstream_vc_connection(
    FabricDatamoverBuilderBase* downstream_builder,
    uint32_t upstream_vc_idx,
    uint32_t downstream_vc_idx,
    uint32_t absolute_channel_id,
    uint32_t vc_relative_channel_id) {
    TT_FATAL(
        upstream_vc_idx < builder_config::MAX_NUM_VCS,
        "Upstream VC index {} exceeds maximum supported VCs ({})",
        upstream_vc_idx,
        builder_config::MAX_NUM_VCS);
    TT_FATAL(
        downstream_vc_idx < builder_config::MAX_NUM_VCS,
        "Downstream VC index {} exceeds maximum supported VCs ({})",
        downstream_vc_idx,
        builder_config::MAX_NUM_VCS);

    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const bool is_2D_routing = fabric_context.is_2D_routing_enabled();

    // VC1 is only supported for 2D routing
    if (upstream_vc_idx == 1 || downstream_vc_idx == 1) {
        TT_FATAL(is_2D_routing, "VC1 is only supported for 2D routing");
    }

    // Validate upstream VC1 usage
    if (upstream_vc_idx == 1) {
        TT_FATAL(
            config.num_used_receiver_channels_per_vc[1] > 0,
            "VC1 receiver channels not configured on upstream router.");
    }

    const auto ds_noc_x = downstream_builder->get_noc_x();
    const auto ds_noc_y = downstream_builder->get_noc_y();
    eth_chan_directions ds_dir = downstream_builder->get_direction();

    // Build connection using absolute and VC-relative indices (both computed and validated by caller)
    // For ERISC builders, use the VC-aware overload; for others, use base class interface (defaults to VC0)
    SenderWorkerAdapterSpec adapter_spec;
    if (auto* downstream_erisc_builder = dynamic_cast<FabricEriscDatamoverBuilder*>(downstream_builder)) {
        adapter_spec = downstream_erisc_builder->build_connection_to_fabric_channel(
            downstream_vc_idx, absolute_channel_id, vc_relative_channel_id);
    } else {
        // For non-ERISC builders (e.g., TENSIX), use base class interface (treats as VC0-relative)
        adapter_spec = downstream_builder->build_connection_to_fabric_channel(vc_relative_channel_id);
    }

    if (auto* downstream_tensix_builder = dynamic_cast<FabricTensixDatamoverBuilder*>(downstream_builder)) {
        this->num_downstream_tensix_connections++;
        downstream_tensix_builder->append_upstream_routers_noc_xy(this->noc_x_, this->noc_y_);
    }

    auto* channel_allocator = config.channel_allocator.get();
    auto* const static_channel_allocator =
        dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(channel_allocator);
    TT_FATAL(static_channel_allocator != nullptr, "Channel allocator must be a FabricStaticSizedChannelsAllocator.");
    auto* adapter_ptr = this->receiver_channel_to_downstream_adapter.get();
    TT_FATAL(adapter_ptr != nullptr, "Adapter is not set. Failed to build TT-Fabric router. Internal error.");
    adapter_ptr->add_downstream_connection(
        adapter_spec, upstream_vc_idx, absolute_channel_id, ds_dir, CoreCoord(ds_noc_x, ds_noc_y), is_2D_routing);
}

size_t FabricEriscDatamoverBuilder::get_configured_risc_count() const { return this->config.risc_configs.size(); }

void FabricEriscDatamoverBuilder::teardown_from_host(
    tt::tt_metal::IDevice* d, tt::tt_fabric::TerminationSignal termination_signal) const {
    std::vector<uint32_t> val(1, termination_signal);
    tt::tt_metal::detail::WriteToDeviceL1(
        d,
        d->logical_core_from_ethernet_core(CoreCoord(this->noc_x_, this->noc_y_)),
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
