// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <enchantum/enchantum.hpp>

#include <stdint.h>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <unordered_set>
#include <variant>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "core_coord.hpp"
#include "fabric_edm_types.hpp"
#include <tt-logger/tt-logger.hpp>
#include <umd/device/tt_core_coordinates.h>

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
static void configure_risc_settings(
    size_t num_riscv_cores,
    size_t risc_id,
    tt::ARCH arch,
    bool& enable_handshake,
    bool& enable_context_switch,
    bool& enable_interrupts,
    std::array<bool, FabricEriscDatamoverConfig::num_sender_channels>& is_sender_channel_serviced,
    std::array<bool, FabricEriscDatamoverConfig::num_receiver_channels>& is_receiver_channel_serviced) {
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
            enable_context_switch = false;
            enable_interrupts = false;
            is_sender_channel_serviced.fill(true);
            is_receiver_channel_serviced.fill(true);
        } else {
            // Blackhole: Distribute sender/receiver across two RISC cores
            if (risc_id == 0) {
                // ERISC0: Handle sender channels only
                enable_handshake = true;
                enable_context_switch = false;
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

static size_t get_num_riscv_cores() {
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_is_fabric_2_erisc_mode_enabled()) {
        size_t nriscs = tt::tt_metal::MetalContext::instance().hal().get_processor_classes_count(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
        if (nriscs > 1) {
            log_warning(tt::LogFabric, "Launching fabric in experimental 2-erisc mode.");
        }
        return nriscs;
    } else {
        return 1;
    }
}

FabricRiscConfig::FabricRiscConfig(uint32_t risc_id) :
    noc_(risc_id == 0 ? tt::tt_metal::NOC::NOC_0 : tt::tt_metal::NOC::NOC_1),
    enable_handshake_(true),
    enable_context_switch_(true),
    enable_interrupts_(true),
    iterations_between_ctx_switch_and_teardown_checks_(
        FabricEriscDatamoverConfig::default_iterations_between_ctx_switch_and_teardown_checks) {
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

FabricEriscDatamoverConfig::FabricEriscDatamoverConfig(Topology topology) {
    this->topology = topology;
    uint32_t num_sender_channels = get_sender_channel_count(topology);
    uint32_t num_downstream_edms = get_downstream_edm_count(topology);
    // Global
    size_t next_l1_addr = tt::tt_metal::hal::get_erisc_l1_unreserved_base();

    // https://github.com/tenstorrent/tt-metal/issues/26354 to track fix for this hack where we always set aside the
    // memory for the telemetry buffer in Blackhole
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_enable_fabric_telemetry() || tt::tt_metal::MetalContext::instance().hal().get_arch() == tt::ARCH::BLACKHOLE) {
        // Avoid a bug on BH, always allocate the space for the telemetry buffer
        this->perf_telemetry_buffer_address = next_l1_addr;
        next_l1_addr += 32;
    }

    this->handshake_addr = next_l1_addr;
    next_l1_addr += eth_channel_sync_size;

    // Ethernet txq IDs on WH are 0,1 and on BH are 0,1,2.
    if (tt::tt_metal::MetalContext::instance().hal().get_arch() == tt::ARCH::BLACKHOLE) {
        this->receiver_txq_id = 1;
    }
    this->num_riscv_cores = get_num_riscv_cores();
    for (uint32_t risc_id = 0; risc_id < this->num_riscv_cores; risc_id++) {
        this->risc_configs.emplace_back(risc_id);
    }

    if (this->sender_txq_id != this->receiver_txq_id) {
        for (size_t i = 0; i < num_sender_channels; i++) {
            this->to_sender_channel_remote_ack_counter_addrs[i] = next_l1_addr;
            next_l1_addr += field_size;
        }
        for (size_t i = 0; i < num_sender_channels; i++) {
            this->to_sender_channel_remote_completion_counter_addrs[i] = next_l1_addr;
            next_l1_addr += field_size;
        }
        for (size_t i = 0; i < num_receiver_channels; i++) {
            this->receiver_channel_remote_ack_counter_addrs[i] = next_l1_addr;
            next_l1_addr += field_size;
        }
        for (size_t i = 0; i < num_receiver_channels; i++) {
            this->receiver_channel_remote_completion_counter_addrs[i] = next_l1_addr;
            next_l1_addr += field_size;
        }
    }

    this->edm_channel_ack_addr = next_l1_addr;
    this->termination_signal_address =
        edm_channel_ack_addr +
        (4 * eth_channel_sync_size);  // pad extra bytes to match old EDM so handshake logic will still work
    this->edm_local_sync_address = termination_signal_address + field_size;
    this->edm_status_address = edm_local_sync_address + field_size;

    uint32_t buffer_address = edm_status_address + field_size;

    for (uint32_t i = 0; i < FabricEriscDatamoverConfig::num_receiver_channels; i++) {
        this->receiver_channels_counters_address[i] = buffer_address;
        buffer_address += receiver_channel_counters_size_bytes;
    }
    for (uint32_t i = 0; i < num_sender_channels; i++) {
        this->sender_channels_counters_address[i] = buffer_address;
        buffer_address += sender_channel_counters_size_bytes;
    }

    // Packet header history buffer(s)
    for (uint32_t i = 0; i < FabricEriscDatamoverConfig::num_receiver_channels; i++) {
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
        this->receiver_channels_local_buffer_index_address[i] = buffer_address;
        buffer_address += field_size;
        // persistent mode field
        this->receiver_channels_downstream_flow_control_semaphore_address[i] = buffer_address;
        buffer_address += field_size;
        this->receiver_channels_downstream_teardown_semaphore_address[i] = buffer_address;
        buffer_address += field_size;
    }

    // Channel Allocations
    this->max_l1_loading_size =
        tt::tt_metal::hal::get_erisc_l1_unreserved_size() + tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    this->buffer_region_start = (buffer_address + buffer_alignment) & ~(buffer_alignment - 1);  // Align
    this->available_channel_buffering_space = max_l1_loading_size - buffer_region_start;
}

void FabricEriscDatamoverConfig::configure_buffer_slots_helper(
    Topology topology,
    const FabricEriscDatamoverOptions& options,
    std::array<size_t, num_sender_channels>& num_sender_buffer_slots,
    std::array<size_t, num_sender_channels>& num_remote_sender_buffer_slots,
    std::array<size_t, num_receiver_channels>& num_receiver_buffer_slots,
    std::array<size_t, num_receiver_channels>& num_remote_receiver_buffer_slots,
    std::array<size_t, num_downstream_sender_channels>& num_downstream_sender_buffer_slots) {
    static const std::vector<std::vector<std::pair<size_t, size_t>>> ring_buffer_slot_options = {
        {{8, 8}, {4, 8}}, {{8, 8}, {4, 8}}};

    static const std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>> ring_buffer_slot_options_dateline = {
        {{{8, 16}, {8, 8}}, {{16, 16}, {8, 16}, {8, 8}}}, {{{8, 16}, {8, 8}}, {{16, 16}, {8, 16}, {8, 8}}}};

    static const std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>>
        ring_buffer_slot_options_dateline_upstream = {
            {{{8, 16}, {8, 8}}, {{16, 16}, {8, 16}, {8, 8}}}, {{{8, 16}, {8, 8}}, {{16, 16}, {8, 16}, {8, 8}}}};

    static const std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>>
        ring_buffer_slot_options_dateline_upstream_adjcent = {
            {{{16, 8}, {8, 8}}, {{16, 8}, {8, 8}}}, {{{16, 8}, {8, 8}}, {{16, 8}, {8, 8}}}};

    auto get_num_buffer_slots = [](Topology topology,
                                   size_t arch_index) -> const std::vector<std::pair<size_t, size_t>>& {
        // Architecture-specific buffer slot configurations
        static const std::vector<std::vector<std::pair<size_t, size_t>>> mesh_buffer_slot_options = {
            {{7, 11}, {4, 8}},  // WORMHOLE_B0: {sender_slots, receiver_slots}
            {{8, 16}, {4, 8}}   // BLACKHOLE: {sender_slots, receiver_slots}
        };
        static const std::vector<std::vector<std::pair<size_t, size_t>>> other_buffer_slot_options = {
            {{8, 16}},  // WORMHOLE_B0: {sender_slots, receiver_slots}
            {{8, 16}}   // BLACKHOLE: {sender_slots, receiver_slots}
        };

        static tt::stl::Indestructible<std::vector<std::vector<std::pair<size_t, size_t>>>> mesh_slots(
            mesh_buffer_slot_options);
        static tt::stl::Indestructible<std::vector<std::vector<std::pair<size_t, size_t>>>> other_slots(
            other_buffer_slot_options);

        if (topology == Topology::Mesh) {
            return mesh_slots.get()[arch_index];
        } else {
            return other_slots.get()[arch_index];
        }
    };

    auto get_optimal_num_slots = [this](
                                     auto& buffer_slot_options,
                                     size_t num_sender_channels,
                                     size_t num_receiver_channels,
                                     size_t& num_sender_buffer_slots,
                                     size_t& num_receiver_buffer_slots,
                                     std::optional<size_t> worker_num_sender_buffer_slots = std::nullopt) {
        for (auto& option : buffer_slot_options) {
            num_sender_buffer_slots = option.first;
            num_receiver_buffer_slots = option.second;
            auto num_total_sender_slots = num_sender_channels * num_sender_buffer_slots;
            auto num_total_receiver_slots = num_receiver_channels * num_receiver_buffer_slots;
            if (worker_num_sender_buffer_slots.has_value()) {
                num_total_sender_slots =
                    worker_num_sender_buffer_slots.value() + (num_sender_channels - 1) * num_sender_buffer_slots;
            }
            auto total_num_bytes =
                (num_total_sender_slots + num_total_receiver_slots) * this->channel_buffer_size_bytes;
            if (total_num_bytes <= this->available_channel_buffering_space) {
                break;
            }
        }
    };

    auto fill_sender_buffer_slots = [&](auto& num_buffer_slots,
                                        size_t channel_skip_idx,
                                        uint32_t default_num_buffer_slots,
                                        uint32_t extra_num_buffer_slots) {
        for (size_t i = 0; i < this->num_used_sender_channels; ++i) {
            if (i == channel_skip_idx) {
                num_buffer_slots[i] = 0;
            } else {
                // tensix worker on channel 0, otherwise extra_num_buffer_slots
                num_buffer_slots[i] = (i == 0 ? default_num_buffer_slots : extra_num_buffer_slots);
            }
        }
    };

    auto fill_receiver_buffer_slots =
        [&](auto& num_buffer_slots, size_t channel_skip_idx, uint32_t extra_num_buffer_slots) {
            for (size_t i = 0; i < this->num_receiver_channels; ++i) {
                if (i == channel_skip_idx) {
                    num_buffer_slots[i] = 0;
                } else {
                    num_buffer_slots[i] = extra_num_buffer_slots;
                }
            }
        };

    auto axis_index = static_cast<std::size_t>(options.edm_axis);
    auto arch = tt::tt_metal::MetalContext::instance().hal().get_arch();
    size_t arch_index;
    if (arch == tt::ARCH::WORMHOLE_B0) {
        arch_index = 0;
    } else if (arch == tt::ARCH::BLACKHOLE) {
        arch_index = 1;
    } else {
        TT_THROW("Unsupported architecture: {}", enchantum::to_string(arch));
    }

    if (topology == Topology::Ring) {
        size_t default_num_sender_buffer_slots;
        size_t default_num_receiver_buffer_slots;
        // get the default buffer slots
        get_optimal_num_slots(
            ring_buffer_slot_options[arch_index],
            this->num_used_sender_channels,
            this->num_used_receiver_channels,
            default_num_sender_buffer_slots,
            default_num_receiver_buffer_slots);
        // get the dateline buffer slots
        size_t dateline_num_sender_buffer_slots;
        size_t dateline_num_receiver_buffer_slots;
        get_optimal_num_slots(
            ring_buffer_slot_options_dateline[arch_index][axis_index],
            this->num_used_sender_channels - 1,
            this->num_used_receiver_channels - 1,
            dateline_num_sender_buffer_slots,
            dateline_num_receiver_buffer_slots,
            default_num_sender_buffer_slots);
        // get the dateline upstream buffer slots
        size_t dateline_upstream_num_sender_buffer_slots;
        size_t dateline_upstream_num_receiver_buffer_slots;
        get_optimal_num_slots(
            ring_buffer_slot_options_dateline_upstream[arch_index][axis_index],
            this->num_used_sender_channels - 1,
            this->num_used_receiver_channels - 1,
            dateline_upstream_num_sender_buffer_slots,
            dateline_upstream_num_receiver_buffer_slots,
            default_num_sender_buffer_slots);
        // get the dateline upstream adjacent device buffer slots
        size_t dateline_upstream_adjcent_num_sender_buffer_slots;
        size_t dateline_upstream_adjcent_num_receiver_buffer_slots;
        get_optimal_num_slots(
            ring_buffer_slot_options_dateline_upstream_adjcent[arch_index][axis_index],
            this->num_used_sender_channels - 1,
            this->num_used_receiver_channels,
            dateline_upstream_adjcent_num_sender_buffer_slots,
            dateline_upstream_adjcent_num_receiver_buffer_slots,
            default_num_sender_buffer_slots);
        // set default buffer slots.
        num_sender_buffer_slots.fill(default_num_sender_buffer_slots);
        num_remote_sender_buffer_slots.fill(default_num_sender_buffer_slots);
        num_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
        num_remote_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
        num_downstream_sender_buffer_slots.fill(default_num_sender_buffer_slots);

        auto buffer_config = options.edm_buffer_config;
        switch (options.edm_type) {
            case FabricEriscDatamoverType::Dateline:
                if (buffer_config.enable_dateline_sender_extra_buffer_slots) {
                    // set num_sender_buffer_slots
                    fill_sender_buffer_slots(
                        num_sender_buffer_slots,
                        this->dateline_sender_channel_skip_idx,
                        default_num_sender_buffer_slots,
                        dateline_num_sender_buffer_slots);
                    // set remote sender buffer slots equal to local sender, since remote is also dateline
                    num_remote_sender_buffer_slots = num_sender_buffer_slots;
                }
                if (buffer_config.enable_dateline_receiver_extra_buffer_slots) {
                    // set num_receiver_buffer_slots
                    fill_receiver_buffer_slots(
                        num_receiver_buffer_slots,
                        this->dateline_receiver_channel_skip_idx,
                        dateline_num_receiver_buffer_slots);
                    // set remote receiver buffer slots equal to local receiver, since remote is also dateline
                    num_remote_receiver_buffer_slots = num_receiver_buffer_slots;
                }
                if (buffer_config.enable_dateline_upstream_sender_extra_buffer_slots) {
                    // set downstream_num_sender_buffer_slots, downstream is dateline-upstream-edm
                    num_downstream_sender_buffer_slots.fill(dateline_upstream_num_sender_buffer_slots);
                }
                break;
            case FabricEriscDatamoverType::DatelineUpstream:
                if (buffer_config.enable_dateline_upstream_sender_extra_buffer_slots) {
                    // set num_sender_buffer_slots
                    fill_sender_buffer_slots(
                        num_sender_buffer_slots,
                        this->dateline_upstream_sender_channel_skip_idx,
                        default_num_sender_buffer_slots,
                        dateline_upstream_num_sender_buffer_slots);
                    this->skip_sender_channel_1_connection = true;
                }
                // set num_receiver_buffer_slots
                if (buffer_config.enable_dateline_upstream_receiver_extra_buffer_slots) {
                    fill_receiver_buffer_slots(
                        num_receiver_buffer_slots,
                        this->dateline_upstream_receiver_channel_skip_idx,
                        dateline_upstream_num_receiver_buffer_slots);
                    this->skip_receiver_channel_1_connection = true;
                }
                if (buffer_config.enable_dateline_sender_extra_buffer_slots) {
                    // set downstream_num_sender_buffer_slots, downstream is dateline edm
                    num_downstream_sender_buffer_slots.fill(dateline_num_sender_buffer_slots);
                }
                if (buffer_config.enable_dateline_upstream_adjacent_sender_extra_buffer_slots) {
                    // set remote sender buffer slots equal to dateline upstream dajcent sender buffer slots
                    fill_sender_buffer_slots(
                        num_remote_sender_buffer_slots,
                        this->dateline_upstream_adjcent_sender_channel_skip_idx,
                        default_num_sender_buffer_slots,
                        dateline_upstream_adjcent_num_sender_buffer_slots);
                }
                break;
            case FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice:
                if (buffer_config.enable_dateline_upstream_adjacent_sender_extra_buffer_slots) {
                    // set num_sender_buffer_slots
                    fill_sender_buffer_slots(
                        num_sender_buffer_slots,
                        this->dateline_upstream_adjcent_sender_channel_skip_idx,
                        default_num_sender_buffer_slots,
                        dateline_upstream_adjcent_num_sender_buffer_slots);
                    this->skip_sender_vc1_channel_connection = true;
                }
                if (buffer_config.enable_dateline_upstream_sender_extra_buffer_slots) {
                    // set remote sender buffer slots equal to dateline upstream sender buffer slots
                    fill_sender_buffer_slots(
                        num_remote_sender_buffer_slots,
                        this->dateline_upstream_sender_channel_skip_idx,
                        default_num_sender_buffer_slots,
                        dateline_upstream_num_sender_buffer_slots);
                }
                if (buffer_config.enable_dateline_upstream_receiver_extra_buffer_slots) {
                    // set remote sender buffer slots equal to dateline upstream sender buffer slots
                    fill_receiver_buffer_slots(
                        num_remote_receiver_buffer_slots,
                        this->dateline_upstream_receiver_channel_skip_idx,
                        dateline_upstream_num_receiver_buffer_slots);
                }
                break;
            case FabricEriscDatamoverType::DatelineUpstreamAdjacentDeviceUpstream:
                if (buffer_config.enable_dateline_upstream_adjacent_sender_extra_buffer_slots) {
                    // set downstream_num_sender_buffer_slots, downstream is dateline upstream adjcent edm
                    num_downstream_sender_buffer_slots.fill(dateline_upstream_adjcent_num_sender_buffer_slots);
                }
                break;
            default: break;
        }
    } else {
        size_t default_num_sender_buffer_slots;
        size_t default_num_receiver_buffer_slots;
        get_optimal_num_slots(
            get_num_buffer_slots(topology, arch_index),
            this->num_used_sender_channels,
            this->num_used_receiver_channels,
            default_num_sender_buffer_slots,
            default_num_receiver_buffer_slots);
        // set default buffer slots.
        num_sender_buffer_slots.fill(default_num_sender_buffer_slots);
        num_remote_sender_buffer_slots.fill(default_num_sender_buffer_slots);
        num_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
        num_remote_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
        num_downstream_sender_buffer_slots.fill(default_num_sender_buffer_slots);
    }
}

FabricEriscDatamoverConfig::FabricEriscDatamoverConfig(
    std::size_t channel_buffer_size_bytes, Topology topology, FabricEriscDatamoverOptions options) :
    FabricEriscDatamoverConfig(topology) {
    this->sender_txq_id = 0;
    this->receiver_txq_id = 0;

    this->channel_buffer_size_bytes = channel_buffer_size_bytes;
    this->num_used_sender_channels = get_sender_channel_count(topology);
    this->num_used_receiver_channels = FabricEriscDatamoverConfig::num_receiver_channels;

    if (topology == Topology::Mesh) {
        // For 2D there is no forwarding to self but we are still initialize the settings for it.
        // Routers ignore the settings at self index.
        this->num_fwd_paths = this->num_used_sender_channels;
    } else {
        this->num_fwd_paths = this->num_used_sender_channels - 1;
    }

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

    std::array<size_t, num_sender_channels> num_sender_buffer_slots = {0};
    std::array<size_t, num_sender_channels> num_remote_sender_buffer_slots = {0};
    std::array<size_t, num_receiver_channels> num_receiver_buffer_slots = {0};
    std::array<size_t, num_receiver_channels> num_remote_receiver_buffer_slots = {0};
    std::array<size_t, num_downstream_sender_channels> num_downstream_sender_buffer_slots = {0};

    bool is_dateline = options.edm_type == FabricEriscDatamoverType::Dateline;
    bool is_dateline_upstream = options.edm_type == FabricEriscDatamoverType::DatelineUpstream;
    bool is_dateline_upstream_adj_dev = options.edm_type == FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice;
    bool is_dateline_upstream_adj_dev_upstream =
        options.edm_type == FabricEriscDatamoverType::DatelineUpstreamAdjacentDeviceUpstream;

    configure_buffer_slots_helper(
        topology,
        options,
        num_sender_buffer_slots,
        num_remote_sender_buffer_slots,
        num_receiver_buffer_slots,
        num_remote_receiver_buffer_slots,
        num_downstream_sender_buffer_slots);

    log_trace(
        tt::LogOp,
        "is_dateline {} is_dateline_upstream {} is_dateline_upstream_adj_dev {}, is_dateline_upstream_adj_dev_upstream "
        "{}",
        is_dateline,
        is_dateline_upstream,
        is_dateline_upstream_adj_dev,
        is_dateline_upstream_adj_dev_upstream);
    log_trace(tt::LogOp, "num_sender_buffer_slots: {}", num_sender_buffer_slots);
    log_trace(tt::LogOp, "num_remote_sender_buffer_slots: {}", num_remote_sender_buffer_slots);
    log_trace(tt::LogOp, "num_receiver_buffer_slots: {}", num_receiver_buffer_slots);
    log_trace(tt::LogOp, "num_remote_receiver_buffer_slots: {}", num_remote_receiver_buffer_slots);
    log_trace(tt::LogOp, "num_downstream_sender_buffer_slots: {}", num_downstream_sender_buffer_slots);

    size_t total_sender_slots = std::accumulate(
        num_sender_buffer_slots.begin(), num_sender_buffer_slots.begin() + this->num_used_sender_channels, size_t{0});
    size_t total_receiver_slots = std::accumulate(
        num_receiver_buffer_slots.begin(),
        num_receiver_buffer_slots.begin() + this->num_used_receiver_channels,
        size_t{0});
    std::size_t total_slot_count = total_sender_slots + total_receiver_slots;
    TT_FATAL(
        total_slot_count * channel_buffer_size_bytes <= available_channel_buffering_space,
        "Total channel size of {} B exceeds available space of {} B",
        total_slot_count * channel_buffer_size_bytes,
        available_channel_buffering_space);

    log_trace(tt::LogOp, "Available channel buffering space: {}", this->available_channel_buffering_space);
    // set the local sender channel sizes
    this->sender_channels_num_buffers = num_sender_buffer_slots;
    for (uint32_t i = 0; i < this->num_used_sender_channels; i++) {
        this->sender_channels_size_bytes[i] = channel_buffer_size_bytes * num_sender_buffer_slots[i];
    }
    // set the remote sender channel sizes
    this->remote_sender_channels_num_buffers = num_remote_sender_buffer_slots;
    for (uint32_t i = 0; i < this->num_used_sender_channels; i++) {
        this->remote_sender_channels_size_bytes[i] = channel_buffer_size_bytes * num_remote_sender_buffer_slots[i];
    }
    // set the local receiver channel sizes
    this->receiver_channels_num_buffers = num_receiver_buffer_slots;
    for (uint32_t i = 0; i < this->num_used_receiver_channels; i++) {
        this->receiver_channels_size_bytes[i] = channel_buffer_size_bytes * num_receiver_buffer_slots[i];
    }
    // set the remote receiver channel sizes
    this->remote_receiver_channels_num_buffers = num_remote_receiver_buffer_slots;
    for (uint32_t i = 0; i < this->num_used_receiver_channels; i++) {
        this->remote_receiver_channels_size_bytes[i] = channel_buffer_size_bytes * num_remote_receiver_buffer_slots[i];
    }
    // set the downstream channel sizes
    this->downstream_sender_channels_num_buffers = num_downstream_sender_buffer_slots;

    // set the base addresses for the local channels
    uint32_t sender_buffer_addr = buffer_region_start;
    for (uint32_t i = 0; i < this->num_used_sender_channels; i++) {
        this->sender_channels_base_address[i] = sender_buffer_addr;
        sender_buffer_addr += this->sender_channels_size_bytes[i];
        log_trace(tt::LogOp, "Sender {} channel_start: {}", i, this->sender_channels_base_address[i]);
    }
    uint32_t receiver_buffer_addr = sender_buffer_addr;
    for (uint32_t i = 0; i < this->num_used_receiver_channels; i++) {
        this->receiver_channels_base_address[i] = receiver_buffer_addr;
        receiver_buffer_addr += this->receiver_channels_size_bytes[i];
        log_trace(tt::LogOp, "Receiver {} channel_start: {}", i, this->receiver_channels_base_address[i]);
    }
    uint32_t buffer_addr_end = receiver_buffer_addr;
    // set the base addresses for the remote channels
    uint32_t remote_sender_buffer_addr = buffer_region_start;
    for (uint32_t i = 0; i < this->num_used_sender_channels; i++) {
        this->remote_sender_channels_base_address[i] = remote_sender_buffer_addr;
        remote_sender_buffer_addr += this->remote_sender_channels_size_bytes[i];
        log_trace(tt::LogOp, "Remote Sender {} channel_start: {}", i, this->remote_sender_channels_base_address[i]);
    }
    uint32_t remote_receiver_buffer_addr = remote_sender_buffer_addr;
    for (uint32_t i = 0; i < this->num_used_receiver_channels; i++) {
        this->remote_receiver_channels_base_address[i] = remote_receiver_buffer_addr;
        remote_receiver_buffer_addr += this->remote_receiver_channels_size_bytes[i];
        log_trace(tt::LogOp, "Remote Receiver {} channel_start: {}", i, this->remote_receiver_channels_base_address[i]);
    }

    log_trace(tt::LogOp, "Available channel buffering space: {}", this->available_channel_buffering_space);

    auto skip_current_sender_channel = [&](uint32_t idx) -> bool {
        return (idx == this->dateline_sender_channel_skip_idx && is_dateline) ||
               (idx == this->dateline_upstream_sender_channel_skip_idx && is_dateline_upstream) ||
               (idx == this->dateline_upstream_adjcent_sender_channel_skip_idx && is_dateline_upstream_adj_dev);
    };

    for (uint32_t i = 0; i < this->num_used_sender_channels; i++) {
        if (!skip_current_sender_channel(i)) {
            TT_FATAL(
                this->sender_channels_size_bytes[i] > 0,
                "Internal error when computing `sender_channels_size_bytes[{}]` which was computed to be size 0",
                i);
        }
    }

    auto skip_current_receiver_channel = [&](uint32_t idx) -> bool {
        return (idx == this->dateline_receiver_channel_skip_idx && is_dateline) ||
               (idx == this->dateline_upstream_receiver_channel_skip_idx && is_dateline_upstream);
    };

    for (uint32_t i = 0; i < this->num_used_receiver_channels; i++) {
        if (!skip_current_receiver_channel(i)) {
            TT_FATAL(
                this->receiver_channels_size_bytes[i] > 0,
                "Internal error when computing `receiver_channels_size_bytes[{}]` which was computed to be size 0",
                i);
        }
    }
    TT_FATAL(
        std::accumulate(
            this->sender_channels_size_bytes.begin(),
            this->sender_channels_size_bytes.begin() + this->num_used_sender_channels,
            0ul) +
                std::accumulate(
                    this->receiver_channels_size_bytes.begin(),
                    this->receiver_channels_size_bytes.begin() + this->num_used_receiver_channels,
                    0ul) <=
            this->available_channel_buffering_space,
        "Internal error when computing channel sizes. Total channel size exceeds available space");
    TT_FATAL(
        buffer_addr_end < this->max_l1_loading_size,
        "Internal error - channel buffers spilled past the end of usable L1 region.");

    // set default noc and cmd bufs (current setup in TG 4U)
    for (uint32_t i = 0; i < FabricEriscDatamoverConfig::num_receiver_channels; i++) {
        this->receiver_channel_forwarding_noc_ids[i] = FabricEriscDatamoverConfig::DEFAULT_RECEIVER_FORWARDING_NOC;
        this->receiver_channel_forwarding_data_cmd_buf_ids[i] = FabricEriscDatamoverConfig::WR_REG_CMD_BUF;
        this->receiver_channel_forwarding_sync_cmd_buf_ids[i] = FabricEriscDatamoverConfig::RD_CMD_BUF;
        this->receiver_channel_local_write_noc_ids[i] = FabricEriscDatamoverConfig::DEFAULT_RECEIVER_LOCAL_WRITE_NOC;
        this->receiver_channel_local_write_cmd_buf_ids[i] = FabricEriscDatamoverConfig::WR_CMD_BUF;
    }
    for (uint32_t i = 0; i < FabricEriscDatamoverConfig::num_sender_channels; i++) {
        this->sender_channel_ack_noc_ids[i] = FabricEriscDatamoverConfig::DEFAULT_SENDER_ACK_NOC;
        this->sender_channel_ack_cmd_buf_ids[i] = FabricEriscDatamoverConfig::AT_CMD_BUF;
    }
    this->edm_noc_vc = FabricEriscDatamoverConfig::DEFAULT_NOC_VC;
}

void get_runtime_args_for_edm_termination_infos(
    const std::vector<edm_termination_info_t>& edm_termination_infos, std::vector<uint32_t>& args_out) {
    args_out.reserve(args_out.size() + edm_termination_infos.size() * 4 + 1);
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
    chip_id_t chip_id,
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
    connection_info.edm_l1_sem_addr = connection.edm_l1_sem_addr;
    connection_info.edm_connection_handshake_addr = connection.edm_connection_handshake_addr;
    connection_info.edm_worker_location_info_addr = connection.edm_worker_location_info_addr;
    connection_info.buffer_size_bytes = connection.buffer_size_bytes;
    connection_info.buffer_index_semaphore_id = connection.buffer_index_semaphore_id;
    // NOTE: valid_connections_mask is not copied to L1 from performance reason
    //       because this callstack will be deprecated and not used in WorkerToFabricEdmSenderImpl yet
    //       we want to reduce the number of write_core calls
    fabric_connections.valid_connections_mask |= (1u << eth_channel);

    size_t connection_offset = offsetof(tt::tt_fabric::tensix_fabric_connections_l1_info_t, read_only) +
                               eth_channel * sizeof(tt::tt_fabric::fabric_connection_info_t);
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
    log_trace(tt::LogOp, "Worker to fabric EDM Sender has {} RT Args: {}", args.size(), args);
    log_trace(tt::LogOp, "arg[{}]: edm_noc_xy {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogOp, "arg[{}]: edm_buffer_base_addr {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogOp, "arg[{}]: num_buffers_per_channel {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogOp, "arg[{}]: edm_l1_sem_addr {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogOp, "arg[{}]: edm_connection_handshake_addr {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogOp, "arg[{}]: edm_worker_location_info_addr {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogOp, "arg[{}]: buffer_size_bytes {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogOp, "arg[{}]: buffer_index_semaphore_id {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(
        tt::LogOp, "arg[{}]: sender_worker_flow_control_semaphore_id {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(
        tt::LogOp, "arg[{}]: sender_worker_buffer_index_semaphore_id {}", starting_arg_idx, args[starting_arg_idx++]);
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
    const std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels>&
        sender_channels_flow_control_semaphore_id,
    const std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels>& sender_channels_connection_semaphore_id,
    const std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels>&
        sender_channels_buffer_index_semaphore_id,

    const FabricEriscDatamoverConfig& config,
    eth_chan_directions direction,
    bool build_in_worker_connection_mode,
    bool dateline_connection) :
    my_eth_core_logical(my_eth_core_logical),
    my_eth_channel(my_eth_core_logical.y),
    my_noc_x(my_noc_x),
    my_noc_y(my_noc_y),
    config(config),
    direction(direction),
    local_fabric_node_id(local_fabric_node_id),
    peer_fabric_node_id(peer_fabric_node_id),
    handshake_address(tt::round_up(
        tt::tt_metal::hal::get_erisc_l1_unreserved_base(), FabricEriscDatamoverConfig::eth_channel_sync_size)),
    channel_buffer_size(config.channel_buffer_size_bytes),
    sender_channels_num_buffers(config.sender_channels_num_buffers),
    receiver_channels_num_buffers(config.receiver_channels_num_buffers),
    remote_receiver_channels_num_buffers(config.remote_receiver_channels_num_buffers),
    downstream_sender_channels_num_buffers(config.downstream_sender_channels_num_buffers),

    // this is the receiver channel's local sem for flow controlling with downstream fabric sender
    receiver_channels_downstream_flow_control_semaphore_id(receiver_channels_downstream_flow_control_semaphore_id),
    receiver_channels_downstream_teardown_semaphore_id(receiver_channels_downstream_teardown_semaphore_id),
    sender_channels_flow_control_semaphore_id(sender_channels_flow_control_semaphore_id),
    sender_channels_connection_semaphore_id(sender_channels_connection_semaphore_id),
    sender_channels_buffer_index_semaphore_id(sender_channels_buffer_index_semaphore_id),
    downstream_vcs_sender_channel_buffer_index_semaphore_id(sender_channels_buffer_index_semaphore_id),

    receiver_channels_local_buffer_index_address(config.receiver_channels_local_buffer_index_address),
    local_sender_channels_buffer_address(config.sender_channels_base_address),
    remote_sender_channels_base_address(config.remote_sender_channels_base_address),
    local_sender_channels_connection_info_addr(config.sender_channels_worker_conn_info_base_address),
    local_receiver_channels_buffer_address(config.receiver_channels_base_address),
    remote_receiver_channels_base_address(config.remote_receiver_channels_base_address),

    termination_signal_ptr(config.termination_signal_address),
    edm_local_sync_ptr(config.edm_local_sync_address),
    edm_status_ptr(config.edm_status_address),
    build_in_worker_connection_mode(build_in_worker_connection_mode),
    dateline_connection(dateline_connection) {
    std::fill(
        sender_channel_connection_liveness_check_disable_array.begin(),
        sender_channel_connection_liveness_check_disable_array.end(),
        false);
}

void FabricEriscDatamoverBuilder::get_telemetry_compile_time_args(std::vector<uint32_t>& ct_args) const {
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    uint32_t telemetry_mode = static_cast<uint32_t>(
        rtoptions.get_enable_fabric_telemetry() ? 1 : 0);
    ct_args.push_back(telemetry_mode);

    // Add telemetry buffer address (16B aligned)
    ct_args.push_back(static_cast<uint32_t>(config.perf_telemetry_buffer_address));
}

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

    for (uint32_t i = 0; i < FabricEriscDatamoverConfig::num_sender_channels; i++) {
        log_trace(tt::LogTest, "Sender {} num buffers: {}", i, this->sender_channels_num_buffers[i]);
        log_trace(tt::LogTest, "Sender {} channel address: {}", i, this->local_sender_channels_buffer_address[i]);
    }
    for (uint32_t i = 0; i < FabricEriscDatamoverConfig::num_receiver_channels; i++) {
        log_trace(tt::LogTest, "Receiver {} num buffers: {}", i, this->receiver_channels_num_buffers[i]);
        log_trace(tt::LogTest, "Receiver {} channel address: {}", i, this->local_receiver_channels_buffer_address[i]);
    }

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

    size_t sender_channel_num_buffers = this->sender_channels_num_buffers[0];
    size_t receiver_channel_num_buffers =
        this->dateline_connection ? this->receiver_channels_num_buffers[1] : this->receiver_channels_num_buffers[0];

    TT_FATAL(sender_channel_num_buffers > 0, "Sender channel num buffers must be greater than 0");
    TT_FATAL(receiver_channel_num_buffers > 0, "Receiver channel num buffers must be greater than 0");

    auto ct_args = std::vector<uint32_t>{
        num_sender_channels,
        num_receiver_channels,
        config.num_fwd_paths,
        this->wait_for_host_signal ? 1 : 0,

        this->firmware_context_switch_interval,
        this->enable_first_level_ack,
        this->fuse_receiver_flush_and_completion_ptr,
        config.topology == Topology::Ring,
        this->dateline_connection,
        is_handshake_master,
        this->handshake_address,
        this->channel_buffer_size,

        config.skip_receiver_channel_1_connection,
        config.skip_sender_channel_1_connection,
        config.skip_sender_vc1_channel_connection,

        config.sender_channels_base_address[0],
        config.sender_channels_worker_conn_info_base_address[0],
        config.sender_channels_base_address[1],
        config.sender_channels_worker_conn_info_base_address[1],
        config.sender_channels_base_address[2],
        config.sender_channels_worker_conn_info_base_address[2],
        config.sender_channels_base_address[3],
        config.sender_channels_worker_conn_info_base_address[3],
        config.sender_channels_base_address[4],
        config.sender_channels_worker_conn_info_base_address[4],

        config.receiver_channels_base_address[0],         // local
        config.remote_receiver_channels_base_address[0],  // remote
        config.receiver_channels_base_address[1],         // local
        config.remote_receiver_channels_base_address[1],  // remote

        config.remote_sender_channels_base_address[0],  // remote
        config.remote_sender_channels_base_address[1],  // remote
        config.remote_sender_channels_base_address[2],  // remote
        config.remote_sender_channels_base_address[3],  // remote
        config.remote_sender_channels_base_address[4],  // remote

        this->termination_signal_ptr,
        this->edm_local_sync_ptr,
        this->edm_status_ptr,

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
        config.topology == Topology::Mesh,
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

        // Special marker to help with identifying misalignment bugs
        0x00c0ffee};

    // insert the sender channel num buffers
    const size_t sender_channel_num_buffers_idx = 12;
    ct_args.insert(
        ct_args.begin() + sender_channel_num_buffers_idx,
        this->sender_channels_num_buffers.begin(),
        this->sender_channels_num_buffers.begin() + num_sender_channels);
    // insert the receiver channel num buffers
    const size_t receiver_channel_num_buffers_idx = sender_channel_num_buffers_idx + num_sender_channels;
    ct_args.insert(
        ct_args.begin() + receiver_channel_num_buffers_idx,
        this->receiver_channels_num_buffers.begin(),
        this->receiver_channels_num_buffers.begin() + num_receiver_channels);
    // insert the remote receiver channel num buffers
    const size_t remote_receiver_channel_num_buffers_idx = receiver_channel_num_buffers_idx + num_receiver_channels;
    ct_args.insert(
        ct_args.begin() + remote_receiver_channel_num_buffers_idx,
        this->remote_receiver_channels_num_buffers.begin(),
        this->remote_receiver_channels_num_buffers.begin() + num_receiver_channels);
    // insert the downstream sender channel num buffers
    const size_t downstream_sender_channel_num_buffers_idx =
        remote_receiver_channel_num_buffers_idx + num_receiver_channels;
    ct_args.insert(
        ct_args.begin() + downstream_sender_channel_num_buffers_idx,
        this->downstream_sender_channels_num_buffers.begin(),
        this->downstream_sender_channels_num_buffers.begin() + config.num_fwd_paths);

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
        for (size_t i = 0; i < num_sender_channels; i++) {
            ct_args.push_back(config.to_sender_channel_remote_ack_counter_addrs[i]);
        }
        for (size_t i = 0; i < num_sender_channels; i++) {
            ct_args.push_back(config.to_sender_channel_remote_completion_counter_addrs[i]);
        }
        for (size_t i = 0; i < num_receiver_channels; i++) {
            ct_args.push_back(config.receiver_channel_remote_ack_counter_addrs[i]);
        }
        for (size_t i = 0; i < num_receiver_channels; i++) {
            ct_args.push_back(config.receiver_channel_remote_completion_counter_addrs[i]);
        }
    }

    ct_args.push_back(0x30c0ffee);
    return ct_args;
}

std::vector<uint32_t> FabricEriscDatamoverBuilder::get_runtime_args() const {
    return std::vector<uint32_t>{
        this->sender_channels_connection_semaphore_id[0],
        this->sender_channels_connection_semaphore_id[1],
        this->sender_channels_connection_semaphore_id[2],
        this->sender_channels_connection_semaphore_id[3],
        this->sender_channels_connection_semaphore_id[4],

        this->downstream_vcs_sender_channel_buffer_index_semaphore_id[0],
        this->downstream_vcs_sender_channel_buffer_index_semaphore_id[1],
        this->downstream_vcs_sender_channel_buffer_index_semaphore_id[2],
        this->downstream_vcs_sender_channel_buffer_index_semaphore_id[3],
        this->downstream_vcs_sender_channel_buffer_index_semaphore_id[4],

        this->downstream_edms_connected,
        this->downstream_edm_vcs_buffer_base_address[1].value_or(0),
        this->downstream_edm_vcs_noc_x[1].value_or(0),
        this->downstream_edm_vcs_noc_y[1].value_or(0),
        this->downstream_edm_vcs_semaphore_address[1].value_or(-1),
        this->downstream_edm_vcs_worker_registration_address[1].value_or(0),
        this->downstream_edm_vcs_worker_location_info_address[1].value_or(0),
        this->receiver_channels_local_buffer_index_address[0],  // extend the following 3 for 2D. need 3 each for 2D.

        this->downstream_edm_vcs_buffer_base_address[2] != std::nullopt,
        this->downstream_edm_vcs_buffer_base_address[2].value_or(0),
        this->downstream_edm_vcs_noc_x[2].value_or(0),
        this->downstream_edm_vcs_noc_y[2].value_or(0),
        this->downstream_edm_vcs_semaphore_address[2].value_or(-1),
        this->downstream_edm_vcs_worker_registration_address[2].value_or(0),
        this->downstream_edm_vcs_worker_location_info_address[2].value_or(0),
        this->receiver_channels_local_buffer_index_address[1],
        // this is the receiver channel's local sem for flow controlling with downstream fabric sender
        this->receiver_channels_downstream_flow_control_semaphore_id[0].value_or(-1),
        this->receiver_channels_downstream_flow_control_semaphore_id[1].value_or(-1),
        this->receiver_channels_downstream_flow_control_semaphore_id[2].value_or(-1),
        this->receiver_channels_downstream_flow_control_semaphore_id[3].value_or(-1),
        this->receiver_channels_downstream_flow_control_semaphore_id[4].value_or(-1),
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
}

FabricEriscDatamoverBuilder FabricEriscDatamoverBuilder::build(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    const CoreCoord& ethernet_core,
    chip_id_t local_physical_chip_id,
    chip_id_t peer_physical_chip_id,
    const FabricEriscDatamoverConfig& config,
    bool build_in_worker_connection_mode,
    bool dateline_connection,
    eth_chan_directions direction) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    return FabricEriscDatamoverBuilder::build(
        device,
        program,
        ethernet_core,
        control_plane.get_fabric_node_id_from_physical_chip_id(local_physical_chip_id),
        control_plane.get_fabric_node_id_from_physical_chip_id(peer_physical_chip_id),
        config,
        build_in_worker_connection_mode,
        dateline_connection,
        direction);
}

FabricEriscDatamoverBuilder FabricEriscDatamoverBuilder::build(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    const CoreCoord& ethernet_core,
    const FabricNodeId& local_fabric_node_id,
    const FabricNodeId& peer_fabric_node_id,
    const FabricEriscDatamoverConfig& config,
    bool build_in_worker_connection_mode,
    bool dateline_connection,
    eth_chan_directions direction) {
    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> sender_channels_buffer_index_semaphore_id;
    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> sender_channels_flow_control_semaphore_id;
    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> sender_channels_connection_semaphore_id;
    std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>
        receiver_channels_downstream_flow_control_semaphore_id;
    std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>
        receiver_channels_downstream_teardown_semaphore_id;
    if (build_in_worker_connection_mode) {
        for (uint32_t i = 0; i < FabricEriscDatamoverConfig::num_receiver_channels; i++) {
            receiver_channels_downstream_flow_control_semaphore_id[i] = 0;
            receiver_channels_downstream_teardown_semaphore_id[i] = 0;
        }
        // Sender channel 0 uses addresses instead of ids in persistent mode
        sender_channels_buffer_index_semaphore_id[0] = config.sender_channels_buffer_index_semaphore_address[0];
        sender_channels_flow_control_semaphore_id[0] = config.sender_channels_local_flow_control_semaphore_address[0];
        sender_channels_connection_semaphore_id[0] = config.sender_channels_connection_semaphore_address[0];
        for (uint32_t i = 1; i < FabricEriscDatamoverConfig::num_sender_channels; i++) {
            sender_channels_flow_control_semaphore_id[i] = 0;
            sender_channels_connection_semaphore_id[i] = 0;
            sender_channels_buffer_index_semaphore_id[i] = 0;
        }
    } else {
        bool mesh = config.topology == Topology::Mesh;
        uint32_t num_vc0_downstream_edms = mesh ? FabricEriscDatamoverConfig::num_downstream_edms_2d_vc0
                                                : FabricEriscDatamoverConfig::num_downstream_edms_vc0;

        // Setup VC0 downstrteam edm semaphore settings.
        // 1D has 1 downstream edm. 2D has 3 downstream EDMs
        // 2D uses the reserved addresses in L1 from FabricEriscDatamoverConfig
        for (uint32_t i = 0; i < num_vc0_downstream_edms; i++) {
            if (mesh) {
                receiver_channels_downstream_flow_control_semaphore_id[i] =
                    config.receiver_channels_downstream_flow_control_semaphore_address[i];
                receiver_channels_downstream_teardown_semaphore_id[i] =
                    config.receiver_channels_downstream_teardown_semaphore_address[i];
            } else {
                receiver_channels_downstream_flow_control_semaphore_id[i] =
                    tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
                receiver_channels_downstream_teardown_semaphore_id[i] =
                    tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
            }
        }
        // Setup VC1 downstream edm
        // 1D and 2D have 1 downstream edm for VC1 in the diretion of respective axis
        if (mesh) {
            receiver_channels_downstream_flow_control_semaphore_id[num_vc0_downstream_edms] =
                config.receiver_channels_downstream_flow_control_semaphore_address[num_vc0_downstream_edms];
            receiver_channels_downstream_teardown_semaphore_id[num_vc0_downstream_edms] =
                config.receiver_channels_downstream_teardown_semaphore_address[num_vc0_downstream_edms];

        } else {
            receiver_channels_downstream_flow_control_semaphore_id[num_vc0_downstream_edms] =
                tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
            receiver_channels_downstream_teardown_semaphore_id[num_vc0_downstream_edms] =
                tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
        }
        uint32_t num_sender_channels = mesh ? FabricEriscDatamoverConfig::num_sender_channels_2d
                                            : FabricEriscDatamoverConfig::num_sender_channels_1d;
        for (uint32_t i = 0; i < num_sender_channels; i++) {
            if (mesh) {
                sender_channels_buffer_index_semaphore_id[i] = config.sender_channels_buffer_index_semaphore_address[i];
                sender_channels_flow_control_semaphore_id[i] =
                    config.sender_channels_local_flow_control_semaphore_address[i];
                sender_channels_connection_semaphore_id[i] = config.sender_channels_connection_semaphore_address[i];
            } else {
                if (i == 0) {
                    // Sender channel 0 uses addresses instead of ids in persistent mode
                    sender_channels_buffer_index_semaphore_id[i] =
                        config.sender_channels_buffer_index_semaphore_address[i];
                    sender_channels_flow_control_semaphore_id[i] =
                        config.sender_channels_local_flow_control_semaphore_address[i];
                    sender_channels_connection_semaphore_id[i] = config.sender_channels_connection_semaphore_address[i];
                } else {
                    sender_channels_flow_control_semaphore_id[i] =
                        tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
                    sender_channels_connection_semaphore_id[i] =
                        tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
                    sender_channels_buffer_index_semaphore_id[i] =
                        tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
                }
            }
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
        dateline_connection);
}

SenderWorkerAdapterSpec FabricEriscDatamoverBuilder::build_connection_to_worker_channel() const {
    log_trace(tt::LogOp, "Building connection to persistent fabric");
    static constexpr uint32_t worker_chan = 0;
    TT_FATAL(
        sender_channels_buffer_index_semaphore_id[worker_chan] !=
            sender_channels_flow_control_semaphore_id[worker_chan],
        "Internal error - sender_channel_buffer_index_semaphore_id and sender_channel_flow_control_semaphore_id "
        "aliased eachother");
    return SenderWorkerAdapterSpec{
        this->my_noc_x,
        this->my_noc_y,
        this->local_sender_channels_buffer_address[worker_chan],
        this->sender_channels_num_buffers[worker_chan],
        this->sender_channels_flow_control_semaphore_id[worker_chan],
        this->sender_channels_connection_semaphore_id[worker_chan],
        this->config.sender_channels_worker_conn_info_base_address[worker_chan],
        this->config.channel_buffer_size_bytes,
        this->sender_channels_buffer_index_semaphore_id[worker_chan],
        this->direction};
}

SenderWorkerAdapterSpec FabricEriscDatamoverBuilder::build_connection_to_fabric_channel(uint32_t ds_edm) {
    auto max_ds_edm_count = config.topology == Topology::Mesh ? FabricEriscDatamoverConfig::num_sender_channels_2d
                                                              : FabricEriscDatamoverConfig::num_sender_channels_1d;
    if (ds_edm >= max_ds_edm_count) {
        TT_THROW("Invalid VC");
    }
    this->sender_channel_connection_liveness_check_disable_array[ds_edm] = true;
    return SenderWorkerAdapterSpec{
        this->my_noc_x,
        this->my_noc_y,
        this->local_sender_channels_buffer_address[ds_edm],
        this->sender_channels_num_buffers[ds_edm],
        this->sender_channels_flow_control_semaphore_id[ds_edm],
        this->sender_channels_connection_semaphore_id[ds_edm],
        this->config.sender_channels_worker_conn_info_base_address[ds_edm],
        this->config.channel_buffer_size_bytes,
        this->sender_channels_buffer_index_semaphore_id[ds_edm],
        eth_chan_directions::EAST};
}

void FabricEriscDatamoverBuilder::connect_to_downstream_edm(FabricEriscDatamoverBuilder& downstream_edm) {
    TT_FATAL(
        !this->build_in_worker_connection_mode, "Tried to connect two EDMs to each other in worker connection mode");
    const auto ds_noc_x = downstream_edm.get_noc_x();
    const auto ds_noc_y = downstream_edm.get_noc_y();
    eth_chan_directions ds_dir = downstream_edm.get_direction();

    log_debug(
        tt::LogTest,
        "EDM at x={}, y={}, Direction={} :: Connecting to downstream EDM at x={}, y={}, VC={}, Direction={}",
        my_noc_x,
        my_noc_y,
        direction,
        ds_noc_x,
        ds_noc_y,
        0,
        ds_dir);

    // VC 0
    // For non-mesh, downstream channel is always 1 because channel 0 is always reserved for worker connections
    auto ds_edm_send_chan = config.topology == Topology::Mesh ? this->direction : 1;
    auto adapter_spec = downstream_edm.build_connection_to_fabric_channel(ds_edm_send_chan);

    if (config.topology == Topology::Mesh) {
        uint32_t val = this->downstream_edm_vcs_noc_x[1].value_or(0);
        val |= (ds_noc_x << (ds_dir * 8));
        this->downstream_edm_vcs_noc_x[1] = val;

        val = this->downstream_edm_vcs_noc_y[1].value_or(0);
        val |= (ds_noc_y << (ds_dir * 8));
        this->downstream_edm_vcs_noc_y[1] = val;

        this->downstream_edms_connected |= 0x1 << ds_dir;
    } else {
        this->downstream_edm_vcs_noc_x[1] = ds_noc_x;
        this->downstream_edm_vcs_noc_y[1] = ds_noc_y;
        this->downstream_vcs_sender_channel_buffer_index_semaphore_id[1] = adapter_spec.buffer_index_semaphore_id;
        this->downstream_edms_connected = 1;
    }

    this->downstream_edm_vcs_buffer_base_address[1] = adapter_spec.edm_buffer_base_addr;
    this->downstream_edm_vcs_semaphore_address[1] = adapter_spec.edm_l1_sem_addr;
    this->downstream_edm_vcs_worker_registration_address[1] = adapter_spec.edm_connection_handshake_addr;
    this->downstream_edm_vcs_worker_location_info_address[1] = adapter_spec.edm_worker_location_info_addr;

    // VC 1
    // Indexing from the back -- grabbing the last channel as it is the designated VC1 channel
    ds_edm_send_chan = config.topology == Topology::Mesh ? FabricEriscDatamoverConfig::num_sender_channels_2d - 1
                                                         : FabricEriscDatamoverConfig::num_sender_channels_1d - 1;
    adapter_spec = downstream_edm.build_connection_to_fabric_channel(ds_edm_send_chan);

    bool connect_vc1 = config.topology != Topology::Mesh ||
                       (this->direction == eth_chan_directions::EAST && ds_dir == eth_chan_directions::WEST) ||
                       (this->direction == eth_chan_directions::WEST && ds_dir == eth_chan_directions::EAST) ||
                       (this->direction == eth_chan_directions::NORTH && ds_dir == eth_chan_directions::SOUTH) ||
                       (this->direction == eth_chan_directions::SOUTH && ds_dir == eth_chan_directions::NORTH);

    if (connect_vc1) {
        log_debug(
            tt::LogTest,
            "EDM at x={}, y={}, Direction={} :: Connecting to downstream EDM at x={}, y={}, VC={}, Direction={}",
            my_noc_x,
            my_noc_y,
            direction,
            ds_noc_x,
            ds_noc_y,
            1,
            ds_dir);

        this->downstream_edm_vcs_noc_x[2] = ds_noc_x;
        this->downstream_edm_vcs_noc_y[2] = ds_noc_y;
        this->downstream_edm_vcs_buffer_base_address[2] = adapter_spec.edm_buffer_base_addr;
        this->downstream_edm_vcs_semaphore_address[2] = adapter_spec.edm_l1_sem_addr;
        this->downstream_edm_vcs_worker_registration_address[2] = adapter_spec.edm_connection_handshake_addr;
        this->downstream_edm_vcs_worker_location_info_address[2] = adapter_spec.edm_worker_location_info_addr;
        if (config.topology != Topology::Mesh) {
            this->downstream_vcs_sender_channel_buffer_index_semaphore_id[2] = adapter_spec.buffer_index_semaphore_id;
        }
    }
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
