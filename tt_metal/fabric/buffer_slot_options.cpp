#include "tt_metal/fabric/buffer_slot_options.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/common/stl/indestructible.hpp"
#include "tt_metal/common/metal_context.hpp"
#include "tt_metal/common/assert.hpp"

namespace tt::tt_fabric {

// Buffer slot configuration helper functions
namespace BufferSlotConfigHelper {

// Shared helper functions
namespace {

template <typename BufferSlotOptions>
void get_optimal_num_slots(
    FabricEriscDatamoverConfig* config,
    const BufferSlotOptions& buffer_slot_options,
    size_t num_sender_channels,
    size_t num_receiver_channels,
    size_t& num_sender_buffer_slots,
    size_t& num_receiver_buffer_slots,
    std::optional<size_t> worker_num_sender_buffer_slots = std::nullopt) {
    for (const auto& option : buffer_slot_options) {
        num_sender_buffer_slots = option.first;
        num_receiver_buffer_slots = option.second;
        auto num_total_sender_slots = num_sender_channels * num_sender_buffer_slots;
        auto num_total_receiver_slots = num_receiver_channels * num_receiver_buffer_slots;
        if (worker_num_sender_buffer_slots.has_value()) {
            num_total_sender_slots =
                worker_num_sender_buffer_slots.value() + (num_sender_channels - 1) * num_sender_buffer_slots;
        }
        auto total_num_bytes = (num_total_sender_slots + num_total_receiver_slots) * config->channel_buffer_size_bytes;
        if (total_num_bytes <= config->available_channel_buffering_space) {
            break;
        }
    }
}

template <typename BufferSlotArray>
void fill_sender_buffer_slots(
    FabricEriscDatamoverConfig* config,
    BufferSlotArray& num_buffer_slots,
    size_t channel_skip_idx,
    uint32_t default_num_buffer_slots,
    uint32_t extra_num_buffer_slots) {
    for (size_t i = 0; i < config->num_used_sender_channels; ++i) {
        if (i == channel_skip_idx) {
            num_buffer_slots[i] = 0;
        } else {
            // tensix worker on channel 0, otherwise extra_num_buffer_slots
            num_buffer_slots[i] = (i == 0 ? default_num_buffer_slots : extra_num_buffer_slots);
        }
    }
}

template <typename BufferSlotArray>
void fill_receiver_buffer_slots(
    FabricEriscDatamoverConfig* config,
    BufferSlotArray& num_buffer_slots,
    size_t channel_skip_idx,
    uint32_t extra_num_buffer_slots) {
    for (size_t i = 0; i < config->num_receiver_channels; ++i) {
        if (i == channel_skip_idx) {
            num_buffer_slots[i] = 0;
        } else {
            num_buffer_slots[i] = extra_num_buffer_slots;
        }
    }
}

}  // anonymous namespace

void configure_buffer_slots_helper(
    FabricEriscDatamoverConfig* config,
    Topology topology,
    const FabricEriscDatamoverOptions& options,
    std::array<size_t, 4>& num_sender_buffer_slots,
    std::array<size_t, 4>& num_remote_sender_buffer_slots,
    std::array<size_t, 4>& num_receiver_buffer_slots,
    std::array<size_t, 4>& num_remote_receiver_buffer_slots,
    eth_chan_directions direction) {
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

    switch (options.fabric_tensix_config) {
        case tt::tt_fabric::FabricTensixConfig::MUX: {
            uint32_t num_sender_channels = config->num_sender_channels_with_tensix_config;
            if (topology == tt::tt_fabric::Topology::Ring || topology == tt::tt_fabric::Topology::Torus) {
                // extra sender channel for vc1
                num_sender_channels = config->num_sender_channels_with_tensix_config_deadlock_avoidance;
            }
            uint32_t target_channel = config->get_worker_connected_sender_channel(direction, topology);
            uint32_t vc1_target_channel = config->get_worker_or_vc1_connected_sender_channel(direction, topology);
            size_t default_num_sender_buffer_slots;
            size_t default_num_receiver_buffer_slots;
            // get the default buffer slots
            get_optimal_num_slots(
                config,
                BufferSlotOptions::default_with_tensix_buffer_slot_options[arch_index],
                num_sender_channels,
                config->num_used_receiver_channels,
                default_num_sender_buffer_slots,
                default_num_receiver_buffer_slots);
            // set default buffer slots.
            num_sender_buffer_slots[target_channel] = default_num_sender_buffer_slots;
            num_sender_buffer_slots[vc1_target_channel] = default_num_sender_buffer_slots;
            num_remote_sender_buffer_slots[target_channel] = default_num_sender_buffer_slots;
            num_remote_sender_buffer_slots[vc1_target_channel] = default_num_sender_buffer_slots;
            num_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
            num_remote_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
            return;
        }
        default: break;
    }

    if (topology == Topology::Ring) {
        configure_ring_buffer_slots(
            config,
            options,
            num_sender_buffer_slots,
            num_remote_sender_buffer_slots,
            num_receiver_buffer_slots,
            num_remote_receiver_buffer_slots,
            arch_index,
            axis_index);
    } else if (topology == Topology::Torus) {
        configure_torus_buffer_slots(
            config,
            options,
            num_sender_buffer_slots,
            num_remote_sender_buffer_slots,
            num_receiver_buffer_slots,
            num_remote_receiver_buffer_slots,
            arch_index,
            axis_index);
    } else if (topology == Topology::Linear) {
        configure_linear_buffer_slots(
            config,
            options,
            num_sender_buffer_slots,
            num_remote_sender_buffer_slots,
            num_receiver_buffer_slots,
            num_remote_receiver_buffer_slots,
            arch_index);
    } else if (topology == Topology::Mesh) {
        configure_mesh_buffer_slots(
            config,
            options,
            num_sender_buffer_slots,
            num_remote_sender_buffer_slots,
            num_receiver_buffer_slots,
            num_remote_receiver_buffer_slots,
            arch_index);
    }
}

void configure_ring_buffer_slots(
    FabricEriscDatamoverConfig* config,
    const FabricEriscDatamoverOptions& options,
    std::array<size_t, 4>& num_sender_buffer_slots,
    std::array<size_t, 4>& num_remote_sender_buffer_slots,
    std::array<size_t, 4>& num_receiver_buffer_slots,
    std::array<size_t, 4>& num_remote_receiver_buffer_slots,
    size_t arch_index,
    size_t axis_index) {
    size_t default_num_sender_buffer_slots;
    size_t default_num_receiver_buffer_slots;
    // get the default buffer slots
    get_optimal_num_slots(
        config,
        BufferSlotOptions::ring_buffer_slot_options[arch_index],
        config->num_used_sender_channels,
        config->num_used_receiver_channels,
        default_num_sender_buffer_slots,
        default_num_receiver_buffer_slots);
    // get the dateline buffer slots
    size_t dateline_num_sender_buffer_slots;
    size_t dateline_num_receiver_buffer_slots;
    get_optimal_num_slots(
        config,
        BufferSlotOptions::ring_buffer_slot_options_dateline[arch_index][axis_index],
        config->num_used_sender_channels - 1,
        config->num_used_receiver_channels - 1,
        dateline_num_sender_buffer_slots,
        dateline_num_receiver_buffer_slots,
        default_num_sender_buffer_slots);
    // get the dateline upstream buffer slots
    size_t dateline_upstream_num_sender_buffer_slots;
    size_t dateline_upstream_num_receiver_buffer_slots;
    get_optimal_num_slots(
        config,
        BufferSlotOptions::ring_buffer_slot_options_dateline_upstream[arch_index][axis_index],
        config->num_used_sender_channels - 1,
        config->num_used_receiver_channels - 1,
        dateline_upstream_num_sender_buffer_slots,
        dateline_upstream_num_receiver_buffer_slots,
        default_num_sender_buffer_slots);
    // get the dateline upstream adjacent device buffer slots
    size_t dateline_upstream_adjcent_num_sender_buffer_slots;
    size_t dateline_upstream_adjcent_num_receiver_buffer_slots;
    get_optimal_num_slots(
        config,
        BufferSlotOptions::ring_buffer_slot_options_dateline_upstream_adjacent[arch_index][axis_index],
        config->num_used_sender_channels - 1,
        config->num_used_receiver_channels,
        dateline_upstream_adjcent_num_sender_buffer_slots,
        dateline_upstream_adjcent_num_receiver_buffer_slots,
        default_num_sender_buffer_slots);
    // set default buffer slots.
    num_sender_buffer_slots.fill(default_num_sender_buffer_slots);
    num_remote_sender_buffer_slots.fill(default_num_sender_buffer_slots);
    num_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
    num_remote_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);

    auto buffer_config = options.edm_buffer_config;
    switch (options.edm_type) {
        case FabricEriscDatamoverType::Dateline:
            if (buffer_config.enable_dateline_sender_extra_buffer_slots) {
                // set num_sender_buffer_slots
                fill_sender_buffer_slots(
                    config,
                    num_sender_buffer_slots,
                    config->dateline_sender_channel_skip_idx,
                    default_num_sender_buffer_slots,
                    dateline_num_sender_buffer_slots);
                // set remote sender buffer slots equal to local sender, since remote is also dateline
                num_remote_sender_buffer_slots = num_sender_buffer_slots;
            }
            if (buffer_config.enable_dateline_receiver_extra_buffer_slots) {
                // set num_receiver_buffer_slots
                fill_receiver_buffer_slots(
                    config,
                    num_receiver_buffer_slots,
                    config->dateline_receiver_channel_skip_idx,
                    dateline_num_receiver_buffer_slots);
                // set remote receiver buffer slots equal to local receiver, since remote is also dateline
                num_remote_receiver_buffer_slots = num_receiver_buffer_slots;
            }
            break;
        case FabricEriscDatamoverType::DatelineUpstream:
            if (buffer_config.enable_dateline_upstream_sender_extra_buffer_slots) {
                // set num_sender_buffer_slots
                fill_sender_buffer_slots(
                    config,
                    num_sender_buffer_slots,
                    config->dateline_upstream_sender_channel_skip_idx,
                    default_num_sender_buffer_slots,
                    dateline_upstream_num_sender_buffer_slots);
                config->skip_sender_channel_1_connection = true;
            }
            // set num_receiver_buffer_slots
            if (buffer_config.enable_dateline_upstream_receiver_extra_buffer_slots) {
                fill_receiver_buffer_slots(
                    config,
                    num_receiver_buffer_slots,
                    config->dateline_upstream_receiver_channel_skip_idx,
                    dateline_upstream_num_receiver_buffer_slots);
                config->skip_receiver_channel_1_connection = true;
            }
            if (buffer_config.enable_dateline_upstream_adjacent_sender_extra_buffer_slots) {
                // set remote sender buffer slots equal to dateline upstream dajcent sender buffer slots
                fill_sender_buffer_slots(
                    config,
                    num_remote_sender_buffer_slots,
                    config->dateline_upstream_adjcent_sender_channel_skip_idx,
                    default_num_sender_buffer_slots,
                    dateline_upstream_adjcent_num_sender_buffer_slots);
            }
            break;
        case FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice:
            if (buffer_config.enable_dateline_upstream_adjacent_sender_extra_buffer_slots) {
                // set num_sender_buffer_slots
                fill_sender_buffer_slots(
                    config,
                    num_sender_buffer_slots,
                    config->dateline_upstream_adjcent_sender_channel_skip_idx,
                    default_num_sender_buffer_slots,
                    dateline_upstream_adjcent_num_sender_buffer_slots);
                config->skip_sender_vc1_channel_connection = true;
            }
            if (buffer_config.enable_dateline_upstream_sender_extra_buffer_slots) {
                // set remote sender buffer slots equal to dateline upstream sender buffer slots
                fill_sender_buffer_slots(
                    config,
                    num_remote_sender_buffer_slots,
                    config->dateline_upstream_sender_channel_skip_idx,
                    default_num_sender_buffer_slots,
                    dateline_upstream_num_sender_buffer_slots);
            }
            if (buffer_config.enable_dateline_upstream_receiver_extra_buffer_slots) {
                // set remote sender buffer slots equal to dateline upstream sender buffer slots
                fill_receiver_buffer_slots(
                    config,
                    num_remote_receiver_buffer_slots,
                    config->dateline_upstream_receiver_channel_skip_idx,
                    dateline_upstream_num_receiver_buffer_slots);
            }
            break;
        default: break;
    }
}

void configure_torus_buffer_slots(
    FabricEriscDatamoverConfig* config,
    const FabricEriscDatamoverOptions& options,
    std::array<size_t, 4>& num_sender_buffer_slots,
    std::array<size_t, 4>& num_remote_sender_buffer_slots,
    std::array<size_t, 4>& num_receiver_buffer_slots,
    std::array<size_t, 4>& num_remote_receiver_buffer_slots,
    size_t arch_index,
    size_t axis_index) {
    // TODO: only handing default and dateline config for now, need to handle other edm types as well
    size_t default_num_sender_buffer_slots;
    size_t default_num_receiver_buffer_slots;
    // get the default buffer slots
    get_optimal_num_slots(
        config,
        BufferSlotOptions::torus_buffer_slot_options[arch_index],
        config->num_used_sender_channels,
        config->num_used_receiver_channels,
        default_num_sender_buffer_slots,
        default_num_receiver_buffer_slots);

    // get the dateline buffer slots
    size_t dateline_num_sender_buffer_slots;
    size_t dateline_num_receiver_buffer_slots;
    get_optimal_num_slots(
        config,
        BufferSlotOptions::torus_buffer_slot_options_dateline[arch_index][axis_index],
        config->num_used_sender_channels - 1,
        config->num_used_receiver_channels - 1,
        dateline_num_sender_buffer_slots,
        dateline_num_receiver_buffer_slots,
        default_num_sender_buffer_slots);

    // set default buffer slots.
    num_sender_buffer_slots.fill(default_num_sender_buffer_slots);
    num_remote_sender_buffer_slots.fill(default_num_sender_buffer_slots);
    num_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
    num_remote_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);

    auto buffer_config = options.edm_buffer_config;
    if (options.edm_type == FabricEriscDatamoverType::Dateline) {
        if (buffer_config.enable_dateline_sender_extra_buffer_slots) {
            // set num_sender_buffer_slots
            fill_sender_buffer_slots(
                config,
                num_sender_buffer_slots,
                config->dateline_sender_channel_skip_idx_2d,
                default_num_sender_buffer_slots,
                dateline_num_sender_buffer_slots);
            // set remote sender buffer slots equal to local sender, since remote is also dateline
            num_remote_sender_buffer_slots = num_sender_buffer_slots;
        }
        if (buffer_config.enable_dateline_receiver_extra_buffer_slots) {
            // set num_receiver_buffer_slots
            fill_receiver_buffer_slots(
                config,
                num_receiver_buffer_slots,
                config->dateline_receiver_channel_skip_idx,
                dateline_num_receiver_buffer_slots);
            // set remote receiver buffer slots equal to local receiver, since remote is also dateline
            num_remote_receiver_buffer_slots = num_receiver_buffer_slots;
        }
    }
}

void configure_linear_buffer_slots(
    FabricEriscDatamoverConfig* config,
    const FabricEriscDatamoverOptions& options,
    std::array<size_t, 4>& num_sender_buffer_slots,
    std::array<size_t, 4>& num_remote_sender_buffer_slots,
    std::array<size_t, 4>& num_receiver_buffer_slots,
    std::array<size_t, 4>& num_remote_receiver_buffer_slots,
    size_t arch_index) {
    size_t default_num_sender_buffer_slots;
    size_t default_num_receiver_buffer_slots;
    get_optimal_num_slots(
        config,
        BufferSlotOptions::linear_buffer_slot_options[arch_index],
        config->num_used_sender_channels,
        config->num_used_receiver_channels,
        default_num_sender_buffer_slots,
        default_num_receiver_buffer_slots);
    // set default buffer slots.
    num_sender_buffer_slots.fill(default_num_sender_buffer_slots);
    num_remote_sender_buffer_slots.fill(default_num_sender_buffer_slots);
    num_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
    num_remote_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
}

void configure_mesh_buffer_slots(
    FabricEriscDatamoverConfig* config,
    const FabricEriscDatamoverOptions& options,
    std::array<size_t, 4>& num_sender_buffer_slots,
    std::array<size_t, 4>& num_remote_sender_buffer_slots,
    std::array<size_t, 4>& num_receiver_buffer_slots,
    std::array<size_t, 4>& num_remote_receiver_buffer_slots,
    size_t arch_index) {
    size_t default_num_sender_buffer_slots;
    size_t default_num_receiver_buffer_slots;
    get_optimal_num_slots(
        config,
        BufferSlotOptions::mesh_buffer_slot_options[arch_index],
        config->num_used_sender_channels,
        config->num_used_receiver_channels,
        default_num_sender_buffer_slots,
        default_num_receiver_buffer_slots);
    // set default buffer slots.
    num_sender_buffer_slots.fill(default_num_sender_buffer_slots);
    num_remote_sender_buffer_slots.fill(default_num_sender_buffer_slots);
    num_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
    num_remote_receiver_buffer_slots.fill(default_num_receiver_buffer_slots);
}

}  // namespace BufferSlotConfigHelper

}  // namespace tt::tt_fabric
