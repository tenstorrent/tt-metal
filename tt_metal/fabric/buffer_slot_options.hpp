#pragma once

#include <vector>
#include <utility>
#include <array>
#include <optional>
#include "tt_metal/fabric/types.hpp"

namespace tt::tt_fabric {

// Forward declarations
struct FabricEriscDatamoverConfig;
struct FabricEriscDatamoverOptions;

/**
 * @brief Configuration options for buffer slots across different topologies and architectures
 *
 * This struct contains static configuration arrays for buffer slot options
 * used by different fabric topologies (Linear, Ring, Mesh, Torus) across
 * different architectures (WORMHOLE_B0, BLACKHOLE).
 */
struct BufferSlotOptions {
    // Architecture-specific buffer slot configuration tables
    // Each vector contains {sender_slots, receiver_slots} pairs for different configurations

    // Default configuration with tensix extension support
    static inline const std::vector<std::vector<std::pair<size_t, size_t>>> default_with_tensix_buffer_slot_options = {
        {{16, 16}, {8, 16}, {8, 8}},  // WORMHOLE_B0: {sender_slots, receiver_slots}
        {{16, 16}, {8, 16}, {8, 8}}   // BLACKHOLE: {sender_slots, receiver_slots}
    };

    // Ring topology buffer slot configurations
    static inline const std::vector<std::vector<std::pair<size_t, size_t>>> ring_buffer_slot_options = {
        {{16, 16}, {8, 16}, {8, 8}},  // WORMHOLE_B0: {sender_slots, receiver_slots}
        {{16, 16}, {8, 16}, {8, 8}}   // BLACKHOLE: {sender_slots, receiver_slots}
    };

    // Ring topology dateline buffer slot configurations (per axis)
    static inline const std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>>
        ring_buffer_slot_options_dateline = {
            {{{8, 8}}, {{8, 8}}},  // WORMHOLE_B0: [x_axis][y_axis] = {sender_slots, receiver_slots}
            {{{8, 8}}, {{8, 8}}}   // BLACKHOLE: [x_axis][y_axis] = {sender_slots, receiver_slots}
    };

    // Ring topology dateline upstream buffer slot configurations (per axis)
    static inline const std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>>
        ring_buffer_slot_options_dateline_upstream = {
            {{{8, 8}}, {{8, 8}}},  // WORMHOLE_B0: [x_axis][y_axis] = {sender_slots, receiver_slots}
            {{{8, 8}}, {{8, 8}}}   // BLACKHOLE: [x_axis][y_axis] = {sender_slots, receiver_slots}
    };

    // Ring topology dateline upstream adjacent buffer slot configurations (per axis)
    static inline const std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>>
        ring_buffer_slot_options_dateline_upstream_adjacent = {
            {{{8, 8}}, {{8, 8}}},  // WORMHOLE_B0: [x_axis][y_axis] = {sender_slots, receiver_slots}
            {{{8, 8}}, {{8, 8}}}   // BLACKHOLE: [x_axis][y_axis] = {sender_slots, receiver_slots}
    };

    // Mesh topology buffer slot configurations
    static inline const std::vector<std::vector<std::pair<size_t, size_t>>> mesh_buffer_slot_options = {
        {{16, 16}, {8, 16}, {8, 8}},  // WORMHOLE_B0: {sender_slots, receiver_slots}
        {{16, 16}, {8, 16}, {8, 8}}   // BLACKHOLE: {sender_slots, receiver_slots}
    };

    // Torus topology buffer slot configurations
    static inline const std::vector<std::vector<std::pair<size_t, size_t>>> torus_buffer_slot_options = {
        {{16, 16}, {8, 16}, {8, 8}},  // WORMHOLE_B0: {sender_slots, receiver_slots}
        {{16, 16}, {8, 16}, {8, 8}}   // BLACKHOLE: {sender_slots, receiver_slots}
    };

    // Torus topology dateline buffer slot configurations (per axis)
    static inline const std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>>
        torus_buffer_slot_options_dateline = {
            {{{8, 8}}, {{8, 8}}},  // WORMHOLE_B0: [x_axis][y_axis] = {sender_slots, receiver_slots}
            {{{8, 8}}, {{8, 8}}}   // BLACKHOLE: [x_axis][y_axis] = {sender_slots, receiver_slots}
    };

    // Linear topology buffer slot configurations
    static inline const std::vector<std::vector<std::pair<size_t, size_t>>> linear_buffer_slot_options = {
        {{8, 16}},  // WORMHOLE_B0: {sender_slots, receiver_slots}
        {{8, 16}}   // BLACKHOLE: {sender_slots, receiver_slots}
    };
};

/**
 * @brief Buffer slot configuration helper functions
 */
namespace BufferSlotConfigHelper {
// Main configuration function
void configure_buffer_slots_helper(
    FabricEriscDatamoverConfig* config,
    Topology topology,
    const FabricEriscDatamoverOptions& options,
    std::array<size_t, 4>& num_sender_buffer_slots,
    std::array<size_t, 4>& num_remote_sender_buffer_slots,
    std::array<size_t, 4>& num_receiver_buffer_slots,
    std::array<size_t, 4>& num_remote_receiver_buffer_slots,
    eth_chan_directions direction);

// Topology-specific configuration functions
void configure_ring_buffer_slots(
    FabricEriscDatamoverConfig* config,
    const FabricEriscDatamoverOptions& options,
    std::array<size_t, 4>& num_sender_buffer_slots,
    std::array<size_t, 4>& num_remote_sender_buffer_slots,
    std::array<size_t, 4>& num_receiver_buffer_slots,
    std::array<size_t, 4>& num_remote_receiver_buffer_slots,
    size_t arch_index,
    size_t axis_index);

void configure_torus_buffer_slots(
    FabricEriscDatamoverConfig* config,
    const FabricEriscDatamoverOptions& options,
    std::array<size_t, 4>& num_sender_buffer_slots,
    std::array<size_t, 4>& num_remote_sender_buffer_slots,
    std::array<size_t, 4>& num_receiver_buffer_slots,
    std::array<size_t, 4>& num_remote_receiver_buffer_slots,
    size_t arch_index,
    size_t axis_index);

void configure_linear_buffer_slots(
    FabricEriscDatamoverConfig* config,
    const FabricEriscDatamoverOptions& options,
    std::array<size_t, 4>& num_sender_buffer_slots,
    std::array<size_t, 4>& num_remote_sender_buffer_slots,
    std::array<size_t, 4>& num_receiver_buffer_slots,
    std::array<size_t, 4>& num_remote_receiver_buffer_slots,
    size_t arch_index);

void configure_mesh_buffer_slots(
    FabricEriscDatamoverConfig* config,
    const FabricEriscDatamoverOptions& options,
    std::array<size_t, 4>& num_sender_buffer_slots,
    std::array<size_t, 4>& num_remote_sender_buffer_slots,
    std::array<size_t, 4>& num_receiver_buffer_slots,
    std::array<size_t, 4>& num_remote_receiver_buffer_slots,
    size_t arch_index);
}  // namespace BufferSlotConfigHelper

}  // namespace tt::tt_fabric
