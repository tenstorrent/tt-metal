// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <array>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"

namespace tt::tt_fabric {

struct IntermeshVCConfig;

/**
 * Defines the channel structure for a fabric router.
 * Source of truth for channel counts - other components derive from this.
 *
 * This struct encapsulates the channel configuration including:
 * - Number of virtual channels (VCs)
 * - Sender channels per VC (for transmitting data)
 * - Receiver channels per VC (for receiving data)
 * - Downstream EDMs per VC (for forwarding/routing)
 *
 * For meshes with Z routers, separate arrays track the Z router channel structure.
 */
struct MeshChannelSpec {
    // ═══════════════════════════════════════════════════════════
    // Factory Methods
    // ═══════════════════════════════════════════════════════════

    /**
     * Create spec for compute mesh router.
     *
     * Returns the MAXIMUM channel capacity for the given topology.
     * For 2D topologies, always includes both VC0 and VC1 configuration
     * (Z routers always need VC1 regardless of intermesh config).
     *
     * This is a pure capacity definition - individual router instances
     * may use fewer channels based on their runtime configuration.
     *
     * @param topology The fabric topology (Linear, Ring, Mesh, Torus, etc.)
     */
    static MeshChannelSpec create_for_compute_mesh(Topology topology);

    // ═══════════════════════════════════════════════════════════
    // Accessors
    // ═══════════════════════════════════════════════════════════

    size_t get_num_max_virtual_channels() const { return num_vcs; }

    /**
     * Check if a specific VC is active/present in this spec.
     * @param vc The VC index to check
     * @return true if the VC exists (vc < num_vcs), false otherwise
     */
    bool has_vc(uint32_t vc) const { return vc < num_vcs; }

    size_t get_num_max_sender_channels_for_vc(uint32_t vc) const {
        TT_ASSERT(vc < num_vcs, "VC {} out of bounds (max {})", vc, num_vcs);
        return sender_channels_per_vc[vc];
    }

    const std::array<size_t, builder_config::MAX_NUM_VCS>& get_max_sender_channels_per_vc() const {
        return sender_channels_per_vc;
    }

    size_t get_num_max_receiver_channels_for_vc(uint32_t vc) const {
        TT_ASSERT(vc < num_vcs, "VC {} out of bounds (max {})", vc, num_vcs);
        return receiver_channels_per_vc[vc];
    }

    const std::array<size_t, builder_config::MAX_NUM_VCS>& get_max_receiver_channels_per_vc() const {
        return receiver_channels_per_vc;
    }

    size_t get_num_max_downstream_edms_for_vc(uint32_t vc) const {
        TT_ASSERT(vc < num_vcs, "VC {} out of bounds (max {})", vc, num_vcs);
        return downstream_edms_per_vc[vc];
    }

    // Note: get_max_downstream_edms_per_vc() array accessor removed (unused)

    // Z router accessors
    size_t get_num_max_z_router_sender_channels_for_vc(uint32_t vc) const {
        TT_ASSERT(vc < num_vcs, "VC {} out of bounds (max {})", vc, num_vcs);
        return z_router_sender_channels_per_vc[vc];
    }

    // Note: get_max_z_router_sender_channels_per_vc() array accessor removed (unused)

    size_t get_num_max_z_router_receiver_channels_for_vc(uint32_t vc) const {
        TT_ASSERT(vc < num_vcs, "VC {} out of bounds (max {})", vc, num_vcs);
        return z_router_receiver_channels_per_vc[vc];
    }

    // Note: get_max_z_router_receiver_channels_per_vc() array accessor removed (unused)

    size_t get_total_sender_channels() const {
        size_t total = 0;
        for (size_t vc = 0; vc < num_vcs; ++vc) {
            total += sender_channels_per_vc[vc];
        }
        return total;
    }

    size_t get_total_receiver_channels() const {
        size_t total = 0;
        for (size_t vc = 0; vc < num_vcs; ++vc) {
            total += receiver_channels_per_vc[vc];
        }
        return total;
    }

    size_t get_total_downstream_edms() const {
        size_t total = 0;
        for (size_t vc = 0; vc < num_vcs; ++vc) {
            total += downstream_edms_per_vc[vc];
        }
        return total;
    }

    // ═══════════════════════════════════════════════════════════
    // Validation
    // ═══════════════════════════════════════════════════════════

    void validate() const {
        TT_FATAL(
            num_vcs <= builder_config::MAX_NUM_VCS, "VCs {} exceeds maximum {}", num_vcs, builder_config::MAX_NUM_VCS);
        TT_FATAL(
            get_total_sender_channels() <= builder_config::num_max_sender_channels,
            "Sender channels {} exceeds maximum {}",
            get_total_sender_channels(),
            builder_config::num_max_sender_channels);
        TT_FATAL(
            get_total_receiver_channels() <= builder_config::num_max_receiver_channels,
            "Receiver channels {} exceeds maximum {}",
            get_total_receiver_channels(),
            builder_config::num_max_receiver_channels);
        TT_FATAL(
            get_total_downstream_edms() <= builder_config::max_downstream_edms,
            "Downstream EDMs {} exceeds maximum {}",
            get_total_downstream_edms(),
            builder_config::max_downstream_edms);
    }

private:
    size_t num_vcs = 0;
    std::array<size_t, builder_config::MAX_NUM_VCS> sender_channels_per_vc = {};
    std::array<size_t, builder_config::MAX_NUM_VCS> receiver_channels_per_vc = {};
    std::array<size_t, builder_config::MAX_NUM_VCS> downstream_edms_per_vc = {};
    std::array<size_t, builder_config::MAX_NUM_VCS> z_router_sender_channels_per_vc = {};
    std::array<size_t, builder_config::MAX_NUM_VCS> z_router_receiver_channels_per_vc = {};
};

}  // namespace tt::tt_fabric
