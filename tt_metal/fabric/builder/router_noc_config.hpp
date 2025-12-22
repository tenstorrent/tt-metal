// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <array>
#include <vector>
#include <cstdint>

#include "fabric_builder_config.hpp"
#include "mesh_channel_spec.hpp"

namespace tt::tt_fabric {

// Forward declaration for constants
struct FabricEriscDatamoverConfig;

/**
 * Router-wide NOC and command buffer configuration.
 * Configures NOC/CmdBuf assignments for all sender and receiver channels.
 * Groups related assignments by operation type.
 */
struct RouterNocConfig {
    // ═══════════════════════════════════════════════════════════
    // Receiver Channel Config
    // ═══════════════════════════════════════════════════════════

    // Forwarding (receiver → downstream sender)
    std::array<size_t, builder_config::num_max_receiver_channels> receiver_channel_forwarding_noc{};
    std::array<size_t, builder_config::num_max_receiver_channels> receiver_channel_forwarding_data_cmd_buf{};
    std::array<size_t, builder_config::num_max_receiver_channels> receiver_channel_forwarding_sync_cmd_buf{};

    // Local write (receiver → local L1)
    std::array<size_t, builder_config::num_max_receiver_channels> receiver_channel_local_write_noc{};
    std::array<size_t, builder_config::num_max_receiver_channels> receiver_channel_local_write_cmd_buf{};

    // ═══════════════════════════════════════════════════════════
    // Sender Channel Config
    // ═══════════════════════════════════════════════════════════

    // Ack (sender receives acks from remote)
    std::array<size_t, builder_config::num_max_sender_channels> sender_channel_ack_noc{};
    std::array<size_t, builder_config::num_max_sender_channels> sender_channel_ack_cmd_buf{};

    // ═══════════════════════════════════════════════════════════
    // Global NOC Config
    // ═══════════════════════════════════════════════════════════

    // EDM NOC virtual channel (can be modified at runtime based on link_idx)
    size_t edm_noc_vc = 2;  // Default: DEFAULT_NOC_VC from FabricEriscDatamoverConfig

    // ═══════════════════════════════════════════════════════════
    // Factory Methods
    // ═══════════════════════════════════════════════════════════

    /**
     * Create default NOC/CmdBuf assignment.
     * @param is_blackhole_single_erisc_mode Forces all operations to NOC1
     */
    static RouterNocConfig create_for_default(bool is_blackhole_single_erisc_mode);

    // ═══════════════════════════════════════════════════════════
    // Per-Operation Accessors
    // ═══════════════════════════════════════════════════════════

    struct ForwardingConfig {
        size_t noc;
        size_t data_cmd_buf;
        size_t sync_cmd_buf;
    };

    struct LocalWriteConfig {
        size_t noc;
        size_t cmd_buf;
    };

    struct AckConfig {
        size_t noc;
        size_t cmd_buf;
    };

    ForwardingConfig get_receiver_channel_forwarding(size_t ch) const {
        return {
            receiver_channel_forwarding_noc[ch],
            receiver_channel_forwarding_data_cmd_buf[ch],
            receiver_channel_forwarding_sync_cmd_buf[ch]};
    }

    LocalWriteConfig get_receiver_channel_local_write(size_t ch) const {
        return {receiver_channel_local_write_noc[ch], receiver_channel_local_write_cmd_buf[ch]};
    }

    AckConfig get_sender_channel_ack(size_t ch) const {
        return {sender_channel_ack_noc[ch], sender_channel_ack_cmd_buf[ch]};
    }

    // EDM NOC VC accessors (runtime modifiable)
    size_t get_edm_noc_vc() const { return edm_noc_vc; }
    void set_edm_noc_vc(size_t vc) { edm_noc_vc = vc; }

    // Setters for individual channel NOC assignments
    void set_receiver_channel_forwarding_noc(size_t ch, uint8_t noc_id) {
        receiver_channel_forwarding_noc[ch] = noc_id;
    }
    void set_receiver_channel_local_write_noc(size_t ch, uint8_t noc_id) {
        receiver_channel_local_write_noc[ch] = noc_id;
    }
    void set_sender_channel_ack_noc(size_t ch, uint8_t noc_id) { sender_channel_ack_noc[ch] = noc_id; }

    // ═══════════════════════════════════════════════════════════
    // CT Args Emission Helper
    // ═══════════════════════════════════════════════════════════

    /**
     * Emit NOC/CmdBuf configuration to compile-time args.
     * Emits in kernel-expected order.
     */
    void emit_ct_args(std::vector<uint32_t>& ct_args, const MeshChannelSpec& spec) const;
};

}  // namespace tt::tt_fabric
