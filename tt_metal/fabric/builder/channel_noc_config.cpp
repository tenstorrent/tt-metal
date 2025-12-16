// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "channel_noc_config.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"

namespace tt::tt_fabric {

RouterNocConfig RouterNocConfig::create_for_default(bool is_blackhole_single_erisc_mode) {
    RouterNocConfig config;

    if (is_blackhole_single_erisc_mode) {
        // Blackhole single-ERISC mode: everything on NOC1
        config.receiver_forwarding_noc.fill(
            FabricEriscDatamoverConfig::BLACKHOLE_SINGLE_ERISC_MODE_RECEIVER_FORWARDING_NOC);
        config.receiver_forwarding_data_cmd_buf.fill(FabricEriscDatamoverConfig::WR_CMD_BUF);
        config.receiver_forwarding_sync_cmd_buf.fill(FabricEriscDatamoverConfig::RD_CMD_BUF);
        config.receiver_local_write_noc.fill(
            FabricEriscDatamoverConfig::BLACKHOLE_SINGLE_ERISC_MODE_RECEIVER_LOCAL_WRITE_NOC);
        config.receiver_local_write_cmd_buf.fill(FabricEriscDatamoverConfig::WR_CMD_BUF);
        config.sender_ack_noc.fill(FabricEriscDatamoverConfig::BLACKHOLE_SINGLE_ERISC_MODE_SENDER_ACK_NOC);
        config.sender_ack_cmd_buf.fill(FabricEriscDatamoverConfig::WR_REG_CMD_BUF);
    } else {
        // Default: receiver on NOC1, sender ack on NOC0
        config.receiver_forwarding_noc.fill(FabricEriscDatamoverConfig::DEFAULT_RECEIVER_FORWARDING_NOC);
        config.receiver_forwarding_data_cmd_buf.fill(FabricEriscDatamoverConfig::WR_REG_CMD_BUF);
        config.receiver_forwarding_sync_cmd_buf.fill(FabricEriscDatamoverConfig::RD_CMD_BUF);
        config.receiver_local_write_noc.fill(FabricEriscDatamoverConfig::DEFAULT_RECEIVER_LOCAL_WRITE_NOC);
        config.receiver_local_write_cmd_buf.fill(FabricEriscDatamoverConfig::WR_CMD_BUF);
        config.sender_ack_noc.fill(FabricEriscDatamoverConfig::DEFAULT_SENDER_ACK_NOC);
        config.sender_ack_cmd_buf.fill(FabricEriscDatamoverConfig::WR_REG_CMD_BUF);
    }

    // edm_noc_vc uses default value (DEFAULT_NOC_VC = 2)
    // Can be modified later via set_edm_noc_vc() based on link_idx
    config.edm_noc_vc = FabricEriscDatamoverConfig::DEFAULT_NOC_VC;

    return config;
}

void RouterNocConfig::emit_ct_args(std::vector<uint32_t>& ct_args, const MeshChannelSpec& spec) const {
    size_t num_sender = spec.get_total_sender_channels();
    size_t num_receiver = spec.get_total_receiver_channels();

    // Sender ack NOC/CmdBuf
    for (size_t i = 0; i < num_sender; i++) {
        ct_args.push_back(static_cast<uint32_t>(sender_ack_noc[i]));
    }
    for (size_t i = 0; i < num_sender; i++) {
        ct_args.push_back(static_cast<uint32_t>(sender_ack_cmd_buf[i]));
    }

    // Receiver forwarding NOC/CmdBuf
    for (size_t i = 0; i < num_receiver; i++) {
        ct_args.push_back(static_cast<uint32_t>(receiver_forwarding_noc[i]));
    }
    for (size_t i = 0; i < num_receiver; i++) {
        ct_args.push_back(static_cast<uint32_t>(receiver_forwarding_data_cmd_buf[i]));
    }
    for (size_t i = 0; i < num_receiver; i++) {
        ct_args.push_back(static_cast<uint32_t>(receiver_forwarding_sync_cmd_buf[i]));
    }

    // Receiver local write NOC/CmdBuf
    for (size_t i = 0; i < num_receiver; i++) {
        ct_args.push_back(static_cast<uint32_t>(receiver_local_write_noc[i]));
    }
    for (size_t i = 0; i < num_receiver; i++) {
        ct_args.push_back(static_cast<uint32_t>(receiver_local_write_cmd_buf[i]));
    }

    // EDM NOC virtual channel
    ct_args.push_back(static_cast<uint32_t>(edm_noc_vc));
}

}  // namespace tt::tt_fabric
