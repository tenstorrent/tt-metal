// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/scaleout/validation/utils/ethernet_link_api.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/distributed.hpp>
#include <llrt/tt_cluster.hpp>

namespace tt::scaleout_tools {

// ============================================================================
// Wormhole-specific helpers (write to specific L1 addresses)
// ============================================================================

void reset_links_wh(const std::vector<ResetLink>& links_to_reset) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();

    tt::tt_metal::DeviceAddr eth_retrain_addr = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::RETRAIN_FORCE);
    std::vector<uint32_t> set_reset = {1};

    // Send to all links to be reset
    log_warning(tt::LogDistributed, "Sending reset messages to all links");
    for (const auto& link : links_to_reset) {
        log_warning(tt::LogDistributed, "  " + link.log_message);

        const auto& soc_desc = cluster.get_soc_desc(link.chip_id);
        auto logical_coord = soc_desc.get_eth_core_for_channel(link.channel, CoordSystem::LOGICAL);
        auto coord = cluster.get_virtual_coordinate_from_logical_coordinates(
            link.chip_id, tt_xy_pair(logical_coord.x, logical_coord.y), CoreType::ETH);

        cluster.write_core(link.chip_id, coord, set_reset, eth_retrain_addr);
    }

    // Wait for FW to process
    log_warning(tt::LogDistributed, "Waiting for all messages to be processed");
    for (const auto& link : links_to_reset) {
        const auto& soc_desc = cluster.get_soc_desc(link.chip_id);
        auto logical_coord = soc_desc.get_eth_core_for_channel(link.channel, CoordSystem::LOGICAL);
        auto coord = cluster.get_virtual_coordinate_from_logical_coordinates(
            link.chip_id, tt_xy_pair(logical_coord.x, logical_coord.y), CoreType::ETH);

        // Check that reset has been processed
        std::vector<uint32_t> reset_status = {0};
        do {
            cluster.read_core(reset_status, sizeof(uint32_t), tt_cxy_pair(link.chip_id, coord), eth_retrain_addr);
        } while (reset_status[0]);
    }
}

// ============================================================================
// Blackhole-specific defines (write to mailbox)
// ============================================================================

struct BHEthMsg {
    FWMailboxMsg msg_type;
    std::vector<uint32_t> msg_args;
    std::string log_message;
};

// ============================================================================
// Blackhole-specific helpers (write to mailbox)
// ============================================================================

bool eth_mailbox_ready(ChipId chip_id, uint32_t channel, bool wait_for_ready = true) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();

    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    auto logical_coord = soc_desc.get_eth_core_for_channel(channel, CoordSystem::LOGICAL);
    auto coord = cluster.get_virtual_coordinate_from_logical_coordinates(
        chip_id, tt_xy_pair(logical_coord.x, logical_coord.y), CoreType::ETH);

    // Check mailbox is empty/ready
    const auto mailbox_addr = hal.get_eth_fw_mailbox_address(0);
    const auto status_mask = hal.get_eth_fw_mailbox_val(FWMailboxMsg::ETH_MSG_STATUS_MASK);
    const auto done_message = hal.get_eth_fw_mailbox_val(FWMailboxMsg::ETH_MSG_DONE);
    uint32_t msg_status = 0;
    std::vector<uint32_t> msg_vec = {0};
    do {
        cluster.read_core(msg_vec, sizeof(uint32_t), tt_cxy_pair(chip_id, coord), mailbox_addr);
        msg_status = msg_vec[0] & status_mask;

        if ((msg_status != done_message && msg_status != 0) && !wait_for_ready) {
            return false;
        }
    } while (msg_status != done_message && msg_status != 0);

    return true;
}

void send_eth_msg(
    ChipId chip_id,
    uint32_t channel,
    FWMailboxMsg msg_type,
    std::vector<uint32_t> args,
    bool wait_for_ready,
    bool wait_for_done) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();

    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    auto logical_coord = soc_desc.get_eth_core_for_channel(channel, CoordSystem::LOGICAL);
    auto coord = cluster.get_virtual_coordinate_from_logical_coordinates(
        chip_id, tt_xy_pair(logical_coord.x, logical_coord.y), CoreType::ETH);

    // Check mailbox is empty/ready
    if (!eth_mailbox_ready(chip_id, channel, wait_for_ready)) {
        log_warning(
            tt::LogDistributed,
            "Ethernet mailbox not ready for new message on chip " + std::to_string(chip_id) + " channel " +
                std::to_string(channel) + ". Skipping message send.");
        return;
    }

    // Write to the mailbox -> write args first in case
    // service_eth_msg picks up message call before args are fully populated
    const auto mailbox_addr = hal.get_eth_fw_mailbox_address(0);
    const auto first_arg_addr = hal.get_eth_fw_mailbox_arg_addr(0, 0);
    const auto call = hal.get_eth_fw_mailbox_val(FWMailboxMsg::ETH_MSG_CALL);
    const auto msg_val = hal.get_eth_fw_mailbox_val(msg_type);
    std::vector<uint32_t> msg_vec = {call | msg_val};

    // Ensure we always write the full mailbox arg window to avoid stale values
    const std::size_t mailbox_arg_count = hal.get_eth_fw_mailbox_arg_count();
    if (args.size() > mailbox_arg_count) {
        log_warning(
            tt::LogDistributed,
            "send_eth_msg: too many mailbox args ({}) for ETH FW mailbox capacity ({})",
            args.size(),
            mailbox_arg_count);
    }
    args.resize(mailbox_arg_count, 0);

    cluster.write_core(chip_id, coord, args, first_arg_addr);
    cluster.write_core(chip_id, coord, msg_vec, mailbox_addr);

    // Wait for the message to be serviced if requested
    if (wait_for_done) {
        // Wait for mailbox to be ready again, indicating message has been processed
        eth_mailbox_ready(chip_id, channel, true /*wait_for_ready*/);
    }
}

void send_eth_msg_to_links(const std::vector<ResetLink>& links, BHEthMsg eth_msg) {
    log_warning(tt::LogDistributed, eth_msg.log_message);
    for (const auto& link : links) {
        log_warning(tt::LogDistributed, "  " + link.log_message);
        send_eth_msg(link.chip_id, link.channel, eth_msg.msg_type, eth_msg.msg_args, true, false);
    }

    log_warning(tt::LogDistributed, "Waiting for all messages to be processed");
    for (const auto& link : links) {
        eth_mailbox_ready(link.chip_id, link.channel, true);
    }
}

void reset_links_bh(const std::vector<ResetLink>& links_to_reset) {
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    const BHEthMsg ETH_MSG_PORT_DOWN = {
        FWMailboxMsg::ETH_MSG_PORT_ACTION, {2, 0, 0}, "Sending ETH_MSG_PORT_ACTION to bring ports down on all links"};

    const BHEthMsg ETH_MSG_PORT_REINIT = {
        FWMailboxMsg::ETH_MSG_PORT_REINIT_MACPCS,
        {1, 2, 0},
        "Sending ETH_MSG_PORT_REINIT_MACPCS to reinitialize MAC/PCS on all links"};

    // Send port down messages to all links
    send_eth_msg_to_links(links_to_reset, ETH_MSG_PORT_DOWN);

    // Barrier to ensure all hosts have brought their links down before reinitialization
    distributed_context.barrier();

    // Send port reinit messages to all links
    send_eth_msg_to_links(links_to_reset, ETH_MSG_PORT_REINIT);
}

// ============================================================================
// Consolidated helpers (should be arch agnostic)
// ============================================================================

void send_reset_msg_to_links(const std::vector<ResetLink>& links_to_reset) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    if (cluster.arch() == tt::ARCH::WORMHOLE_B0) {
        reset_links_wh(links_to_reset);
    } else if (cluster.arch() == tt::ARCH::BLACKHOLE) {
        reset_links_bh(links_to_reset);
    } else {
        TT_THROW("Unsupported cluster architecture for ethernet link reset: {}", cluster.arch());
    }
}

}  // namespace tt::scaleout_tools
