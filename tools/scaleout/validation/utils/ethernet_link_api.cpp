// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/scaleout/validation/utils/ethernet_link_api.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/distributed.hpp>
#include <llrt/tt_cluster.hpp>

#include <chrono>
#include <set>
#include <utility>

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
    tt_metal::FWMailboxMsg msg_type;
    std::vector<uint32_t> msg_args;
    std::string log_message;
    // How long base firmware gets to process the message on all links before they count as failed.
    std::chrono::seconds timeout;
};

// ============================================================================
// Blackhole-specific helpers (write to mailbox)
// ============================================================================

namespace {

using Deadline = std::chrono::steady_clock::time_point;

CoreCoord eth_virtual_core(ChipId chip_id, uint32_t channel) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    auto logical_coord = soc_desc.get_eth_core_for_channel(channel, CoordSystem::LOGICAL);
    return cluster.get_virtual_coordinate_from_logical_coordinates(
        chip_id, tt_xy_pair(logical_coord.x, logical_coord.y), CoreType::ETH);
}

// True if the base firmware mailbox status is DONE (or empty, when allow_empty is set).
bool eth_mailbox_idle(ChipId chip_id, const CoreCoord& core, bool allow_empty) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();

    const auto mailbox_addr = hal.get_eth_fw_mailbox_address(0);
    const auto status_mask = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::ETH_MSG_STATUS_MASK);
    const auto done_message = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::ETH_MSG_DONE);
    std::vector<uint32_t> msg_vec = {0};
    cluster.read_core(msg_vec, sizeof(uint32_t), tt_cxy_pair(chip_id, core), mailbox_addr);
    const uint32_t msg_status = msg_vec[0] & status_mask;
    return msg_status == done_message || (allow_empty && msg_status == 0);
}

// Returns false if the mailbox is not idle by the deadline; it is always checked at least once.
bool wait_for_eth_mailbox(ChipId chip_id, const CoreCoord& core, bool allow_empty, Deadline deadline) {
    while (!eth_mailbox_idle(chip_id, core, allow_empty)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
    }
    return true;
}

// Writes a message to an idle base firmware mailbox without waiting for it to be processed.
void post_eth_msg(ChipId chip_id, const CoreCoord& core, tt_metal::FWMailboxMsg msg_type, std::vector<uint32_t> args) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();

    // Write to the mailbox -> write args first in case
    // service_eth_msg picks up message call before args are fully populated
    const auto mailbox_addr = hal.get_eth_fw_mailbox_address(0);
    const auto first_arg_addr = hal.get_eth_fw_mailbox_arg_addr(0, 0);
    const auto call = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::ETH_MSG_CALL);
    const auto msg_val = hal.get_eth_fw_mailbox_val(msg_type);
    std::vector<uint32_t> msg_vec = {call | msg_val};

    // Ensure we always write the full mailbox arg window to avoid stale values
    const std::size_t mailbox_arg_count = hal.get_eth_fw_mailbox_arg_count();
    if (args.size() > mailbox_arg_count) {
        log_warning(
            tt::LogDistributed,
            "post_eth_msg: too many mailbox args ({}) for ETH FW mailbox capacity ({})",
            args.size(),
            mailbox_arg_count);
    }
    args.resize(mailbox_arg_count, 0);

    cluster.write_core(chip_id, core, args, first_arg_addr);
    cluster.write_core(chip_id, core, msg_vec, mailbox_addr);
}

}  // namespace

// Posts eth_msg to every link, then waits for all of them under one shared timeout.
// Returns the links whose firmware did not process the message in time.
std::vector<ResetLink> send_eth_msg_to_links(const std::vector<ResetLink>& links, const BHEthMsg& eth_msg) {
    using LinkCore = std::pair<const ResetLink*, CoreCoord>;
    const auto deadline = std::chrono::steady_clock::now() + eth_msg.timeout;
    std::vector<ResetLink> failed_links;
    std::vector<LinkCore> pending;
    std::vector<LinkCore> sent;

    log_warning(tt::LogDistributed, eth_msg.log_message);
    for (const auto& link : links) {
        log_warning(tt::LogDistributed, "  " + link.log_message);
        pending.emplace_back(&link, eth_virtual_core(link.chip_id, link.channel));
    }

    // Post in rounds so a mailbox that stays busy does not hold up the links after it.
    while (true) {
        std::vector<LinkCore> busy;
        for (const auto& entry : pending) {
            const auto& [link, core] = entry;
            if (eth_mailbox_idle(link->chip_id, core, /*allow_empty=*/true)) {
                post_eth_msg(link->chip_id, core, eth_msg.msg_type, eth_msg.msg_args);
                sent.push_back(entry);
            } else {
                busy.push_back(entry);
            }
        }
        pending = std::move(busy);
        if (pending.empty() || std::chrono::steady_clock::now() >= deadline) {
            break;
        }
    }
    for (const auto& [link, core] : pending) {
        log_error(tt::LogDistributed, "Ethernet mailbox busy with an earlier message, not sent: {}", link->log_message);
        failed_links.push_back(*link);
    }

    log_warning(tt::LogDistributed, "Waiting for all messages to be processed");
    for (const auto& [link, core] : sent) {
        if (!wait_for_eth_mailbox(link->chip_id, core, /*allow_empty=*/false, deadline)) {
            log_error(
                tt::LogDistributed,
                "Ethernet firmware did not process the message within {} s: {}",
                eth_msg.timeout.count(),
                link->log_message);
            failed_links.push_back(*link);
        }
    }
    return failed_links;
}

bool reset_links_bh(const std::vector<ResetLink>& links_to_reset) {
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    // Reinit retrains the link: ~8 s when it comes back up, ~70 s when training times out in firmware.
    const BHEthMsg ETH_MSG_PORT_REINIT = {
        tt_metal::FWMailboxMsg::ETH_MSG_PORT_REINIT_MACPCS,
        {1, 2, 0},
        "Sending ETH_MSG_PORT_REINIT_MACPCS to reinitialize MAC/PCS on all links",
        std::chrono::seconds(120)};

    // Send port down messages to all links
    const bool ports_down = send_port_down_msg_to_links(links_to_reset).empty();

    // Barrier to ensure all hosts have brought their links down before reinitialization
    distributed_context.barrier();

    // Send port reinit messages to all links
    const bool ports_reinit = send_eth_msg_to_links(links_to_reset, ETH_MSG_PORT_REINIT).empty();
    return ports_down && ports_reinit;
}

// ============================================================================
// Consolidated helpers (should be arch agnostic)
// ============================================================================

void return_links_to_base_firmware(const std::vector<ResetLink>& links) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    TT_FATAL(cluster.arch() == tt::ARCH::BLACKHOLE, "Returning ERISCs to base firmware is only supported on Blackhole");

    namespace dev_msgs = tt::tt_metal::dev_msgs;
    constexpr auto k_core_type = tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH;
    const auto& factory = hal.get_dev_msgs_factory(k_core_type);
    const auto launch_addr = hal.get_dev_addr(k_core_type, tt::tt_metal::HalL1MemAddrType::LAUNCH);
    const auto rd_ptr_addr = hal.get_dev_addr(k_core_type, tt::tt_metal::HalL1MemAddrType::LAUNCH_MSG_BUFFER_RD_PTR);
    const auto run_flag_addr = hal.get_dev_addr(k_core_type, tt::tt_metal::HalL1MemAddrType::MAILBOX) +
                               factory.offset_of<dev_msgs::mailboxes_t>(dev_msgs::mailboxes_t::Field::aerisc_run_flag);
    const auto launch_msg_size = factory.size_of<dev_msgs::launch_msg_t>();
    auto launch_msg = factory.create<dev_msgs::launch_msg_t>();
    const uint32_t run_flag_off = 0;

    std::set<ChipId> chips;
    for (const auto& link : links) {
        const tt_cxy_pair target(link.chip_id, eth_virtual_core(link.chip_id, link.channel));

        // A running Metal kernel (e.g. a fabric router) exits once exit_erisc_kernel is set in its launch message.
        uint32_t rd_ptr = 0;
        cluster.read_reg(&rd_ptr, target, rd_ptr_addr);
        rd_ptr &= (dev_msgs::launch_msg_buffer_num_entries - 1);
        const auto launch_slot_addr = launch_addr + (rd_ptr * launch_msg_size);
        cluster.read_core(launch_msg.data(), launch_msg.size(), target, launch_slot_addr);
        launch_msg.view().kernel_config().exit_erisc_kernel() = 1;
        cluster.write_core(launch_msg.data(), launch_msg.size(), target, launch_slot_addr);

        // Idle Metal firmware hands the core back to base firmware once its run flag is cleared.
        cluster.write_reg(&run_flag_off, target, run_flag_addr);
        chips.insert(link.chip_id);
    }
    for (const auto chip_id : chips) {
        cluster.l1_barrier(chip_id);
    }
}

std::vector<ResetLink> send_port_down_msg_to_links(const std::vector<ResetLink>& links_to_reset) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    TT_FATAL(cluster.arch() == tt::ARCH::BLACKHOLE, "Port-down messages are only supported on Blackhole");

    // Port down takes milliseconds, unless base firmware is stuck, e.g. retraining a link that failed to come up.
    const BHEthMsg eth_msg_port_down = {
        tt_metal::FWMailboxMsg::ETH_MSG_PORT_ACTION,
        {2, 0, 0},
        "Sending ETH_MSG_PORT_ACTION to bring ports down on all links",
        std::chrono::seconds(5)};
    return send_eth_msg_to_links(links_to_reset, eth_msg_port_down);
}

bool send_reset_msg_to_links(const std::vector<ResetLink>& links_to_reset) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    if (cluster.arch() == tt::ARCH::WORMHOLE_B0) {
        reset_links_wh(links_to_reset);
        return true;
    }
    if (cluster.arch() == tt::ARCH::BLACKHOLE) {
        return reset_links_bh(links_to_reset);
    }
    TT_THROW("Unsupported cluster architecture for ethernet link reset: {}", cluster.arch());
}

}  // namespace tt::scaleout_tools
