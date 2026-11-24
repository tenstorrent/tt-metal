// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/ethernet_link_reset.hpp"

#include <chrono>
#include <set>
#include <utility>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-umd/types/core_coordinates.hpp>
#include <tt-umd/types/xy_pair.hpp>
#include <tt_stl/assert.hpp>

#include "llrt/hal.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include <llrt/tt_cluster.hpp>

namespace tt::tt_fabric {

namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::FWMailboxMsg;
using Deadline = std::chrono::steady_clock::time_point;

// Wormhole has no firmware timeout of its own, so this bounds the wait for RETRAIN_FORCE to clear. Matches the
// Blackhole reinit budget below.
constexpr std::chrono::seconds k_wh_retrain_timeout{120};

struct BHEthMsg {
    FWMailboxMsg msg_type;
    std::vector<uint32_t> msg_args;
    std::string log_message;
    // How long base firmware gets to process the message on all links before they count as failed.
    std::chrono::seconds timeout;
};

CoreCoord eth_virtual_core(const tt::Cluster& cluster, ChipId chip_id, uint32_t channel) {
    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    auto logical_coord = soc_desc.get_eth_core_for_channel(channel, CoordSystem::LOGICAL);
    return cluster.get_virtual_coordinate_from_logical_coordinates(
        chip_id, tt_xy_pair(logical_coord.x, logical_coord.y), CoreType::ETH);
}

// ============================================================================
// Wormhole (write to a specific L1 address)
// ============================================================================

std::vector<EthLinkEndpoint> retrain_links_wh(
    const tt::Cluster& cluster, const tt::tt_metal::Hal& hal, const std::vector<EthLinkEndpoint>& endpoints) {
    const tt::tt_metal::DeviceAddr eth_retrain_addr = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::RETRAIN_FORCE);
    const std::vector<uint32_t> set_reset = {1};

    log_warning(tt::LogDistributed, "Sending reset messages to all links");
    for (const auto& endpoint : endpoints) {
        log_warning(tt::LogDistributed, "  " + endpoint.log_message);
        cluster.write_core(
            endpoint.chip_id,
            eth_virtual_core(cluster, endpoint.chip_id, endpoint.channel),
            set_reset,
            eth_retrain_addr);
    }

    // Firmware clears the flag once it has processed the retrain.
    log_warning(tt::LogDistributed, "Waiting for all messages to be processed");
    const auto deadline = std::chrono::steady_clock::now() + k_wh_retrain_timeout;
    std::vector<EthLinkEndpoint> failed;
    for (const auto& endpoint : endpoints) {
        const auto core = eth_virtual_core(cluster, endpoint.chip_id, endpoint.channel);
        std::vector<uint32_t> reset_status = {0};
        while (true) {
            cluster.read_core(reset_status, sizeof(uint32_t), tt_cxy_pair(endpoint.chip_id, core), eth_retrain_addr);
            if (reset_status[0] == 0) {
                break;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                log_error(
                    tt::LogDistributed,
                    "Ethernet firmware did not clear RETRAIN_FORCE within {} s: {}",
                    k_wh_retrain_timeout.count(),
                    endpoint.log_message);
                failed.push_back(endpoint);
                break;
            }
        }
    }
    return failed;
}

// ============================================================================
// Blackhole (write to the base firmware mailbox)
// ============================================================================

// True if the base firmware mailbox status is DONE (or empty, when allow_empty is set).
bool eth_mailbox_idle(
    const tt::Cluster& cluster, const tt::tt_metal::Hal& hal, ChipId chip_id, const CoreCoord& core, bool allow_empty) {
    const auto mailbox_addr = hal.get_eth_fw_mailbox_address(0);
    const auto status_mask = hal.get_eth_fw_mailbox_val(FWMailboxMsg::ETH_MSG_STATUS_MASK);
    const auto done_message = hal.get_eth_fw_mailbox_val(FWMailboxMsg::ETH_MSG_DONE);
    std::vector<uint32_t> msg_vec = {0};
    cluster.read_core(msg_vec, sizeof(uint32_t), tt_cxy_pair(chip_id, core), mailbox_addr);
    const uint32_t msg_status = msg_vec[0] & status_mask;
    return msg_status == done_message || (allow_empty && msg_status == 0);
}

// Returns false if the mailbox is not idle by the deadline; it is always checked at least once.
bool wait_for_eth_mailbox(
    const tt::Cluster& cluster,
    const tt::tt_metal::Hal& hal,
    ChipId chip_id,
    const CoreCoord& core,
    bool allow_empty,
    Deadline deadline) {
    while (!eth_mailbox_idle(cluster, hal, chip_id, core, allow_empty)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
    }
    return true;
}

// Writes a message to an idle base firmware mailbox without waiting for it to be processed.
void post_eth_msg(
    const tt::Cluster& cluster,
    const tt::tt_metal::Hal& hal,
    ChipId chip_id,
    const CoreCoord& core,
    FWMailboxMsg msg_type,
    std::vector<uint32_t> args) {
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
            "post_eth_msg: too many mailbox args ({}) for ETH FW mailbox capacity ({})",
            args.size(),
            mailbox_arg_count);
    }
    args.resize(mailbox_arg_count, 0);

    cluster.write_core(chip_id, core, args, first_arg_addr);
    cluster.write_core(chip_id, core, msg_vec, mailbox_addr);
}

// Posts eth_msg to every endpoint, then waits for all of them under one shared timeout.
// Returns the endpoints whose firmware did not process the message in time.
std::vector<EthLinkEndpoint> send_eth_msg_to_links(
    const tt::Cluster& cluster,
    const tt::tt_metal::Hal& hal,
    const std::vector<EthLinkEndpoint>& endpoints,
    const BHEthMsg& eth_msg) {
    using EndpointCore = std::pair<const EthLinkEndpoint*, CoreCoord>;
    const auto deadline = std::chrono::steady_clock::now() + eth_msg.timeout;
    std::vector<EthLinkEndpoint> failed;
    std::vector<EndpointCore> pending;
    std::vector<EndpointCore> sent;

    log_warning(tt::LogDistributed, eth_msg.log_message);
    for (const auto& endpoint : endpoints) {
        log_warning(tt::LogDistributed, "  " + endpoint.log_message);
        pending.emplace_back(&endpoint, eth_virtual_core(cluster, endpoint.chip_id, endpoint.channel));
    }

    // Post in rounds so a mailbox that stays busy does not hold up the endpoints after it.
    while (true) {
        std::vector<EndpointCore> busy;
        for (const auto& entry : pending) {
            const auto& [endpoint, core] = entry;
            if (eth_mailbox_idle(cluster, hal, endpoint->chip_id, core, /*allow_empty=*/true)) {
                post_eth_msg(cluster, hal, endpoint->chip_id, core, eth_msg.msg_type, eth_msg.msg_args);
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
    for (const auto& [endpoint, core] : pending) {
        log_error(
            tt::LogDistributed, "Ethernet mailbox busy with an earlier message, not sent: {}", endpoint->log_message);
        failed.push_back(*endpoint);
    }

    log_warning(tt::LogDistributed, "Waiting for all messages to be processed");
    for (const auto& [endpoint, core] : sent) {
        if (!wait_for_eth_mailbox(cluster, hal, endpoint->chip_id, core, /*allow_empty=*/false, deadline)) {
            log_error(
                tt::LogDistributed,
                "Ethernet firmware did not process the message within {} s: {}",
                eth_msg.timeout.count(),
                endpoint->log_message);
            failed.push_back(*endpoint);
        }
    }
    return failed;
}

std::vector<EthLinkEndpoint> retrain_links_bh(
    const tt::Cluster& cluster,
    const tt::tt_metal::Hal& hal,
    const tt::tt_metal::distributed::multihost::DistributedContext& distributed_context,
    const std::vector<EthLinkEndpoint>& endpoints) {
    // Reinit retrains the link: ~8 s when it comes back up, ~70 s when training times out in firmware.
    const BHEthMsg eth_msg_port_reinit = {
        FWMailboxMsg::ETH_MSG_PORT_REINIT_MACPCS,
        {1, 2, 0},
        "Sending ETH_MSG_PORT_REINIT_MACPCS to reinitialize MAC/PCS on all links",
        std::chrono::seconds(120)};

    auto failed = send_port_down_msg_to_links(cluster, hal, endpoints);

    // Every host brings its ports down before any host reinitializes.
    distributed_context.barrier();

    auto reinit_failed = send_eth_msg_to_links(cluster, hal, endpoints, eth_msg_port_reinit);
    failed.insert(failed.end(), reinit_failed.begin(), reinit_failed.end());
    return failed;
}

}  // namespace

void return_links_to_base_firmware(
    const tt::Cluster& cluster, const tt::tt_metal::Hal& hal, const std::vector<EthLinkEndpoint>& endpoints) {
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
    for (const auto& endpoint : endpoints) {
        const tt_cxy_pair target(endpoint.chip_id, eth_virtual_core(cluster, endpoint.chip_id, endpoint.channel));

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
        chips.insert(endpoint.chip_id);
    }
    for (const auto chip_id : chips) {
        cluster.l1_barrier(chip_id);
    }
}

std::vector<EthLinkEndpoint> send_port_down_msg_to_links(
    const tt::Cluster& cluster, const tt::tt_metal::Hal& hal, const std::vector<EthLinkEndpoint>& endpoints) {
    TT_FATAL(cluster.arch() == tt::ARCH::BLACKHOLE, "Port-down messages are only supported on Blackhole");

    // Port down takes milliseconds, unless base firmware is stuck, e.g. retraining a link that failed to come up.
    const BHEthMsg eth_msg_port_down = {
        FWMailboxMsg::ETH_MSG_PORT_ACTION,
        {2, 0, 0},
        "Sending ETH_MSG_PORT_ACTION to bring ports down on all links",
        std::chrono::seconds(5)};
    return send_eth_msg_to_links(cluster, hal, endpoints, eth_msg_port_down);
}

std::vector<EthLinkEndpoint> retrain_ethernet_links(
    const tt::Cluster& cluster,
    const tt::tt_metal::Hal& hal,
    const tt::tt_metal::distributed::multihost::DistributedContext& distributed_context,
    const std::vector<EthLinkEndpoint>& endpoints) {
    if (cluster.arch() == tt::ARCH::WORMHOLE_B0) {
        return retrain_links_wh(cluster, hal, endpoints);
    }
    if (cluster.arch() == tt::ARCH::BLACKHOLE) {
        return retrain_links_bh(cluster, hal, distributed_context, endpoints);
    }
    TT_THROW("Unsupported cluster architecture for ethernet link reset: {}", cluster.arch());
}

}  // namespace tt::tt_fabric
