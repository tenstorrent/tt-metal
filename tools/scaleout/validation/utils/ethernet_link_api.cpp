// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/scaleout/validation/utils/ethernet_link_api.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/distributed.hpp>
#include <llrt/tt_cluster.hpp>
#include <umd/device/cluster.hpp>

#include <chrono>
#include <set>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

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
    const auto status_mask = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::ETH_MSG_STATUS_MASK);
    const auto done_message = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::ETH_MSG_DONE);
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
    tt_metal::FWMailboxMsg msg_type,
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
    const auto call = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::ETH_MSG_CALL);
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

void send_eth_msg_to_links(const std::vector<ResetLink>& links, const BHEthMsg& eth_msg) {
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

void down_links_bh(const std::vector<ResetLink>& links_to_reset) {
    const BHEthMsg ETH_MSG_PORT_DOWN = {
        tt_metal::FWMailboxMsg::ETH_MSG_PORT_ACTION,
        {2, 0, 0},
        "Sending ETH_MSG_PORT_ACTION to bring ports down on all links"};

    // Send port down messages to all links. Ports stay down until reinitialized or the chip is reset.
    send_eth_msg_to_links(links_to_reset, ETH_MSG_PORT_DOWN);
}

void up_links_bh(const std::vector<ResetLink>& links_to_reset) {
    const BHEthMsg ETH_MSG_PORT_REINIT = {
        tt_metal::FWMailboxMsg::ETH_MSG_PORT_REINIT_MACPCS,
        {1, 2, 0},
        "Sending ETH_MSG_PORT_REINIT_MACPCS to reinitialize MAC/PCS on all links"};

    // Send port reinit messages to all links
    send_eth_msg_to_links(links_to_reset, ETH_MSG_PORT_REINIT);
}

// Shared implementation behind down_links_bh_unsafe() (all links) and down_links_bh_single_ended_unsafe()
// (one end per link). The only difference between the two is which (chip, eth channel) endpoints get the
// link-down write sequence; see the endpoint-selection block below and the two public wrappers at the end.
static void down_links_bh_unsafe_impl(bool single_ended) {
    // Build a standalone HAL purely to resolve the Blackhole ETH FW mailbox layout and message
    // encodings. This touches no device -- it is host-side architecture description only.
    // [[maybe_unused]]: only referenced by the (currently disabled) PORT_ACTION mailbox setup below.
    [[maybe_unused]] const tt::tt_metal::Hal hal(
        tt::ARCH::BLACKHOLE,
        /*is_base_routing_fw_enabled=*/false,
        /*enable_2_erisc_mode=*/false,
        /*profiler_dram_bank_size_per_risc_bytes=*/0,
        /*enable_dram_backed_cq=*/false);

    // Construct our own UMD cluster. The constructor maps the device BARs and sets up read/write
    // access, but does NOT call start_device(), so it never takes the CHIP_IN_USE mutex that gates
    // the safe path. That is exactly what lets this run while a test process holds the chip.
    tt::umd::Cluster cluster;

    // PORT_ACTION (port-down) message setup, kept here (disabled) in case we go back to it. This mirrors
    // the ETH_MSG_PORT_DOWN message used by down_links_bh(): ETH_MSG_PORT_ACTION with the "bring port down"
    // argument. To re-enable, uncomment these and the matching write_to_device calls in the loop below.
    // const auto call = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::ETH_MSG_CALL);
    // const auto msg_val = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::ETH_MSG_PORT_ACTION);
    // const auto mailbox_addr = hal.get_eth_fw_mailbox_address(0);
    // const auto first_arg_addr = hal.get_eth_fw_mailbox_arg_addr(0, 0);
    // const std::size_t mailbox_arg_count = hal.get_eth_fw_mailbox_arg_count();
    // std::vector<uint32_t> args = {2, 0, 0};
    // args.resize(mailbox_arg_count, 0);
    // const std::vector<uint32_t> msg_vec = {call | msg_val};

    // After downing the port we also forge boot_results->eth_status.train_status to look like a genuine
    // (recoverable) training failure. The port-down above is a *latched administrative down*, which the
    // FW link-recovery path deliberately refuses to override; stamping the train_status field makes the
    // link instead present as having failed training at the manual-EQ stage -- the spontaneous-failure
    // stimulus that recover_eth_link_if_down() is meant to act on. get_eth_fw_mailbox_val(TRAIN_STATUS)
    // returns the L1 address of that field (same offset on every eth core); the value is
    // link_train_status_e::LINK_TRAIN_TIMEOUT_MANUAL_EQ (index 5 of the enum in eth_fw_api.h).
    // const auto train_status_addr = hal.get_eth_fw_mailbox_val(tt_metal::FWMailboxMsg::TRAIN_STATUS);
    // constexpr uint32_t LINK_TRAIN_TIMEOUT_MANUAL_EQ = 5;
    // const std::vector<uint32_t> train_status_vec = {LINK_TRAIN_TIMEOUT_MANUAL_EQ};

    // After the PORT_ACTION + train_status writes we also clear the syseng FW PORT_DOWN record at
    // 0x7CF68 (write 0). PORT_ACTION leaves the link latched as administratively/PORT_DOWN, and the FW
    // link-recovery path refuses to run on a link in that state. Clearing this record removes the
    // administrative-down latch so recovery is no longer precluded -- i.e. it turns the deliberate
    // down into a fault recover_eth_link_if_down() is allowed to act on.
    // (Address sits in the boot_results spare region in metal's struct view; it is a live FW field.)
    // constexpr uint64_t ETH_FW_PORT_DOWN_RECORD_ADDR = 0x7CF68;
    // const std::vector<uint32_t> zero_vec = {0};

    // Finally force boot_results->eth_status.port_status back to PORT_UP (1). 0x7CC04 is the port_status
    // field (boot_results base 0x7CC00 + offsetof(eth_status_t, port_status) == +4; value 1 == PORT_UP per
    // port_status_e in eth_fw_api.h). PORT_ACTION leaves it at PORT_DOWN, which also precludes recovery;
    // overwriting it with PORT_UP clears that gate so recover_eth_link_if_down() can act on the link.
    // constexpr uint64_t ETH_FW_PORT_STATUS_ADDR = 0x7CC04;
    // constexpr uint32_t PORT_UP = 1;
    // const std::vector<uint32_t> port_up_vec = {PORT_UP};

    log_warning(
        tt::LogDistributed,
        "UNSAFE: writing ETH_MSG_PORT_ACTION (port down) directly to {} without acquiring the CHIP_IN_USE lock",
        single_ended ? "one specific ethernet link" : "all ethernet links");

    // Only local MMIO chips are reachable: without start_device() there is no ethernet routing, so we can
    // only poke chips we have a direct PCIe BAR mapping for.
    const auto mmio_chips = cluster.get_target_mmio_device_ids();

    // Build the set of (chip, eth channel) endpoints to bring down.
    std::vector<std::pair<int, uint32_t>> endpoints;
    if (!single_ended) {
        // All links: every eth channel on every local chip.
        for (auto chip_id : mmio_chips) {
            const auto& soc_desc = cluster.get_soc_descriptor(chip_id);
            const uint32_t num_eth_channels = soc_desc.get_num_eth_channels();
            for (uint32_t channel = 0; channel < num_eth_channels; channel++) {
                endpoints.emplace_back(chip_id, channel);
            }
        }
    } else {
        // Single-ended: bring down exactly ONE endpoint per link (across ALL links), leaving each link's
        // partner end UP so the FW link-recovery retrain has a live peer to handshake with.
        //
        // get_ethernet_connections() reports every link twice (once from each end), so we canonicalize by
        // link and take the first locally-reachable endpoint we encounter. For a link between two local
        // chips that picks one of the two ends; for a link whose partner is not locally reachable, the
        // local end is the only one we ever see, so it is the one chosen. get_eth_core_for_channel() in the
        // loop below takes the LOGICAL channel, which is exactly what the connections map is keyed on.
        const auto* cluster_desc = cluster.get_cluster_description();
        const auto& eth_conns = cluster_desc->get_ethernet_connections();
        std::set<std::pair<std::pair<int, int>, std::pair<int, int>>> handled_links;
        for (auto chip_id : mmio_chips) {
            auto chip_it = eth_conns.find(chip_id);
            if (chip_it == eth_conns.end()) {
                continue;  // no active links on this chip
            }
            for (const auto& [channel, remote] : chip_it->second) {
                const auto [remote_chip, remote_channel] = remote;
                const std::pair<int, int> here = {chip_id, channel};
                const std::pair<int, int> there = {remote_chip, remote_channel};
                const auto link_key = std::minmax(here, there);
                if (!handled_links.insert({link_key.first, link_key.second}).second) {
                    continue;  // partner end of this link was already chosen
                }
                endpoints.emplace_back(chip_id, static_cast<uint32_t>(channel));
            }
        }
    }

    for (const auto& [chip_id, channel] : endpoints) {
        const auto& soc_desc = cluster.get_soc_descriptor(chip_id);
        TT_FATAL(
            soc_desc.arch == tt::ARCH::BLACKHOLE,
            "down_links_bh_unsafe only supports Blackhole (chip {} arch: {})",
            chip_id,
            soc_desc.arch);
        const auto core = soc_desc.get_eth_core_for_channel(channel, CoordSystem::TRANSLATED);
        log_warning(
            tt::LogDistributed,
            "  UNSAFE port-down chip " + std::to_string(chip_id) + " channel " + std::to_string(channel));
        // PORT_ACTION args (op 1): still disabled. If re-enabled it must come before the call word below,
        // matching send_eth_msg() ordering so the FW never acts on a half-populated arg window.
        // cluster.write_to_device(args.data(), args.size() * sizeof(uint32_t), chip_id, core, first_arg_addr);
        // RMW: set bit 0 of register 0xFFBA2200, preserving the other bits. Uses the register access
        // path (read_from_device_reg/write_to_device_reg) since this is an MMIO register, not memory.
        constexpr uint64_t kReg = 0xFFBA2200;
        uint32_t reg_val = 0;
        cluster.read_from_device_reg(&reg_val, chip_id, core, kReg, sizeof(reg_val));
        reg_val |= 0x1u;
        cluster.write_to_device_reg(&reg_val, sizeof(reg_val), chip_id, core, kReg);
        // PORT_ACTION call word (op 2): disabled. Fire the mailbox message AFTER the RMW. No
        // wait-for-ready/done: this is fire-and-forget.
        // cluster.write_to_device(msg_vec.data(), msg_vec.size() * sizeof(uint32_t), chip_id, core, mailbox_addr);
        // Give the FW time to finish processing PORT_ACTION before we stamp train_status. The FW writes
        // train_status = LINK_TRAIN_REQUESTED_DOWN as part of bringing the port down; if we stamp our value
        // first it just gets clobbered. Waiting lets the FW settle so our write below is the last word.
        // std::this_thread::sleep_for(std::chrono::seconds(5));
        // Stamp train_status = LINK_TRAIN_TIMEOUT_MANUAL_EQ so the link presents as a recoverable training
        // failure (see comment above where train_status_addr is computed).
        // cluster.write_to_device(
        //     train_status_vec.data(), train_status_vec.size() * sizeof(uint32_t), chip_id, core, train_status_addr);
        // Force port_status back to PORT_UP at 0x7CC04 so the PORT_DOWN status doesn't preclude recovery
        // (see comment above).
        // cluster.write_to_device(
        //     port_up_vec.data(), port_up_vec.size() * sizeof(uint32_t), chip_id, core, ETH_FW_PORT_STATUS_ADDR);
    }
}

void down_links_bh_unsafe() { down_links_bh_unsafe_impl(/*single_ended=*/false); }

void down_links_bh_single_ended_unsafe() { down_links_bh_unsafe_impl(/*single_ended=*/true); }

void reset_links_bh(const std::vector<ResetLink>& links_to_reset) {
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    // Send port down messages to all links
    down_links_bh(links_to_reset);

    // Barrier to ensure all hosts have brought their links down before reinitialization
    distributed_context.barrier();

    // Send port reinit messages to all links
    up_links_bh(links_to_reset);
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

void dump_eth_peers_json() {
    // Build a standalone UMD cluster (no CHIP_IN_USE lock) to read the cluster descriptor.
    tt::umd::Cluster cluster;
    const auto* cluster_desc = cluster.get_cluster_description();
    const auto& eth_conns = cluster_desc->get_ethernet_connections();

    // eth_conns: chip_id -> { channel -> (remote_chip, remote_channel) }
    // Emit as a JSON array; each entry has local and remote chip/channel/NOC0 coords.
    std::cout << "{\n  \"peers\": [\n";
    bool first = true;
    for (const auto& [chip_id, channels] : eth_conns) {
        const auto& local_soc = cluster.get_soc_descriptor(chip_id);
        for (const auto& [channel, remote] : channels) {
            const auto [remote_chip, remote_channel] = remote;
            const auto& remote_soc = cluster.get_soc_descriptor(remote_chip);
            // NOC0 is the default and matches the watcher log's core(x,y) output.
            auto local_core = local_soc.get_eth_core_for_channel(channel);
            auto remote_core = remote_soc.get_eth_core_for_channel(remote_channel);
            if (!first) {
                std::cout << ",\n";
            }
            first = false;
            std::cout << "    {"
                      << "\"chip\": " << chip_id
                      << ", \"channel\": " << channel
                      << ", \"noc_x\": " << local_core.x
                      << ", \"noc_y\": " << local_core.y
                      << ", \"remote_chip\": " << remote_chip
                      << ", \"remote_channel\": " << remote_channel
                      << ", \"remote_noc_x\": " << remote_core.x
                      << ", \"remote_noc_y\": " << remote_core.y
                      << "}";
        }
    }
    std::cout << "\n  ]\n}\n";
}

}  // namespace tt::scaleout_tools
