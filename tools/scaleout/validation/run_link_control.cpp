// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Utility to manually control the state of the ethernet links between chips on a Blackhole host.
//
// Two separate utilities live in this one executable:
//   run_link_control down          # Bring every ethernet link down and leave it down.
//   run_link_control up            # Bring every ethernet link back up (reinitialize MAC/PCS).
//   run_link_control down_unsafe   # Like "down", but does NOT acquire UMD's CHIP_IN_USE lock, so it
//                                    can run while another process (e.g. a fabric test) holds the chip.
//
// After "down"/"down_unsafe", links stay down until "up" is run (or the chip is reset, e.g. tt-smi -r).
//
// WARNING: "down_unsafe" pokes the chip from a second process concurrently with whoever already owns
// it. This is a deliberate fault-injection tool for exercising link recovery; do not use it otherwise.

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <llrt/tt_cluster.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include "tools/scaleout/validation/utils/ethernet_link_api.hpp"

namespace tt::scaleout_tools {

void set_config_vars() {
    // This tool writes custom messages to ethernet cores, which shouldn't be running fabric routers,
    // so it must be run with slow dispatch mode.
    setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", 1);

    // Only set these if they are not already set.
    if (getenv("TT_MESH_HOST_RANK") == nullptr) {
        setenv("TT_MESH_HOST_RANK", "0", 1);
    }
    if (getenv("TT_MESH_ID") == nullptr) {
        setenv("TT_MESH_ID", "0", 1);
    }
    // Disable 2-ERISC mode for Blackhole.
    if (getenv("TT_METAL_DISABLE_MULTI_AERISC") == nullptr) {
        setenv("TT_METAL_DISABLE_MULTI_AERISC", "1", 1);
    }
}

// Collect every ethernet link on every chip controlled by this host.
//
// NOTE: we enumerate eth channels from the SoC descriptor rather than from
// cluster.get_ethernet_connections(). The connection map is built by UMD topology
// discovery, which skips any link whose training status is not SUCCESS (see
// topology_discovery.cpp is_eth_trained). So once links are brought down, they
// disappear from that map and "up" would find nothing to reinitialize. The SoC
// descriptor lists every (non-harvested) eth core regardless of link/training state,
// and those cores have FW running to service the mailbox, so this works for both
// "down" (links currently up) and "up" (links currently down).
std::vector<ResetLink> collect_all_local_links() {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::vector<ResetLink> links;
    for (auto chip_id : cluster.all_chip_ids()) {
        const auto& soc_desc = cluster.get_soc_desc(chip_id);
        const uint32_t num_eth_channels = soc_desc.get_num_eth_channels();
        for (uint32_t channel = 0; channel < num_eth_channels; channel++) {
            links.push_back(
                ResetLink{chip_id, channel, "chip " + std::to_string(chip_id) + " channel " + std::to_string(channel)});
        }
    }
    return links;
}

void print_usage() {
    std::cout << "Usage:\n"
              << "  run_link_control down          # Bring all ethernet links down (they stay down)\n"
              << "  run_link_control up            # Bring all ethernet links back up (reinitialize MAC/PCS)\n"
              << "  run_link_control down_unsafe   # Bring all links down WITHOUT taking the CHIP_IN_USE lock\n"
              << "                                 # (use while a test holds the chip; racy, recovery testing only)\n"
              << "  run_link_control down_single_unsafe\n"
              << "                                 # Like down_unsafe, but downs only ONE end of each link so\n"
              << "                                 # the partner stays up for recovery's retrain handshake\n"
              << "  run_link_control dump_peers    # Dump all ethernet peer connections as JSON (stdout)\n"
              << "                                 # Runs without taking the CHIP_IN_USE lock\n";
}

}  // namespace tt::scaleout_tools

int main(int argc, char* argv[]) {
    using namespace tt::scaleout_tools;

    if (argc != 2) {
        print_usage();
        return 1;
    }
    const std::string command = argv[1];
    if (command == "-h" || command == "--help") {
        print_usage();
        return 0;
    }
    if (command != "down" && command != "up" && command != "down_unsafe" && command != "down_single_unsafe" && command != "dump_peers") {
        std::cerr << "Unknown command: " << command << "\n";
        print_usage();
        return 1;
    }

    set_config_vars();

    // The unsafe paths construct their own UMD cluster and never go through MetalContext, so they must
    // branch out BEFORE we touch MetalContext::get_cluster() (which calls start_device() and would
    // block on the CHIP_IN_USE lock held by whoever currently owns the chip).
    if (command == "down_unsafe") {
        std::cout << "Bringing all ethernet links DOWN (UNSAFE: bypassing CHIP_IN_USE lock)..." << std::endl;
        down_links_bh_unsafe();
        std::cout << "Done. Links will stay down until 'run_link_control up' or a chip reset (tt-smi -r)." << std::endl;
        return 0;
    }
    if (command == "down_single_unsafe") {
        std::cout << "Bringing ONE end of each ethernet link DOWN (UNSAFE: bypassing CHIP_IN_USE lock)..." << std::endl;
        down_links_bh_single_ended_unsafe();
        std::cout << "Done. Links will stay down until 'run_link_control up' or a chip reset (tt-smi -r)." << std::endl;
        return 0;
    }
    if (command == "dump_peers") {
        dump_eth_peers_json();
        return 0;
    }

    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    TT_FATAL(
        cluster.arch() == tt::ARCH::BLACKHOLE,
        "run_link_control only supports Blackhole (detected arch: {})",
        cluster.arch());

    const auto links = collect_all_local_links();
    std::cout << "Found " << links.size() << " ethernet link(s) on this host." << std::endl;

    if (command == "down") {
        std::cout << "Bringing all ethernet links DOWN..." << std::endl;
        down_links_bh(links);
        std::cout << "Done. Links will stay down until 'run_link_control up' or a chip reset (tt-smi -r)." << std::endl;
    } else {  // "up"
        std::cout << "Bringing all ethernet links UP..." << std::endl;
        up_links_bh(links);
        std::cout << "Done." << std::endl;
    }

    return 0;
}
