// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <system_error>

#include <fmt/core.h>

#include "tools/scaleout/tt-ethtool/cli/format.hpp"
#include "tools/scaleout/tt-ethtool/lib/eth_fw.hpp"
#include "tools/scaleout/tt-ethtool/lib/operations.hpp"
#include "tools/scaleout/tt-ethtool/lib/types.hpp"
#include "umd/device/cluster.hpp"

namespace {

constexpr unsigned int kDefaultReinitOption = 2;  // MAC + SERDES from reset.
constexpr unsigned int kDefaultReinitRetries = 1;

// Logger that mirrors the original CLI's behavior of dumping mailbox poll
// status messages to stdout while operations are in flight.
tt_ethtool::Logger make_stdout_logger() {
    return tt_ethtool::Logger{
        .info = [](std::string_view msg) { std::cout << msg << "\n"; },
        .warn = [](std::string_view msg) { std::cerr << msg << "\n"; },
    };
}

void print_usage() {
    std::cout << "TT Ethernet Tool\n"
                 "Usage: tt-ethtool <command> [args]\n"
                 "Commands:\n"
                 "  help                           Show this help message\n"
                 "  list                           List all Ethernet ports visible on the local system\n"
                 "  link up      <chip>:<channel>  Bring the specified ethernet link up\n"
                 "  link down    <chip>:<channel>  Bring the specified ethernet link down\n"
                 "  link status  <chip>:<channel>  Show status of the specified ethernet link\n"
                 "  link reinit  <chip>:<channel> [<reinit-option>] [<retries>]\n"
                 "                                 Re-initialize MAC/PCS (and optionally SERDES).\n"
                 "                                 reinit-option: 0=mac-only, 1=mac+serdes-retrain,\n"
                 "                                                2=mac+serdes-reset (default),\n"
                 "                                                3=mac+serdes-reset-tx-barrier\n"
                 "                                 retries:       FW retry attempts (default 1)\n"
                 "  link stats   <chip>:<channel> [<copy-addr>]\n"
                 "                                 Refresh and print eth_live_status counters\n"
                 "                                 (MAC TX/RX MIB, FEC corrected/uncorrected\n"
                 "                                 codewords, retrain count, rx link up).\n"
                 "                                 copy-addr: optional L1 address (hex or dec) for\n"
                 "                                            the FW to copy the live status block\n"
                 "                                            to. Default 0xFFFFFFFF (no copy).\n";
}

unsigned int parse_uint(std::string_view s, std::string_view what) {
    unsigned int value = 0;
    auto* begin = s.data();
    auto* end = s.data() + s.size();
    auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument(fmt::format("{}: '{}' is not a valid unsigned integer", what, s));
    }
    return value;
}

// Parse a 32-bit unsigned int as hex (when prefixed with 0x/0X) or decimal.
std::uint32_t parse_u32(std::string_view s, std::string_view what) {
    int base = 10;
    std::string_view body = s;
    if (body.size() > 2 && body[0] == '0' && (body[1] == 'x' || body[1] == 'X')) {
        body.remove_prefix(2);
        base = 16;
    }
    std::uint32_t value = 0;
    auto* begin = body.data();
    auto* end = body.data() + body.size();
    auto result = std::from_chars(begin, end, value, base);
    if (result.ec != std::errc{} || result.ptr != end || body.empty()) {
        throw std::invalid_argument(fmt::format("{}: '{}' is not a valid 32-bit unsigned integer", what, s));
    }
    return value;
}

// Validate against the cluster descriptor before constructing the heavy
// `Cluster` (which performs full device init). All link subcommands share
// this prelude.
void validate_link_or_throw(tt_ethtool::LinkRef link) {
    tt_ethtool::validate_link_ref(*tt::umd::Cluster::create_cluster_descriptor(), link);
}

int cmd_list() {
    const auto chips = tt_ethtool::list_eth_ports();
    tt_ethtool::cli::print_list(std::cout, chips);
    return EXIT_SUCCESS;
}

int cmd_link_up_or_down(tt_ethtool::LinkRef link, std::string_view action_name) {
    validate_link_or_throw(link);
    tt::umd::Cluster cluster;
    const auto context = tt_ethtool::resolve_link_context(cluster, link);
    const auto logger = make_stdout_logger();
    tt_ethtool::cli::print_link_action_header(std::cout, action_name, link, context);
    if (action_name == "up") {
        tt_ethtool::link_up(cluster, link, logger);
    } else {
        tt_ethtool::link_down(cluster, link, logger);
    }
    tt_ethtool::cli::print_link_action_footer(std::cout, action_name, link);
    return EXIT_SUCCESS;
}

int cmd_link_status(tt_ethtool::LinkRef link) {
    validate_link_or_throw(link);
    tt::umd::Cluster cluster;
    const auto logger = make_stdout_logger();
    const auto status = tt_ethtool::get_link_status(cluster, link, logger);
    tt_ethtool::cli::print_link_status(std::cout, link, status);
    const bool link_up = std::visit([](const auto& s) { return s.link_up; }, status);
    return link_up ? EXIT_SUCCESS : EXIT_FAILURE;
}

int cmd_link_reinit(tt_ethtool::LinkRef link, std::uint32_t reinit_option, std::uint32_t retries) {
    validate_link_or_throw(link);
    tt::umd::Cluster cluster;
    const auto context = tt_ethtool::resolve_link_context(cluster, link);
    const auto logger = make_stdout_logger();
    tt_ethtool::cli::print_link_reinit_header(std::cout, link, context, reinit_option, retries);
    const auto result = tt_ethtool::link_reinit(cluster, link, reinit_option, retries, logger);
    tt_ethtool::cli::print_link_reinit_footer(std::cout, link, result.fw_result);
    return EXIT_SUCCESS;
}

int cmd_link_stats(tt_ethtool::LinkRef link, std::uint32_t copy_addr) {
    validate_link_or_throw(link);
    tt::umd::Cluster cluster;
    const auto context = tt_ethtool::resolve_link_context(cluster, link);
    const auto logger = make_stdout_logger();
    tt_ethtool::cli::print_link_stats_header(std::cout, link, context, copy_addr);
    const auto result = tt_ethtool::get_link_stats(cluster, link, copy_addr, logger);
    tt_ethtool::cli::print_link_stats(std::cout, link, result.stats);
    return EXIT_SUCCESS;
}

int dispatch_link(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "link subcommand requires: link <up|down|status|reinit|stats> <chip>:<channel> [args]\n\n";
        print_usage();
        return EXIT_FAILURE;
    }
    const std::string_view action = argv[2];
    const tt_ethtool::LinkRef link = tt_ethtool::parse_link_ref(argv[3]);

    if (action == "up" || action == "down") {
        return cmd_link_up_or_down(link, action);
    }
    if (action == "status") {
        return cmd_link_status(link);
    }
    if (action == "reinit") {
        unsigned int reinit_option = kDefaultReinitOption;
        unsigned int retries = kDefaultReinitRetries;
        if (argc >= 5) {
            reinit_option = parse_uint(argv[4], "reinit-option");
        }
        if (argc >= 6) {
            retries = parse_uint(argv[5], "retries");
        }
        if (argc > 6) {
            std::cerr << "link reinit takes at most two optional args: <reinit-option> <retries>\n\n";
            print_usage();
            return EXIT_FAILURE;
        }
        return cmd_link_reinit(link, reinit_option, retries);
    }
    if (action == "stats") {
        std::uint32_t copy_addr = tt_ethtool::eth_fw::blackhole::ETH_LIVE_STATUS_NO_COPY;
        if (argc >= 5) {
            copy_addr = parse_u32(argv[4], "copy-addr");
        }
        if (argc > 5) {
            std::cerr << "link stats takes at most one optional arg: <copy-addr>\n\n";
            print_usage();
            return EXIT_FAILURE;
        }
        return cmd_link_stats(link, copy_addr);
    }
    std::cerr << fmt::format(
        "Unknown link action: '{}' (expected 'up', 'down', 'status', 'reinit', or 'stats')\n\n", action);
    print_usage();
    return EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return EXIT_FAILURE;
    }

    const std::string_view command = argv[1];

    try {
        if (command == "help" || command == "-h" || command == "--help") {
            print_usage();
            return EXIT_SUCCESS;
        }
        if (command == "list") {
            return cmd_list();
        }
        if (command == "link") {
            return dispatch_link(argc, argv);
        }

        std::cerr << fmt::format("Unknown command: {}\n\n", command);
        print_usage();
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "tt-ethtool: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
