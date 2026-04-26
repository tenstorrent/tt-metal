// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <charconv>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include <fmt/core.h>

#include "tools/scaleout/tt-ethtool/commands.hpp"

namespace {

constexpr unsigned int kDefaultReinitOption = 2;  // MAC + SERDES from reset.
constexpr unsigned int kDefaultReinitRetries = 1;

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
                 "                                 retries:       FW retry attempts (default 1)\n";
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

int dispatch_link(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "link subcommand requires: link <up|down|status|reinit> <chip>:<channel> [args]\n\n";
        print_usage();
        return EXIT_FAILURE;
    }
    const std::string_view action = argv[2];
    const tt_ethtool::LinkRef link = tt_ethtool::parse_link_ref(argv[3]);

    if (action == "up") {
        return tt_ethtool::run_link_up(link);
    }
    if (action == "down") {
        return tt_ethtool::run_link_down(link);
    }
    if (action == "status") {
        return tt_ethtool::run_link_status(link);
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
        return tt_ethtool::run_link_reinit(link, reinit_option, retries);
    }
    std::cerr << fmt::format(
        "Unknown link action: '{}' (expected 'up', 'down', 'status', or 'reinit')\n\n", action);
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
            return tt_ethtool::run_list();
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
