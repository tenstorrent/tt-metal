// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string_view>

#include <fmt/core.h>

#include "tools/scaleout/tt-ethtool/commands.hpp"

namespace {

void print_usage() {
    std::cout << "TT Ethernet Tool\n"
                 "Usage: tt-ethtool <command> [args]\n"
                 "Commands:\n"
                 "  help                          Show this help message\n"
                 "  list                          List all Ethernet ports visible on the local system\n"
                 "  link up     <chip>:<channel>  Bring the specified ethernet link up\n"
                 "  link down   <chip>:<channel>  Bring the specified ethernet link down\n"
                 "  link status <chip>:<channel>  Show status of the specified ethernet link\n";
}

int dispatch_link(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "link subcommand requires: link <up|down|status> <chip>:<channel>\n\n";
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
    std::cerr << fmt::format("Unknown link action: '{}' (expected 'up', 'down', or 'status')\n\n", action);
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
