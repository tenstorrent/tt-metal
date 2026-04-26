// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>
#include <string_view>

namespace tt_ethtool {

// A link reference provided on the command line, parsed from "<chip>:<channel>".
struct LinkRef {
    int chip_id;
    int channel;
};

// Parse a link reference of the form "<chip>:<channel>". Throws std::invalid_argument on bad input.
LinkRef parse_link_ref(std::string_view input);

// Enumerate all ethernet ports visible on the local system. Returns process exit code.
int run_list();

// Bring a single ethernet link up. Returns process exit code.
int run_link_up(LinkRef link);

// Bring a single ethernet link down. Returns process exit code.
int run_link_down(LinkRef link);

// Query and print the status of a single ethernet link. Returns process exit code.
int run_link_status(LinkRef link);

// Re-initialize a single ethernet link at the MAC/PCS level (and optionally SERDES).
// `reinit_option` selects the flavor (see `eth_fw.hpp::ETH_PORT_REINIT_OPT_*`).
// `retries` is the number of retry attempts the FW will perform. Returns process exit code.
int run_link_reinit(LinkRef link, unsigned int reinit_option, unsigned int retries);

}  // namespace tt_ethtool
