// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Formatting helpers used by the tt-ethtool CLI to turn library result
// structures into the human-readable text output that the binary has
// historically emitted on stdout.
//
// These functions intentionally only depend on the public library types so
// that other consumers of the library can re-use them if they want CLI-style
// rendering.

#pragma once

#include <cstdint>
#include <ostream>
#include <string_view>
#include <vector>

#include "tools/scaleout/tt-ethtool/lib/types.hpp"

namespace tt_ethtool::cli {

// String renderings of enum-like values exposed by the library.
std::string_view to_str(EthChannelState state);
std::string_view bh_port_status_to_str(std::uint32_t status);
std::string_view bh_reinit_option_to_str(std::uint32_t option);
std::string_view wh_train_status_to_str(std::uint32_t status);

// Top-level renderers for each operation's result.
void print_list(std::ostream& os, const std::vector<ChipEthInfo>& chips);

// Operations are split into a "header" and a "footer" so that the CLI can
// interleave the library's `Logger::info` diagnostics (e.g. mailbox poll
// messages) between them and preserve the historical output ordering:
//
//     <header>
//     <Logger::info messages from the operation>
//     <footer>
void print_link_action_header(std::ostream& os, std::string_view action_name, LinkRef link, const LinkContext& context);
void print_link_action_footer(std::ostream& os, std::string_view action_name, LinkRef link);

void print_link_status(std::ostream& os, LinkRef link, const LinkStatus& status);

void print_link_reinit_header(
    std::ostream& os, LinkRef link, const LinkContext& context, std::uint32_t reinit_option, std::uint32_t retries);
void print_link_reinit_footer(std::ostream& os, LinkRef link, std::uint32_t fw_result);

void print_link_stats_header(std::ostream& os, LinkRef link, const LinkContext& context, std::uint32_t copy_addr);
void print_link_stats(std::ostream& os, LinkRef link, const EthLiveStats& stats);

}  // namespace tt_ethtool::cli
