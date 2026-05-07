// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Public API of the tt-ethtool library.
//
// All operations are pure C++ functions that return structured results and
// throw `std::invalid_argument` / `std::runtime_error` on failure. They never
// touch stdout/stderr; route diagnostics through the optional `Logger` arg.
//
// Two flavors of each link operation are provided:
//   * Low-level: takes a `tt::umd::Cluster&` supplied by the caller. Use this
//     when the caller is already managing a cluster and wants to perform
//     several operations against it.
//   * Convenience: constructs a fresh `tt::umd::Cluster` internally. Mirrors
//     the original CLI behavior and is what the `tt-ethtool` binary calls.
//
// `list_eth_ports` only needs a `tt::umd::ClusterDescriptor`, which is much
// cheaper to obtain than a full `Cluster`.

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "tools/scaleout/tt-ethtool/lib/eth_fw.hpp"
#include "tools/scaleout/tt-ethtool/lib/types.hpp"

namespace tt {
namespace umd {
class Cluster;
class ClusterDescriptor;
}  // namespace umd
}  // namespace tt

namespace tt_ethtool {

// ---------------------------------------------------------------------------
// Parsing & validation
// ---------------------------------------------------------------------------

// Parse a link reference of the form "<chip>:<channel>". Throws
// `std::invalid_argument` on malformed input.
LinkRef parse_link_ref(std::string_view input);

// Validate that the link reference points at a real chip + channel on the
// current host, given the cluster descriptor. Throws `std::invalid_argument`
// if the chip is unknown or the channel is out of range for that chip's arch.
//
// Note: takes a non-const reference because the underlying UMD accessors used
// to compute valid channel ranges are not (yet) marked const.
void validate_link_ref(tt::umd::ClusterDescriptor& desc, LinkRef link);

// Resolve the architecture and NOC0 coordinate of an ethernet link. Useful
// for printing diagnostic headers before invoking a longer-running operation.
LinkContext resolve_link_context(tt::umd::Cluster& cluster, LinkRef link);

// ---------------------------------------------------------------------------
// `list`
// ---------------------------------------------------------------------------

// Enumerate ethernet ports visible on the local system, sorted by chip id.
// Uses only the cluster descriptor (no full device init).
//
// Note: takes a non-const reference because the underlying UMD accessors used
// internally are not (yet) marked const.
std::vector<ChipEthInfo> list_eth_ports(tt::umd::ClusterDescriptor& desc);

// Convenience overload: builds the cluster descriptor internally.
std::vector<ChipEthInfo> list_eth_ports();

// ---------------------------------------------------------------------------
// Link control: up / down
// ---------------------------------------------------------------------------

// Bring a link up.
//   * Wormhole: triggers a SerDes retrain via ETH_RETRAIN_ADDR.
//   * Blackhole: sends ETH_MSG_PORT_ACTION(ETH_PORT_ACTION_LINK_UP).
// Throws `std::runtime_error` on FW timeout or unsupported architecture.
LinkActionResult link_up(tt::umd::Cluster& cluster, LinkRef link, const Logger& logger = {});
LinkActionResult link_up(LinkRef link, const Logger& logger = {});

// Bring a link down. Only supported on Blackhole; throws on Wormhole.
LinkActionResult link_down(tt::umd::Cluster& cluster, LinkRef link, const Logger& logger = {});
LinkActionResult link_down(LinkRef link, const Logger& logger = {});

// ---------------------------------------------------------------------------
// Link status
// ---------------------------------------------------------------------------

// Read link state. Returns an arch-specific variant.
LinkStatus get_link_status(tt::umd::Cluster& cluster, LinkRef link, const Logger& logger = {});
LinkStatus get_link_status(LinkRef link, const Logger& logger = {});

// ---------------------------------------------------------------------------
// Link reinit (Blackhole only)
// ---------------------------------------------------------------------------

// Re-initialize MAC/PCS (and optionally SerDes) on a Blackhole link.
// `reinit_option` selects the flavor (see eth_fw::blackhole::ETH_PORT_REINIT_OPT_*);
// `retries` is the FW retry budget. Throws on unsupported architecture or
// invalid `reinit_option`.
ReinitResult link_reinit(
    tt::umd::Cluster& cluster,
    LinkRef link,
    std::uint32_t reinit_option,
    std::uint32_t retries,
    const Logger& logger = {});
ReinitResult link_reinit(LinkRef link, std::uint32_t reinit_option, std::uint32_t retries, const Logger& logger = {});

// ---------------------------------------------------------------------------
// Link stats (Blackhole only)
// ---------------------------------------------------------------------------

// Trigger ETH_MSG_LINK_STATUS_CHECK on the specified link to refresh the
// boot_results.eth_live_status counters in place, then read them back.
// Pass `eth_fw::blackhole::ETH_LIVE_STATUS_NO_COPY` to skip the optional
// FW-side DMA copy of the live status block.
LinkStatsResult get_link_stats(
    tt::umd::Cluster& cluster,
    LinkRef link,
    std::uint32_t copy_addr = eth_fw::blackhole::ETH_LIVE_STATUS_NO_COPY,
    const Logger& logger = {});
LinkStatsResult get_link_stats(
    LinkRef link, std::uint32_t copy_addr = eth_fw::blackhole::ETH_LIVE_STATUS_NO_COPY, const Logger& logger = {});

}  // namespace tt_ethtool
