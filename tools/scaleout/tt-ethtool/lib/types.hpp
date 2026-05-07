// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pure data types returned and consumed by the tt-ethtool library.
//
// The library is intentionally I/O-free: all functions return structured
// results, and any progress / diagnostic messages are routed through the
// `Logger` callback bag below. The `tt-ethtool` CLI (or any other consumer)
// is responsible for turning these structures into human-readable output.

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "umd/device/types/arch.hpp"
#include "umd/device/types/cluster_descriptor_types.hpp"
#include "umd/device/types/core_coordinates.hpp"

namespace tt_ethtool {

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

// Library diagnostic sink. Defaults to no-op callbacks; the CLI installs a
// logger that prints to stdout/stderr. Library functions never write to
// stdout/stderr directly.
struct Logger {
    using Callback = std::function<void(std::string_view)>;
    Callback info = [](std::string_view) {};
    Callback warn = [](std::string_view) {};
};

// ---------------------------------------------------------------------------
// Link reference
// ---------------------------------------------------------------------------

// Identifies a single ethernet link by chip id + channel index.
struct LinkRef {
    tt::ChipId chip_id;
    std::uint32_t channel;
};

// Resolved per-link context that is useful for both diagnostics and display.
// Populated by the library and returned alongside operation-specific results.
struct LinkContext {
    tt::ARCH arch;
    tt::umd::CoreCoord noc0_coord;
};

// ---------------------------------------------------------------------------
// `list` results
// ---------------------------------------------------------------------------

enum class EthChannelState : std::uint8_t {
    ACTIVE,
    IDLE,
    HARVESTED,
};

// Peer in the same local cluster (resolved via cluster descriptor).
struct LocalPeer {
    tt::ChipId chip_id;
    std::uint32_t channel;
};

// Peer that lives in a remote cluster (only the FW-assigned unique id is
// known on this host).
struct RemotePeer {
    std::uint64_t remote_unique_id;
    std::uint32_t channel;
};

using EthPeer = std::variant<LocalPeer, RemotePeer>;

struct EthChannelInfo {
    std::uint32_t channel;
    EthChannelState state;
    // Unset when harvested or when the SoC descriptor cannot resolve the core.
    std::optional<tt::umd::CoreCoord> noc0_coord;
    // Unset when the channel has no known peer (e.g. idle, harvested,
    // dangling).
    std::optional<EthPeer> peer;
};

struct ChipEthInfo {
    tt::ChipId chip_id;
    tt::ARCH arch;
    tt::BoardType board_type;
    bool is_mmio_capable;
    // Empty when the chip is not MMIO-capable (only MMIO chips have a BDF).
    std::string pci_bdf;
    std::vector<EthChannelInfo> channels;
};

// ---------------------------------------------------------------------------
// `link up` / `link down` result
// ---------------------------------------------------------------------------

struct LinkActionResult {
    LinkContext context;
};

// ---------------------------------------------------------------------------
// `link status` result
// ---------------------------------------------------------------------------

struct WormholeLinkStatus {
    LinkContext context;
    bool link_up;
    // Raw value at ETH_TRAIN_STATUS_ADDR (matches eth_fw::wormhole::TRAIN_STATUS_*).
    std::uint32_t train_status;
    bool retrain_in_progress;
};

struct BlackholeLinkStatus {
    LinkContext context;
    bool link_up;                // Convenience: true when true_link_up == 1.
    std::uint32_t true_link_up;  // Mailbox arg0 returned by ETH_MSG_PORT_UP_CHECK.
    std::uint32_t rx_link_up;    // eth_live_status.rx_link_up.
    std::uint32_t port_status;   // Raw eth_status_t.port_status (matches eth_fw::blackhole::PORT_STATUS_*).
    std::uint32_t retrain_count;
};

using LinkStatus = std::variant<WormholeLinkStatus, BlackholeLinkStatus>;

// ---------------------------------------------------------------------------
// `link reinit` result
// ---------------------------------------------------------------------------

struct ReinitResult {
    LinkContext context;
    // Mailbox arg0 returned by FW after the reinit completes.
    std::uint32_t fw_result;
    // Echoed back so callers can re-display them without re-parsing.
    std::uint32_t reinit_option;
    std::uint32_t retries;
};

// ---------------------------------------------------------------------------
// `link stats` result
// ---------------------------------------------------------------------------

// Mirrors the eth_live_status_t struct refreshed by ETH_MSG_LINK_STATUS_CHECK.
struct EthLiveStats {
    std::uint32_t retrain_count;
    std::uint32_t rx_link_up;
    std::uint64_t frames_txd;
    std::uint64_t frames_txd_ok;
    std::uint64_t frames_txd_badfcs;
    std::uint64_t bytes_txd;
    std::uint64_t bytes_txd_ok;
    std::uint64_t bytes_txd_badfcs;
    std::uint64_t frames_rxd;
    std::uint64_t frames_rxd_ok;
    std::uint64_t frames_rxd_badfcs;
    std::uint64_t frames_rxd_dropped;
    std::uint64_t bytes_rxd;
    std::uint64_t bytes_rxd_ok;
    std::uint64_t bytes_rxd_badfcs;
    std::uint64_t bytes_rxd_dropped;
    std::uint64_t corr_cw;
    std::uint64_t uncorr_cw;
};

struct LinkStatsResult {
    LinkContext context;
    // Echo of the `copy_addr` argument passed to the FW.
    std::uint32_t copy_addr;
    EthLiveStats stats;
};

}  // namespace tt_ethtool
