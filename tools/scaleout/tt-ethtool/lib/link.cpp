// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/scaleout/tt-ethtool/lib/operations.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>

#include <fmt/core.h>

#include "tools/scaleout/tt-ethtool/lib/eth_fw.hpp"
#include "tools/scaleout/tt-ethtool/lib/types.hpp"
#include "umd/device/cluster.hpp"
#include "umd/device/cluster_descriptor.hpp"
#include "umd/device/soc_descriptor.hpp"
#include "umd/device/types/arch.hpp"
#include "umd/device/types/cluster_descriptor_types.hpp"
#include "umd/device/types/core_coordinates.hpp"

namespace tt_ethtool {

namespace {

constexpr auto kMailboxPollTimeout = std::chrono::seconds(60);
constexpr auto kRetrainPollTimeout = std::chrono::seconds(10);
constexpr auto kPollInterval = std::chrono::milliseconds(1);

// ---------- Cluster / SoC helpers ----------

tt::umd::CoreCoord get_eth_core(tt::umd::Cluster& cluster, tt::ChipId chip_id, std::uint32_t channel) {
    const auto& soc_desc = cluster.get_soc_descriptor(chip_id);
    return soc_desc.get_eth_core_for_channel(channel, tt::CoordSystem::LOGICAL);
}

std::uint32_t read_u32(tt::umd::Cluster& cluster, tt::ChipId chip_id, tt::umd::CoreCoord core, std::uint64_t addr) {
    std::uint32_t value = 0;
    cluster.read_from_device(&value, chip_id, core, addr, sizeof(value));
    return value;
}

std::uint64_t read_u64(tt::umd::Cluster& cluster, tt::ChipId chip_id, tt::umd::CoreCoord core, std::uint64_t addr) {
    // The device stores 64-bit counters as two consecutive little-endian
    // 32-bit words. Compose them on the host to keep the operation
    // independent of UMD's chunked-read alignment requirements.
    const std::uint32_t lo = read_u32(cluster, chip_id, core, addr);
    const std::uint32_t hi = read_u32(cluster, chip_id, core, addr + sizeof(std::uint32_t));
    return (static_cast<std::uint64_t>(hi) << 32) | lo;
}

void write_u32(
    tt::umd::Cluster& cluster, tt::ChipId chip_id, tt::umd::CoreCoord core, std::uint64_t addr, std::uint32_t value) {
    cluster.write_to_device(&value, sizeof(value), chip_id, core, addr);
}

// ---------- Wormhole retrain flow ----------

void wormhole_trigger_retrain(
    tt::umd::Cluster& cluster, tt::ChipId chip_id, std::uint32_t channel, tt::umd::CoreCoord core) {
    namespace wh = eth_fw::wormhole;

    write_u32(cluster, chip_id, core, wh::ETH_RETRAIN_ADDR, wh::ETH_TRIGGER_RETRAIN_VAL);

    const auto deadline = std::chrono::steady_clock::now() + kRetrainPollTimeout;
    while (read_u32(cluster, chip_id, core, wh::ETH_RETRAIN_ADDR) != 0) {
        if (std::chrono::steady_clock::now() > deadline) {
            throw std::runtime_error(fmt::format(
                "Timed out waiting for ETH FW to acknowledge retrain on chip {} channel {}", chip_id, channel));
        }
        std::this_thread::sleep_for(kPollInterval);
    }
}

// ---------- Blackhole mailbox flow ----------

// Wait for the host mailbox slot to be empty (value 0) or signal DONE.
void bh_mailbox_wait_ready(
    tt::umd::Cluster& cluster,
    tt::ChipId chip_id,
    std::uint32_t channel,
    tt::umd::CoreCoord core,
    const Logger& logger) {
    namespace bh = eth_fw::blackhole;

    const auto deadline = std::chrono::steady_clock::now() + kMailboxPollTimeout;
    while (true) {
        const std::uint32_t msg = read_u32(cluster, chip_id, core, bh::ETH_MAILBOX_HOST_MSG_ADDR);
        const std::uint32_t status = msg & bh::ETH_MSG_STATUS_MASK;
        if (status == 0) {
            logger.info(
                fmt::format("ETH FW mailbox (status=0x{:08x}) on chip {} channel {} is empty", msg, chip_id, channel));
            return;
        }
        if (status == bh::ETH_MSG_DONE) {
            logger.info(
                fmt::format("ETH FW mailbox (status=0x{:08x}) on chip {} channel {} is done", msg, chip_id, channel));
            return;
        }
        if (std::chrono::steady_clock::now() > deadline) {
            throw std::runtime_error(fmt::format(
                "Timed out waiting for ETH FW mailbox (status=0x{:08x}) on chip {} channel {}", msg, chip_id, channel));
        }
        std::this_thread::sleep_for(kPollInterval);
    }
}

void bh_send_mailbox_msg(
    tt::umd::Cluster& cluster,
    tt::ChipId chip_id,
    std::uint32_t channel,
    tt::umd::CoreCoord core,
    std::uint32_t msg_type,
    std::array<std::uint32_t, eth_fw::blackhole::ETH_MAILBOX_NUM_ARGS> args,
    bool wait_for_done,
    const Logger& logger) {
    namespace bh = eth_fw::blackhole;

    bh_mailbox_wait_ready(cluster, chip_id, channel, core, logger);

    // Write args first, then the CALL word last to avoid the FW observing a
    // partially populated command.
    for (std::uint32_t i = 0; i < bh::ETH_MAILBOX_NUM_ARGS; ++i) {
        const std::uint64_t addr = bh::ETH_MAILBOX_HOST_ARG0_ADDR + i * sizeof(std::uint32_t);
        write_u32(cluster, chip_id, core, addr, args[i]);
    }

    const std::uint32_t cmd = bh::ETH_MSG_CALL | msg_type;
    write_u32(cluster, chip_id, core, bh::ETH_MAILBOX_HOST_MSG_ADDR, cmd);

    if (wait_for_done) {
        bh_mailbox_wait_ready(cluster, chip_id, channel, core, logger);
    }
}

std::array<std::uint32_t, eth_fw::blackhole::ETH_MAILBOX_NUM_ARGS> bh_read_mailbox_args(
    tt::umd::Cluster& cluster, tt::ChipId chip_id, tt::umd::CoreCoord core) {
    namespace bh = eth_fw::blackhole;
    std::array<std::uint32_t, bh::ETH_MAILBOX_NUM_ARGS> args{};
    for (std::uint32_t i = 0; i < bh::ETH_MAILBOX_NUM_ARGS; ++i) {
        const std::uint64_t addr = bh::ETH_MAILBOX_HOST_ARG0_ADDR + i * sizeof(std::uint32_t);
        args[i] = read_u32(cluster, chip_id, core, addr);
    }
    return args;
}

// ---------- Link up / down core ----------

enum class LinkAction {
    UP,
    DOWN,
};

LinkActionResult run_link_action(tt::umd::Cluster& cluster, LinkRef link, LinkAction action, const Logger& logger) {
    const LinkContext ctx = resolve_link_context(cluster, link);

    switch (ctx.arch) {
        case tt::ARCH::WORMHOLE_B0: {
            if (action == LinkAction::DOWN) {
                throw std::runtime_error(
                    "Wormhole ETH FW does not expose a runtime link-down; only 'link up' (retrain) is supported.");
            }
            wormhole_trigger_retrain(cluster, link.chip_id, link.channel, ctx.noc0_coord);
            break;
        }
        case tt::ARCH::BLACKHOLE: {
            namespace bh = eth_fw::blackhole;
            const std::uint32_t arg0 =
                (action == LinkAction::UP) ? bh::ETH_PORT_ACTION_LINK_UP : bh::ETH_PORT_ACTION_LINK_DOWN;
            bh_send_mailbox_msg(
                cluster,
                link.chip_id,
                link.channel,
                ctx.noc0_coord,
                bh::ETH_MSG_TYPE_PORT_ACTION,
                {arg0, 0, 0},
                /*wait_for_done=*/true,
                logger);
            break;
        }
        default:
            throw std::runtime_error(
                fmt::format("Unsupported architecture for link control: {}", tt::arch_to_str(ctx.arch)));
    }

    return LinkActionResult{.context = ctx};
}

// ---------- Validation against an existing cluster ----------

void validate_link_against_cluster(tt::umd::Cluster& cluster, LinkRef link) {
    validate_link_ref(*cluster.get_cluster_description(), link);
}

}  // namespace

LinkContext resolve_link_context(tt::umd::Cluster& cluster, LinkRef link) {
    const tt::ARCH arch = cluster.get_cluster_description()->get_arch(link.chip_id);
    const tt::umd::CoreCoord core = get_eth_core(cluster, link.chip_id, link.channel);
    return LinkContext{.arch = arch, .noc0_coord = core};
}

// ---------------------------------------------------------------------------
// Link up / down
// ---------------------------------------------------------------------------

LinkActionResult link_up(tt::umd::Cluster& cluster, LinkRef link, const Logger& logger) {
    validate_link_against_cluster(cluster, link);
    return run_link_action(cluster, link, LinkAction::UP, logger);
}

LinkActionResult link_down(tt::umd::Cluster& cluster, LinkRef link, const Logger& logger) {
    validate_link_against_cluster(cluster, link);
    return run_link_action(cluster, link, LinkAction::DOWN, logger);
}

LinkActionResult link_up(LinkRef link, const Logger& logger) {
    validate_link_ref(*tt::umd::Cluster::create_cluster_descriptor(), link);
    tt::umd::Cluster cluster;
    return run_link_action(cluster, link, LinkAction::UP, logger);
}

LinkActionResult link_down(LinkRef link, const Logger& logger) {
    validate_link_ref(*tt::umd::Cluster::create_cluster_descriptor(), link);
    tt::umd::Cluster cluster;
    return run_link_action(cluster, link, LinkAction::DOWN, logger);
}

// ---------------------------------------------------------------------------
// Link status
// ---------------------------------------------------------------------------

LinkStatus get_link_status(tt::umd::Cluster& cluster, LinkRef link, const Logger& logger) {
    validate_link_against_cluster(cluster, link);
    const LinkContext ctx = resolve_link_context(cluster, link);

    switch (ctx.arch) {
        case tt::ARCH::WORMHOLE_B0: {
            namespace wh = eth_fw::wormhole;
            const std::uint32_t retrain = read_u32(cluster, link.chip_id, ctx.noc0_coord, wh::ETH_RETRAIN_ADDR);
            const bool retrain_in_progress = (retrain == wh::ETH_TRIGGER_RETRAIN_VAL);
            const std::uint32_t train_status =
                read_u32(cluster, link.chip_id, ctx.noc0_coord, wh::ETH_TRAIN_STATUS_ADDR);
            const bool link_up_b = !retrain_in_progress && train_status == wh::TRAIN_STATUS_SUCCESS;
            return WormholeLinkStatus{
                .context = ctx,
                .link_up = link_up_b,
                .train_status = train_status,
                .retrain_in_progress = retrain_in_progress,
            };
        }
        case tt::ARCH::BLACKHOLE: {
            namespace bh = eth_fw::blackhole;
            // ETH_MSG_PORT_UP_CHECK: no input arguments. FW performs a fast
            // link-up check (result written to eth_live_status.rx_link_up) and
            // a true/full link-up check (returned via arg0).
            bh_send_mailbox_msg(
                cluster,
                link.chip_id,
                link.channel,
                ctx.noc0_coord,
                bh::ETH_MSG_TYPE_PORT_UP_CHECK,
                {0, 0, 0},
                /*wait_for_done=*/true,
                logger);

            const auto args = bh_read_mailbox_args(cluster, link.chip_id, ctx.noc0_coord);
            const std::uint32_t true_link_up = args[0];
            const std::uint32_t rx_link_up = read_u32(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_RX_LINK_UP_ADDR);
            const std::uint32_t port_status = read_u32(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_PORT_STATUS_ADDR);
            const std::uint32_t retrain_count =
                read_u32(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_RETRAIN_COUNT_ADDR);
            return BlackholeLinkStatus{
                .context = ctx,
                .link_up = (true_link_up == 1),
                .true_link_up = true_link_up,
                .rx_link_up = rx_link_up,
                .port_status = port_status,
                .retrain_count = retrain_count,
            };
        }
        default:
            throw std::runtime_error(
                fmt::format("Unsupported architecture for link status: {}", tt::arch_to_str(ctx.arch)));
    }
}

LinkStatus get_link_status(LinkRef link, const Logger& logger) {
    validate_link_ref(*tt::umd::Cluster::create_cluster_descriptor(), link);
    tt::umd::Cluster cluster;
    return get_link_status(cluster, link, logger);
}

// ---------------------------------------------------------------------------
// Link reinit
// ---------------------------------------------------------------------------

ReinitResult link_reinit(
    tt::umd::Cluster& cluster, LinkRef link, std::uint32_t reinit_option, std::uint32_t retries, const Logger& logger) {
    namespace bh = eth_fw::blackhole;
    if (reinit_option > bh::ETH_PORT_REINIT_OPT_MAC_SERDES_TX_BARRIER) {
        throw std::invalid_argument(fmt::format(
            "Invalid reinit option {}; expected 0..{}.", reinit_option, bh::ETH_PORT_REINIT_OPT_MAC_SERDES_TX_BARRIER));
    }
    validate_link_against_cluster(cluster, link);

    const LinkContext ctx = resolve_link_context(cluster, link);
    if (ctx.arch != tt::ARCH::BLACKHOLE) {
        throw std::runtime_error(fmt::format(
            "Unsupported architecture for link reinit: {}. Only Blackhole exposes "
            "ETH_MSG_PORT_REINIT_MACPCS; use 'link up' on Wormhole to trigger a retrain.",
            tt::arch_to_str(ctx.arch)));
    }

    bh_send_mailbox_msg(
        cluster,
        link.chip_id,
        link.channel,
        ctx.noc0_coord,
        bh::ETH_MSG_TYPE_PORT_REINIT_MACPCS,
        {retries, reinit_option, 0},
        /*wait_for_done=*/true,
        logger);

    const auto args = bh_read_mailbox_args(cluster, link.chip_id, ctx.noc0_coord);
    return ReinitResult{
        .context = ctx,
        .fw_result = args[0],
        .reinit_option = reinit_option,
        .retries = retries,
    };
}

ReinitResult link_reinit(LinkRef link, std::uint32_t reinit_option, std::uint32_t retries, const Logger& logger) {
    validate_link_ref(*tt::umd::Cluster::create_cluster_descriptor(), link);
    tt::umd::Cluster cluster;
    return link_reinit(cluster, link, reinit_option, retries, logger);
}

// ---------------------------------------------------------------------------
// Link stats
// ---------------------------------------------------------------------------

LinkStatsResult get_link_stats(tt::umd::Cluster& cluster, LinkRef link, std::uint32_t copy_addr, const Logger& logger) {
    validate_link_against_cluster(cluster, link);
    const LinkContext ctx = resolve_link_context(cluster, link);

    if (ctx.arch != tt::ARCH::BLACKHOLE) {
        throw std::runtime_error(fmt::format(
            "Unsupported architecture for link stats: {}. Only Blackhole exposes "
            "ETH_MSG_LINK_STATUS_CHECK with a snapshotting eth_live_status block.",
            tt::arch_to_str(ctx.arch)));
    }

    namespace bh = eth_fw::blackhole;

    // Trigger the FW to refresh boot_results.eth_live_status. arg0 doubles as
    // the destination L1 address for an optional copy of the live status
    // block; ETH_LIVE_STATUS_NO_COPY (0xFFFFFFFF) skips that copy.
    bh_send_mailbox_msg(
        cluster,
        link.chip_id,
        link.channel,
        ctx.noc0_coord,
        bh::ETH_MSG_TYPE_LINK_STATUS_CHECK,
        {copy_addr, 0, 0},
        /*wait_for_done=*/true,
        logger);

    EthLiveStats stats{};
    stats.retrain_count = read_u32(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_RETRAIN_COUNT_ADDR);
    stats.rx_link_up = read_u32(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_RX_LINK_UP_ADDR);
    stats.frames_txd = read_u64(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_LIVE_STATUS_FRAMES_TXD_ADDR);
    stats.frames_txd_ok = read_u64(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_LIVE_STATUS_FRAMES_TXD_OK_ADDR);
    stats.frames_txd_badfcs =
        read_u64(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_LIVE_STATUS_FRAMES_TXD_BADFCS_ADDR);
    stats.bytes_txd = read_u64(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_LIVE_STATUS_BYTES_TXD_ADDR);
    stats.bytes_txd_ok = read_u64(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_LIVE_STATUS_BYTES_TXD_OK_ADDR);
    stats.bytes_txd_badfcs = read_u64(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_LIVE_STATUS_BYTES_TXD_BADFCS_ADDR);
    stats.frames_rxd = read_u64(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_LIVE_STATUS_FRAMES_RXD_ADDR);
    stats.frames_rxd_ok = read_u64(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_LIVE_STATUS_FRAMES_RXD_OK_ADDR);
    stats.frames_rxd_badfcs =
        read_u64(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_LIVE_STATUS_FRAMES_RXD_BADFCS_ADDR);
    stats.frames_rxd_dropped =
        read_u64(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_LIVE_STATUS_FRAMES_RXD_DROPPED_ADDR);
    stats.bytes_rxd = read_u64(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_LIVE_STATUS_BYTES_RXD_ADDR);
    stats.bytes_rxd_ok = read_u64(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_LIVE_STATUS_BYTES_RXD_OK_ADDR);
    stats.bytes_rxd_badfcs = read_u64(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_LIVE_STATUS_BYTES_RXD_BADFCS_ADDR);
    stats.bytes_rxd_dropped =
        read_u64(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_LIVE_STATUS_BYTES_RXD_DROPPED_ADDR);
    stats.corr_cw = read_u64(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_LIVE_STATUS_CORR_CW_ADDR);
    stats.uncorr_cw = read_u64(cluster, link.chip_id, ctx.noc0_coord, bh::ETH_LIVE_STATUS_UNCORR_CW_ADDR);

    return LinkStatsResult{.context = ctx, .copy_addr = copy_addr, .stats = stats};
}

LinkStatsResult get_link_stats(LinkRef link, std::uint32_t copy_addr, const Logger& logger) {
    validate_link_ref(*tt::umd::Cluster::create_cluster_descriptor(), link);
    tt::umd::Cluster cluster;
    return get_link_stats(cluster, link, copy_addr, logger);
}

}  // namespace tt_ethtool
