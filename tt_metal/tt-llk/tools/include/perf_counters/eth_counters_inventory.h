// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "perf_counters/eth_counters_blackhole.h"
#include "perf_counters/eth_counters_quasar.h"
#include "perf_counters/eth_counters_types.h"
#include "perf_counters/eth_counters_wormhole.h"

namespace llk::perf::eth
{

namespace detail
{

template <std::size_t N>
constexpr QueueTable as_queue_table(const std::array<QueueEntry, N>& entries)
{
    return QueueTable {entries.data(), entries.size()};
}

template <std::size_t N>
constexpr L1Table as_l1_table(const std::array<L1Entry, N>& entries)
{
    return L1Table {entries.data(), entries.size()};
}

} // namespace detail

// The host picks the architecture at run time, so there is no ARCH_* switch here.
constexpr QueueTable tx_queue_table(Arch arch)
{
    switch (arch)
    {
        case Arch::BLACKHOLE:
            return detail::as_queue_table(blackhole::tx_queue_counters);
        case Arch::WORMHOLE:
            return detail::as_queue_table(wormhole::tx_queue_counters);
        case Arch::QUASAR:
            return detail::as_queue_table(quasar::tx_queue_counters);
    }
    return QueueTable {nullptr, 0};
}

constexpr QueueTable rx_queue_table(Arch arch)
{
    switch (arch)
    {
        case Arch::BLACKHOLE:
            return detail::as_queue_table(blackhole::rx_queue_counters);
        case Arch::WORMHOLE:
            return detail::as_queue_table(wormhole::rx_queue_counters);
        case Arch::QUASAR:
            return detail::as_queue_table(quasar::rx_queue_counters);
    }
    return QueueTable {nullptr, 0};
}

constexpr L1Table l1_table(Arch arch)
{
    switch (arch)
    {
        case Arch::BLACKHOLE:
            return detail::as_l1_table(blackhole::l1_counters);
        case Arch::WORMHOLE:
            return detail::as_l1_table(wormhole::l1_counters);
        case Arch::QUASAR:
            return detail::as_l1_table(quasar::l1_counters);
    }
    return L1Table {nullptr, 0};
}

constexpr Layout layout_for(Arch arch)
{
    switch (arch)
    {
        case Arch::BLACKHOLE:
            return blackhole::layout;
        case Arch::WORMHOLE:
            return wormhole::layout;
        case Arch::QUASAR:
            return quasar::layout;
    }
    return Layout {0, 0, 0, 0, 0, 0, 0};
}

// Register address of a queue counter: the offset moves with the queue.
constexpr std::uint32_t tx_queue_address(Arch arch, std::uint32_t queue, std::uint32_t offset)
{
    const Layout l = layout_for(arch);
    return l.txq_base + queue * l.queue_stride + offset;
}

constexpr std::uint32_t rx_queue_address(Arch arch, std::uint32_t queue, std::uint32_t offset)
{
    const Layout l = layout_for(arch);
    return l.rxq_base + queue * l.queue_stride + offset;
}

} // namespace llk::perf::eth
