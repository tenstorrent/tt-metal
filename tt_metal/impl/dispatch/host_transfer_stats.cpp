// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "host_transfer_stats.hpp"

#include <algorithm>
#include <tt_stl/fmt.hpp>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal {

namespace {

constexpr std::array<const char*, 8> kLabels = {
    "buffer writes, fast dispatch",
    "buffer writes, slow dispatch",
    "buffer reads, fast dispatch",
    "buffer reads, slow dispatch",
    "core writes, fast dispatch",
    "core reads, fast dispatch",
    "H2D socket writes, host push",
    "H2D socket writes, device pull",
};
static_assert(kLabels.size() == static_cast<size_t>(HostTransfer::SocketPull) + 1);

std::string format_bytes(uint64_t bytes) {
    const auto b = static_cast<double>(bytes);
    if (b >= 1e9) {
        return fmt::format("{:.2f} GB", b / 1e9);
    }
    if (b >= 1e6) {
        return fmt::format("{:.2f} MB", b / 1e6);
    }
    if (b >= 1e3) {
        return fmt::format("{:.2f} KB", b / 1e3);
    }
    return fmt::format("{} B", bytes);
}

}  // namespace

void HostTransferStats::begin(HostTransfer kind) {
    auto& s = streams_[static_cast<size_t>(kind)];
    std::lock_guard lk(s.mutex);
    ++s.in_flight;
}

void HostTransferStats::end(HostTransfer kind, Clock::time_point start, Clock::time_point end, uint64_t bytes) {
    auto& s = streams_[static_cast<size_t>(kind)];
    std::lock_guard lk(s.mutex);
    if (s.in_flight == 0) {
        return;
    }
    if (bytes > 0) {
        s.bytes += bytes;
        ++s.transfers;
        s.stretch_start = std::min(s.stretch_start, start);
        s.stretch_end = std::max(s.stretch_end, end);
    }
    if (--s.in_flight == 0) {
        if (s.stretch_start <= s.stretch_end) {
            s.busy += s.stretch_end - s.stretch_start;
        }
        s.stretch_start = Clock::time_point::max();
        s.stretch_end = Clock::time_point::min();
    }
}

void HostTransferStats::add_bytes(HostTransfer kind, uint64_t bytes) {
    auto& s = streams_[static_cast<size_t>(kind)];
    std::lock_guard lk(s.mutex);
    s.bytes += bytes;
    ++s.transfers;
}

HostTransferStats::Totals HostTransferStats::totals(HostTransfer kind) const {
    const auto& s = streams_[static_cast<size_t>(kind)];
    std::lock_guard lk(s.mutex);
    Totals totals{.bytes = s.bytes, .transfers = s.transfers, .busy = s.busy};
    if (s.stretch_start <= s.stretch_end) {
        totals.busy += s.stretch_end - s.stretch_start;
    }
    return totals;
}

std::vector<std::pair<std::string, std::string>> HostTransferStats::summary_rows() const {
    std::vector<std::pair<std::string, std::string>> rows;
    for (size_t i = 0; i < streams_.size(); ++i) {
        const auto kind = static_cast<HostTransfer>(i);
        const Totals t = totals(kind);
        if (t.transfers == 0) {
            continue;
        }
        const double seconds = std::chrono::duration<double>(t.busy).count();
        std::string value;
        if (kind == HostTransfer::SocketPull) {
            value = fmt::format("{} staged, {} transfers", format_bytes(t.bytes), t.transfers);
        } else if (seconds > 0) {
            value = fmt::format(
                "{} in {:.3f} s ({:.2f} GB/s), {} transfers",
                format_bytes(t.bytes),
                seconds,
                static_cast<double>(t.bytes) / 1e9 / seconds,
                t.transfers);
        } else {
            value = fmt::format("{}, {} transfers", format_bytes(t.bytes), t.transfers);
        }
        rows.emplace_back(kLabels[i], std::move(value));
    }
    return rows;
}

void HostTransferStats::log_summary() const {
    if (!enabled()) {
        return;
    }
    const auto rows = summary_rows();
    if (rows.empty()) {
        log_info(tt::LogMetal, "Host transfer stats | no transfers recorded");
        return;
    }
    log_info(tt::LogMetal, "Host transfer stats | rates are totals across all devices and command queues");
    for (const auto& [label, value] : rows) {
        log_info(tt::LogMetal, "Host transfer stats | {:<32} {}", label, value);
    }
}

void HostTransferStats::reset() {
    for (auto& s : streams_) {
        std::lock_guard lk(s.mutex);
        s.bytes = 0;
        s.transfers = 0;
        s.busy = {};
        s.stretch_start = Clock::time_point::max();
        s.stretch_end = Clock::time_point::min();
    }
}

ScopedHostTransfer::ScopedHostTransfer(HostTransferStats* stats, HostTransfer kind, uint64_t bytes) :
    kind_(kind), bytes_(bytes) {
    if (stats == nullptr || !stats->enabled()) {
        return;
    }
    stats_ = stats;
    stats_->begin(kind_);
    start_ = HostTransferStats::Clock::now();
}

ScopedHostTransfer::~ScopedHostTransfer() {
    if (stats_ != nullptr) {
        stats_->end(kind_, start_, HostTransferStats::Clock::now(), bytes_);
    }
}

}  // namespace tt::tt_metal
