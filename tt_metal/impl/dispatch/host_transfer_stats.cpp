// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "host_transfer_stats.hpp"

#include <algorithm>
#include <tt_stl/fmt.hpp>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal {

namespace {

struct CategoryInfo {
    const char* label;
    // The device reads these bytes after the host call returns, so they are reported without a transfer time.
    bool staged = false;
};

// One entry per HostTransferCategory, in enum order.
constexpr auto kCategories = std::to_array<CategoryInfo>({
    {"buffer writes, fast dispatch"},
    {"buffer writes, slow dispatch"},
    {"buffer reads, fast dispatch"},
    {"buffer reads, slow dispatch"},
    {"core writes, fast dispatch"},
    {"core reads, fast dispatch"},
    {"program binaries, fast dispatch"},
    {"program binaries, slow dispatch"},
    {"program config, fast dispatch", /*staged=*/true},
    {"program config, slow dispatch"},
    {"firmware"},
    {"H2D socket writes, host push"},
    {"H2D socket writes, device pull", /*staged=*/true},
});
static_assert(kCategories.size() == kNumHostTransferCategories);

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

std::string format_seconds(double seconds) {
    if (seconds >= 1) {
        return fmt::format("{:.3f} s", seconds);
    }
    if (seconds >= 1e-3) {
        return fmt::format("{:.3f} ms", seconds * 1e3);
    }
    return fmt::format("{:.3f} us", seconds * 1e6);
}

}  // namespace

void HostTransferStats::begin(HostTransferCategory category) {
    auto& s = streams_[static_cast<size_t>(category)];
    std::lock_guard lk(s.mutex);
    ++s.in_flight;
}

void HostTransferStats::end(
    HostTransferCategory category, Clock::time_point start, Clock::time_point end, uint64_t bytes) {
    auto& s = streams_[static_cast<size_t>(category)];
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

void HostTransferStats::add_bytes(HostTransferCategory category, uint64_t bytes) {
    auto& s = streams_[static_cast<size_t>(category)];
    std::lock_guard lk(s.mutex);
    s.bytes += bytes;
    ++s.transfers;
}

HostTransferStats::Totals HostTransferStats::totals(HostTransferCategory category) const {
    const auto& s = streams_[static_cast<size_t>(category)];
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
        const Totals t = totals(static_cast<HostTransferCategory>(i));
        if (t.transfers == 0) {
            continue;
        }
        const double seconds = std::chrono::duration<double>(t.busy).count();
        const std::string count = fmt::format("{} transfer{}", t.transfers, t.transfers == 1 ? "" : "s");
        std::string value;
        if (kCategories[i].staged) {
            value = fmt::format("{} staged, {}", format_bytes(t.bytes), count);
        } else if (seconds > 0) {
            value = fmt::format(
                "{} in {} ({:.2f} GB/s), {}",
                format_bytes(t.bytes),
                format_seconds(seconds),
                static_cast<double>(t.bytes) / 1e9 / seconds,
                count);
        } else {
            value = fmt::format("{}, {}", format_bytes(t.bytes), count);
        }
        rows.emplace_back(kCategories[i].label, std::move(value));
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

ScopedHostTransfer::ScopedHostTransfer(HostTransferStats* stats, HostTransferCategory category, uint64_t bytes) :
    category_(category), bytes_(bytes) {
    if (stats == nullptr || !stats->enabled()) {
        return;
    }
    stats_ = stats;
    stats_->begin(category_);
    start_ = HostTransferStats::Clock::now();
}

ScopedHostTransfer::~ScopedHostTransfer() {
    if (stats_ != nullptr) {
        stats_->end(category_, start_, HostTransferStats::Clock::now(), bytes_);
    }
}

}  // namespace tt::tt_metal
