// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "noc_debugging.hpp"

#include <map>
#include <type_traits>

#include <fmt/base.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/xy_pair.hpp>
#include "tt_metal/third_party/umd/device/api/umd/device/types/xy_pair.hpp"

namespace tt::tt_metal {

namespace detail {

std::string format_core_info(tt_cxy_pair core, int processor_id) {
    return fmt::format("Device:{}, Virtual Core:{}, Processor:{}", core.chip, core.str(), processor_id);
}

inline bool wrap_ge(uint32_t a, uint32_t b) {
    // wrapping comparison from RFC 1982
    // same number of bits from LocalNocEventDstTrailer
    constexpr uint32_t COUNTER_WIDTH = 12;
    constexpr uint32_t shift = 32 - COUNTER_WIDTH;
    int32_t diff = static_cast<int32_t>(a - b);
    return (diff << shift) >= 0;
}

}  // namespace detail

NOCDebugState::CoreDebugState& NOCDebugState::get_state(tt_cxy_pair core) { return cores[core]; }

const NOCDebugState::CoreDebugState& NOCDebugState::get_state(tt_cxy_pair core) const { return cores[core]; }

void NOCDebugState::handle_write_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocWriteEvent event) {
    CoreDebugState& state = get_state(core);
    uint8_t noc_id = event.noc;
    uint32_t src_addr = event.src_addr;
    bool posted = event.posted;
    bool is_semaphore = event.is_semaphore;
    bool is_mcast = event.is_mcast;
    std::string problem_type;

    // Multiple writes from the same source address without a barrier in between
    // Source data potentially overwritten before being flushed
    if (posted && state.posted_writes_pending[noc_id].contains(src_addr)) {
        problem_type = "multiple posted writes from the same source address without a barrier in between";
    } else if (!posted && state.nonposted_writes_pending[noc_id].contains(src_addr)) {
        problem_type = "multiple non-posted writes from the same source address without a barrier in between";
    }

    // Check if transaction counter has not increased (should always increase)
    // This detects cases where counters wrap incorrectly or don't advance
    if (posted) {
        if (state.any_posted_writes[noc_id] &&
            !detail::wrap_ge(event.counter_snapshot, state.posted_write_counter_snapshot[noc_id])) {
            problem_type =
                "multiple posted writes from the same source address without a barrier/flush (transaction counters are "
                "the same)";
        }
    } else {
        if (state.any_nonposted_writes[noc_id] &&
            !detail::wrap_ge(event.counter_snapshot, state.nonposted_write_counter_snapshot[noc_id])) {
            problem_type =
                "multiple non-posted writes from the same source address without a barrier/flush (transaction counters "
                "are the same)";
        }
    }

    if (!problem_type.empty()) {
        // Classify write type for more detailed error reporting
        NOCDebugIssueType issue_type(NOCDebugIssueBaseType::WRITE_FLUSH_BARRIER, is_mcast, is_semaphore);
        state.issue[processor_id].set_issue(issue_type);
    }

    if (event.posted) {
        state.posted_writes_pending[noc_id][src_addr] = {processor_id, is_semaphore, is_mcast};
        state.posted_write_counter_snapshot[noc_id] = event.counter_snapshot;
        state.any_posted_writes[noc_id] = true;
    } else {
        state.nonposted_writes_pending[noc_id][src_addr] = {processor_id, is_semaphore, is_mcast};
        state.nonposted_write_counter_snapshot[noc_id] = event.counter_snapshot;
        state.any_nonposted_writes[noc_id] = true;
    }
    update_latest_risc_timestamp(core, processor_id, timestamp);
}

void NOCDebugState::handle_read_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocReadEvent event) {
    CoreDebugState& state = get_state(core);
    uint8_t noc_id = event.noc;
    uint32_t dst_addr = event.dst_addr;
    std::string problem_type;

    // Multiple reads to the same destination address without a barrier in between
    // Destination data potentially being read before a read barrier has ensured that data has fully arrived
    if (state.reads_not_flushed[noc_id].contains(dst_addr)) {
        problem_type = "multiple reads to the same address without read barrier";
    }

    // Check if transaction counter has not increased (should always increase)
    // This detects cases where counters wrap incorrectly or don't advance
    if (state.any_reads[noc_id]) {
        if (!detail::wrap_ge(event.counter_snapshot, state.read_counter_snapshot[noc_id])) {
            if (!problem_type.empty()) {
                problem_type += "; ";
            }
            problem_type +=
                "multiple reads to the same address without read barrier (transaction counters are the same)";
        }
    }

    if (!problem_type.empty()) {
        state.issue[processor_id].set_issue(NOCDebugIssueType(NOCDebugIssueBaseType::READ_BARRIER));
    }

    update_latest_risc_timestamp(core, processor_id, timestamp);

    state.reads_not_flushed[noc_id].insert(event.dst_addr);
    state.read_counter_snapshot[noc_id] = event.counter_snapshot;
    state.any_reads[noc_id] = true;
}

void NOCDebugState::handle_read_barrier_event(
    tt_cxy_pair core, int processor_id, uint64_t timestamp, NocReadBarrierEvent event) {
    CoreDebugState& state = get_state(core);
    uint8_t noc_id = event.noc;
    update_latest_risc_timestamp(core, processor_id, timestamp);

    state.reads_not_flushed[noc_id].clear();
}

void NOCDebugState::handle_write_barrier_event(
    tt_cxy_pair core, int processor_id, uint64_t timestamp, NocWriteBarrierEvent event) {
    CoreDebugState& state = get_state(core);
    uint8_t noc_id = event.noc;
    update_latest_risc_timestamp(core, processor_id, timestamp);

    if (event.posted) {
        state.posted_writes_pending[noc_id].clear();
    } else {
        state.nonposted_writes_pending[noc_id].clear();
    }
}

void NOCDebugState::handle_write_flush_event(
    tt_cxy_pair core, int processor_id, uint64_t timestamp, NocWriteFlushEvent event) {
    CoreDebugState& state = get_state(core);
    uint8_t noc_id = event.noc;
    update_latest_risc_timestamp(core, processor_id, timestamp);

    if (event.posted) {
        state.posted_writes_pending[noc_id].clear();
    } else {
        state.nonposted_writes_pending[noc_id].clear();
    }
}

void NOCDebugState::update_latest_risc_timestamp(tt_cxy_pair core, int processor_id, uint64_t timestamp) {
    cores[core].latest_risc_timestamp[processor_id] = timestamp;
}

void NOCDebugState::finish_cores() {
    std::unique_lock<std::mutex> lock{cores_mutex};

    const auto get_unflushed_write_issue_type = [](const NOCDebugState::PendingWriteInfo& info) {
        return NOCDebugIssueType(NOCDebugIssueBaseType::UNFLUSHED_WRITE_AT_END, info.is_mcast, info.is_semaphore);
    };

    for (auto& [core, state] : cores) {
        for (size_t noc_id = 0; noc_id < CoreDebugState::MAX_NOCS; ++noc_id) {
            // Set issues for the specific processor that initiated each pending write
            for (const auto& [addr, info] : state.posted_writes_pending[noc_id]) {
                state.issue[info.processor_id].set_issue(get_unflushed_write_issue_type(info));
            }
            for (const auto& [addr, info] : state.nonposted_writes_pending[noc_id]) {
                state.issue[info.processor_id].set_issue(get_unflushed_write_issue_type(info));
            }
        }
    }
}

NOCDebugIssue NOCDebugState::get_issues(tt_cxy_pair core, int processor_id) const {
    std::unique_lock<std::mutex> lock{cores_mutex};
    const CoreDebugState& state = get_state(core);
    return state.issue[processor_id];
}

void NOCDebugState::reset_state() {
    std::unique_lock<std::mutex> lock{cores_mutex};
    cores.clear();
}

std::string NOCDebugState::get_issue_description(const NOCDebugIssueType& issue_type) {
    std::string desc;
    if (issue_type.base_type == NOCDebugIssueBaseType::READ_BARRIER) {
        return "read";
    }

    // For writes, build description based on flags
    if (issue_type.is_semaphore) {
        desc = "semaphore";
    } else {
        desc = "write";
    }

    if (issue_type.is_mcast) {
        desc += " mcast";
    } else {
        desc += " unicast";
    }

    return desc;
}

void NOCDebugState::print_aggregated_errors() const {
    std::unique_lock<std::mutex> lock{cores_mutex};

    // Collect issues by category, grouped by core
    struct CoreIssues {
        std::vector<std::string> write_barrier_issues;
        std::vector<std::string> unflushed_write_issues;  // at end of kernel
        bool has_read_barrier = false;
    };
    std::map<std::string, CoreIssues> issues_by_core;

    for (const auto& [core, state] : cores) {
        for (size_t proc = 0; proc < CoreDebugState::MAX_PROCESSORS; ++proc) {
            const auto& issue = state.issue[proc];
            if (!issue.any_issue()) {
                continue;
            }

            std::string core_key = fmt::format("Device {} ({},{}) Processor {}", core.chip, core.x, core.y, proc);
            CoreIssues& core_issues = issues_by_core[core_key];

            // Iterate through all issues and categorize them
            for (const auto& issue_type : issue.issues) {
                if (issue_type.base_type == NOCDebugIssueBaseType::WRITE_FLUSH_BARRIER) {
                    core_issues.write_barrier_issues.push_back(get_issue_description(issue_type));
                } else if (issue_type.base_type == NOCDebugIssueBaseType::READ_BARRIER) {
                    core_issues.has_read_barrier = true;
                } else if (issue_type.base_type == NOCDebugIssueBaseType::UNFLUSHED_WRITE_AT_END) {
                    core_issues.unflushed_write_issues.push_back(get_issue_description(issue_type));
                }
            }
        }
    }

    if (issues_by_core.empty()) {
        return;
    }

    log_error(tt::LogMetal, "========== NOC Debug Summary ==========");

    for (const auto& [core_key, core_issues] : issues_by_core) {
        if (!core_issues.write_barrier_issues.empty()) {
            log_error(tt::LogMetal, "Missing write barrier/flush (same src addr written multiple times):");
            break;
        }
    }
    for (const auto& [core_key, core_issues] : issues_by_core) {
        if (!core_issues.write_barrier_issues.empty()) {
            log_error(tt::LogMetal, "  {} [{}]", core_key, fmt::join(core_issues.write_barrier_issues, ", "));
        }
    }

    bool has_read_barrier = false;
    for (const auto& [core_key, core_issues] : issues_by_core) {
        if (core_issues.has_read_barrier) {
            has_read_barrier = true;
            break;
        }
    }
    if (has_read_barrier) {
        log_error(tt::LogMetal, "Missing read barrier (same dst addr read multiple times):");
        for (const auto& [core_key, core_issues] : issues_by_core) {
            if (core_issues.has_read_barrier) {
                log_error(tt::LogMetal, "  {}", core_key);
            }
        }
    }

    for (const auto& [core_key, core_issues] : issues_by_core) {
        if (!core_issues.unflushed_write_issues.empty()) {
            log_error(tt::LogMetal, "Unflushed async writes at kernel end (missing noc_async_write_barrier):");
            break;
        }
    }
    for (const auto& [core_key, core_issues] : issues_by_core) {
        if (!core_issues.unflushed_write_issues.empty()) {
            std::string issues_str;
            for (size_t i = 0; i < core_issues.unflushed_write_issues.size(); ++i) {
                if (i > 0) {
                    issues_str += ", ";
                }
                issues_str += core_issues.unflushed_write_issues[i];
            }
            log_error(tt::LogMetal, "  {} [{}]", core_key, issues_str);
        }
    }

    log_error(tt::LogMetal, "========================================");
    log_error(tt::LogMetal, "");
}

void NOCDebugState::push_event(size_t chip_id, uint64_t timestamp, int processor_id, const NOCDebugEvent& event) {
    std::unique_lock<std::mutex> lock{cores_mutex};
    std::visit(
        [&](auto&& e) {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, NocWriteEvent>) {
                tt_cxy_pair key{chip_id, {static_cast<size_t>(e.src_x), static_cast<size_t>(e.src_y)}};
                handle_write_event(key, processor_id, timestamp, e);
            } else if constexpr (std::is_same_v<T, NocReadEvent>) {
                tt_cxy_pair key{chip_id, {static_cast<size_t>(e.dst_x), static_cast<size_t>(e.dst_y)}};
                handle_read_event(key, processor_id, timestamp, e);
            } else if constexpr (std::is_same_v<T, NocReadBarrierEvent>) {
                tt_cxy_pair key{chip_id, {static_cast<size_t>(e.src_x), static_cast<size_t>(e.src_y)}};
                handle_read_barrier_event(key, processor_id, timestamp, e);
            } else if constexpr (std::is_same_v<T, NocWriteBarrierEvent>) {
                tt_cxy_pair key{chip_id, {static_cast<size_t>(e.src_x), static_cast<size_t>(e.src_y)}};
                handle_write_barrier_event(key, processor_id, timestamp, e);
            } else if constexpr (std::is_same_v<T, NocWriteFlushEvent>) {
                tt_cxy_pair key{chip_id, {static_cast<size_t>(e.src_x), static_cast<size_t>(e.src_y)}};
                handle_write_flush_event(key, processor_id, timestamp, e);
            }
        },
        event);
}

}  // namespace tt::tt_metal
