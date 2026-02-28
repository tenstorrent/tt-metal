// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "noc_debugging.hpp"

#include <algorithm>
#include <map>
#include <type_traits>

#include <fmt/base.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/xy_pair.hpp>
#include "tt_metal/third_party/umd/device/api/umd/device/types/xy_pair.hpp"
#include "tt_stl/assert.hpp"

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

NOCDebugState::LockedBufferInfo::LockType get_lock_type(NocDebuggingEventMetadata::NocDebugEventType event_type) {
    if (event_type == NocDebuggingEventMetadata::NocDebugEventType::CB_LOCK ||
        event_type == NocDebuggingEventMetadata::NocDebugEventType::CB_UNLOCK) {
        return NOCDebugState::LockedBufferInfo::LockType::CB;
    }

    if (event_type == NocDebuggingEventMetadata::NocDebugEventType::MEM_LOCK ||
        event_type == NocDebuggingEventMetadata::NocDebugEventType::MEM_UNLOCK) {
        return NOCDebugState::LockedBufferInfo::LockType::MEM;
    }

    TT_THROW("Invalid lock type: {}", enchantum::to_string(event_type));
}

}  // namespace detail

NOCDebugState::CoreDebugState& NOCDebugState::get_state(tt_cxy_pair core) { return cores[core]; }

const NOCDebugState::CoreDebugState& NOCDebugState::get_state(tt_cxy_pair core) const { return cores[core]; }

bool NOCDebugState::has_state(tt_cxy_pair core) const { return cores.contains(core); }

void NOCDebugState::handle_write_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocWriteEvent event) {
    CoreDebugState& state = get_state(core);
    uint8_t noc_id = event.noc;
    uint32_t src_addr = event.src_addr;
    bool posted = event.posted;
    bool is_semaphore = event.is_semaphore;
    bool is_mcast = event.is_mcast;
    bool issue_found = false;

    // Multiple writes from the same source address without a barrier in between
    // Source data potentially overwritten before being flushed
    if (posted && state.posted_writes_pending[noc_id].contains(src_addr)) {
        issue_found = true;
    } else if (!posted && state.nonposted_writes_pending[noc_id].contains(src_addr)) {
        issue_found = true;
    }

    // Check if transaction counter has not increased (should always increase)
    // This detects cases where counters wrap incorrectly or don't advance
    if (posted) {
        if (state.any_posted_writes[noc_id] &&
            !detail::wrap_ge(event.counter_snapshot, state.posted_write_counter_snapshot[noc_id])) {
            issue_found = true;
        }
    } else {
        if (state.any_nonposted_writes[noc_id] &&
            !detail::wrap_ge(event.counter_snapshot, state.nonposted_write_counter_snapshot[noc_id])) {
            issue_found = true;
        }
    }

    if (issue_found) {
        // Classify write type for more detailed error reporting
        NOCDebugIssueType issue_type(NOCDebugIssueBaseType::WRITE_FLUSH_BARRIER, is_mcast, is_semaphore);
        state.issue[processor_id].set_issue(issue_type);
    }

    // Check if the write hit a locked buffer in the destination core
    tt_cxy_pair dst_core{core.chip, static_cast<size_t>(event.dst_x), static_cast<size_t>(event.dst_y)};
    if (has_state(dst_core)) {
        CoreDebugState& dst_state = get_state(dst_core);
        if (const auto* locked_buf = dst_state.get_noc_write_to_lock_buffer(event); locked_buf != nullptr) {
            NOCDebugIssueType issue_type;
            if (locked_buf->lock_type == NOCDebugState::LockedBufferInfo::LockType::MEM) {
                issue_type.base_type = NOCDebugIssueBaseType::WRITE_TO_LOCKED_CORE_LOCAL_MEM;
            } else {
                issue_type.base_type = NOCDebugIssueBaseType::WRITE_TO_LOCKED_CB;
            }
            issue_type.issue_address = locked_buf->address;
            issue_type.issue_size = locked_buf->size;
            issue_type.src_x = event.src_x;
            issue_type.src_y = event.src_y;
            issue_type.dst_x = event.dst_x;
            issue_type.dst_y = event.dst_y;
            state.issue[processor_id].set_issue(issue_type);
        }
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
    bool issue_found = false;

    // Multiple reads to the same destination address without a barrier in between
    // Destination data potentially being read before a read barrier has ensured that data has fully arrived
    if (state.reads_not_flushed[noc_id].contains(dst_addr)) {
        issue_found = true;
    }

    // Check if transaction counter has not increased (should always increase)
    // This detects cases where counters wrap incorrectly or don't advance
    if (state.any_reads[noc_id]) {
        if (!detail::wrap_ge(event.counter_snapshot, state.read_counter_snapshot[noc_id])) {
            issue_found = true;
        }
    }

    if (issue_found) {
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

void NOCDebugState::handle_scoped_lock_event(
    tt_cxy_pair core, int processor_id, uint64_t timestamp, ScopedLockEvent event) {
    CoreDebugState& state = get_state(core);

    // Merge intervals is not required.
    // for unlocking, the start and end address of the request will be the same as from the lock request.
    // if multiple locks and unlocks are requested for the same buffer, then only one will be removed as eventually
    // the Lock destructor will release the lock
    auto& bufs = state.locked_buffers[processor_id];
    auto lock_type = detail::get_lock_type(event.event_type);
    if (event.is_lock()) {
        bufs.insert({event.locked_address_base, event.num_bytes, lock_type});
    } else {
        bufs.erase({event.locked_address_base, event.num_bytes, lock_type});
    }

    update_latest_risc_timestamp(core, processor_id, timestamp);
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
    {
        std::lock_guard<std::mutex> lock{pending_events_mutex_};
        pending_events_.clear();
    }
    std::unique_lock<std::mutex> lock{cores_mutex};
    cores.clear();
}

std::string NOCDebugState::get_issue_description(const NOCDebugIssueType& issue_type) {
    if (issue_type.base_type == NOCDebugIssueBaseType::READ_BARRIER) {
        return "read";
    }

    if (issue_type.base_type == NOCDebugIssueBaseType::WRITE_TO_LOCKED_CORE_LOCAL_MEM ||
        issue_type.base_type == NOCDebugIssueBaseType::WRITE_TO_LOCKED_CB) {
        const char* locked_type =
            (issue_type.base_type == NOCDebugIssueBaseType::WRITE_TO_LOCKED_CB) ? "CB" : "core local mem";
        return fmt::format(
            "from ({},{}) to ({},{}) addr 0x{:08X} size {} locked {}",
            static_cast<int>(issue_type.src_x),
            static_cast<int>(issue_type.src_y),
            static_cast<int>(issue_type.dst_x),
            static_cast<int>(issue_type.dst_y),
            issue_type.issue_address,
            issue_type.issue_size,
            locked_type);
    }

    std::string desc;
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
        std::vector<std::string> locked_buffer_issues;
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
                } else if (issue_type.base_type == NOCDebugIssueBaseType::WRITE_TO_LOCKED_CORE_LOCAL_MEM) {
                    core_issues.locked_buffer_issues.push_back(get_issue_description(issue_type));
                } else if (issue_type.base_type == NOCDebugIssueBaseType::WRITE_TO_LOCKED_CB) {
                    core_issues.locked_buffer_issues.push_back(get_issue_description(issue_type));
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

    for (const auto& [core_key, core_issues] : issues_by_core) {
        if (!core_issues.locked_buffer_issues.empty()) {
            log_error(tt::LogMetal, "Write to locked buffer:");
            break;
        }
    }
    for (const auto& [core_key, core_issues] : issues_by_core) {
        if (!core_issues.locked_buffer_issues.empty()) {
            log_error(tt::LogMetal, "  {} [{}]", core_key, fmt::join(core_issues.locked_buffer_issues, ", "));
        }
    }

    log_error(tt::LogMetal, "========================================");
    log_error(tt::LogMetal, "");
}

void NOCDebugState::push_event(size_t chip_id, uint64_t timestamp, int processor_id, const NOCDebugEvent& event) {
    std::lock_guard<std::mutex> lock{pending_events_mutex_};
    pending_events_.push_back({chip_id, timestamp, processor_id, event});
}

void NOCDebugState::process_accumulated_events_all_chips() {
    // Store them immediately because the main thread will still insert events while we are
    // processing the current batch
    std::vector<PendingEvent> to_process;
    {
        std::lock_guard<std::mutex> lock{pending_events_mutex_};
        to_process = std::move(pending_events_);
        pending_events_.clear();
    }
    // process in global sorted order
    std::sort(to_process.begin(), to_process.end(), [](const PendingEvent& a, const PendingEvent& b) {
        return a.timestamp < b.timestamp;
    });
    for (const PendingEvent& entry : to_process) {
        std::visit(
            [this, &entry](auto&& e) {
                using T = std::decay_t<decltype(e)>;
                const size_t chip_id = entry.chip_id;
                const uint64_t timestamp = entry.timestamp;
                const int processor_id = entry.processor_id;
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
                } else if constexpr (std::is_same_v<T, ScopedLockEvent>) {
                    tt_cxy_pair key{chip_id, {static_cast<size_t>(e.src_x), static_cast<size_t>(e.src_y)}};
                    handle_scoped_lock_event(key, processor_id, timestamp, e);
                }
            },
            entry.event);
    }
}

const NOCDebugState::LockedBufferInfo* NOCDebugState::CoreDebugState::get_noc_write_to_lock_buffer(
    const NocWriteEvent& event) const {
    const uint32_t write_start = event.dst_addr;
    const uint32_t write_end = event.dst_addr + event.num_bytes;
    const auto& bufs = this->locked_buffers;
    for (auto proc_id = 0; proc_id < CoreDebugState::MAX_PROCESSORS; ++proc_id) {
        for (const auto& buf : bufs[proc_id]) {
            const uint32_t buf_end = buf.address + buf.size;
            if (write_end > buf.address && buf_end > write_start) {
                return &buf;
            }
        }
    }
    return nullptr;
}

}  // namespace tt::tt_metal
