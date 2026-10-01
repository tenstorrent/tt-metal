// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "noc_debugging.hpp"

#include <algorithm>
#include <map>
#include <type_traits>
#include <enchantum/enchantum.hpp>

#include <fmt/base.h>
#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/xy_pair.hpp>
#include "tt_metal/third_party/umd/device/api/umd/device/types/xy_pair.hpp"
#include "tt_stl/assert.hpp"
#include "hostdev/profiler_common.h"

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

// Same wrapping comparison as above, for the device profiler marker timestamp. That timestamp is narrower than the
// uint64_t carrying it (the device discards the wall-clock bits that do not fit the marker, see
// PROFILER_MARKER_TS_BITS), so it wraps and a plain <= would misorder events across the wrap.
//
// Wrapping (a - b) as a SIGNED offset, by sign-extending the difference at the timestamp's real width. Valid whenever
// a and b are within half a wrap window (hours) of each other, which always holds because we only compare timestamps
// within a single short (seconds-wide) batch. Used both for comparison (below) and to sort a batch that may straddle
// the wrap: sorting by ts_sub(ts, ref) against a fixed per-chip reference is a valid strict-weak-ordering.
inline int64_t ts_sub(uint64_t a, uint64_t b) {
    constexpr uint32_t shift = 64 - kernel_profiler::PROFILER_MARKER_TS_BITS;
    return static_cast<int64_t>((a - b) << shift) >> shift;
}

// Wrapping >= (RFC 1982): a is at-or-after b.
inline bool ts_wrap_ge(uint64_t a, uint64_t b) { return ts_sub(a, b) >= 0; }

NOCDebugState::LockedBufferInfo::LockType get_lock_type(NocDebuggingEventMetadata::NocDebugEventType event_type) {
    if (event_type == NocDebuggingEventMetadata::NocDebugEventType::CB_LOCK ||
        event_type == NocDebuggingEventMetadata::NocDebugEventType::CB_UNLOCK) {
        return NOCDebugState::LockedBufferInfo::LockType::CB;
    }

    if (event_type == NocDebuggingEventMetadata::NocDebugEventType::MEM_LOCK ||
        event_type == NocDebuggingEventMetadata::NocDebugEventType::MEM_UNLOCK) {
        return NOCDebugState::LockedBufferInfo::LockType::MEM;
    }

    if (event_type == NocDebuggingEventMetadata::NocDebugEventType::DFB_LOCK ||
        event_type == NocDebuggingEventMetadata::NocDebugEventType::DFB_UNLOCK) {
        return NOCDebugState::LockedBufferInfo::LockType::DFB;
    }

    TT_THROW("Invalid lock type: {}", enchantum::to_string(event_type));
}

NOCDebugIssueBaseType locked_buffer_issue_base_type(NOCDebugState::LockedBufferInfo::LockType lock_type) {
    switch (lock_type) {
        case NOCDebugState::LockedBufferInfo::LockType::MEM:
            return NOCDebugIssueBaseType::WRITE_TO_LOCKED_CORE_LOCAL_MEM;
        case NOCDebugState::LockedBufferInfo::LockType::CB: return NOCDebugIssueBaseType::WRITE_TO_LOCKED_CB;
        case NOCDebugState::LockedBufferInfo::LockType::DFB: return NOCDebugIssueBaseType::WRITE_TO_LOCKED_DFB;
    }
    TT_THROW("Invalid lock type");
}

// May be called for non-write issue types, so the default case returns nullptr.
const char* locked_buffer_type_name(NOCDebugIssueBaseType base_type) {
    switch (base_type) {
        case NOCDebugIssueBaseType::WRITE_TO_LOCKED_CORE_LOCAL_MEM: return "core local mem";
        case NOCDebugIssueBaseType::WRITE_TO_LOCKED_CB: return "circular buffer";
        case NOCDebugIssueBaseType::WRITE_TO_LOCKED_DFB: return "dataflow buffer";
        default: return nullptr;
    }
}

}  // namespace detail

NOCDebugState::CoreDebugState& NOCDebugState::get_state(tt_cxy_pair core) { return cores[core]; }

const NOCDebugState::CoreDebugState& NOCDebugState::get_state(tt_cxy_pair core) const { return cores[core]; }

bool NOCDebugState::has_state(tt_cxy_pair core) const { return cores.contains(core); }

void NOCDebugState::handle_write_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocWriteEvent event) {
    CoreDebugState& state = get_state(core);
    uint8_t noc_id = event.noc;
    uint64_t src_addr = event.src_addr;
    bool posted = event.posted;
    bool is_semaphore = event.is_semaphore;
    bool is_mcast = event.is_mcast;
    bool issue_found = false;

    // The source-reuse and counter-monotonicity checks only make sense for writes that carry an L1 source buffer
    // and a usable NIU write-counter snapshot. An inline dword write has neither (its value is immediate and it
    // does not advance the tracked write counter), so skip both to avoid false positives.
    if (event.has_source_buffer) {
        // Multiple writes from the same source address without a barrier in between
        // Source data potentially overwritten before being flushed
        if ((posted && state.posted_writes_pending[noc_id].contains(src_addr)) ||
            (!posted && state.nonposted_writes_pending[noc_id].contains(src_addr))) {
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
    }

    if (issue_found) {
        // Classify write type for more detailed error reporting
        NOCDebugIssueType issue_type(NOCDebugIssueBaseType::WRITE_FLUSH_BARRIER, is_mcast, is_semaphore);
        state.issue[processor_id].set_issue(issue_type);
    }

    // Resolve where this write actually lands. For a stateful write the destination was programmed in two halves
    // by the hardware: the earlier WRITE_SET_STATE supplied the coordinates and everything above the low address
    // word (on Blackhole the NOC_RET_ADDR_MID high bits), while this write supplied only the low word
    // (NOC_RET_ADDR_LO). Its own dst_x/dst_y are therefore placeholders (0,0) and its address only the bottom of
    // the real one, so both come from the tracked write state instead.
    // Coords are non-negative NOC grid coordinates; cast through uint8_t so the signed int8_t fields widen without
    // a signed-char conversion warning (the -1 sentinel in an unused mcast_end field maps to 255 but is never read).
    bool have_dst = true;
    int dst_x = static_cast<uint8_t>(event.dst_x);
    int dst_y = static_cast<uint8_t>(event.dst_y);
    int mcast_end_x = static_cast<uint8_t>(event.mcast_end_dst_x);
    int mcast_end_y = static_cast<uint8_t>(event.mcast_end_dst_y);
    bool dst_is_mcast = event.is_mcast;
    uint64_t write_addr = event.dst_addr;
    uint32_t write_size = event.num_bytes;
    if (!event.has_valid_dst) {
        const CoreDebugState::WriteStateInfo& ws = state.current_write_state[processor_id][noc_id];
        have_dst = ws.valid;  // a stateful write with no preceding set_state has no resolvable destination
        dst_x = static_cast<uint8_t>(ws.dst_x);
        dst_y = static_cast<uint8_t>(ws.dst_y);
        mcast_end_x = static_cast<uint8_t>(ws.mcast_end_dst_x);
        mcast_end_y = static_cast<uint8_t>(ws.mcast_end_dst_y);
        dst_is_mcast = ws.is_mcast;
        // Reassemble the address from the two halves the hardware programmed separately.
        constexpr uint64_t low_word_mask = 0xFFFF'FFFFull;  // NOC_RET_ADDR_LO, the part each write programs
        write_addr = (ws.dst_addr & ~low_word_mask) | (write_addr & low_word_mask);
        // The non-trid stateful write records num_bytes == 0 (its size lives in the set_state); the trid variant
        // carries its own size, so prefer a non-zero event size and fall back to the tracked one.
        if (write_size == 0) {
            write_size = ws.num_bytes;
        }
    }

    // For each core the write lands on, flag a write to locked buffer, or a WRITE_TO_UNLOCKED_DFB on the writer's own
    // core.
    const auto flag_write_into_core = [&](tt_cxy_pair dst_core) {
        if (!has_state(dst_core)) {
            return;
        }
        const bool same_core = (core == dst_core);
        CoreDebugState& dst_state = get_state(dst_core);
        NOCDebugIssueType issue_type;
        if (const auto* locked_buf =
                dst_state.get_noc_write_to_lock_buffer(write_addr, write_size, processor_id, same_core)) {
            issue_type.base_type = detail::locked_buffer_issue_base_type(locked_buf->lock_type);
        } else if (same_core && dst_state.write_into_unlocked_dfb(write_addr, write_size, processor_id)) {
            issue_type.base_type = NOCDebugIssueBaseType::WRITE_TO_UNLOCKED_DFB;
        } else {
            return;
        }
        issue_type.issue_address = write_addr;
        issue_type.issue_size = write_size;
        issue_type.src_x = event.src_x;
        issue_type.src_y = event.src_y;
        issue_type.dst_x = static_cast<uint8_t>(dst_core.x);
        issue_type.dst_y = static_cast<uint8_t>(dst_core.y);
        issue_type.is_mcast = dst_is_mcast;
        state.issue[processor_id].set_issue(issue_type);
    };

    if (have_dst) {
        if (dst_is_mcast) {
            // dst is the rectangle start corner and mcast_end_* the end corner (the two can be reversed on NOC1),
            // so check every core in the inclusive bounding box.
            for (int x = std::min(dst_x, mcast_end_x); x <= std::max(dst_x, mcast_end_x); ++x) {
                for (int y = std::min(dst_y, mcast_end_y); y <= std::max(dst_y, mcast_end_y); ++y) {
                    flag_write_into_core(tt_cxy_pair{core.chip, static_cast<size_t>(x), static_cast<size_t>(y)});
                }
            }
        } else {
            flag_write_into_core(tt_cxy_pair{core.chip, static_cast<size_t>(dst_x), static_cast<size_t>(dst_y)});
        }
    }

    // Always track the pending write so the end-of-kernel unflushed check sees it, but only advance the
    // counter-monotonicity baseline for writes that carry a real counter snapshot (see has_source_buffer above).
    if (event.posted) {
        state.posted_writes_pending[noc_id][src_addr] = {processor_id, is_semaphore, is_mcast};
        if (event.has_source_buffer) {
            state.posted_write_counter_snapshot[noc_id] = event.counter_snapshot;
            state.any_posted_writes[noc_id] = true;
        }
    } else {
        state.nonposted_writes_pending[noc_id][src_addr] = {processor_id, is_semaphore, is_mcast};
        if (event.has_source_buffer) {
            state.nonposted_write_counter_snapshot[noc_id] = event.counter_snapshot;
            state.any_nonposted_writes[noc_id] = true;
        }
    }
    update_latest_risc_timestamp(core, processor_id, timestamp);
}

void NOCDebugState::handle_write_set_state_event(
    tt_cxy_pair core, int processor_id, uint64_t timestamp, NocWriteSetStateEvent event) {
    CoreDebugState& state = get_state(core);
    uint8_t noc_id = event.noc;

    // Record the destination programmed for this (processor, noc) so subsequent stateful writes can resolve their
    // real destination core. Events are processed in timestamp order, so a set_state always lands before the writes
    // that reuse it, and a later set_state overwrites this one.
    CoreDebugState::WriteStateInfo& ws = state.current_write_state[processor_id][noc_id];
    ws.dst_addr = event.dst_addr;
    ws.dst_x = event.dst_x;
    ws.dst_y = event.dst_y;
    ws.mcast_end_dst_x = event.mcast_end_dst_x;
    ws.mcast_end_dst_y = event.mcast_end_dst_y;
    ws.num_bytes = event.num_bytes;
    ws.is_mcast = event.is_mcast;
    ws.valid = true;

    update_latest_risc_timestamp(core, processor_id, timestamp);
}

void NOCDebugState::handle_read_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocReadEvent event) {
    CoreDebugState& state = get_state(core);
    uint8_t noc_id = event.noc;
    uint64_t dst_addr = event.dst_addr;
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

void NOCDebugState::handle_full_barrier_event(
    tt_cxy_pair core, int processor_id, uint64_t timestamp, NocFullBarrierEvent event) {
    CoreDebugState& state = get_state(core);
    uint8_t noc_id = event.noc;
    update_latest_risc_timestamp(core, processor_id, timestamp);

    // A full barrier waits for all outstanding reads, writes and atomics to complete, so every pending set clears.
    state.reads_not_flushed[noc_id].clear();
    state.posted_writes_pending[noc_id].clear();
    state.nonposted_writes_pending[noc_id].clear();
    state.atomics_pending[noc_id].clear();
}

void NOCDebugState::handle_semaphore_inc_event(
    tt_cxy_pair core, int processor_id, uint64_t timestamp, NocSemaphoreIncEvent event) {
    CoreDebugState& state = get_state(core);
    uint8_t noc_id = event.noc;
    update_latest_risc_timestamp(core, processor_id, timestamp);

    // An atomic increment carries no source buffer and does not advance the NIU write counter, so neither the
    // source-reuse nor the counter-monotonicity check applies. Only a non-posted increment expects an ack and must
    // be flushed (via an atomic/full barrier) before kernel end; a posted increment is fire-and-forget.
    if (!event.posted) {
        state.atomics_pending[noc_id][event.dst_addr] = {processor_id, /*is_semaphore=*/true, event.is_mcast};
    }
}

void NOCDebugState::handle_atomic_barrier_event(
    tt_cxy_pair core, int processor_id, uint64_t timestamp, NocAtomicBarrierEvent event) {
    CoreDebugState& state = get_state(core);
    uint8_t noc_id = event.noc;
    update_latest_risc_timestamp(core, processor_id, timestamp);

    // An atomic barrier waits only for outstanding atomics (separate NIU counter from writes), so it clears the
    // atomics pending set and leaves reads/writes untouched.
    state.atomics_pending[noc_id].clear();
}

void NOCDebugState::handle_scoped_lock_event(
    tt_cxy_pair core, int processor_id, uint64_t timestamp, ScopedLockEvent event) {
    CoreDebugState& state = get_state(core);

    // DFB region bookkeeping, so a write into an unlocked DFB can be flagged.
    using EventType = NocDebuggingEventMetadata::NocDebugEventType;
    if (event.event_type == EventType::DFB_REGION_START) {
        state.dfb_regions[processor_id].insert({event.locked_address_base, event.num_bytes});
        update_latest_risc_timestamp(core, processor_id, timestamp);
        return;
    }
    if (event.event_type == EventType::DFB_REGION_CLEAR) {
        // Carries no extent; unregisters everything this RISC declared.
        state.dfb_regions[processor_id].clear();
        update_latest_risc_timestamp(core, processor_id, timestamp);
        return;
    }

    // Merging intervals is not required: unlock carries the same start address and size as its lock, so the two
    // always key to the same entry. Refcounted so nested or duplicate locks over the same region are only released
    // once the matching number of unlocks arrive; decrement only when the region is actually tracked, so a
    // stray/unmatched unlock cannot underflow the count.
    auto& bufs = state.locked_buffers[processor_id];
    const LockedBufferInfo buf{{event.locked_address_base, event.num_bytes}, detail::get_lock_type(event.event_type)};
    if (event.is_lock()) {
        ++bufs[buf];
    } else if (auto it = bufs.find(buf); it != bufs.end() && --it->second == 0) {
        bufs.erase(it);
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
            // Non-posted atomics (semaphore incs) left outstanding at kernel end (no atomic/full barrier).
            for (const auto& [addr, info] : state.atomics_pending[noc_id]) {
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

NOCDebugState::StateSummary NOCDebugState::get_state_summary() const {
    StateSummary summary;
    {
        std::lock_guard<std::mutex> lock{pending_events_mutex_};
        summary.pending_events = pending_events_.size();
    }
    std::unique_lock<std::mutex> lock{cores_mutex};
    // Iterate the map directly (do NOT use get_state, which would insert empty entries).
    for (const auto& [core, state] : cores) {
        for (size_t processor_id = 0; processor_id < CoreDebugState::MAX_PROCESSORS; ++processor_id) {
            summary.issues += state.issue[processor_id].issues.size();
        }
    }
    return summary;
}

std::string NOCDebugState::get_issue_description(const NOCDebugIssueType& issue_type) {
    if (issue_type.base_type == NOCDebugIssueBaseType::READ_BARRIER) {
        return "read";
    }

    if (issue_type.base_type == NOCDebugIssueBaseType::WRITE_TO_UNLOCKED_DFB) {
        return fmt::format(
            "from ({},{}) to ({},{}) addr 0x{:08X} size {} unlocked dataflow buffer",
            static_cast<int>(issue_type.src_x),
            static_cast<int>(issue_type.src_y),
            static_cast<int>(issue_type.dst_x),
            static_cast<int>(issue_type.dst_y),
            issue_type.issue_address,
            issue_type.issue_size);
    }

    if (const char* locked_type = detail::locked_buffer_type_name(issue_type.base_type)) {
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
        std::vector<std::string> unlocked_dfb_issues;
        bool has_read_barrier = false;
    };
    std::map<std::string, CoreIssues> issues_by_core;

    for (auto& [core, state] : cores) {
        for (size_t proc = 0; proc < CoreDebugState::MAX_PROCESSORS; ++proc) {
            auto& issue = state.issue[proc];
            if (!issue.any_issue()) {
                continue;
            }
            issue.unreported.clear();

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
                } else if (issue_type.base_type == NOCDebugIssueBaseType::WRITE_TO_UNLOCKED_DFB) {
                    core_issues.unlocked_dfb_issues.push_back(get_issue_description(issue_type));
                } else if (detail::locked_buffer_type_name(issue_type.base_type) != nullptr) {
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

    for (const auto& [core_key, core_issues] : issues_by_core) {
        if (!core_issues.unlocked_dfb_issues.empty()) {
            log_error(tt::LogMetal, "Write to unlocked DFB (wrote a DFB region without holding its lock):");
            break;
        }
    }
    for (const auto& [core_key, core_issues] : issues_by_core) {
        if (!core_issues.unlocked_dfb_issues.empty()) {
            log_error(tt::LogMetal, "  {} [{}]", core_key, fmt::join(core_issues.unlocked_dfb_issues, ", "));
        }
    }

    log_error(tt::LogMetal, "========================================");
    log_error(tt::LogMetal, "");
}

void NOCDebugState::report_new_issues() const {
    std::lock_guard<std::mutex> lock{cores_mutex};

    for (auto& [core, state] : cores) {
        for (size_t proc = 0; proc < CoreDebugState::MAX_PROCESSORS; ++proc) {
            auto& unreported = state.issue[proc].unreported;
            for (const auto& issue_type : unreported) {
                log_error(
                    tt::LogMetal,
                    "[NOC issue] Device {} ({},{}) Processor {}: {}",
                    core.chip,
                    core.x,
                    core.y,
                    proc,
                    get_issue_description(issue_type));
            }
            unreported.clear();
        }
    }
}

void NOCDebugState::push_event(size_t chip_id, uint64_t timestamp, int processor_id, const NOCDebugEvent& event) {
    std::lock_guard<std::mutex> lock{pending_events_mutex_};
    pending_events_.push_back({chip_id, timestamp, processor_id, event});
}

void NOCDebugState::process_events_locked(std::vector<PendingEvent>& to_process) {
    // Caller must hold cores_mutex. Sort by (chip, then wrap-aware time within that chip) and fold each event into
    // the per-core state machine. Timestamps are only comparable WITHIN a chip.
    std::unordered_map<size_t, uint64_t> chip_ref;
    for (const auto& e : to_process) {
        auto [it, inserted] = chip_ref.try_emplace(e.chip_id, e.timestamp);
        if (!inserted && detail::ts_wrap_ge(e.timestamp, it->second)) {
            it->second = e.timestamp;
        }
    }

    // The linearisation above is only correct while every event sits within half a wrap window of the reference.
    constexpr int64_t implausible_span = int64_t(1) << (kernel_profiler::PROFILER_MARKER_TS_BITS - 2);
    for (const auto& e : to_process) {
        if (detail::ts_sub(e.timestamp, chip_ref.at(e.chip_id)) < -implausible_span) {
            log_warning(
                tt::LogMetal,
                "NOC debug: a batch of {} events spans more than a quarter of the device timestamp wrap window; "
                "event ordering within it may be wrong. Process events more often (see "
                "TT_METAL_NOC_DEBUG_FULL_READ_INTERVAL_MS).",
                to_process.size());
            break;
        }
    }
    std::sort(to_process.begin(), to_process.end(), [&chip_ref](const PendingEvent& a, const PendingEvent& b) {
        if (a.chip_id != b.chip_id) {
            return a.chip_id < b.chip_id;
        }
        const uint64_t ref = chip_ref.at(a.chip_id);
        return detail::ts_sub(a.timestamp, ref) < detail::ts_sub(b.timestamp, ref);
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
                } else if constexpr (std::is_same_v<T, NocWriteSetStateEvent>) {
                    tt_cxy_pair key{chip_id, {static_cast<size_t>(e.src_x), static_cast<size_t>(e.src_y)}};
                    handle_write_set_state_event(key, processor_id, timestamp, e);
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
                } else if constexpr (std::is_same_v<T, NocFullBarrierEvent>) {
                    tt_cxy_pair key{chip_id, {static_cast<size_t>(e.src_x), static_cast<size_t>(e.src_y)}};
                    handle_full_barrier_event(key, processor_id, timestamp, e);
                } else if constexpr (std::is_same_v<T, NocSemaphoreIncEvent>) {
                    tt_cxy_pair key{chip_id, {static_cast<size_t>(e.src_x), static_cast<size_t>(e.src_y)}};
                    handle_semaphore_inc_event(key, processor_id, timestamp, e);
                } else if constexpr (std::is_same_v<T, NocAtomicBarrierEvent>) {
                    tt_cxy_pair key{chip_id, {static_cast<size_t>(e.src_x), static_cast<size_t>(e.src_y)}};
                    handle_atomic_barrier_event(key, processor_id, timestamp, e);
                } else if constexpr (std::is_same_v<T, ScopedLockEvent>) {
                    tt_cxy_pair key{chip_id, {static_cast<size_t>(e.src_x), static_cast<size_t>(e.src_y)}};
                    handle_scoped_lock_event(key, processor_id, timestamp, e);
                }
            },
            entry.event);
    }
}

void NOCDebugState::process_accumulated_events_all_chips() {
    std::lock_guard<std::mutex> cores_lock{cores_mutex};

    // Move the whole queue out immediately because another thread may still push events while we process this batch.
    std::vector<PendingEvent> to_process;
    {
        std::lock_guard<std::mutex> lock{pending_events_mutex_};
        to_process = std::move(pending_events_);
        pending_events_.clear();
    }
    // Process everything: this is called only at a quiescent point (a user read at a kernel boundary, or device
    // close), where every core has flushed, so the batch is a complete snapshot and no watermark is needed.
    process_events_locked(to_process);
}

void NOCDebugState::process_accumulated_events_up_to(uint64_t margin_ticks) {
    std::lock_guard<std::mutex> cores_lock{cores_mutex};

    std::vector<PendingEvent> batch;
    {
        std::lock_guard<std::mutex> lock{pending_events_mutex_};
        batch = std::move(pending_events_);
        pending_events_.clear();
    }

    // Bounded-lateness watermark. This runs mid-run (from the background full read), where the snapshot is NOT
    // guaranteed complete: a momentarily-stalled core may not yet have recorded an older event.
    std::unordered_map<size_t, uint64_t> chip_latest;
    for (const auto& e : batch) {
        auto [it, inserted] = chip_latest.try_emplace(e.chip_id, e.timestamp);
        if (!inserted && detail::ts_wrap_ge(e.timestamp, it->second)) {
            it->second = e.timestamp;
        }
    }

    std::vector<PendingEvent> to_process;
    std::vector<PendingEvent> retained;
    to_process.reserve(batch.size());
    for (const auto& e : batch) {
        // May underflow past zero; that is fine, ts_wrap_ge only looks at the low PROFILER_MARKER_TS_BITS of the
        // difference, so the value stays correct modulo the timestamp width without masking it back down.
        const uint64_t watermark = chip_latest.at(e.chip_id) - margin_ticks;
        // Process if the event is at/before the watermark; the retained tail (newer than watermark, i.e. within
        // margin_ticks of the latest) waits for the next call.
        if (detail::ts_wrap_ge(watermark, e.timestamp)) {
            to_process.push_back(e);
        } else {
            retained.push_back(e);
        }
    }

    process_events_locked(to_process);

    if (!retained.empty()) {
        std::lock_guard<std::mutex> lock{pending_events_mutex_};
        pending_events_.insert(pending_events_.end(), retained.begin(), retained.end());
    }
}

const NOCDebugState::LockedBufferInfo* NOCDebugState::CoreDebugState::get_noc_write_to_lock_buffer(
    uint64_t write_start, uint32_t write_size, int writer_processor_id, bool same_core) const {
    const uint64_t write_end = write_start + write_size;
    const auto& bufs = this->locked_buffers;
    for (auto proc_id = 0; proc_id < CoreDebugState::MAX_PROCESSORS; ++proc_id) {
        // Self-writing while holding a lock is allowed.
        if (same_core && proc_id == writer_processor_id) {
            continue;
        }
        for (const auto& [buf, hold_count] : bufs[proc_id]) {
            const uint64_t buf_end = buf.extent.address + buf.extent.size;
            if (write_end > buf.extent.address && buf_end > write_start) {
                return &buf;
            }
        }
    }
    return nullptr;
}

bool NOCDebugState::CoreDebugState::write_into_unlocked_dfb(
    uint64_t write_start, uint32_t write_size, int writer_processor_id) const {
    const uint64_t write_end = write_start + write_size;

    const auto overlaps_write = [&](const L1Extent& region) {
        return write_end > region.address && (region.address + region.size) > write_start;
    };
    const bool write_into_dfb =
        std::any_of(dfb_regions.begin(), dfb_regions.end(), [&](const std::set<L1Extent>& regions) {
            return std::any_of(regions.begin(), regions.end(), overlaps_write);
        });
    if (!write_into_dfb) {
        return false;
    }

    // The write must be fully covered by the union of the writer's own locks.
    // locked_buffers[proc] is ordered by (address, size, type), so a single ascending sweep suffices.
    uint64_t covered_to = write_start;
    for (const auto& [buf, hold_count] : locked_buffers[writer_processor_id]) {
        if (buf.extent.address > covered_to) {
            break;  // gap before this lock -> not fully covered
        }
        const uint64_t buf_end = buf.extent.address + buf.extent.size;
        covered_to = std::max(buf_end, covered_to);
        if (covered_to >= write_end) {
            break;
        }
    }
    // Flag iff the writer's own locks do not cover the whole write.
    return covered_to < write_end;
}

}  // namespace tt::tt_metal
