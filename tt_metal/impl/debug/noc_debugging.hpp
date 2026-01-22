// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bitset>
#include <cstdint>
#include <mutex>
#include <umd/device/types/xy_pair.hpp>
#include <unordered_map>
#include <map>
#include <array>
#include <tools/profiler/event_metadata.hpp>
#include <unordered_set>

namespace tt::tt_metal {

namespace detail {

inline bool check_overlap(const std::map<uint32_t, uint32_t>& ranges, uint32_t start_addr, uint32_t end_addr) {
    if (ranges.empty()) {
        return false;
    }

    // greater than or equal to start_addr
    auto it = ranges.lower_bound(start_addr);

    // previous value end is greater than or equal to start addr which is an overlap
    if (it != ranges.begin()) {
        auto prev = std::prev(it);
        if (prev->second >= start_addr) {
            return true;
        }
    }

    // existing value is greater than start addr and less than end addr which is an overlap
    if (it != ranges.end() && it->first <= end_addr) {
        return true;
    }

    return false;
}

inline void insert_range(std::map<uint32_t, uint32_t>& ranges, uint32_t start_addr, uint32_t end_addr) {
    auto it = ranges.find(start_addr);
    if (it != ranges.end()) {
        // Same start exists - take the larger end to cover both ranges
        it->second = std::max(it->second, end_addr);
    } else {
        ranges.emplace(start_addr, end_addr);
    }
}

}  // namespace detail

struct NocWriteEvent {
    uint32_t src_addr;
    uint32_t dst_addr;
    uint32_t num_bytes;
    uint32_t counter_snapshot;  // nonposted_write_reqs_sent or posted_write_reqs_sent
    int8_t src_x;
    int8_t src_y;
    int8_t dst_x;
    int8_t dst_y;
    bool posted;
    uint8_t noc;
};

struct NocReadEvent {
    uint32_t src_addr;
    uint32_t dst_addr;
    uint32_t num_bytes;
    uint32_t counter_snapshot;  // read_resps_recv
    int8_t src_x;
    int8_t src_y;
    int8_t dst_x;
    int8_t dst_y;
    uint8_t noc;
};

struct NocReadBarrierEvent {
    int8_t src_x;
    int8_t src_y;
    uint8_t noc;
};

struct NocWriteBarrierEvent {
    int8_t src_x;
    int8_t src_y;
    bool posted;
    uint8_t noc;
};

struct NocWriteFlushEvent {
    int8_t src_x;
    int8_t src_y;
    bool posted;
    uint8_t noc;
};

struct LocalMemWriteEvent {
    uint32_t local_start_addr;
    uint32_t local_end_addr;
    // Nonposted writes sent counter value
    uint32_t counter_snapshot;
    int8_t src_x;
    int8_t src_y;
};

struct LocalMemReadEvent {
    uint32_t local_start_addr;
    uint32_t local_end_addr;
    // Read received counter value
    uint32_t counter_snapshot;
    int8_t src_x;
    int8_t src_y;
};

struct UnknownNocEvent {};

using NOCDebugEvent = std::variant<
    NocWriteEvent,
    NocReadEvent,
    NocReadBarrierEvent,
    NocWriteBarrierEvent,
    NocWriteFlushEvent,
    LocalMemWriteEvent,
    LocalMemReadEvent,
    UnknownNocEvent>;

enum class NOCDebugIssueType : uint8_t {
    // Write with missing flush or barrier at the source core.
    WRITE_FLUSH_BARRIER,
    // Read with missing barrier at the destination core.
    READ_BARRIER,
    // Number of issue types
    COUNT,
};

struct NOCDebugIssue {
    std::bitset<static_cast<size_t>(NOCDebugIssueType::COUNT)> issue_bits;

    NOCDebugIssue() { issue_bits.reset(); }

    void set_issue(NOCDebugIssueType issue_type) { issue_bits[static_cast<size_t>(issue_type)] = true; }

    bool has_issue(NOCDebugIssueType issue_type) const { return issue_bits[static_cast<size_t>(issue_type)]; }

    bool any_issue() const { return issue_bits.any(); }
};

class NOCDebugState {
public:
    // Add an event
    void push_event(size_t chip_id, uint64_t timestamp, int processor_id, const NOCDebugEvent& event);

    // Get the issue reported for a given core and processor during the lifetime of the debug state
    NOCDebugIssue get_issues(tt_cxy_pair core, int processor_id) const;

    // Reset the debug state for all cores and clear all issues
    void reset_state();

private:
    struct CoreDebugState {
        static constexpr size_t MAX_PROCESSORS = 5;
        static constexpr size_t MAX_NOCS = 2;

        // Counter snapshots for each NOC
        std::array<uint32_t, MAX_NOCS> read_counter_snapshot{};
        std::array<uint32_t, MAX_NOCS> nonposted_write_counter_snapshot{};
        std::array<uint32_t, MAX_NOCS> posted_write_counter_snapshot{};

        // Pending addresses not flushed yet for each NOC
        std::array<std::map<uint32_t, uint32_t>, MAX_NOCS> reads_not_flushed{};
        std::array<std::map<uint32_t, uint32_t>, MAX_NOCS> posted_writes_not_flushed{};
        std::array<std::map<uint32_t, uint32_t>, MAX_NOCS> nonposted_writes_not_flushed{};

        // Captures if any read or write has occured yet for each NOC
        std::array<bool, MAX_NOCS> any_reads{};
        std::array<bool, MAX_NOCS> any_posted_writes{};
        std::array<bool, MAX_NOCS> any_nonposted_writes{};

        // Latest RISC timestamp for each processor
        std::array<uint64_t, MAX_PROCESSORS> latest_risc_timestamp{};

        // Keep track of reported issues for each processor
        std::array<NOCDebugIssue, MAX_PROCESSORS> issue{};

        void insert_read_not_flushed(uint8_t noc, uint32_t start_addr, uint32_t end_addr) {
            detail::insert_range(reads_not_flushed[noc], start_addr, end_addr);
        }

        void insert_posted_write_not_flushed(uint8_t noc, uint32_t start_addr, uint32_t end_addr) {
            detail::insert_range(posted_writes_not_flushed[noc], start_addr, end_addr);
        }

        void insert_nonposted_write_not_flushed(uint8_t noc, uint32_t start_addr, uint32_t end_addr) {
            detail::insert_range(nonposted_writes_not_flushed[noc], start_addr, end_addr);
        }

        void flush_reads(uint8_t noc) { reads_not_flushed[noc].clear(); }

        void flush_posted_writes(uint8_t noc) { posted_writes_not_flushed[noc].clear(); }

        void flush_nonposted_writes(uint8_t noc) { nonposted_writes_not_flushed[noc].clear(); }

        bool overlaps_reads(uint8_t noc, uint32_t start_addr, uint32_t end_addr) const {
            return detail::check_overlap(reads_not_flushed[noc], start_addr, end_addr);
        }

        bool overlaps_posted_writes(uint8_t noc, uint32_t start_addr, uint32_t end_addr) const {
            return detail::check_overlap(posted_writes_not_flushed[noc], start_addr, end_addr);
        }

        bool overlaps_nonposted_writes(uint8_t noc, uint32_t start_addr, uint32_t end_addr) const {
            return detail::check_overlap(nonposted_writes_not_flushed[noc], start_addr, end_addr);
        }
    };

    void handle_write_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocWriteEvent event);
    void handle_read_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocReadEvent event);
    void handle_read_barrier_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocReadBarrierEvent event);
    void handle_write_barrier_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocWriteBarrierEvent event);
    void handle_write_flush_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocWriteFlushEvent event);

    void handle_local_mem_read_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, LocalMemReadEvent event);
    void handle_local_mem_write_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, LocalMemWriteEvent event);

    void update_latest_risc_timestamp(tt_cxy_pair core, int processor_id, uint64_t timestamp);

    CoreDebugState& get_state(tt_cxy_pair core);

    const CoreDebugState& get_state(tt_cxy_pair core) const;

    mutable std::unordered_map<tt_cxy_pair, CoreDebugState> cores;
    mutable std::mutex cores_mutex;
};

}  // namespace tt::tt_metal
