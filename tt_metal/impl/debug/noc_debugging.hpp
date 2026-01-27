// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bitset>
#include <cstdint>
#include <mutex>
#include <umd/device/types/xy_pair.hpp>
#include <unordered_map>
#include <array>
#include <tools/profiler/event_metadata.hpp>
#include <unordered_set>
#include <string_view>

namespace tt::tt_metal {

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
    bool is_semaphore;
    bool is_mcast;
    int8_t mcast_end_dst_x;
    int8_t mcast_end_dst_y;
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

struct UnknownNocEvent {};

using NOCDebugEvent = std::variant<
    NocWriteEvent,
    NocReadEvent,
    NocReadBarrierEvent,
    NocWriteBarrierEvent,
    NocWriteFlushEvent,
    UnknownNocEvent>;

enum class NOCDebugIssueType : uint8_t {
    // Write with missing flush or barrier at the source core (by write type)
    WRITE_FLUSH_BARRIER,
    WRITE_FLUSH_BARRIER_MCAST,
    WRITE_FLUSH_BARRIER_SEMAPHORE,
    WRITE_FLUSH_BARRIER_SEMAPHORE_MCAST,
    // Read with missing barrier at the destination core.
    READ_BARRIER,
    // Unflushed writes at end of kernel execution (by write type)
    UNFLUSHED_WRITE_AT_END,
    UNFLUSHED_WRITE_MCAST_AT_END,
    UNFLUSHED_SEMAPHORE_AT_END,
    UNFLUSHED_SEMAPHORE_MCAST_AT_END,
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

    // Print aggregated errors summary (grouped by error type with affected cores)
    void print_aggregated_errors() const;

    // This should be called after kernels are done (Finish()). It will check for unflushed reads/writes at the end of
    // the kernel.
    void finish_cores();

    // Tracks info about a pending write for end-of-kernel checking
    struct PendingWriteInfo {
        int processor_id = 0;
        bool is_semaphore = false;
        bool is_mcast = false;
    };

private:
    struct CoreDebugState {
        static constexpr size_t MAX_PROCESSORS = 5;
        static constexpr size_t MAX_NOCS = 2;

        // Counter snapshots for each NOC
        std::array<uint32_t, MAX_NOCS> read_counter_snapshot{};
        std::array<uint32_t, MAX_NOCS> nonposted_write_counter_snapshot{};
        std::array<uint32_t, MAX_NOCS> posted_write_counter_snapshot{};

        // Pending reads not flushed yet for each NOC (dst_addr set)
        std::array<std::unordered_set<uint32_t>, MAX_NOCS> reads_not_flushed{};

        // Pending writes not flushed yet for each NOC (src_addr -> write type info)
        std::array<std::unordered_map<uint32_t, PendingWriteInfo>, MAX_NOCS> posted_writes_pending{};
        std::array<std::unordered_map<uint32_t, PendingWriteInfo>, MAX_NOCS> nonposted_writes_pending{};

        // Captures if any read or write has occurred yet for each NOC
        std::array<bool, MAX_NOCS> any_reads{};
        std::array<bool, MAX_NOCS> any_posted_writes{};
        std::array<bool, MAX_NOCS> any_nonposted_writes{};

        // Latest RISC timestamp for each processor
        std::array<uint64_t, MAX_PROCESSORS> latest_risc_timestamp{};

        // Keep track of reported issues for each processor
        std::array<NOCDebugIssue, MAX_PROCESSORS> issue{};
    };

    void handle_write_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocWriteEvent event);
    void handle_read_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocReadEvent event);
    void handle_read_barrier_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocReadBarrierEvent event);
    void handle_write_barrier_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocWriteBarrierEvent event);
    void handle_write_flush_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocWriteFlushEvent event);

    void update_latest_risc_timestamp(tt_cxy_pair core, int processor_id, uint64_t timestamp);

    CoreDebugState& get_state(tt_cxy_pair core);

    const CoreDebugState& get_state(tt_cxy_pair core) const;

    static std::string_view get_issue_description(NOCDebugIssueType issue_type);

    mutable std::unordered_map<tt_cxy_pair, CoreDebugState> cores;
    mutable std::mutex cores_mutex;
};

}  // namespace tt::tt_metal
