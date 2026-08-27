// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bitset>
#include <compare>
#include <cstdint>
#include <map>
#include <mutex>
#include <umd/device/types/xy_pair.hpp>
#include <unordered_map>
#include <array>
#include <tools/profiler/event_metadata.hpp>
#include <tools/profiler/noc_debugging_metadata.hpp>
#include <unordered_set>
#include <set>
#include <string>
#include <vector>

namespace tt::tt_metal {

struct NocWriteEvent {
    uint64_t src_addr;
    uint64_t dst_addr;
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
    bool has_source_buffer = true;  // False for writes that carry no L1 source buffer
    bool has_valid_dst = true;      // False for stateful writes (WRITE_WITH_STATE / WRITE_WITH_TRID_WITH_STATE)
};

struct NocWriteSetStateEvent {
    uint64_t dst_addr;
    uint32_t num_bytes;  // 0 for the trid variant (its size is supplied at the with-state call)
    int8_t src_x;
    int8_t src_y;
    int8_t dst_x;
    int8_t dst_y;
    bool is_mcast;
    int8_t mcast_end_dst_x;
    int8_t mcast_end_dst_y;
    uint8_t noc;
};

struct NocReadEvent {
    uint64_t src_addr;
    uint64_t dst_addr;
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

// A full barrier (noc_async_full_barrier) waits for all outstanding reads, writes AND atomics to complete, so it
// clears every pending set for the noc.
struct NocFullBarrierEvent {
    int8_t src_x;
    int8_t src_y;
    uint8_t noc;
};

// A remote atomic increment (noc_semaphore_inc / noc_semaphore_inc_multicast). Unlike a write it carries no source
// buffer (the increment value is immediate) and does not advance the NIU write counter, so the source-reuse and
// counter-monotonicity checks do not apply. Only a non-posted increment must be flushed before kernel end. For a
// multicast increment dst_x/dst_y are the rectangle start and mcast_end_dst_x/y the end.
struct NocSemaphoreIncEvent {
    uint64_t dst_addr;
    int8_t src_x;
    int8_t src_y;
    int8_t dst_x;
    int8_t dst_y;
    bool posted;
    uint8_t noc;
    bool is_mcast;
    int8_t mcast_end_dst_x;
    int8_t mcast_end_dst_y;
};

// An atomic barrier (noc_async_atomic_barrier) waits only for outstanding atomics; on device it uses a counter
// separate from writes, so it clears the atomics pending set for the noc but leaves reads/writes untouched.
struct NocAtomicBarrierEvent {
    int8_t src_x;
    int8_t src_y;
    uint8_t noc;
};

struct UnknownNocEvent {};

struct ScopedLockEvent {
    int8_t src_x;
    int8_t src_y;
    NocDebuggingEventMetadata::NocDebugEventType event_type;
    uint64_t locked_address_base;
    uint32_t num_bytes;

    bool is_lock() const {
        return event_type == NocDebuggingEventMetadata::NocDebugEventType::CB_LOCK ||
               event_type == NocDebuggingEventMetadata::NocDebugEventType::MEM_LOCK ||
               event_type == NocDebuggingEventMetadata::NocDebugEventType::DFB_LOCK;
    }
};

using NOCDebugEvent = std::variant<
    NocWriteEvent,
    NocWriteSetStateEvent,
    NocReadEvent,
    NocReadBarrierEvent,
    NocWriteBarrierEvent,
    NocWriteFlushEvent,
    NocFullBarrierEvent,
    NocSemaphoreIncEvent,
    NocAtomicBarrierEvent,
    ScopedLockEvent,
    UnknownNocEvent>;

enum class NOCDebugIssueBaseType : uint8_t {
    WRITE_FLUSH_BARRIER,
    READ_BARRIER,
    UNFLUSHED_WRITE_AT_END,
    WRITE_TO_LOCKED_CORE_LOCAL_MEM,
    WRITE_TO_LOCKED_CB,
    WRITE_TO_LOCKED_DFB,
    WRITE_TO_UNLOCKED_DFB,
    COUNT,
};

// TODO: Move metadata out into a variant so we can have different metadata for each issue types
struct NOCDebugIssueType {
    NOCDebugIssueBaseType base_type = NOCDebugIssueBaseType::WRITE_FLUSH_BARRIER;
    uint64_t issue_address = 0;  // The destination address of the violating NOC transaction
    uint32_t issue_size = 0;     // The size of the violating NOC transaction in bytes
    uint8_t src_x = 0;
    uint8_t src_y = 0;
    uint8_t dst_x = 0;
    uint8_t dst_y = 0;
    bool is_mcast : 1 = false;      // True if the issue involved a multicast
    bool is_semaphore : 1 = false;  // True if the issue involved a semaphore operation

    NOCDebugIssueType() = default;

    NOCDebugIssueType(NOCDebugIssueBaseType type, bool mcast = false, bool semaphore = false) :
        base_type(type), is_mcast(mcast), is_semaphore(semaphore) {}

    auto operator<=>(const NOCDebugIssueType& other) const = default;
};

struct NOCDebugIssue {
    std::set<NOCDebugIssueType> issues;

    void set_issue(const NOCDebugIssueType& issue_type) { issues.insert(issue_type); }

    bool has_issue(const NOCDebugIssueType& issue_type) const { return issues.contains(issue_type); }

    bool any_issue() const { return !issues.empty(); }

    // Check if any issue with a specific base type exists
    bool has_base_issue(NOCDebugIssueBaseType base_type) const {
        for (const auto& issue : issues) {
            if (issue.base_type == base_type) {
                return true;
            }
        }
        return false;
    }

    // Get all issues with a specific base type
    std::vector<NOCDebugIssueType> get_issues_by_base(NOCDebugIssueBaseType base_type) const {
        std::vector<NOCDebugIssueType> result;
        result.reserve(issues.size());
        for (const auto& issue : issues) {
            if (issue.base_type == base_type) {
                result.push_back(issue);
            }
        }
        return result;
    }
};

class NOCDebugState {
public:
    // Accumulate an event (processed in timestamp order when process_accumulated_events is called)
    void push_event(size_t chip_id, uint64_t timestamp, int processor_id, const NOCDebugEvent& event);

    // Sort accumulated events by timestamp, process them, then clear the queue. Call after a poll is complete.
    void process_accumulated_events_all_chips();

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

    struct L1Extent {
        // 64-bit to match the address fields on the events these extents are compared against (see NocWriteEvent).
        uint64_t address;
        uint32_t size;
        auto operator<=>(const L1Extent&) const = default;
    };

    struct LockedBufferInfo {
        enum class LockType {
            CB,
            MEM,
            DFB,
        };

        L1Extent extent;
        LockType lock_type;

        auto operator<=>(const LockedBufferInfo& other) const = default;
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
        std::array<std::unordered_set<uint64_t>, MAX_NOCS> reads_not_flushed{};

        // Pending writes not flushed yet for each NOC (src_addr -> write type info)
        std::array<std::unordered_map<uint64_t, PendingWriteInfo>, MAX_NOCS> posted_writes_pending{};
        std::array<std::unordered_map<uint64_t, PendingWriteInfo>, MAX_NOCS> nonposted_writes_pending{};

        // Pending non-posted atomic increments (semaphore inc) not yet flushed for each NOC (dst_addr -> info).
        // Kept separate from writes because on device atomics use their own counter: they are released by an
        // atomic/full barrier, never by a write barrier.
        std::array<std::unordered_map<uint64_t, PendingWriteInfo>, MAX_NOCS> atomics_pending{};

        // Captures if any read or write has occurred yet for each NOC
        std::array<bool, MAX_NOCS> any_reads{};
        std::array<bool, MAX_NOCS> any_posted_writes{};
        std::array<bool, MAX_NOCS> any_nonposted_writes{};

        // Buffers currently locked by each RISC, reference-counted so nested or duplicate locks over the same
        // region are only released once the matching number of unlocks arrive.
        std::array<std::map<LockedBufferInfo, uint32_t>, MAX_PROCESSORS> locked_buffers{};

        // Live DFB L1 extents on each RISC. A DFB's constructor declares its ring extent, and the firmware
        // clears this RISC's whole set once the kernel exits.
        std::array<std::set<L1Extent>, MAX_PROCESSORS> dfb_regions{};

        // Destination programmed by the most recent WRITE_SET_STATE per (processor, noc). Subsequent stateful writes
        // (WRITE_WITH_STATE / WRITE_WITH_TRID_WITH_STATE), whose own event records the destination core as a
        // placeholder, resolve their real destination from here.
        struct WriteStateInfo {
            uint64_t dst_addr = 0;  // programmed destination address; supplies the bits above the low word
            uint32_t num_bytes = 0;
            int8_t dst_x = 0;
            int8_t dst_y = 0;
            int8_t mcast_end_dst_x = 0;
            int8_t mcast_end_dst_y = 0;
            bool is_mcast = false;
            bool valid = false;  // false until a WRITE_SET_STATE has been seen on this (processor, noc)
        };
        std::array<std::array<WriteStateInfo, MAX_NOCS>, MAX_PROCESSORS> current_write_state{};

        // Latest RISC timestamp for each processor
        std::array<uint64_t, MAX_PROCESSORS> latest_risc_timestamp{};

        // Keep track of reported issues for each processor
        std::array<NOCDebugIssue, MAX_PROCESSORS> issue{};

        // Check if a NOC write of [write_start, write_start + write_size) hit a locked buffer in this core.
        // The extent is passed explicitly rather than read from the event because a stateful write's real
        // destination is reassembled by the caller (see handle_write_event).
        const LockedBufferInfo* get_noc_write_to_lock_buffer(
            uint64_t write_start, uint32_t write_size, int writer_processor_id, bool same_core) const;

        // Check if the write lands in a unlocked live DFB region on this core
        bool write_into_unlocked_dfb(uint64_t write_start, uint32_t write_size, int writer_processor_id) const;
    };

    void handle_write_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocWriteEvent event);
    void handle_write_set_state_event(
        tt_cxy_pair core, int processor_id, uint64_t timestamp, NocWriteSetStateEvent event);
    void handle_read_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocReadEvent event);
    void handle_read_barrier_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocReadBarrierEvent event);
    void handle_write_barrier_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocWriteBarrierEvent event);
    void handle_write_flush_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocWriteFlushEvent event);
    void handle_full_barrier_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocFullBarrierEvent event);
    void handle_semaphore_inc_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, NocSemaphoreIncEvent event);
    void handle_atomic_barrier_event(
        tt_cxy_pair core, int processor_id, uint64_t timestamp, NocAtomicBarrierEvent event);
    void handle_scoped_lock_event(tt_cxy_pair core, int processor_id, uint64_t timestamp, ScopedLockEvent event);

    void update_latest_risc_timestamp(tt_cxy_pair core, int processor_id, uint64_t timestamp);

    CoreDebugState& get_state(tt_cxy_pair core);

    const CoreDebugState& get_state(tt_cxy_pair core) const;

    bool has_state(tt_cxy_pair core) const;

    static std::string get_issue_description(const NOCDebugIssueType& issue_type);

    struct PendingEvent {
        size_t chip_id;
        uint64_t timestamp;
        int processor_id;
        NOCDebugEvent event;
    };
    mutable std::vector<PendingEvent> pending_events_;
    mutable std::mutex pending_events_mutex_;

    mutable std::unordered_map<tt_cxy_pair, CoreDebugState> cores;
    mutable std::mutex cores_mutex;
};

}  // namespace tt::tt_metal
