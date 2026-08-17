// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Wire contract for the cross-process CounterChannel SHM segment.
//
// One POSIX shared-memory region per (owner, shm_name) carries this
// struct verbatim. The owner-side `InterProcessCounterChannel` creates
// and initialises it; one connector at a time attaches via shm_open
// + mmap, drains events, and detaches at shutdown. Successive
// connectors can sequentially attach to the SAME owner-side segment
// without losing progress — the cursor and clean-shutdown flag
// persist across connector lifetimes.
//
// Producer / consumer split:
//   * `producer_counter` — sole writer is the owner process;
//     concurrently read by the active connector while the producer
//     atomic-adds. This is the only field that needs atomicity. No
//     release/acquire pairing required; the counter carries no
//     "this-happens-before-that" semantics with any other field.
//   * `consumer_cursor`  — sole accessor is the active connector at
//     any moment. Reads + writes happen-before each other within one
//     thread. The NEXT connector reads after the previous one has
//     fully exited (process termination + reattach is a hard OS-level
//     barrier). No concurrent access ever ⇒ plain uint32_t.
//   * `prior_clean_shutdown` — same sequential-only access pattern.
//
// Persistence semantics:
//   The cursor is the high-water mark of events ALREADY observed by
//   the system. A new connector inherits whatever the previous one
//   left behind — even after a crash — because that value is the
//   truth of "what's been seen." Higher-layer code decides what to do
//   with events-pending-but-not-yet-consumed; the SHM layer imposes
//   no policy.
//
// Cache-line layout:
//   `producer_counter` and `consumer_cursor` sit on separate cache
//   lines so producer writes and connector writes don't false-share.
//   `prior_clean_shutdown` is cold (read at attach, written at clean
//   shutdown) so it packs onto the consumer's line at no cost.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace tt::tt_metal::distributed {

// Cache-line alignment for the SHM region — keeps each hot field on
// its own line so cross-process writes don't ping-pong unrelated state.
inline constexpr std::size_t kInterProcessCounterCacheLine = 64;

struct InterProcessCounterSegment {
    // Monotonic count of events emitted by the owner. Producer is the
    // sole writer (`fetch_add`); the active connector concurrently
    // reads. Atomic because that concurrency is real.
    //
    // u32 wraps at ~4B events, which doesn't happen in practice.
    // Wraparound semantics are correct under unsigned subtraction
    // (mod 2^32) as long as the unconsumed delta stays below 2^32 —
    // trivially satisfied when the consumer polls in a hot loop.
    //
    // Own cache line, isolated from consumer-side fields so the
    // producer's fetch_add doesn't invalidate the consumer's line on
    // every increment.
    alignas(kInterProcessCounterCacheLine) std::atomic<uint32_t> producer_counter;

    // High-water mark of events the system has already observed. Sole
    // accessor at any moment is the active connector. The NEXT
    // connector reads this only AFTER the previous one has fully
    // exited — process termination + reattach is a hard OS-level
    // memory barrier, so no atomic semantics are needed across the
    // handoff either.
    //
    // Own cache line for the same false-sharing reason as
    // producer_counter above.
    alignas(kInterProcessCounterCacheLine) uint32_t consumer_cursor;

    // Set to 1 by a cleanly-exiting connector in its destructor (and
    // by the owner ctor for the first-attach case); cleared back to 0
    // by the next connector at attach time. 0 at attach means the
    // prior connector crashed without running its dtor. Cold field
    // (read once / written once) — packs onto consumer_cursor's line
    // at no hot-path cost.
    uint32_t prior_clean_shutdown;
};

}  // namespace tt::tt_metal::distributed
