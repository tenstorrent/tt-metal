// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sys/types.h>

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

namespace tt::tt_metal::internal::drisc_service {

// One DRISC. All seats of a chip share one file in $TT_DRISC_SEAT_LOCK_DIR (default /dev/shm/tt-drisc-seats),
// chip-<asic id>.lock. Seat i = noc0_y * 256 + noc0_x owns:
//   lock byte 2i      a write lock by an exclusive owner (tt-metal's DRAM programmable cores) or a sole client; a read
//                     lock while clients share it.
//   lock byte 2i + 1  the gate: a write lock for the length of one client's attach.
//   data bytes 4i..   the last client generation handed out.
// The locks are Linux open file description locks, taken through one open of the file per process, so a process
// holds one file open per chip however many seats it holds. Holders in one process exclude each other through a
// table in the process, as holders in different processes do through the locks. A process's locks last until it
// releases them or exits. A child forked without exec neither takes nor releases its parent's locks, but shares the
// parent's open file, so a lock the parent never released lasts until the child is gone too.
struct SeatId {
    uint64_t asic_id;
    uint32_t noc0_x;
    uint32_t noc0_y;
};

// A claim on a DRISC through its lock byte. Movable, not copyable. Two kinds, each from its own factory:
// - exclusive, tt-metal's for its DRAM programmable cores: a write lock for as long as it is held;
// - client, a service client's: a write lock while it is the seat's only client, a read lock once share() lets others
//   join. Take it under the SeatGate so sole() cannot go stale while the client acts on it.
class SeatLock {
public:
    // Returns nullopt when anyone else, in this process or another, holds the seat. Creates the lock dir (mode 1777)
    // and file (mode 0666) when missing, leaving the umask alone. Throws when the dir or file cannot be created or
    // opened, the file is a symlink, or locking fails for a reason other than contention.
    static std::optional<SeatLock> try_acquire_exclusive(const SeatId& seat);

    // A write lock when no other client holds the seat, else a read lock. Returns nullopt when an exclusive owner, or
    // a client that has not shared yet, holds it. Throws like try_acquire_exclusive.
    static std::optional<SeatLock> try_acquire_client(const SeatId& seat);

    // For a client lock, true from a write lock acquire until share(). Always false for an exclusive lock.
    bool sole() const { return kind_ == Kind::sole_client; }

    // Turns a client's write lock into a read lock in one step, so the seat is never free in between. Throws for an
    // exclusive lock. A no-op once shared.
    void share();

    ~SeatLock();
    SeatLock(SeatLock&& other) noexcept;
    SeatLock& operator=(SeatLock&& other) noexcept;
    SeatLock(const SeatLock&) = delete;
    SeatLock& operator=(const SeatLock&) = delete;

private:
    enum class Kind : uint8_t { exclusive, sole_client, shared_client };

    SeatLock(std::string path, uint32_t index, Kind kind);
    void release();

    std::string path_;  // Empty once moved from.
    uint32_t index_;
    Kind kind_;
    pid_t pid_;  // The process that took it: a forked child must not release it.
};

// A write lock on the seat's gate byte, so one client attaches at a time. Not copyable or movable.
class SeatGate {
public:
    // Waits up to `timeout` for the gate. Throws on timeout, or when the file cannot be created, opened or locked.
    SeatGate(const SeatId& seat, std::chrono::milliseconds timeout);
    ~SeatGate();
    SeatGate(const SeatGate&) = delete;
    SeatGate& operator=(const SeatGate&) = delete;
    SeatGate(SeatGate&&) = delete;
    SeatGate& operator=(SeatGate&&) = delete;

    // The seat's next client generation, in [1, max]: one past the last one handed out, wrapping from `max` to 1.
    // Throws when the file cannot be read or written.
    uint32_t next_gen(uint32_t max);

private:
    std::string path_;
    uint32_t index_;
    pid_t pid_;
};

}  // namespace tt::tt_metal::internal::drisc_service
