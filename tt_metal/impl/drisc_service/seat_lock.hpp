// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

namespace tt::tt_metal::drisc_service {

// One DRISC, as named by its files in $KV_MANAGER_SEAT_LOCK_DIR (default /dev/shm/tt-kv), the dir tt-d-gen uses too:
//   dmk-seat-<asic id>-<noc0 x>-<noc0 y>.lock  LOCK_EX by an exclusive owner (tt-metal's DRAM programmable cores, or
//                                              tt-d-gen's own kernel until it adopts the service); LOCK_SH per
//                                              attached service client.
//   dmk-seat-<asic id>-<noc0 x>-<noc0 y>.gate  LOCK_EX for the length of one client's attach, and while tt-metal
//                                              takes the .lock.
//   dmk-seat-<asic id>-<noc0 x>-<noc0 y>.gen   the last client generation handed out.
// Each lock is an flock on its own open file description, so two holders in one process exclude each other the way
// two processes do. A lock lasts until every process sharing its file has closed it or exited, so a child forked
// without exec keeps its parent's locks.
struct SeatId {
    uint64_t asic_id;
    uint32_t noc0_x;
    uint32_t noc0_y;
};

// tt-metal's exclusive claim on a DRISC, for its DRAM programmable cores: LOCK_EX on the .lock file. Movable, not
// copyable.
class SeatLock {
public:
    // Returns nullopt when anyone else, in this process or another, holds the seat or its gate. Creates the
    // lock dir (mode 1777) and files when missing. Throws when the dir or a file cannot be created or opened, or flock
    // fails for a reason other than contention.
    static std::optional<SeatLock> try_acquire(const SeatId& seat);

    ~SeatLock();
    SeatLock(SeatLock&& other) noexcept;
    SeatLock& operator=(SeatLock&& other) noexcept;
    SeatLock(const SeatLock&) = delete;
    SeatLock& operator=(const SeatLock&) = delete;

private:
    explicit SeatLock(int fd) : fd_(fd) {}

    int fd_ = -1;  // -1 once moved from.
};

// A service client's claim on a DRISC: LOCK_EX while it is the seat's only client, LOCK_SH once share() lets others
// join. Take it under the SeatGate so sole() cannot go stale while the client acts on it. Movable, not copyable.
class ClientSeatLock {
public:
    // LOCK_EX when no other client holds the seat, else LOCK_SH. Returns nullopt when an exclusive owner holds it.
    // Throws like SeatLock::try_acquire.
    static std::optional<ClientSeatLock> try_acquire(const SeatId& seat);

    // True from a LOCK_EX acquire until share().
    bool sole() const { return sole_; }

    // Downgrades to LOCK_SH. Linux drops LOCK_EX before taking LOCK_SH. tt-metal cannot take the seat in between,
    // since it takes the gate first; tt-d-gen's own kernel does not take the gate, so it can, and then this throws,
    // holding nothing. A no-op once shared.
    void share();

    ~ClientSeatLock();
    ClientSeatLock(ClientSeatLock&& other) noexcept;
    ClientSeatLock& operator=(ClientSeatLock&& other) noexcept;
    ClientSeatLock(const ClientSeatLock&) = delete;
    ClientSeatLock& operator=(const ClientSeatLock&) = delete;

private:
    ClientSeatLock(int fd, bool sole) : fd_(fd), sole_(sole) {}

    int fd_ = -1;  // -1 once moved from.
    bool sole_ = false;
};

// LOCK_EX on the seat's .gate file, so one client attaches at a time. Not copyable or movable.
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
    // Throws when the .gen file cannot be opened, read or written.
    uint32_t next_gen(uint32_t max);

private:
    std::string gen_path_;
    int fd_ = -1;
};

}  // namespace tt::tt_metal::drisc_service
