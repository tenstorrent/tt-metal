// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// InterProcessCounterChannel — POSIX-SHM-backed CounterChannel impl. One
// class, dual role (owner vs connector), modelled on
// `tt::tt_metal::distributed::H2DStreamService`:
//
//   * Constructor      → owner side. Creates the SHM segment under
//                        /dev/shm/<shm_name>, owns its lifetime,
//                        unlinks on shutdown.
//   * `connect(name)`  → connector side. Attaches to an owner-exported
//                        segment by the same shm_name, inherits the
//                        cursor + `prior_clean_shutdown` flag the
//                        previous connector left behind.
//
// The class is intentionally agnostic to WHAT is being counted —
// callers compute domain-specific /dev/shm names (e.g. layer-ack vs.
// migration-resp) and pass them in. The channel itself just operates
// on a /dev/shm path.
//
// Wire layout: see `inter_process_counter_layout.hpp`
// (`InterProcessCounterSegment`).
//
// Lifecycle:
//   * Owner    : ctor → inject*N → shutdown() (or dtor)
//                  shutdown ⇒ munmap + shm_unlink (segment removed
//                  from /dev/shm; in-flight connectors keep their
//                  mappings until they themselves munmap, but the
//                  name is gone).
//   * Connector: connect() → try_consume_all*N → shutdown() (or dtor)
//                  shutdown ⇒ write seg_->prior_clean_shutdown = 1 →
//                  munmap. The segment SURVIVES for the next
//                  connector (only the owner side ever unlinks).

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace tt::tt_metal::distributed {

// Forward decl. Definition lives in the .cpp's private companion
// header (`tt_metal/distributed/inter_process_counter_layout.hpp`);
// only the .cpp needs the full type so the public header stays light.
struct InterProcessCounterSegment;

class InterProcessCounterChannel {
public:
    // ===== Owner-side construction =====
    //
    // Creates /dev/shm/<shm_name> with the InterProcessCounterSegment
    // layout: shm_open(O_CREAT|O_EXCL|O_RDWR), ftruncate to
    // sizeof(InterProcessCounterSegment), mmap.
    //
    // Throws std::runtime_error if a segment with this shm_name
    // already exists. The owner is responsible for unlinking a stale
    // segment from a prior crashed run before constructing here.
    //
    // shm_name must be a POSIX-shm-valid string: leading '/' and no
    // other slashes.
    explicit InterProcessCounterChannel(const std::string& shm_name);

    // ===== Connector-side factory =====
    //
    // Attaches to an owner-created segment by the same shm_name.
    // Blocks up to `connect_timeout_ms` waiting for /dev/shm/<…> to
    // appear; throws std::runtime_error on timeout.
    //
    // At attach time, atomically reads `prior_clean_shutdown` and
    // resets it to 0 (so the value reported by
    // `had_clean_prior_shutdown()` reflects strictly the IMMEDIATELY
    // PREVIOUS connector, not any earlier chain). The
    // `consumer_cursor` is inherited as-is — it is the high-water
    // mark of events already observed by the system, and persistence
    // across connector lifetimes is the point of having it in SHM.
    static std::unique_ptr<InterProcessCounterChannel> connect(
        const std::string& shm_name, uint32_t connect_timeout_ms = 30'000);

    // Runs shutdown() if the caller didn't. Idempotent.
    ~InterProcessCounterChannel();

    InterProcessCounterChannel(const InterProcessCounterChannel&) = delete;
    InterProcessCounterChannel& operator=(const InterProcessCounterChannel&) = delete;
    InterProcessCounterChannel(InterProcessCounterChannel&&) = delete;
    InterProcessCounterChannel& operator=(InterProcessCounterChannel&&) = delete;

    // ===== CounterChannel-style API — role-dispatched =====

    // Owner-only. Atomic-add `n` to seg_->producer_counter. Throws
    // std::runtime_error if called on a connector instance.
    void inject(uint32_t n);

    // Connector-only. Returns producer_counter - consumer_cursor and
    // advances the cursor to the producer's head. Throws on owner.
    uint32_t try_consume_all();

    // Connector-only. Snapshot of producer_counter - consumer_cursor
    // (i.e. what the next try_consume_all would return).
    // Non-destructive. Throws on owner — pending is a consumer-facing
    // notion; the owner doesn't read consumer_cursor.
    uint32_t pending() const;

    // ===== Explicit teardown =====
    //
    // Idempotent. Role-dispatched:
    //   * owner     — munmap + shm_unlink (segment removed from
    //                 /dev/shm).
    //   * connector — write seg_->prior_clean_shutdown = 1 → munmap
    //                 (segment survives; next connector reads the
    //                 flag).
    // Subsequent calls on either side are no-ops. The dtor calls
    // this if the caller didn't.
    void shutdown();

    // ===== Connector diagnostics =====
    //
    // Snapshot of seg_->prior_clean_shutdown taken at connect() time
    // (before this connector cleared it). True iff the immediately-
    // previous connector exited cleanly via shutdown() (i.e. its
    // dtor ran to completion). False on first attach to a fresh
    // segment, or after a connector that crashed without running its
    // dtor.
    //
    // The SHM imposes no recovery policy — this is informational;
    // the caller decides what (if anything) to do about a non-clean
    // prior. Throws on owner.
    bool had_clean_prior_shutdown() const;

    // ===== Both roles =====

    // /dev/shm-relative path the segment lives at.
    const std::string& shm_name() const noexcept { return shm_path_; }

private:
    enum class Role : uint8_t { Owner, Connector };

    // Private ctor used by the static connect() factory after it has
    // shm_open'd + mmap'd the existing segment and read the
    // had_clean_prior_shutdown bit.
    InterProcessCounterChannel(
        std::string shm_path, Role role, int fd, InterProcessCounterSegment* seg, bool had_clean_prior_shutdown);

    // Throws std::runtime_error if this instance's role != expected,
    // tagged with `op` (the method name) for debuggability.
    void assert_role(Role expected, const char* op) const;

    std::string shm_path_;
    Role role_;
    int fd_ = -1;
    InterProcessCounterSegment* seg_ = nullptr;
    bool had_clean_prior_shutdown_ = false;

    std::atomic<bool> shutdown_called_{false};
};

}  // namespace tt::tt_metal::distributed
