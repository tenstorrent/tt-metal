// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/service/inter_process_counter_channel.hpp>

#include "inter_process_counter_layout.hpp"
#include <tt-metalium/experimental/sockets/shm_resource_tracker.hpp>

#include <tt-logger/tt-logger.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace tt::tt_metal::distributed {

namespace {

// posix_errno_str — strerror() for errno at call time. Kept in this TU
// so the header doesn't have to include <cerrno> / <cstring>.
std::string posix_errno_str() { return std::strerror(errno); }

[[noreturn]] void throw_posix(const std::string& op, const std::string& detail) {
    throw std::runtime_error("InterProcessCounterChannel: " + op + " failed (" + detail + "): " + posix_errno_str());
}

// Creates the segment name exclusively. If the name is already taken, the
// tracker's stale scan runs once and the open is retried: a copy left by a
// killed predecessor is listed in that predecessor's manifest and gets
// reaped, whereas a live owner's segment survives the scan and the retry
// fails with EEXIST like the first attempt.
int open_fresh_counter_segment(const std::string& shm_path) {
    constexpr int kFlags = O_CREAT | O_EXCL | O_RDWR;
    int fd = ::shm_open(shm_path.c_str(), kFlags, S_IRUSR | S_IWUSR);
    if (fd == -1 && errno == EEXIST) {
        ShmResourceTracker::cleanup_stale_resources();
        fd = ::shm_open(shm_path.c_str(), kFlags, S_IRUSR | S_IWUSR);
    }
    return fd;
}

}  // namespace

// =============================================================================
// Owner-side construction.
//
// Creates the SHM segment fresh:
//   shm_open(O_CREAT|O_EXCL|O_RDWR)   ← fails if a stale segment exists
//   ftruncate to sizeof(InterProcessCounterSegment)
//   mmap PROT_READ|PROT_WRITE, MAP_SHARED
//   register with ShmResourceTracker
//
// The registration is what ties the segment's lifetime to the owner
// process: the tracker unlinks it on SIGINT/SIGTERM and at exit, and
// lists it in this process's manifest so the next owner reaps it if
// this process is killed outright (SIGKILL, OOM): the exclusive open
// below runs the stale scan when it finds the name taken.
//
// On any failure mid-way, undo what was done so we don't leak a half-
// initialised segment on /dev/shm.
// =============================================================================
InterProcessCounterChannel::InterProcessCounterChannel(const std::string& shm_name) :
    shm_path_(shm_name), role_(Role::Owner), fd_(open_fresh_counter_segment(shm_path_)) {
    if (fd_ == -1) {
        // EEXIST here means the name is held by a segment the stale scan
        // could not attribute to a dead process (a live owner, or a pid
        // in use again); the owner is responsible for unlinking it before
        // re-creating.
        throw_posix("shm_open(O_CREAT|O_EXCL)", shm_path_);
    }

    // Size the new segment to the layout struct exactly. POSIX
    // guarantees the new region is zero-filled.
    if (::ftruncate(fd_, sizeof(InterProcessCounterSegment)) != 0) {
        const int saved = errno;
        ::close(fd_);
        ::shm_unlink(shm_path_.c_str());
        errno = saved;
        fd_ = -1;
        throw_posix("ftruncate", shm_path_);
    }

    void* mapped = ::mmap(
        nullptr,
        sizeof(InterProcessCounterSegment),
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd_,
        /*offset=*/0);
    if (mapped == MAP_FAILED) {
        const int saved = errno;
        ::close(fd_);
        ::shm_unlink(shm_path_.c_str());
        errno = saved;
        fd_ = -1;
        throw_posix("mmap", shm_path_);
    }
    seg_ = static_cast<InterProcessCounterSegment*>(mapped);
    ::close(fd_);
    fd_ = -1;

    // Initial state — producer_counter=0, consumer_cursor=0 — is
    // guaranteed by POSIX ftruncate.
    //
    // prior_clean_shutdown is explicitly stamped to 1 so the first
    // connector's `had_clean_prior_shutdown()` returns true — there
    // was no predecessor to have exited uncleanly.
    seg_->prior_clean_shutdown = 1;

    // Last step; if registering throws, undo like the earlier failure paths so
    // no unregistered segment is left behind.
    try {
        ShmResourceTracker::instance().track_shm(shm_path_);
    } catch (...) {
        ::munmap(seg_, sizeof(InterProcessCounterSegment));
        seg_ = nullptr;
        ::shm_unlink(shm_path_.c_str());
        throw;
    }
}

// =============================================================================
// Connector-side factory.
//
// Polls shm_open until the owner has exported the segment or
// connect_timeout_ms elapses. On attach, reads
// prior_clean_shutdown and resets it to 0 (so this connector's own
// dtor write is the only bit the NEXT connector sees).
// =============================================================================
std::unique_ptr<InterProcessCounterChannel> InterProcessCounterChannel::connect(
    const std::string& shm_name, uint32_t connect_timeout_ms) {
    if (shm_name.empty() || shm_name[0] != '/' || shm_name.find('/', 1) != std::string::npos) {
        throw std::runtime_error(
            "InterProcessCounterChannel::connect: shm_name must start with '/' and contain no other '/'");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(connect_timeout_ms);
    int fd = -1;
    while (true) {
        fd = ::shm_open(shm_name.c_str(), O_RDWR, 0);
        if (fd != -1) {
            break;
        }
        if (errno != ENOENT) {
            // Anything other than "not found yet" is fatal — bad
            // permissions, EMFILE, etc.
            throw_posix("shm_open(O_RDWR)", shm_name);
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error(
                "InterProcessCounterChannel::connect: timed out after " + std::to_string(connect_timeout_ms) +
                " ms waiting for " + shm_name + " — owner process has not exported this shm_name");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    void* mapped = ::mmap(
        nullptr,
        sizeof(InterProcessCounterSegment),
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        /*offset=*/0);
    if (mapped == MAP_FAILED) {
        const int saved = errno;
        ::close(fd);
        errno = saved;
        throw_posix("mmap", shm_name);
    }
    auto* seg = static_cast<InterProcessCounterSegment*>(mapped);

    // Read prior_clean_shutdown ONCE — this is our snapshot of the
    // predecessor's exit. Clear it so the next connector's snapshot
    // captures THIS connector's exit, not the entire history.
    //
    // No atomic semantics needed: the prior connector has fully
    // exited (process termination is a hard barrier) before we
    // attach, so the value is durable; only one connector at a time
    // accesses this field, so the read-and-clear is sequential.
    const bool had_clean_prior_shutdown = (seg->prior_clean_shutdown != 0);
    seg->prior_clean_shutdown = 0;

    // consumer_cursor is left UNTOUCHED — it represents the
    // high-water mark of events already observed by the system,
    // regardless of whether the predecessor exited cleanly.

    // Private ctor — make_unique can't reach a private constructor so
    // we wrap with `new` here. Owned via unique_ptr from this point on.
    return std::unique_ptr<InterProcessCounterChannel>(
        new InterProcessCounterChannel(shm_name, Role::Connector, fd, seg, had_clean_prior_shutdown));
}

// Private ctor used only by connect() above.
InterProcessCounterChannel::InterProcessCounterChannel(
    std::string shm_path, Role role, int fd, InterProcessCounterSegment* seg, bool had_clean_prior_shutdown) :
    shm_path_(std::move(shm_path)),
    role_(role),
    fd_(fd),
    seg_(seg),
    had_clean_prior_shutdown_(had_clean_prior_shutdown) {}

// =============================================================================
// Destructor — calls shutdown() if not already. Idempotent.
// =============================================================================
InterProcessCounterChannel::~InterProcessCounterChannel() {
    // Dtors must not throw. If shutdown() fails (e.g. mmap is already gone for
    // some external reason) we log and swallow — the OS will reclaim our
    // handles on process exit anyway.
    try {
        shutdown();
    } catch (const std::exception& e) {
        log_warning(LogMetal, "InterProcessCounterChannel destructor: shutdown failed: {}", e.what());
    } catch (...) {
        log_warning(LogMetal, "InterProcessCounterChannel destructor: shutdown failed with unknown exception");
    }
}

// =============================================================================
// CounterChannel API — role-dispatched.
// =============================================================================

void InterProcessCounterChannel::inject(uint32_t n) {
    assert_role(Role::Owner, "inject");
    // memory_order_release pairs with the connector's acquire-load in
    // try_consume_all / pending. Producer is sole writer of this
    // field.
    seg_->producer_counter.fetch_add(n, std::memory_order_release);
}

uint32_t InterProcessCounterChannel::try_consume_all() {
    assert_role(Role::Connector, "try_consume_all");
    const uint32_t produced = seg_->producer_counter.load(std::memory_order_acquire);
    const uint32_t cursor = seg_->consumer_cursor;
    // u32 unsigned subtraction handles producer wrap-around mod 2^32
    // correctly as long as the unconsumed delta stays below 2^32 —
    // trivially satisfied since the consumer polls in the hot loop.
    const uint32_t delta = produced - cursor;
    if (delta == 0) {
        return 0;
    }
    // Advance the cursor BEFORE returning. If this connector crashes
    // between the advance and the caller acting on the return value,
    // the events are accounted for as "observed" by the system — they
    // do NOT replay for the next connector. The cursor is the truth.
    seg_->consumer_cursor = produced;
    return delta;
}

uint32_t InterProcessCounterChannel::pending() const {
    assert_role(Role::Connector, "pending");
    const uint32_t produced = seg_->producer_counter.load(std::memory_order_acquire);
    return produced - seg_->consumer_cursor;
}

bool InterProcessCounterChannel::had_clean_prior_shutdown() const {
    assert_role(Role::Connector, "had_clean_prior_shutdown");
    return had_clean_prior_shutdown_;
}

// =============================================================================
// shutdown() — idempotent, role-dispatched.
//
//   * Owner    : munmap → close(fd) → shm_unlink → untrack (only if
//                the name is gone). Segment removed from /dev/shm; any
//                still-attached connector's
//                mapping survives until that connector itself unmaps
//                (POSIX semantics), but no new connector can find
//                the name.
//   * Connector: write prior_clean_shutdown=1 → munmap → close(fd).
//                Segment is NOT unlinked — only the owner can
//                release the /dev/shm name. The next connector reads
//                the flag we just wrote.
// =============================================================================
void InterProcessCounterChannel::shutdown() {
    if (shutdown_called_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }

    if (seg_ != nullptr) {
        if (role_ == Role::Connector) {
            // Signal to the next connector that we exited cleanly.
            // Sequential access — no atomic semantics needed; the
            // next connector reads only after our process has fully
            // exited (or we've called munmap, whichever comes
            // first).
            seg_->prior_clean_shutdown = 1;
        }
        ::munmap(seg_, sizeof(InterProcessCounterSegment));
        seg_ = nullptr;
    }
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
    if (role_ == Role::Owner) {
        // Drop the registration only once the name is gone (unlinked here
        // or already removed by someone else). If the unlink fails for any
        // other reason the segment stays tracked, so the tracker's exit and
        // signal paths retry it and the manifest still names it for the
        // next owner's stale scan, as NamedShm::unlink does.
        if (::shm_unlink(shm_path_.c_str()) == 0 || errno == ENOENT) {
            ShmResourceTracker::instance().untrack_shm(shm_path_);
        }
    }
}

// =============================================================================
// Role check — throws with a useful message if a caller violates
// the role separation (e.g. inject on a connector).
// =============================================================================
void InterProcessCounterChannel::assert_role(Role expected, const char* op) const {
    if (role_ != expected) {
        const char* current = (role_ == Role::Owner) ? "owner" : "connector";
        const char* needed = (expected == Role::Owner) ? "owner" : "connector";
        throw std::runtime_error(
            std::string("InterProcessCounterChannel::") + op + ": called on a " + current +
            " instance but requires a " + needed + " instance");
    }
}

}  // namespace tt::tt_metal::distributed
