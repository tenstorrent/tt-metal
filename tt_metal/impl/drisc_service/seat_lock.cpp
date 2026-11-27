// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/drisc_service/seat_lock.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::internal::drisc_service {

namespace {

// Creates `dir` with mode 01777 when missing. Shared between users, it gets its mode from chmod after it is created:
// chmod ignores the umask, and changing the umask instead would change it for every thread of the process. A process
// that dies between the two calls leaves it with the umask's mode, closed to other users until it is deleted.
void create_dir(const std::string& dir) {
    if (::mkdir(dir.c_str(), 01777) != 0) {
        TT_FATAL(
            errno == EEXIST,
            "drisc_service: cannot create seat lock dir {} ({}); set TT_DRISC_SEAT_LOCK_DIR to a writable directory",
            dir,
            std::strerror(errno));
        return;
    }
    TT_FATAL(::chmod(dir.c_str(), 01777) == 0, "drisc_service: cannot chmod {} ({})", dir, std::strerror(errno));
}

// Creates the lock dir when missing and returns the file that holds the seats of chip `asic_id`.
std::string chip_file(uint64_t asic_id) {
    const char* env = std::getenv("TT_DRISC_SEAT_LOCK_DIR");
    const std::string dir = (env != nullptr && *env != '\0') ? env : "/dev/shm/tt-drisc-seats";
    create_dir(dir);
    return fmt::format("{}/chip-{}.lock", dir, asic_id);
}

// A Blackhole NoC node id has 6 bits per coordinate (NOC_ADDR_NODE_ID_BITS), so NOC0 x and y are below this.
constexpr uint32_t noc0_coord_limit = 1u << 6;

// Each seat's locks: its lock byte, then its gate byte.
constexpr off_t lock_bytes_per_seat = 2;

// Each seat's data: its last client generation.
constexpr off_t data_bytes_per_seat = sizeof(uint32_t);

uint32_t seat_index(const SeatId& seat) {
    TT_FATAL(
        seat.noc0_x < noc0_coord_limit && seat.noc0_y < noc0_coord_limit,
        "drisc_service: seat ({}, {}) is outside the NoC grid",
        seat.noc0_x,
        seat.noc0_y);
    return (seat.noc0_y * noc0_coord_limit) + seat.noc0_x;
}

off_t lock_byte(uint32_t index) { return lock_bytes_per_seat * index; }
off_t gate_byte(uint32_t index) { return lock_byte(index) + 1; }
off_t gen_offset(uint32_t index) { return data_bytes_per_seat * index; }

// Opens `path` with `flags`, creating it with mode 0666 when missing, set the way create_dir sets the dir's. Never
// follows a symlink: whoever owns the dir could point a seat file at a file the caller can write. An existing file is
// opened without O_CREAT, which fs.protected_regular refuses in a sticky dir for a file another user owns.
int open_seat_file(const std::string& path, int flags) {
    while (true) {
        int fd = ::open(path.c_str(), flags | O_CLOEXEC | O_NOFOLLOW);
        if (fd >= 0) {
            return fd;
        }
        const int err = errno;
        TT_FATAL(err != ELOOP, "drisc_service: seat file {} is a symlink", path);
        TT_FATAL(
            err == ENOENT,
            "drisc_service: cannot open seat file {} ({}); if it is not mode 0666, delete it",
            path,
            std::strerror(err));
        fd = ::open(path.c_str(), flags | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0666);
        if (fd < 0 && errno == EEXIST) {
            continue;
        }
        TT_FATAL(
            fd >= 0,
            "drisc_service: cannot create seat file {} ({}); if its dir is not mode 1777, delete the dir",
            path,
            std::strerror(errno));
        if (::fchmod(fd, 0666) != 0) {
            const int chmod_err = errno;
            ::close(fd);
            TT_THROW("drisc_service: cannot chmod {} ({})", path, std::strerror(chmod_err));
        }
        return fd;
    }
}

// What this process holds on one seat.
struct SeatState {
    bool exclusive = false;  // An exclusive owner or a sole client.
    uint32_t shared = 0;     // Shared clients.
    bool gate = false;
};

// A chip file, open while this process holds anything in it.
struct ChipFile {
    int fd = -1;
    uint32_t holders = 0;  // Live SeatLocks and SeatGates on its seats.
    std::unordered_map<uint32_t, SeatState> seats;
};

struct Registry {
    pid_t pid = ::getpid();
    std::mutex mutex;
    std::unordered_map<std::string, ChipFile> chips;
};

// This process's holds. Never destroyed, so holders destroyed at exit can still reach it. A forked child starts a new
// one: its parent's holds are not the child's to change.
Registry& registry() {
    static std::atomic<Registry*> current{new Registry};
    Registry* reg = current.load();
    while (reg->pid != ::getpid()) {
        auto* fresh = new Registry;
        if (current.compare_exchange_strong(reg, fresh)) {
            reg = fresh;
        } else {
            delete fresh;
        }
    }
    return *reg;
}

// Opens the chip file at `path` unless this process already has it open.
ChipFile& open_chip(Registry& reg, const std::string& path) {
    auto it = reg.chips.find(path);
    if (it == reg.chips.end()) {
        it = reg.chips.emplace(path, ChipFile{.fd = open_seat_file(path, O_RDWR)}).first;
    }
    return it->second;
}

// Closes the chip file at `path`, dropping every lock taken through it, once nothing in this process holds it.
void close_if_unheld(Registry& reg, const std::string& path) {
    const auto it = reg.chips.find(path);
    if (it->second.holders == 0) {
        ::close(it->second.fd);
        reg.chips.erase(it);
    }
}

// Sets this process's lock on byte `offset` of `fd` to `type` (F_WRLCK, F_RDLCK or F_UNLCK) in one fcntl call, which
// turns a write lock into a read lock without unlocking the byte in between. False on failure, with errno set.
bool set_lock(int fd, off_t offset, short type) {
    struct flock lock{};
    lock.l_type = type;
    lock.l_whence = SEEK_SET;
    lock.l_start = offset;
    lock.l_len = 1;
    return ::fcntl(fd, F_OFD_SETLK, &lock) == 0;
}

// False on contention.
bool try_lock(int fd, off_t offset, short type, const std::string& path) {
    if (set_lock(fd, offset, type)) {
        return true;
    }
    TT_FATAL(errno == EAGAIN || errno == EACCES, "drisc_service: cannot lock {} ({})", path, std::strerror(errno));
    return false;
}

// The lock that this process's holds on a seat add up to.
short lock_type(const SeatState& state) {
    if (state.exclusive) {
        return F_WRLCK;
    }
    return state.shared > 0 ? F_RDLCK : F_UNLCK;
}

}  // namespace

std::optional<SeatLock> SeatLock::try_acquire_exclusive(const SeatId& seat) {
    std::string path = chip_file(seat.asic_id);
    const uint32_t index = seat_index(seat);
    Registry& reg = registry();
    const std::lock_guard guard(reg.mutex);
    ChipFile& chip = open_chip(reg, path);
    SeatState& state = chip.seats[index];
    if (state.exclusive || state.shared > 0 || !try_lock(chip.fd, lock_byte(index), F_WRLCK, path)) {
        close_if_unheld(reg, path);
        return std::nullopt;
    }
    state.exclusive = true;
    chip.holders++;
    return SeatLock(std::move(path), index, Kind::exclusive);
}

std::optional<SeatLock> SeatLock::try_acquire_client(const SeatId& seat) {
    std::string path = chip_file(seat.asic_id);
    const uint32_t index = seat_index(seat);
    Registry& reg = registry();
    const std::lock_guard guard(reg.mutex);
    ChipFile& chip = open_chip(reg, path);
    SeatState& state = chip.seats[index];
    Kind kind = Kind::shared_client;
    if (state.exclusive) {
        close_if_unheld(reg, path);
        return std::nullopt;
    }
    // A client of this process already shares the seat, so this process holds its read lock.
    if (state.shared == 0) {
        if (try_lock(chip.fd, lock_byte(index), F_WRLCK, path)) {
            kind = Kind::sole_client;
        } else if (!try_lock(chip.fd, lock_byte(index), F_RDLCK, path)) {
            close_if_unheld(reg, path);
            return std::nullopt;
        }
    }
    if (kind == Kind::sole_client) {
        state.exclusive = true;
    } else {
        state.shared++;
    }
    chip.holders++;
    return SeatLock(std::move(path), index, kind);
}

SeatLock::SeatLock(std::string path, uint32_t index, Kind kind) :
    path_(std::move(path)), index_(index), kind_(kind), pid_(::getpid()) {}

void SeatLock::share() {
    TT_FATAL(kind_ != Kind::exclusive, "drisc_service: an exclusive seat lock cannot be shared");
    if (kind_ == Kind::shared_client) {
        return;
    }
    TT_FATAL(pid_ == ::getpid(), "drisc_service: a seat lock taken before a fork cannot be shared after it");
    Registry& reg = registry();
    const std::lock_guard guard(reg.mutex);
    ChipFile& chip = reg.chips.at(path_);
    SeatState& state = chip.seats.at(index_);
    TT_FATAL(
        set_lock(chip.fd, lock_byte(index_), F_RDLCK),
        "drisc_service: cannot share {} ({})",
        path_,
        std::strerror(errno));
    state.exclusive = false;
    state.shared++;
    kind_ = Kind::shared_client;
}

void SeatLock::release() {
    if (path_.empty() || pid_ != ::getpid()) {
        return;
    }
    Registry& reg = registry();
    const std::lock_guard guard(reg.mutex);
    ChipFile& chip = reg.chips.at(path_);
    SeatState& state = chip.seats.at(index_);
    if (kind_ == Kind::shared_client) {
        state.shared--;
    } else {
        state.exclusive = false;
    }
    set_lock(chip.fd, lock_byte(index_), lock_type(state));
    chip.holders--;
    close_if_unheld(reg, path_);
    path_.clear();
}

SeatLock::~SeatLock() { release(); }

SeatLock::SeatLock(SeatLock&& other) noexcept :
    path_(std::exchange(other.path_, {})), index_(other.index_), kind_(other.kind_), pid_(other.pid_) {}

SeatLock& SeatLock::operator=(SeatLock&& other) noexcept {
    if (this != &other) {
        release();
        path_ = std::exchange(other.path_, {});
        index_ = other.index_;
        kind_ = other.kind_;
        pid_ = other.pid_;
    }
    return *this;
}

SeatGate::SeatGate(const SeatId& seat, std::chrono::milliseconds timeout) :
    path_(chip_file(seat.asic_id)), index_(seat_index(seat)), pid_(::getpid()) {
    Registry& reg = registry();
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true) {
        {
            const std::lock_guard guard(reg.mutex);
            ChipFile& chip = open_chip(reg, path_);
            SeatState& state = chip.seats[index_];
            if (!state.gate && try_lock(chip.fd, gate_byte(index_), F_WRLCK, path_)) {
                state.gate = true;
                chip.holders++;
                return;
            }
            close_if_unheld(reg, path_);
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            TT_THROW(
                "drisc_service: timed out after {} ms waiting for the gate of seat {} in {}",
                timeout.count(),
                index_,
                path_);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

SeatGate::~SeatGate() {
    if (pid_ != ::getpid()) {
        return;
    }
    Registry& reg = registry();
    const std::lock_guard guard(reg.mutex);
    ChipFile& chip = reg.chips.at(path_);
    chip.seats.at(index_).gate = false;
    set_lock(chip.fd, gate_byte(index_), F_UNLCK);
    chip.holders--;
    close_if_unheld(reg, path_);
}

uint32_t SeatGate::next_gen(uint32_t max) {
    Registry& reg = registry();
    const std::lock_guard guard(reg.mutex);
    const int fd = reg.chips.at(path_).fd;
    uint32_t last = 0;
    const ssize_t got = ::pread(fd, &last, sizeof(last), gen_offset(index_));
    const uint32_t gen = (got == sizeof(last) && last < max) ? last + 1 : 1;
    const ssize_t put = got < 0 ? -1 : ::pwrite(fd, &gen, sizeof(gen), gen_offset(index_));
    TT_FATAL(put == sizeof(gen), "drisc_service: cannot update {} ({})", path_, std::strerror(errno));
    return gen;
}

}  // namespace tt::tt_metal::internal::drisc_service
