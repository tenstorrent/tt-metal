// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/drisc_service/seat_lock.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <utility>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::drisc_service {

namespace {

// Creates the lock dir when missing and returns the seat's file with extension `ext`.
std::string seat_file(const SeatId& seat, const char* ext) {
    const char* env = std::getenv("KV_MANAGER_SEAT_LOCK_DIR");
    const std::string dir = (env != nullptr && *env != '\0') ? env : "/dev/shm/tt-kv";
    if (::mkdir(dir.c_str(), 0777) == 0) {
        // mkdir applies the umask; other users must be able to create seat files here too.
        ::chmod(dir.c_str(), 01777);
    } else {
        TT_FATAL(
            errno == EEXIST,
            "drisc_service: cannot create seat lock dir {} ({}); set KV_MANAGER_SEAT_LOCK_DIR to a writable directory",
            dir,
            std::strerror(errno));
    }
    return fmt::format("{}/dmk-seat-{}-{}-{}.{}", dir, seat.asic_id, seat.noc0_x, seat.noc0_y, ext);
}

// Opens `path` with `flags`, creating it mode 0666 when missing. An existing file is opened without O_CREAT, which
// fs.protected_regular refuses in a sticky dir for a file another user owns.
int open_seat_file(const std::string& path, int flags) {
    while (true) {
        int fd = ::open(path.c_str(), flags | O_CLOEXEC);
        if (fd < 0 && errno == ENOENT) {
            fd = ::open(path.c_str(), flags | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
            if (fd >= 0) {
                // open applies the umask; other users must be able to open the file too.
                ::fchmod(fd, 0666);
            } else if (errno == EEXIST) {
                continue;
            }
        }
        TT_FATAL(fd >= 0, "drisc_service: cannot open seat file {} ({})", path, std::strerror(errno));
        return fd;
    }
}

// An open seat file, closed (dropping any flock taken through it) unless release()d.
class SeatFile {
public:
    explicit SeatFile(const std::string& path, int flags) : fd_(open_seat_file(path, flags)) {}
    ~SeatFile() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }
    SeatFile(const SeatFile&) = delete;
    SeatFile& operator=(const SeatFile&) = delete;
    int get() const { return fd_; }
    int release() { return std::exchange(fd_, -1); }

private:
    int fd_;
};

// False on contention.
bool try_flock(int fd, int operation, const std::string& path) {
    if (::flock(fd, operation | LOCK_NB) == 0) {
        return true;
    }
    TT_FATAL(errno == EWOULDBLOCK, "drisc_service: cannot lock seat file {} ({})", path, std::strerror(errno));
    return false;
}

}  // namespace

std::optional<SeatLock> SeatLock::try_acquire(const SeatId& seat) {
    const std::string path = seat_file(seat, "lock");
    // A client holds the gate across its LOCK_EX to LOCK_SH downgrade, when the lock alone looks free.
    const std::string gate_path = seat_file(seat, "gate");
    const SeatFile gate(gate_path, O_RDONLY);
    if (!try_flock(gate.get(), LOCK_EX, gate_path)) {
        return std::nullopt;
    }
    SeatFile file(path, O_RDONLY);
    if (!try_flock(file.get(), LOCK_EX, path)) {
        return std::nullopt;
    }
    return SeatLock(file.release());
}

SeatLock::~SeatLock() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

SeatLock::SeatLock(SeatLock&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}

SeatLock& SeatLock::operator=(SeatLock&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) {
            ::close(fd_);
        }
        fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
}

std::optional<ClientSeatLock> ClientSeatLock::try_acquire(const SeatId& seat) {
    const std::string path = seat_file(seat, "lock");
    SeatFile file(path, O_RDONLY);
    if (try_flock(file.get(), LOCK_EX, path)) {
        return ClientSeatLock(file.release(), true);
    }
    if (try_flock(file.get(), LOCK_SH, path)) {
        return ClientSeatLock(file.release(), false);
    }
    return std::nullopt;
}

void ClientSeatLock::share() {
    if (!sole_) {
        return;
    }
    if (::flock(fd_, LOCK_SH | LOCK_NB) != 0) {
        const int err = errno;
        ::close(fd_);
        fd_ = -1;
        TT_THROW(
            "drisc_service: lost the DRISC seat to tt-d-gen's own kernel while sharing it ({})", std::strerror(err));
    }
    sole_ = false;
}

ClientSeatLock::~ClientSeatLock() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

ClientSeatLock::ClientSeatLock(ClientSeatLock&& other) noexcept : fd_(other.fd_), sole_(other.sole_) { other.fd_ = -1; }

ClientSeatLock& ClientSeatLock::operator=(ClientSeatLock&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) {
            ::close(fd_);
        }
        fd_ = other.fd_;
        sole_ = other.sole_;
        other.fd_ = -1;
    }
    return *this;
}

SeatGate::SeatGate(const SeatId& seat, std::chrono::milliseconds timeout) : gen_path_(seat_file(seat, "gen")) {
    const std::string path = seat_file(seat, "gate");
    SeatFile file(path, O_RDONLY);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!try_flock(file.get(), LOCK_EX, path)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            TT_THROW("drisc_service: timed out after {} ms waiting for {}", timeout.count(), path);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    fd_ = file.release();
}

SeatGate::~SeatGate() { ::close(fd_); }

uint32_t SeatGate::next_gen(uint32_t max) {
    const int fd = open_seat_file(gen_path_, O_RDWR);
    uint32_t last = 0;
    const ssize_t got = ::pread(fd, &last, sizeof(last), 0);
    const uint32_t gen = (got == sizeof(last) && last < max) ? last + 1 : 1;
    const ssize_t put = got < 0 ? -1 : ::pwrite(fd, &gen, sizeof(gen), 0);
    const int err = errno;
    ::close(fd);
    TT_FATAL(put == sizeof(gen), "drisc_service: cannot update seat file {} ({})", gen_path_, std::strerror(err));
    return gen;
}

}  // namespace tt::tt_metal::drisc_service
