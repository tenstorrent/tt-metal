// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRISC seat locks without a device: each test gets its own lock directory through TT_DRISC_SEAT_LOCK_DIR.

#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "impl/drisc_service/seat_lock.hpp"

namespace tt::tt_metal::internal::drisc_service {
namespace {

constexpr SeatId seat{.asic_id = 1, .noc0_x = 2, .noc0_y = 3};
constexpr SeatId other_seat{.asic_id = 1, .noc0_x = 2, .noc0_y = 4};
constexpr std::chrono::milliseconds gate_timeout{1000};

class SeatLockFiles : public ::testing::Test {
protected:
    void SetUp() override {
        if (const char* value = std::getenv("TT_DRISC_SEAT_LOCK_DIR")) {
            saved_dir_ = value;
        }
        static int calls = 0;
        dir_ = std::filesystem::temp_directory_path() /
               ("seat_lock_test_" + std::to_string(::getpid()) + "_" + std::to_string(++calls));
        setenv("TT_DRISC_SEAT_LOCK_DIR", dir_.c_str(), /*overwrite=*/1);
    }

    void TearDown() override {
        if (saved_dir_.has_value()) {
            setenv("TT_DRISC_SEAT_LOCK_DIR", saved_dir_->c_str(), /*overwrite=*/1);
        } else {
            unsetenv("TT_DRISC_SEAT_LOCK_DIR");
        }
        std::filesystem::remove_all(dir_);
    }

    // The file of `seat`'s chip.
    std::filesystem::path chip_file() const { return dir_ / "chip-1.lock"; }
    const std::filesystem::path& dir() const { return dir_; }

private:
    std::filesystem::path dir_;
    std::optional<std::string> saved_dir_;
};

// 1) An exclusive lock keeps out clients and a second exclusive lock, on its own seat only.
// 2) An exclusive lock is never sole and cannot be shared.
// 3) A client lock keeps exclusive locks out, both before and after it shares.
TEST_F(SeatLockFiles, CPU_ExclusiveAndClientLocksExcludeEachOther) {
    std::optional<SeatLock> exclusive = SeatLock::try_acquire_exclusive(seat);
    ASSERT_TRUE(exclusive.has_value());
    EXPECT_FALSE(exclusive->sole());
    EXPECT_ANY_THROW(exclusive->share());
    EXPECT_FALSE(SeatLock::try_acquire_client(seat).has_value()) << "a client took a seat held exclusively";
    EXPECT_FALSE(SeatLock::try_acquire_exclusive(seat).has_value()) << "two exclusive locks on one seat";
    EXPECT_TRUE(SeatLock::try_acquire_exclusive(other_seat).has_value()) << "another seat was blocked";
    exclusive.reset();

    std::optional<SeatLock> client = SeatLock::try_acquire_client(seat);
    ASSERT_TRUE(client.has_value());
    EXPECT_TRUE(client->sole());
    EXPECT_FALSE(SeatLock::try_acquire_exclusive(seat).has_value()) << "metal took a seat a sole client holds";
    client->share();
    EXPECT_FALSE(SeatLock::try_acquire_exclusive(seat).has_value()) << "metal took a seat a shared client holds";
}

// 1) A second client is refused while the first is sole, as metal would be; clients take the seat's gate around their
//    attach, so the first has shared before the second looks.
// 2) Once the first shares, the second joins and is not sole.
// 3) Sharing an already shared lock, as a joining client's attach does, keeps it.
// 4) The seat stays held until the last client leaves.
TEST_F(SeatLockFiles, CPU_SecondClientSharesOnlyAfterTheFirstShares) {
    std::optional<SeatLock> first = SeatLock::try_acquire_client(seat);
    ASSERT_TRUE(first.has_value());
    EXPECT_FALSE(SeatLock::try_acquire_client(seat).has_value()) << "a second client joined a sole client's seat";

    first->share();
    EXPECT_FALSE(first->sole());
    std::optional<SeatLock> second = SeatLock::try_acquire_client(seat);
    ASSERT_TRUE(second.has_value());
    EXPECT_FALSE(second->sole()) << "the second client thinks it is alone";
    second->share();
    EXPECT_FALSE(second->sole());

    first.reset();
    EXPECT_FALSE(SeatLock::try_acquire_exclusive(seat).has_value()) << "metal took a seat a client still holds";
    second.reset();
    EXPECT_TRUE(SeatLock::try_acquire_exclusive(seat).has_value()) << "the seat stayed held after both clients left";
}

// The gate admits one attach at a time on its own seat only.
TEST_F(SeatLockFiles, CPU_GateAdmitsOneAttachAtATime) {
    {
        const SeatGate gate(seat, gate_timeout);
        EXPECT_ANY_THROW({ const SeatGate second(seat, std::chrono::milliseconds(10)); }) << "two attaches at once";
        const SeatGate other(other_seat, gate_timeout);
    }
    const SeatGate again(seat, gate_timeout);
}

size_t open_files() {
    size_t count = 0;
    for ([[maybe_unused]] const auto& entry : std::filesystem::directory_iterator("/proc/self/fd")) {
        count++;
    }
    return count;
}

// A process keeps one file open per chip, however many seats it holds there and however it holds them.
TEST_F(SeatLockFiles, CPU_OneOpenFilePerChip) {
    const size_t before = open_files();
    std::vector<SeatLock> held;
    for (uint32_t y = 0; y < 12; y++) {
        held.push_back(SeatLock::try_acquire_exclusive({.asic_id = 1, .noc0_x = 0, .noc0_y = y}).value());
        held.push_back(SeatLock::try_acquire_client({.asic_id = 1, .noc0_x = 9, .noc0_y = y}).value());
    }
    {
        const SeatGate gate(seat, gate_timeout);
        EXPECT_EQ(open_files(), before + 1);
    }
    held.push_back(SeatLock::try_acquire_exclusive({.asic_id = 2, .noc0_x = 0, .noc0_y = 0}).value());
    EXPECT_EQ(open_files(), before + 2);
    held.clear();
    EXPECT_EQ(open_files(), before) << "a chip file stayed open with nothing held in it";
}

TEST_F(SeatLockFiles, CPU_LockIsReleasedWhenItsLastOwnerGoes) {
    std::optional<SeatLock> client = SeatLock::try_acquire_client(seat);
    ASSERT_TRUE(client.has_value());
    SeatLock moved = std::move(*client);
    client.reset();
    EXPECT_FALSE(SeatLock::try_acquire_exclusive(seat).has_value()) << "destroying a moved-from lock freed the seat";

    std::optional<SeatLock> other = SeatLock::try_acquire_exclusive(other_seat);
    ASSERT_TRUE(other.has_value());
    *other = std::move(moved);
    EXPECT_TRUE(SeatLock::try_acquire_exclusive(other_seat).has_value()) << "move-assigning kept the old seat";
    EXPECT_FALSE(SeatLock::try_acquire_exclusive(seat).has_value()) << "move-assigning dropped the new seat";
    other.reset();
    EXPECT_TRUE(SeatLock::try_acquire_exclusive(seat).has_value());
}

// The OS drops a process's locks when it dies, which is how a crashed client's seat is freed.
TEST_F(SeatLockFiles, CPU_LockOfAKilledProcessIsReleased) {
    int ready[2];
    ASSERT_EQ(::pipe(ready), 0);
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        const std::optional<SeatLock> lock = SeatLock::try_acquire_exclusive(seat);
        const char held = lock.has_value() ? 'y' : 'n';
        if (::write(ready[1], &held, 1) != 1) {
            ::_exit(1);
        }
        ::pause();
        ::_exit(0);
    }
    struct Reaper {
        pid_t pid;
        ~Reaper() {
            if (pid > 0) {
                ::kill(pid, SIGKILL);
                ::waitpid(pid, nullptr, 0);
            }
        }
    } reaper{child};

    char held = 0;
    ASSERT_EQ(::read(ready[0], &held, 1), 1);
    ::close(ready[0]);
    ::close(ready[1]);
    ASSERT_EQ(held, 'y') << "the child could not take the seat";
    EXPECT_FALSE(SeatLock::try_acquire_exclusive(seat).has_value()) << "took a seat another process holds";
    ASSERT_EQ(::kill(child, SIGKILL), 0);
    ASSERT_EQ(::waitpid(child, nullptr, 0), child);
    reaper.pid = 0;
    EXPECT_TRUE(SeatLock::try_acquire_exclusive(seat).has_value()) << "the dead child's seat stayed held";
}

// A child forked without exec takes its parent's seat no more than another process could, and destroying the copy of
// the lock it inherited leaves the seat held.
TEST_F(SeatLockFiles, CPU_ForkedChildLeavesItsParentsLocksAlone) {
    std::optional<SeatLock> lock = SeatLock::try_acquire_exclusive(seat);
    ASSERT_TRUE(lock.has_value());
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        lock.reset();
        ::_exit(SeatLock::try_acquire_exclusive(seat).has_value() ? 1 : 0);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << "the child took its parent's seat, status " << status;
}

// share() turns a sole client's lock into a shared one without freeing the seat in between: another process that
// keeps trying to take the seat across many shares never gets it.
TEST_F(SeatLockFiles, CPU_SharingNeverFreesTheSeat) {
    struct Rounds {
        std::atomic<int> started{0};  // The round the parent started, -1 to stop the child.
        std::atomic<int> trying{0};   // The round the child is trying to take the seat in.
        std::atomic<int> stopped{0};  // The round the parent finished sharing in.
        std::atomic<int> done{0};     // The round the child stopped trying in.
        std::atomic<int> taken{0};    // Times the child took the seat.
    };
    void* mem = ::mmap(nullptr, sizeof(Rounds), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(mem, MAP_FAILED);
    auto* rounds = new (mem) Rounds;
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        for (int round = 1;; round++) {
            int started = 0;
            while ((started = rounds->started.load()) != round && started != -1) {
            }
            if (started == -1) {
                ::_exit(0);
            }
            rounds->trying = round;
            while (rounds->stopped.load() != round) {
                if (SeatLock::try_acquire_exclusive(seat).has_value()) {
                    rounds->taken++;
                }
            }
            rounds->done = round;
        }
    }
    struct Reaper {
        pid_t pid;
        ~Reaper() {
            ::kill(pid, SIGKILL);
            ::waitpid(pid, nullptr, 0);
        }
    } reaper{child};

    constexpr int num_rounds = 500;
    for (int round = 1; round <= num_rounds; round++) {
        std::optional<SeatLock> client = SeatLock::try_acquire_client(seat);
        ASSERT_TRUE(client.has_value() && client->sole()) << "round " << round;
        rounds->started = round;
        while (rounds->trying.load() != round) {
        }
        client->share();
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        rounds->stopped = round;
        while (rounds->done.load() != round) {
        }
    }
    rounds->started = -1;
    EXPECT_EQ(rounds->taken.load(), 0) << "another process took the seat while a client shared it";
    ::munmap(mem, sizeof(Rounds));
}

TEST_F(SeatLockFiles, CPU_GenerationsCountUpWrapAndRecover) {
    SeatGate gate(seat, gate_timeout);
    EXPECT_EQ(gate.next_gen(3), 1u);
    EXPECT_EQ(gate.next_gen(3), 2u);
    EXPECT_EQ(gate.next_gen(3), 3u);
    EXPECT_EQ(gate.next_gen(3), 1u) << "did not wrap from max to 1";

    std::filesystem::resize_file(chip_file(), 0);
    EXPECT_EQ(gate.next_gen(3), 1u) << "a short chip file did not restart the count";
    const std::string past_max(64 * 1024, '\x07');
    std::ofstream(chip_file(), std::ios::binary | std::ios::trunc) << past_max;
    EXPECT_EQ(gate.next_gen(3), 1u) << "a generation past max did not restart the count";
}

mode_t mode_of(const std::filesystem::path& path) {
    struct stat st{};
    EXPECT_EQ(::lstat(path.c_str(), &st), 0) << path;
    return st.st_mode & 07777;
}

// 1) The lock dir and files get the modes other users need under a restrictive umask.
// 2) Creating them leaves the process's umask as it was.
TEST_F(SeatLockFiles, CPU_FilesAreSharedWhateverTheUmask) {
    struct UmaskRestorer {
        mode_t old;
        ~UmaskRestorer() { ::umask(old); }
    } restorer{::umask(077)};

    const std::optional<SeatLock> lock = SeatLock::try_acquire_client(seat);
    ASSERT_TRUE(lock.has_value());
    SeatGate gate(seat, gate_timeout);
    gate.next_gen(3);

    EXPECT_EQ(::umask(077), 077) << "taking a seat changed the umask";
    EXPECT_EQ(mode_of(dir()), 01777);
    EXPECT_EQ(mode_of(chip_file()), 0666);
}

// 1) A chip file that is a symlink is refused, not followed: whoever owns the shared dir could plant one.
// 2) The file it points to is left untouched.
TEST_F(SeatLockFiles, CPU_SymlinkedChipFilesAreRefused) {
    std::filesystem::create_directories(dir());
    const std::filesystem::path target = dir() / "target";
    std::ofstream(target) << "keep";

    std::filesystem::create_symlink(target, chip_file());
    EXPECT_ANY_THROW({ const SeatGate gate(seat, gate_timeout); }) << "the gate followed it";
    EXPECT_ANY_THROW(SeatLock::try_acquire_client(seat)) << "the seat lock followed it";

    std::string content;
    std::ifstream(target) >> content;
    EXPECT_EQ(content, "keep");
}

}  // namespace
}  // namespace tt::tt_metal::internal::drisc_service
