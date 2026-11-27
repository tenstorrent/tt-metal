// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRISC seat locks without a device: each test gets its own lock directory through TT_DRISC_SEAT_LOCK_DIR.

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>

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

    // `seat`'s file with extension `ext`.
    std::filesystem::path seat_file(const char* ext) const { return dir_ / (std::string("seat-1-2-3.") + ext); }
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

TEST_F(SeatLockFiles, CPU_ExclusiveBacksOffWhileAClientIsAttaching) {
    {
        const SeatGate gate(seat, gate_timeout);
        EXPECT_FALSE(SeatLock::try_acquire_exclusive(seat).has_value()) << "metal took a seat during an attach";
    }
    EXPECT_TRUE(SeatLock::try_acquire_exclusive(seat).has_value());
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

// A child forked without exec shares the parent's lock, which lasts until the child is gone too; the OS drops it
// when the child dies, which is how a crashed client's seat is freed.
TEST_F(SeatLockFiles, CPU_LockOfAKilledProcessIsReleased) {
    std::optional<SeatLock> lock = SeatLock::try_acquire_exclusive(seat);
    ASSERT_TRUE(lock.has_value());
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
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

    lock.reset();
    EXPECT_FALSE(SeatLock::try_acquire_exclusive(seat).has_value()) << "the seat was freed while the child lived";
    ASSERT_EQ(::kill(child, SIGKILL), 0);
    ASSERT_EQ(::waitpid(child, nullptr, 0), child);
    reaper.pid = 0;
    EXPECT_TRUE(SeatLock::try_acquire_exclusive(seat).has_value()) << "the dead child's seat stayed held";
}

TEST_F(SeatLockFiles, CPU_GenerationsCountUpWrapAndRecover) {
    SeatGate gate(seat, gate_timeout);
    EXPECT_EQ(gate.next_gen(3), 1u);
    EXPECT_EQ(gate.next_gen(3), 2u);
    EXPECT_EQ(gate.next_gen(3), 3u);
    EXPECT_EQ(gate.next_gen(3), 1u) << "did not wrap from max to 1";

    std::ofstream(seat_file("gen"), std::ios::binary | std::ios::trunc) << "x";
    EXPECT_EQ(gate.next_gen(3), 1u) << "a short .gen file did not restart the count";
    const uint32_t past_max = 7;
    std::ofstream(seat_file("gen"), std::ios::binary | std::ios::trunc)
        .write(reinterpret_cast<const char*>(&past_max), sizeof(past_max));
    EXPECT_EQ(gate.next_gen(3), 1u) << "a .gen value past max did not restart the count";
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
    for (const char* ext : {"lock", "gate", "gen"}) {
        EXPECT_EQ(mode_of(seat_file(ext)), 0666) << ext;
    }
}

// 1) A seat file that is a symlink is refused, not followed: whoever owns the shared dir could plant one.
// 2) The file it points to is left untouched.
TEST_F(SeatLockFiles, CPU_SymlinkedSeatFilesAreRefused) {
    std::filesystem::create_directories(dir());
    const std::filesystem::path target = dir() / "target";
    std::ofstream(target) << "keep";

    std::filesystem::create_symlink(target, seat_file("gen"));
    SeatGate gate(seat, gate_timeout);
    EXPECT_ANY_THROW(gate.next_gen(3)) << ".gen was followed";
    std::filesystem::create_symlink(target, seat_file("lock"));
    EXPECT_ANY_THROW(SeatLock::try_acquire_client(seat)) << ".lock was followed";

    std::string content;
    std::ifstream(target) >> content;
    EXPECT_EQ(content, "keep");
}

}  // namespace
}  // namespace tt::tt_metal::internal::drisc_service
