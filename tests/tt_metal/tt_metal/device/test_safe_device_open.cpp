// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <string>

#include <gtest/gtest.h>
#include <umd/device/utils/robust_mutex.hpp>

#include "impl/device/safe_device_open.hpp"

namespace tt::tt_metal {
namespace {

// High chip ID that won't correspond to a real device on any test host.
constexpr tt::ChipId kTestId = 250;

// Names that match SafeDeviceGuard's internal naming for kTestId.
const std::string kDirtyShmName = "/TT_METAL_DEVICE_DIRTY." + std::to_string(kTestId);
// RobustMutex opens "/dev/shm/<SHM_FILE_PREFIX><mutex_name>".
const std::string kMutexShmName =
    std::string(tt::umd::RobustMutex::SHM_FILE_PREFIX) + "tt-metal-device-" + std::to_string(kTestId);

// Remove any leftover shm objects from a previous run.
void wipe_shm() {
    shm_unlink(kDirtyShmName.c_str());
    shm_unlink(kMutexShmName.c_str());
}

// Read the dirty byte directly. Returns 0 if the shm doesn't exist.
uint8_t read_dirty() {
    int fd = shm_open(kDirtyShmName.c_str(), O_RDONLY, 0);
    if (fd == -1) {
        return 0;
    }
    void* ptr = mmap(nullptr, 1, PROT_READ, MAP_SHARED, fd, 0);
    uint8_t val = 0;
    if (ptr != MAP_FAILED) {
        val = *static_cast<uint8_t*>(ptr);
        munmap(ptr, 1);
    }
    close(fd);
    return val;
}

class SafeDeviceGuardTest : public ::testing::Test {
protected:
    void SetUp() override { wipe_shm(); }
    void TearDown() override { wipe_shm(); }
};

// Dirty bit is set to 1 after construction and cleared to 0 after graceful destruction.
TEST_F(SafeDeviceGuardTest, DirtyBitLifecycle) {
    {
        SafeDeviceGuard guard({kTestId});
        EXPECT_EQ(read_dirty(), 1) << "dirty bit must be 1 while guard is live";
    }
    EXPECT_EQ(read_dirty(), 0) << "dirty bit must be cleared on graceful exit";
}

// After on_hang(), the dirty bit persists through destruction.
// The next acquirer will see dirty=1 and trigger a reset before starting work.
TEST_F(SafeDeviceGuardTest, HangLeavesDeviceDirty) {
    {
        SafeDeviceGuard guard({kTestId});
        guard.on_hang();
    }
    EXPECT_EQ(read_dirty(), 1) << "dirty bit must stay set when the guard exits via hang path";
}

// on_hang() must be safe to call from multiple threads or repeated calls (idempotent).
TEST_F(SafeDeviceGuardTest, OnHangIsIdempotent) {
    SafeDeviceGuard guard({kTestId});
    EXPECT_NO_THROW({
        guard.on_hang();
        guard.on_hang();
        guard.on_hang();
    });
}

// A child process holds the guard; the parent verifies the dirty bit is set while the child
// is alive, then acquires the guard itself after the child exits cleanly.
TEST_F(SafeDeviceGuardTest, MutexSerializesAcrossProcesses) {
    int to_child[2], to_parent[2];
    ASSERT_EQ(pipe(to_child), 0);
    ASSERT_EQ(pipe(to_parent), 0);

    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork failed: " << errno;

    if (pid == 0) {
        close(to_child[1]);
        close(to_parent[0]);
        {
            SafeDeviceGuard guard({kTestId});
            char locked = 'L';
            write(to_parent[1], &locked, 1);
            char go = 0;
            read(to_child[0], &go, 1);
        }
        close(to_child[0]);
        close(to_parent[1]);
        _exit(0);
    }

    close(to_child[0]);
    close(to_parent[1]);

    char locked = 0;
    ASSERT_EQ(read(to_parent[0], &locked, 1), 1);
    EXPECT_EQ(locked, 'L');
    EXPECT_EQ(read_dirty(), 1) << "dirty bit must be 1 while the child holds the guard";

    char go = 'G';
    write(to_child[1], &go, 1);
    close(to_child[1]);
    close(to_parent[0]);

    int status = 0;
    waitpid(pid, &status, 0);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);

    // Child exited cleanly, so it cleared the dirty bit. Parent can now re-acquire.
    {
        SafeDeviceGuard guard({kTestId});
        EXPECT_EQ(read_dirty(), 1);
    }
    EXPECT_EQ(read_dirty(), 0);
}

// A process that dies without releasing the guard (simulated via _exit without running dtors)
// leaves the dirty bit set. The next acquirer must be able to construct a guard (the robust
// mutex recovers from EOWNERDEAD) and will observe dirty=1, triggering the reset path.
TEST_F(SafeDeviceGuardTest, CrashLeavesDeviceDirtyNextAcquirerRecovers) {
    pid_t pid = fork();
    ASSERT_NE(pid, -1);

    if (pid == 0) {
        SafeDeviceGuard* guard = new SafeDeviceGuard({kTestId});
        (void)guard;
        _exit(0);  // skip destructors — mutex is abandoned, dirty bit stays 1
    }

    int status = 0;
    waitpid(pid, &status, 0);
    EXPECT_TRUE(WIFEXITED(status));

    EXPECT_EQ(read_dirty(), 1) << "crash must leave dirty bit set";

    // Guard construction must succeed despite the abandoned mutex (RobustMutex handles EOWNERDEAD).
    // It will attempt tt-smi -r 250 which fails on non-hardware hosts; that is acceptable here.
    EXPECT_NO_THROW({
        SafeDeviceGuard guard({kTestId});
        EXPECT_EQ(read_dirty(), 1);
    });
    EXPECT_EQ(read_dirty(), 0);
}

}  // namespace
}  // namespace tt::tt_metal
