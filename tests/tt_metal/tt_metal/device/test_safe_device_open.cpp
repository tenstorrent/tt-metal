// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <umd/device/utils/robust_mutex.hpp>

#include "impl/device/safe_device_open.hpp"

namespace tt::tt_metal {
namespace {

// High chip IDs that won't correspond to real devices on any test host.
constexpr tt::ChipId kA = 250;
constexpr tt::ChipId kB = 251;

// Canonical mesh key: sorted chip IDs joined with '-'.
std::string mesh_key(std::vector<tt::ChipId> ids) {
    std::sort(ids.begin(), ids.end());
    std::string key;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) key += '-';
        key += std::to_string(ids[i]);
    }
    return key;
}

std::string dirty_shm_name(std::vector<tt::ChipId> ids) {
    return "/TT_METAL_DEVICE_DIRTY.mesh-" + mesh_key(std::move(ids));
}

std::string mutex_shm_name(std::vector<tt::ChipId> ids) {
    return std::string(tt::umd::RobustMutex::SHM_FILE_PREFIX) + "tt-metal-mesh-" + mesh_key(std::move(ids));
}

void wipe_shm(std::vector<tt::ChipId> ids) {
    shm_unlink(dirty_shm_name(ids).c_str());
    shm_unlink(mutex_shm_name(ids).c_str());
}

// Read the dirty byte directly. Returns 0 if the shm doesn't exist yet.
uint8_t read_dirty(std::vector<tt::ChipId> ids) {
    const std::string name = dirty_shm_name(std::move(ids));
    int fd = shm_open(name.c_str(), O_RDONLY, 0);
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
    void SetUp() override {
        wipe_shm({kA});
        wipe_shm({kB});
        wipe_shm({kA, kB});
    }
    void TearDown() override {
        wipe_shm({kA});
        wipe_shm({kB});
        wipe_shm({kA, kB});
    }
};

// Dirty bit is set to 1 after construction and cleared to 0 after graceful destruction.
TEST_F(SafeDeviceGuardTest, DirtyBitLifecycle) {
    {
        SafeDeviceGuard guard({kA});
        EXPECT_EQ(read_dirty({kA}), 1) << "dirty bit must be 1 while guard is live";
    }
    EXPECT_EQ(read_dirty({kA}), 0) << "dirty bit must be cleared on graceful exit";
}

// After on_hang(), the dirty bit persists through destruction so the next acquirer resets.
TEST_F(SafeDeviceGuardTest, HangLeavesDeviceDirty) {
    {
        SafeDeviceGuard guard({kA});
        guard.on_hang();
    }
    EXPECT_EQ(read_dirty({kA}), 1) << "dirty bit must stay set when the guard exits via hang path";
}

// on_hang() must be safe to call multiple times (idempotent).
TEST_F(SafeDeviceGuardTest, OnHangIsIdempotent) {
    SafeDeviceGuard guard({kA});
    EXPECT_NO_THROW({
        guard.on_hang();
        guard.on_hang();
        guard.on_hang();
    });
}

// A child holds the guard; parent verifies dirty=1 while child is live, then re-acquires cleanly.
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
            SafeDeviceGuard guard({kA});
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
    EXPECT_EQ(read_dirty({kA}), 1) << "dirty bit must be 1 while child holds the guard";

    char go = 'G';
    write(to_child[1], &go, 1);
    close(to_child[1]);
    close(to_parent[0]);

    int status = 0;
    waitpid(pid, &status, 0);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);

    {
        SafeDeviceGuard guard({kA});
        EXPECT_EQ(read_dirty({kA}), 1);
    }
    EXPECT_EQ(read_dirty({kA}), 0);
}

// A process that exits without running destructors leaves the dirty bit set; the next acquirer
// recovers via RobustMutex EOWNERDEAD handling.
TEST_F(SafeDeviceGuardTest, CrashLeavesDeviceDirtyNextAcquirerRecovers) {
    pid_t pid = fork();
    ASSERT_NE(pid, -1);

    if (pid == 0) {
        SafeDeviceGuard* guard = new SafeDeviceGuard({kA});
        (void)guard;
        _exit(0);  // skip destructors — mutex abandoned, dirty bit stays 1
    }

    int status = 0;
    waitpid(pid, &status, 0);
    EXPECT_TRUE(WIFEXITED(status));

    EXPECT_EQ(read_dirty({kA}), 1) << "crash must leave dirty bit set";

    EXPECT_NO_THROW({
        SafeDeviceGuard guard({kA});
        EXPECT_EQ(read_dirty({kA}), 1);
    });
    EXPECT_EQ(read_dirty({kA}), 0);
}

// A multi-chip mesh uses a single, mesh-scoped dirty shm distinct from any single-chip shm.
TEST_F(SafeDeviceGuardTest, MultiChipMeshUsesSingleShm) {
    {
        SafeDeviceGuard guard({kA, kB});
        // The mesh shm is set; the individual per-chip shms are untouched.
        EXPECT_EQ(read_dirty({kA, kB}), 1) << "mesh dirty bit must be 1";
        EXPECT_EQ(read_dirty({kA}), 0) << "single-chip shm must be untouched by mesh guard";
        EXPECT_EQ(read_dirty({kB}), 0) << "single-chip shm must be untouched by mesh guard";
    }
    EXPECT_EQ(read_dirty({kA, kB}), 0);
}

// Two non-overlapping meshes can be held simultaneously — they use different mutex names.
TEST_F(SafeDeviceGuardTest, NonOverlappingMeshesRunInParallel) {
    int to_child[2], to_parent[2];
    ASSERT_EQ(pipe(to_child), 0);
    ASSERT_EQ(pipe(to_parent), 0);

    pid_t pid = fork();
    ASSERT_NE(pid, -1);

    if (pid == 0) {
        close(to_child[1]);
        close(to_parent[0]);
        {
            SafeDeviceGuard guard_b({kB});  // child holds mesh {kB}
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

    // Parent can acquire mesh {kA} immediately — different lock from child's {kB}.
    EXPECT_NO_THROW({
        SafeDeviceGuard guard_a({kA});
        EXPECT_EQ(read_dirty({kA}), 1);
        EXPECT_EQ(read_dirty({kB}), 1);

        char go = 'G';
        write(to_child[1], &go, 1);
        close(to_child[1]);
        close(to_parent[0]);

        int status = 0;
        waitpid(pid, &status, 0);
        EXPECT_TRUE(WIFEXITED(status));
        EXPECT_EQ(WEXITSTATUS(status), 0);
    });
}

}  // namespace
}  // namespace tt::tt_metal
