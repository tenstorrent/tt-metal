// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The owner side of InterProcessCounterChannel registers its segment with
// ShmResourceTracker, so the segment leaves /dev/shm together with its owner
// (exit or SIGTERM) and a copy left by a killed predecessor is reaped by the
// tracker's stale scan instead of blocking re-creation. No device needed.
//
// The tracker's signal path and stale scan are driven directly rather than
// through a child process: this binary opens the devices at startup, so a
// re-executed child (gtest death tests) would redo device initialisation.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <internal/service/inter_process_counter_channel.hpp>
#include "tt_metal/distributed/inter_process_counter_layout.hpp"
#include <tt-metalium/experimental/sockets/shm_resource_tracker.hpp>

namespace tt::tt_metal::distributed {
namespace {

std::string tracking_manifest_path(pid_t pid) { return fmt::format("/dev/shm/tt_socket_manifest_{}", pid); }

bool tracking_manifest_names(pid_t pid, const std::string& shm_name) {
    std::ifstream ifs(tracking_manifest_path(pid));
    std::string line;
    while (std::getline(ifs, line)) {
        if (line == "shm " + shm_name) {
            return true;
        }
    }
    return false;
}

bool tracking_segment_exists(const std::string& shm_name) {
    const int fd = ::shm_open(shm_name.c_str(), O_RDONLY, 0);
    if (fd == -1) {
        return false;
    }
    ::close(fd);
    return true;
}

// A pid that was alive a moment ago and is now gone.
pid_t tracking_reaped_child_pid() {
    const pid_t pid = fork();
    if (pid == 0) {
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return pid;
}

// What a killed owner leaves behind: the segment itself plus a manifest naming it.
void tracking_plant_dead_owner_segment(const std::string& shm_name, pid_t dead_owner) {
    const int fd = ::shm_open(shm_name.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    ASSERT_NE(fd, -1) << std::strerror(errno);
    ASSERT_EQ(::ftruncate(fd, sizeof(InterProcessCounterSegment)), 0);
    ::close(fd);
    const std::string path = tracking_manifest_path(dead_owner);
    const std::string tmp_path = path + ".planting";
    {
        std::ofstream manifest(tmp_path, std::ios::trunc);
        manifest << "shm " << shm_name << "\n";
        ASSERT_TRUE(manifest.good());
    }
    ASSERT_EQ(std::rename(tmp_path.c_str(), path.c_str()), 0) << std::strerror(errno);
}

TEST(CounterChannelTracking, OwnerSegmentIsListedInManifestUntilShutdown) {
    const std::string name = fmt::format("/tt_test_ctr_manifest_{}", getpid());
    ::shm_unlink(name.c_str());

    InterProcessCounterChannel owner(name);
    EXPECT_TRUE(tracking_segment_exists(name));
    EXPECT_TRUE(tracking_manifest_names(getpid(), name));

    owner.shutdown();
    EXPECT_FALSE(tracking_segment_exists(name));
    EXPECT_FALSE(tracking_manifest_names(getpid(), name));
}

TEST(CounterChannelTracking, OwnerSegmentIsUnlinkedBySignalCleanup) {
    const std::string name = fmt::format("/tt_test_ctr_signal_{}", getpid());
    ::shm_unlink(name.c_str());

    InterProcessCounterChannel owner(name);
    ASSERT_TRUE(tracking_segment_exists(name));

    // What the tracker's SIGINT/SIGTERM handler runs before the process dies.
    ShmResourceTracker::instance().cleanup_from_signal();
    EXPECT_FALSE(tracking_segment_exists(name)) << name << " survived the signal cleanup";
    EXPECT_FALSE(tracking_manifest_names(getpid(), name));

    // The owner's own teardown afterwards is harmless.
    owner.shutdown();
}

TEST(CounterChannelTracking, SegmentOfKilledPredecessorIsReapedBeforeCreate) {
    const std::string name = fmt::format("/tt_test_ctr_stale_{}", getpid());
    const pid_t predecessor = tracking_reaped_child_pid();
    ASSERT_GT(predecessor, 0) << "fork failed: " << std::strerror(errno);
    tracking_plant_dead_owner_segment(name, predecessor);
    ASSERT_TRUE(tracking_segment_exists(name));

    // In a fresh process the owner constructor triggers this scan by touching
    // the tracker before its O_EXCL open; here the tracker already exists, so
    // the scan is run explicitly. Without it construction fails with EEXIST.
    ShmResourceTracker::cleanup_stale_resources();
    EXPECT_FALSE(tracking_segment_exists(name)) << "planted segment of a dead owner was not reaped";
    EXPECT_FALSE(std::ifstream(tracking_manifest_path(predecessor)).good()) << "stale manifest was not reaped";

    InterProcessCounterChannel owner(name);
    EXPECT_TRUE(tracking_segment_exists(name));
    owner.shutdown();
    EXPECT_FALSE(tracking_segment_exists(name));
}

}  // namespace
}  // namespace tt::tt_metal::distributed
