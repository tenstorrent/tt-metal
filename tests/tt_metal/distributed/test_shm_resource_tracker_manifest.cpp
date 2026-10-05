// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The tracker's manifest records the owner's process start time, so the stale
// scan reaps the resources of an owner that died even when its pid has since
// been handed to another process, and keeps those of an owner that is alive.
// Manifests without the start line (older images) are judged by pid alone.
// No device needed.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <tt-metalium/experimental/sockets/shm_resource_tracker.hpp>
#include "tt_metal/distributed/shm_owner_liveness.hpp"
#include "tt_metal/distributed/shm_stale_scan.hpp"

namespace tt::tt_metal::distributed {
namespace {

std::string manifest_test_path(pid_t pid) { return fmt::format("/dev/shm/tt_socket_manifest_{}", pid); }

bool manifest_test_file_exists(const std::string& path) { return std::ifstream(path).good(); }

bool manifest_test_shm_exists(const std::string& shm_name) {
    const int fd = ::shm_open(shm_name.c_str(), O_RDONLY, 0);
    if (fd == -1) {
        return false;
    }
    ::close(fd);
    return true;
}

bool manifest_test_has_line(const std::string& path, const std::string& wanted) {
    std::ifstream ifs(path);
    std::string line;
    while (std::getline(ifs, line)) {
        if (line == wanted) {
            return true;
        }
    }
    return false;
}

// A process that stays alive for the duration of a test, standing in for an
// unrelated process that was handed a dead owner's pid.
class ManifestTestChild {
public:
    ManifestTestChild() : parent_(getpid()), pid_(fork()) {
        if (pid_ == 0) {
            // Die with the test process, and do not run the tracker's
            // inherited SIGINT/SIGTERM handler on the parent's resources.
            prctl(PR_SET_PDEATHSIG, SIGKILL);
            if (getppid() != parent_) {
                _exit(0);
            }
            signal(SIGINT, SIG_DFL);
            signal(SIGTERM, SIG_DFL);
            for (;;) {
                pause();
            }
        }
    }
    ~ManifestTestChild() {
        if (pid_ > 0) {
            kill(pid_, SIGKILL);
            int status = 0;
            waitpid(pid_, &status, 0);
        }
    }
    pid_t pid() const { return pid_; }

private:
    pid_t parent_;
    pid_t pid_;
};

// A pid that was alive a moment ago and is now gone.
pid_t manifest_test_reaped_pid() {
    const pid_t pid = fork();
    if (pid == 0) {
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return pid;
}

// What an owner leaves behind: a shm object and a manifest listing it, with or
// without the start line.
void manifest_test_plant(pid_t owner, std::optional<uint64_t> start_time, const std::string& shm_name) {
    const int fd = ::shm_open(shm_name.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    ASSERT_NE(fd, -1) << std::strerror(errno);
    ASSERT_EQ(::ftruncate(fd, 64), 0);
    ::close(fd);
    const std::string path = manifest_test_path(owner);
    const std::string tmp_path = path + ".planting";
    {
        std::ofstream manifest(tmp_path, std::ios::trunc);
        if (start_time) {
            manifest << "start " << *start_time << "\n";
        }
        manifest << "shm " << shm_name << "\n";
        ASSERT_TRUE(manifest.good());
    }
    ASSERT_EQ(std::rename(tmp_path.c_str(), path.c_str()), 0) << std::strerror(errno);
}

void manifest_test_remove(pid_t owner, const std::string& shm_name) {
    ::shm_unlink(shm_name.c_str());
    std::remove(manifest_test_path(owner).c_str());
}

TEST(ShmResourceTrackerManifest, OwnManifestRecordsStartTime) {
    const std::string name = fmt::format("/tt_test_manifest_own_{}", getpid());
    auto& tracker = ShmResourceTracker::instance();
    tracker.track_shm(name);

    const std::string manifest = manifest_test_path(getpid());
    std::ifstream ifs(manifest);
    std::string first_line;
    ASSERT_TRUE(std::getline(ifs, first_line));
    EXPECT_EQ(first_line, fmt::format("start {}", process_start_time(getpid())));
    EXPECT_TRUE(manifest_test_has_line(manifest, "shm " + name));

    tracker.untrack_shm(name);
    EXPECT_FALSE(manifest_test_has_line(manifest, "shm " + name));
}

TEST(ShmResourceTrackerManifest, ScanReapsOwnerWhosePidBelongsToAnotherProcess) {
    ManifestTestChild bystander;
    ASSERT_GT(bystander.pid(), 0) << std::strerror(errno);
    const uint64_t bystander_start = process_start_time(bystander.pid());
    ASSERT_NE(bystander_start, 0u);
    const std::string name = fmt::format("/tt_test_manifest_reused_{}", getpid());
    // The dead owner started at a different time than the process now holding its pid.
    manifest_test_plant(bystander.pid(), bystander_start + 1, name);
    ASSERT_TRUE(manifest_test_shm_exists(name));

    ShmResourceTracker::cleanup_stale_resources();

    EXPECT_TRUE(ShmResourceTracker::is_pid_alive(bystander.pid())) << "bystander died; the test proved nothing";
    EXPECT_FALSE(manifest_test_shm_exists(name)) << "stale shm of a reused pid was kept";
    EXPECT_FALSE(manifest_test_file_exists(manifest_test_path(bystander.pid()))) << "stale manifest was kept";
    manifest_test_remove(bystander.pid(), name);
}

TEST(ShmResourceTrackerManifest, ScanReapsPredecessorThatHeldOurOwnPid) {
    // The recreated worker was handed the dead owner's pid, so the stale
    // manifest sits at this process's own manifest path.
    const uint64_t my_start = process_start_time(getpid());
    ASSERT_NE(my_start, 0u);
    const std::string name = fmt::format("/tt_test_manifest_ownpid_{}", getpid());
    manifest_test_plant(getpid(), my_start + 1, name);

    ShmResourceTracker::cleanup_stale_resources();

    EXPECT_FALSE(manifest_test_shm_exists(name)) << "predecessor's shm at our own pid was kept";
    EXPECT_FALSE(manifest_test_file_exists(manifest_test_path(getpid()))) << "predecessor's manifest was kept";

    // A manifest at our own path with our own start time is ours and stays.
    manifest_test_plant(getpid(), my_start, name);
    ShmResourceTracker::cleanup_stale_resources();
    EXPECT_TRUE(manifest_test_shm_exists(name)) << "our own manifest was reaped";
    manifest_test_remove(getpid(), name);

    // Let the tracker rewrite its real manifest from its in-memory state.
    auto& tracker = ShmResourceTracker::instance();
    tracker.track_shm(name);
    tracker.untrack_shm(name);
}

TEST(ShmResourceTrackerManifest, ScanKeepsLiveOwnerWithMatchingStartTime) {
    ManifestTestChild owner;
    ASSERT_GT(owner.pid(), 0) << std::strerror(errno);
    const uint64_t owner_start = process_start_time(owner.pid());
    ASSERT_NE(owner_start, 0u);
    const std::string name = fmt::format("/tt_test_manifest_live_{}", getpid());
    manifest_test_plant(owner.pid(), owner_start, name);

    ShmResourceTracker::cleanup_stale_resources();

    EXPECT_TRUE(manifest_test_shm_exists(name)) << "shm of a live owner was removed";
    EXPECT_TRUE(manifest_test_file_exists(manifest_test_path(owner.pid()))) << "manifest of a live owner was removed";
    manifest_test_remove(owner.pid(), name);
}

TEST(ShmResourceTrackerManifest, ScanJudgesManifestWithoutStartLineByPidAlone) {
    ManifestTestChild live;
    ASSERT_GT(live.pid(), 0) << std::strerror(errno);
    const pid_t dead = manifest_test_reaped_pid();
    ASSERT_GT(dead, 0) << std::strerror(errno);
    const std::string live_name = fmt::format("/tt_test_manifest_legacy_live_{}", getpid());
    const std::string dead_name = fmt::format("/tt_test_manifest_legacy_dead_{}", getpid());
    manifest_test_plant(live.pid(), std::nullopt, live_name);
    manifest_test_plant(dead, std::nullopt, dead_name);

    ShmResourceTracker::cleanup_stale_resources();

    EXPECT_TRUE(manifest_test_shm_exists(live_name)) << "legacy manifest of a live pid was reaped";
    EXPECT_TRUE(manifest_test_file_exists(manifest_test_path(live.pid())));
    EXPECT_FALSE(manifest_test_shm_exists(dead_name)) << "legacy manifest of a dead pid was kept";
    EXPECT_FALSE(manifest_test_file_exists(manifest_test_path(dead)));
    manifest_test_remove(live.pid(), live_name);
    manifest_test_remove(dead, dead_name);
}

TEST(ShmResourceTrackerManifest, ManifestRepublishedBetweenJudgementAndReapIsKept) {
    // A scanner judges the predecessor's manifest at a pid that a live process has since
    // been handed. Before the scanner reaps, that live owner reaps its predecessor itself
    // and publishes its own manifest at the same path. Only what the judged manifest
    // listed may go; the live owner's manifest and segment must stay.
    ManifestTestChild live_owner;
    ASSERT_GT(live_owner.pid(), 0) << std::strerror(errno);
    const uint64_t live_start = process_start_time(live_owner.pid());
    ASSERT_NE(live_start, 0u);
    const std::string stale_name = fmt::format("/tt_test_manifest_racestale_{}", getpid());
    const std::string live_name = fmt::format("/tt_test_manifest_racelive_{}", getpid());
    manifest_test_plant(live_owner.pid(), live_start + 1, stale_name);  // the dead predecessor's
    if (HasFatalFailure()) {
        return;
    }

    const StaleScan scan = collect_stale_shm_resources();
    const auto judged = std::find_if(scan.manifests.begin(), scan.manifests.end(), [&](const StaleManifest& m) {
        return m.path == manifest_test_path(live_owner.pid());
    });
    ASSERT_NE(judged, scan.manifests.end()) << "predecessor's manifest was not judged stale";
    ASSERT_EQ(judged->shm_names, std::vector<std::string>{stale_name});

    // The live owner publishes its own manifest (new file, renamed into place) and segment.
    manifest_test_plant(live_owner.pid(), live_start, live_name);
    if (HasFatalFailure()) {
        return;
    }

    reap_stale_shm_resources(scan);

    EXPECT_FALSE(manifest_test_shm_exists(stale_name)) << "predecessor's segment was kept";
    EXPECT_TRUE(manifest_test_shm_exists(live_name)) << "live owner's segment was removed";
    EXPECT_TRUE(manifest_test_has_line(manifest_test_path(live_owner.pid()), "shm " + live_name))
        << "live owner's republished manifest was removed";
    manifest_test_remove(live_owner.pid(), live_name);
}

}  // namespace
}  // namespace tt::tt_metal::distributed
