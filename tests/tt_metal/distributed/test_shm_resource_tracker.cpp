// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/shm_resource_tracker.hpp>

#include <gtest/gtest.h>

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <system_error>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

namespace tt::tt_metal::distributed {
namespace {

// Each owner starts the actual tracker in a child. The test process never
// initializes the singleton, so forks do not inherit a tracker or its lease.
class OwnerProcess {
public:
    OwnerProcess(const std::string& shm_name, const std::string& file_path) {
        int commands[2], replies[2];
        if (::pipe(commands) != 0) {
            throw std::system_error(errno, std::generic_category(), "command pipe");
        }
        if (::pipe(replies) != 0) {
            ::close(commands[0]);
            ::close(commands[1]);
            throw std::system_error(errno, std::generic_category(), "reply pipe");
        }
        pid_ = ::fork();
        if (pid_ == 0) {
            ::close(commands[1]);
            ::close(replies[0]);
            try {
                auto& tracker = ShmResourceTracker::instance();
                const int fd = ::shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
                if (fd < 0 || ::ftruncate(fd, 1) != 0 || ::pwrite(fd, "x", 1, 0) != 1) {
                    ::_exit(5);
                }
                ::close(fd);
                std::ofstream(file_path) << "descriptor";
                tracker.track_shm(shm_name);
                tracker.track_file(file_path);
                char command = 0;
                do {
                    if (::write(replies[1], "r", 1) != 1 || ::read(commands[0], &command, 1) != 1) {
                        ::_exit(2);
                    }
                    if (command == 'u') {
                        tracker.untrack_file(file_path);
                        tracker.track_file(file_path);
                    } else if (command == 'c') {
                        tracker.cleanup_all();
                    } else if (command == 't') {
                        tracker.track_shm(shm_name);
                        tracker.track_file(file_path);
                    } else if (command == 'e') {
                        std::exit(0);
                    } else {
                        ::_exit(3);
                    }
                } while (true);
            } catch (...) {
                ::_exit(4);
            }
        }
        const int error = errno;
        ::close(commands[0]);
        ::close(replies[1]);
        commands_ = commands[1];
        replies_ = replies[0];
        if (pid_ < 0 || !ready()) {
            stop(SIGKILL);
            ::close(commands_);
            ::close(replies_);
            throw std::system_error(error, std::generic_category(), "start tracker owner");
        }
    }

    ~OwnerProcess() {
        stop(SIGKILL);
        ::close(commands_);
        ::close(replies_);
    }

    OwnerProcess(const OwnerProcess&) = delete;
    OwnerProcess& operator=(const OwnerProcess&) = delete;

    bool command(char command) { return ::write(commands_, &command, 1) == 1 && ready(); }
    pid_t pid() const { return pid_; }

    int stop(int signal) {
        int status = 0;
        if (pid_ > 0) {
            ::kill(pid_, signal);
            while (::waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
            }
            pid_ = -1;
        }
        return status;
    }

    int exit_normally() {
        if (::write(commands_, "e", 1) != 1) {
            return -1;
        }
        int status = 0;
        while (::waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
        }
        pid_ = -1;
        return status;
    }

private:
    bool ready() {
        char reply = 0;
        return ::read(replies_, &reply, 1) == 1 && reply == 'r';
    }
    pid_t pid_ = -1;
    int commands_ = -1;
    int replies_ = -1;
};

class ShmResourceTrackerTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::string pattern = "/tmp/tt_shm_tracker_XXXXXX";
        ASSERT_NE(::mkdtemp(pattern.data()), nullptr);
        directory_ = pattern;
        file_path_ = directory_ + "/descriptor";
        // This PID is deliberately absent locally, as a foreign namespace's PID
        // can be. The old orphan scan deletes this live, tracked segment.
        shm_name_ = "/tt_h2d_2147483647_" + std::filesystem::path(directory_).filename().string();
    }

    void TearDown() override {
        owner_.reset();
        ShmResourceTracker::cleanup_stale_resources();
        ::shm_unlink(shm_name_.c_str());
        std::filesystem::remove_all(directory_);
    }

    void create_resources() {
        const int fd = ::shm_open(shm_name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        ASSERT_GE(fd, 0);
        EXPECT_EQ(::ftruncate(fd, 1), 0);
        EXPECT_EQ(::pwrite(fd, "x", 1, 0), 1);
        ::close(fd);
        std::ofstream(file_path_) << "descriptor";
    }

    void start_owner() { owner_ = std::make_unique<OwnerProcess>(shm_name_, file_path_); }

    std::string manifest_path() const {
        for (const auto& entry : std::filesystem::directory_iterator("/dev/shm")) {
            const auto name = entry.path().filename().string();
            if (!name.starts_with("tt_socket_manifest_v2_") || name.ends_with(".lock") || name.ends_with(".tmp")) {
                continue;
            }
            std::ifstream input(entry.path());
            const std::string contents{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
            if (contents.find("shm " + shm_name_ + "\n") != std::string::npos) {
                return entry.path();
            }
        }
        return {};
    }

    void expect_attachable() const {
        const int fd = ::shm_open(shm_name_.c_str(), O_RDWR, 0600);
        ASSERT_GE(fd, 0);
        char byte = 0;
        EXPECT_EQ(::pread(fd, &byte, 1, 0), 1);
        EXPECT_EQ(byte, 'x');
        ::close(fd);
        EXPECT_TRUE(std::filesystem::exists(file_path_));
    }

    void expect_removed() const {
        const int fd = ::shm_open(shm_name_.c_str(), O_RDWR, 0600);
        EXPECT_EQ(fd, -1);
        if (fd >= 0) {
            ::close(fd);
        }
        EXPECT_FALSE(std::filesystem::exists(file_path_));
    }

    std::unique_ptr<OwnerProcess> owner_;
    std::string directory_, shm_name_, file_path_;
};

TEST_F(ShmResourceTrackerTest, LiveOwnerSurvivesManifestReplacementAndRepeatedSweeps) {
    start_owner();
    const auto manifest = manifest_path();
    ASSERT_FALSE(manifest.empty());
    for (int i = 0; i < 10; ++i) {
        ASSERT_TRUE(owner_->command('u'));
        ShmResourceTracker::cleanup_stale_resources();
        expect_attachable();
        EXPECT_EQ(manifest_path(), manifest);
    }
}

TEST_F(ShmResourceTrackerTest, KilledOwnerIsReapedWithManifestAndTemporaryFile) {
    start_owner();
    const auto manifest = manifest_path();
    ASSERT_FALSE(manifest.empty());
    std::ofstream(manifest + ".tmp") << "incomplete replacement";
    const auto status = owner_->stop(SIGKILL);
    ASSERT_TRUE(WIFSIGNALED(status));
    ShmResourceTracker::cleanup_stale_resources();
    expect_removed();
    EXPECT_FALSE(std::filesystem::exists(manifest));
    EXPECT_FALSE(std::filesystem::exists(manifest + ".tmp"));
    EXPECT_FALSE(std::filesystem::exists(manifest + ".lock"));
    ShmResourceTracker::cleanup_stale_resources();
}

TEST_F(ShmResourceTrackerTest, NormalExitCleansResourcesAndLease) {
    start_owner();
    const auto manifest = manifest_path();
    ASSERT_FALSE(manifest.empty());
    const auto status = owner_->exit_normally();
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0);
    expect_removed();
    EXPECT_FALSE(std::filesystem::exists(manifest));
    EXPECT_FALSE(std::filesystem::exists(manifest + ".lock"));
}

TEST_F(ShmResourceTrackerTest, SignalExitCleansResources) {
    start_owner();
    const auto status = owner_->stop(SIGTERM);
    ASSERT_TRUE(WIFSIGNALED(status));
    EXPECT_EQ(WTERMSIG(status), SIGTERM);
    expect_removed();
}

TEST_F(ShmResourceTrackerTest, EmptyTrackerRetainsLeaseForLaterResources) {
    start_owner();
    const auto manifest = manifest_path();
    ASSERT_FALSE(manifest.empty());
    ASSERT_TRUE(owner_->command('c'));
    expect_removed();
    EXPECT_FALSE(std::filesystem::exists(manifest));
    ShmResourceTracker::cleanup_stale_resources();
    EXPECT_TRUE(std::filesystem::exists(manifest + ".lock"));
    create_resources();
    ASSERT_TRUE(owner_->command('t'));
    ShmResourceTracker::cleanup_stale_resources();
    expect_attachable();
    EXPECT_EQ(manifest_path(), manifest);
}

TEST_F(ShmResourceTrackerTest, LegacyResourcesWithoutLeaseAreLeftAlone) {
    create_resources();
    const auto manifest = "/dev/shm/tt_socket_manifest_2147483647";
    ASSERT_FALSE(std::filesystem::exists(manifest));
    std::ofstream(manifest) << "shm " << shm_name_ << "\nfile " << file_path_ << "\n";
    ShmResourceTracker::cleanup_stale_resources();
    expect_attachable();
    EXPECT_TRUE(std::filesystem::exists(manifest));
    ::unlink(manifest);
}

TEST_F(ShmResourceTrackerTest, ConcurrentSweepersDoNotRaceLiveManifestUpdates) {
    start_owner();
    const pid_t sweeper = ::fork();
    ASSERT_GE(sweeper, 0);
    if (sweeper == 0) {
        for (int i = 0; i < 100; ++i) {
            ShmResourceTracker::cleanup_stale_resources();
        }
        ::_exit(0);
    }
    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(owner_->command('u'));
        ShmResourceTracker::cleanup_stale_resources();
    }
    int status = 0;
    ASSERT_EQ(::waitpid(sweeper, &status, 0), sweeper);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
    expect_attachable();
}

TEST_F(ShmResourceTrackerTest, SweeperInForeignPidNamespaceKeepsLiveOwner) {
    start_owner();
    const pid_t owner_pid = owner_->pid();
    ASSERT_GT(owner_pid, 1);
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        if (::unshare(CLONE_NEWPID) != 0) {
            ::_exit(77);
        }
        const pid_t sweeper = ::fork();
        if (sweeper == 0) {
            // A fresh PID namespace contains only this PID 1. The live owner's
            // PID is absent, reproducing the Kubernetes hostIPC failure.
            if (::kill(owner_pid, 0) != -1 || errno != ESRCH) {
                ::_exit(2);
            }
            ShmResourceTracker::cleanup_stale_resources();
            const int fd = ::shm_open(shm_name_.c_str(), O_RDWR, 0600);
            ::_exit(fd >= 0 && std::filesystem::exists(file_path_) ? 0 : 1);
        }
        int status = 0;
        if (sweeper < 0 || ::waitpid(sweeper, &status, 0) < 0 || !WIFEXITED(status)) {
            ::_exit(3);
        }
        ::_exit(WEXITSTATUS(status));
    }
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    if (WEXITSTATUS(status) == 77) {
        GTEST_SKIP() << "Creating a PID namespace requires CAP_SYS_ADMIN";
    }
    EXPECT_EQ(WEXITSTATUS(status), 0);
    expect_attachable();
}

}  // namespace
}  // namespace tt::tt_metal::distributed
