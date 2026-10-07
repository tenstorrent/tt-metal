// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/experimental/sockets/named_shm.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <sys/statfs.h>
#include <sys/wait.h>
#include <unistd.h>

namespace tt::tt_metal::distributed::test {
namespace {
class SocketHugepageEnv {
public:
    explicit SocketHugepageEnv(const char* value) {
        if (const char* old = std::getenv(key)) {
            previous_ = old;
        }
        if (value) {
            setenv(key, value, 1);
        } else {
            unsetenv(key);
        }
    }
    ~SocketHugepageEnv() {
        if (previous_) {
            setenv(key, previous_->c_str(), 1);
        } else {
            unsetenv(key);
        }
    }

private:
    static constexpr const char* key = "TT_METAL_SOCKET_HUGEPAGE_DIR";
    std::optional<std::string> previous_;
};

TEST(NamedShmHost, PosixRoundTripWithoutHugepages) {
    SocketHugepageEnv env(nullptr);
    const auto name = generate_shm_name("test");
    auto owner = NamedShm::create(name, 36864);
    auto peer = NamedShm::open(owner.name(), owner.size());
    EXPECT_EQ(owner.name(), name);
    EXPECT_EQ(owner.size(), 36864);
    static_cast<char*>(peer.ptr())[36863] = 42;
    EXPECT_EQ(static_cast<char*>(owner.ptr())[36863], 42);
    peer.close();
    owner.unlink();
    EXPECT_THROW(NamedShm::open(name, 36864), std::exception);
}

TEST(NamedShmHost, SmallRegionDoesNotNeedHugepageMount) {
    SocketHugepageEnv env("/nonexistent/socket/hugepages");
    const auto name = generate_shm_name("test");
    auto owner = NamedShm::create(name, 4096);
    EXPECT_EQ(owner.name(), name);
    EXPECT_EQ(owner.size(), 4096);
    owner.unlink();
}

TEST(NamedShmHost, RejectsOrdinaryFilesystemAndRemovesNewBacking) {
    SocketHugepageEnv env("/tmp");
    const auto name = generate_shm_name("test");
    const auto file = std::filesystem::path("/tmp") / name.substr(1);
    EXPECT_THROW(NamedShm::create(name, 36864), std::exception);
    EXPECT_FALSE(std::filesystem::exists(file));
}

TEST(NamedShmHost, RejectsOpeningOrdinaryFileWithoutRemovingIt) {
    const auto file = std::filesystem::path("/tmp") / generate_shm_name("test").substr(1);
    {
        std::ofstream out(file);
        out << "keep";
    }
    EXPECT_THROW(NamedShm::open(file.string(), 4096), std::exception);
    EXPECT_TRUE(std::filesystem::exists(file));
    std::filesystem::remove(file);
}

TEST(NamedShmHost, HugepageCrossProcessAndSizeValidation) {
    const char* dir = std::getenv("TT_TEST_SOCKET_HUGEPAGE_DIR");
    if (!dir) {
        GTEST_SKIP() << "Set TT_TEST_SOCKET_HUGEPAGE_DIR to a writable hugetlbfs mount with free pages";
    }
    SocketHugepageEnv env(dir);
    auto owner = NamedShm::create(generate_shm_name("test"), 36864);
    const auto name = owner.name();
    EXPECT_GT(owner.size(), 36864);
    EXPECT_EQ(static_cast<char*>(owner.ptr())[owner.size() - 1], 0);
    EXPECT_THROW(NamedShm::open(name, 4096), std::exception);
    EXPECT_THROW(NamedShm::create(generate_shm_name("test"), owner.size() + 1), std::exception);
    const pid_t pid = fork();
    ASSERT_NE(pid, -1);
    if (pid == 0) {
        unsetenv("TT_METAL_SOCKET_HUGEPAGE_DIR");
        try {
            auto peer = NamedShm::open(name, owner.size());
            static_cast<char*>(peer.ptr())[0] = 17;
            static_cast<char*>(peer.ptr())[36863] = 23;
            peer.close();
            _exit(0);
        } catch (...) {
            _exit(1);
        }
    }
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
    EXPECT_EQ(static_cast<char*>(owner.ptr())[0], 17);
    EXPECT_EQ(static_cast<char*>(owner.ptr())[36863], 23);
    owner.unlink();
    EXPECT_FALSE(std::filesystem::exists(name));
}
}  // namespace
}  // namespace tt::tt_metal::distributed::test
