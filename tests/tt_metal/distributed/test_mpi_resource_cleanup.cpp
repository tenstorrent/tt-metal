// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression tests for https://github.com/tenstorrent/tt-metal/issues/63870: a tt-metal process that was not
// launched as part of an MPI job must not leave OpenMPI files (/dev/shm/sm_segment.*, /tmp/ompi.*) behind.
//
// Each test re-executes this binary with a filter that selects one MpiResourceCleanupChild test. The child gets an
// environment without launcher variables, as a process started from a shell would, and its TMPDIR and OpenMPI
// shared-memory backing directories point into a private scratch directory. Leftovers are counted there only, so
// other MPI jobs on the same host cannot affect the result.
//
// This file is its own executable: the parent must not open devices or start MPI, and test registration in
// distributed_unit_tests does both.

#include <mpi.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "tt_metal/distributed/host_h2h_socket.hpp"

namespace tt::tt_metal::distributed {
namespace {

namespace fs = std::filesystem;
using multihost::DistributedContext;

constexpr const char* kScratchDirEnv = "TT_METAL_MPI_CLEANUP_TEST_DIR";

// Names OpenMPI 4/5, PRRTE and PMIx give the session directories and rendezvous files they create under TMPDIR.
constexpr std::array<std::string_view, 7> kMpiTmpPrefixes = {
    "ompi.", "openmpi-sessions-", "prte.", "prterun.", "prted.", "prun.", "pmix"};

// Variables set by mpirun / PRRTE / PMIx / tt-run. A child that inherited them would look like an MPI rank.
constexpr std::array<std::string_view, 5> kLauncherEnvPrefixes = {"OMPI_", "PMIX_", "PMI_", "PRTE_", "TT_RUN_"};

fs::path tmp_dir(const fs::path& scratch) { return scratch / "t"; }
fs::path shm_dir(const fs::path& scratch) { return scratch / "s"; }

// Everything OpenMPI left in the scratch directory: all of the shared-memory backing directory, plus MPI-named
// entries under TMPDIR (tt-metal may put unrelated files there).
std::vector<fs::path> mpi_leftovers(const fs::path& scratch) {
    std::vector<fs::path> roots;
    for (const auto& entry : fs::directory_iterator(shm_dir(scratch))) {
        roots.push_back(entry.path());
    }
    for (const auto& entry : fs::directory_iterator(tmp_dir(scratch))) {
        const std::string name = entry.path().filename().string();
        for (std::string_view prefix : kMpiTmpPrefixes) {
            if (name.starts_with(prefix)) {
                roots.push_back(entry.path());
                break;
            }
        }
    }
    std::vector<fs::path> leftovers;
    for (const auto& root : roots) {
        leftovers.push_back(root);
        if (fs::is_directory(root)) {
            for (const auto& entry : fs::recursive_directory_iterator(root)) {
                leftovers.push_back(entry.path());
            }
        }
    }
    return leftovers;
}

std::string describe(const std::vector<fs::path>& paths, const fs::path& scratch) {
    std::ostringstream out;
    for (const auto& path : paths) {
        out << "\n  " << fs::relative(path, scratch).string();
        if (fs::is_regular_file(path)) {
            out << " (" << fs::file_size(path) << " bytes)";
        }
    }
    return out.str();
}

std::optional<fs::path> child_scratch_dir() {
    const char* dir = std::getenv(kScratchDirEnv);
    if (dir == nullptr) {
        return std::nullopt;
    }
    return fs::path(dir);
}

// Proves the child test body actually ran, so a skipped or filtered-out child cannot pass as "left nothing behind".
void mark_child_ran(const fs::path& scratch, int rank = 0) {
    std::ofstream(scratch / ("ran." + std::to_string(rank))) << getpid() << "\n";
}

std::optional<std::string> find_mpirun() {
    if (const char* launcher = std::getenv("TT_METAL_TEST_MPIRUN"); launcher != nullptr) {
        return std::string(launcher);
    }
    const char* path = std::getenv("PATH");
    if (path == nullptr) {
        return std::nullopt;
    }
    // Same preference as tt-run: the ULFM launcher matches the ULFM libmpi tt-metal links against.
    for (std::string_view name : {"mpirun-ulfm", "mpirun"}) {
        std::string_view dirs(path);
        while (!dirs.empty()) {
            const auto colon = dirs.find(':');
            const fs::path candidate = fs::path(dirs.substr(0, colon)) / name;
            if (access(candidate.c_str(), X_OK) == 0) {
                return candidate.string();
            }
            dirs = colon == std::string_view::npos ? std::string_view{} : dirs.substr(colon + 1);
        }
    }
    return std::nullopt;
}

class MpiResourceCleanupTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Short path: PMIx puts Unix sockets under TMPDIR and sun_path is limited to 108 bytes.
        std::string pattern = (fs::temp_directory_path() / "ttmpiXXXXXX").string();
        ASSERT_NE(mkdtemp(pattern.data()), nullptr) << "mkdtemp failed: " << std::strerror(errno);
        scratch_ = pattern;
        fs::create_directory(tmp_dir(scratch_));
        fs::create_directory(shm_dir(scratch_));
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(scratch_, ec);
    }

    struct ChildResult {
        bool spawned = false;
        bool timed_out = false;
        int status = 0;
    };

    std::vector<std::string> child_environment(const std::vector<std::string>& extra_env) const {
        std::vector<std::string> env;
        for (char** entry = environ; *entry != nullptr; ++entry) {
            const std::string_view var(*entry);
            bool drop = var.starts_with("TMPDIR=") || var.starts_with("TT_METAL_MPI_INIT=");
            for (std::string_view prefix : kLauncherEnvPrefixes) {
                drop = drop || var.starts_with(prefix);
            }
            if (!drop) {
                env.emplace_back(var);
            }
        }
        env.push_back("TMPDIR=" + tmp_dir(scratch_).string());
        env.push_back("OMPI_MCA_btl_sm_backing_directory=" + shm_dir(scratch_).string());
        env.push_back("OMPI_MCA_osc_sm_backing_directory=" + shm_dir(scratch_).string());
        env.push_back(std::string(kScratchDirEnv) + "=" + scratch_.string());
        env.insert(env.end(), extra_env.begin(), extra_env.end());
        return env;
    }

    ChildResult spawn_and_wait(
        const std::vector<std::string>& args,
        std::chrono::seconds timeout,
        const std::vector<std::string>& extra_env = {}) const {
        std::vector<std::string> env = child_environment(extra_env);
        std::vector<char*> envp;
        envp.reserve(env.size() + 1);
        for (auto& var : env) {
            envp.push_back(var.data());
        }
        envp.push_back(nullptr);
        std::vector<std::string> argv_storage = args;
        std::vector<char*> argv;
        argv.reserve(argv_storage.size() + 1);
        for (auto& arg : argv_storage) {
            argv.push_back(arg.data());
        }
        argv.push_back(nullptr);

        ChildResult result;
        pid_t pid = 0;
        // The child leads a new process group, so a timeout can signal mpirun and the ranks it started together.
        posix_spawnattr_t attr;
        posix_spawnattr_init(&attr);
        posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
        posix_spawnattr_setpgroup(&attr, 0);
        // posix_spawn rather than fork(): the parent may already be multi-threaded.
        const int spawn_rc = posix_spawn(&pid, argv[0], nullptr, &attr, argv.data(), envp.data());
        posix_spawnattr_destroy(&attr);
        if (spawn_rc != 0) {
            return result;
        }
        result.spawned = true;

        if (!wait_until(pid, result.status, std::chrono::steady_clock::now() + timeout)) {
            result.timed_out = true;
            // SIGTERM first: mpirun then tears down its ranks, including any it moved out of this process group.
            kill(-pid, SIGTERM);
            if (!wait_until(pid, result.status, std::chrono::steady_clock::now() + std::chrono::seconds(10))) {
                kill(-pid, SIGKILL);
                while (waitpid(pid, &result.status, 0) < 0 && errno == EINTR) {
                }
            }
            // Ranks that outlive the launcher keep its stdout open, and with it the CI log pipe.
            kill(-pid, SIGKILL);
        }
        return result;
    }

    // False if pid is still running at the deadline.
    static bool wait_until(pid_t pid, int& status, std::chrono::steady_clock::time_point deadline) {
        while (true) {
            const pid_t wait_result = waitpid(pid, &status, WNOHANG);
            if (wait_result < 0 && errno == EINTR) {
                continue;
            }
            if (wait_result != 0) {
                return true;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    ChildResult run_child(
        std::string_view child_test,
        std::chrono::seconds timeout,
        const std::vector<std::string>& extra_env = {}) const {
        return spawn_and_wait(
            {"/proc/self/exe", "--gtest_filter=MpiResourceCleanupChild." + std::string(child_test)},
            timeout,
            extra_env);
    }

    void expect_clean_exit(const ChildResult& result, std::chrono::seconds timeout) const {
        ASSERT_TRUE(result.spawned) << "posix_spawn failed";
        ASSERT_FALSE(result.timed_out) << "child did not exit within " << timeout.count() << "s and was killed";
        ASSERT_TRUE(WIFEXITED(result.status)) << "child terminated abnormally (status " << result.status << ")";
        ASSERT_EQ(WEXITSTATUS(result.status), 0) << "child test failed; see its output above";
    }

    void expect_child_ran(int rank = 0) const {
        EXPECT_TRUE(fs::exists(scratch_ / ("ran." + std::to_string(rank))))
            << "child rank " << rank << " did not run its test body (skipped or filtered out?)";
    }

    void expect_no_mpi_leftovers() const {
        const auto leftovers = mpi_leftovers(scratch_);
        EXPECT_TRUE(leftovers.empty()) << "OpenMPI files left behind after the child exited (scratch " << scratch_
                                       << "):" << describe(leftovers, scratch_);
    }

    fs::path scratch_;
};

// Positive control: a real MPI singleton is visible to mpi_leftovers() while it runs. Without this, a change in
// where OpenMPI puts its files would make the leak checks below pass vacuously.
TEST_F(MpiResourceCleanupTest, HarnessDetectsLiveMpiSingleton) {
    constexpr auto timeout = std::chrono::seconds(120);
    const auto result = run_child("RawMpiSingletonIsVisible", timeout);
    expect_clean_exit(result, timeout);
    expect_child_ran();
}

// The trigger named in #63870: asking for the current world starts an MPI runtime when no launcher started one.
TEST_F(MpiResourceCleanupTest, CurrentWorldWithoutLauncherLeavesNoMpiFiles) {
    constexpr auto timeout = std::chrono::seconds(120);
    const auto result = run_child("GetCurrentWorldWithoutMpi", timeout);
    expect_clean_exit(result, timeout);
    expect_child_ran();
    expect_no_mpi_leftovers();
}

TEST_F(MpiResourceCleanupTest, MpiInitOverrideStartsMpiWithoutLauncher) {
    constexpr auto timeout = std::chrono::seconds(120);
    const auto result = run_child("GetCurrentWorldWithMpi", timeout, {"TT_METAL_MPI_INIT=always"});
    expect_clean_exit(result, timeout);
    expect_child_ran();
}

// What every single-process tt-metal program does. Uses real hardware, or a mock cluster when
// TT_METAL_MOCK_CLUSTER_DESC_PATH is set.
TEST_F(MpiResourceCleanupTest, DeviceOpenWithoutLauncherLeavesNoMpiFiles) {
    constexpr auto timeout = std::chrono::seconds(600);
    const auto result = run_child("OpenAndCloseUnitMesh", timeout);
    expect_clean_exit(result, timeout);
    expect_child_ran();
    expect_no_mpi_leftovers();
}

// An application that initializes MPI itself (mpi4py, for example) before its first tt-metal call. tt-metal must
// reuse that runtime: a second MPI_Init, or a second MPI_Finalize at exit, aborts the process.
TEST_F(MpiResourceCleanupTest, ExternallyInitializedMpiIsReused) {
    constexpr auto timeout = std::chrono::seconds(120);
    const auto result = run_child("GetCurrentWorldAfterExternalMpiInit", timeout);
    expect_clean_exit(result, timeout);
    expect_child_ran();
}

// Socket bringup runs MPI collectives, which abort the process when no launcher started MPI. It must report an
// error instead.
TEST_F(MpiResourceCleanupTest, H2HSocketWithoutLauncherReportsError) {
    constexpr auto timeout = std::chrono::seconds(120);
    const auto result = run_child("CreateH2HSocketWithoutMpi", timeout);
    expect_clean_exit(result, timeout);
    expect_child_ran();
    expect_no_mpi_leftovers();
}

// A real multi-rank job must still get an MPI world (no false negative in launcher detection) and the launcher must
// clean up after it.
TEST_F(MpiResourceCleanupTest, MpirunJobGetsMpiWorldAndLeavesNoMpiFiles) {
    const auto mpirun = find_mpirun();
    if (!mpirun.has_value()) {
        GTEST_SKIP() << "no mpirun-ulfm or mpirun in PATH (set TT_METAL_TEST_MPIRUN to override)";
    }
    std::vector<std::string> args = {*mpirun, "-np", "2", "--oversubscribe"};
    if (geteuid() == 0) {
        args.emplace_back("--allow-run-as-root");
    }
    for (const char* var : {"TMPDIR", kScratchDirEnv}) {
        args.emplace_back("-x");
        args.emplace_back(var);
    }
    args.emplace_back(fs::canonical("/proc/self/exe").string());
    args.emplace_back("--gtest_filter=MpiResourceCleanupChild.MpiWorldOfTwo");

    constexpr auto timeout = std::chrono::seconds(180);
    const auto result = spawn_and_wait(args, timeout);
    expect_clean_exit(result, timeout);
    expect_child_ran(0);
    expect_child_ran(1);
    expect_no_mpi_leftovers();
}

// ---------------------------------------------------------------------------------------------------------------
// Child bodies. They only do something when run by MpiResourceCleanupTest, which sets kScratchDirEnv.

TEST(MpiResourceCleanupChild, RawMpiSingletonIsVisible) {
    const auto scratch = child_scratch_dir();
    if (!scratch.has_value()) {
        GTEST_SKIP() << "runs only as a child of MpiResourceCleanupTest";
    }
    int provided = 0;
    ASSERT_EQ(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided), MPI_SUCCESS);
    const auto live = mpi_leftovers(*scratch);
    MPI_Finalize();
    EXPECT_FALSE(live.empty()) << "a live MPI singleton created nothing under " << *scratch
                               << "; the leak checks in this file cannot see OpenMPI's files";
    mark_child_ran(*scratch);
}

// Asks for the current world and checks whether that started MPI; a size-1 world is expected either way.
void get_current_world_and_expect_mpi(bool expect_mpi) {
    const auto scratch = child_scratch_dir();
    if (!scratch.has_value()) {
        GTEST_SKIP() << "runs only as a child of MpiResourceCleanupTest";
    }
    const auto& world = DistributedContext::get_current_world();
    ASSERT_NE(world, nullptr);
    EXPECT_EQ(*world->size(), 1);
    EXPECT_EQ(*world->rank(), 0);
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    EXPECT_EQ(mpi_initialized != 0, expect_mpi) << "MPI_Initialized after get_current_world()";
    mark_child_ran(*scratch);
}

TEST(MpiResourceCleanupChild, GetCurrentWorldWithoutMpi) { get_current_world_and_expect_mpi(false); }

TEST(MpiResourceCleanupChild, GetCurrentWorldWithMpi) { get_current_world_and_expect_mpi(true); }

TEST(MpiResourceCleanupChild, OpenAndCloseUnitMesh) {
    const auto scratch = child_scratch_dir();
    if (!scratch.has_value()) {
        GTEST_SKIP() << "runs only as a child of MpiResourceCleanupTest";
    }
    auto mesh_device = MeshDevice::create_unit_mesh(0);
    ASSERT_NE(mesh_device, nullptr);
    mesh_device->close();
    mesh_device.reset();
    mark_child_ran(*scratch);
}

TEST(MpiResourceCleanupChild, GetCurrentWorldAfterExternalMpiInit) {
    const auto scratch = child_scratch_dir();
    if (!scratch.has_value()) {
        GTEST_SKIP() << "runs only as a child of MpiResourceCleanupTest";
    }
    int provided = 0;
    ASSERT_EQ(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided), MPI_SUCCESS);
    const auto& world = DistributedContext::get_current_world();
    ASSERT_NE(world, nullptr);
    EXPECT_EQ(*world->size(), 1);
    EXPECT_EQ(*world->rank(), 0);
    world->barrier();
    // The application owns MPI_Finalize. If tt-metal registered its own as well, the exit after this aborts.
    EXPECT_EQ(MPI_Finalize(), MPI_SUCCESS);
    mark_child_ran(*scratch);
}

TEST(MpiResourceCleanupChild, CreateH2HSocketWithoutMpi) {
    const auto scratch = child_scratch_dir();
    if (!scratch.has_value()) {
        GTEST_SKIP() << "runs only as a child of MpiResourceCleanupTest";
    }
    // A default config fails the local checks, so create() goes straight to the agreement collective.
    std::string err;
    EXPECT_EQ(experimental::H2HSocket::create({}, err), nullptr);
    EXPECT_NE(err.find("MPI is not initialized"), std::string::npos) << "unexpected error: " << err;
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    EXPECT_EQ(mpi_initialized, 0) << "socket bringup started MPI";
    mark_child_ran(*scratch);
}

TEST(MpiResourceCleanupChild, MpiWorldOfTwo) {
    const auto scratch = child_scratch_dir();
    if (!scratch.has_value()) {
        GTEST_SKIP() << "runs only as a child of MpiResourceCleanupTest";
    }
    const auto& world = DistributedContext::get_current_world();
    ASSERT_NE(world, nullptr);
    ASSERT_EQ(*world->size(), 2) << "launched by mpirun -np 2 but did not get an MPI world of two ranks";
    world->barrier();
    mark_child_ran(*scratch, *world->rank());
}

}  // namespace
}  // namespace tt::tt_metal::distributed
