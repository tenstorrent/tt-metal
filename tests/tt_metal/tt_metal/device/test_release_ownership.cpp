// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/wait.h>
#include <unistd.h>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

using namespace tt::tt_metal;

namespace tt::tt_metal {

// Helper function to create and close a device
static void open_and_close_device() {
    std::vector<ChipId> ids;
    for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids()) {
        ids.push_back(id);
    }
    ASSERT_GT(ids.size(), 0);
    const auto& dispatch_core_config = tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();

    auto devices = distributed::MeshDevice::create_unit_meshes(
        ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config);

    ASSERT_EQ(devices.size(), 1);
    devices[0]->close();
}

TEST(TensixReleaseOwnership, BasicReleaseOwnership) {
    // Open and close a device
    open_and_close_device();

    // Release ownership of the MetalContext
    detail::ReleaseOwnership();

    // Verify context can be re-created by opening a device again
    open_and_close_device();
}

TEST(TensixReleaseOwnership, ReleaseOwnershipWithSubprocess) {
    // Open and close a device in the parent process
    open_and_close_device();

    // Release ownership of the MetalContext
    detail::ReleaseOwnership();

    // Find the test_clean_init executable
    std::filesystem::path current_exe = std::filesystem::canonical("/proc/self/exe");
    std::filesystem::path test_dir = current_exe.parent_path();
    std::filesystem::path test_clean_init_path = test_dir / "test_clean_init";

    ASSERT_TRUE(std::filesystem::exists(test_clean_init_path))
        << "Could not find test_clean_init executable at: " << test_clean_init_path;

    log_info(tt::LogTest, "Spawning subprocess: {}", test_clean_init_path.string());

    // Spawn a subprocess that runs test_clean_init
    pid_t pid = fork();

    if (pid == -1) {
        FAIL() << "Failed to fork subprocess";
    }

    if (pid == 0) {
        // Child process
        const char* args[] = {test_clean_init_path.c_str(), nullptr};
        execv(test_clean_init_path.c_str(), const_cast<char* const*>(args));

        // If execv returns, it failed
        log_error(tt::LogTest, "Failed to execute subprocess: {}", strerror(errno));
        _exit(1);
    }

    // Parent process - wait for child to complete
    int status;
    pid_t result = waitpid(pid, &status, 0);

    ASSERT_EQ(result, pid) << "waitpid failed";

    if (WIFEXITED(status)) {
        int exit_code = WEXITSTATUS(status);
        log_info(tt::LogTest, "Subprocess exited with code: {}", exit_code);
        ASSERT_EQ(exit_code, 0) << "Subprocess failed with exit code: " << exit_code;
    } else if (WIFSIGNALED(status)) {
        int signal = WTERMSIG(status);
        FAIL() << "Subprocess terminated by signal: " << signal;
    } else {
        FAIL() << "Subprocess terminated abnormally";
    }

    log_info(tt::LogTest, "Subprocess completed successfully, verifying parent can still open device");

    // Verify the parent process can still open a device after subprocess completes
    open_and_close_device();

    log_info(tt::LogTest, "Test passed: parent process successfully opened device after subprocess");
}

}  // namespace tt::tt_metal
