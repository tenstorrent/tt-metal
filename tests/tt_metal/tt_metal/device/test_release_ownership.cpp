// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/wait.h>
#include <unistd.h>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include <umd/device/firmware/firmware_info_provider.hpp>
#include <umd/device/tt_device/tt_device.hpp>
#include <umd/device/types/arch.hpp>

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

    // Open one unit mesh per MMIO chip. Multi-chip boards (e.g. P300) expose more than one
    // MMIO chip, so assert against the number of chips opened rather than hardcoding a single device.
    auto devices = distributed::MeshDevice::create_unit_meshes(
        ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config);

    ASSERT_EQ(devices.size(), ids.size());
    for (auto& [id, device] : devices) {
        (void)id;
        device->close();
    }
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

namespace {

constexpr const char* kTdpLimitEnvVar = "TT_METAL_TDP_LIMIT_WATTS";
constexpr uint32_t kTestTdpLimitWatts = 200;

// Mirrors TDP_LIMIT_MIN_FIRMWARE_VERSION in UMD's firmware_utils.cpp, which is file-local there.
const umd::FirmwareBundleVersion kMinTdpLimitFirmware(19, 11, 0);

// Rebuilds the cluster with TT_METAL_TDP_LIMIT_WATTS set to `value`, then reports the limit
// firmware ends up enforcing on every PCIe-attached chip, since the limit is applied per ASIC. The
// knob is read while a context builds its RunTimeOptions, so any existing context has to go first.
std::map<ChipId, std::optional<uint32_t>> open_cluster_with_tdp_limit(const std::optional<std::string>& value) {
    if (value.has_value()) {
        setenv(kTdpLimitEnvVar, value->c_str(), /*overwrite=*/1);
    } else {
        unsetenv(kTdpLimitEnvVar);
    }
    if (MetalContext::instance_exists()) {
        detail::ReleaseOwnership();
    }

    const Cluster& cluster = MetalContext::instance().get_cluster();
    std::map<ChipId, std::optional<uint32_t>> limits;
    for (const ChipId chip_id : cluster.mmio_chip_ids()) {
        limits[chip_id] =
            cluster.get_driver()->get_chip(chip_id)->get_tt_device()->get_firmware_info_provider()->get_tdp_limit();
    }
    return limits;
}

// Restores TT_METAL_TDP_LIMIT_WATTS and drops the context it configured, so the tests that follow
// in this binary start from the environment they expect. Parsing of the variable itself is covered
// by the TdpLimitEnv tests in test_device_init_and_teardown.cpp, which need no device.
class TdpLimitFixture : public ::testing::Test {
protected:
    void SetUp() override {
        const char* prev = getenv(kTdpLimitEnvVar);
        prev_ = prev != nullptr ? std::optional<std::string>(prev) : std::nullopt;
    }

    void TearDown() override {
        if (prev_.has_value()) {
            setenv(kTdpLimitEnvVar, prev_->c_str(), /*overwrite=*/1);
        } else {
            unsetenv(kTdpLimitEnvVar);
        }
        if (MetalContext::instance_exists()) {
            detail::ReleaseOwnership();
        }
    }

private:
    std::optional<std::string> prev_;
};

}  // namespace

// The knob is optional, so nothing it can be handed may take a run down. This holds on any
// architecture: Wormhole cannot set a limit at all, and Blackhole firmware rejects one below the
// range it accepts. Both have to come out as a warning rather than a failed cluster open.
TEST_F(TdpLimitFixture, RejectedLimitDoesNotFailClusterOpen) { EXPECT_NO_THROW(open_cluster_with_tdp_limit("1")); }

// Surviving the rejection is not enough: firmware must still be enforcing what it was before.
TEST_F(TdpLimitFixture, RejectedLimitLeavesFirmwareUntouched) {
    const std::map<ChipId, std::optional<uint32_t>> before = open_cluster_with_tdp_limit(std::nullopt);

    // 1 W is below the [50, 500] W window firmware accepts, so the write is refused and warned about.
    for (const auto& [chip_id, limit] : open_cluster_with_tdp_limit("1")) {
        EXPECT_EQ(limit, before.at(chip_id)) << "chip " << chip_id << " moved after a refused limit";
    }
}

// The whole lifecycle: the limit is applied, it outlives the cluster that set it, an unset knob
// leaves it alone, and the 0 sentinel puts the board default back. The last step is also cleanup.
TEST_F(TdpLimitFixture, LimitOutlivesClusterAndSentinelRestoresIt) {
    const std::map<ChipId, std::optional<uint32_t>> board_defaults = open_cluster_with_tdp_limit(std::nullopt);

    // Metal requires one architecture across the cluster, so the first chip gates this for all.
    const Cluster& cluster = MetalContext::instance().get_cluster();
    const ChipId first_chip_id = *cluster.mmio_chip_ids().begin();
    if (cluster.arch() != tt::ARCH::BLACKHOLE || cluster.get_driver()
                                                         ->get_chip(first_chip_id)
                                                         ->get_tt_device()
                                                         ->get_firmware_info_provider()
                                                         ->get_firmware_version() < kMinTdpLimitFirmware) {
        GTEST_SKIP() << "TDP limit needs Blackhole with firmware " << kMinTdpLimitFirmware.to_string() << " or newer";
    }
    for (const auto& [chip_id, limit] : board_defaults) {
        ASSERT_NE(limit, kTestTdpLimitWatts) << "chip " << chip_id << " already sits at the limit under test";
    }

    for (const auto& [chip_id, limit] : open_cluster_with_tdp_limit(std::to_string(kTestTdpLimitWatts))) {
        EXPECT_EQ(limit, kTestTdpLimitWatts) << "chip " << chip_id << " did not take the limit";
    }
    for (const auto& [chip_id, limit] : open_cluster_with_tdp_limit(std::nullopt)) {
        EXPECT_EQ(limit, kTestTdpLimitWatts) << "chip " << chip_id << " lost the limit when its cluster closed";
    }
    for (const auto& [chip_id, limit] :
         open_cluster_with_tdp_limit(std::to_string(llrt::TDP_LIMIT_RESTORE_DEFAULT_SENTINEL))) {
        EXPECT_EQ(limit, board_defaults.at(chip_id)) << "chip " << chip_id << " was not put back to its default";
    }
}

}  // namespace tt::tt_metal
