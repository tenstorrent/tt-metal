// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TT_METAL_EMULE_MODE selects an emulated device: the simulator takes precedence over it, it takes precedence over
// mock (both read TT_METAL_MOCK_CLUSTER_DESC_PATH), and TT_METAL_EMULE_SOC_DESC_PATH needs it. None of this needs a
// device: RunTimeOptions reads the environment.

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

#include "llrt/rtoptions.hpp"

namespace {

// Set (or clear) one environment variable for the test's life and put the old value back.
class ScopedEmuleModeEnv {
public:
    ScopedEmuleModeEnv(const char* name, const std::optional<std::string>& value) : name_(name) {
        if (const char* old = std::getenv(name)) {
            old_ = old;
        }
        apply(value);
    }
    ~ScopedEmuleModeEnv() { apply(old_); }
    ScopedEmuleModeEnv(const ScopedEmuleModeEnv&) = delete;
    ScopedEmuleModeEnv& operator=(const ScopedEmuleModeEnv&) = delete;

private:
    void apply(const std::optional<std::string>& value) {
        if (value.has_value()) {
            setenv(name_, value->c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }
    const char* name_;
    std::optional<std::string> old_;
};

struct TargetEnv {
    std::optional<std::string> simulator;
    std::optional<std::string> emule_mode;
    std::optional<std::string> mock_cluster_desc;
    std::optional<std::string> emule_soc_desc;
};

// The variables under test set as given, every other target variable cleared.
class RunTimeOptionsEmuleMode : public ::testing::Test {
protected:
    static constexpr const char* kClusterDesc = "/descriptors/wormhole_N150.yaml";

    void set(const TargetEnv& env) {
        simulator_.emplace("TT_METAL_SIMULATOR", env.simulator);
        emule_mode_.emplace("TT_METAL_EMULE_MODE", env.emule_mode);
        mock_cluster_desc_.emplace("TT_METAL_MOCK_CLUSTER_DESC_PATH", env.mock_cluster_desc);
        emule_soc_desc_.emplace("TT_METAL_EMULE_SOC_DESC_PATH", env.emule_soc_desc);
    }

    // Throws what RunTimeOptions throws, with its message.
    static std::string construction_error() {
        try {
            tt::llrt::RunTimeOptions opts;
        } catch (const std::exception& e) {
            return e.what();
        }
        return {};
    }

private:
    std::optional<ScopedEmuleModeEnv> simulator_, emule_mode_, mock_cluster_desc_, emule_soc_desc_;
};

TEST_F(RunTimeOptionsEmuleMode, CPU_EmuleModeIsAnEmulatedTargetNotAMockOne) {
    set({.emule_mode = "1", .mock_cluster_desc = kClusterDesc});
    tt::llrt::RunTimeOptions opts;
    EXPECT_EQ(opts.get_target_device(), tt::TargetDevice::Emule);
    EXPECT_TRUE(opts.has_mock_cluster_desc()) << "an emulated device takes its cluster shape from the descriptor";
    EXPECT_FALSE(opts.get_mock_enabled()) << "an emulated device runs kernels, so it is not mock";
}

TEST_F(RunTimeOptionsEmuleMode, CPU_ClusterDescriptorAloneIsMock) {
    set({.mock_cluster_desc = kClusterDesc});
    tt::llrt::RunTimeOptions opts;
    EXPECT_EQ(opts.get_target_device(), tt::TargetDevice::Mock);
    EXPECT_TRUE(opts.has_mock_cluster_desc());
    EXPECT_TRUE(opts.get_mock_enabled());
}

TEST_F(RunTimeOptionsEmuleMode, CPU_SimulatorTakesPrecedenceOverEmuleMode) {
    set({.simulator = "/models/emu-quasar-2x3", .emule_mode = "1", .mock_cluster_desc = kClusterDesc});
    tt::llrt::RunTimeOptions opts;
    EXPECT_EQ(opts.get_target_device(), tt::TargetDevice::Simulator);
}

TEST_F(RunTimeOptionsEmuleMode, CPU_EmuleModeNeedsAClusterDescriptor) {
    set({.emule_mode = "1"});
    EXPECT_NE(construction_error().find("requires TT_METAL_MOCK_CLUSTER_DESC_PATH"), std::string::npos);
}

TEST_F(RunTimeOptionsEmuleMode, CPU_EmuleSocDescriptorIsTakenInEmulatedMode) {
    set({.emule_mode = "1", .mock_cluster_desc = kClusterDesc, .emule_soc_desc = "/descriptors/soc.yaml"});
    tt::llrt::RunTimeOptions opts;
    EXPECT_EQ(opts.get_emule_soc_desc_path(), "/descriptors/soc.yaml");
}

TEST_F(RunTimeOptionsEmuleMode, CPU_EmuleSocDescriptorRequiresEmuleMode) {
    set({.mock_cluster_desc = kClusterDesc, .emule_soc_desc = "/descriptors/soc.yaml"});
    EXPECT_NE(construction_error().find("requires TT_METAL_EMULE_MODE=1"), std::string::npos);
}

TEST_F(RunTimeOptionsEmuleMode, CPU_EmuleSocDescriptorUnderTheSimulatorNamesThePrecedence) {
    set(
        {.simulator = "/models/emu-quasar-2x3",
         .emule_mode = "1",
         .mock_cluster_desc = kClusterDesc,
         .emule_soc_desc = "/descriptors/soc.yaml"});
    const std::string error = construction_error();
    EXPECT_NE(error.find("takes precedence over TT_METAL_EMULE_MODE"), std::string::npos) << error;
}

}  // namespace
