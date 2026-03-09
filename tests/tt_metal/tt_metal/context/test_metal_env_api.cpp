// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>
#include <tt-metalium/experimental/context/metal_env.hpp>

#include "impl/context/metal_env_accessor.hpp"
#include "impl/device/mock_device_util.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal {

TEST(MetalEnv, Init) { auto env = MetalEnv(); }

TEST(MetalEnv, Mock) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value();
    MetalEnvDescriptor settings(mock_path);
    MetalEnv env_1(settings);
    MetalEnv env_2(settings);
    EXPECT_EQ(env_1.get_arch(), tt::ARCH::WORMHOLE_B0);
    EXPECT_EQ(env_2.get_arch(), tt::ARCH::WORMHOLE_B0);
}

TEST(MetalEnv, OnePhysicalMultipleMock) {
    auto env_physical = MetalEnv();

    auto mock_path_1 = experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2).value();
    MetalEnv env_mock_1{MetalEnvDescriptor(mock_path_1)};

    auto mock_path_2 = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 2).value();
    MetalEnv env_mock_2{MetalEnvDescriptor(mock_path_2)};

    EXPECT_EQ(MetalEnvAccessor(env_mock_1).get_cluster().arch(), tt::ARCH::BLACKHOLE);
    EXPECT_EQ(MetalEnvAccessor(env_mock_2).get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
}

TEST(MetalEnv, EnvQueries) {
    MetalEnv env;
    EXPECT_NO_THROW(env.get_arch());
    EXPECT_NO_THROW(env.get_arch_name());
    EXPECT_NO_THROW(env.get_num_pcie_devices());
    EXPECT_NO_THROW(env.get_l1_size());
    EXPECT_NO_THROW(env.get_dram_alignment());
    EXPECT_NO_THROW(env.get_l1_alignment());
    EXPECT_NO_THROW(env.get_arch_num_circular_buffers());
    EXPECT_NO_THROW(env.get_max_worker_l1_unreserved_size());
    EXPECT_NO_THROW(env.get_eps());
    EXPECT_NO_THROW(env.get_nan());
    EXPECT_NO_THROW(env.get_inf());
}

TEST(MetalEnv, EnvQueriesMock) {
    auto env_settings = MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2));
    MetalEnv env(env_settings);
    EXPECT_EQ(env.get_arch(), tt::ARCH::BLACKHOLE);
    EXPECT_NO_THROW(env.get_l1_size());
    EXPECT_EQ(env.get_num_pcie_devices(), 2);
    EXPECT_NO_THROW(env.get_dram_alignment());
    EXPECT_NO_THROW(env.get_l1_alignment());
    EXPECT_NO_THROW(env.get_arch_num_circular_buffers());
    EXPECT_NO_THROW(env.get_max_worker_l1_unreserved_size());
    EXPECT_NO_THROW(env.get_eps());
    EXPECT_NO_THROW(env.get_nan());
    EXPECT_NO_THROW(env.get_inf());
}

TEST(MetalEnv, ForkSafety) {
    MetalEnv env;

    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";

    if (pid == 0) {
        // Child: the prefork prepare handler already ran in the parent before
        // the fork.  If it didn't crash/abort, the child just exits cleanly.
        _exit(0);
    }

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(MetalEnv, ForkSafetyMultipleEnvs) {
    MetalEnv env_physical;

    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2).value();
    MetalEnv env_mock{MetalEnvDescriptor(mock_path)};

    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";

    if (pid == 0) {
        _exit(0);
    }

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(MetalEnv, ForkSafetyActiveEnv) {
    MetalEnv env;
    MetalEnvAccessor accessor(env);
    accessor.acquire();

    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "fork() failed";

    if (pid == 0) {
        // Child inherits the nonzero use count from the parent.
        // check_use_count_zero() should return false.
        _exit(accessor.check_use_count_zero() ? 0 : 1);
    }

    accessor.release();

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 1);
}

}  // namespace tt::tt_metal
