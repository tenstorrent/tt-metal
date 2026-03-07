// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <thread>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

#include "context/context_descriptor.hpp"
#include <tt-metalium/experimental/context/metal_env.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_cluster.hpp"
#include "impl/device/mock_device_util.hpp"

namespace tt::tt_metal {

class MetalContextTest : public ::testing::Test {
protected:
    void TearDown() override { MetalContext::destroy_all_instances(); }
};

TEST_F(MetalContextTest, CreateSiliconInstance) {
    MetalEnv env;
    ContextId context_id = MetalContext::create_instance(env);
    EXPECT_EQ(context_id, DEFAULT_CONTEXT_ID);
}

TEST_F(MetalContextTest, AccessInvalidContextId) {
    EXPECT_THROW(MetalContext::instance(ContextId{-1}), std::runtime_error);
    EXPECT_THROW(MetalContext::instance(ContextId{MAX_CONTEXT_COUNT}), std::runtime_error);
}

TEST_F(MetalContextTest, MultipleSiliconInstancesSameEnv) {
    MetalEnv env;
    ContextId context_id = MetalContext::create_instance(env);
    EXPECT_EQ(context_id, DEFAULT_CONTEXT_ID);
    EXPECT_THROW(MetalContext::create_instance(env), std::runtime_error);
}

TEST_F(MetalContextTest, LegacyImplicitSiliconInstance) {
    // Implicit init to support legacy behaviour
    auto& context = MetalContext::instance();
    EXPECT_FALSE(context.rtoptions().get_mock_enabled());
}

TEST_F(MetalContextTest, CreateMockInstances) {
    MetalEnv env_wh{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value())};
    ContextId context_id_wh = MetalContext::create_instance(env_wh);
    EXPECT_EQ(context_id_wh, 1);

    MetalEnv env_bh{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 1).value())};
    ContextId context_id_bh = MetalContext::create_instance(env_bh);
    EXPECT_EQ(context_id_bh, 2);
}

TEST_F(MetalContextTest, CreateSiliconInstanceWithMockInstances) {
    MetalEnv env;
    ContextId context_id = MetalContext::create_instance(env);
    EXPECT_EQ(context_id, DEFAULT_CONTEXT_ID);

    MetalEnv env_wh{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value())};
    ContextId context_id_wh = MetalContext::create_instance(env_wh);

    MetalEnv env_bh{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 1).value())};
    ContextId context_id_bh = MetalContext::create_instance(env_bh);

    ASSERT_EQ(MetalContext::instance(context_id_wh).get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
    ASSERT_EQ(MetalContext::instance(context_id_bh).get_cluster().arch(), tt::ARCH::BLACKHOLE);
}

TEST_F(MetalContextTest, DestroyInstanceExplicit) {
    // Instance should not exist after being destroyed
    MetalEnv env_wh{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value())};
    ContextId context_id_wh = MetalContext::create_instance(env_wh);
    ASSERT_EQ(MetalContext::instance(context_id_wh).get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
    MetalContext::destroy_instance(false, context_id_wh);
    EXPECT_THROW(MetalContext::instance(context_id_wh), std::runtime_error);
}

TEST_F(MetalContextTest, CreateImplicitAfterDestroy) {
    ContextId context_id;
    {
        MetalEnv env;
        context_id = MetalContext::create_instance(env);
        MetalContext::destroy_instance(false, context_id);
    }
    EXPECT_EQ(context_id, DEFAULT_CONTEXT_ID);
    auto& context = MetalContext::instance(context_id);
    EXPECT_EQ(context.rtoptions().get_mock_enabled(), false);
}

TEST_F(MetalContextTest, ThreadIsolation) {
    ContextId mock_context_id{};
    ContextId DEFAULT_CONTEXT_ID{};
    bool mock_ok{false};
    bool silicon_ok{false};

    std::thread mock_thread([&]() {
        MetalEnv env{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value())};
        ContextId id = MetalContext::create_instance(env);
        mock_context_id = id;
        mock_ok = MetalContext::instance(id).rtoptions().get_mock_enabled() &&
                  MetalContext::instance(id).get_cluster().arch() == tt::ARCH::WORMHOLE_B0;
        MetalContext::destroy_instance(false, id);
    });
    std::thread silicon_thread([&]() {
        MetalEnv env;
        ContextId id = MetalContext::create_instance(env);
        DEFAULT_CONTEXT_ID = id;
        silicon_ok = (id == DEFAULT_CONTEXT_ID) && !MetalContext::instance(id).rtoptions().get_mock_enabled();
        MetalContext::destroy_instance(false, id);
    });

    mock_thread.join();
    silicon_thread.join();

    EXPECT_TRUE(mock_ok);
    EXPECT_TRUE(silicon_ok);
    EXPECT_EQ(DEFAULT_CONTEXT_ID, DEFAULT_CONTEXT_ID);
    EXPECT_GE(mock_context_id, 1);
}

TEST_F(MetalContextTest, ForkIsolation) {
    // Try to open a mock cluster on the parent process while the child process as a mock cluster open.
    int pipe_fd[2];
    ASSERT_EQ(pipe(pipe_fd), 0) << "pipe() failed";

    pid_t pid = fork();
    if (pid == -1) {
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        FAIL() << "Failed to fork";
    }
    if (pid == 0) {
        close(pipe_fd[1]);
        MetalEnv child_mock_env{
            MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value())};
        ContextId child_mock_id = MetalContext::create_instance(child_mock_env);
        if (child_mock_id.get() < 1 || !MetalContext::instance(child_mock_id).rtoptions().get_mock_enabled()) {
            MetalContext::destroy_all_instances(false);
            _exit(1);
        }
        char byte = 0;
        if (read(pipe_fd[0], &byte, 1) != 1) {
            MetalContext::destroy_all_instances(false);
            _exit(1);
        }
        close(pipe_fd[0]);
        MetalContext::destroy_instance(false, child_mock_id);
        _exit(0);
    }

    close(pipe_fd[0]);
    MetalEnv parent_mock_env{
        MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value())};
    ContextId parent_mock_id = MetalContext::create_instance(parent_mock_env);
    ASSERT_GT(parent_mock_id, DEFAULT_CONTEXT_ID);
    ASSERT_TRUE(MetalContext::instance(parent_mock_id).rtoptions().get_mock_enabled());

    // Signal child to exit
    char byte = 1;
    ASSERT_EQ(write(pipe_fd[1], &byte, 1), 1) << "write(pipe) failed";
    close(pipe_fd[1]);

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);

    EXPECT_TRUE(MetalContext::instance(parent_mock_id).rtoptions().get_mock_enabled());
    EXPECT_EQ(MetalContext::instance(parent_mock_id).get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
}

TEST_F(MetalContextTest, MaxContexts) {
    std::vector<ContextId> context_ids;
    context_ids.reserve(MAX_CONTEXT_COUNT - 1);

    auto mock_cluster_desc_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value();
    for (int i = 0; i < MAX_CONTEXT_COUNT - 1; ++i) {
        MetalEnv env{MetalEnvDescriptor(mock_cluster_desc_path)};
        context_ids.push_back(MetalContext::create_instance(env));
    }
    MetalEnv env{MetalEnvDescriptor(mock_cluster_desc_path)};
    EXPECT_THROW(MetalContext::create_instance(env), std::runtime_error);
}

TEST_F(MetalContextTest, DoubleAcquireSameEnvFails) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value();
    MetalEnv env{MetalEnvDescriptor(mock_path)};
    MetalContext::create_instance(env);
    EXPECT_THROW(MetalContext::create_instance(env), std::runtime_error);
}

TEST_F(MetalContextTest, ReuseEnvAfterDestroy) {
    auto mock_path = experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1).value();
    MetalEnv env{MetalEnvDescriptor(mock_path)};
    ContextId id = MetalContext::create_instance(env);
    MetalContext::destroy_instance(false, id);
    ContextId id2 = MetalContext::create_instance(env);
    EXPECT_EQ(MetalContext::instance(id2).get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
}

}  // namespace tt::tt_metal
