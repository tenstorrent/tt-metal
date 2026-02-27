// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

#include "device/mock_device_common.hpp"
#include "impl/context/metallium_object.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/context_id.hpp"
#include "tt_cluster.hpp"

namespace tt::tt_metal {

class MetalContextTest : public ::testing::Test {
protected:
    void TearDown() override { MetalContext::destroy_all_instances(); }
};

TEST_F(MetalContextTest, CreateSiliconInstance) {
    auto descriptor = std::make_shared<MetalliumObjectDescriptor>();
    ContextId context_id = MetalContext::create_instance(descriptor);
    EXPECT_EQ(context_id, SILICON_CONTEXT_ID);
}

TEST_F(MetalContextTest, MultipleSiliconInstances) {
    auto descriptor = std::make_shared<MetalliumObjectDescriptor>();
    ContextId context_id = MetalContext::create_instance(descriptor);
    EXPECT_EQ(context_id, SILICON_CONTEXT_ID);
    // Only one silicon instance is allowed
    EXPECT_THROW(MetalContext::create_instance(descriptor), std::runtime_error);
}

TEST_F(MetalContextTest, LegacyImplicitSiliconInstance) {
    // Implicit init to support legacy behaviour
    auto& context = MetalContext::instance();
    EXPECT_FALSE(context.rtoptions().get_mock_enabled());
}

TEST_F(MetalContextTest, CreateMockInstances) {
    auto descriptor = std::make_shared<MetalliumObjectDescriptor>(
        experimental::get_mock_cluster_desc_for_config(tt::ARCH::WORMHOLE_B0, 1).value());
    ContextId context_id_wh = MetalContext::create_instance(descriptor);
    EXPECT_EQ(context_id_wh, 1);

    descriptor = std::make_shared<MetalliumObjectDescriptor>(
        experimental::get_mock_cluster_desc_for_config(tt::ARCH::BLACKHOLE, 1).value());
    ContextId context_id_bh = MetalContext::create_instance(descriptor);
    EXPECT_EQ(context_id_bh, 2);
}

TEST_F(MetalContextTest, CreateSiliconInstanceWithMockInstances) {
    auto descriptor = std::make_shared<MetalliumObjectDescriptor>();
    ContextId context_id = MetalContext::create_instance(descriptor);
    EXPECT_EQ(context_id, SILICON_CONTEXT_ID);

    auto descriptor_wh = std::make_shared<MetalliumObjectDescriptor>(
        experimental::get_mock_cluster_desc_for_config(tt::ARCH::WORMHOLE_B0, 1).value());
    ContextId context_id_wh = MetalContext::create_instance(descriptor_wh);

    auto descriptor_bh = std::make_shared<MetalliumObjectDescriptor>(
        experimental::get_mock_cluster_desc_for_config(tt::ARCH::BLACKHOLE, 1).value());
    ContextId context_id_bh = MetalContext::create_instance(descriptor_bh);

    ASSERT_EQ(MetalContext::instance(context_id_wh).get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
    ASSERT_EQ(MetalContext::instance(context_id_bh).get_cluster().arch(), tt::ARCH::BLACKHOLE);
}

TEST_F(MetalContextTest, DestroyInstanceExplicit) {
    auto descriptor_wh = std::make_shared<MetalliumObjectDescriptor>(
        experimental::get_mock_cluster_desc_for_config(tt::ARCH::WORMHOLE_B0, 1).value());
    ContextId context_id_wh = MetalContext::create_instance(descriptor_wh);
    ASSERT_EQ(MetalContext::instance(context_id_wh).get_cluster().arch(), tt::ARCH::WORMHOLE_B0);
    MetalContext::destroy_instance(context_id_wh);
    EXPECT_THROW(MetalContext::instance(context_id_wh), std::runtime_error);
}

TEST_F(MetalContextTest, CreateImplicitAfterDestroy) {
    auto descriptor = std::make_shared<MetalliumObjectDescriptor>();
    ContextId context_id = MetalContext::create_instance(descriptor);
    MetalContext::destroy_instance(context_id);
    auto& context = MetalContext::instance(context_id);
    EXPECT_EQ(context.rtoptions().get_mock_enabled(), false);
}

TEST_F(MetalContextTest, ThreadIsolation) {
    ContextId mock_context_id{};
    ContextId silicon_context_id{};
    bool mock_ok{false};
    bool silicon_ok{false};

    std::thread mock_thread([&]() {
        auto descriptor = std::make_shared<MetalliumObjectDescriptor>(
            experimental::get_mock_cluster_desc_for_config(tt::ARCH::WORMHOLE_B0, 1).value());
        ContextId id = MetalContext::create_instance(descriptor);
        mock_context_id = id;
        mock_ok = MetalContext::instance(id).rtoptions().get_mock_enabled() &&
                  MetalContext::instance(id).get_cluster().arch() == tt::ARCH::WORMHOLE_B0;
        MetalContext::destroy_instance(id, false);
    });
    std::thread silicon_thread([&]() {
        auto descriptor = std::make_shared<MetalliumObjectDescriptor>();
        ContextId id = MetalContext::create_instance(descriptor);
        silicon_context_id = id;
        silicon_ok = (id == SILICON_CONTEXT_ID) && !MetalContext::instance(id).rtoptions().get_mock_enabled();
        MetalContext::destroy_instance(id, false);
    });

    mock_thread.join();
    silicon_thread.join();

    EXPECT_TRUE(mock_ok);
    EXPECT_TRUE(silicon_ok);
    EXPECT_EQ(silicon_context_id, SILICON_CONTEXT_ID);
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
        auto child_mock_descriptor = std::make_shared<MetalliumObjectDescriptor>(
            experimental::get_mock_cluster_desc_for_config(tt::ARCH::WORMHOLE_B0, 1).value());
        ContextId child_mock_id = MetalContext::create_instance(child_mock_descriptor);
        if (child_mock_id < 1 || !MetalContext::instance(child_mock_id).rtoptions().get_mock_enabled()) {
            _exit(1);
        }
        char byte = 0;
        if (read(pipe_fd[0], &byte, 1) != 1) {
            _exit(1);
        }
        close(pipe_fd[0]);
        _exit(0);
    }

    close(pipe_fd[0]);
    auto parent_mock_descriptor = std::make_shared<MetalliumObjectDescriptor>(
        experimental::get_mock_cluster_desc_for_config(tt::ARCH::WORMHOLE_B0, 1).value());
    ContextId parent_mock_id = MetalContext::create_instance(parent_mock_descriptor);
    ASSERT_GT(parent_mock_id, SILICON_CONTEXT_ID);
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

}  // namespace tt::tt_metal
