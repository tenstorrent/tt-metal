// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <impl/debug/inspector/rpc_server_controller.hpp>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>


class InspectorFixture : public ::testing::Test {
  protected:
    // Helper to find a free port
    int find_free_port() {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) return 0;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = 0; // Let OS pick
        if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(sock);
            return 0;
        }
        socklen_t len = sizeof(addr);
        if (getsockname(sock, (struct sockaddr*)&addr, &len) == -1) {
            close(sock);
            return 0;
        }
        int port = ntohs(addr.sin_port);
        close(sock);
        return port;
    }
};

TEST_F(InspectorFixture, WrongHostnameFailure) {
    tt::tt_metal::inspector::RpcServerController controller;
    EXPECT_ANY_THROW(controller.start("265.265.265.265", 12345));
    EXPECT_ANY_THROW(controller.start("how-is-this-possible.invalid-hostname.my-nonexisting.company.aiqwe", 12345));
}

TEST_F(InspectorFixture, SecondServerDoesntStart) {
    tt::tt_metal::inspector::RpcServerController controller_success;
    int free_port = find_free_port();
    EXPECT_NO_THROW(controller_success.start("localhost", free_port));

    tt::tt_metal::inspector::RpcServerController controller_fail;
    EXPECT_ANY_THROW(controller_fail.start("localhost", free_port));
}

TEST_F(InspectorFixture, StartTwoServersOnDifferentPorts) {
    tt::tt_metal::inspector::RpcServerController controller1;
    EXPECT_NO_THROW(controller1.start("localhost", find_free_port()));

    tt::tt_metal::inspector::RpcServerController controller2;
    EXPECT_NO_THROW(controller2.start("localhost", find_free_port()));
}
