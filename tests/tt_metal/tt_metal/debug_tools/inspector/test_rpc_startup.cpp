// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <kj/async-io.h>
#include <capnp/rpc-twoparty.h>
#include <gtest/gtest.h>
#include <impl/debug/inspector/rpc_server_controller.hpp>
#include <string>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>


class InspectorFixture : public ::testing::Test {
  protected:
    // Helper to find a free port
    int find_free_port() {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            return 0;
        }
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

    std::string find_free_port_address() {
        int port = find_free_port();
        if (port == 0) {
            return "";
        }
        return "localhost:" + std::to_string(port);
    }

    void try_connect(const std::string& address) {
        auto io = ::kj::setupAsyncIo();
        auto& waitScope = io.waitScope;
        kj::Network& network = io.provider->getNetwork();
        kj::Own<kj::NetworkAddress> addr = network.parseAddress(address).wait(waitScope);
        kj::Own<kj::AsyncIoStream> conn = addr->connect().wait(waitScope);
        capnp::TwoPartyClient client(*conn);
        tt::tt_metal::inspector::rpc::Inspector::Client inspector = client.bootstrap().castAs<tt::tt_metal::inspector::rpc::Inspector>();
        auto request = inspector.getProgramsRequest();
        auto response = request.send().wait(waitScope);
    }

    void start(tt::tt_metal::inspector::RpcServerController& controller, const std::string& address) {
        controller.get_rpc_server().setGetProgramsCallback([](auto result) {
            result.initPrograms(0);
        });
        controller.start(address);
    }
};

TEST_F(InspectorFixture, WrongHostnameFailure) {
    tt::tt_metal::inspector::RpcServerController controller;
    EXPECT_ANY_THROW(start(controller, "265.265.265.265:12345"));
    EXPECT_ANY_THROW(start(controller, "how-is-this-possible.invalid-hostname.my-nonexisting.company.aiqwe:12345"));
}

TEST_F(InspectorFixture, UnixSocketStart) {
    // First remove any existing socket file
    std::filesystem::path path = std::filesystem::temp_directory_path() / "inspector_test_socket";
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }
    std::string address = "unix:" + path.string();
    tt::tt_metal::inspector::RpcServerController controller;
    EXPECT_NO_THROW(start(controller, address));
    EXPECT_NO_THROW(try_connect(address));
}

TEST_F(InspectorFixture, SecondServerDoesntStart) {
    tt::tt_metal::inspector::RpcServerController controller_success;
    std::string address = find_free_port_address();
    EXPECT_NO_THROW(start(controller_success, address));
    EXPECT_NO_THROW(try_connect(address));

    tt::tt_metal::inspector::RpcServerController controller_fail;
    EXPECT_ANY_THROW(start(controller_fail, address));
    EXPECT_NO_THROW(try_connect(address));
}

TEST_F(InspectorFixture, StartTwoServersOnDifferentPorts) {
    tt::tt_metal::inspector::RpcServerController controller1;
    auto address1 = find_free_port_address();
    EXPECT_NO_THROW(start(controller1, address1));
    EXPECT_NO_THROW(try_connect(address1));

    tt::tt_metal::inspector::RpcServerController controller2;
    auto address2 = find_free_port_address();
    EXPECT_NO_THROW(start(controller2, address2));
    EXPECT_NO_THROW(try_connect(address2));
}
