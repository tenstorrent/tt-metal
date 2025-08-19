// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <capnp/ez-rpc.h>
#include <memory>
#include <thread>
#include <string>
#include "impl/debug/inspector/rpc_server_generated.hpp"

namespace tt::tt_metal::inspector {

class RpcServerController {
public:
    RpcServerController(const std::string& host, uint16_t port);
    ~RpcServerController();

    void start();
    void stop();
    bool is_running() const { return server_thread.joinable(); }

    RpcServer& get_rpc_server();

private:
    std::string host;
    uint16_t port;
    std::thread server_thread;
    std::unique_ptr<::capnp::EzRpcServer> rpc_server;
    RpcServer* rpc_server_implementation;
    bool should_stop;

    void run_server();
};

} // namespace tt::tt_metal::inspector