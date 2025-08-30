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
    ~RpcServerController();

    void start(const std::string& host, uint16_t port);
    void stop();
    bool is_running() const { return rpc_server != nullptr; }

    RpcServer& get_rpc_server();

private:
    std::thread server_thread;
    std::unique_ptr<::capnp::EzRpcServer> rpc_server;
    RpcServer* rpc_server_implementation;
    bool should_stop;

    // temp data used in background thread as initialization
    std::string host;
    uint16_t port;
    ::kj::Own<RpcServer> temp_rpc_server_implementation;

    void run_server();
};

} // namespace tt::tt_metal::inspector
