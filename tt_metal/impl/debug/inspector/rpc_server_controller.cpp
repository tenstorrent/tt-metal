// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rpc_server_controller.hpp"
#include <kj/async.h>
#include <tt-logger/tt-logger.hpp>
#include <chrono>
#include <thread>

namespace tt::tt_metal::inspector {

RpcServerController::~RpcServerController() {
    stop();
}

void RpcServerController::start(const std::string& host, uint16_t port) {
    if (is_running()) {
        log_warning(tt::LogInspector, "Inspector RPC server already running");
        return;
    }

    // Create the RPC implementation and set up callbacks
    this->host = host;
    this->port = port;
    temp_rpc_server_implementation = ::kj::heap<RpcServer>();
    rpc_server_implementation = temp_rpc_server_implementation.get();

    should_stop = false;
    server_thread = std::thread(&RpcServerController::run_server, this);
}

void RpcServerController::stop() {
    if (!is_running()) {
        return;
    }

    should_stop = true;

    if (server_thread.joinable()) {
        server_thread.join();
    }

    log_info(tt::LogInspector, "Inspector RPC server stopped");
}

void RpcServerController::run_server() {
    try {
        // Create and configure the RPC server
        rpc_server = std::make_unique<::capnp::EzRpcServer>(::kj::mv(temp_rpc_server_implementation), host, port);
        log_info(tt::LogInspector, "Inspector RPC server listening on {}:{}", host, port);

        // Keep server running until stopped
        auto& waitScope = rpc_server->getWaitScope();
        auto last_events = std::chrono::high_resolution_clock::now();

        while (!should_stop) {
            auto count = waitScope.poll();
            if (count > 0) {
                last_events = std::chrono::high_resolution_clock::now();
            }
            else if (std::chrono::high_resolution_clock::now() - last_events > std::chrono::milliseconds(10)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    } catch (const std::exception& e) {
        log_error(tt::LogInspector, "Inspector RPC server error: {}", e.what());
    }
    rpc_server.reset();
}

RpcServer& RpcServerController::get_rpc_server() {
    return *rpc_server_implementation;
}

} // namespace tt::tt_metal::inspector
