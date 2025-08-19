// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rpc_server_controller.hpp"
#include <tt-logger/tt-logger.hpp>
#include <chrono>
#include <thread>

namespace tt::tt_metal::inspector {

RpcServerController::RpcServerController(const std::string& host, uint16_t port)
    : host(host), port(port), should_stop(false) {
}

RpcServerController::~RpcServerController() {
    stop();
}

void RpcServerController::start() {
    if (server_thread.joinable()) {
        log_warning(tt::LogInspector, "Inspector RPC server already running");
        return;
    }

    should_stop = false;
    server_thread = std::thread(&RpcServerController::run_server, this);
    log_info(tt::LogInspector, "Inspector RPC server starting on {}:{}", host, port);
}

void RpcServerController::stop() {
    if (!server_thread.joinable()) {
        return;
    }

    should_stop = true;
    
    if (server_thread.joinable()) {
        server_thread.join();
    }
    rpc_server.reset();
    log_info(tt::LogInspector, "Inspector RPC server stopped");
}

void RpcServerController::run_server() {
    try {
        // Create the RPC implementation and set up callbacks
        auto rpc_impl = ::kj::heap<RpcServer>();
        rpc_server_implementation = rpc_impl.get();

        // Create and configure the RPC server
        rpc_server = std::make_unique<::capnp::EzRpcServer>(::kj::mv(rpc_impl), host, port);

        log_info(tt::LogInspector, "Inspector RPC server listening on {}:{}", host, port);

// TODO: Instead of probing, we should use conditional variables to wait for the server to stop.
        // Keep server running until stopped
        while (!should_stop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
    } catch (const std::exception& e) {
        log_error(tt::LogInspector, "Inspector RPC server error: {}", e.what());
    }
}

RpcServer& RpcServerController::get_rpc_server() {
    return *rpc_server_implementation;
}

} // namespace tt::tt_metal::inspector