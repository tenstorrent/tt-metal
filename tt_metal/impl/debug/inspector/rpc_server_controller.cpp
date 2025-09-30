// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    if (is_running) {
        log_warning(tt::LogInspector, "Inspector RPC server already running");
        return;
    }

    std::lock_guard<std::mutex> lock(start_stop_mutex);

    if (is_running) {
        log_warning(tt::LogInspector, "Inspector RPC server already running");
        return;
    }

    // Create the RPC implementation and set up callbacks
    this->host = host;
    this->port = port;

    should_stop = false;
    is_running = true;
    server_start_finished = false;
    server_thread = std::thread(&RpcServerController::run_server, this);

    // Wait for server to start or fail
    std::unique_lock start_lock(server_start_mutex);
    server_start_cv.wait(start_lock, [this] { return this->server_start_finished.load(); });
    if (!is_running) {
        if (server_thread.joinable()) {
            server_thread.join();
        }
        throw std::runtime_error(server_start_error_message);
    }
}

void RpcServerController::stop() {
    if (!is_running) {
        return;
    }

    std::lock_guard<std::mutex> lock(start_stop_mutex);

    if (!is_running) {
        return;
    }

    should_stop = true;

    if (server_thread.joinable()) {
        server_thread.join();
    }

    log_info(tt::LogInspector, "Inspector RPC server stopped");
    is_running = false;
}

void RpcServerController::run_server() {
    try {
        // Create and configure the RPC server
        ::capnp::EzRpcServer rpc_server(::kj::Own<RpcServer>(&rpc_server_implementation, ::kj::NullDisposer::instance), host, port);

        // Check if server is started correctly
        auto& waitScope = rpc_server.getWaitScope();
        rpc_server.getPort().wait(waitScope);

        // Signal back to RpcServerController::start that the server is ready to accept connections
        {
            std::lock_guard lock(server_start_mutex);
            server_start_finished = true;
        }
        server_start_cv.notify_one();
        log_info(tt::LogInspector, "Inspector RPC server listening on {}:{}", host, port);

        // Keep server running until stopped
        auto last_events = std::chrono::high_resolution_clock::now();

        while (!should_stop) {
            auto count = waitScope.poll();
            // If external client is querying, avoid sleeping too much
            if (count > 0) {
                last_events = std::chrono::high_resolution_clock::now();
            }
            // If no events for a while, sleep a bit to reduce CPU usage
            else if (std::chrono::high_resolution_clock::now() - last_events > std::chrono::milliseconds(10)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    } catch (const kj::Exception& e) {
        server_start_error_message = e.getDescription().cStr();
    } catch (const std::exception& e) {
        server_start_error_message = e.what();
    }
    is_running = false;

    // In case of failure, signal back to RpcServerController::start that the server start failed
    {
        std::lock_guard lock(server_start_mutex);
        server_start_finished = true;
    }
    server_start_cv.notify_one();
}

RpcServer& RpcServerController::get_rpc_server() {
    return rpc_server_implementation;
}

} // namespace tt::tt_metal::inspector
