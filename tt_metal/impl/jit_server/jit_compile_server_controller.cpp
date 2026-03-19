// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/jit_server/jit_compile_server_controller.hpp"

#include <capnp/rpc-twoparty.h>
#include <kj/async-io.h>
#include <tt-logger/tt-logger.hpp>

#include <chrono>
#include <stdexcept>
#include <thread>
#include <utility>

namespace tt::tt_metal::jit_server {

JitCompileServerController::JitCompileServerController(JitCompileService::CompileCallback compile_callback) :
    jit_compile_service_(std::move(compile_callback)) {}

JitCompileServerController::~JitCompileServerController() { stop(); }

void JitCompileServerController::start(std::string address) {
    if (is_running_) {
        log_warning(tt::LogMetal, "JIT compile RPC server already running");
        return;
    }

    std::lock_guard<std::mutex> lock(start_stop_mutex_);
    if (is_running_) {
        log_warning(tt::LogMetal, "JIT compile RPC server already running");
        return;
    }

    address_ = std::move(address);
    should_stop_ = false;
    is_running_ = true;
    server_start_finished_ = false;
    server_thread_ = std::thread(&JitCompileServerController::run_server, this);

    std::unique_lock start_lock(server_start_mutex_);
    server_start_cv_.wait(start_lock, [this] { return this->server_start_finished_.load(); });
    if (!is_running_) {
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
        throw std::runtime_error(server_start_error_message_);
    }
}

void JitCompileServerController::stop() {
    if (!is_running_) {
        return;
    }

    std::lock_guard<std::mutex> lock(start_stop_mutex_);
    if (!is_running_) {
        return;
    }

    should_stop_ = true;
    if (server_thread_.joinable()) {
        server_thread_.join();
    }

    log_trace(tt::LogMetal, "JIT compile RPC server stopped");
    is_running_ = false;
}

void JitCompileServerController::run_server() {
    try {
        auto io = ::kj::setupAsyncIo();
        auto& wait_scope = io.waitScope;

        kj::Network& network = io.provider->getNetwork();
        kj::Own<kj::NetworkAddress> address = network.parseAddress(address_).wait(wait_scope);
        kj::Own<kj::ConnectionReceiver> listener = address->listen();

        capnp::TwoPartyServer server(::kj::Own<JitCompileService>(&jit_compile_service_, ::kj::NullDisposer::instance));
        auto listen_promise = server.listen(*listener);

        {
            std::lock_guard lock(server_start_mutex_);
            server_start_finished_ = true;
        }
        server_start_cv_.notify_one();

        auto last_events = std::chrono::high_resolution_clock::now();
        while (!should_stop_) {
            auto count = wait_scope.poll();
            listen_promise.poll(wait_scope);

            if (count > 0) {
                last_events = std::chrono::high_resolution_clock::now();
            } else if (std::chrono::high_resolution_clock::now() - last_events > std::chrono::milliseconds(10)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    } catch (const kj::Exception& e) {
        server_start_error_message_ = e.getDescription().cStr();
    } catch (const std::exception& e) {
        server_start_error_message_ = e.what();
    }
    is_running_ = false;

    {
        std::lock_guard lock(server_start_mutex_);
        server_start_finished_ = true;
    }
    server_start_cv_.notify_one();
}

}  // namespace tt::tt_metal::jit_server
