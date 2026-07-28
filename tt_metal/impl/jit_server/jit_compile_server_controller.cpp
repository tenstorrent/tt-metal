// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/jit_server/jit_compile_server_controller.hpp"

#include <capnp/rpc-twoparty.h>
#include <kj/async.h>
#include <kj/async-io.h>
#include <tt-logger/tt-logger.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

namespace tt::tt_metal::jit_server {

JitCompileServerController::JitCompileServerController(
    JitCompileService::CompileCallback compile_callback, JitCompileService::UploadFirmwareCallback upload_fw_callback) :
    jit_compile_service_(std::move(compile_callback), std::move(upload_fw_callback)) {}

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
    jit_compile_service_.set_listen_address(address_);
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
    // Wake the blocking event loop via the cross-thread shutdown fulfiller so wait() returns.
    {
        std::lock_guard lock(shutdown_mutex_);
        if (shutdown_signal_) {
            shutdown_signal_();
        }
    }
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

        // Idiomatic blocking event loop: block in epoll via wait(), woken by both socket I/O
        // (new connections, request reads, response writes) AND cross-thread compile-completion
        // fulfillments. This replaces the former non-blocking poll()-spin + 1ms sleep, which was
        // the single pump for all connection I/O and every cross-thread fulfill(); under high
        // concurrency that fragile single-pump could fail to deliver a response, wedging the
        // client forever. Shutdown is a cross-thread promise fulfilled by stop().
        auto shutdown_paf = kj::newPromiseAndCrossThreadFulfiller<void>();
        auto shutdown_fulfiller =
            std::make_shared<kj::Own<kj::CrossThreadPromiseFulfiller<void>>>(kj::mv(shutdown_paf.fulfiller));
        {
            std::lock_guard lock(shutdown_mutex_);
            shutdown_signal_ = [shutdown_fulfiller]() { (*shutdown_fulfiller)->fulfill(); };
        }

        {
            std::lock_guard lock(server_start_mutex_);
            server_start_finished_ = true;
        }
        server_start_cv_.notify_one();

        // Returns when stop() fulfills the shutdown promise, or if the listener fails.
        listen_promise.exclusiveJoin(kj::mv(shutdown_paf.promise)).wait(wait_scope);

        {
            std::lock_guard lock(shutdown_mutex_);
            shutdown_signal_ = nullptr;
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
