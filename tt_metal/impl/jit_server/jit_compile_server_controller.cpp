// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/jit_server/jit_compile_server_controller.hpp"

#include <capnp/rpc-twoparty.h>
#include <kj/async.h>
#include <kj/async-io.h>
#include <tt-logger/tt-logger.hpp>

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

    {
        std::lock_guard wake_lock(wake_mutex_);
        if (wake_) {
            wake_();
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

        // The loop blocks until there is work: socket I/O, or a compile finishing on a pool thread (its
        // cross-thread fulfill wakes the loop). stop() wakes it the same way.
        auto stop_paf = kj::newPromiseAndCrossThreadFulfiller<void>();
        {
            std::lock_guard lock(wake_mutex_);
            wake_ = [fulfiller = std::make_shared<kj::Own<kj::CrossThreadPromiseFulfiller<void>>>(
                         kj::mv(stop_paf.fulfiller))]() { (*fulfiller)->fulfill(); };
        }

        {
            std::lock_guard lock(server_start_mutex_);
            server_start_finished_ = true;
        }
        server_start_cv_.notify_one();

        listen_promise.exclusiveJoin(kj::mv(stop_paf.promise)).wait(wait_scope);
    } catch (const kj::Exception& e) {
        server_start_error_message_ = e.getDescription().cStr();
    } catch (const std::exception& e) {
        server_start_error_message_ = e.what();
    }
    {
        std::lock_guard lock(wake_mutex_);
        wake_ = nullptr;
    }
    is_running_ = false;

    {
        std::lock_guard lock(server_start_mutex_);
        server_start_finished_ = true;
    }
    server_start_cv_.notify_one();
}

}  // namespace tt::tt_metal::jit_server
