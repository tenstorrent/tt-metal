// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#include <capnp/rpc-twoparty.h>
#include <kj/async-io.h>

#include "impl/jit_server/jit_broker_rpc_client.hpp"
#include "impl/jit_server/jit_broker_service.hpp"
#include "impl/jit_server/jit_compile_server_controller.hpp"
#include "impl/jit_server/remote_compile_coordinator.hpp"

namespace tt::tt_metal {
namespace {

class ScopedBroker {
public:
    explicit ScopedBroker(std::string endpoint) : endpoint_(std::move(endpoint)) {
        worker_ = std::thread([this] { this->run(); });
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return started_; });
        if (!start_error_.empty()) {
            throw std::runtime_error(start_error_);
        }
    }

    ~ScopedBroker() {
        stop_.store(true);
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    const std::string& endpoint() const { return endpoint_; }

private:
    void run() {
        try {
            auto io = kj::setupAsyncIo();
            kj::Network& network = io.provider->getNetwork();
            auto address = network.parseAddress(endpoint_).wait(io.waitScope);
            auto listener = address->listen();
            capnp::TwoPartyServer server(kj::Own<jit_server::JitBrokerService>(&service_, kj::NullDisposer::instance));
            auto listen_promise = server.listen(*listener);
            {
                std::lock_guard lock(mutex_);
                started_ = true;
            }
            cv_.notify_all();

            while (!stop_.load()) {
                io.waitScope.poll();
                listen_promise.poll(io.waitScope);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } catch (const std::exception& e) {
            std::lock_guard lock(mutex_);
            start_error_ = e.what();
            started_ = true;
            cv_.notify_all();
        }
    }

    std::string endpoint_;
    jit_server::JitBrokerService service_;
    std::atomic<bool> stop_{false};
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool started_ = false;
    std::string start_error_;
};

std::string next_endpoint(int base_port) {
    static std::atomic<int> port_offset{0};
    return "localhost:" + std::to_string(base_port + port_offset.fetch_add(1));
}

}  // namespace

TEST(JitBrokerEndToEndTest, CoordinatorDispatchesAndServersReleaseToBroker) {
    ScopedBroker broker(next_endpoint(20000));
    jit_server::JitBrokerRpcSession broker_session(broker.endpoint());

    std::mutex mutex;
    std::unordered_map<std::string, int> compile_count_by_server;
    std::unordered_map<std::string, int> firmware_uploads_by_server;
    std::unordered_map<std::string, std::string> kernel_to_server;

    auto start_mock_server = [&](const std::string& endpoint) {
        auto compile_callback = [&,
                                 endpoint](const jit_server::CompileRequest& request) -> jit_server::CompileResponse {
            {
                std::lock_guard lock(mutex);
                compile_count_by_server[endpoint] += 1;
                kernel_to_server[request.kernel_name] = endpoint;
            }
            jit_server::JitBrokerRpcClient(broker.endpoint())
                .release(
                    request.handle,
                    jit_server::KernelKey{request.build_key, request.kernel_name},
                    /*was_real_compile=*/true);
            jit_server::CompileResponse response;
            response.success = true;
            return response;
        };

        auto upload_callback =
            [&, endpoint](const jit_server::UploadFirmwareRequest&) -> jit_server::UploadFirmwareResponse {
            std::lock_guard lock(mutex);
            firmware_uploads_by_server[endpoint] += 1;
            jit_server::UploadFirmwareResponse response;
            response.success = true;
            return response;
        };

        auto controller = std::make_unique<jit_server::JitCompileServerController>(
            std::move(compile_callback), std::move(upload_callback));
        controller->start(endpoint);
        broker_session.register_server(endpoint);
        broker_session.report_cache_state(endpoint, {}, {42});
        return controller;
    };

    const std::string server_a = next_endpoint(21000);
    const std::string server_b = next_endpoint(22000);
    auto controller_a = start_mock_server(server_a);
    auto controller_b = start_mock_server(server_b);

    RemoteCompileCoordinator coordinator(broker.endpoint(), /*device_build_id=*/0, /*build_key=*/42);
    coordinator.submit(101, [] {
        KernelCompileDescriptor desc;
        desc.kernel_hash = 101;
        desc.request.build_key = 42;
        desc.request.kernel_name = "kernel_a/101";
        return desc;
    });
    coordinator.submit(102, [] {
        KernelCompileDescriptor desc;
        desc.kernel_hash = 102;
        desc.request.build_key = 42;
        desc.request.kernel_name = "kernel_b/102";
        return desc;
    });
    coordinator.finish();

    controller_a->stop();
    controller_b->stop();

    {
        std::lock_guard lock(mutex);
        EXPECT_EQ(compile_count_by_server[server_a] + compile_count_by_server[server_b], 2);
        EXPECT_EQ(kernel_to_server.size(), 2);
        EXPECT_EQ(firmware_uploads_by_server[server_a], 0);
        EXPECT_EQ(firmware_uploads_by_server[server_b], 0);
    }
}

}  // namespace tt::tt_metal
