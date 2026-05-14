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
#include <vector>

#include <capnp/rpc-twoparty.h>
#include <kj/async-io.h>

#include "impl/jit_server/jit_broker_rpc_client.hpp"
#include "impl/jit_server/jit_broker_service.hpp"

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

std::string next_broker_endpoint() {
    static std::atomic<int> port{19075};
    return "localhost:" + std::to_string(port.fetch_add(1));
}

}  // namespace

TEST(JitBrokerRoutingTest, AffinityPendingAndFirmwareClaims) {
    ScopedBroker broker(next_broker_endpoint());
    jit_server::JitBrokerRpcSession client(broker.endpoint());

    client.register_server("server-a:1111");
    client.register_server("server-b:2222");
    client.report_cache_state("server-a:1111", {}, {});
    client.report_cache_state("server-b:2222", {}, {});

    jit_server::BrokerAssignRequest req;
    req.build_key = 7;
    req.kernel_keys = {"kernel_x/123"};
    auto first = client.assign(req);
    ASSERT_EQ(first.assignments.size(), 1);
    const auto chosen_server = first.assignments[0].server_endpoint;

    auto second = client.assign(req);
    ASSERT_EQ(second.assignments.size(), 1);
    EXPECT_EQ(second.assignments[0].server_endpoint, chosen_server);

    client.release(
        first.assignments[0].handle,
        jit_server::KernelKey{req.build_key, req.kernel_keys[0]},
        /*was_real_compile=*/true);
    client.release(
        second.assignments[0].handle,
        jit_server::KernelKey{req.build_key, req.kernel_keys[0]},
        /*was_real_compile=*/false);

    auto third = client.assign(req);
    ASSERT_EQ(third.assignments.size(), 1);
    EXPECT_EQ(third.assignments[0].server_endpoint, chosen_server);

    const auto upload1 = client.claim_firmware_upload(7, chosen_server);
    EXPECT_EQ(upload1, jit_server::FirmwareUploadAction::YOU_UPLOAD);
    const auto upload2 = client.claim_firmware_upload(7, chosen_server);
    EXPECT_EQ(upload2, jit_server::FirmwareUploadAction::WAIT_FOR_OTHER);
    client.release_firmware_upload(7, chosen_server, true);
    const auto upload3 = client.claim_firmware_upload(7, chosen_server);
    EXPECT_EQ(upload3, jit_server::FirmwareUploadAction::SKIP_ALREADY_PRESENT);
}

}  // namespace tt::tt_metal
