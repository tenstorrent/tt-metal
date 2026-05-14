// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <string>
#include <thread>

#include <capnp/rpc-twoparty.h>
#include <kj/async-io.h>
#include <tt-logger/tt-logger.hpp>

#include "impl/jit_server/jit_broker_service.hpp"

namespace {

std::atomic<bool> g_keep_running{true};
constexpr const char* kBrokerEndpointEnv = "TT_METAL_JIT_BROKER_ENDPOINT";
constexpr const char* kDefaultBrokerEndpoint = "localhost:9875";

void handle_signal(int /*signal*/) { g_keep_running.store(false); }

}  // namespace

int main() {
    const char* endpoint_env = std::getenv(kBrokerEndpointEnv);
    const std::string endpoint = endpoint_env != nullptr ? endpoint_env : kDefaultBrokerEndpoint;

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    tt::tt_metal::jit_server::JitBrokerService broker_service;
    auto io = kj::setupAsyncIo();
    kj::Network& network = io.provider->getNetwork();
    auto address = network.parseAddress(endpoint).wait(io.waitScope);
    auto listener = address->listen();
    capnp::TwoPartyServer server(
        kj::Own<tt::tt_metal::jit_server::JitBrokerService>(&broker_service, kj::NullDisposer::instance));
    auto listen_promise = server.listen(*listener);

    log_info(tt::LogMetal, "JIT dispatch broker listening on {}", endpoint);

    auto last_events = std::chrono::high_resolution_clock::now();
    while (g_keep_running.load()) {
        auto count = io.waitScope.poll();
        listen_promise.poll(io.waitScope);
        if (count > 0) {
            last_events = std::chrono::high_resolution_clock::now();
        } else if (std::chrono::high_resolution_clock::now() - last_events > std::chrono::milliseconds(10)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    return 0;
}
