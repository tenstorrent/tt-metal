// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <capnp/ez-rpc.h>
#include <condition_variable>
#include <memory>
#include <thread>
#include <string>
#include "impl/debug/inspector/rpc_server_generated.hpp"

namespace tt::tt_metal::inspector {

class RpcServerController {
public:
    ~RpcServerController();

    void start(std::string address);
    void stop();

    RpcServer& get_rpc_server();

private:
    std::thread server_thread;
    RpcServer rpc_server_implementation;
    std::mutex start_stop_mutex;
    std::atomic<bool> should_stop{false};
    std::atomic<bool> is_running{false};

    // Used to signal when the server has started
    std::condition_variable server_start_cv;
    std::mutex server_start_mutex;
    std::atomic<bool> server_start_finished{false};
    std::string server_start_error_message{};

    // temp data used in background thread as initialization
    std::string address{};

    void run_server();
};

} // namespace tt::tt_metal::inspector
