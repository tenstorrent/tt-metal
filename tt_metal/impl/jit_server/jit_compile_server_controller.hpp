// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "impl/jit_server/jit_compile_service.hpp"

namespace tt::tt_metal::jit_server {

class JitCompileServerController {
public:
    explicit JitCompileServerController(
        JitCompileService::CompileCallback compile_callback,
        JitCompileService::UploadFirmwareCallback upload_fw_callback = {});
    ~JitCompileServerController();

    void start(std::string address);
    void stop();

private:
    void run_server();

    std::thread server_thread_;
    JitCompileService jit_compile_service_;
    std::mutex start_stop_mutex_;
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> is_running_{false};

    std::condition_variable server_start_cv_;
    std::mutex server_start_mutex_;
    std::atomic<bool> server_start_finished_{false};
    std::string server_start_error_message_;

    std::string address_;
};

}  // namespace tt::tt_metal::jit_server
