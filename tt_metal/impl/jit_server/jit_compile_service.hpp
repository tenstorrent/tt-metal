// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include <taskflow/taskflow.hpp>

#include "common/host_threading.hpp"
#include "impl/jit_server/in_flight_compile_deduper.hpp"
#include "impl/jit_server/rpc.capnp.h"
#include "impl/jit_server/types.hpp"

namespace tt::tt_metal::jit_server {

class JitCompileService final : public rpc::JitCompile::Server {
public:
    using CompileCallback = std::function<CompileResponse(const CompileRequest&)>;
    using UploadFirmwareCallback = std::function<UploadFirmwareResponse(const UploadFirmwareRequest&)>;

    explicit JitCompileService(CompileCallback compile_callback, UploadFirmwareCallback upload_fw_callback = {});
    ~JitCompileService();

    kj::Promise<void> compile(CompileContext context) override;
    kj::Promise<void> uploadFirmware(UploadFirmwareContext context) override;

    void set_listen_address(std::string listen_address);

private:
    struct MetricsSnapshot {
        std::uint64_t total_compiles = 0;
        std::uint64_t dedup_hits = 0;
        std::uint64_t total_compile_time_ns = 0;
        std::uint64_t queued = 0;
        std::uint64_t current_inflight = 0;
        std::uint64_t peak_inflight = 0;
        std::uint64_t total_bytes_in = 0;
        std::uint64_t total_bytes_out = 0;
    };

    std::string get_log_address();
    MetricsSnapshot get_metrics_snapshot() const;
    void run_periodic_logger_loop();
    void log_metrics_summary();

    std::uint64_t estimate_compile_request_bytes_in(const CompileRequest& request) const;
    std::uint64_t calculate_compile_response_bytes_out(const CompileResponse& response) const;

    static std::uint64_t current_time_ms_since_epoch();
    static std::chrono::milliseconds get_periodic_log_interval();

    std::string make_dedup_key(const CompileRequest& request) const;

    CompileCallback compile_callback_;
    UploadFirmwareCallback upload_fw_callback_;
    InFlightCompileDeduper<CompileResponse> compile_deduper_;
    tf::Executor thread_pool_{tt::tt_metal::detail::get_host_worker_threads()};

    // total_compiles_, queued_, current_inflight_, peak_inflight_, total_bytes_in_, total_bytes_out_, and
    // dedup_hits_ all count incoming requests (dedup hits included). total_compile_time_ns_ measures only
    // unique compile work, so it can be ratioed against (total_compiles_ - dedup_hits_) for per-compile time.
    std::atomic<std::uint64_t> total_compiles_{0};
    std::atomic<std::uint64_t> dedup_hits_{0};
    std::atomic<std::uint64_t> total_compile_time_ns_{0};
    std::atomic<std::uint64_t> queued_{0};
    std::atomic<std::uint64_t> current_inflight_{0};
    std::atomic<std::uint64_t> peak_inflight_{0};
    std::atomic<std::uint64_t> total_bytes_in_{0};
    std::atomic<std::uint64_t> total_bytes_out_{0};

    std::mutex listen_address_mutex_;
    std::string listen_address_ = "unknown:0";
    std::atomic<bool> listen_address_known_{false};
    std::once_flag warned_missing_listen_address_;

    const std::chrono::milliseconds periodic_log_interval_;
    std::mutex periodic_logger_mutex_;
    std::condition_variable periodic_logger_cv_;
    bool should_stop_periodic_logger_ = false;
    std::thread periodic_logger_thread_;
};

}  // namespace tt::tt_metal::jit_server
