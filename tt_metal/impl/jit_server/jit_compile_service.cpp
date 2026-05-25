// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/jit_server/jit_compile_service.hpp"

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <kj/async.h>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::jit_server {

namespace {

// Periodic metrics logging is opt-in via TT_METAL_JIT_SERVER_LOG_INTERVAL_MS.
// A positive integer enables periodic logging at that millisecond interval.
// Anything else -- unset, empty, "0", negative, or unparseable -- disables it.
constexpr char kJitServerLogIntervalEnv[] = "TT_METAL_JIT_SERVER_LOG_INTERVAL_MS";

std::chrono::milliseconds parse_log_interval_ms() {
    const char* interval_env = std::getenv(kJitServerLogIntervalEnv);
    if (interval_env == nullptr || *interval_env == '\0') {
        return std::chrono::milliseconds(0);
    }

    const std::string value(interval_env);
    try {
        const auto parsed_value = std::stoll(value);
        if (parsed_value <= 0) {
            return std::chrono::milliseconds(0);
        }
        return std::chrono::milliseconds(parsed_value);
    } catch (const std::exception&) {
        log_warning(
            tt::LogMetal,
            "Invalid value '{}' for {}. Periodic metrics logging is disabled.",
            value,
            kJitServerLogIntervalEnv);
        return std::chrono::milliseconds(0);
    }
}

void update_peak_inflight(std::atomic<std::uint64_t>& peak_inflight, std::uint64_t current_inflight) {
    auto peak = peak_inflight.load(std::memory_order_relaxed);
    while (peak < current_inflight &&
           !peak_inflight.compare_exchange_weak(peak, current_inflight, std::memory_order_relaxed)) {
    }
}

}  // namespace

JitCompileService::JitCompileService(CompileCallback compile_callback, UploadFirmwareCallback upload_fw_callback) :
    compile_callback_(std::move(compile_callback)),
    upload_fw_callback_(std::move(upload_fw_callback)),
    periodic_log_interval_(get_periodic_log_interval()) {
    if (periodic_log_interval_.count() > 0) {
        periodic_logger_thread_ = std::thread(&JitCompileService::run_periodic_logger_loop, this);
    }
}

JitCompileService::~JitCompileService() {
    {
        std::lock_guard<std::mutex> lock(periodic_logger_mutex_);
        should_stop_periodic_logger_ = true;
    }
    periodic_logger_cv_.notify_one();

    if (periodic_logger_thread_.joinable()) {
        periodic_logger_thread_.join();
    }

    if (total_compiles_.load(std::memory_order_relaxed) > 0) {
        log_metrics_summary();
    }
}

void JitCompileService::set_listen_address(std::string listen_address) {
    std::lock_guard<std::mutex> lock(listen_address_mutex_);
    if (listen_address.empty()) {
        listen_address_ = "unknown:0";
        listen_address_known_.store(false, std::memory_order_relaxed);
        return;
    }
    listen_address_ = std::move(listen_address);
    listen_address_known_.store(true, std::memory_order_relaxed);
}

std::string JitCompileService::make_dedup_key(const CompileRequest& request) const {
    return std::to_string(request.build_key) + ":" + request.kernel_name;
}

std::string JitCompileService::get_log_address() {
    if (!listen_address_known_.load(std::memory_order_relaxed)) {
        std::call_once(warned_missing_listen_address_, [&] {
            log_warning(tt::LogMetal, "JIT compile service listen address unavailable, using fallback id unknown:0");
        });
        return "unknown:0";
    }

    std::lock_guard<std::mutex> lock(listen_address_mutex_);
    return listen_address_;
}

JitCompileService::MetricsSnapshot JitCompileService::get_metrics_snapshot() const {
    MetricsSnapshot snapshot;
    snapshot.total_compiles = total_compiles_.load(std::memory_order_relaxed);
    snapshot.dedup_hits = dedup_hits_.load(std::memory_order_relaxed);
    snapshot.total_compile_time_ns = total_compile_time_ns_.load(std::memory_order_relaxed);
    snapshot.queued = queued_.load(std::memory_order_relaxed);
    snapshot.current_inflight = current_inflight_.load(std::memory_order_relaxed);
    snapshot.peak_inflight = peak_inflight_.load(std::memory_order_relaxed);
    snapshot.total_bytes_in = total_bytes_in_.load(std::memory_order_relaxed);
    snapshot.total_bytes_out = total_bytes_out_.load(std::memory_order_relaxed);
    return snapshot;
}

void JitCompileService::run_periodic_logger_loop() {
    std::unique_lock<std::mutex> lock(periodic_logger_mutex_);
    while (!should_stop_periodic_logger_) {
        if (periodic_logger_cv_.wait_for(
                lock, periodic_log_interval_, [this] { return should_stop_periodic_logger_; })) {
            break;
        }
        lock.unlock();
        log_metrics_summary();
        lock.lock();
    }
}

void JitCompileService::log_metrics_summary() {
    const auto snapshot = get_metrics_snapshot();
    const auto total_compile_time_ms = snapshot.total_compile_time_ns / 1000000;
    log_info(
        tt::LogMetal,
        "[jit_server addr={} ts={}] count={} dedup_hits={} total_compile_time_ms={} queued={} inflight={} "
        "peak_inflight={} bytes_in={} bytes_out={}",
        get_log_address(),
        current_time_ms_since_epoch(),
        snapshot.total_compiles,
        snapshot.dedup_hits,
        total_compile_time_ms,
        snapshot.queued,
        snapshot.current_inflight,
        snapshot.peak_inflight,
        snapshot.total_bytes_in,
        snapshot.total_bytes_out);
}

std::uint64_t JitCompileService::estimate_compile_request_bytes_in(const CompileRequest& request) const {
    // Approximate request payload bytes as the sum of:
    //   * shared toolchain string bytes (`gpp`, `kernel_name`)
    //   * all per-target string fields (`target_name`, flags, includes, linker fields, and list string entries)
    //   * generated file names and generated file content bytes
    // This intentionally excludes scalar fields and container overhead.
    std::uint64_t bytes = request.gpp.size() + request.kernel_name.size();

    for (const auto& target : request.targets) {
        bytes += target.target_name.size();
        bytes += target.cflags.size();
        bytes += target.includes.size();
        bytes += target.compiler_opt_level.size();
        bytes += target.lflags.size();
        bytes += target.extra_link_objs.size();
        bytes += target.linker_script.size();
        bytes += target.weakened_firmware_name.size();
        bytes += target.linker_opt_level.size();

        for (const auto& define : target.defines) {
            bytes += define.size();
        }
        for (const auto& src : target.srcs) {
            bytes += src.size();
        }
        for (const auto& obj : target.objs) {
            bytes += obj.size();
        }
    }

    for (const auto& generated_file : request.generated_files) {
        bytes += generated_file.name.size();
        bytes += generated_file.content.size();
    }

    return bytes;
}

std::uint64_t JitCompileService::calculate_compile_response_bytes_out(const CompileResponse& response) const {
    std::uint64_t bytes = 0;
    for (const auto& elf_blob : response.elf_blobs) {
        bytes += elf_blob.data.size();
    }
    return bytes;
}

std::uint64_t JitCompileService::current_time_ms_since_epoch() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count());
}

std::chrono::milliseconds JitCompileService::get_periodic_log_interval() { return parse_log_interval_ms(); }

kj::Promise<void> JitCompileService::compile(CompileContext context) {
    CompileRequest request;
    auto reader = context.getParams().getRequest();
    request.build_key = reader.getBuildKey();
    request.kernel_name = reader.getKernelName().cStr();
    request.gpp = reader.getGpp().cStr();
    for (auto target : reader.getTargets()) {
        TargetRecipe t;
        t.target_name = target.getTargetName().cStr();
        t.cflags = target.getCflags().cStr();
        for (auto define : target.getDefines()) {
            t.defines.push_back(define.cStr());
        }
        t.includes = target.getIncludes().cStr();
        t.compiler_opt_level = target.getCompilerOptLevel().cStr();
        for (auto src : target.getSrcs()) {
            t.srcs.push_back(src.cStr());
        }
        for (auto obj : target.getObjs()) {
            t.objs.push_back(obj.cStr());
        }
        t.lflags = target.getLflags().cStr();
        t.extra_link_objs = target.getExtraLinkObjs().cStr();
        t.linker_script = target.getLinkerScript().cStr();
        t.weakened_firmware_name = target.getWeakenedFirmwareName().cStr();
        t.firmware_is_kernel_object = target.getFirmwareIsKernelObject();
        t.linker_opt_level = target.getLinkerOptLevel().cStr();
        request.targets.push_back(std::move(t));
    }
    for (auto file : reader.getGeneratedFiles()) {
        GeneratedFile gf;
        gf.name = file.getName().cStr();
        auto content = file.getContent();
        gf.content.assign(content.begin(), content.end());
        request.generated_files.push_back(std::move(gf));
    }

    total_bytes_in_.fetch_add(estimate_compile_request_bytes_in(request), std::memory_order_relaxed);

    const std::string dedup_key = make_dedup_key(request);

    auto paf = kj::newPromiseAndCrossThreadFulfiller<CompileResponse>();
    auto fulfiller = std::make_shared<kj::Own<kj::CrossThreadPromiseFulfiller<CompileResponse>>>(kj::mv(paf.fulfiller));

    queued_.fetch_add(1, std::memory_order_relaxed);
    try {
        thread_pool_.silent_async([this, dedup_key, request = std::move(request), fulfiller]() mutable {
            queued_.fetch_sub(1, std::memory_order_relaxed);

            const auto inflight_now = current_inflight_.fetch_add(1, std::memory_order_relaxed) + 1;
            update_peak_inflight(peak_inflight_, inflight_now);

            bool was_dedup_owner = false;
            CompileResponse response;
            try {
                response = compile_deduper_.run(dedup_key, [&]() {
                    was_dedup_owner = true;
                    if (!compile_callback_) {
                        throw std::runtime_error("No JIT compile callback configured on server");
                    }

                    const auto compile_start = std::chrono::steady_clock::now();
                    auto record_compile_time = [this, compile_start]() {
                        const auto compile_time_ns =
                            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                           std::chrono::steady_clock::now() - compile_start)
                                                           .count());
                        total_compile_time_ns_.fetch_add(compile_time_ns, std::memory_order_relaxed);
                    };

                    try {
                        auto compile_response = compile_callback_(request);
                        record_compile_time();
                        return compile_response;
                    } catch (...) {
                        record_compile_time();
                        throw;
                    }
                });
            } catch (const std::exception& e) {
                response.success = false;
                response.error_message = e.what();
            }

            if (!was_dedup_owner) {
                dedup_hits_.fetch_add(1, std::memory_order_relaxed);
            }
            total_compiles_.fetch_add(1, std::memory_order_relaxed);
            current_inflight_.fetch_sub(1, std::memory_order_relaxed);
            total_bytes_out_.fetch_add(calculate_compile_response_bytes_out(response), std::memory_order_relaxed);
            (*fulfiller)->fulfill(kj::mv(response));
        });
    } catch (...) {
        queued_.fetch_sub(1, std::memory_order_relaxed);
        throw;
    }

    return paf.promise.then([context](CompileResponse response) mutable {
        if (!response.success) {
            log_warning(tt::LogMetal, "compile FAIL: {}", response.error_message);
        }

        auto response_builder = context.getResults().initResponse();
        response_builder.setSuccess(response.success);
        response_builder.setErrorMessage(response.error_message);
        auto blobs_builder = response_builder.initElfBlobs(response.elf_blobs.size());
        for (std::size_t i = 0; i < response.elf_blobs.size(); ++i) {
            blobs_builder[i].setName(response.elf_blobs[i].name);
            blobs_builder[i].setData(
                kj::arrayPtr(response.elf_blobs[i].data.data(), response.elf_blobs[i].data.size()));
        }
    });
}

kj::Promise<void> JitCompileService::uploadFirmware(UploadFirmwareContext context) {
    UploadFirmwareRequest request;
    auto reader = context.getParams().getRequest();
    request.build_key = reader.getBuildKey();
    for (auto artifact : reader.getArtifacts()) {
        FirmwareArtifact a;
        a.target_name = artifact.getTargetName().cStr();
        a.file_name = artifact.getFileName().cStr();
        a.is_kernel_object = artifact.getIsKernelObject();
        auto data = artifact.getData();
        a.data.assign(data.begin(), data.end());
        request.artifacts.push_back(std::move(a));
    }

    auto paf = kj::newPromiseAndCrossThreadFulfiller<UploadFirmwareResponse>();
    auto fulfiller =
        std::make_shared<kj::Own<kj::CrossThreadPromiseFulfiller<UploadFirmwareResponse>>>(kj::mv(paf.fulfiller));

    thread_pool_.silent_async([this, request = std::move(request), fulfiller]() mutable {
        UploadFirmwareResponse response;
        try {
            if (!upload_fw_callback_) {
                throw std::runtime_error("No firmware upload callback configured on server");
            }
            response = upload_fw_callback_(request);
        } catch (const std::exception& e) {
            response.success = false;
            response.error_message = e.what();
        }
        (*fulfiller)->fulfill(kj::mv(response));
    });

    return paf.promise.then([context](UploadFirmwareResponse response) mutable {
        if (!response.success) {
            log_warning(tt::LogMetal, "uploadFirmware FAIL: {}", response.error_message);
        }
        auto response_builder = context.getResults().initResponse();
        response_builder.setSuccess(response.success);
        response_builder.setErrorMessage(response.error_message);
    });
}

}  // namespace tt::tt_metal::jit_server
