// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "build_cache_telemetry.hpp"

#include <atomic>
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>

#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal {

// --- TelemetryToken ---

TelemetryToken::TelemetryToken(std::string name) : name_(std::move(name)) {}

void TelemetryToken::set_recording_enabled(bool enabled) {
    recording_enabled_.store(enabled, std::memory_order_release);
}

void TelemetryToken::record(double value) {
    if (!recording_enabled_.load(std::memory_order_relaxed)) {
        return;
    }
    std::lock_guard lk(data_mutex_);
    ++data_.count;
    data_.total += value;
    data_.min_val = std::min(data_.min_val, value);
    data_.max_val = std::max(data_.max_val, value);
}

TelemetryTokenData TelemetryToken::snapshot() const {
    std::lock_guard lk(data_mutex_);
    return data_;
}

// --- BuildCacheTelemetry ---

struct BuildCacheTelemetryImpl {
    std::atomic<uint64_t> srcs_and_compiled{0};
    std::atomic<uint32_t> cached_hit_count{0};
    std::atomic<uint32_t> merged_artifacts{0};
    std::atomic<uint32_t> merged_genfiles{0};
    std::atomic<uint32_t> jit_once_dedup_count{0};

    std::mutex token_registry_mutex;
    std::vector<TelemetryToken*> registered_tokens;
};

BuildCacheTelemetry::BuildCacheTelemetry() { enable(); }

BuildCacheTelemetry::~BuildCacheTelemetry() {
    // Dump metrics here rather than via std::atexit. The atexit handler is registered inside
    // the singleton constructor, so glibc queues it *before* the constructor completes.
    // Because atexit/static-dtor ordering is LIFO over a shared queue, the singleton destructor
    // would fire *before* the atexit handler, leaving impl_ already reset when dump_metrics()
    // ran — causing a silent no-op.
    //
    // The destructor body is the correct place: impl_ is still valid here (member destructors
    // run after the body), so dump_metrics() can read all counters and registered tokens.
    //
    // LoggerRegistry::instance() is a leaky singleton (allocated with `new`, never freed),
    // so its loggers and sinks are alive for the entire process lifetime — including here.
    dump_metrics();
}

BuildCacheTelemetry& BuildCacheTelemetry::inst() {
    static BuildCacheTelemetry instance;
    return instance;
}

void BuildCacheTelemetry::enable() {
    if (impl_) {
        return;
    }
    impl_ = std::make_unique<BuildCacheTelemetryImpl>();
    std::lock_guard owned_lk(owned_tokens_mutex_);
    for (auto& token : owned_tokens_) {
        token->set_recording_enabled(true);
    }
    std::lock_guard reg_lk(impl_->token_registry_mutex);
    for (auto& token : owned_tokens_) {
        impl_->registered_tokens.push_back(token.get());
    }
}

void BuildCacheTelemetry::disable() {
    {
        std::lock_guard owned_lk(owned_tokens_mutex_);
        for (auto& token : owned_tokens_) {
            token->set_recording_enabled(false);
        }
    }
    impl_.reset();
}

void BuildCacheTelemetry::record_compile(uint32_t num_srcs, uint32_t num_compiled) {
    if (!impl_) {
        return;
    }
    impl_->srcs_and_compiled.fetch_add(
        (static_cast<uint64_t>(num_srcs) << 32) | static_cast<uint64_t>(num_compiled), std::memory_order_release);
}

void BuildCacheTelemetry::record_cache_hit() {
    if (!impl_) {
        return;
    }
    impl_->cached_hit_count.fetch_add(1, std::memory_order_release);
}

void BuildCacheTelemetry::record_merge(uint32_t count) {
    if (!impl_) {
        return;
    }
    impl_->merged_artifacts.fetch_add(count, std::memory_order_release);
}

void BuildCacheTelemetry::record_genfile_merge(uint32_t count) {
    if (!impl_) {
        return;
    }
    impl_->merged_genfiles.fetch_add(count, std::memory_order_release);
}

void BuildCacheTelemetry::record_jit_once_dedup() {
    if (!impl_) {
        return;
    }
    impl_->jit_once_dedup_count.fetch_add(1, std::memory_order_release);
}

uint32_t BuildCacheTelemetry::get_srcs_count() const {
    if (!impl_) {
        return 0;
    }
    return impl_->srcs_and_compiled.load(std::memory_order_acquire) >> 32;
}

uint32_t BuildCacheTelemetry::get_compile_count() const {
    if (!impl_) {
        return 0;
    }
    return impl_->srcs_and_compiled.load(std::memory_order_acquire) & 0xFFFF'FFFFu;
}

uint32_t BuildCacheTelemetry::get_cache_hit_count() const {
    if (!impl_) {
        return 0;
    }
    return impl_->cached_hit_count.load(std::memory_order_acquire);
}

uint32_t BuildCacheTelemetry::get_merge_count() const {
    if (!impl_) {
        return 0;
    }
    return impl_->merged_artifacts.load(std::memory_order_acquire);
}

uint32_t BuildCacheTelemetry::get_genfile_merge_count() const {
    if (!impl_) {
        return 0;
    }
    return impl_->merged_genfiles.load(std::memory_order_acquire);
}

uint32_t BuildCacheTelemetry::get_jit_once_dedup_count() const {
    if (!impl_) {
        return 0;
    }
    return impl_->jit_once_dedup_count.load(std::memory_order_acquire);
}

void BuildCacheTelemetry::log_compile_summary() const {
    if (!impl_) {
        return;
    }
    const uint64_t packed = impl_->srcs_and_compiled.load(std::memory_order_acquire);
    const uint32_t total = packed >> 32;
    const uint32_t compiled = packed & 0xFFFF'FFFFu;

    if (total == 0) {
        log_info(tt::LogBuildKernels, "JIT cache stats: no build info recorded");
        return;
    }

    const uint32_t hits = total - compiled;
    const uint32_t cached = impl_->cached_hit_count.load(std::memory_order_acquire);
    const uint32_t artifacts = impl_->merged_artifacts.load(std::memory_order_acquire);
    const uint32_t genfiles = impl_->merged_genfiles.load(std::memory_order_acquire);
    const uint32_t dedup = impl_->jit_once_dedup_count.load(std::memory_order_acquire);
    log_info(
        tt::LogBuildKernels,
        "JIT cache stats: {}/{} hits ({:.1f}%) [{} cached, {} build-once dedup, "
        "{} merged artifacts, {} merged genfiles]",
        hits,
        total,
        100.0 * static_cast<double>(hits) / static_cast<double>(total),
        cached,
        dedup,
        artifacts,
        genfiles);
}

TelemetryToken& BuildCacheTelemetry::register_metric(const std::string& name) {
    std::lock_guard lk(owned_tokens_mutex_);
    owned_tokens_.push_back(std::make_unique<TelemetryToken>(name));
    auto* token = owned_tokens_.back().get();
    token->set_recording_enabled(impl_ != nullptr);
    if (impl_) {
        std::lock_guard reg_lk(impl_->token_registry_mutex);
        impl_->registered_tokens.push_back(token);
    }
    return *token;
}

void BuildCacheTelemetry::dump_metrics() const {
    if (!impl_) {
        return;
    }

    log_compile_summary();

    std::lock_guard lk(impl_->token_registry_mutex);
    log_info(tt::LogBuildKernels, "JIT telemetry: {} registered TelemetryTokens", impl_->registered_tokens.size());

    for (const auto* token : impl_->registered_tokens) {
        const TelemetryTokenData snap = token->snapshot();
        if (snap.count == 0) {
            continue;
        }
        const double mean_val = snap.total / static_cast<double>(snap.count);
        log_info(
            tt::LogBuildKernels,
            "JIT telemetry [{}]: count={}, total={:.3f}ms, min={:.3f}ms, max={:.3f}ms, mean={:.3f}ms",
            token->name(),
            snap.count,
            snap.total,
            snap.min_val,
            snap.max_val,
            mean_val);
    }
}

}  // namespace tt::tt_metal
