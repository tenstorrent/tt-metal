// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "build_cache_telemetry.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>

#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal {

// --- TelemetryToken ---

TelemetryToken::TelemetryToken(std::string name) : name_(std::move(name)) {}

void TelemetryToken::record(double value) { values_.push_back(value); }

// --- BuildCacheTelemetry ---

BuildCacheTelemetry::BuildCacheTelemetry() { enable(); }

BuildCacheTelemetry::~BuildCacheTelemetry() {
    if (impl_) {
        dump_metrics();
    }
}

BuildCacheTelemetry& BuildCacheTelemetry::inst() {
    static BuildCacheTelemetry instance;
    return instance;
}

void BuildCacheTelemetry::enable() { impl_ = std::make_unique<BuildCacheTelemetryImpl>(); }

void BuildCacheTelemetry::disable() { impl_.reset(); }

void BuildCacheTelemetry::record_compile(size_t num_srcs, size_t num_compiled) {
    if (!impl_) {
        return;
    }
    impl_->total_srcs.fetch_add(num_srcs, std::memory_order_relaxed);
    impl_->compiled_count.fetch_add(num_compiled, std::memory_order_relaxed);
}

void BuildCacheTelemetry::record_cache_hit() {
    if (!impl_) {
        return;
    }
    impl_->cached_hit_count.fetch_add(1, std::memory_order_relaxed);
}

void BuildCacheTelemetry::record_merge(size_t count) {
    if (!impl_) {
        return;
    }
    impl_->merged_artifacts.fetch_add(count, std::memory_order_relaxed);
}

void BuildCacheTelemetry::record_genfile_merge(size_t count) {
    if (!impl_) {
        return;
    }
    impl_->merged_genfiles.fetch_add(count, std::memory_order_relaxed);
}

void BuildCacheTelemetry::record_jit_once_dedup() {
    if (!impl_) {
        return;
    }
    impl_->jit_once_dedup_count.fetch_add(1, std::memory_order_relaxed);
}

void BuildCacheTelemetry::log_compile_summary(bool state_changed) const {
    if (!impl_) {
        return;
    }
    const size_t total = impl_->total_srcs.load(std::memory_order_relaxed);
    if (total == 0) {
        return;
    }
    const size_t compiled = impl_->compiled_count.load(std::memory_order_relaxed);
    const size_t hits = total - compiled;
    const size_t cached = impl_->cached_hit_count.load(std::memory_order_relaxed);
    const size_t artifacts = impl_->merged_artifacts.load(std::memory_order_relaxed);
    const size_t genfiles = impl_->merged_genfiles.load(std::memory_order_relaxed);
    const size_t dedup = impl_->jit_once_dedup_count.load(std::memory_order_relaxed);
    log_info(
        tt::LogBuildKernels,
        "JIT cache stats: {}/{} hits ({:.1f}%) [{} cached, {} build-once dedup, "
        "{} merged artifacts, {} merged genfiles]{}",
        hits,
        total,
        100.0 * static_cast<double>(hits) / static_cast<double>(total),
        cached,
        dedup,
        artifacts,
        genfiles,
        state_changed ? " [state changed, full recompile]" : "");
}

TelemetryToken& BuildCacheTelemetry::register_metric(const std::string& name) {
    std::lock_guard lk(owned_tokens_mutex_);
    owned_tokens_.push_back(std::make_unique<TelemetryToken>(name));
    auto* token = owned_tokens_.back().get();
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
    std::lock_guard lk(impl_->token_registry_mutex);
    for (const auto* token : impl_->registered_tokens) {
        const auto& vals = token->values();
        if (vals.empty()) {
            continue;
        }
        double min_val = std::numeric_limits<double>::max();
        double max_val = std::numeric_limits<double>::lowest();
        double sum = 0.0;
        for (double v : vals) {
            min_val = std::min(v, min_val);
            max_val = std::max(v, max_val);
            sum += v;
        }
        double mean_val = sum / static_cast<double>(vals.size());
        log_info(
            tt::LogBuildKernels,
            "JIT telemetry [{}]: count={}, min={:.3f}ms, max={:.3f}ms, mean={:.3f}ms",
            token->name(),
            vals.size(),
            min_val,
            max_val,
            mean_val);
    }
}

}  // namespace tt::tt_metal
