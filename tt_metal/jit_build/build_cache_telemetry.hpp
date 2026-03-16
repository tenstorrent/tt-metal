// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tt::tt_metal {

// Opaque handle returned by BuildCacheTelemetry::register_metric().
// Records a stream of double values into a deque.
// NOT thread-safe for concurrent record() calls on the SAME token.
// Different tokens may be used concurrently from different threads.
// Typically each token is stored as a function-local static and only
// accessed from a single call site, so no synchronization is needed.
class TelemetryToken {
public:
    TelemetryToken() = default;
    explicit TelemetryToken(std::string name);

    void record(double value);

    const std::string& name() const { return name_; }
    const std::deque<double>& values() const { return values_; }
    bool empty() const { return values_.empty(); }

private:
    std::string name_;
    std::deque<double> values_;
};

struct BuildCacheTelemetryImpl {
    std::atomic<size_t> total_srcs{0};
    std::atomic<size_t> compiled_count{0};
    std::atomic<size_t> cached_hit_count{0};
    std::atomic<size_t> merged_artifacts{0};
    std::atomic<size_t> merged_genfiles{0};
    std::atomic<size_t> jit_once_dedup_count{0};

    std::mutex token_registry_mutex;
    std::vector<TelemetryToken*> registered_tokens;
};

// Process-wide telemetry for JIT build cache merge diagnostics.
//
// Wraps BuildCacheTelemetryImpl via unique_ptr: when the impl exists, telemetry
// is active; when null, every method is a no-op.  Enabled by default.
//
// enable()/disable() are NOT thread-safe with respect to recording methods;
// call them only during setup/teardown when no concurrent builds are running.
//
// register_metric() returns a TelemetryToken that stores a stream of values
// in a std::deque.  At process exit, dump_metrics() iterates all registered
// tokens and prints name, min, max, and mean via log_info.
class BuildCacheTelemetry {
public:
    static BuildCacheTelemetry& inst();

    void enable();
    void disable();
    bool is_enabled() const { return impl_ != nullptr; }

    void record_compile(size_t num_srcs, size_t num_compiled);
    void record_cache_hit();
    void record_merge(size_t count);
    void record_genfile_merge(size_t count);
    void record_jit_once_dedup();
    void log_compile_summary(bool state_changed) const;

    // Register a named metric stream. The returned token is owned by the caller
    // and must outlive the telemetry instance (or at least until dump_metrics).
    // Typically stored as a static local.
    TelemetryToken& register_metric(const std::string& name);

    // Print min/max/mean for all registered metrics. Called automatically at
    // process exit when telemetry is enabled.
    void dump_metrics() const;

private:
    BuildCacheTelemetry();
    ~BuildCacheTelemetry();
    std::unique_ptr<BuildCacheTelemetryImpl> impl_;
    // Tokens are owned by the telemetry instance for lifetime management.
    std::vector<std::unique_ptr<TelemetryToken>> owned_tokens_;
    std::mutex owned_tokens_mutex_;
};

}  // namespace tt::tt_metal
