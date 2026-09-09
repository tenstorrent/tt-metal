// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tt::tt_metal {

class BuildCacheTelemetry;

struct TelemetryTokenData {
    uint32_t count{0};
    double total{0};
    double min_val{std::numeric_limits<double>::infinity()};
    double max_val{-std::numeric_limits<double>::infinity()};
};

// Opaque handle returned by BuildCacheTelemetry::get_or_register_metric().
// Maintains mutex-protected running total/count/min/max per value stream.
// record() and snapshot() are safe for concurrent use.
// References stay valid for the lifetime of BuildCacheTelemetry::inst();
// record() is a no-op while process-wide telemetry is disabled so values are
// not appended after disable(), but the token object is not destroyed.
class TelemetryToken {
public:
    TelemetryToken() = default;
    explicit TelemetryToken(std::string name, std::string unit = "ms");

    void record(double value);
    TelemetryTokenData snapshot() const;

    const std::string& name() const { return name_; }
    const std::string& unit() const { return unit_; }

private:
    friend class BuildCacheTelemetry;
    void set_recording_enabled(bool enabled);

    std::string name_;
    std::string unit_{"ms"};
    std::atomic<bool> recording_enabled_{true};
    mutable std::mutex data_mutex_;
    TelemetryTokenData data_;
};

struct BuildCacheTelemetryImpl;  // forward declaration

// Process-wide telemetry for JIT build cache merge diagnostics.
//
// get_or_register_metric() returns a TelemetryToken (running aggregate stats) that is
// owned in owned_tokens_ for the life of the inst() singleton; disable()/enable()
// tear down impl_ and rebuild the token registry (enable() is a no-op if already
// enabled). Tokens are not destroyed across disable/enable; only recording is toggled.
// dump_metrics() is called from the destructor body (before member dtors run), so impl_
// is still valid. tt::LoggerRegistry is a leaky singleton (allocated with `new`, never freed),
// so its loggers and sinks are alive for the entire process lifetime and are safe to use
// from the destructor.
class BuildCacheTelemetry {
public:
    static BuildCacheTelemetry& inst();

    // enable()/disable() are NOT thread-safe with respect to recording methods;
    // call them only during setup/teardown when no concurrent builds are running.
    // enable() is a no-op if telemetry is already enabled.
    void enable();
    void disable();
    bool is_enabled() const { return impl_ != nullptr; }

    void record_compile(uint32_t num_srcs, uint32_t num_compiled);
    void record_cache_hit();
    void record_merge(uint32_t count);
    // Counter for future genfile cache reuse; generation paths currently always rewrite files.
    void record_genfile_merge(uint32_t count);
    void record_jit_once_dedup();

    uint32_t get_srcs_count() const;
    uint32_t get_compile_count() const;
    uint32_t get_cache_hit_count() const;
    uint32_t get_merge_count() const;
    uint32_t get_genfile_merge_count() const;
    uint32_t get_jit_once_dedup_count() const;

    // Extend the process-wide JIT build window with [start, end]. The window is the wall-clock
    // span from the earliest build entry to the latest build exit, recorded as a single sample
    // into the "jit_build_window" metric by dump_metrics(). It is the parallelism-aware companion
    // to the per-call build metrics: summing those over-counts when builds run on the thread pool,
    // whereas the window is what a wall clock next to the process would show. A workload that
    // compiles lazily in bursts folds the idle gaps between bursts into the span.
    void note_build_window(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end);

    void log_compile_summary() const;

    // Return the metric stream named `name`, registering it on first use. The returned
    // TelemetryToken is owned in owned_tokens_ and the reference remains valid until the
    // process-wide singleton (inst()) is destroyed; it is not invalidated by disable()/enable(),
    // which only clear impl_ and rebuild the registry while leaving tokens allocated. Callers
    // must not take ownership. While telemetry is disabled, TelemetryToken::record() is a no-op;
    // snapshot() still reflects aggregates recorded while enabled. Repeat calls for the same name
    // must agree on `unit` -- a mismatch means two call sites are feeding one stream values of
    // different kinds, so it fails loudly rather than silently mislabelling the dump.
    TelemetryToken& get_or_register_metric(const std::string& name, std::string unit = "ms");

    void dump_metrics() const;

private:
    BuildCacheTelemetry();
    ~BuildCacheTelemetry();
    std::unique_ptr<BuildCacheTelemetryImpl> impl_;
    std::vector<std::unique_ptr<TelemetryToken>> owned_tokens_;
    // Name -> token index into owned_tokens_; guarded by owned_tokens_mutex_ alongside the vector.
    std::unordered_map<std::string, TelemetryToken*> tokens_by_name_;
    std::mutex owned_tokens_mutex_;
    // Registered in the constructor so the const dump_metrics() can record the window span
    // without registering a metric during teardown.
    TelemetryToken* build_window_token_{nullptr};
};

// Times the scope it lives in: takes a steady_clock timestamp on construction and records the
// elapsed milliseconds in `token` on destruction. The token must outlive the timer -- references
// from BuildCacheTelemetry::get_or_register_metric() are valid for the life of the inst() singleton.
// The delta is recorded even when the scope is left by an exception.
class ScopedTelemetryTimer {
public:
    explicit ScopedTelemetryTimer(TelemetryToken& token) :
        token_(token), start_(std::chrono::steady_clock::now()) {}

    ~ScopedTelemetryTimer() {
        token_.record(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count());
    }

    ScopedTelemetryTimer(const ScopedTelemetryTimer&) = delete;
    ScopedTelemetryTimer& operator=(const ScopedTelemetryTimer&) = delete;
    ScopedTelemetryTimer(ScopedTelemetryTimer&&) = delete;
    ScopedTelemetryTimer& operator=(ScopedTelemetryTimer&&) = delete;

private:
    TelemetryToken& token_;
    std::chrono::steady_clock::time_point start_;
};

// Folds the scope it lives in into the process-wide JIT build window (see
// BuildCacheTelemetry::note_build_window). Deliberately not a ScopedTelemetryTimer with static
// storage: such a timer would only record at process exit, so every test, inference, or idle
// stretch after the last compile would land in the metric. This ends the window at the last
// build instead. Contributes on every exit path from the scope, including exceptions and the
// build-cache early returns.
class ScopedBuildWindow {
public:
    ScopedBuildWindow();
    ~ScopedBuildWindow();

    ScopedBuildWindow(const ScopedBuildWindow&) = delete;
    ScopedBuildWindow& operator=(const ScopedBuildWindow&) = delete;
    ScopedBuildWindow(ScopedBuildWindow&&) = delete;
    ScopedBuildWindow& operator=(ScopedBuildWindow&&) = delete;

    std::chrono::steady_clock::time_point start() const { return start_; }

private:
    std::chrono::steady_clock::time_point start_;
};

// Invokes `fn(args...)` and records how long it took (in ms) in `token`, forwarding both the
// arguments and the return value through unchanged. The timing is recorded even if `fn` throws.
template <typename Fn, typename... Args>
decltype(auto) record_elapsed(TelemetryToken& token, Fn&& fn, Args&&... args) {
    ScopedTelemetryTimer timer(token);
    return std::forward<Fn>(fn)(std::forward<Args>(args)...);
}

// Convenience wrapper over BuildCacheTelemetry::get_or_register_metric() for metrics that are
// broken down by target: builds the "<metric_name>.<target_name>" key and returns that stream.
TelemetryToken& per_target_telemetry_token(
    std::string_view metric_name, std::string_view target_name, std::string_view unit);

}  // namespace tt::tt_metal
