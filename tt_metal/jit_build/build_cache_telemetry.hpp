// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tt::tt_metal {

class BuildCacheTelemetry;

// Point-in-time copy of running stats from a TelemetryToken.
// Snapshots are transactional and are internally consistent.
struct TelemetryTokenSnapshot {
    size_t count{0};
    double total{0};
    double min_val{std::numeric_limits<double>::infinity()};
    double max_val{-std::numeric_limits<double>::infinity()};
};

// Opaque handle returned by BuildCacheTelemetry::register_metric().
// Maintains lock-free running total/count/min/max per value stream.
// record() and snapshot() are safe for concurrent use.
// References stay valid for the lifetime of BuildCacheTelemetry::inst();
// record() is a no-op while process-wide telemetry is disabled so values are
// not appended after disable(), but the token object is not destroyed.
// snapshots are guarenteed to be consistent
class TelemetryToken {
public:
    TelemetryToken() = default;
    explicit TelemetryToken(std::string name);

    void record(double value);
    TelemetryTokenSnapshot snapshot() const;

    const std::string& name() const { return name_; }

private:
    friend class BuildCacheTelemetry;
    void set_recording_enabled(bool enabled);

    struct RunningStats {
        std::atomic<size_t> seq{0};
        std::atomic<double> total{0};
        std::atomic<double> min_val{std::numeric_limits<double>::infinity()};
        std::atomic<double> max_val{-std::numeric_limits<double>::infinity()};
    };

    std::string name_;
    std::atomic<bool> recording_enabled_{true};
    RunningStats stats_;
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
// register_metric() returns a TelemetryToken (running aggregate stats) that is
// owned in owned_tokens_ for the life of the inst() singleton; disable()/enable()
// tear down impl_ and the token registry but do not destroy tokens, so returned
// references stay valid across disable/enable; only recording is toggled.
// dump_metrics() is called from the destructor body (before member dtors run), so impl_
// is still valid. tt::LoggerRegistry is a leaky singleton (allocated with `new`, never freed),
// so its loggers and sinks are alive for the entire process lifetime and are safe to use
// from the destructor.
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

    size_t get_srcs_count() const;
    size_t get_compile_count() const;
    size_t get_cache_hit_count() const;
    size_t get_merge_count() const;
    size_t get_genfile_merge_count() const;
    size_t get_jit_once_dedup_count() const;

    void log_compile_summary() const;

    // Register a named metric stream and return a non-owning reference to a
    // TelemetryToken owned in owned_tokens_. The reference remains valid until
    // the process-wide singleton (inst()) is destroyed; it is not invalidated by
    // disable()/enable(), which only clear impl_ and rebuild the registry while
    // leaving tokens allocated. Callers must not take ownership. While telemetry
    // is disabled, TelemetryToken::record() is a no-op; snapshot() still reflects
    // aggregates recorded while enabled.
    TelemetryToken& register_metric(const std::string& name);

    void dump_metrics() const;

private:
    BuildCacheTelemetry();
    ~BuildCacheTelemetry();
    std::unique_ptr<BuildCacheTelemetryImpl> impl_;
    std::vector<std::unique_ptr<TelemetryToken>> owned_tokens_;
    std::mutex owned_tokens_mutex_;
};

}  // namespace tt::tt_metal
