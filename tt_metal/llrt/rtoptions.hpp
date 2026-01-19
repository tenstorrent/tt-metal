// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Run Time Options
//
// Reads env vars and sets up a global object which contains run time
// configuration options (such as debug logging)
//

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <unordered_set>
#include <string>
#include <vector>
#include <atomic>
#include "llrt/hal.hpp"
#include "core_coord.hpp"
#include "dispatch_core_common.hpp"  // For DispatchCoreConfig
#include "tt_target_device.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include "tt_metal/hw/inc/hostdev/fabric_telemetry_msgs.h"

namespace tt::llrt {
// Forward declaration - full definition in rtoptions.cpp
enum class EnvVarID;

inline std::string g_root_dir;
inline std::once_flag g_root_once;

// Enumerates the debug features that can be enabled at runtime. These features allow for
// fine-grained control over targeted cores, chips, harts, etc.
enum RunTimeDebugFeatures {
    RunTimeDebugFeatureDprint,
    RunTimeDebugFeatureReadDebugDelay,
    RunTimeDebugFeatureWriteDebugDelay,
    RunTimeDebugFeatureAtomicDebugDelay,
    RunTimeDebugFeatureEnableL1DataCache,
    // NOTE: Update RunTimeDebugFeatureNames if adding new features
    RunTimeDebugFeatureCount
};

// Enumerates a class of cores to enable features on at runtime.
enum RunTimeDebugClass {
    RunTimeDebugClassNoneSpecified,
    RunTimeDebugClassWorker,
    RunTimeDebugClassDispatch,
    RunTimeDebugClassAll,
    RunTimeDebugClassCount
};

extern const char* RunTimeDebugFeatureNames[RunTimeDebugFeatureCount];
extern const char* RunTimeDebugClassNames[RunTimeDebugClassCount];

// TargetSelection stores the targets for a given debug feature. I.e. for which chips, cores, harts
// to enable the feature.
struct TargetSelection {
    std::map<CoreType, std::vector<CoreCoord>> cores;
    std::map<CoreType, int> all_cores;
    bool enabled{};
    std::vector<int> chip_ids;
    bool all_chips = false;
    tt_metal::HalProcessorSet processors;
    std::string file_name;  // File name to write output to.
    bool one_file_per_risc = false;
    bool prepend_device_core_risc{};
};

struct WatcherSettings {
    std::atomic<bool> enabled = false;
    std::atomic<bool> dump_all = false;
    std::atomic<bool> append = false;
    std::atomic<bool> auto_unpause = false;
    std::atomic<bool> noinline = false;
    bool phys_coords = false;
    bool text_start = false;
    bool skip_logging = false;
    bool noc_sanitize_linked_transaction = false;
    std::atomic<int> interval_ms = 0;
};

struct InspectorSettings {
    bool enabled = true;
    bool initialization_is_important = false;
    bool warn_on_write_exceptions = true;
    std::filesystem::path log_path;
    std::string rpc_server_host = "localhost";
    uint16_t rpc_server_port = 50051;
    bool rpc_server_enabled = true;
    bool serialize_on_dispatch_timeout = true;
    std::string rpc_server_address() const { return rpc_server_host + ":" + std::to_string(rpc_server_port); }
};

template <typename T>
struct FabricTelemetrySelection {
    bool monitor_all = true;
    std::unordered_set<T> ids;

    bool matches(T value) const { return monitor_all || ids.count(value) > 0; }

    void set_monitor_all(bool value) {
        monitor_all = value;
        if (monitor_all) {
            ids.clear();
        }
    }
};

struct FabricTelemetrySettings {
    static constexpr uint8_t kAllStatsMask =
        static_cast<uint8_t>(DynamicStatistics::ROUTER_STATE) | static_cast<uint8_t>(DynamicStatistics::BANDWIDTH) |
        static_cast<uint8_t>(DynamicStatistics::HEARTBEAT_TX) | static_cast<uint8_t>(DynamicStatistics::HEARTBEAT_RX);

    bool enabled = false;
    FabricTelemetrySelection<uint32_t> chips;
    FabricTelemetrySelection<uint32_t> channels;
    FabricTelemetrySelection<uint32_t> eriscs;
    uint8_t stats_mask = kAllStatsMask;
    bool is_telemetry_enabled(uint32_t phys_chip_id, uint32_t channel_id, uint32_t risc_id) const {
        return chips.matches(phys_chip_id) && channels.matches(channel_id) && eriscs.matches(risc_id);
    }
};

class RunTimeOptions {
    std::string root_dir;

    bool is_cache_dir_env_var_set = false;
    std::string cache_dir_;

    std::string logs_dir_ = (std::filesystem::current_path() / "").string();

    bool is_kernel_dir_env_var_set = false;
    std::string kernel_dir;
    std::string system_kernel_dir;

    bool is_core_grid_override_todeprecate_env_var_set = false;
    std::string core_grid_override_todeprecate;

    bool is_custom_fabric_mesh_graph_desc_path_set = false;
    std::string custom_fabric_mesh_graph_desc_path;

    bool build_map_enabled = false;

    WatcherSettings watcher_settings;
    bool record_noc_transfer_data = false;

    InspectorSettings inspector_settings;

    bool lightweight_kernel_asserts = false;

    bool enable_llk_asserts = false;

    // Fabric profiling settings
    struct FabricProfilingSettings {
        bool enable_rx_ch_fwd = false;
    } fabric_profiling_settings;

    TargetSelection feature_targets[RunTimeDebugFeatureCount];

    std::atomic<bool> test_mode_enabled = false;

    bool profiler_enabled = false;
    bool profile_dispatch_cores = false;
    bool profiler_sync_enabled = false;
    bool profiler_mid_run_dump = false;
    bool profiler_trace_profiler = false;
    bool profiler_trace_tracking = false;
    bool profiler_cpp_post_process = false;
    bool profiler_sum = false;
    bool profiler_buffer_usage_enabled = false;
    bool profiler_noc_events_enabled = false;
    uint32_t profiler_perf_counter_mode = 0;
    std::string profiler_noc_events_report_path;
    bool profiler_disable_dump_to_files = false;
    bool profiler_disable_push_to_tracy = false;
    std::optional<uint32_t> profiler_program_support_count = std::nullopt;
    bool experimental_device_debug_dump_enabled = false;

    bool null_kernels = false;
    // Kernels should return early, skipping the rest of the kernel. Kernels
    // should remain the same size as normal, unlike with null_kernels.
    bool kernels_early_return = false;

    bool clear_l1 = false;
    bool clear_dram = false;

    bool skip_loading_fw = false;

    bool jit_analytics_enabled = false;
    bool riscv_debug_info_enabled = false;
    uint32_t watcher_debug_delay = 0;

    bool validate_kernel_binaries = false;
    unsigned num_hw_cqs = 1;

    bool using_slow_dispatch = false;

    bool enable_dispatch_data_collection = false;

    // HW can clear Blackhole's L1 data cache psuedo-randomly once every 128 transactions
    // This option will enable this feature to help flush out whether there is a missing cache invalidation
    bool enable_hw_cache_invalidation = false;

    tt_metal::DispatchCoreType dispatch_core_type = tt_metal::DispatchCoreType::WORKER;

    bool skip_deleting_built_cache = false;

    std::filesystem::path simulator_path = "";

    bool erisc_iram_enabled = false;
    // a copy for an intermittent period until the environment variable TT_METAL_ENABLE_ERISC_IRAM is removed
    // we keep a copy so that when we teardown the fabric (which enables erisc iram internally), we can recover
    // to the user override (if it existed)
    std::optional<bool> erisc_iram_enabled_env_var = std::nullopt;

    bool fast_dispatch = true;

    bool skip_eth_cores_with_retrain = false;

    // Relaxed ordering on BH allows loads to bypass stores when going to separate addresses
    // e.g. Store A followed by Load A will be unchanges but Store A followed by Load B may return B before A is written
    // This option will disable the relaxed ordering
    bool disable_relaxed_memory_ordering = false;

    // Enable instruction gathering in Tensix core.
    bool enable_gathering = false;

    // Buffer in DRAM to store various ARC processor samples. Feature not ready yet
    uint32_t arc_debug_buffer_size = 0;

    // Force disables using DMA for reads and writes
    std::atomic<bool> disable_dma_ops = false;

    // Forces MetalContext re-init on Device creation. Workaround for upstream issues that require re-init each time
    // (#25048) TODO: Once all of init is moved to MetalContext, investigate removing this option.
    bool force_context_reinit = false;
    // Comma-separated list of device IDs to make visible to the runtime
    std::string visible_devices;

    // Sets the architecture name (only necessary during simulation)
    std::string arch_name;

    // Forces Tracy profiler pushes during execution for real-time profiling
    bool tracy_mid_run_push = false;

    // presence-based override to force-disable fabric 2-ERISC regardless of defaults
    bool disable_fabric_2_erisc_mode = false;

    // feature flag to enable 2-erisc mode on Blackhole (general, not fabric-specific)
    bool enable_2_erisc_mode = true;

    // Log kernels compilation commands
    bool log_kernels_compilation_commands = false;

    // Enable fabric performance telemetry
    bool enable_fabric_bw_telemetry = false;

    // Enable fabric telemetry
    bool enable_fabric_telemetry = false;
    FabricTelemetrySettings fabric_telemetry_settings;

    // Mock cluster initialization using a provided cluster descriptor
    std::string mock_cluster_desc_path;

    // Consolidated target device selection
    TargetDevice runtime_target_device_ = TargetDevice::Silicon;
    // Timeout duration for operations
    std::chrono::duration<float> timeout_duration_for_operations = std::chrono::duration<float>(0.0f);
    // Command to run when a dispatch timeout occurs
    std::string dispatch_timeout_command_to_execute;

    // Using MGD 2.0 syntax for mesh graph descriptor
    bool use_mesh_graph_descriptor_2_0 = false;

    // Reliability mode override parsed from environment (RELIABILITY_MODE)
    std::optional<tt::tt_fabric::FabricReliabilityMode> reliability_mode = std::nullopt;

    // Force JIT compile even if dependencies are up to date
    bool force_jit_compile = false;

    // To be used for NUMA node based thread binding
    bool numa_based_affinity = false;

    // Fabric router sync timeout configuration (in milliseconds)
    // If not set, fabric code will use its own default
    std::optional<uint32_t> fabric_router_sync_timeout_ms = std::nullopt;

    // Disable XIP dump
    bool disable_xip_dump = false;

    // Dump JIT build commands to stdout for debugging
    bool dump_build_commands = false;

public:
    RunTimeOptions();
    RunTimeOptions(const RunTimeOptions&) = delete;
    RunTimeOptions& operator=(const RunTimeOptions&) = delete;

    static void set_root_dir(const std::string& root_dir);
    const std::string& get_root_dir() const;

    bool is_cache_dir_specified() const { return this->is_cache_dir_env_var_set; }
    const std::string& get_cache_dir() const;

    // Returns the logs directory for generated output (dprint, watcher, profiler, etc.)
    // Uses TT_METAL_LOGS_PATH if set, otherwise defaults to current working directory
    const std::string& get_logs_dir() const;

    bool is_kernel_dir_specified() const { return this->is_kernel_dir_env_var_set; }
    const std::string& get_kernel_dir() const;
    // Location where kernels are installed via package manager.
    const std::string& get_system_kernel_dir() const;

    bool is_core_grid_override_todeprecate() const { return this->is_core_grid_override_todeprecate_env_var_set; }
    const std::string& get_core_grid_override_todeprecate() const;

    bool get_build_map_enabled() const { return build_map_enabled; }

    // Info from watcher environment variables, setters included so that user
    // can override with a SW call.
    bool get_watcher_enabled() const { return watcher_settings.enabled.load(std::memory_order_relaxed); }
    void set_watcher_enabled(bool enabled) { watcher_settings.enabled.store(enabled, std::memory_order_relaxed); }
    // Return a hash of which watcher features are enabled
    uint32_t get_watcher_hash() const;
    int get_watcher_interval() const { return watcher_settings.interval_ms.load(std::memory_order_relaxed); }
    void set_watcher_interval(int interval_ms) {
        watcher_settings.interval_ms.store(interval_ms, std::memory_order_relaxed);
    }
    bool get_watcher_dump_all() const { return watcher_settings.dump_all.load(std::memory_order_relaxed); }
    void set_watcher_dump_all(bool dump_all) { watcher_settings.dump_all.store(dump_all, std::memory_order_relaxed); }
    bool get_watcher_append() const { return watcher_settings.append.load(std::memory_order_relaxed); }
    void set_watcher_append(bool append) { watcher_settings.append.store(append, std::memory_order_relaxed); }
    bool get_watcher_auto_unpause() const { return watcher_settings.auto_unpause.load(std::memory_order_relaxed); }
    void set_watcher_auto_unpause(bool auto_unpause) {
        watcher_settings.auto_unpause.store(auto_unpause, std::memory_order_relaxed);
    }
    bool get_watcher_noinline() const { return watcher_settings.noinline.load(std::memory_order_relaxed); }
    void set_watcher_noinline(bool noinline) { watcher_settings.noinline.store(noinline, std::memory_order_relaxed); }
    bool get_watcher_phys_coords() const { return watcher_settings.phys_coords; }
    void set_watcher_phys_coords(bool phys_coords) { watcher_settings.phys_coords = phys_coords; }
    bool get_watcher_text_start() const { return watcher_settings.text_start; }
    void set_watcher_text_start(bool text_start) { watcher_settings.text_start = text_start; }
    bool get_watcher_skip_logging() const { return watcher_settings.skip_logging; }
    void set_watcher_skip_logging(bool skip_logging) { watcher_settings.skip_logging = skip_logging; }
    bool get_inspector_rpc_server_enabled() const { return inspector_settings.rpc_server_enabled; }
    const std::string& get_inspector_rpc_server_host() const { return inspector_settings.rpc_server_host; }
    uint16_t get_inspector_rpc_server_port() const { return inspector_settings.rpc_server_port; }
    bool get_serialize_inspector_on_dispatch_timeout() const {
        return inspector_settings.serialize_on_dispatch_timeout;
    }
    bool get_watcher_noc_sanitize_linked_transaction() const {
        return watcher_settings.noc_sanitize_linked_transaction;
    }
    void set_watcher_noc_sanitize_linked_transaction(bool enabled) {
        watcher_settings.noc_sanitize_linked_transaction = enabled;
    }
    const std::set<std::string>& get_watcher_disabled_features() const { return watcher_disabled_features; }
    bool watcher_status_disabled() const { return watcher_feature_disabled(watcher_waypoint_str); }
    bool watcher_noc_sanitize_disabled() const { return watcher_feature_disabled(watcher_noc_sanitize_str); }
    bool watcher_assert_disabled() const { return watcher_feature_disabled(watcher_assert_str); }
    bool watcher_pause_disabled() const { return watcher_feature_disabled(watcher_pause_str); }
    bool watcher_ring_buffer_disabled() const { return watcher_feature_disabled(watcher_ring_buffer_str); }
    bool watcher_stack_usage_disabled() const { return watcher_feature_disabled(watcher_stack_usage_str); }
    bool watcher_dispatch_disabled() const { return watcher_feature_disabled(watcher_dispatch_str); }
    bool watcher_eth_link_status_disabled() const { return watcher_feature_disabled(watcher_eth_link_status_str); }

    bool get_lightweight_kernel_asserts() const { return lightweight_kernel_asserts; }
    void set_lightweight_kernel_asserts(bool enabled) { lightweight_kernel_asserts = enabled; }

    bool get_llk_asserts() const { return enable_llk_asserts; }
    void set_llk_asserts(bool enabled) { enable_llk_asserts = enabled; }

    // Info from inspector environment variables, setters included so that user
    // can override with a SW call.
    const std::filesystem::path& get_inspector_log_path() const { return inspector_settings.log_path; }
    bool get_inspector_enabled() const { return inspector_settings.enabled; }
    void set_inspector_enabled(bool enabled) { inspector_settings.enabled = enabled; }
    bool get_inspector_initialization_is_important() const { return inspector_settings.initialization_is_important; }
    void set_inspector_initialization_is_important(bool important) {
        inspector_settings.initialization_is_important = important;
    }
    bool get_inspector_warn_on_write_exceptions() const { return inspector_settings.warn_on_write_exceptions; }
    void set_inspector_warn_on_write_exceptions(bool warn) { inspector_settings.warn_on_write_exceptions = warn; }
    std::string get_inspector_rpc_server_address() const {
        return inspector_settings.rpc_server_host + ":" + std::to_string(inspector_settings.rpc_server_port);
    }
    void set_inspector_rpc_server_enabled(bool enabled) { inspector_settings.rpc_server_enabled = enabled; }
    // Info from DPrint environment variables, setters included so that user can
    // override with a SW call.
    bool get_feature_enabled(RunTimeDebugFeatures feature) const { return feature_targets[feature].enabled; }
    void set_feature_enabled(RunTimeDebugFeatures feature, bool enabled) { feature_targets[feature].enabled = enabled; }
    // Note: dprint cores are logical
    const std::map<CoreType, std::vector<CoreCoord>>& get_feature_cores(RunTimeDebugFeatures feature) const {
        return feature_targets[feature].cores;
    }
    void set_feature_cores(RunTimeDebugFeatures feature, std::map<CoreType, std::vector<CoreCoord>> cores) {
        feature_targets[feature].cores = std::move(cores);
    }
    // An alternative to setting cores by range, a flag to enable all.
    void set_feature_all_cores(RunTimeDebugFeatures feature, CoreType core_type, int all_cores) {
        feature_targets[feature].all_cores[core_type] = all_cores;
    }
    int get_feature_all_cores(RunTimeDebugFeatures feature, CoreType core_type) const {
        return feature_targets[feature].all_cores.at(core_type);
    }
    // Note: core range is inclusive
    void set_feature_core_range(RunTimeDebugFeatures feature, CoreCoord start, CoreCoord end, CoreType core_type) {
        feature_targets[feature].cores[core_type] = std::vector<CoreCoord>();
        for (uint32_t x = start.x; x <= end.x; x++) {
            for (uint32_t y = start.y; y <= end.y; y++) {
                feature_targets[feature].cores[core_type].push_back({x, y});
            }
        }
    }
    const std::vector<int>& get_feature_chip_ids(RunTimeDebugFeatures feature) const {
        return feature_targets[feature].chip_ids;
    }
    void set_feature_chip_ids(RunTimeDebugFeatures feature, std::vector<int> chip_ids) {
        feature_targets[feature].chip_ids = std::move(chip_ids);
    }
    // An alternative to setting cores by range, a flag to enable all.
    void set_feature_all_chips(RunTimeDebugFeatures feature, bool all_chips) {
        feature_targets[feature].all_chips = all_chips;
    }
    bool get_feature_all_chips(RunTimeDebugFeatures feature) const { return feature_targets[feature].all_chips; }
    const tt_metal::HalProcessorSet& get_feature_processors(RunTimeDebugFeatures feature) const {
        return feature_targets[feature].processors;
    }
    void set_feature_processors(RunTimeDebugFeatures feature, tt_metal::HalProcessorSet processors) {
        feature_targets[feature].processors = processors;
    }
    std::string get_feature_file_name(RunTimeDebugFeatures feature) const { return feature_targets[feature].file_name; }
    void set_feature_file_name(RunTimeDebugFeatures feature, const std::string& file_name) {
        feature_targets[feature].file_name = file_name;
    }
    bool get_feature_one_file_per_risc(RunTimeDebugFeatures feature) const {
        return feature_targets[feature].one_file_per_risc;
    }
    void set_feature_one_file_per_risc(RunTimeDebugFeatures feature, bool one_file_per_risc) {
        feature_targets[feature].one_file_per_risc = one_file_per_risc;
    }
    bool get_feature_prepend_device_core_risc(RunTimeDebugFeatures feature) const {
        return feature_targets[feature].prepend_device_core_risc;
    }
    void set_feature_prepend_device_core_risc(RunTimeDebugFeatures feature, bool prepend_device_core_risc) {
        feature_targets[feature].prepend_device_core_risc = prepend_device_core_risc;
    }
    TargetSelection get_feature_targets(RunTimeDebugFeatures feature) const { return feature_targets[feature]; }
    void set_feature_targets(RunTimeDebugFeatures feature, const TargetSelection& targets) {
        feature_targets[feature] = targets;
    }

    bool get_record_noc_transfers() const { return record_noc_transfer_data; }
    void set_record_noc_transfers(bool val) { record_noc_transfer_data = val; }

    bool get_validate_kernel_binaries() const { return validate_kernel_binaries; }
    void set_validate_kernel_binaries(bool val) { validate_kernel_binaries = val; }

    // Returns the string representation for hash computation.
    std::string get_feature_hash_string(RunTimeDebugFeatures feature) const {
        switch (feature) {
            case RunTimeDebugFeatureDprint: {
                std::string hash_str = std::to_string(get_feature_enabled(feature));
                hash_str += std::to_string(get_feature_all_chips(feature));
                return hash_str;
            }
            case RunTimeDebugFeatureReadDebugDelay:
            case RunTimeDebugFeatureWriteDebugDelay:
            case RunTimeDebugFeatureAtomicDebugDelay:
                if (get_feature_enabled(feature)) {
                    return std::to_string(get_watcher_debug_delay());
                } else {
                    return "false";
                }
            case RunTimeDebugFeatureEnableL1DataCache: return std::to_string(get_feature_enabled(feature));
            default: return "";
        }
    }
    std::string get_compile_hash_string() const {
        std::string compile_hash_str = fmt::format(
            "{}_{}_{}_{}_{}",
            get_watcher_hash(),
            get_kernels_early_return(),
            get_erisc_iram_enabled(),
            get_enable_2_erisc_mode(),
            get_disable_fabric_2_erisc_mode());
        for (int i = 0; i < RunTimeDebugFeatureCount; i++) {
            compile_hash_str += "_";
            compile_hash_str += get_feature_hash_string((llrt::RunTimeDebugFeatures)i);
        }
        return compile_hash_str;
    }

    // Used for both watcher and dprint servers, this dev option (no corresponding env var) sets
    // whether to catch exceptions (test mode = true) coming from debug servers or to throw them
    // (test mode = false). We need to catch for gtesting, since an unhandled exception will kill
    // the gtest (and can't catch an exception from the server thread in main thread), but by
    // default we should throw so that the user can see the exception as soon as it happens.
    bool get_test_mode_enabled() const { return test_mode_enabled.load(std::memory_order_relaxed); }
    void set_test_mode_enabled(bool enable) { test_mode_enabled.store(enable, std::memory_order_relaxed); }

    bool get_profiler_enabled() const { return profiler_enabled; }
    bool get_profiler_do_dispatch_cores() const { return profile_dispatch_cores; }
    bool get_profiler_sync_enabled() const { return profiler_sync_enabled; }
    bool get_profiler_trace_only() const { return profiler_trace_profiler; }
    bool get_profiler_trace_tracking() const { return profiler_trace_tracking; }
    bool get_profiler_mid_run_dump() const { return profiler_mid_run_dump; }
    bool get_profiler_cpp_post_process() const { return profiler_cpp_post_process; }
    bool get_profiler_sum() const { return profiler_sum; }
    std::optional<uint32_t> get_profiler_program_support_count() const { return profiler_program_support_count; }
    void set_profiler_program_support_count(uint32_t profiler_program_support_count) {
        this->profiler_program_support_count = profiler_program_support_count;
    }
    bool get_profiler_buffer_usage_enabled() const { return profiler_buffer_usage_enabled; }
    bool get_profiler_noc_events_enabled() const { return profiler_noc_events_enabled; }
    uint32_t get_profiler_perf_counter_mode() const { return profiler_perf_counter_mode; }
    std::string get_profiler_noc_events_report_path() const { return profiler_noc_events_report_path; }
    bool get_profiler_disable_dump_to_files() const { return profiler_disable_dump_to_files; }
    bool get_profiler_disable_push_to_tracy() const { return profiler_disable_push_to_tracy; }
    bool get_experimental_device_debug_dump_enabled() const { return experimental_device_debug_dump_enabled; }

    void set_kernels_nullified(bool v) { null_kernels = v; }
    bool get_kernels_nullified() const { return null_kernels; }

    void set_kernels_early_return(bool v) { kernels_early_return = v; }
    bool get_kernels_early_return() const { return kernels_early_return; }

    bool get_clear_l1() const { return clear_l1; }
    void set_clear_l1(bool clear) { clear_l1 = clear; }

    bool get_clear_dram() const { return clear_dram; }
    void set_clear_dram(bool clear) { clear_dram = clear; }

    std::string get_visible_devices() const { return visible_devices; }
    std::string get_arch_name() const { return arch_name; }
    bool get_tracy_mid_run_push() const { return tracy_mid_run_push; }

    bool get_skip_loading_fw() const { return skip_loading_fw; }

    bool get_jit_analytics_enabled() const { return jit_analytics_enabled; }
    void set_jit_analytics_enabled(bool enable) { jit_analytics_enabled = enable; }

    // Whether to compile with -g to include DWARF debug info in the binary.
    bool get_riscv_debug_info_enabled() const { return riscv_debug_info_enabled; }
    void set_riscv_debug_info_enabled(bool enable) { riscv_debug_info_enabled = enable; }

    unsigned get_num_hw_cqs() const { return num_hw_cqs; }
    void set_num_hw_cqs(unsigned num) { num_hw_cqs = num; }

    uint32_t get_watcher_debug_delay() const { return watcher_debug_delay; }
    void set_watcher_debug_delay(uint32_t delay) { watcher_debug_delay = delay; }

    bool get_dispatch_data_collection_enabled() const { return enable_dispatch_data_collection; }
    void set_dispatch_data_collection_enabled(bool enable) { enable_dispatch_data_collection = enable; }

    bool get_hw_cache_invalidation_enabled() const { return this->enable_hw_cache_invalidation; }

    bool get_relaxed_memory_ordering_disabled() const { return this->disable_relaxed_memory_ordering; }
    bool get_gathering_enabled() const { return this->enable_gathering; }

    tt_metal::DispatchCoreConfig get_dispatch_core_config() const;

    bool get_skip_deleting_built_cache() const { return skip_deleting_built_cache; }

    bool get_simulator_enabled() const { return runtime_target_device_ == TargetDevice::Simulator; }
    const std::filesystem::path& get_simulator_path() const { return simulator_path; }

    bool get_erisc_iram_enabled() const {
        // Disabled when debug tools are enabled due to IRAM size
        return erisc_iram_enabled && !get_watcher_enabled() && !get_feature_enabled(RunTimeDebugFeatureDprint);
    }
    bool get_erisc_iram_env_var_enabled() const {
        return erisc_iram_enabled_env_var.has_value() && erisc_iram_enabled_env_var.value();
    }
    bool get_erisc_iram_env_var_disabled() const {
        return erisc_iram_enabled_env_var.has_value() && !erisc_iram_enabled_env_var.value();
    }
    bool get_fast_dispatch() const { return fast_dispatch; }

    // Temporary API until all multi-device workloads are ported to run on fabric.
    // It's currently not possible to enable Erisc IRAM by default for all legacy CCL
    // workloads. In those workloads, erisc kernels are loaded every CCL op; the binary
    // copy to IRAM can noticeably degrade legacy CCL op performance in those cases.
    void set_erisc_iram_enabled(bool enable) { erisc_iram_enabled = enable; }

    bool get_skip_eth_cores_with_retrain() const { return skip_eth_cores_with_retrain; }

    uint32_t get_arc_debug_buffer_size() const { return arc_debug_buffer_size; }
    void set_arc_debug_buffer_size(uint32_t size) { arc_debug_buffer_size = size; }

    bool get_disable_dma_ops() const { return disable_dma_ops.load(std::memory_order_relaxed); }
    void set_disable_dma_ops(bool disable) { disable_dma_ops.store(disable, std::memory_order_relaxed); }

    bool get_force_context_reinit() const { return force_context_reinit; }

    // Presence-based override to force-disable fabric 2-ERISC
    bool get_disable_fabric_2_erisc_mode() const { return disable_fabric_2_erisc_mode; }

    // Feature flag to enable 2-erisc mode on Blackhole
    bool get_enable_2_erisc_mode() const { return enable_2_erisc_mode; }

    void set_enable_2_erisc_mode(bool enable) { enable_2_erisc_mode = enable; }

    bool is_custom_fabric_mesh_graph_desc_path_specified() const { return is_custom_fabric_mesh_graph_desc_path_set; }
    std::string get_custom_fabric_mesh_graph_desc_path() const { return custom_fabric_mesh_graph_desc_path; }

    bool get_log_kernels_compilation_commands() const { return log_kernels_compilation_commands; }

    // If true, the fabric (routers) will collect coarse grain telemetry data in software. This flag's state does not
    // affect the ability to capture Ethernet Subsystem register-read-based telemetry data.
    // This BW telemetry is coarse grain and records the total time that the reouter has unsent and inflight packets.
    //
    // NOTE: Enabling this option will lead to a 0-2% performance degradation for fabric traffic.
    bool get_enable_fabric_bw_telemetry() const { return enable_fabric_bw_telemetry; }
    void set_enable_fabric_bw_telemetry(bool enable) { enable_fabric_bw_telemetry = enable; }

    bool get_enable_fabric_telemetry() const { return enable_fabric_telemetry; }
    void set_enable_fabric_telemetry(bool enable) { enable_fabric_telemetry = enable; }
    const FabricTelemetrySettings& get_fabric_telemetry_settings() const { return fabric_telemetry_settings; }

    // If true, enables code profiling for receiver channel forward operations
    bool get_enable_fabric_code_profiling_rx_ch_fwd() const { return fabric_profiling_settings.enable_rx_ch_fwd; }
    void set_enable_fabric_code_profiling_rx_ch_fwd(bool enable) {
        fabric_profiling_settings.enable_rx_ch_fwd = enable;
    }

    // Reliability mode override accessor
    std::optional<tt::tt_fabric::FabricReliabilityMode> get_reliability_mode() const { return reliability_mode; }

    // Mock cluster accessors
    bool get_mock_enabled() const { return !mock_cluster_desc_path.empty(); }
    const std::string& get_mock_cluster_desc_path() const { return mock_cluster_desc_path; }
    // Set mock cluster descriptor from filename (prepends base path automatically)
    // NOTE: Must be called before Cluster is created (e.g., in MetalContext constructor).
    // Path depends on UMD's cluster_descriptor_examples directory structure.
    void set_mock_cluster_desc(const std::string& filename) {
        if (filename.empty()) {
            return;
        }
        mock_cluster_desc_path =
            get_root_dir() + "/tt_metal/third_party/umd/tests/cluster_descriptor_examples/" + filename;
        // Set target device to Mock if simulator is not enabled
        if (simulator_path.empty()) {
            runtime_target_device_ = tt::TargetDevice::Mock;
        }
    }

    // Target device accessor
    TargetDevice get_target_device() const { return runtime_target_device_; }

    std::chrono::duration<float> get_timeout_duration_for_operations() const { return timeout_duration_for_operations; }
    std::string get_dispatch_timeout_command_to_execute() const { return dispatch_timeout_command_to_execute; }
    // Mesh graph descriptor version accessor
    bool get_use_mesh_graph_descriptor_2_0() const { return use_mesh_graph_descriptor_2_0; }

    bool get_force_jit_compile() const { return force_jit_compile; }
    void set_force_jit_compile(bool enable) { force_jit_compile = enable; }

    bool get_numa_based_affinity() const { return numa_based_affinity; }

    std::optional<uint32_t> get_fabric_router_sync_timeout_ms() const { return fabric_router_sync_timeout_ms; }

    bool get_disable_xip_dump() const { return disable_xip_dump; }

    bool get_dump_build_commands() const { return dump_build_commands; }

    // Parse all feature-specific environment variables, after hal is initialized.
    // (Needed because syntax of some env vars is arch-dependent.)
    void ParseAllFeatureEnv(const tt_metal::Hal& hal) {
        for (int i = 0; i < RunTimeDebugFeatureCount; i++) {
            ParseFeatureEnv((RunTimeDebugFeatures)i, hal);
        }
    }

private:
    // Helper functions to parse feature-specific environment vaiables.
    void ParseFeatureEnv(RunTimeDebugFeatures feature, const tt_metal::Hal& hal);
    void ParseFeatureCoreRange(RunTimeDebugFeatures feature, const std::string& env_var, CoreType core_type);
    void ParseFeatureChipIds(RunTimeDebugFeatures feature, const std::string& env_var);
    void ParseFeatureRiscvMask(RunTimeDebugFeatures feature, const std::string& env_var, const tt_metal::Hal& hal);
    void ParseFeatureFileName(RunTimeDebugFeatures feature, const std::string& env_var);
    void ParseFeatureOneFilePerRisc(RunTimeDebugFeatures feature, const std::string& env_var);
    void ParseFeaturePrependDeviceCoreRisc(RunTimeDebugFeatures feature, const std::string& env_var);
    void ParseFabricTelemetryEnv(const char* value);
    void HandleEnvVar(
        EnvVarID id, const char* value);  // Handle single env var (value usually non-null, see cpp for details)
    void InitializeFromEnvVars();         // Initialize all environment variables from table
    // Helper function to parse watcher-specific environment variables.
    void ParseWatcherEnv();

    // Watcher feature name strings (used in env vars + defines in the device code), as well as a
    // set to track disabled features.
    const std::string watcher_waypoint_str = "WAYPOINT";
    const std::string watcher_noc_sanitize_str = "NOC_SANITIZE";
    const std::string watcher_assert_str = "ASSERT";
    const std::string watcher_pause_str = "PAUSE";
    const std::string watcher_ring_buffer_str = "RING_BUFFER";
    const std::string watcher_stack_usage_str = "STACK_USAGE";
    const std::string watcher_dispatch_str = "DISPATCH";
    const std::string watcher_eth_link_status_str = "ETH_LINK_STATUS";
    const std::string watcher_sanitize_read_only_l1_str = "SANITIZE_READ_ONLY_L1";
    const std::string watcher_sanitize_write_only_l1_str = "SANITIZE_WRITE_ONLY_L1";
    std::set<std::string> watcher_disabled_features;
    bool watcher_feature_disabled(const std::string& name) const { return watcher_disabled_features.contains(name); }
};

// Function declarations for operation timeout and synchronization
std::chrono::duration<float> get_timeout_duration_for_operations();

}  // namespace tt::llrt
