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
#include <string>
#include <utility>
#include <vector>

#include "llrt/hal.hpp"
#include "core_coord.hpp"
#include "dispatch_core_common.hpp"  // For DispatchCoreConfig
#include "tt_target_device.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt {

namespace llrt {

// Enumerates the debug features that can be enabled at runtime. These features allow for
// fine-grained control over targeted cores, chips, harts, etc.
enum RunTimeDebugFeatures {
    RunTimeDebugFeatureDprint,
    RunTimeDebugFeatureReadDebugDelay,
    RunTimeDebugFeatureWriteDebugDelay,
    RunTimeDebugFeatureAtomicDebugDelay,
    RunTimeDebugFeatureDisableL1DataCache,
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
    bool enabled = false;
    bool dump_all = false;
    bool append = false;
    bool auto_unpause = false;
    bool noinline = false;
    bool phys_coords = false;
    bool text_start = false;
    bool skip_logging = false;
    bool noc_sanitize_linked_transaction = false;
    int interval_ms = 0;
};

struct InspectorSettings {
    bool enabled = true;
    bool initialization_is_important = false;
    bool warn_on_write_exceptions = true;
    std::filesystem::path log_path;
};

class RunTimeOptions {
    bool is_root_dir_env_var_set = false;
    std::string root_dir;

    bool is_cache_dir_env_var_set = false;
    std::string cache_dir_;

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

    TargetSelection feature_targets[RunTimeDebugFeatureCount];

    bool test_mode_enabled = false;

    bool profiler_enabled = false;
    bool profile_dispatch_cores = false;
    bool profiler_sync_enabled = false;
    bool profiler_mid_run_dump = false;
    bool profiler_trace_profiler = false;
    bool profiler_buffer_usage_enabled = false;
    bool profiler_noc_events_enabled = false;
    std::string profiler_noc_events_report_path;

    bool null_kernels = false;
    // Kernels should return early, skipping the rest of the kernel. Kernels
    // should remain the same size as normal, unlike with null_kernels.
    bool kernels_early_return = false;

    bool clear_l1 = false;
    bool clear_dram = false;

    bool skip_loading_fw = false;
    bool skip_reset_cores_on_init = false;

    bool riscv_debug_info_enabled = false;
    uint32_t watcher_debug_delay = 0;

    bool validate_kernel_binaries = false;
    unsigned num_hw_cqs = 1;

    bool fd_fabric_en = false;
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
    bool disable_dma_ops = false;

    // Forces MetalContext re-init on Device creation. Workaround for upstream issues that require re-init each time
    // (#25048) TODO: Once all of init is moved to MetalContext, investigate removing this option.
    bool force_context_reinit = false;

    // feature flag to enable 2-erisc mode with fabric on Blackhole, until it is enabled by default
    bool enable_2_erisc_mode_with_fabric = false;

    // Log kernels compilation commands
    bool log_kernels_compilation_commands = false;

    // Enable fabric performance telemetry
    bool enable_fabric_telemetry = false;

    // Mock cluster initialization using a provided cluster descriptor
    std::string mock_cluster_desc_path = "";

    // Consolidated target device selection
    TargetDevice runtime_target_device_ = TargetDevice::Silicon;
    // Timeout duration for operations
    std::chrono::duration<float> timeout_duration_for_operations = std::chrono::duration<float>(0.0f);

    // Using MGD 2.0 syntax for mesh graph descriptor in Fabric Control Plane
    bool use_mesh_graph_descriptor_2_0 = false;

public:
    RunTimeOptions();
    RunTimeOptions(const RunTimeOptions&) = delete;
    RunTimeOptions& operator=(const RunTimeOptions&) = delete;

    bool is_root_dir_specified() const { return this->is_root_dir_env_var_set; }
    const std::string& get_root_dir() const;

    bool is_cache_dir_specified() const { return this->is_cache_dir_env_var_set; }
    const std::string& get_cache_dir() const;

    bool is_kernel_dir_specified() const { return this->is_kernel_dir_env_var_set; }
    const std::string& get_kernel_dir() const;
    // Location where kernels are installed via package manager.
    const std::string& get_system_kernel_dir() const;

    bool is_core_grid_override_todeprecate() const { return this->is_core_grid_override_todeprecate_env_var_set; }
    const std::string& get_core_grid_override_todeprecate() const;

    bool get_build_map_enabled() const { return build_map_enabled; }

    // Info from watcher environment variables, setters included so that user
    // can override with a SW call.
    bool get_watcher_enabled() const { return watcher_settings.enabled; }
    void set_watcher_enabled(bool enabled) { watcher_settings.enabled = enabled; }
    int get_watcher_interval() const { return watcher_settings.interval_ms; }
    void set_watcher_interval(int interval_ms) { watcher_settings.interval_ms = interval_ms; }
    int get_watcher_dump_all() const { return watcher_settings.dump_all; }
    void set_watcher_dump_all(bool dump_all) { watcher_settings.dump_all = dump_all; }
    int get_watcher_append() const { return watcher_settings.append; }
    void set_watcher_append(bool append) { watcher_settings.append = append; }
    int get_watcher_auto_unpause() const { return watcher_settings.auto_unpause; }
    void set_watcher_auto_unpause(bool auto_unpause) { watcher_settings.auto_unpause = auto_unpause; }
    int get_watcher_noinline() const { return watcher_settings.noinline; }
    void set_watcher_noinline(bool noinline) { watcher_settings.noinline = noinline; }
    int get_watcher_phys_coords() const { return watcher_settings.phys_coords; }
    void set_watcher_phys_coords(bool phys_coords) { watcher_settings.phys_coords = phys_coords; }
    bool get_watcher_text_start() const { return watcher_settings.text_start; }
    void set_watcher_text_start(bool text_start) { watcher_settings.text_start = text_start; }
    bool get_watcher_skip_logging() const { return watcher_settings.skip_logging; }
    void set_watcher_skip_logging(bool skip_logging) { watcher_settings.skip_logging = skip_logging; }
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
    void set_feature_file_name(RunTimeDebugFeatures feature, std::string file_name) {
        feature_targets[feature].file_name = std::move(file_name);
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
    void set_feature_targets(RunTimeDebugFeatures feature, TargetSelection targets) {
        feature_targets[feature] = std::move(targets);
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
            case RunTimeDebugFeatureDisableL1DataCache: return std::to_string(get_feature_enabled(feature));
            default: return "";
        }
    }
    std::string get_compile_hash_string() const {
        std::string compile_hash_str =
            fmt::format("{}_{}_{}", get_watcher_enabled(), get_kernels_early_return(), get_erisc_iram_enabled());
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
    bool get_test_mode_enabled() const { return test_mode_enabled; }
    void set_test_mode_enabled(bool enable) { test_mode_enabled = enable; }

    bool get_profiler_enabled() const { return profiler_enabled; }
    bool get_profiler_do_dispatch_cores() const { return profile_dispatch_cores; }
    bool get_profiler_sync_enabled() const { return profiler_sync_enabled; }
    bool get_profiler_trace_only() const { return profiler_trace_profiler; }
    bool get_profiler_mid_run_dump() const { return profiler_mid_run_dump; }
    bool get_profiler_buffer_usage_enabled() const { return profiler_buffer_usage_enabled; }
    bool get_profiler_noc_events_enabled() const { return profiler_noc_events_enabled; }
    std::string get_profiler_noc_events_report_path() const { return profiler_noc_events_report_path; }

    void set_kernels_nullified(bool v) { null_kernels = v; }
    bool get_kernels_nullified() const { return null_kernels; }

    void set_kernels_early_return(bool v) { kernels_early_return = v; }
    bool get_kernels_early_return() const { return kernels_early_return; }

    bool get_clear_l1() const { return clear_l1; }
    void set_clear_l1(bool clear) { clear_l1 = clear; }

    bool get_clear_dram() const { return clear_dram; }
    void set_clear_dram(bool clear) { clear_dram = clear; }

    bool get_skip_loading_fw() const { return skip_loading_fw; }
    bool get_skip_reset_cores_on_init() const { return skip_reset_cores_on_init; }

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

    uint32_t get_arc_debug_buffer_size() { return arc_debug_buffer_size; }
    void set_arc_debug_buffer_size(uint32_t size) { arc_debug_buffer_size = size; }

    bool get_disable_dma_ops() const { return disable_dma_ops; }
    void set_disable_dma_ops(bool disable) { disable_dma_ops = disable; }

    bool get_force_context_reinit() const { return force_context_reinit; }

    // Feature flag to specify if fabric is enabled in 2-erisc mode or not.
    // if true, then the fabric router is parallelized across two eriscs in the Ethernet core
    bool get_is_fabric_2_erisc_mode_enabled() const { return enable_2_erisc_mode_with_fabric; }

    bool is_custom_fabric_mesh_graph_desc_path_specified() const { return is_custom_fabric_mesh_graph_desc_path_set; }
    std::string get_custom_fabric_mesh_graph_desc_path() const { return custom_fabric_mesh_graph_desc_path; }

    bool get_log_kernels_compilation_commands() const { return log_kernels_compilation_commands; }

    // If true, the fabric (routers) will collect coarse grain telemetry data in software. This flag's state does not
    // affect the ability to capture Ethernet Subsystem register-read-based telemetry data.
    // This BW telemetry is coarse grain and records the total time that the reouter has unsent and inflight packets.
    //
    // NOTE: Enabling this option will lead to a 0-2% performance degradation for fabric traffic.
    bool get_enable_fabric_telemetry() const { return enable_fabric_telemetry; }
    void set_enable_fabric_telemetry(bool enable) { enable_fabric_telemetry = enable; }

    // Mock cluster accessors
    bool get_mock_enabled() const { return runtime_target_device_ == TargetDevice::Mock; }
    const std::string& get_mock_cluster_desc_path() const { return mock_cluster_desc_path; }

    // Target device accessor
    TargetDevice get_target_device() const { return runtime_target_device_; }

    std::chrono::duration<float> get_timeout_duration_for_operations() const { return timeout_duration_for_operations; }

    // Using MGD 2.0 syntax for mesh graph descriptor in Fabric Control Plane
    // TODO: This will be removed after MGD 1.0 is deprecated
    bool get_use_mesh_graph_descriptor_2_0() const { return use_mesh_graph_descriptor_2_0; }

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
    std::set<std::string> watcher_disabled_features;
    bool watcher_feature_disabled(const std::string& name) const {
        return watcher_disabled_features.find(name) != watcher_disabled_features.end();
    }

    // Helper function to parse inspector-specific environment variables.
    void ParseInspectorEnv();
};

// Function declarations for operation timeout and synchronization
std::chrono::duration<float> get_timeout_duration_for_operations();

}  // namespace llrt

}  // namespace tt
