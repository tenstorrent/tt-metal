// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Run Time Options
//
// Reads env vars and sets up a global object which contains run time
// configuration options (such as debug logging)
//

#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "core_coord.hpp"
#include "dispatch_core_common.hpp"  // For DispatchCoreConfig
#include <umd/device/types/xy_pair.h>

enum class CoreType;

namespace tt {

namespace llrt {

// TODO: This should come from the HAL
enum DebugHartFlags : unsigned int {
    RISCV_NC = 1,
    RISCV_TR0 = 2,
    RISCV_TR1 = 4,
    RISCV_TR2 = 8,
    RISCV_BR = 16,
    RISCV_ER0 = 32,
    RISCV_ER1 = 64
};

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
    bool enabled;
    std::vector<int> chip_ids;
    bool all_chips = false;
    uint32_t riscv_mask = 0;
    std::string file_name;  // File name to write output to.
    bool one_file_per_risc = false;
    bool prepend_device_core_risc;
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
    int interval_ms = 0;
};

struct InspectorSettings {
    bool enabled = true;
    bool initialization_is_important = true;
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

    bool build_map_enabled = false;

    WatcherSettings watcher_settings;
    bool record_noc_transfer_data = false;

    InspectorSettings inspector_settings;

    TargetSelection feature_targets[RunTimeDebugFeatureCount];

    bool test_mode_enabled = false;

    bool profiler_enabled = false;
    bool profile_dispatch_cores = false;
    bool profiler_sync_enabled = false;
    bool profiler_mid_run_tracy_push = false;
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

    bool fb_fabric_en = false;

    bool enable_dispatch_data_collection = false;

    // HW can clear Blackhole's L1 data cache psuedo-randomly once every 128 transactions
    // This option will enable this feature to help flush out whether there is a missing cache invalidation
    bool enable_hw_cache_invalidation = false;

    tt_metal::DispatchCoreType dispatch_core_type = tt_metal::DispatchCoreType::WORKER;

    bool skip_deleting_built_cache = false;

    bool simulator_enabled = false;
    std::filesystem::path simulator_path = "";

    bool erisc_iram_enabled = false;

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

public:
    RunTimeOptions();
    RunTimeOptions(const RunTimeOptions&) = delete;
    RunTimeOptions& operator=(const RunTimeOptions&) = delete;

    inline bool is_root_dir_specified() const { return this->is_root_dir_env_var_set; }
    const std::string& get_root_dir() const;

    inline bool is_cache_dir_specified() const { return this->is_cache_dir_env_var_set; }
    const std::string& get_cache_dir() const;

    inline bool is_kernel_dir_specified() const { return this->is_kernel_dir_env_var_set; }
    const std::string& get_kernel_dir() const;
    // Location where kernels are installed via package manager.
    const std::string& get_system_kernel_dir() const;

    inline bool get_build_map_enabled() const { return build_map_enabled; }

    // Info from watcher environment variables, setters included so that user
    // can override with a SW call.
    inline bool get_watcher_enabled() const { return watcher_settings.enabled; }
    inline void set_watcher_enabled(bool enabled) { watcher_settings.enabled = enabled; }
    inline int get_watcher_interval() const { return watcher_settings.interval_ms; }
    inline void set_watcher_interval(int interval_ms) { watcher_settings.interval_ms = interval_ms; }
    inline int get_watcher_dump_all() const { return watcher_settings.dump_all; }
    inline void set_watcher_dump_all(bool dump_all) { watcher_settings.dump_all = dump_all; }
    inline int get_watcher_append() const { return watcher_settings.append; }
    inline void set_watcher_append(bool append) { watcher_settings.append = append; }
    inline int get_watcher_auto_unpause() const { return watcher_settings.auto_unpause; }
    inline void set_watcher_auto_unpause(bool auto_unpause) { watcher_settings.auto_unpause = auto_unpause; }
    inline int get_watcher_noinline() const { return watcher_settings.noinline; }
    inline void set_watcher_noinline(bool noinline) { watcher_settings.noinline = noinline; }
    inline int get_watcher_phys_coords() const { return watcher_settings.phys_coords; }
    inline void set_watcher_phys_coords(bool phys_coords) { watcher_settings.phys_coords = phys_coords; }
    inline bool get_watcher_text_start() const { return watcher_settings.text_start; }
    inline void set_watcher_text_start(bool text_start) { watcher_settings.text_start = text_start; }
    inline bool get_watcher_skip_logging() const { return watcher_settings.skip_logging; }
    inline void set_watcher_skip_logging(bool skip_logging) { watcher_settings.skip_logging = skip_logging; }
    inline const std::set<std::string>& get_watcher_disabled_features() const { return watcher_disabled_features; }
    inline bool watcher_status_disabled() const { return watcher_feature_disabled(watcher_waypoint_str); }
    inline bool watcher_noc_sanitize_disabled() const { return watcher_feature_disabled(watcher_noc_sanitize_str); }
    inline bool watcher_assert_disabled() const { return watcher_feature_disabled(watcher_assert_str); }
    inline bool watcher_pause_disabled() const { return watcher_feature_disabled(watcher_pause_str); }
    inline bool watcher_ring_buffer_disabled() const { return watcher_feature_disabled(watcher_ring_buffer_str); }
    inline bool watcher_stack_usage_disabled() const { return watcher_feature_disabled(watcher_stack_usage_str); }
    inline bool watcher_dispatch_disabled() const { return watcher_feature_disabled(watcher_dispatch_str); }

    // Info from inspector environment variables, setters included so that user
    // can override with a SW call.
    inline const std::filesystem::path& get_inspector_log_path() const { return inspector_settings.log_path; }
    inline bool get_inspector_enabled() const { return inspector_settings.enabled; }
    inline void set_inspector_enabled(bool enabled) { inspector_settings.enabled = enabled; }
    inline bool get_inspector_initialization_is_important() const { return inspector_settings.initialization_is_important; }
    inline void set_inspector_initialization_is_important(bool important) { inspector_settings.initialization_is_important = important; }
    inline bool get_inspector_warn_on_write_exceptions() const { return inspector_settings.warn_on_write_exceptions; }
    inline void set_inspector_warn_on_write_exceptions(bool warn) { inspector_settings.warn_on_write_exceptions = warn; }

    // Info from DPrint environment variables, setters included so that user can
    // override with a SW call.
    inline bool get_feature_enabled(RunTimeDebugFeatures feature) const { return feature_targets[feature].enabled; }
    inline void set_feature_enabled(RunTimeDebugFeatures feature, bool enabled) {
        feature_targets[feature].enabled = enabled;
    }
    // Note: dprint cores are logical
    inline const std::map<CoreType, std::vector<CoreCoord>>& get_feature_cores(RunTimeDebugFeatures feature) const {
        return feature_targets[feature].cores;
    }
    inline void set_feature_cores(RunTimeDebugFeatures feature, std::map<CoreType, std::vector<CoreCoord>> cores) {
        feature_targets[feature].cores = cores;
    }
    // An alternative to setting cores by range, a flag to enable all.
    inline void set_feature_all_cores(RunTimeDebugFeatures feature, CoreType core_type, int all_cores) {
        feature_targets[feature].all_cores[core_type] = all_cores;
    }
    inline int get_feature_all_cores(RunTimeDebugFeatures feature, CoreType core_type) const {
        return feature_targets[feature].all_cores.at(core_type);
    }
    // Note: core range is inclusive
    inline void set_feature_core_range(
        RunTimeDebugFeatures feature, CoreCoord start, CoreCoord end, CoreType core_type) {
        feature_targets[feature].cores[core_type] = std::vector<CoreCoord>();
        for (uint32_t x = start.x; x <= end.x; x++) {
            for (uint32_t y = start.y; y <= end.y; y++) {
                feature_targets[feature].cores[core_type].push_back({x, y});
            }
        }
    }
    inline const std::vector<int>& get_feature_chip_ids(RunTimeDebugFeatures feature) const {
        return feature_targets[feature].chip_ids;
    }
    inline void set_feature_chip_ids(RunTimeDebugFeatures feature, std::vector<int> chip_ids) {
        feature_targets[feature].chip_ids = chip_ids;
    }
    // An alternative to setting cores by range, a flag to enable all.
    inline void set_feature_all_chips(RunTimeDebugFeatures feature, bool all_chips) {
        feature_targets[feature].all_chips = all_chips;
    }
    inline bool get_feature_all_chips(RunTimeDebugFeatures feature) const { return feature_targets[feature].all_chips; }
    inline uint32_t get_feature_riscv_mask(RunTimeDebugFeatures feature) const {
        return feature_targets[feature].riscv_mask;
    }
    inline void set_feature_riscv_mask(RunTimeDebugFeatures feature, uint32_t riscv_mask) {
        feature_targets[feature].riscv_mask = riscv_mask;
    }
    inline std::string get_feature_file_name(RunTimeDebugFeatures feature) const {
        return feature_targets[feature].file_name;
    }
    inline void set_feature_file_name(RunTimeDebugFeatures feature, std::string file_name) {
        feature_targets[feature].file_name = file_name;
    }
    inline bool get_feature_one_file_per_risc(RunTimeDebugFeatures feature) const {
        return feature_targets[feature].one_file_per_risc;
    }
    inline void set_feature_one_file_per_risc(RunTimeDebugFeatures feature, bool one_file_per_risc) {
        feature_targets[feature].one_file_per_risc = one_file_per_risc;
    }
    inline bool get_feature_prepend_device_core_risc(RunTimeDebugFeatures feature) const {
        return feature_targets[feature].prepend_device_core_risc;
    }
    inline void set_feature_prepend_device_core_risc(RunTimeDebugFeatures feature, bool prepend_device_core_risc) {
        feature_targets[feature].prepend_device_core_risc = prepend_device_core_risc;
    }
    inline TargetSelection get_feature_targets(RunTimeDebugFeatures feature) const { return feature_targets[feature]; }
    inline void set_feature_targets(RunTimeDebugFeatures feature, TargetSelection targets) {
        feature_targets[feature] = targets;
    }

    inline bool get_record_noc_transfers() const { return record_noc_transfer_data; }
    inline void set_record_noc_transfers(bool val) { record_noc_transfer_data = val; }

    inline bool get_validate_kernel_binaries() const { return validate_kernel_binaries; }
    inline void set_validate_kernel_binaries(bool val) { validate_kernel_binaries = val; }

    // Returns the string representation for hash computation.
    inline std::string get_feature_hash_string(RunTimeDebugFeatures feature) const {
        switch (feature) {
            case RunTimeDebugFeatureDprint: return std::to_string(get_feature_enabled(feature));
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
    inline std::string get_compile_hash_string() const {
        std::string compile_hash_str = fmt::format("{}_{}", get_watcher_enabled(), get_kernels_early_return());
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
    inline void set_test_mode_enabled(bool enable) { test_mode_enabled = enable; }

    inline bool get_profiler_enabled() const { return profiler_enabled; }
    inline bool get_profiler_do_dispatch_cores() const { return profile_dispatch_cores; }
    inline bool get_profiler_sync_enabled() const { return profiler_sync_enabled; }
    inline bool get_profiler_tracy_mid_run_push() const { return profiler_mid_run_tracy_push; }
    inline bool get_profiler_buffer_usage_enabled() const { return profiler_buffer_usage_enabled; }
    inline bool get_profiler_noc_events_enabled() const { return profiler_noc_events_enabled; }
    inline std::string get_profiler_noc_events_report_path() const { return profiler_noc_events_report_path; }

    inline void set_kernels_nullified(bool v) { null_kernels = v; }
    inline bool get_kernels_nullified() const { return null_kernels; }

    inline void set_kernels_early_return(bool v) { kernels_early_return = v; }
    inline bool get_kernels_early_return() const { return kernels_early_return; }

    inline bool get_clear_l1() const { return clear_l1; }
    inline void set_clear_l1(bool clear) { clear_l1 = clear; }

    inline bool get_clear_dram() const { return clear_dram; }
    inline void set_clear_dram(bool clear) { clear_dram = clear; }

    inline bool get_skip_loading_fw() const { return skip_loading_fw; }
    inline bool get_skip_reset_cores_on_init() const { return skip_reset_cores_on_init; }

    // Whether to compile with -g to include DWARF debug info in the binary.
    inline bool get_riscv_debug_info_enabled() const { return riscv_debug_info_enabled; }
    inline void set_riscv_debug_info_enabled(bool enable) { riscv_debug_info_enabled = enable; }

    inline unsigned get_num_hw_cqs() const { return num_hw_cqs; }
    inline void set_num_hw_cqs(unsigned num) { num_hw_cqs = num; }

    inline bool get_fd_fabric() const { return fb_fabric_en; }

    inline uint32_t get_watcher_debug_delay() const { return watcher_debug_delay; }
    inline void set_watcher_debug_delay(uint32_t delay) { watcher_debug_delay = delay; }

    inline bool get_dispatch_data_collection_enabled() const { return enable_dispatch_data_collection; }
    inline void set_dispatch_data_collection_enabled(bool enable) { enable_dispatch_data_collection = enable; }

    inline bool get_hw_cache_invalidation_enabled() const { return this->enable_hw_cache_invalidation; }

    inline bool get_relaxed_memory_ordering_disabled() const { return this->disable_relaxed_memory_ordering; }
    inline bool get_gathering_enabled() const { return this->enable_gathering; }

    tt_metal::DispatchCoreConfig get_dispatch_core_config() const;

    inline bool get_skip_deleting_built_cache() const { return skip_deleting_built_cache; }

    inline bool get_simulator_enabled() const { return simulator_enabled; }
    inline const std::filesystem::path& get_simulator_path() const { return simulator_path; }

    inline bool get_erisc_iram_enabled() const { return erisc_iram_enabled; }

    inline bool get_skip_eth_cores_with_retrain() const { return skip_eth_cores_with_retrain; }

    inline uint32_t get_arc_debug_buffer_size() { return arc_debug_buffer_size; }
    inline void set_arc_debug_buffer_size(uint32_t size) { arc_debug_buffer_size = size; }

    inline bool get_disable_dma_ops() const { return disable_dma_ops; }
    inline void set_disable_dma_ops(bool disable) { disable_dma_ops = disable; }

private:
    // Helper functions to parse feature-specific environment vaiables.
    void ParseFeatureEnv(RunTimeDebugFeatures feature);
    void ParseFeatureCoreRange(RunTimeDebugFeatures feature, const std::string& env_var, CoreType core_type);
    void ParseFeatureChipIds(RunTimeDebugFeatures feature, const std::string& env_var);
    void ParseFeatureRiscvMask(RunTimeDebugFeatures feature, const std::string& env_var);
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
    std::set<std::string> watcher_disabled_features;
    bool watcher_feature_disabled(const std::string& name) const {
        return watcher_disabled_features.find(name) != watcher_disabled_features.end();
    }

    // Helper function to parse inspector-specific environment variables.
    void ParseInspectorEnv();
};

}  // namespace llrt

}  // namespace tt
