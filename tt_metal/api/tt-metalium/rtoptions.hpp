// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Run Time Options
//
// Reads env vars and sets up a global object which contains run time
// configuration options (such as debug logging)
//

#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "core_coord.hpp"
#include "dispatch_core_manager.hpp"
#include "umd/device/tt_soc_descriptor.h"  // For CoreType

namespace tt {

namespace llrt {

static inline const char* get_core_type_name(CoreType ct) {
    switch (ct) {
        case CoreType::ARC: return "ARC";
        case CoreType::DRAM: return "DRAM";
        case CoreType::ETH: return "ethernet";
        case CoreType::PCIE: return "PCIE";
        case CoreType::WORKER: return "worker";
        case CoreType::HARVESTED: return "harvested";
        case CoreType::ROUTER_ONLY: return "router_only";
        default: return "UNKNOWN";
    }
}

// TODO: This should come from the HAL
enum DebugHartFlags : unsigned int {
    RISCV_NC = 1,
    RISCV_TR0 = 2,
    RISCV_TR1 = 4,
    RISCV_TR2 = 8,
    RISCV_BR = 16,
    RISCV_ER = 32
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

class RunTimeOptions {
    bool is_root_dir_env_var_set = false;
    std::string root_dir;

    bool is_kernel_dir_env_var_set = false;
    std::string kernel_dir;

    bool build_map_enabled = false;

    bool watcher_enabled = false;
    int watcher_interval_ms;
    bool watcher_dump_all = false;
    bool watcher_append = false;
    bool watcher_auto_unpause = false;
    bool watcher_noinline = false;
    bool record_noc_transfer_data = false;

    TargetSelection feature_targets[RunTimeDebugFeatureCount];

    bool test_mode_enabled = false;

    bool profiler_enabled = false;
    bool profile_dispatch_cores = false;
    bool profiler_sync_enabled = false;
    bool profiler_buffer_usage_enabled = false;

    bool null_kernels = false;

    bool clear_l1 = false;

    bool skip_loading_fw = false;
    bool skip_reset_cores_on_init = false;

    bool riscv_debug_info_enabled = false;
    uint32_t watcher_debug_delay = 0;

    bool validate_kernel_binaries = false;
    unsigned num_hw_cqs = 1;

    bool enable_dispatch_data_collection = false;

    // HW can clear Blackhole's L1 data cache psuedo-randomly once every 128 transactions
    // This option will enable this feature to help flush out whether there is a missing cache invalidation
    bool enable_hw_cache_invalidation = false;

    tt_metal::DispatchCoreConfig dispatch_core_config = tt_metal::DispatchCoreConfig{};

    bool skip_deleting_built_cache = false;

    RunTimeOptions();

public:
    static RunTimeOptions& get_instance() {
        static RunTimeOptions instance;
        return instance;
    }

    RunTimeOptions(const RunTimeOptions&) = delete;
    RunTimeOptions& operator=(const RunTimeOptions&) = delete;

    inline bool is_root_dir_specified() const { return this->is_root_dir_env_var_set; }
    const std::string& get_root_dir();

    inline bool is_kernel_dir_specified() const { return this->is_kernel_dir_env_var_set; }
    const std::string& get_kernel_dir() const;

    inline bool get_build_map_enabled() { return build_map_enabled; }

    // Info from watcher environment variables, setters included so that user
    // can override with a SW call.
    inline bool get_watcher_enabled() { return watcher_enabled; }
    inline void set_watcher_enabled(bool enabled) { watcher_enabled = enabled; }
    inline int get_watcher_interval() { return watcher_interval_ms; }
    inline void set_watcher_interval(int interval_ms) { watcher_interval_ms = interval_ms; }
    inline int get_watcher_dump_all() { return watcher_dump_all; }
    inline void set_watcher_dump_all(bool dump_all) { watcher_dump_all = dump_all; }
    inline int get_watcher_append() { return watcher_append; }
    inline void set_watcher_append(bool append) { watcher_append = append; }
    inline int get_watcher_auto_unpause() { return watcher_auto_unpause; }
    inline void set_watcher_auto_unpause(bool auto_unpause) { watcher_auto_unpause = auto_unpause; }
    inline int get_watcher_noinline() { return watcher_noinline; }
    inline void set_watcher_noinline(bool noinline) { watcher_noinline = noinline; }
    inline std::set<std::string>& get_watcher_disabled_features() { return watcher_disabled_features; }
    inline bool watcher_status_disabled() { return watcher_feature_disabled(watcher_waypoint_str); }
    inline bool watcher_noc_sanitize_disabled() { return watcher_feature_disabled(watcher_noc_sanitize_str); }
    inline bool watcher_assert_disabled() { return watcher_feature_disabled(watcher_assert_str); }
    inline bool watcher_pause_disabled() { return watcher_feature_disabled(watcher_pause_str); }
    inline bool watcher_ring_buffer_disabled() { return watcher_feature_disabled(watcher_ring_buffer_str); }
    inline bool watcher_stack_usage_disabled() { return watcher_feature_disabled(watcher_stack_usage_str); }
    inline bool watcher_dispatch_disabled() { return watcher_feature_disabled(watcher_dispatch_str); }

    // Info from DPrint environment variables, setters included so that user can
    // override with a SW call.
    inline bool get_feature_enabled(RunTimeDebugFeatures feature) { return feature_targets[feature].enabled; }
    inline void set_feature_enabled(RunTimeDebugFeatures feature, bool enabled) {
        feature_targets[feature].enabled = enabled;
    }
    // Note: dprint cores are logical
    inline std::map<CoreType, std::vector<CoreCoord>>& get_feature_cores(RunTimeDebugFeatures feature) {
        return feature_targets[feature].cores;
    }
    inline void set_feature_cores(RunTimeDebugFeatures feature, std::map<CoreType, std::vector<CoreCoord>> cores) {
        feature_targets[feature].cores = cores;
    }
    // An alternative to setting cores by range, a flag to enable all.
    inline void set_feature_all_cores(RunTimeDebugFeatures feature, CoreType core_type, int all_cores) {
        feature_targets[feature].all_cores[core_type] = all_cores;
    }
    inline int get_feature_all_cores(RunTimeDebugFeatures feature, CoreType core_type) {
        return feature_targets[feature].all_cores[core_type];
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
    inline std::vector<int>& get_feature_chip_ids(RunTimeDebugFeatures feature) {
        return feature_targets[feature].chip_ids;
    }
    inline void set_feature_chip_ids(RunTimeDebugFeatures feature, std::vector<int> chip_ids) {
        feature_targets[feature].chip_ids = chip_ids;
    }
    // An alternative to setting cores by range, a flag to enable all.
    inline void set_feature_all_chips(RunTimeDebugFeatures feature, bool all_chips) {
        feature_targets[feature].all_chips = all_chips;
    }
    inline bool get_feature_all_chips(RunTimeDebugFeatures feature) { return feature_targets[feature].all_chips; }
    inline uint32_t get_feature_riscv_mask(RunTimeDebugFeatures feature) { return feature_targets[feature].riscv_mask; }
    inline void set_feature_riscv_mask(RunTimeDebugFeatures feature, uint32_t riscv_mask) {
        feature_targets[feature].riscv_mask = riscv_mask;
    }
    inline std::string get_feature_file_name(RunTimeDebugFeatures feature) {
        return feature_targets[feature].file_name;
    }
    inline void set_feature_file_name(RunTimeDebugFeatures feature, std::string file_name) {
        feature_targets[feature].file_name = file_name;
    }
    inline bool get_feature_one_file_per_risc(RunTimeDebugFeatures feature) {
        return feature_targets[feature].one_file_per_risc;
    }
    inline void set_feature_one_file_per_risc(RunTimeDebugFeatures feature, bool one_file_per_risc) {
        feature_targets[feature].one_file_per_risc = one_file_per_risc;
    }
    inline bool get_feature_prepend_device_core_risc(RunTimeDebugFeatures feature) {
        return feature_targets[feature].prepend_device_core_risc;
    }
    inline void set_feature_prepend_device_core_risc(RunTimeDebugFeatures feature, bool prepend_device_core_risc) {
        feature_targets[feature].prepend_device_core_risc = prepend_device_core_risc;
    }
    inline TargetSelection get_feature_targets(RunTimeDebugFeatures feature) { return feature_targets[feature]; }
    inline void set_feature_targets(RunTimeDebugFeatures feature, TargetSelection targets) {
        feature_targets[feature] = targets;
    }

    inline bool get_record_noc_transfers() { return record_noc_transfer_data; }
    inline void set_record_noc_transfers(bool val) { record_noc_transfer_data = val; }

    inline bool get_validate_kernel_binaries() { return validate_kernel_binaries; }
    inline void set_validate_kernel_binaries(bool val) { validate_kernel_binaries = val; }

    // Returns the string representation for hash computation.
    inline std::string get_feature_hash_string(RunTimeDebugFeatures feature) {
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

    // Used for both watcher and dprint servers, this dev option (no corresponding env var) sets
    // whether to catch exceptions (test mode = true) coming from debug servers or to throw them
    // (test mode = false). We need to catch for gtesting, since an unhandled exception will kill
    // the gtest (and can't catch an exception from the server thread in main thread), but by
    // default we should throw so that the user can see the exception as soon as it happens.
    bool get_test_mode_enabled() { return test_mode_enabled; }
    inline void set_test_mode_enabled(bool enable) { test_mode_enabled = enable; }

    inline bool get_profiler_enabled() { return profiler_enabled; }
    inline bool get_profiler_do_dispatch_cores() { return profile_dispatch_cores; }
    inline bool get_profiler_sync_enabled() { return profiler_sync_enabled; }
    inline bool get_profiler_buffer_usage_enabled() { return profiler_buffer_usage_enabled; }

    inline void set_kernels_nullified(bool v) { null_kernels = v; }
    inline bool get_kernels_nullified() { return null_kernels; }

    inline bool get_clear_l1() { return clear_l1; }
    inline void set_clear_l1(bool clear) { clear_l1 = clear; }

    inline bool get_skip_loading_fw() { return skip_loading_fw; }
    inline bool get_skip_reset_cores_on_init() { return skip_reset_cores_on_init; }

    // Whether to compile with -g to include DWARF debug info in the binary.
    inline bool get_riscv_debug_info_enabled() { return riscv_debug_info_enabled; }
    inline void set_riscv_debug_info_enabled(bool enable) { riscv_debug_info_enabled = enable; }

    inline unsigned get_num_hw_cqs() { return num_hw_cqs; }
    inline void set_num_hw_cqs(unsigned num) { num_hw_cqs = num; }

    inline uint32_t get_watcher_debug_delay() { return watcher_debug_delay; }
    inline void set_watcher_debug_delay(uint32_t delay) { watcher_debug_delay = delay; }

    inline bool get_dispatch_data_collection_enabled() { return enable_dispatch_data_collection; }
    inline void set_dispatch_data_collection_enabled(bool enable) { enable_dispatch_data_collection = enable; }

    inline bool get_hw_cache_invalidation_enabled() const { return this->enable_hw_cache_invalidation; }

    inline tt_metal::DispatchCoreConfig get_dispatch_core_config() { return dispatch_core_config; }

    inline bool get_skip_deleting_built_cache() { return skip_deleting_built_cache; }

private:
    // Helper functions to parse feature-specific environment vaiables.
    void ParseFeatureEnv(RunTimeDebugFeatures feature);
    void ParseFeatureCoreRange(RunTimeDebugFeatures feature, const std::string &env_var, CoreType core_type);
    void ParseFeatureChipIds(RunTimeDebugFeatures feature, const std::string &env_var);
    void ParseFeatureRiscvMask(RunTimeDebugFeatures feature, const std::string &env_var);
    void ParseFeatureFileName(RunTimeDebugFeatures feature, const std::string &env_var);
    void ParseFeatureOneFilePerRisc(RunTimeDebugFeatures feature, const std::string &env_var);
    void ParseFeaturePrependDeviceCoreRisc(RunTimeDebugFeatures feature, const std::string &env_var);

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
    bool watcher_feature_disabled(const std::string& name) {
        return watcher_disabled_features.find(name) != watcher_disabled_features.end();
    }
};

}  // namespace llrt

}  // namespace tt
