// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rtoptions.hpp"

#include <ctype.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "assert.hpp"
#include <umd/device/tt_core_coordinates.h>

using std::vector;

namespace tt {

namespace llrt {

const char* RunTimeDebugFeatureNames[RunTimeDebugFeatureCount] = {
    "DPRINT",
    "READ_DEBUG_DELAY",
    "WRITE_DEBUG_DELAY",
    "ATOMIC_DEBUG_DELAY",
    "DISABLE_L1_DATA_CACHE",
};

const char* RunTimeDebugClassNames[RunTimeDebugClassCount] = {"N/A", "worker", "dispatch", "all"};

static const char* TT_METAL_HOME_ENV_VAR = "TT_METAL_HOME";
static const char* TT_METAL_KERNEL_PATH_ENV_VAR = "TT_METAL_KERNEL_PATH";
// Set this var to change the cache dir.
static const char* TT_METAL_CACHE_ENV_VAR = "TT_METAL_CACHE";
// Used for demonstration purposes and will be removed in the future.
static const char* TT_METAL_FD_FABRIC_DEMO = "TT_METAL_FD_FABRIC";
static const char* TT_METAL_VISIBLE_DEVICES_ENV_VAR = "TT_METAL_VISIBLE_DEVICES";

RunTimeOptions::RunTimeOptions() {
    const char* root_dir_str = std::getenv(TT_METAL_HOME_ENV_VAR);
    if (root_dir_str != nullptr) {
        this->is_root_dir_env_var_set = true;
        this->root_dir = std::string(root_dir_str) + "/";
    }

    // Check if user has specified a cache path.
    const char* cache_dir_str = std::getenv(TT_METAL_CACHE_ENV_VAR);
    if (cache_dir_str != nullptr) {
        this->is_cache_dir_env_var_set = true;
        this->cache_dir_ = std::string(cache_dir_str) + "/tt-metal-cache/";
    }

    const char* kernel_dir_str = std::getenv(TT_METAL_KERNEL_PATH_ENV_VAR);
    if (kernel_dir_str != nullptr) {
        this->is_kernel_dir_env_var_set = true;
        this->kernel_dir = std::string(kernel_dir_str) + "/";
    }
    this->system_kernel_dir = "/usr/share/tenstorrent/kernels/";

    const char* visible_devices_str = std::getenv(TT_METAL_VISIBLE_DEVICES_ENV_VAR);
    if (visible_devices_str != nullptr) {
        this->is_visible_devices_env_var_set = true;
        std::string devices_string(visible_devices_str);
        size_t pos = 0;
        while ((pos = devices_string.find(',')) != std::string::npos) {
            std::string device_str = devices_string.substr(0, pos);
            this->visible_devices.push_back(std::stoi(device_str));
            devices_string.erase(0, pos + 1);
        }
        if (!devices_string.empty()) {
            this->visible_devices.push_back(std::stoi(devices_string));
        }
    }

    build_map_enabled = (getenv("TT_METAL_KERNEL_MAP") != nullptr);

    ParseWatcherEnv();
    ParseInspectorEnv();

    for (int i = 0; i < RunTimeDebugFeatureCount; i++) {
        ParseFeatureEnv((RunTimeDebugFeatures)i);
    }

    // Test mode has no env var, default is disabled
    test_mode_enabled = false;

    profiler_enabled = false;
    profile_dispatch_cores = false;
    profiler_sync_enabled = false;
    profiler_mid_run_tracy_push = false;
    profiler_buffer_usage_enabled = false;
#if defined(TRACY_ENABLE)
    const char* profiler_enabled_str = std::getenv("TT_METAL_DEVICE_PROFILER");
    if (profiler_enabled_str != nullptr && profiler_enabled_str[0] == '1') {
        profiler_enabled = true;
        const char* profile_dispatch_str = std::getenv("TT_METAL_DEVICE_PROFILER_DISPATCH");
        if (profile_dispatch_str != nullptr && profile_dispatch_str[0] == '1') {
            profile_dispatch_cores = true;
        }
        const char* profiler_sync_enabled_str = std::getenv("TT_METAL_PROFILER_SYNC");
        if (profiler_enabled && profiler_sync_enabled_str != nullptr && profiler_sync_enabled_str[0] == '1') {
            profiler_sync_enabled = true;
        }
        const char* profiler_force_push_enabled_str = std::getenv("TT_METAL_TRACY_MID_RUN_PUSH");
        if (profiler_enabled && profiler_force_push_enabled_str != nullptr &&
            profiler_force_push_enabled_str[0] == '1') {
            profiler_mid_run_tracy_push = true;
        }
    }

    const char *profiler_noc_events_str = std::getenv("TT_METAL_DEVICE_PROFILER_NOC_EVENTS");
    if (profiler_noc_events_str != nullptr && profiler_noc_events_str[0] == '1') {
        profiler_enabled = true;
        profiler_noc_events_enabled = true;
    }

    const char *profiler_noc_events_report_path_str = std::getenv("TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH");
    if (profiler_noc_events_report_path_str != nullptr) {
        profiler_noc_events_report_path = profiler_noc_events_report_path_str;
    }

    const char *profile_buffer_usage_str = std::getenv("TT_METAL_MEM_PROFILER");
    if (profile_buffer_usage_str != nullptr && profile_buffer_usage_str[0] == '1') {
        profiler_buffer_usage_enabled = true;
    }
#endif
    TT_FATAL(
        !(get_feature_enabled(RunTimeDebugFeatureDprint) && get_profiler_enabled()),
        "Cannot enable both debug printing and profiling");

    null_kernels = (std::getenv("TT_METAL_NULL_KERNELS") != nullptr);

    kernels_early_return = (std::getenv("TT_METAL_KERNELS_EARLY_RETURN") != nullptr);

    this->clear_l1 = false;
    const char* clear_l1_enabled_str = std::getenv("TT_METAL_CLEAR_L1");
    if (clear_l1_enabled_str != nullptr && clear_l1_enabled_str[0] == '1') {
        this->clear_l1 = true;
    }

    this->clear_dram = false;
    const char* clear_dram_enabled_str = std::getenv("TT_METAL_CLEAR_DRAM");
    if (clear_dram_enabled_str != nullptr && clear_dram_enabled_str[0] == '1') {
        this->clear_dram = true;
    }

    const char* skip_eth_cores_with_retrain_str = std::getenv("TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN");
    if (skip_eth_cores_with_retrain_str != nullptr) {
        if (skip_eth_cores_with_retrain_str[0] == '0') {
            skip_eth_cores_with_retrain = false;
        }
        if (skip_eth_cores_with_retrain_str[0] == '1') {
            skip_eth_cores_with_retrain = true;
        }
    }

    const char* riscv_debug_info_enabled_str = std::getenv("TT_METAL_RISCV_DEBUG_INFO");
    bool enable_riscv_debug_info = get_inspector_enabled();
    if (riscv_debug_info_enabled_str != nullptr) {
        enable_riscv_debug_info = true;
        if (strcmp(riscv_debug_info_enabled_str, "0") == 0) {
            enable_riscv_debug_info = false;
        }
    }
    set_riscv_debug_info_enabled(enable_riscv_debug_info);

    const char* validate_kernel_binaries = std::getenv("TT_METAL_VALIDATE_PROGRAM_BINARIES");
    set_validate_kernel_binaries(validate_kernel_binaries != nullptr && validate_kernel_binaries[0] == '1');

    const char* num_cqs = getenv("TT_METAL_GTEST_NUM_HW_CQS");
    if (num_cqs != nullptr) {
        try {
            set_num_hw_cqs(std::stoi(num_cqs));
        } catch (const std::invalid_argument& ia) {
            TT_THROW("Invalid TT_METAL_GTEST_NUM_HW_CQS: {}", num_cqs);
        }
    }

    using_slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr;

    const char* dispatch_data_collection_str = std::getenv("TT_METAL_DISPATCH_DATA_COLLECTION");
    if (dispatch_data_collection_str != nullptr) {
        enable_dispatch_data_collection = true;
    }

    if (getenv("TT_METAL_GTEST_ETH_DISPATCH")) {
        this->dispatch_core_type = tt_metal::DispatchCoreType::ETH;
    }

    if (getenv("TT_METAL_SKIP_LOADING_FW")) {
        this->skip_loading_fw = true;
    }

    if (getenv("TT_METAL_SKIP_DELETING_BUILT_CACHE")) {
        this->skip_deleting_built_cache = true;
    }

    if (getenv("TT_METAL_ENABLE_HW_CACHE_INVALIDATION")) {
        this->enable_hw_cache_invalidation = true;
    }

    if (std::getenv("TT_METAL_SIMULATOR")) {
        this->simulator_enabled = true;
        this->simulator_path = std::getenv("TT_METAL_SIMULATOR");
    }

    if (auto str = getenv("TT_METAL_ENABLE_ERISC_IRAM")) {
        bool disabled = strcmp(str, "0") == 0;
        this->erisc_iram_enabled = !disabled;
        this->erisc_iram_enabled_env_var = !disabled;
    }
    this->fast_dispatch = (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr);

    if (getenv("TT_METAL_DISABLE_RELAXED_MEM_ORDERING")) {
        this->disable_relaxed_memory_ordering = true;
    }

    if (getenv("TT_METAL_ENABLE_GATHERING")) {
        this->enable_gathering = true;
    }

    const char* arc_debug_enabled_str = std::getenv("TT_METAL_ARC_DEBUG_BUFFER_SIZE");
    if (arc_debug_enabled_str != nullptr) {
        sscanf(arc_debug_enabled_str, "%u", &arc_debug_buffer_size);
    }

    const char* disable_dma_ops_str = std::getenv("TT_METAL_DISABLE_DMA_OPS");
    if (disable_dma_ops_str != nullptr) {
        if (disable_dma_ops_str[0] == '1') {
            this->disable_dma_ops = true;
        }
    }

    if (getenv("TT_METAL_FORCE_REINIT")) {
        force_context_reinit = true;
    }
}

const std::string& RunTimeOptions::get_root_dir() const {
    if (!this->is_root_dir_specified()) {
        TT_THROW("Env var {} is not set.", TT_METAL_HOME_ENV_VAR);
    }

    return root_dir;
}

const std::string& RunTimeOptions::get_cache_dir() const {
    if (!this->is_cache_dir_specified()) {
        TT_THROW("Env var {} is not set.", TT_METAL_CACHE_ENV_VAR);
    }
    return this->cache_dir_;
}

const std::string& RunTimeOptions::get_kernel_dir() const {
    if (!this->is_kernel_dir_specified()) {
        TT_THROW("Env var {} is not set.", TT_METAL_KERNEL_PATH_ENV_VAR);
    }

    return this->kernel_dir;
}

const std::string& RunTimeOptions::get_system_kernel_dir() const { return this->system_kernel_dir; }

void RunTimeOptions::ParseWatcherEnv() {
    const char* watcher_enable_str = getenv("TT_METAL_WATCHER");
    if (watcher_enable_str != nullptr) {
        int sleep_val = 0;
        sscanf(watcher_enable_str, "%d", &sleep_val);
        if (strstr(watcher_enable_str, "ms") == nullptr) {
            sleep_val *= 1000;
        }
        watcher_settings.enabled = true;
        watcher_settings.interval_ms = sleep_val;
    }

    watcher_settings.dump_all = (getenv("TT_METAL_WATCHER_DUMP_ALL") != nullptr);
    watcher_settings.append = (getenv("TT_METAL_WATCHER_APPEND") != nullptr);
    watcher_settings.noinline = (getenv("TT_METAL_WATCHER_NOINLINE") != nullptr);
    watcher_settings.phys_coords = (getenv("TT_METAL_WATCHER_PHYS_COORDS") != nullptr);
    watcher_settings.text_start = (getenv("TT_METAL_WATCHER_TEXT_START") != nullptr);
    watcher_settings.skip_logging = (getenv("TT_METAL_WATCHER_SKIP_LOGGING") != nullptr);
    // Auto unpause is for testing only, no env var.
    watcher_settings.auto_unpause = false;

    // Any watcher features to disabled based on env var.
    std::set all_features = {
        watcher_waypoint_str,
        watcher_noc_sanitize_str,
        watcher_assert_str,
        watcher_pause_str,
        watcher_ring_buffer_str,
        watcher_stack_usage_str,
        watcher_dispatch_str};
    for (const std::string& feature : all_features) {
        std::string env_var("TT_METAL_WATCHER_DISABLE_");
        env_var += feature;
        if (getenv(env_var.c_str()) != nullptr) {
            watcher_disabled_features.insert(feature);
        }
    }

    const char* watcher_debug_delay_str = getenv("TT_METAL_WATCHER_DEBUG_DELAY");
    if (watcher_debug_delay_str != nullptr) {
        sscanf(watcher_debug_delay_str, "%u", &watcher_debug_delay);
        // Assert watcher is also enabled (TT_METAL_WATCHER=1)
        TT_ASSERT(watcher_settings.enabled, "TT_METAL_WATCHER_DEBUG_DELAY requires TT_METAL_WATCHER");
        // Assert TT_METAL_WATCHER_DISABLE_NOC_SANITIZE is either not set or set to 0
        TT_ASSERT(
            watcher_disabled_features.find(watcher_noc_sanitize_str) == watcher_disabled_features.end(),
            "TT_METAL_WATCHER_DEBUG_DELAY requires TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=0");
    }
}

void RunTimeOptions::ParseInspectorEnv() {
    const char* inspector_enable_str = getenv("TT_METAL_INSPECTOR");
    if (inspector_enable_str != nullptr) {
        inspector_settings.enabled = true;
        if (strcmp(inspector_enable_str, "0") == 0) {
            inspector_settings.enabled = false;
        }
    }

    const char* inspector_log_path_str = getenv("TT_METAL_INSPECTOR_LOG_PATH");
    if (inspector_log_path_str != nullptr) {
        inspector_settings.log_path = std::filesystem::path(inspector_log_path_str);
    } else {
        inspector_settings.log_path = std::filesystem::path(get_root_dir()) / "generated/inspector";
    }

    const char* inspector_initialization_is_important_str = getenv("TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT");
    if (inspector_initialization_is_important_str != nullptr) {
        inspector_settings.initialization_is_important = true;
        if (strcmp(inspector_initialization_is_important_str, "0") == 0) {
            inspector_settings.initialization_is_important = false;
        }
    }

    const char* inspector_warn_on_write_exceptions_str = getenv("TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS");
    if (inspector_warn_on_write_exceptions_str != nullptr) {
        inspector_settings.warn_on_write_exceptions = true;
        if (strcmp(inspector_warn_on_write_exceptions_str, "0") == 0) {
            inspector_settings.warn_on_write_exceptions = false;
        }
    }
}

void RunTimeOptions::ParseFeatureEnv(RunTimeDebugFeatures feature) {
    std::string feature_env_prefix("TT_METAL_");
    feature_env_prefix += RunTimeDebugFeatureNames[feature];

    ParseFeatureCoreRange(feature, feature_env_prefix + "_CORES", CoreType::WORKER);
    ParseFeatureCoreRange(feature, feature_env_prefix + "_ETH_CORES", CoreType::ETH);
    ParseFeatureChipIds(feature, feature_env_prefix + "_CHIPS");
    ParseFeatureRiscvMask(feature, feature_env_prefix + "_RISCVS");
    ParseFeatureFileName(feature, feature_env_prefix + "_FILE");
    ParseFeatureOneFilePerRisc(feature, feature_env_prefix + "_ONE_FILE_PER_RISC");
    ParseFeaturePrependDeviceCoreRisc(feature, feature_env_prefix + "_PREPEND_DEVICE_CORE_RISC");

    // Set feature enabled if the user asked for any feature cores
    feature_targets[feature].enabled = false;
    for (auto& core_type_and_all_flag : feature_targets[feature].all_cores) {
        if (core_type_and_all_flag.second != RunTimeDebugClassNoneSpecified) {
            feature_targets[feature].enabled = true;
        }
    }
    for (auto& core_type_and_cores : feature_targets[feature].cores) {
        if (core_type_and_cores.second.size() > 0) {
            feature_targets[feature].enabled = true;
        }
    }

    const char* print_noc_xfers = std::getenv("TT_METAL_RECORD_NOC_TRANSFER_DATA");
    if (print_noc_xfers != nullptr) {
        record_noc_transfer_data = true;
    }
};

void RunTimeOptions::ParseFeatureCoreRange(
    RunTimeDebugFeatures feature, const std::string& env_var, CoreType core_type) {
    char* str = std::getenv(env_var.c_str());
    std::vector<CoreCoord> cores;

    // Check if "all" is specified, rather than a range of cores.
    feature_targets[feature].all_cores[core_type] = RunTimeDebugClassNoneSpecified;
    if (str != nullptr) {
        for (int idx = 0; idx < RunTimeDebugClassCount; idx++) {
            if (strcmp(str, RunTimeDebugClassNames[idx]) == 0) {
                feature_targets[feature].all_cores[core_type] = idx;
                return;
            }
        }
    }
    if (str != nullptr) {
        if (isdigit(str[0])) {
            // Assume this is a single core
            uint32_t x, y;
            if (sscanf(str, "%d,%d", &x, &y) != 2) {
                TT_THROW("Invalid {}", env_var);
            }
            cores.push_back({x, y});
        } else if (str[0] == '(') {
            if (strchr(str, '-')) {
                // Assume this is a range
                CoreCoord start, end;
                if (sscanf(str, "(%zu,%zu)", &start.x, &start.y) != 2) {
                    TT_THROW("Invalid {}", env_var);
                }
                str = strchr(str, '-');
                if (sscanf(str, "-(%zu,%zu)", &end.x, &end.y) != 2) {
                    TT_THROW("Invalid {}", env_var);
                }
                for (uint32_t x = start.x; x <= end.x; x++) {
                    for (uint32_t y = start.y; y <= end.y; y++) {
                        cores.push_back({x, y});
                    }
                }
            } else {
                // Assume this is a list of coordinates (maybe just one)
                while (str != nullptr) {
                    uint32_t x, y;
                    if (sscanf(str, "(%d,%d)", &x, &y) != 2) {
                        TT_THROW("Invalid {}", env_var);
                    }
                    cores.push_back({x, y});
                    str = strchr(str, ',');
                    str = strchr(str + 1, ',');
                    if (str != nullptr) {
                        str++;
                    }
                }
            }
        } else {
            TT_THROW("Invalid {}", env_var);
        }
    }

    // Set the core range
    feature_targets[feature].cores[core_type] = cores;
}

void RunTimeOptions::ParseFeatureChipIds(RunTimeDebugFeatures feature, const std::string& env_var) {
    std::vector<int> chips;
    char* env_var_str = std::getenv(env_var.c_str());

    // If the environment variable is not empty, parse it.
    while (env_var_str != nullptr) {
        // Can also have "all"
        if (strcmp(env_var_str, "all") == 0) {
            feature_targets[feature].all_chips = true;
            break;
        }
        uint32_t chip;
        if (sscanf(env_var_str, "%d", &chip) != 1) {
            TT_THROW("Invalid {}", env_var_str);
        }
        chips.push_back(chip);
        env_var_str = strchr(env_var_str, ',');
        if (env_var_str != nullptr) {
            env_var_str++;
        }
    }

    // Default is no chips are specified is all
    if (chips.size() == 0) {
        feature_targets[feature].all_chips = true;
    }
    feature_targets[feature].chip_ids = chips;
}

void RunTimeOptions::ParseFeatureRiscvMask(RunTimeDebugFeatures feature, const std::string& env_var) {
    uint32_t riscv_mask = 0;
    char* env_var_str = std::getenv(env_var.c_str());

    if (env_var_str != nullptr) {
        if (strstr(env_var_str, "BR")) {
            riscv_mask |= RISCV_BR;
        }
        if (strstr(env_var_str, "NC")) {
            riscv_mask |= RISCV_NC;
        }
        if (strstr(env_var_str, "TR0")) {
            riscv_mask |= RISCV_TR0;
        }
        if (strstr(env_var_str, "TR1")) {
            riscv_mask |= RISCV_TR1;
        }
        if (strstr(env_var_str, "TR2")) {
            riscv_mask |= RISCV_TR2;
        }
        if (strstr(env_var_str, "TR*")) {
            riscv_mask |= (RISCV_TR0 | RISCV_TR1 | RISCV_TR2);
        }
        if (strstr(env_var_str, "ER0")) {
            riscv_mask |= RISCV_ER0;
        }
        if (strstr(env_var_str, "ER1")) {
            riscv_mask |= RISCV_ER1;
        }
        if (strstr(env_var_str, "ER*")) {
            riscv_mask |= (RISCV_ER0 | RISCV_ER1);
        }
        if (riscv_mask == 0) {
            TT_THROW(
                "Invalid RISC selection: \"{}\". Valid values are BR,NC,TR0,TR1,TR2,TR*,ER0,ER1,ER*.", env_var_str);
        }
    } else {
        // Default is all RISCVs enabled.
        bool default_disabled = (feature == RunTimeDebugFeatures::RunTimeDebugFeatureDisableL1DataCache);
        riscv_mask =
            default_disabled ? 0 : (RISCV_ER0 | RISCV_ER1 | RISCV_BR | RISCV_TR0 | RISCV_TR1 | RISCV_TR2 | RISCV_NC);
    }

    feature_targets[feature].riscv_mask = riscv_mask;
}

void RunTimeOptions::ParseFeatureFileName(RunTimeDebugFeatures feature, const std::string& env_var) {
    char* env_var_str = std::getenv(env_var.c_str());
    feature_targets[feature].file_name = (env_var_str != nullptr) ? std::string(env_var_str) : "";
}

void RunTimeOptions::ParseFeatureOneFilePerRisc(RunTimeDebugFeatures feature, const std::string& env_var) {
    char* env_var_str = std::getenv(env_var.c_str());
    feature_targets[feature].one_file_per_risc = (env_var_str != nullptr);
}

void RunTimeOptions::ParseFeaturePrependDeviceCoreRisc(RunTimeDebugFeatures feature, const std::string &env_var) {
    char *env_var_str = std::getenv(env_var.c_str());
    feature_targets[feature].prepend_device_core_risc =
        (env_var_str != nullptr) ? (strcmp(env_var_str, "1") == 0) : true;
}

// Can't create a DispatchCoreConfig as part of the RTOptions constructor because the DispatchCoreConfig constructor
// depends on RTOptions settings.
tt_metal::DispatchCoreConfig RunTimeOptions::get_dispatch_core_config() const {
    tt_metal::DispatchCoreConfig dispatch_core_config = tt_metal::DispatchCoreConfig{};
    dispatch_core_config.set_dispatch_core_type(this->dispatch_core_type);
    return dispatch_core_config;
}

}  // namespace llrt

}  // namespace tt
