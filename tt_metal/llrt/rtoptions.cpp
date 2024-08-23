// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rtoptions.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>

#include "impl/debug/dprint_server.hpp"
#include "tools/profiler/profiler_state.hpp"

using std::vector;

namespace tt {

namespace llrt {

const char *RunTimeDebugFeatureNames[RunTimeDebugFeatureCount] = {
    "DPRINT",
    "READ_DEBUG_DELAY",
    "WRITE_DEBUG_DELAY",
    "ATOMIC_DEBUG_DELAY",
};

const char *RunTimeDebugClassNames[RunTimeDebugClassCount] = {
    "N/A",
    "worker",
    "dispatch",
    "all"
};

// Note: global initialization order is non-deterministic
// This is ok so long as this gets initialized before decisions are based on
// env state
RunTimeOptions OptionsG;

RunTimeOptions::RunTimeOptions() {
    if (const char *root_dir_ptr = std::getenv("TT_METAL_HOME")) {
        root_dir = std::string(root_dir_ptr) + "/";
    }

    build_map_enabled = (getenv("TT_METAL_KERNEL_MAP") != nullptr);

    ParseWatcherEnv();

    for (int i = 0; i < RunTimeDebugFeatureCount; i++) {
        ParseFeatureEnv((RunTimeDebugFeatures)i);
    }

    // Test mode has no env var, default is disabled
    test_mode_enabled = false;

    profiler_enabled = false;
    profile_dispatch_cores = false;
    profiler_sync_enabled = false;
#if defined(TRACY_ENABLE)
    const char *profiler_enabled_str = std::getenv("TT_METAL_DEVICE_PROFILER");
    if (profiler_enabled_str != nullptr && profiler_enabled_str[0] == '1') {
        profiler_enabled = true;
        const char *profile_dispatch_str = std::getenv("TT_METAL_DEVICE_PROFILER_DISPATCH");
        if (profile_dispatch_str != nullptr && profile_dispatch_str[0] == '1') {
            profile_dispatch_cores = true;
        }
        const char *profiler_sync_enabled_str = std::getenv("TT_METAL_PROFILER_SYNC");
        if (profiler_enabled && profiler_sync_enabled_str != nullptr && profiler_sync_enabled_str[0] == '1') {
            profiler_sync_enabled = true;
        }
    }
#endif
    TT_FATAL(
        !(get_feature_enabled(RunTimeDebugFeatureDprint) && get_profiler_enabled()),
        "Cannot enable both debug printing and profiling");

    null_kernels = (std::getenv("TT_METAL_NULL_KERNELS") != nullptr);

    clear_l1 = false;
    const char *clear_l1_enabled_str = std::getenv("TT_METAL_CLEAR_L1");
    if (clear_l1_enabled_str != nullptr) {
        if (clear_l1_enabled_str[0] == '0')
            clear_l1 = false;
        if (clear_l1_enabled_str[0] == '1')
            clear_l1 = true;
    }

    const char *riscv_debug_info_enabled_str = std::getenv("TT_METAL_RISCV_DEBUG_INFO");
    set_riscv_debug_info_enabled(riscv_debug_info_enabled_str != nullptr);

    const char *validate_kernel_binaries = std::getenv("TT_METAL_VALIDATE_PROGRAM_BINARIES");
    set_validate_kernel_binaries(validate_kernel_binaries != nullptr && validate_kernel_binaries[0] == '1');

    const char *num_cqs = getenv("TT_METAL_GTEST_NUM_HW_CQS");
    if (num_cqs != nullptr) {
        try {
            set_num_hw_cqs(std::stoi(num_cqs));
        } catch (const std::invalid_argument& ia) {
            TT_THROW("Invalid TT_METAL_GTEST_NUM_HW_CQS: {}", num_cqs);
        }
    }

    const char *dispatch_data_collection_str = std::getenv("TT_METAL_DISPATCH_DATA_COLLECTION");
    if (dispatch_data_collection_str != nullptr) {
        enable_dispatch_data_collection = true;
    }

    if (getenv("TT_METAL_GTEST_ETH_DISPATCH")) {
        this->dispatch_core_type = tt_metal::DispatchCoreType::ETH;
    }

    if (getenv("TT_METAL_SKIP_LOADING_FW")) {
        this->skip_loading_fw = true;
    }
}

const std::string &RunTimeOptions::get_root_dir() {
    if (root_dir == "") {
        TT_THROW("Env var " + std::string("TT_METAL_HOME") + " is not set.");
    }

    return root_dir;
}

void RunTimeOptions::ParseWatcherEnv() {
    watcher_interval_ms = 0;
    const char *watcher_enable_str = getenv("TT_METAL_WATCHER");
    watcher_enabled = (watcher_enable_str != nullptr);
    if (watcher_enabled) {
        int sleep_val = 0;
        sscanf(watcher_enable_str, "%d", &sleep_val);
        if (strstr(watcher_enable_str, "ms") == nullptr) {
            sleep_val *= 1000;
        }
        watcher_interval_ms = sleep_val;
    }

    const char *watcher_dump_all_str = getenv("TT_METAL_WATCHER_DUMP_ALL");
    watcher_dump_all = (watcher_dump_all_str != nullptr);

    const char *watcher_append_str = getenv("TT_METAL_WATCHER_APPEND");
    watcher_append = (watcher_append_str != nullptr);

    const char *watcher_noinline_str = getenv("TT_METAL_WATCHER_NOINLINE");
    watcher_noinline = (watcher_noinline_str != nullptr);

    // Auto unpause is for testing only, no env var.
    watcher_auto_unpause = false;

    // Any watcher features to disabled based on env var.
    std::set all_features = {
        watcher_status_str,
        watcher_noc_sanitize_str,
        watcher_assert_str,
        watcher_pause_str,
        watcher_ring_buffer_str,
        watcher_stack_usage_str};
    for (std::string feature : all_features) {
        std::string env_var("TT_METAL_WATCHER_DISABLE_");
        env_var += feature;
        if (getenv(env_var.c_str()) != nullptr) {
            watcher_disabled_features.insert(feature);
        }
    }

    const char *watcher_debug_delay_str = getenv("TT_METAL_WATCHER_DEBUG_DELAY");
    if (watcher_debug_delay_str != nullptr) {
        sscanf(watcher_debug_delay_str, "%u", &watcher_debug_delay);
        // Assert watcher is also enabled (TT_METAL_WATCHER=1)
        TT_ASSERT(watcher_enabled, "TT_METAL_WATCHER_DEBUG_DELAY requires TT_METAL_WATCHER");
        // Assert TT_METAL_WATCHER_DISABLE_NOC_SANITIZE is either not set or set to 0
        TT_ASSERT(
            watcher_disabled_features.find(watcher_noc_sanitize_str) == watcher_disabled_features.end(),
            "TT_METAL_WATCHER_DEBUG_DELAY requires TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=0");
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

    // Set feature enabled if the user asked for any feature cores
    feature_targets[feature].enabled = false;
    for (auto &core_type_and_all_flag : feature_targets[feature].all_cores)
        if (core_type_and_all_flag.second != RunTimeDebugClassNoneSpecified)
            feature_targets[feature].enabled = true;
    for (auto &core_type_and_cores : feature_targets[feature].cores)
        if (core_type_and_cores.second.size() > 0)
            feature_targets[feature].enabled = true;

    const char *print_noc_xfers = std::getenv("TT_METAL_DPRINT_NOC_TRANSFER_DATA");
    if (print_noc_xfers != nullptr)
        dprint_noc_transfer_data = true;
};

void RunTimeOptions::ParseFeatureCoreRange(
    RunTimeDebugFeatures feature, const std::string &env_var, CoreType core_type) {
    char *str = std::getenv(env_var.c_str());
    vector<CoreCoord> cores;

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
                    if (str != nullptr)
                        str++;
                }
            }
        } else {
            TT_THROW("Invalid {}", env_var);
        }
    }

    // Set the core range
    feature_targets[feature].cores[core_type] = cores;
}

void RunTimeOptions::ParseFeatureChipIds(RunTimeDebugFeatures feature, const std::string &env_var) {
    vector<int> chips;
    char *env_var_str = std::getenv(env_var.c_str());

    // If the environment variable is not empty, parse it.
    while (env_var_str != nullptr) {
        uint32_t chip;
        if (sscanf(env_var_str, "%d", &chip) != 1) {
            TT_THROW("Invalid {}", env_var_str);
        }
        chips.push_back(chip);
        env_var_str = strchr(env_var_str, ',');
        if (env_var_str != nullptr)
            env_var_str++;
    }

    // Default is no chips are specified is chip 0.
    if (chips.size() == 0)
        chips.push_back(0);
    feature_targets[feature].chip_ids = chips;
}

void RunTimeOptions::ParseFeatureRiscvMask(RunTimeDebugFeatures feature, const std::string &env_var) {
    // Default is all RISCVs enabled for printing.
    uint32_t riscv_mask = DPRINT_RISCV_BR | DPRINT_RISCV_TR0 | DPRINT_RISCV_TR1 | DPRINT_RISCV_TR2 | DPRINT_RISCV_NC;
    char *env_var_str = std::getenv(env_var.c_str());
    if (env_var_str != nullptr) {
        if (strcmp(env_var_str, "BR") == 0) {
            riscv_mask = DPRINT_RISCV_BR;
        } else if (strcmp(env_var_str, "NC") == 0) {
            riscv_mask = DPRINT_RISCV_NC;
        } else if (strcmp(env_var_str, "TR0") == 0) {
            riscv_mask = DPRINT_RISCV_TR0;
        } else if (strcmp(env_var_str, "TR1") == 0) {
            riscv_mask = DPRINT_RISCV_TR1;
        } else if (strcmp(env_var_str, "TR2") == 0) {
            riscv_mask = DPRINT_RISCV_TR2;
        } else {
            TT_THROW("Invalid TT_DEBUG_PRINT_RISCV");
        }
    }
    feature_targets[feature].riscv_mask = riscv_mask;
}

void RunTimeOptions::ParseFeatureFileName(RunTimeDebugFeatures feature, const std::string &env_var) {
    char *env_var_str = std::getenv(env_var.c_str());
    feature_targets[feature].file_name = (env_var_str != nullptr) ? std::string(env_var_str) : "";
}

}  // namespace llrt

}  // namespace tt
