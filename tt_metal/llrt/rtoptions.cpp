// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdlib.h>
#include <stdio.h>
#include <cstring>

#include "rtoptions.hpp"
#include "impl/debug/dprint_server.hpp"
#include "tools/profiler/profiler_state.hpp"

using std::vector;

namespace tt {

namespace llrt {

// Note: global initialization order is non-deterministic
// This is ok so long as this gets initialized before decisions are based on
// env state
RunTimeOptions OptionsG;

RunTimeOptions::RunTimeOptions() {
    if (const char* root_dir_ptr = std::getenv("TT_METAL_HOME")) {
        root_dir = string(root_dir_ptr) + "/";
    }

    build_map_enabled = (getenv("TT_METAL_KERNEL_MAP") != nullptr);

    ParseWatcherEnv();
    ParseDPrintEnv();

    // Test mode has no env var, default is disabled
    test_mode_enabled = false;

    profiler_enabled = false;
#if defined(PROFILER)
    const char *profiler_enabled_str = std::getenv("TT_METAL_DEVICE_PROFILER");
    if (profiler_enabled_str != nullptr && profiler_enabled_str[0] == '1') {
        profiler_enabled = true;
    }
#endif
    TT_FATAL(!(get_dprint_enabled() && get_profiler_enabled()), "Cannot enable both debug printing and profiling");

    null_kernels = (std::getenv("TT_METAL_NULL_KERNELS") != nullptr);

    clear_l1 = true;
    const char *clear_l1_enabled_str = std::getenv("TT_METAL_CLEAR_L1");
    if (clear_l1_enabled_str != nullptr) {
        if (clear_l1_enabled_str[0] == '0') clear_l1 = false;
        if (clear_l1_enabled_str[0] == '1') clear_l1 = true;
    }
}

const std::string& RunTimeOptions::get_root_dir() {
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

    // Auto unpause is for testing only, no env var.
    watcher_auto_unpause = false;
}

void RunTimeOptions::ParseDPrintEnv() {
    ParseDPrintCoreRange("TT_METAL_DPRINT_CORES", CoreType::WORKER);
    ParseDPrintCoreRange("TT_METAL_DPRINT_ETH_CORES", CoreType::ETH);
    ParseDPrintChipIds("TT_METAL_DPRINT_CHIPS");
    ParseDPrintRiscvMask("TT_METAL_DPRINT_RISCVS");
    ParseDPrintFileName("TT_METAL_DPRINT_FILE");

    // Set dprint enabled if the user asked for any dprint cores
    dprint_enabled = false;
    for (auto &core_type_and_all_flag : dprint_all_cores)
        if (core_type_and_all_flag.second)
            dprint_enabled = true;
    for (auto &core_type_and_cores : dprint_cores)
        if (core_type_and_cores.second.size() > 0)
            dprint_enabled = true;
};

void RunTimeOptions::ParseDPrintCoreRange(const char* env_var, CoreType core_type) {
    char *str = std::getenv(env_var);
    vector<CoreCoord> cores;

    // Check if "all" is specified, rather than a range of cores.
    if (str != nullptr && strcmp(str, "all") == 0) {
        dprint_all_cores[core_type] = true;
        return;
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
                    str = strchr(str+1, ',');
                    if (str != nullptr) str++;
                }
            }
        } else {
            TT_THROW("Invalid {}", env_var);
        }
    }

    // Set the core range
    dprint_cores[core_type] = cores;
}

void RunTimeOptions::ParseDPrintChipIds(const char* env_var) {
    vector<int> chips;
    char *env_var_str = std::getenv(env_var);

    // If the environment variable is not empty, parse it.
    while (env_var_str != nullptr) {
        uint32_t chip;
        if (sscanf(env_var_str, "%d", &chip) != 1) {
            TT_THROW("Invalid {}", env_var_str);
        }
        chips.push_back(chip);
        env_var_str = strchr(env_var_str, ',');
        if (env_var_str != nullptr) env_var_str++;
    }

    // Default is no chips are specified is chip 0.
    if (chips.size() == 0)
        chips.push_back(0);
    dprint_chip_ids = chips;
}

void RunTimeOptions::ParseDPrintRiscvMask(const char* env_var) {
    // Default is all RISCVs enabled for printing.
    uint32_t riscv_mask = DPRINT_RISCV_BR | DPRINT_RISCV_TR0 | DPRINT_RISCV_TR1 | DPRINT_RISCV_TR2 | DPRINT_RISCV_NC;
    char *env_var_str = std::getenv(env_var);
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
    dprint_riscv_mask = riscv_mask;
}

void RunTimeOptions::ParseDPrintFileName(const char* env_var) {
    char *env_var_str = std::getenv(env_var);
    dprint_file_name = (env_var_str != nullptr)? std::string(env_var_str) : "";
}

} // namespace llrt

} // namespace tt
