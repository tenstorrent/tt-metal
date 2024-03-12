// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Run Time Options
//
// Reads env vars and sets up a global object which contains run time
// configuration options (such as debug logging)
//

#pragma once

#include <vector>
#include <cstdint>
#include "tt_metal/common/core_coord.h"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h" // For CoreType

namespace tt {

namespace llrt {

class RunTimeOptions {
    std::string root_dir;

    bool build_map_enabled = false;

    bool watcher_enabled = false;
    int watcher_interval_ms;
    bool watcher_dump_all = false;
    bool watcher_append = false;
    bool watcher_auto_unpause = false;

    std::map<CoreType, std::vector<CoreCoord>> dprint_cores;
    std::map<CoreType, bool> dprint_all_cores;
    bool dprint_enabled;
    std::vector<int> dprint_chip_ids;
    bool dprint_all_chips = false;
    uint32_t dprint_riscv_mask = 0;
    std::string dprint_file_name;

    bool test_mode_enabled = false;

    bool profiler_enabled = false;

    bool null_kernels = false;

    bool clear_l1 = false;

   public:
    RunTimeOptions();

    const std::string& get_root_dir();

    inline bool get_build_map_enabled() { return build_map_enabled; }

    // Info from watcher environment variables, setters included so that user
    // can override with a SW call.
    inline bool get_watcher_enabled()                 { return watcher_enabled; }
    inline void set_watcher_enabled(bool enabled)     { watcher_enabled = enabled; }
    inline int get_watcher_interval()                 { return watcher_interval_ms; }
    inline void set_watcher_interval(int interval_ms) { watcher_interval_ms = interval_ms; }
    inline int get_watcher_dump_all()                 { return watcher_dump_all; }
    inline void set_watcher_dump_all(bool dump_all)   { watcher_dump_all = dump_all; }
    inline int get_watcher_append()                   { return watcher_append; }
    inline void set_watcher_append(bool append)       { watcher_append = append; }
    inline int get_watcher_auto_unpause()             { return watcher_auto_unpause; }
    inline void set_watcher_auto_unpause(bool auto_unpause) { watcher_auto_unpause = auto_unpause; }

    // Info from DPrint environment variables, setters included so that user can
    // override with a SW call.
    inline bool get_dprint_enabled() { return dprint_enabled; }
    inline void set_dprint_enabled(bool enable) { dprint_enabled = enable; }
    // Note: dprint cores are logical
    inline std::map<CoreType, std::vector<CoreCoord>>& get_dprint_cores() {
        return dprint_cores;
    }
    inline void set_dprint_cores(std::map<CoreType, std::vector<CoreCoord>> cores) {
        dprint_cores = cores;
    }
    // An alternative to setting cores by range, a flag to enable all.
    inline void set_dprint_all_cores(CoreType core_type, bool all_cores) {
        dprint_all_cores[core_type] = all_cores;
    }
    inline bool get_dprint_all_cores(CoreType core_type) { return dprint_all_cores[core_type]; }
    // Note: core range is inclusive
    inline void set_dprint_core_range(CoreCoord start, CoreCoord end, CoreType core_type) {
        dprint_cores[core_type] = std::vector<CoreCoord>();
        for (uint32_t x = start.x; x <= end.x; x++) {
            for (uint32_t y = start.y; y <= end.y; y++) {
                dprint_cores[core_type].push_back({x, y});
            }
        }
    }
    inline std::vector<int>& get_dprint_chip_ids() { return dprint_chip_ids; }
    inline void set_dprint_chip_ids(std::vector<int> chip_ids) {
        dprint_chip_ids = chip_ids;
    }
    // An alternative to setting cores by range, a flag to enable all.
    inline void set_dprint_all_chips(bool all_chips) {
        dprint_all_chips = all_chips;
    }
    inline bool get_dprint_all_chips() { return dprint_all_chips; }
    inline uint32_t get_dprint_riscv_mask() { return dprint_riscv_mask; }
    inline void set_dprint_riscv_mask(uint32_t riscv_mask) {
        dprint_riscv_mask = riscv_mask;
    }
    inline std::string get_dprint_file_name() { return dprint_file_name; }
    inline void set_dprint_file_name(std::string file_name) {
        dprint_file_name = file_name;
    }

    // Used for both watcher and dprint servers, this dev option (no corresponding env var) sets
    // whether to catch exceptions (test mode = true) coming from debug servers or to throw them
    // (test mode = false). We need to catch for gtesting, since an unhandled exception will kill
    // the gtest (and can't catch an exception from the server thread in main thread), but by
    // default we should throw so that the user can see the exception as soon as it happens.
    bool get_test_mode_enabled() { return test_mode_enabled; }
    inline void set_test_mode_enabled(bool enable) { test_mode_enabled = enable; }

    inline bool get_profiler_enabled() { return profiler_enabled; }

    inline void set_kernels_nullified(bool v) { null_kernels = v; }
    inline bool get_kernels_nullified() { return null_kernels; }

    inline bool get_clear_l1() { return clear_l1; }

private:
    // Helper functions to parse DPrint-specific environment vaiables.
    void ParseDPrintEnv();
    void ParseDPrintCoreRange(const char* env_var, CoreType core_type);
    void ParseDPrintChipIds(const char* env_var);
    void ParseDPrintRiscvMask(const char* env_var);
    void ParseDPrintFileName(const char* env_var);

    // Helper function to parse watcher-specific environment variables.
    void ParseWatcherEnv();
};


extern RunTimeOptions OptionsG;

} // namespace llrt

} // namespace tt
