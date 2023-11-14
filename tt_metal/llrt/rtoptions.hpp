/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// Run Time Options
//
// Reads env vars and sets up a global object which contains run time
// configuration options (such as debug logging)
//

#pragma once

#include <vector>
#include <cstdint>
#include "tt_metal/common/core_coord.h"

namespace tt {

namespace llrt {

class RunTimeOptions {
    int watcher_interval_ms;
    bool watcher_dump_all;

    std::vector<CoreCoord> dprint_core_range;
    std::vector<int> dprint_chip_ids;
    uint32_t dprint_riscv_mask;
    std::string dprint_file_name;

public:
    RunTimeOptions();

    inline bool get_watcher_enabled() { return watcher_interval_ms != 0; }
    inline int get_watcher_interval() { return watcher_interval_ms; }
    inline int get_watcher_dump_all() { return watcher_dump_all; }

    // Info from DPrint environment variables, setters included so that user can
    // override with a SW call.
    inline bool get_dprint_enabled() { return dprint_core_range.size() != 0; }
    inline std::vector<CoreCoord>& get_dprint_core_range() {
        return dprint_core_range;
    }
    inline void set_dprint_core_range(std::vector<CoreCoord> core_range) {
        dprint_core_range = core_range;
    }
    inline std::vector<int>& get_dprint_chip_ids() { return dprint_chip_ids; }
    inline void set_dprint_chip_ids(std::vector<int> chip_ids) {
        dprint_chip_ids = chip_ids;
    }
    inline uint32_t get_dprint_riscv_mask() { return dprint_riscv_mask; }
    inline void set_dprint_riscv_mask(uint32_t riscv_mask) {
        dprint_riscv_mask = riscv_mask;
    }
    inline std::string get_dprint_file_name() { return dprint_file_name; }
    inline void set_dprint_file_name(std::string file_name) {
        dprint_file_name = file_name;
    }

private:
    // Helper functions to parse DPrint-specific environment vaiables.
    void ParseDPrintEnv();
    void ParseDPrintCoreRange(const char* env_var);
    void ParseDPrintChipIds(const char* env_var);
    void ParseDPrintRiscvMask(const char* env_var);
    void ParseDPrintFileName(const char* env_var);
};


extern RunTimeOptions OptionsG;

} // namespace llrt

} // namespace tt
