// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include "impl/context/metal_context.hpp"
#include <device.hpp>

namespace tt_gdb {
// Debugger info for UI
struct PythonCoreMapInfo {
    const vector<CoreCoord> breakpoint_cores;
    const vector<map<string, int>> breakpoint_lines;
    const vector<string> op_names;

    const CoreCoord current_core = {0, 0};
    const bool reenter = false;
    const string current_risc = "trisc0";
};

// Regex
bool is_print_command(std::string input);
bool is_continue_command(std::string input);
bool is_quit_command(std::string input);
bool is_exit_tt_gdb_context_command(std::string input);
bool is_help_command(std::string input);
bool is_help_documentation_command(std::string input);

// Debugger general apis
inline void prompt(std::string& input);
inline std::string get_second_token(std::string& input);

// Commands
void print_cmd(uint32_t chip_id, CoreCoord core, std::string variable, std::string thread_type, std::string op);
void continue_cmd(uint32_t chip_id, CoreCoord core, std::string thread_type);
void quit_cmd();
void help_cmd();

// Documentation commands
inline void print_documentation();
inline void continue_documentation();
inline void quit_documentation();
inline void exit_tt_gdb_context_documentation();
inline void help_documentation();
inline void display_documentation(std::string input);
void nicely_display_commands();

// Debugger driver and python UI
void launch_core_map(PythonCoreMapInfo info);
void breakpoint_subroutine(int chip_id, const CoreCoord& core, std::string thread_type, std::string op);

void tt_gdb(int chip_id, const vector<CoreCoord> worker_cores, vector<string> ops);
}  // end namespace tt_gdb

namespace tt {
namespace tt_metal {

void tt_gdb(IDevice* device, int chip_id, const vector<CoreCoord> logical_cores, vector<string> ops);

}
}  // namespace tt
