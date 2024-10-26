// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "json.hpp"
#include <fstream>
#include <regex>
#include <iomanip>
#include <string>
#include <filesystem>
#include <thread>

#include "tt_metal/impl/device/device.hpp"
#include "build_kernels_for_riscv/build_kernel_options.hpp"

#include "tt_gdb.hpp"

using json = nlohmann::json;


namespace tt_gdb {

// Globals
const std::map<string, std::function<void()>> documentation_map = {
    {"p", print_documentation},
    {"c", continue_documentation},
    {"q", quit_documentation},
    {"e", exit_tt_gdb_context_documentation},
    {"h", help_documentation},
    {"help", help_documentation},
};

const std::map<string, uint32_t> thread_type_to_sp_pointer_addr = {
    {"ncrisc", NCRISC_SP},
    {"trisc0", TRISC0_SP},
    {"trisc1", TRISC1_SP},
    {"trisc2", TRISC2_SP},
    {"brisc",  BRISC_SP},
};

const std::map<string, uint32_t> thread_type_to_bp_addr = {
    {"ncrisc", NCRISC_BREAKPOINT},
    {"trisc0", TRISC0_BREAKPOINT},
    {"trisc1", TRISC1_BREAKPOINT},
    {"trisc2", TRISC2_BREAKPOINT},
    {"brisc",  BRISC_BREAKPOINT},
};

// Regex matching
bool is_print_command(string input) {
    // The complex part '[a-z]+[a-z0-9_]*' just matches a string that starts with a letter and then
    // has any combination of letters, integers, and underscores
    std::regex self_regex("[ ]*p[ ]+[a-z_]+[a-z0-9_]*[ ]*",
            std::regex_constants::ECMAScript | std::regex_constants::icase);
    return std::regex_match(input, self_regex);
}

bool is_continue_command(string input) {
    std::regex self_regex("[ ]*c[ ]*",
            std::regex_constants::ECMAScript | std::regex_constants::icase);
    return std::regex_match(input, self_regex);
}

bool is_quit_command(string input) {
    std::regex self_regex("[ ]*q[ ]*",
            std::regex_constants::ECMAScript | std::regex_constants::icase);
    return std::regex_match(input, self_regex);
}

bool is_exit_tt_gdb_context_command(string input) {
    std::regex self_regex("[ ]*e[ ]*",
            std::regex_constants::ECMAScript | std::regex_constants::icase);
    return std::regex_match(input, self_regex);
}

bool is_help_command(string input) {
    std::regex self_regex("[ ]*h[ ]*|[ ]*help[ ]*",
            std::regex_constants::ECMAScript | std::regex_constants::icase);
    return std::regex_match(input, self_regex);
}

bool is_help_documentation_command(string input) {
    std::regex self_regex("[ ]*h[ ]+[a-z]+|[ ]*help[ ]+[a-z]+",
            std::regex_constants::ECMAScript | std::regex_constants::icase);
    return std::regex_match(input, self_regex);
}

// Debugger apis
inline void prompt(string &input) {
    std::cout << "(tt_gdb) ";
    std::getline(std::cin, input);
}

inline string get_second_token(string &input) {
    /*
        Given an input of the form "<SPACE>*<TOKEN><SPACE>*<TOKEN>", gets the second token
    */
    int start = 0;
    int end = input.size() - 1;

    while (input.at(start) == ' ') {
        start++;
    }

    while (input.at(start) != ' ') {
        start++;
    }

    while (input.at(start) == ' ') {
        start++;
    }

    while (input.at(end) == ' ') {
        end--;
    }

    return input.substr(start, end - start + 1);
}

void print_cmd(uint32_t chip_id, CoreCoord core, string variable, string thread_type, string op) {
    string debug_file_path = tt::get_kernel_compile_outpath(chip_id) + op + "/" + thread_type + "/" + thread_type + "_debug_dwarf_info.json";
    const string cmd = "python3 tt_metal/tools/tt_gdb/pydwarf2.py " + thread_type + " " + op;
    int ret = system(cmd.c_str());

    // Error
    if (ret) {
        std::cout << "Could not find variable " << variable << std::endl;
        return;
    }
    std::ifstream debug_file(debug_file_path, std::ifstream::binary);


    json debug_data = json::parse(debug_file);

    int offset_from_frame_pointer;
    int variable_offset;
    map<string, int> variable_offset_info = debug_data["variable_offset_info"];

    try {
        offset_from_frame_pointer = debug_data["offset_from_frame_pointer"];
    } catch (std::invalid_argument& e) {
        std::cout << "Could not find offset from frame pointer" << std::endl;
        return;
    }

    try {
        variable_offset = variable_offset_info[variable];
    } catch (std::invalid_argument& e){
        std::cout << "Could not find variable" << std::endl;
        return;
    } catch (std::exception& e) {
        std::cout << "Could not find address of variable" << std::endl;
        return;
    }

    std::uint32_t debug_addr = thread_type_to_sp_pointer_addr.at(thread_type);

    uint32_t sp_pointer = tt::llrt::read_hex_vec_from_core(chip_id, core, debug_addr, sizeof(uint32_t)).at(0);

    uint32_t val = tt::llrt::read_hex_vec_from_core(chip_id, core, sp_pointer + offset_from_frame_pointer + variable_offset, sizeof(uint32_t)).at(0);
    std::cout << val << std::endl;
}

void continue_cmd(uint32_t chip_id, CoreCoord core, string thread_type) {

    const std::vector<uint32_t> breakpoint_flag = {0};

    tt::llrt::write_hex_vec_to_core(chip_id, core, breakpoint_flag, thread_type_to_bp_addr.at(thread_type));
    // std::cout << "Continue command issued for core " << core.x << ", " << core.y << " for thread " << thread_type << std::endl;
}

void quit_cmd() {
    std::filesystem::remove("core_debug_info.json");
    exit(0);
}

// Documentation
inline void print_documentation() {
    // TODO(agrebenisan): Eventually would like this to be able to handle expressions
    std::cout << "p variable" << std::endl;
    std::cout << "\tPrint the value of a variable" << std::endl;
}

inline void continue_documentation() {
    std::cout << "c" << std::endl;
    std::cout << "\tResume the program" << std::endl;
}

inline void quit_documentation() {
    std::cout << "q" << std::endl;
    std::cout << "\tQuit the program" << std::endl;
}

inline void exit_tt_gdb_context_documentation() {
    std::cout << "e" << std::endl;
    std::cout << "\tGo back to debugger grid" << std::endl;
}

inline void help_documentation() {
    std::cout << "h(elp)" << std::endl;
    std::cout << "\tWithout argument, print the list of available commands." << std::endl;
    std::cout << "\tWith a command name as argument, print help about that command." << std::endl;
}

inline void display_documentation(string input) {
    documentation_map.at(input)();
}

void nicely_display_commands() {

    int num_printed = 0;
    for (const auto &kv_pair: documentation_map) {
        string cmd = kv_pair.first;
        std::cout << cmd;
        std::cout << std::setw(9 - cmd.size());
        num_printed++;
        if ((num_printed % 8) == 0) {
            std::cout << std::endl;
        }
    }

    std::cout << std::endl;
}

void help_cmd() {
    string doc_cmd = "Documented commands (type help <topic>):";
    std::cout << doc_cmd << std::endl;
    for (int i = 0; i < doc_cmd.size(); i++) {
        std::cout << "=";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    nicely_display_commands();
}

string disaggregate_python_core_map_info(const PythonCoreMapInfo& info) {
    std::stringstream ss;

    ss << "--cores_with_breakpoint ";

    for (CoreCoord core: info.breakpoint_cores) {
        ss << std::to_string(core.y) << "-" << std::to_string(core.x) << " ";
    }

    ss << "--breakpoint_lines ";
    for (map<string, int> calling_risc_to_breakpoint_line: info.breakpoint_lines) {
        ss << "'{";
        int idx = 0;
        for (const auto& [calling_risc, breakpoint_line]: calling_risc_to_breakpoint_line) {
            idx++;

            if (idx < calling_risc_to_breakpoint_line.size()) {
                ss << "\"" << calling_risc << "\": " << std::to_string(breakpoint_line) << ", ";
            } else {
                ss << "\"" << calling_risc << "\": " << std::to_string(breakpoint_line);
            }
        }
        ss << "}' ";
    }

    ss << "--ops ";

    for (string op: info.op_names) {
        ss << op << " ";
    }

    if (info.reenter) {
        ss << "--reenter ";
    }

    ss << "--start_index " << info.current_core.x << " " << info.current_core.y << " ";
    ss << "--current_risc " << info.current_risc << " " ;

    return ss.str();
}


void breakpoint_subroutine(int chip_id, const CoreCoord &core, string thread_type, string op) {
    auto run_cmd = [&chip_id, &core, &thread_type, &op](string input) {
        bool exit = false;

        if (is_print_command(input)) {
            string variable = get_second_token(input);
            print_cmd(chip_id, core, variable, thread_type, op);
        } else if (is_continue_command(input)) {
            continue_cmd(chip_id, core, thread_type);
            exit = true;
        } else if (is_quit_command(input)) {
            quit_cmd();
        } else if (is_exit_tt_gdb_context_command(input)) {
            exit = true;
        } else if (is_help_command(input)) {
            help_cmd();
        } else if (is_help_documentation_command(input)) {
            string documentation_to_show = get_second_token(input);
            display_documentation(documentation_to_show);
        } else {
            std::cout << "invalid command" << std::endl;
        }

        return exit;
    };

    string input = "";
    bool exit = false;

    while (not exit) {
        prompt(input);
        exit = run_cmd(input);
    }
}

void launch_core_map(PythonCoreMapInfo info) {
    /*
        This function launches python core map, a user interface that allows
        you to choose which core and which risc you want to debug. It also
        shows certain debug info
    */

    const string cmd = "python3 tt_metal/tools/tt_gdb/tt_gdb_table.py " + disaggregate_python_core_map_info(info);

    std::cout << "Launched python core view with this cmd: " << cmd << std::endl;
    int ret = system(cmd.c_str());
    TT_ASSERT(ret == 0, "tt_gdb_table.py must have 0 exit code");
}

void tt_gdb_(int chip_id, const vector<CoreCoord> cores, vector<string> ops) {

    const vector<std::tuple<string, uint32_t, uint32_t>> breakpoint_addresses = {
        std::tuple("ncrisc", NCRISC_BREAKPOINT, NCRISC_BP_LNUM),
        std::tuple("trisc0", TRISC0_BREAKPOINT, TRISC0_BP_LNUM),
        std::tuple("trisc1", TRISC1_BREAKPOINT, TRISC1_BP_LNUM),
        std::tuple("trisc2", TRISC2_BREAKPOINT, TRISC2_BP_LNUM),
        std::tuple("brisc",  BRISC_BREAKPOINT,  BRISC_BP_LNUM)
    };

    std::filesystem::remove("core_debug_info.json");

    // This program loops indefinitely, however should be launched as a detached thread so that its resources are freed after the main thread terminates
    while (true) {

        vector<CoreCoord> breakpoint_cores;
        vector<map<string, int>> breakpoint_lines;
        std::ifstream core_debug_info("core_debug_info.json", std::ifstream::binary);

        CoreCoord current_core = {0, 0};
        bool reenter = false;
        string current_risc = "trisc0";

        if (core_debug_info.good()) {
            json debug_data = json::parse(core_debug_info);

            // Get state from last time we exited python program
            reenter = debug_data["reenter"];
            current_core = {debug_data["current_core_x"], debug_data["current_core_y"]};
            current_risc = debug_data["current_risc"];
        }

        for (const auto &core: cores) {

            bool at_least_one_breakpoint = false;
            map<string, int> breakpoint_lines_for_core;
            for (const auto& [calling_risc, breakpoint_address, breakpoint_line_address]: breakpoint_addresses) {
                uint32_t breakpoint_flag = tt::llrt::read_hex_vec_from_core(chip_id, core, breakpoint_address, sizeof(uint32_t)).at(0);
                at_least_one_breakpoint |= (breakpoint_flag == 1);

                if (breakpoint_flag == 1) {
                    uint32_t breakpoint_line = tt::llrt::read_hex_vec_from_core(chip_id, core, breakpoint_line_address, sizeof(uint32_t)).at(0);
                    breakpoint_lines_for_core.emplace(calling_risc, breakpoint_line);
                }
            }

            if (at_least_one_breakpoint) {
                breakpoint_cores.push_back(core);
                breakpoint_lines.push_back(breakpoint_lines_for_core);
            }
        }

        // Render python UI
        if (not breakpoint_cores.empty()) {
            TT_ASSERT(not breakpoint_lines.empty(), "If breakpoint_cores is not empty, breakpoint_lines cannot be either");
            PythonCoreMapInfo info = {
                // This info is provided by C++
                breakpoint_cores,
                breakpoint_lines,
                ops,

                // This info is state received from python from last exit
                current_core,
                reenter,
                current_risc

            };
            launch_core_map(info);

            // Read python output
            std::ifstream core_debug_info("core_debug_info.json", std::ifstream::binary);
            json debug_data = json::parse(core_debug_info);

            if (debug_data.contains("exit")) {
                std::filesystem::remove("core_debug_info.json");
                exit(0);
            }

            breakpoint_subroutine(
                chip_id,
                {
                    debug_data["current_core_x"],
                    debug_data["current_core_y"],
                },
                debug_data["current_risc"],
                debug_data["op"]
            );
        }

        sleep(2);
    }
}

void tt_gdb(int chip_id, const vector<CoreCoord> worker_cores, vector<string> ops) {
    // Makes this thread completely independent from the rest of execution. Once the main thread finishes, the debugger's resources are freed

    std::thread debug_server(tt_gdb_, chip_id, worker_cores, ops);
    debug_server.detach();
}

} // end namespace tt_gdb

namespace tt {
namespace tt_metal {

void tt_gdb(Device* device, int chip_id, const vector<CoreCoord> logical_cores, vector<string> ops) {
    vector<CoreCoord> worker_cores;

    for (const auto& logical_core: logical_cores) {
        worker_cores.push_back(device->worker_core_from_logical_core(logical_core));
    }

    tt_gdb::tt_gdb(chip_id, worker_cores, ops);
}

} // end namespace tt_metal
} // end namespace tt
