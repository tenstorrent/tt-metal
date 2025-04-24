// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/core.h>

#include <CLI/CLI.hpp>
#include <array>
#include <cstdio>  // for popen, pclose
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#include "autograd/auto_context.hpp"

struct board_entry {
    std::string pci_dev_id;
    std::string board_type;
    std::string device_series;
    std::string board_number;
};

std::string replace_all(const std::string& s, const std::string& old_val, const std::string& new_val) {
    std::string result = s;
    size_t pos = 0;
    while ((pos = result.find(old_val, pos)) != std::string::npos) {
        result.replace(pos, old_val.size(), new_val);
        pos += new_val.size();
    }
    return result;
}

std::vector<board_entry> parse_tt_smi_output(const std::string& command_output) {
    std::vector<board_entry> board_list;

    // Regex for ASCII pipes:
    static const std::regex row_pattern(R"(^\|?\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|?$)");

    std::stringstream stream(command_output);
    std::string line;
    while (std::getline(stream, line)) {
        line = replace_all(line, "\xE2\x94\x82", "|");

        // Now we have a line with normal ASCII pipes, e.g.:
        // "| 0          | wormhole   | n150 L        | 0100018511732049 |"

        std::smatch match;
        if (std::regex_match(line, match, row_pattern)) {
            // Skip header lines containing "Pci Dev ID"
            if (match[1].str().find("Pci Dev ID") != std::string::npos) {
                continue;
            }

            board_entry entry;
            entry.pci_dev_id = match[1];
            entry.board_type = match[2];
            entry.device_series = match[3];
            entry.board_number = match[4];

            board_list.push_back(std::move(entry));
        }
    }

    return board_list;
}

// Runs "tt-smi -ls" via popen, parses the result, and returns a list of boards.
std::vector<board_entry> get_tt_smi_boards() {
    std::array<char, 256> buffer{};
    std::string command_output;

    FILE* pipe_ptr = popen("tt-smi -ls", "r");
    if (!pipe_ptr) {
        throw std::runtime_error("Failed to run tt-smi -ls via popen()");
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe_ptr)) {
        fmt::println("Buffer: {}", buffer.data());
        command_output += buffer.data();
    }

    int return_code = pclose(pipe_ptr);
    if (return_code != 0) {
        std::cerr << "Warning: tt-smi -ls exited with code " << return_code << std::endl;
    }

    return parse_tt_smi_output(command_output);
}

void print_tt_smi() {
    try {
        std::vector<board_entry> boards = get_tt_smi_boards();
        fmt::print("Parsed boards:\n");
        for (const auto& b : boards) {
            fmt::print(
                "PCI Dev ID:      {}\n"
                "Board Type:      {}\n"
                "Device Series:   {}\n"
                "Board Number:    {}\n"
                "-----------------------------\n",
                b.pci_dev_id,
                b.board_type,
                b.device_series,
                b.board_number);
        }
    } catch (const std::exception& ex) {
        fmt::print(stderr, "Error: {}\n", ex.what());
    }
}

int main(int argc, char** argv) {
    ttml::autograd::ctx().init_mpi_context(argc, argv);
    CLI::App app{"NanoGPT Example"};
    argv = app.ensure_utf8(argv);

    bool print_tt_smi_output = false;

    // app.add_option("-c,--config", config_name, "Yaml Config name")->default_val(config_name);
    app.add_option("-t,--tt_smi", print_tt_smi_output, "print tt-smi on all hosts")->default_val(print_tt_smi_output);

    CLI11_PARSE(app, argc, argv);
    if (print_tt_smi_output) {
        print_tt_smi();
    }

    return 0;
}
