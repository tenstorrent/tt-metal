// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/core.h>
#include <mpi.h>

#include <array>
#include <cstdio>  // for popen, pclose
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
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

int main(int argc, char** argv) {
    std::cout << "Starting MPI init" << std::endl;
    MPI_Init(&argc, &argv);
    std::cout << "MPI init complete" << std::endl;
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME] = {};
    int name_len = 0;
    MPI_Get_processor_name(processor_name, &name_len);

    std::cout << "Hello world from processor " << processor_name << ", rank " << world_rank << " out of " << world_size
              << " processors" << std::endl;
    try {
        std::vector<board_entry> boards = get_tt_smi_boards();
        std::cout << "Parsed boards:\n";
        for (const auto& b : boards) {
            std::cout << "PCI Dev ID:      " << b.pci_dev_id << "\n"
                      << "Board Type:      " << b.board_type << "\n"
                      << "Device Series:   " << b.device_series << "\n"
                      << "Board Number:    " << b.board_number << "\n"
                      << "-----------------------------\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }
    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}
