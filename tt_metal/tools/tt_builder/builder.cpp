// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include "tt_metal/common/utils.hpp"

using namespace tt;

int main(int argc, char* argv[]) {
    std::cout << "Running tt_builder and command" << std::endl;
    std::cout << "current path is ";
    std::string cmd = "pwd";
    std::string log_file = "./build/tools/output.log";
    if (!tt::utils::run_command(cmd, log_file, false)) {
        std::cout << "Error running command" << std::endl;
    }
    std::cout << "Finished running" << std::endl;
}
