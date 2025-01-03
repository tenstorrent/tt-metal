// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include "builder.hpp"
#include "tt_metal/common/logger.hpp"
using namespace std;

void print_usage(const char* exec_name) {
    cout << "Usage: " << exec_name << " [OPTION]" << endl;
    cout << "\t-h, --help           : Display this message." << endl;
    cout << "\t--build-firmware     : Build Firmware." << endl;
    cout << "\t--build-dispatch     : Build Dispatch." << endl;
    cout << "\t--build-all          : Build Firmware and Dispatch." << endl;
    cout << "\t--set-output-path    : Set output path for binaries." << endl;
}

int main(int argc, char* argv[]) {
    tt::log_info("Running tt_builder ");

    try {
        bool run_firmware_build = false, run_dispatch_build = false, set_built_path = false;
        string output_path;
        for (int idx = 1; idx < argc; idx++) {
            string input(argv[idx]);
            if (input == "-h" || input == "--help") {
                print_usage(argv[0]);
                return 0;
            } else if (input.rfind("--build-firmware", 0) == 0) {
                run_firmware_build = true;
            } else if (input.rfind("--build-dispatch", 0) == 0) {
                run_dispatch_build = true;
            } else if (input.rfind("--build-all", 0) == 0) {
                run_firmware_build = true;
                run_dispatch_build = true;
            } else if (input.rfind("--set-output-path=", 0) == 0) {
                set_built_path = true;
                output_path = input.substr(input.find("=") + 1);
            } else {
                print_usage(argv[0]);
                throw(runtime_error("Unrecognized option"));
            }
        }

        if (!run_firmware_build && !run_dispatch_build) {
            print_usage(argv[0]);
            throw(runtime_error("No build selected to run"));
        }

        tt::tt_metal::BuilderTool builder;
        if (set_built_path) {
            builder.set_built_path(output_path);
        }

        if (run_firmware_build) {
            builder.build_firmware();
        }
        if (run_dispatch_build) {
            builder.build_dispatch();
        }
    } catch (const exception& e) {
        // Capture the exception error message
        tt::log_error("{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        tt::log_error("System error message: {}", strerror(errno));
    }
    return 0;
}
